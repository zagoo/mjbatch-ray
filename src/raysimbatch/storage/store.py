# SPDX-License-Identifier: Apache-2.0
"""DurableStore and ArtifactRegistry (design §6.8).

``DurableStore`` is the design's four-method protocol (put_blob, get_blob,
acquire_lease, commit) over PostgreSQL + content-addressed blobs; ``db`` exposes the
domain operations used by schedulers and controllers. Async wrappers run the blocking
drivers on worker threads so control-plane event loops never block on storage.
"""

from __future__ import annotations

import asyncio
import hashlib
import urllib.parse
from pathlib import Path

from raysimbatch.config.models import ArtifactRef, RunConfig
from raysimbatch.protocol import Artifact, Lease
from raysimbatch.storage.blob import BlobCorrupt, BlobStore, Retry, open_blob_store
from raysimbatch.storage.db import CommitAck, Database
from raysimbatch.storage.secrets import resolve


class DurableStore:
  def __init__(self, db: Database, blobs: BlobStore):
    self.db, self.blobs = db, blobs

  @classmethod
  def from_config(cls, cfg: RunConfig, *, migrate: bool = True) -> DurableStore:
    if not cfg.artifact_root or not cfg.database_secret:
      raise ValueError("artifact_root and database_secret are required for durable runs")
    st = cfg.storage
    db = Database(resolve(cfg.database_secret), timeout_s=st.database_timeout_s, retry_attempts=st.retry_attempts,
                  backoff=st.retry_backoff_s)
    if migrate:
      db.migrate()
    blobs = open_blob_store(cfg.artifact_root, Retry(st.retry_attempts, st.retry_backoff_s))
    return cls(db, blobs)

  # design protocol (async)
  async def put_blob(self, payload: bytes, media_type: str) -> Artifact:
    return await asyncio.to_thread(self.put_blob_sync, payload, media_type)

  async def get_blob(self, artifact: Artifact) -> bytes:
    return await asyncio.to_thread(self.blobs.get, artifact)

  async def acquire_lease(self, key: str, owner: str, ttl_s: float) -> Lease:
    return await asyncio.to_thread(self.db.acquire_lease, key, owner, ttl_s)

  async def commit(self, key: str, epoch: int, digest: str, artifact: Artifact) -> CommitAck:
    return await asyncio.to_thread(self.db.commit, key, epoch, digest, artifact)

  def put_blob_sync(self, payload: bytes, media_type: str) -> Artifact:
    a = self.blobs.put(payload, media_type)
    return a


def read_ref(ref: ArtifactRef | Artifact) -> bytes:
  """Fetch and verify a content-addressed reference by URI (file:// or s3://)."""
  uri = ref.uri
  sha = ref.sha256
  if uri is None or sha is None:
    raise ValueError("empty artifact reference")
  u = urllib.parse.urlparse(uri)
  if u.scheme == "file":
    data = Path(u.path).read_bytes()
  elif u.scheme == "s3":
    import boto3

    q = urllib.parse.parse_qs(u.query)
    import os

    endpoint = q.get("endpoint", [os.environ.get("RSB_S3_ENDPOINT")])[0]
    data = boto3.client("s3", endpoint_url=endpoint).get_object(Bucket=u.netloc, Key=u.path.lstrip("/"))["Body"].read()
  else:
    raise ValueError(f"unsupported artifact URI {uri!r}")
  if hashlib.sha256(data).hexdigest() != sha:
    raise BlobCorrupt(f"artifact {uri} does not match sha256 {sha}")
  return data


class ArtifactRegistry:
  """Content-addressed artifacts: registration, resolution, per-node materialization."""

  def __init__(self, store: DurableStore):
    self.store = store

  def register_bytes(self, payload: bytes, media_type: str) -> ArtifactRef:
    a = self.store.blobs.put(payload, media_type)
    self.store.db.register_artifact(a)
    return ArtifactRef(uri=a.uri, sha256=a.sha256)

  def register_file(self, path: str | Path, media_type: str = "application/octet-stream") -> ArtifactRef:
    return self.register_bytes(Path(path).read_bytes(), media_type)

  async def register(self, artifact: Artifact) -> str:
    await asyncio.to_thread(self.store.db.register_artifact, artifact)
    return artifact.sha256

  async def resolve(self, digest: str) -> Artifact:
    a = await asyncio.to_thread(self.store.db.artifact, digest)
    if a is None:
      raise KeyError(f"unknown artifact {digest}")
    return a

  def fetch(self, ref: ArtifactRef) -> bytes:
    return read_ref(ref)

  async def materialize(self, digest: str, node_id: str):
    """ObjectRef of the verified bytes on ``node_id``, via that node's AssetCache."""
    from raysimbatch.control.assets import asset_cache_for_node

    cache = asset_cache_for_node(node_id, self.store.blobs.root)
    art = await self.resolve(digest)
    return await cache.get.remote(art.uri, art.sha256, art.byte_length, art.media_type)
