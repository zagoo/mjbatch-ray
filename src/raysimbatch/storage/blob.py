# SPDX-License-Identifier: Apache-2.0
"""Content-addressed immutable blob storage: S3-compatible (production) or filesystem.

Objects are written under their SHA-256 before any database transaction references
them (design §6.6); a re-put of the same bytes is a no-op, and every read verifies
length and digest so corrupt model/weight bytes are never loaded (§8.1).
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from raysimbatch.protocol import Artifact


class BlobError(RuntimeError):
  pass


class BlobCorrupt(BlobError):
  pass


class BlobStore(Protocol):
  root: str

  def put(self, payload: bytes, media_type: str) -> Artifact: ...
  def get(self, artifact: Artifact) -> bytes: ...
  def exists(self, sha256: str) -> bool: ...
  def delete(self, sha256: str) -> None: ...
  def list_hashes(self) -> list[str]: ...


def _digest(payload: bytes) -> str:
  return hashlib.sha256(payload).hexdigest()


def _verify(artifact: Artifact, data: bytes) -> bytes:
  if len(data) != artifact.byte_length or _digest(data) != artifact.sha256:
    raise BlobCorrupt(f"artifact {artifact.sha256} failed length/digest verification")
  return data


@dataclass
class Retry:
  attempts: int = 3
  backoff: tuple[float, ...] = (0.1, 0.5, 2.0)

  def run(self, fn, *args):
    last: Exception | None = None
    for i in range(self.attempts + 1):
      try:
        return fn(*args)
      except BlobCorrupt:
        raise
      except Exception as exc:  # noqa: BLE001 - transport errors are retried with backoff
        last = exc
        if i < self.attempts:
          time.sleep(self.backoff[min(i, len(self.backoff) - 1)])
    raise BlobError(f"blob operation failed after {self.attempts + 1} attempts: {last}") from last


class FileBlobStore:
  """Filesystem store (development and single-host qualification)."""

  def __init__(self, root: str | Path, retry: Retry | None = None):
    self.path = Path(root)
    self.path.mkdir(parents=True, exist_ok=True)
    self.root = f"file://{self.path}"
    self.retry = retry or Retry()

  def _file(self, sha: str) -> Path:
    return self.path / "objects" / sha[:2] / sha

  def put(self, payload: bytes, media_type: str) -> Artifact:
    sha = _digest(payload)
    f = self._file(sha)
    if not f.exists():
      f.parent.mkdir(parents=True, exist_ok=True)
      fd, tmp = tempfile.mkstemp(dir=f.parent)
      with os.fdopen(fd, "wb") as out:
        out.write(payload)
        out.flush()
        os.fsync(out.fileno())
      os.replace(tmp, f)
    return Artifact(f"{self.root}/objects/{sha[:2]}/{sha}", sha, len(payload), media_type)

  def get(self, artifact: Artifact) -> bytes:
    return _verify(artifact, self.retry.run(self._file(artifact.sha256).read_bytes))

  def exists(self, sha256: str) -> bool:
    return self._file(sha256).exists()

  def delete(self, sha256: str) -> None:
    self._file(sha256).unlink(missing_ok=True)

  def list_hashes(self) -> list[str]:
    base = self.path / "objects"
    return sorted(p.name for p in base.glob("*/*")) if base.exists() else []


class S3BlobStore:
  """S3-compatible store; credentials come from the standard AWS environment/secret
  mounts, never from configuration values."""

  def __init__(self, bucket: str, prefix: str = "", endpoint: str | None = None, retry: Retry | None = None,
               timeout_s: float = 30.0):
    import boto3
    from botocore.config import Config

    self.bucket, self.prefix = bucket, prefix.strip("/")
    self.endpoint = endpoint
    cfg = Config(connect_timeout=timeout_s, read_timeout=timeout_s, retries={"max_attempts": 1},
                 s3={"addressing_style": "path"})
    self.client = boto3.client("s3", endpoint_url=endpoint, config=cfg)
    self.retry = retry or Retry()
    q = f"?endpoint={urllib.parse.quote(endpoint, safe='')}" if endpoint else ""
    self.root = f"s3://{bucket}/{self.prefix}{q}"
    self._ensure_bucket()

  def _ensure_bucket(self) -> None:
    try:
      self.client.head_bucket(Bucket=self.bucket)
    except Exception:  # noqa: BLE001 - create on first use (idempotent)
      try:
        self.client.create_bucket(Bucket=self.bucket)
      except Exception as exc:  # noqa: BLE001
        if "BucketAlready" not in str(exc):
          raise BlobError(f"cannot access bucket {self.bucket}: {exc}") from exc

  def _key(self, sha: str) -> str:
    return f"{self.prefix}/objects/{sha[:2]}/{sha}".lstrip("/")

  def put(self, payload: bytes, media_type: str) -> Artifact:
    sha = _digest(payload)
    key = self._key(sha)
    if not self.exists(sha):
      self.retry.run(lambda: self.client.put_object(Bucket=self.bucket, Key=key, Body=payload,
                                                    ContentType=media_type, Metadata={"sha256": sha}))
    return Artifact(f"s3://{self.bucket}/{key}", sha, len(payload), media_type)

  def get(self, artifact: Artifact) -> bytes:
    key = self._key(artifact.sha256)
    body = self.retry.run(lambda: self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read())
    return _verify(artifact, body)

  def exists(self, sha256: str) -> bool:
    try:
      self.client.head_object(Bucket=self.bucket, Key=self._key(sha256))
      return True
    except Exception as exc:  # noqa: BLE001
      if "404" in str(exc) or "NotFound" in str(exc) or "NoSuchKey" in str(exc):
        return False
      raise BlobError(str(exc)) from exc

  def delete(self, sha256: str) -> None:
    self.retry.run(lambda: self.client.delete_object(Bucket=self.bucket, Key=self._key(sha256)))

  def list_hashes(self) -> list[str]:
    out = []
    pages = self.client.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=f"{self.prefix}/objects/".lstrip("/"))
    for page in pages:
      out += [o["Key"].rsplit("/", 1)[-1] for o in page.get("Contents", [])]
    return sorted(out)


_CACHE: dict[str, BlobStore] = {}


def open_blob_store(uri: str, retry: Retry | None = None) -> BlobStore:
  """``file:///abs/path`` or ``s3://bucket/prefix[?endpoint=http://host:port]``."""
  if uri in _CACHE:
    return _CACHE[uri]
  u = urllib.parse.urlparse(uri)
  if u.scheme == "file":
    store: BlobStore = FileBlobStore(u.path, retry)
  elif u.scheme == "s3":
    q = urllib.parse.parse_qs(u.query)
    endpoint = q.get("endpoint", [os.environ.get("RSB_S3_ENDPOINT")])[0]
    store = S3BlobStore(u.netloc, u.path, endpoint, retry)
  else:
    raise BlobError(f"unsupported artifact root {uri!r} (file:// or s3://)")
  _CACHE[uri] = store
  return store
