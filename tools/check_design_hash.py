# SPDX-License-Identifier: Apache-2.0
"""Verify the technical design is the version the reading ledger recorded.

  python tools/check_design_hash.py [--design PATH]

Exit 0 when the SHA-256 matches docs/ledgers/reading-ledger.md, 1 otherwise (a changed
design must be reconciled before dependent work continues), 2 if it cannot be read.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "docs/ledgers/reading-ledger.md"


def recorded() -> tuple[str, str]:
  text = LEDGER.read_text()
  path = re.search(r"\| Path \| `([^`]+)` \|", text).group(1)
  sha = re.search(r"\| SHA-256 \| `([0-9a-f]{64})` \|", text).group(1)
  return path, sha


def main(argv=None) -> int:
  path, sha = recorded()
  ap = argparse.ArgumentParser()
  ap.add_argument("--design", default=path)
  args = ap.parse_args(argv)
  try:
    actual = hashlib.sha256(Path(args.design).read_bytes()).hexdigest()
  except OSError as exc:
    print(f"cannot read design {args.design}: {exc}", file=sys.stderr)
    return 2
  if actual != sha:
    print(f"design changed: recorded {sha}, found {actual} ({args.design})", file=sys.stderr)
    return 1
  print(f"design unchanged: {sha}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
