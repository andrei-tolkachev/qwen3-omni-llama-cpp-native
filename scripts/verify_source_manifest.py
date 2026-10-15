#!/usr/bin/env python3
"""Offline SHA-256 verification of this source release. Zero network access."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parent.parent
expected = json.loads((root / "SOURCE_MANIFEST.json").read_text())
missing = []
mismatch = []
for rel, digest in expected.items():
    path = root / rel
    if not path.is_file():
        missing.append(rel)
    elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        mismatch.append(rel)
extras = [str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()
          and p.relative_to(root).parts[0] not in (".git", "build", "build-omni", "build-omni-ci", "build-cpu-smoke")
          and "__pycache__" not in p.parts and not p.name.endswith(".pyc")
          and p != root / "SOURCE_MANIFEST.json" and str(p.relative_to(root)) not in expected]
print(f"Manifest: {len(expected)} files; missing: {len(missing)}; mismatched: {len(mismatch)}; unexpected: {len(extras)}")
for key,values in (("missing",missing),("mismatch",mismatch),("unexpected",extras)):
    for rel in values[:20]:print(f"{key}: {rel}")
raise SystemExit(1 if missing or mismatch or extras else 0)
