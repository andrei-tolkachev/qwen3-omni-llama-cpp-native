#!/usr/bin/env python3
"""Package exactly the verified source manifests, never a working directory wholesale."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile

root = Path(__file__).resolve().parents[1]
module = root / 'tools/loraai_omni_native'


def archive(source, target, prefix):
    subprocess.run([sys.executable, str(source / 'scripts/verify_source_manifest.py')], check=True)
    files = sorted([*json.loads((source / 'SOURCE_MANIFEST.json').read_text()), 'SOURCE_MANIFEST.json'])
    with target.open('xb') as output, gzip.GzipFile(filename='', mode='wb', fileobj=output, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w|', format=tarfile.PAX_FORMAT) as tar:
            for rel in files:
                path = source / rel
                if Path(rel).is_absolute() or '..' in Path(rel).parts or path.is_symlink():
                    raise ValueError('unsafe archive path: ' + rel)
                info = tar.gettarinfo(str(path), prefix + '/' + rel)
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ''
                info.mode = 0o755 if path.stat().st_mode & 0o111 else 0o644
                with path.open('rb') as content:
                    tar.addfile(info, content)
    return hashlib.sha256(target.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', args.version):
        parser.error('invalid version')
    destination = args.output.resolve()
    if destination == root or root in destination.parents:
        parser.error('output must be outside the source tree')
    destination.mkdir(parents=True, exist_ok=True)
    checksums = []
    for kind, source in [('source', root), ('upgrade-kit', module)]:
        prefix = f'qwen3-omni-native-{args.version}-{kind}'
        name = prefix + '.tar.gz'
        digest = archive(source, destination / name, prefix)
        checksums.append(f'{digest}  {name}\n')
    with (destination / 'SHA256SUMS').open('x') as output:
        output.writelines(checksums)
    print(''.join(checksums), end='')


if __name__ == '__main__':
    main()
