"""Resolve content-addressed artifact locations while checking their size and SHA256."""
import argparse
import hashlib
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--artifact-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.artifact_root.resolve(strict=True)
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    def visit(value):
        if isinstance(value, dict):
            value = {key: visit(item) for key, item in value.items()}
            if isinstance(value.get('path'), str) and value['path'].startswith('artifact://'):
                suffix = value['path'][len('artifact://'):]
                path = (root / suffix).resolve(strict=True)
                if root not in path.parents:
                    raise ValueError('Artifact path is outside artifact-root.')
                if not value.get('sha256') or 'size_bytes' not in value:
                    raise ValueError('Artifact identity is incomplete.')
                digest = hashlib.sha256()
                with path.open('rb') as stream:
                    for chunk in iter(lambda: stream.read(8 << 20), b''):
                        digest.update(chunk)
                if digest.hexdigest() != value['sha256'] or path.stat().st_size != int(value['size_bytes']):
                    raise ValueError('Artifact identity mismatch: ' + str(path))
                value['path'] = str(path)
            return value
        if isinstance(value, list):
            return [visit(item) for item in value]
        return value
    result = visit(manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')

if __name__ == '__main__':
    main()
