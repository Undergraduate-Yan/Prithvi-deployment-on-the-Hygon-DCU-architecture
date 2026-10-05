"""Prepare identity-checked system configurations from portable artifact manifests."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from _entrypoint import ROOT


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=['flood', 'cloud'])
    parser.add_argument('--artifact-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True, help='Fixed development input NPY; cloud expects the retained normalized development pack')
    parser.add_argument('--input-sha256', required=True, help='Expected identity from retained input evidence, not a replacement selected for convenience')
    args = parser.parse_args()
    source = args.input.resolve(strict=True)
    digest = sha256(source)
    if digest != args.input_sha256:
        raise ValueError('Development input SHA256 mismatch')
    if args.task == 'flood' and digest != '7f0e2c08cf337dadf277a91976e6f9f7b3f8e022a2b06ad97547dcb562f5be80':
        raise ValueError('The flood system protocol requires the retained development input')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    configdir = args.output_dir / '00_configs'
    configdir.mkdir()
    if args.task == 'flood':
        specs = [('Mono_FP32', 'configs/flood/runtime/Mono_FP32.json'),
                 ('Mono_FP16_Opt', 'configs/flood/runtime/Mono_FP16_OPT.json'),
                 ('MP_RCS_Opt', 'configs/flood/runtime/MP_RCS_OPT.json')]
    else:
        specs = [(name, f'manifests/artifacts/cloud_{name}.json') for name in ('fp32', 'fp16', 'mp')]
    candidates = []
    for name, relative in specs:
        output = configdir / (name + '.json')
        subprocess.run([sys.executable, str(ROOT / 'scripts/resolve_artifacts.py'),
                        '--manifest', str(ROOT / relative), '--artifact-root', str(args.artifact_root.resolve()),
                        '--output', str(output)], check=True)
        if args.task == 'cloud':
            manifest = json.loads(output.read_text(encoding='utf-8'))
            sessions = manifest['sessions']
            if len(sessions) != 14 or [r['ordinal'] for r in sessions] != list(range(14)):
                raise ValueError('Cloud ordered session identities are incomplete')
            candidates.append(dict(role=manifest['role'], models=[r['model']['path'] for r in sessions], caches=[r['cache']['path'] for r in sessions]))
    if args.task == 'flood':
        target = args.output_dir / '02_input/configuration_validation_input_00.npy'
        target.parent.mkdir()
        shutil.copyfile(source, target)
    else:
        (args.output_dir / 'cloud_candidates.json').write_text(json.dumps({'candidates': candidates}, indent=2)+'\n', encoding='utf-8')
    (args.output_dir / 'input_identity.json').write_text(json.dumps(dict(path=str(source), sha256=digest, size_bytes=source.stat().st_size), indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
