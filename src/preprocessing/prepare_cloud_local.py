"""Prepare the fixed cloud scene set from locally acquired CloudSEN12+ TACO parts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from cloud_decode import decode

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', required=True, type=Path)
    parser.add_argument('--source-root', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    args = parser.parse_args()
    with args.records.open(encoding='utf-8-sig', newline='') as stream:
        records = list(csv.DictReader(stream))
    if len(records) != 300 or len({r['datapoint_id'] for r in records}) != 300 or len({r['roi_id'] for r in records}) != 300:
        raise ValueError('The paper protocol requires 300 distinct scenes and regions.')
    source_root = args.source_root.resolve(strict=True)
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    output = args.output_root / 'scene_payloads'
    output.mkdir(parents=True)
    evidence = []
    for index, record in enumerate(records):
        source = (source_root / record['source_part']).resolve(strict=True)
        if source_root not in source.parents:
            raise ValueError('Source part must be inside source-root.')
        with source.open('rb') as stream:
            stream.seek(int(record['source_begin']))
            payload = stream.read(int(record['source_length']))
        if len(payload) != int(record['source_length']):
            raise ValueError('Truncated source scene: ' + record['datapoint_id'])
        _, row, image, label = decode(index, record, payload)
        target = output / f'{index:03d}.npz'
        np.savez_compressed(target, image=image, label=label)
        row['materialized_npz_sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
        evidence.append(row)
    manifest = {'status': 'REPRODUCTION_PAYLOAD_COMPLETE', 'samples': 300,
                'scope': 'Reproduction inputs; no claim of a new independent evaluation',
                'records_sha256': hashlib.sha256(args.records.read_bytes()).hexdigest(),
                'scenes': evidence}
    # Retain the evaluator's manifest filename for data-format compatibility.
    (args.output_root / 'cloud_phase7r_formal_payload_manifest.json').write_text(
        json.dumps(manifest, indent=2) + '\n', encoding='utf-8')

if __name__ == '__main__':
    main()
