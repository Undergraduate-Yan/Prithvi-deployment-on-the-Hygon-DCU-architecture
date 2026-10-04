"""Run the paper's five-process C++ timing protocol on an explicitly supplied plan."""
import argparse
import json
import os
from pathlib import Path
import subprocess

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runner', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--input-raw', type=Path, required=True)
    parser.add_argument('--cpu', type=int, required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if os.name != 'posix' or not hasattr(os, 'sched_getaffinity'):
        raise RuntimeError('This timing protocol requires the Linux K100 environment.')
    if args.cpu not in os.sched_getaffinity(0):
        raise ValueError('CPU is outside the allowed affinity set.')
    runner = args.runner.resolve(strict=True)
    plan = args.plan.resolve(strict=True)
    input_raw = args.input_raw.resolve(strict=True)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    for index in range(5):
        folder = (args.output_dir / f'process_{index:02d}').resolve()
        folder.mkdir()
        command = ['taskset', '-c', str(args.cpu), str(runner),
                   '--plan', str(plan), '--input-raw', str(input_raw),
                   '--output-raw', str(folder / 'logits.f32'),
                   '--latency-csv', str(folder / 'latency.csv'),
                   '--scope', 'logits', '--device', str(args.device),
                   '--warmups', '50', '--iterations', '200']
        (folder / 'command.json').write_text(json.dumps(command, indent=2) + '\n', encoding='utf-8')
        with (folder / 'stdout.txt').open('w', encoding='utf-8') as stdout, (folder / 'stderr.txt').open('w', encoding='utf-8') as stderr:
            subprocess.run(command, check=True, stdout=stdout, stderr=stderr)

if __name__ == '__main__':
    main()
