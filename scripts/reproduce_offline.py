"""Run the bundled offline analyses and preserve per-command logs and status."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from _entrypoint import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = str(ROOT / 'configs/analysis/retained_records.json')
    jobs = [
        ('tables', 'reproduce_tables.py', []),
        ('flood', 'analyze_results.py', ['flood']),
        ('cloud', 'analyze_results.py', ['cloud']),
        ('configuration', 'analyze_controls.py', ['configuration']),
        ('flood-latency', 'analyze_controls.py', ['flood-latency']),
        ('cloud-latency', 'analyze_controls.py', ['cloud-latency']),
        *[(action, 'analyze_supplementary.py', [action, '--config', config])
          for action in ['clean89', 'margins', 'cloud-confusion', 'process-latency']],
    ]
    results = []
    for name, script, arguments in jobs:
        command = [sys.executable, str(ROOT / 'scripts' / script), *arguments,
                   '--output-dir', str(output / name)]
        started = time.monotonic()
        with (output / (name + '.log')).open('x', encoding='utf-8') as log:
            completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        results.append(dict(analysis=name, command=command, exit_code=completed.returncode,
                            elapsed_seconds=time.monotonic()-started,
                            status='EXECUTED' if completed.returncode == 0 else 'FAILED'))
        (output / 'execution_summary.json').write_text(json.dumps({
            'scope': 'Offline derived-record analyses; successful execution does not establish independent numerical validation.',
            'results': results}, indent=2)+'\n', encoding='utf-8')
        print(name, results[-1]['status'], flush=True)
    return 1 if any(row['exit_code'] for row in results) else 0


if __name__ == '__main__':
    raise SystemExit(main())
