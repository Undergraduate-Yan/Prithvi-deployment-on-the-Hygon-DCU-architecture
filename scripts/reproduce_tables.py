"""Render the retained supplementary CSV tables without recomputing experiments."""
import argparse
import csv
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.output_dir.exists():
        raise FileExistsError('Choose a new output directory: ' + str(args.output_dir))
    args.output_dir.mkdir(parents=True)
    index = json.loads((root / 'results/table_index.json').read_text(encoding='utf-8'))
    for item in index:
        with (root / item['file']).open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.reader(stream))
        def line(row):
            return '| ' + ' | '.join(v.replace('|', '\\|').replace('\n', '<br>') for v in row) + ' |'
        lines = ['# Table ' + item['table'] + '. ' + item['title'], '',
                 'Retained study tabulation; no inference or statistical recomputation.', '',
                 line(rows[0]), line(['---'] * len(rows[0]))]
        lines.extend(line(row) for row in rows[1:])
        lines += ['', item['measurement_scope'], '', 'Input: `' + item['file'] + '`']
        (args.output_dir / (item['table'] + '.md')).write_text('\n'.join(lines) + '\n', encoding='utf-8')

if __name__ == '__main__':
    main()
