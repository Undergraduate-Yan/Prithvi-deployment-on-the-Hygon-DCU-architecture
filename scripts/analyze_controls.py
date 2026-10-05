"""Recompute configuration scores and controlled latency from retained records."""
import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from _entrypoint import ROOT


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write(path, rows):
    if not rows:
        raise ValueError('No records to aggregate.')
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def configuration(output):
    groups = defaultdict(list)
    for row in read(ROOT / 'results/flood/configuration/per_scene_metrics.csv'):
        groups[row['configuration']].append(row)
    values = []
    for name, rows in sorted(groups.items()):
        indexes = [int(r['sample_index']) for r in rows]
        if len(indexes) != 64 or set(indexes) != set(range(64)):
            raise ValueError(f'{name}: configuration scene grid must contain 64 unique indexes')
        matrix = [[0, 0], [0, 0]]
        valid = 0
        for row in rows:
            current = json.loads(row['confusion_matrix'])
            if len(current) != 2 or any(len(r) != 2 for r in current):
                raise ValueError('Expected a binary confusion matrix')
            if any(not isinstance(v, int) or v < 0 for r in current for v in r):
                raise ValueError('Counts must be nonnegative integers')
            if sum(map(sum, current)) != int(row['valid_pixels']):
                raise ValueError('Valid count mismatch')
            for i in range(2):
                for j in range(2):
                    matrix[i][j] += current[i][j]
            valid += int(row['valid_pixels'])
        tn, fp = matrix[0]
        fn, tp = matrix[1]
        background = tn / (tn + fp + fn) if tn + fp + fn else math.nan
        water = tp / (tp + fp + fn) if tp + fp + fn else math.nan
        values.append(dict(configuration=name, scenes=len(rows), valid_pixels=valid,
                           miou=(background + water) / 2, water_iou=water,
                           background_iou=background, pixel_accuracy=(tn + tp) / valid,
                           confusion_matrix=json.dumps(matrix)))
    write(output / 'configuration_pooled.csv', values)


def latency(task, output):
    path = ROOT / ('results/flood/latency_calls.csv' if task == 'flood' else 'results/cloud/process_latency_calls.csv')
    groups = defaultdict(list)
    for row in read(path):
        if row.get('error') or int(row.get('retry', '0')):
            raise ValueError('Error/retry in controlled latency input')
        role = row['variant'] if task == 'flood' else row['role']
        trial = row['round'] if task == 'flood' else row['trial']
        value = float(row['latency_ms'])
        if not math.isfinite(value) or value <= 0:
            raise ValueError('Invalid latency')
        groups[(row['node'], role, trial)].append(value)
    per_process = []
    aggregate = defaultdict(list)
    for (node, role, trial), values in sorted(groups.items()):
        if len(values) != 200:
            raise ValueError(f'{node}/{role}/{trial}: expected 200 measured calls')
        per_process.append(dict(node=node, role=role, trial=trial, calls=len(values), median_ms=statistics.median(values)))
        aggregate[(node, role)].append(values)
    pooled = []
    for (node, role), trials in sorted(aggregate.items()):
        if len(trials) != 5:
            raise ValueError('Expected five fresh processes')
        values = sorted(v for trial in trials for v in trial)
        # Cloud retained summaries use nearest-rank percentiles. Flood uses NumPy linear percentiles.
        def percentile(q):
            if task == 'cloud':
                return values[math.ceil(q * len(values)) - 1]
            position = (len(values) - 1) * q
            low = math.floor(position)
            high = math.ceil(position)
            return values[low] + (values[high] - values[low]) * (position - low)
        pooled.append(dict(node=node, role=role, calls=len(values), fresh_processes=len(trials),
                           pooled_median_ms=statistics.median(values), p95_ms=percentile(.95), p99_ms=percentile(.99),
                           median_of_process_medians_ms=statistics.median(statistics.median(t) for t in trials),
                           percentile_method='nearest_rank' if task == 'cloud' else 'linear'))
    write(output / f'{task}_process_medians.csv', per_process)
    write(output / f'{task}_controlled_latency.csv', pooled)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['configuration', 'flood-latency', 'cloud-latency'])
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.action == 'configuration':
        configuration(args.output_dir)
    else:
        latency(args.action.split('-')[0], args.output_dir)


if __name__ == '__main__':
    main()
