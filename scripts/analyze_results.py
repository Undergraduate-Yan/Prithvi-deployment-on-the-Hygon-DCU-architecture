"""Reanalyse retained derived records without training or model inference."""
import argparse
import csv
import json
import math
from pathlib import Path
from _entrypoint import ROOT, configure

def read(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))

def write(path, rows):
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def flood(output):
    import numpy as np
    from flood_analysis import aggregate
    rows = read(ROOT / 'results/flood/per_scene_metrics.csv')
    groups = {}
    for row in rows:
        groups.setdefault(row['variant'], []).append(row)
    pooled = []
    for role, selected in groups.items():
        values = aggregate(selected)
        values['confusion_matrix'] = json.dumps(values['confusion_matrix'])
        pooled.append({'variant': role, **values})
    write(output / 'flood_pooled_metrics.csv', pooled)
    reference = {int(r['sample_index']): r for r in groups['RCS13-FP32']}
    rng = np.random.default_rng(42)
    margins = {'miou': -0.005, 'water_iou': -0.010, 'boundary_water_iou': -0.010, 'agreement': -0.005}
    summary = []
    for role in sorted(r for r in groups if r != 'RCS13-FP32'):
        selected = groups[role]
        if len(selected) != 90 or {int(r['sample_index']) for r in selected} != set(range(90)):
            raise ValueError('Flood scene identities/count differ from the paper protocol.')
        for metric, margin in margins.items():
            values = np.asarray([float(r[metric]) - float(reference[int(r['sample_index'])][metric]) for r in selected], dtype=float)
            values = values[np.isfinite(values)]
            indices = rng.integers(0, len(values), size=(10000, len(values)))
            samples = values[indices].mean(axis=1)
            low = float(np.percentile(samples, 5))
            summary.append({'variant': role, 'metric': metric, 'mean_paired_difference': float(values.mean()),
                            'lower95': low, 'margin': margin, 'within_descriptive_margin': low > margin,
                            'finite_scenes': len(values), 'resamples': 10000, 'seed': 42})
    write(output / 'flood_descriptive_margins.csv', summary)

def cloud(output):
    import numpy as np
    from cloud_analysis import metric_vector, metric_batch, CLASSES, MARGINS
    roles = [('fp32', 'Cloud-RCS-FP32-Compat'), ('fp16', 'Cloud-RCS-FP16-Opt-v2'), ('mp', 'Cloud-RCS-MP-Task-v2')]
    matrices, valid, agreement = {}, {}, {}
    scenes = read(ROOT / 'results/cloud/per_scene_metrics.csv')
    for key, role in roles:
        rows = read(ROOT / f'results/cloud/per_scene/{key}_confusion.csv')
        if [int(r['scene_index']) for r in rows] != list(range(300)):
            raise ValueError('Cloud scene order/count differs from the paper protocol.')
        matrices[key] = np.asarray([json.loads(r['confusion_matrix_json']) for r in rows], dtype=np.int64)
        valid[key] = np.asarray([int(r['valid_pixels']) for r in rows], dtype=np.int64)
        selected = sorted((r for r in scenes if r['role'] == role), key=lambda r: int(r['scene_index']))
        if [int(r['scene_index']) for r in selected] != list(range(300)):
            raise ValueError('Agreement records are not paired by scene index.')
        # Full-precision ratios and integer denominators recover the retained counts.
        counts = np.asarray([float(r['agreement_vs_fp32']) for r in selected]) * valid[key]
        if not np.all(np.abs(counts - np.rint(counts)) < 1e-5):
            raise ValueError('Agreement ratio cannot recover an unambiguous integer count.')
        agreement[key] = np.rint(counts).astype(np.int64)
        if not np.array_equal(matrices[key].sum(axis=(1, 2)), valid[key]):
            raise ValueError('Confusion matrices and valid pixel counts disagree.')
    if not all(np.array_equal(valid[key], valid['fp32']) for key, _ in roles):
        raise ValueError('Candidate valid pixel counts disagree.')
    rng = np.random.default_rng(42)
    indices = rng.integers(0, 300, size=(10000, 300), endpoint=False)
    reference_point = metric_vector(matrices['fp32'].sum(axis=0))
    reference_boot = metric_batch(matrices['fp32'][indices].sum(axis=1))
    pooled, summary = [], []
    for key, role in roles:
        point = metric_vector(matrices[key].sum(axis=0))
        observed_agreement = float(agreement[key].sum() / valid[key].sum())
        pooled.append({'role': role, **point, 'agreement': observed_agreement})
        if key == 'fp32':
            continue
        candidate_boot = metric_batch(matrices[key][indices].sum(axis=1))
        for metric in ('mIoU', 'MacroF1', *(name + '_IoU' for name in CLASSES), 'agreement'):
            if metric == 'agreement':
                delta = agreement[key][indices].sum(axis=1) / valid[key][indices].sum(axis=1) - 1.0
                estimate = observed_agreement - 1.0
            else:
                delta = candidate_boot[metric] - reference_boot[metric]
                estimate = point[metric] - reference_point[metric]
            low = float(np.nanpercentile(delta, 5.0))
            summary.append({'role': role, 'metric': metric, 'difference': estimate, 'lower95': low,
                            'margin': -MARGINS[metric], 'status': 'PASS' if low >= -MARGINS[metric] else 'FAIL',
                            'resamples': 10000, 'seed': 42})
    write(output / 'cloud_pooled_metrics.csv', pooled)
    write(output / 'cloud_criteria.csv', summary)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=['flood', 'cloud'])
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    configure()
    (flood if args.task == 'flood' else cloud)(args.output_dir)

if __name__ == '__main__':
    main()
