"""Regression coverage for binary scene labels, ignored pools and retained counts."""
import csv
import math
from pathlib import Path
import sys
import unittest
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'src/evaluation'), str(ROOT / 'src/statistics')]
import flood_metrics as evaluation
import flood_analysis as analysis

SCENE_FUNCTIONS = (evaluation.metrics_for_scene, analysis.scene_metrics)


class FloodContractTests(unittest.TestCase):
    def test_hand_computed_scene(self):
        target = np.array([[0, 0, 1], [1, -1, 1]])
        prediction = np.array([[0, 1, 0], [1, 1, 1]])
        reference = np.array([[0, 0, 1], [1, 0, 1]])
        for function in SCENE_FUNCTIONS:
            with self.subTest(module=function.__module__):
                result = function(prediction, target, reference)
                self.assertEqual([result[k] for k in ('tn', 'fp', 'fn', 'tp')], [1, 1, 1, 2])
                self.assertEqual(result['valid_pixels'], 5)
                self.assertAlmostEqual(result['miou'], (1 / 3 + 1 / 2) / 2)
                self.assertAlmostEqual(result['pixel_accuracy'], 3 / 5)
                self.assertAlmostEqual(result['agreement'], 3 / 5)

    def test_empty_and_all_ignored_pools(self):
        target = np.full((3, 4), -1)
        prediction = np.zeros_like(target)
        for module, scene in ((evaluation, evaluation.metrics_for_scene), (analysis, analysis.scene_metrics)):
            with warnings.catch_warnings():
                warnings.simplefilter('error', RuntimeWarning)
                row = scene(prediction, target, prediction)
                for rows in ([], [row], [row, row]):
                    with self.subTest(module=module.__name__, scenes=len(rows)):
                        result = module.aggregate(rows)
                        self.assertEqual(result['valid_pixels'], 0)
                        self.assertEqual(result['confusion_matrix'], [[0, 0], [0, 0]])
                        for key in ('miou', 'water_iou', 'boundary_water_iou', 'pixel_accuracy', 'agreement'):
                            self.assertTrue(math.isnan(result[key]), key)

    def test_invalid_labels_rejected_before_cast(self):
        for function in SCENE_FUNCTIONS:
            for array_index, invalids in ((0, (2, 256, 300, -1, .5, np.nan, np.inf)),
                                         (1, (2, -2, 300, .5, np.nan, np.inf)),
                                         (2, (2, 256, 300, -1, .5, np.nan, np.inf))):
                for value in invalids:
                    arrays = [np.zeros((2, 3)) for _ in range(3)]
                    arrays[array_index][0, 0] = value
                    with self.subTest(module=function.__module__, array=array_index, value=value):
                        with self.assertRaises(ValueError):
                            function(*arrays)

    def test_shape_and_non_numeric_rejected(self):
        for function in SCENE_FUNCTIONS:
            for value in (np.zeros((2, 1)), np.zeros((2, 3, 1)), np.full((2, 3), '0'), [[0, 0, 0]]):
                with self.subTest(module=function.__module__, shape=np.shape(value)):
                    with self.assertRaises(ValueError):
                        function(value, np.zeros((2, 3)), np.zeros((2, 3)))
            with self.assertRaises(ValueError):
                function(np.zeros((2, 3)), [[0]], np.zeros((2, 3)))

    def test_confusion_counts_direct_contract(self):
        target = np.array([[0, 1, -1]])
        prediction = np.array([[0, 1, 0]])
        valid = target >= 0
        self.assertEqual(sum(evaluation.confusion_counts(prediction, target, valid).values()), 2)
        for invalid in (np.array([[300, 1, 0]]), np.array([[256, 1, 0]])):
            with self.assertRaises(ValueError):
                evaluation.confusion_counts(invalid, target, valid)
        for invalid_mask in (np.ones_like(valid), np.ones((1, 1), dtype=bool), valid.astype(int)):
            with self.assertRaises(ValueError):
                evaluation.confusion_counts(prediction, target, invalid_mask)

    def test_modules_agree_for_binary_dtypes_and_ignored_pixels(self):
        rng = np.random.default_rng(7)
        for dtype in (np.int16, np.int64, np.float32, np.float64):
            target = rng.integers(-1, 2, size=(8, 9)).astype(dtype)
            prediction = rng.integers(0, 2, size=target.shape).astype(dtype)
            reference = rng.integers(0, 2, size=target.shape).astype(dtype)
            a, b = (fn(prediction, target, reference) for fn in SCENE_FUNCTIONS)
            for key in a:
                self.assertAlmostEqual(a[key], b[key], msg=key)
        for fn in SCENE_FUNCTIONS:
            result = fn(np.zeros((2, 2), dtype=bool), np.zeros((2, 2), dtype=bool), np.zeros((2, 2), dtype=bool))
            self.assertEqual(result['miou'], 1)
            self.assertTrue(math.isnan(result['water_iou']))

    def test_retained_flood_pools_against_integer_count_oracle(self):
        with (ROOT / 'results/flood/per_scene_metrics.csv').open(encoding='utf-8-sig', newline='') as stream:
            groups = {}
            for row in csv.DictReader(stream):
                groups.setdefault(row['variant'], []).append(row)
        self.assertEqual(len(groups), 6)
        for variant, rows in groups.items():
            self.assertEqual(len(rows), 90)
            tn, fp, fn, tp = [sum(int(r[k]) for r in rows) for k in ('tn', 'fp', 'fn', 'tp')]
            valid = sum(int(r['valid_pixels']) for r in rows)
            changed = sum(int(r['changed_valid_pixels']) for r in rows)
            self.assertEqual(valid, tn + fp + fn + tp)
            for module in (evaluation, analysis):
                with self.subTest(variant=variant, module=module.__name__):
                    result = module.aggregate(rows)
                    self.assertEqual(result['confusion_matrix'], [[tn, fp], [fn, tp]])
                    self.assertEqual(result['miou'], (tn / (tn + fp + fn) + tp / (tp + fp + fn)) / 2)
                    self.assertEqual(result['agreement'], 1 - changed / valid)


if __name__ == '__main__':
    unittest.main()
