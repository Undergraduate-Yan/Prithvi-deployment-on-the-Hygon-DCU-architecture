from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_csv(name: str) -> list[dict[str, str]]:
    with (ROOT / "results" / name).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


class RepositoryLayoutTests(unittest.TestCase):
    def test_expected_source_groups_exist(self) -> None:
        self.assertTrue((ROOT / "src/baseline").is_dir())
        self.assertTrue((ROOT / "src/phase11").is_dir())
        self.assertTrue((ROOT / "paper/main.tex").is_file())
        self.assertTrue((ROOT / "paper/main_CN.tex").is_file())

    def test_paper_metrics_are_complete(self) -> None:
        rows = read_csv("paper_metrics.csv")
        self.assertEqual([row["variant"] for row in rows], ["FP32", "Full FP16", "M0", "M1", "M2", "M3", "M4", "M5"])
        by_name = {row["variant"]: row for row in rows}
        self.assertEqual(by_name["M4"]["strict_mae"], "0.108455")
        self.assertEqual(by_name["M4"]["strict_max_abs"], "0.393831")
        self.assertTrue(all(by_name[name]["task_status"] == "passed" for name in ["Full FP16", "M0", "M1", "M2", "M3", "M4", "M5"]))
        self.assertTrue(all(by_name[name]["strict_status"].startswith("failed") for name in ["Full FP16", "M0", "M1", "M2", "M3", "M4", "M5"]))

    def test_deployment_capacity_and_stability(self) -> None:
        rows = {row["bundle"]: row for row in read_csv("deployment_comparison.csv")}
        m5 = int(rows["M5"]["payload_bytes"])
        fp16 = int(rows["Full FP16"]["payload_bytes"])
        reduction = (fp16 - m5) / fp16 * 100.0
        self.assertAlmostEqual(reduction, 23.463061, places=5)
        for row in rows.values():
            self.assertEqual(row["deployment_status"], "passed")
            self.assertEqual(row["inference_errors"], "0")
            self.assertEqual(row["nonfinite_outputs"], "0")
            self.assertEqual(row["prediction_drifts"], "0")

    def test_external_manifest_hashes_are_well_formed(self) -> None:
        manifest = json.loads((ROOT / "artifacts/required_artifacts.example.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema"], "prithvi_k100_external_artifacts_v1")
        names = [row["name"] for row in manifest["artifacts"]]
        self.assertEqual(len(names), len(set(names)))
        for row in manifest["artifacts"]:
            self.assertEqual(len(row["sha256"]), 64)
            int(row["sha256"], 16)


if __name__ == "__main__":
    unittest.main()
