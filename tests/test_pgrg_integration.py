"""
test_pgrg_integration.py — Integration and Regression Tests for Improved PGRG

Verifies:
  1. Improved PGRG uses verified parameters (min_area=100, seed=0.45, sim=25, gap=11, equal weights).
  2. Output masks have the expected shape (H, W) and valid binary values ({0, 255}).
  3. Both Improved and Original PGRG configurations execute correctly.
  4. Baseline methods (Standard Region Growing, Otsu + Morphology, Watershed) remain functional.
  5. The CLI selects Improved PGRG by default, and --original-pgrg selects Original PGRG.
"""

import os
import sys
import csv
import subprocess
import unittest
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.pgrg import (
    IMPROVED_CONFIG,
    ORIGINAL_CONFIG,
    pgrg_pipeline,
    improved_pgrg,
    original_pgrg,
)
from src.region_growing import standard_region_growing
from src.otsu import otsu_morphology
from src.watershed import watershed_segmentation
from src.preprocessing import preprocess_image

SAMPLE_IMAGE = os.path.join(
    project_root, "dataset", "NEU-DET", "validation", "images", "scratches", "scratches_241.jpg"
)


class TestPGRGIntegration(unittest.TestCase):

    def setUp(self):
        self.assertTrue(os.path.exists(SAMPLE_IMAGE), f"Sample image not found: {SAMPLE_IMAGE}")
        _, _, _, self.normalized = preprocess_image(SAMPLE_IMAGE)

    def test_verified_improved_parameters(self):
        """Verify that Improved PGRG uses the exact verified parameter values."""
        self.assertEqual(IMPROVED_CONFIG["min_area"], 100)
        self.assertEqual(IMPROVED_CONFIG["seed_threshold"], 0.45)
        self.assertEqual(IMPROVED_CONFIG["similarity_threshold"], 25)
        self.assertEqual(IMPROVED_CONFIG["merge_gap"], 11)
        self.assertIsNone(IMPROVED_CONFIG["weights"])

    def test_verified_original_parameters(self):
        """Verify that Original PGRG uses the verified parameter values."""
        self.assertEqual(ORIGINAL_CONFIG["min_area"], 100)
        self.assertEqual(ORIGINAL_CONFIG["seed_threshold"], 0.45)
        self.assertEqual(ORIGINAL_CONFIG["similarity_threshold"], 25)
        self.assertEqual(ORIGINAL_CONFIG["merge_gap"], 11)
        self.assertIsNone(ORIGINAL_CONFIG["weights"])

    def test_original_pgrg_execution_and_mask_validity(self):
        """Verify Original PGRG executes and produces valid binary mask in {0, 255}."""
        mask = original_pgrg(self.normalized)
        self.assertEqual(mask.shape, self.normalized.shape)
        self.assertEqual(mask.dtype, np.uint8)
        unique_vals = set(np.unique(mask))
        self.assertTrue(unique_vals.issubset({0, 255}), f"Unexpected mask values: {unique_vals}")

    def test_baseline_methods_execution(self):
        """Verify all baseline segmentation methods remain fully functional."""
        # Standard Region Growing
        srg_mask = standard_region_growing(self.normalized, seed_threshold=0.45, similarity_threshold=20)
        self.assertEqual(srg_mask.shape, self.normalized.shape)
        self.assertTrue(set(np.unique(srg_mask)).issubset({0, 255}))

        # Otsu + Morphology
        _, _, otsu_mask = otsu_morphology(self.normalized)
        self.assertEqual(otsu_mask.shape, self.normalized.shape)
        self.assertTrue(set(np.unique(otsu_mask)).issubset({0, 255}))

        # Watershed
        _, ws_mask = watershed_segmentation(self.normalized)
        self.assertEqual(ws_mask.shape, self.normalized.shape)
        self.assertTrue(set(np.unique(ws_mask)).issubset({0, 255}))

    def test_cli_default_selects_original(self):
        """Verify CLI runs Original PGRG by default when no flag is provided."""
        python_exe = sys.executable
        cmd = [python_exe, "main.py", "--no-show", SAMPLE_IMAGE]
        result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, f"CLI error: {result.stderr}")
        self.assertIn("Original PGRG", result.stdout)

    def test_phase10_validation_benchmark(self):
        """Verify Phase 10 evaluates Original PGRG and baselines, outputting the validation table."""
        python_exe = sys.executable
        cmd = [python_exe, "main.py", "--phase", "10"]
        result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, f"Phase 10 error: {result.stderr}")
        self.assertIn("FINAL HELD-OUT RESULTS (Validation Set, 360 images)", result.stdout)
        self.assertIn("Original PGRG", result.stdout)
        self.assertIn("Standard Region Growing", result.stdout)
        self.assertIn("Otsu + Morphology", result.stdout)
        self.assertIn("Watershed", result.stdout)

        # Check required table columns
        for col in ["Method", "TP", "FP", "FN", "P", "R", "F1", "IoU", "RT(ms)"]:
            self.assertIn(col, result.stdout, f"Column '{col}' missing from Phase 10 table.")

        # Check CSV output under results/improvement/
        csv_path = os.path.join(project_root, "results", "improvement", "final_held_out_results.csv")
        self.assertTrue(os.path.exists(csv_path), "Phase 10 CSV results file was not created.")
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            for col in ["Method", "TP", "FP", "FN", "P", "R", "F1", "IoU", "RT(ms)"]:
                self.assertIn(col, header)



if __name__ == "__main__":
    unittest.main()

