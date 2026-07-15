import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from misc.scene_evaluation import SceneStitchEvaluator


class SceneStitchEvaluatorTest(unittest.TestCase):

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.metadata_path = self.root / "metadata.csv"
        self.output_dir = self.root / "results"
        self.positions = ((0, 0), (1, 0), (0, 1), (1, 1))
        fieldnames = (
            "patch_name",
            "split",
            "domain",
            "scene",
            "left",
            "top",
            "right",
            "bottom",
            "scene_width",
            "scene_height",
        )
        with self.metadata_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for index, (left, top) in enumerate(self.positions):
                writer.writerow(
                    {
                        "patch_name": f"patch_{index}.png",
                        "split": "test",
                        "domain": "Synthetic",
                        "scene": "Scene1",
                        "left": left,
                        "top": top,
                        "right": left + 3,
                        "bottom": top + 3,
                        "scene_width": 4,
                        "scene_height": 4,
                    }
                )

    def tearDown(self):
        self.temporary_directory.cleanup()

    @staticmethod
    def _normalized_tensor(patch):
        values = torch.from_numpy(patch.astype(np.float32))[None, None]
        return values / 255.0 * 2.0 - 1.0

    def test_reconstructs_overlapping_patches_and_writes_metrics(self):
        image_a = np.arange(16, dtype=np.uint8).reshape(4, 4) * 16
        image_b = 255 - image_a
        ground_truth = np.array(
            [
                [0, 0, 0, 0],
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.uint8,
        )
        evaluator = SceneStitchEvaluator(self.metadata_path, self.output_dir)

        for index, (left, top) in enumerate(self.positions):
            region = np.s_[top:top + 3, left:left + 3]
            label_patch = ground_truth[region]
            logits = np.full((1, 2, 3, 3), -6.0, dtype=np.float32)
            logits[0, 0][label_patch == 0] = 6.0
            logits[0, 1][label_patch == 1] = 6.0
            evaluator.add_batch(
                names=[f"patch_{index}.png"],
                logits=torch.from_numpy(logits),
                images_a=self._normalized_tensor(image_a[region]),
                images_b=self._normalized_tensor(image_b[region]),
                labels=torch.from_numpy(label_patch)[None, None],
            )

        summary = evaluator.finalize()
        scene_dir = self.output_dir / "Synthetic" / "Scene1"
        reconstructed_a = np.asarray(Image.open(scene_dir / "time1.png"))
        prediction = np.asarray(Image.open(scene_dir / "prediction.png"))

        np.testing.assert_array_equal(reconstructed_a, image_a)
        np.testing.assert_array_equal(prediction, ground_truth * 255)
        self.assertEqual(summary["patch_count"], 4)
        self.assertEqual(summary["scene_count"], 1)
        self.assertAlmostEqual(summary["pixel_global"]["f1"], 1.0)
        self.assertAlmostEqual(summary["pixel_global"]["iou"], 1.0)
        self.assertTrue((self.output_dir / "scene_metrics.csv").is_file())
        self.assertTrue((self.output_dir / "summary.json").is_file())
        self.assertTrue((scene_dir / "overview.png").is_file())

    def test_finalize_rejects_missing_test_patches(self):
        evaluator = SceneStitchEvaluator(self.metadata_path, self.output_dir)
        with self.assertRaisesRegex(ValueError, "test patches were not evaluated"):
            evaluator.finalize()


if __name__ == "__main__":
    unittest.main()
