import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from models.losses import (
    WeightedCrossEntropyDiceLoss,
    get_dataset_class_counts,
    sqrt_inverse_frequency_weights,
)


class _DatasetStub:

    def __init__(self, root_dir, names):
        self.root_dir = str(root_dir)
        self.img_name_list = names
        self.label_transform = "norm"


class ImbalanceStrategyTest(unittest.TestCase):

    def test_counts_raw_training_labels_without_augmentation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "label").mkdir()
            labels = {
                "one.png": np.array([[0, 255], [0, 255]], dtype=np.uint8),
                "two.png": np.array([[0, 0], [0, 255]], dtype=np.uint8),
            }
            for name, label in labels.items():
                Image.fromarray(label, mode="L").save(root / "label" / name)

            counts = get_dataset_class_counts(
                _DatasetStub(root, list(labels)), n_classes=2
            )

            np.testing.assert_array_equal(counts, np.array([5, 3]))

    def test_sqrt_inverse_weights_are_moderate_and_mean_normalized(self):
        weights = sqrt_inverse_frequency_weights([95, 5])

        self.assertAlmostEqual(float(weights.mean()), 1.0)
        self.assertAlmostEqual(
            float(weights[1] / weights[0]),
            float(np.sqrt(95.0 / 5.0)),
            places=6,
        )

    def test_ce_dice_prefers_correct_logits_and_backpropagates(self):
        target = torch.tensor([[[[0, 0], [1, 1]]]], dtype=torch.long)
        correct_logits = torch.tensor(
            [[[[6.0, 6.0], [-6.0, -6.0]],
              [[-6.0, -6.0], [6.0, 6.0]]]],
            requires_grad=True,
        )
        wrong_logits = -correct_logits.detach()
        loss_function = WeightedCrossEntropyDiceLoss([0.5, 1.5])

        correct_loss = loss_function(correct_logits, target)
        wrong_loss = loss_function(wrong_logits, target)
        correct_loss.backward()

        self.assertLess(float(correct_loss), float(wrong_loss))
        self.assertTrue(torch.isfinite(correct_logits.grad).all())


if __name__ == "__main__":
    unittest.main()
