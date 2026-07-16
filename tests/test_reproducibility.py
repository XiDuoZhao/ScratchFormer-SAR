import random
import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data import DataLoader, Dataset

import utils
from models.trainer import CDTrainer


class _TinyDataset(Dataset):

    def __init__(self, root_dir, name):
        self.root_dir = str(root_dir)
        self.name = name

    def __len__(self):
        return 1

    def __getitem__(self, index):
        image_a = torch.zeros(1, 2, 2)
        image_b = torch.ones(1, 2, 2) * 0.25
        label = torch.tensor([[[0, 0], [0, 1]]], dtype=torch.uint8)
        return {"name": self.name, "A": image_a, "B": image_b, "L": label}


class _TinyModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.classifier = torch.nn.Conv2d(2, 2, kernel_size=1)

    def forward(self, image_a, image_b):
        return [self.classifier(torch.cat((image_a, image_b), dim=1))]


class ReproducibilityTest(unittest.TestCase):

    def test_global_seed_repeats_python_numpy_and_torch(self):
        utils.set_random_seed(123)
        first = (random.random(), np.random.rand(), torch.rand(3))
        utils.set_random_seed(123)
        second = (random.random(), np.random.rand(), torch.rand(3))

        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        self.assertTrue(torch.equal(first[2], second[2]))

    def test_train_is_shuffled_but_validation_is_sequential(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "list").mkdir()
            names = "one.png\ntwo.png\n"
            (root / "list" / "train.txt").write_text(names, encoding="utf-8")
            (root / "list" / "val.txt").write_text(names, encoding="utf-8")
            args = SimpleNamespace(
                data_name="SAR",
                data_root=str(root),
                split="train",
                split_val="val",
                dataset="CDDataset",
                img_mode="L",
                img_size=128,
                batch_size=1,
                num_workers=0,
                seed=17,
            )

            loaders = utils.get_loaders(args)

            self.assertIsInstance(loaders["train"].sampler, RandomSampler)
            self.assertIsInstance(loaders["val"].sampler, SequentialSampler)
            expected_train_state = utils.seeded_generator(17).get_state()
            expected_val_state = utils.seeded_generator(18).get_state()
            self.assertTrue(
                torch.equal(loaders["train"].generator.get_state(), expected_train_state)
            )
            self.assertTrue(
                torch.equal(loaders["val"].generator.get_state(), expected_val_state)
            )

    def test_trainer_uses_stitched_validation_and_saves_rng_state(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint_dir = root / "checkpoints"
            checkpoint_dir.mkdir()
            metadata_path = root / "metadata.csv"
            fields = (
                "patch_name", "split", "domain", "scene", "left", "top",
                "right", "bottom", "scene_width", "scene_height",
            )
            with metadata_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    {
                        "patch_name": "val.png", "split": "val",
                        "domain": "Synthetic", "scene": "Scene1",
                        "left": 0, "top": 0, "right": 2, "bottom": 2,
                        "scene_width": 2, "scene_height": 2,
                    }
                )

            train_loader = DataLoader(
                _TinyDataset(root, "train.png"), batch_size=1, shuffle=True,
                generator=utils.seeded_generator(5),
            )
            val_loader = DataLoader(
                _TinyDataset(root, "val.png"), batch_size=1, shuffle=False,
                generator=utils.seeded_generator(6),
            )
            args = SimpleNamespace(
                n_class=2, gpu_ids=[], lr=1e-3, optimizer="adamw",
                checkpoint_dir=str(checkpoint_dir), vis_dir=str(root / "vis"),
                batch_size=1, max_epochs=1, shuffle_AB=False,
                multi_scale_train=False, multi_scale_infer=False,
                multi_pred_weights=[1.0], loss="ce", pretrain=None,
                selection_metric="F1_1", scene_metadata=str(metadata_path),
                val_eval_mode="auto", seed=5,
            )

            scheduler_factory = lambda optimizer, args: (
                torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
            )
            with patch("models.trainer.define_G", return_value=_TinyModel()), \
                    patch("models.trainer.get_scheduler", side_effect=scheduler_factory):
                trainer = CDTrainer(args, {"train": train_loader, "val": val_loader})
                trainer.train_models()

            checkpoint = torch.load(
                checkpoint_dir / "best_ckpt.pt", map_location="cpu"
            )
            self.assertEqual(trainer.val_eval_mode, "scene")
            self.assertEqual(checkpoint["seed"], 5)
            self.assertEqual(checkpoint["val_eval_mode"], "scene")
            self.assertIn("rng_state", checkpoint)
            self.assertEqual(
                checkpoint["validation_scene_summary"]["split"], "val"
            )
            self.assertTrue(
                (checkpoint_dir / "validation_scene_evaluation" / "summary.json").is_file()
            )


if __name__ == "__main__":
    unittest.main()
