"""Reconstruct full SAR scenes from patch predictions and evaluate them."""

import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch


RATE_FIELDS = (
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "iou",
    "kappa",
    "mcc",
    "miou",
    "mf1",
    "gt_change_ratio",
    "pred_change_ratio",
)

CSV_FIELDS = (
    "domain",
    "scene",
    "width",
    "height",
    "tn",
    "fp",
    "fn",
    "tp",
) + RATE_FIELDS


def _safe_div(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def binary_metrics(confusion_matrix):
    """Return change-class metrics for a 2x2 [ground truth, prediction] matrix."""
    confusion = np.asarray(confusion_matrix, dtype=np.int64)
    if confusion.shape != (2, 2):
        raise ValueError(f"expected a 2x2 confusion matrix, got {confusion.shape}")

    tn, fp, fn, tp = (int(value) for value in confusion.ravel())
    total = tn + fp + fn + tp
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    iou = _safe_div(tp, tp + fp + fn)

    background_precision = _safe_div(tn, tn + fn)
    background_recall = specificity
    background_f1 = _safe_div(
        2 * background_precision * background_recall,
        background_precision + background_recall,
    )
    background_iou = _safe_div(tn, tn + fp + fn)

    observed_agreement = _safe_div(tp + tn, total)
    expected_agreement = _safe_div(
        (tn + fp) * (tn + fn) + (fn + tp) * (fp + tp),
        total * total,
    )
    kappa = _safe_div(
        observed_agreement - expected_agreement,
        1.0 - expected_agreement,
    )

    mcc_denominator = math.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    )
    mcc = _safe_div(tp * tn - fp * fn, mcc_denominator)

    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": observed_agreement,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "iou": iou,
        "kappa": kappa,
        "mcc": mcc,
        "miou": (background_iou + iou) / 2.0,
        "mf1": (background_f1 + f1) / 2.0,
        "gt_change_ratio": _safe_div(tp + fn, total),
        "pred_change_ratio": _safe_div(tp + fp, total),
    }


def confusion_matrix_binary(ground_truth, prediction):
    ground_truth = np.asarray(ground_truth, dtype=np.uint8)
    prediction = np.asarray(prediction, dtype=np.uint8)
    if ground_truth.shape != prediction.shape:
        raise ValueError(
            f"ground truth and prediction shapes differ: "
            f"{ground_truth.shape} vs {prediction.shape}"
        )
    encoded = 2 * ground_truth.reshape(-1) + prediction.reshape(-1)
    return np.bincount(encoded, minlength=4).reshape(2, 2)


def _mean_rates(metrics_rows):
    if not metrics_rows:
        return {field: 0.0 for field in RATE_FIELDS}
    return {
        field: float(np.mean([float(row[field]) for row in metrics_rows]))
        for field in RATE_FIELDS
    }


def _safe_component(value):
    safe = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in str(value)
    )
    return safe or "unnamed"


def _uint8_image(array):
    return np.clip(np.rint(array), 0, 255).astype(np.uint8)


def _save_overview(path, image_a, image_b, ground_truth, prediction):
    panels = (
        ("Time 1 image (A)", image_a),
        ("Time 2 image (B)", image_b),
        ("Ground truth (GT)", ground_truth),
        ("Prediction (Pred)", prediction),
    )
    height, width = image_a.shape
    title_height = 26
    canvas = Image.new("RGB", (width * 2, (height + title_height) * 2), "black")
    draw = ImageDraw.Draw(canvas)
    for index, (title, panel) in enumerate(panels):
        column = index % 2
        row = index // 2
        left = column * width
        top = row * (height + title_height)
        draw.text((left + 8, top + 7), title, fill="white")
        panel_image = Image.fromarray(panel, mode="L").convert("RGB")
        canvas.paste(panel_image, (left, top + title_height))
    canvas.save(path)


class SceneStitchEvaluator:
    """Accumulate patch logits and reconstruct every held-out test scene."""

    def __init__(self, metadata_path, output_dir, n_class=2):
        if n_class != 2:
            raise ValueError("scene-level SAR evaluation currently requires n_class=2")

        self.metadata_path = Path(metadata_path)
        self.output_dir = Path(output_dir)
        if not self.metadata_path.is_file():
            raise FileNotFoundError(f"metadata does not exist: {self.metadata_path}")

        with self.metadata_path.open(newline="", encoding="utf-8") as stream:
            rows = [
                row for row in csv.DictReader(stream) if row.get("split") == "test"
            ]
        if not rows:
            raise ValueError(f"metadata has no test rows: {self.metadata_path}")

        required_fields = {
            "patch_name",
            "domain",
            "scene",
            "left",
            "top",
            "right",
            "bottom",
            "scene_width",
            "scene_height",
        }
        missing_fields = required_fields - set(rows[0])
        if missing_fields:
            raise ValueError(
                f"metadata is missing required fields: {sorted(missing_fields)}"
            )

        self.rows_by_name = {}
        for row in rows:
            name = row["patch_name"]
            if name in self.rows_by_name:
                raise ValueError(f"duplicate test patch in metadata: {name}")
            self.rows_by_name[name] = row

        self.n_class = n_class
        self.buffers = {}
        self.seen_names = set()

    def _create_buffer(self, row):
        width = int(row["scene_width"])
        height = int(row["scene_height"])
        return {
            "domain": row["domain"],
            "scene": row["scene"],
            "width": width,
            "height": height,
            "probability_sum": np.zeros(
                (self.n_class, height, width), dtype=np.float32
            ),
            "image_a_sum": np.zeros((height, width), dtype=np.float32),
            "image_b_sum": np.zeros((height, width), dtype=np.float32),
            "ground_truth": np.full((height, width), -1, dtype=np.int8),
            "coverage": np.zeros((height, width), dtype=np.uint16),
        }

    @staticmethod
    def _images_to_uint8(images):
        values = images.detach().cpu().float().numpy()
        if values.ndim != 4 or values.shape[1] != 1:
            raise ValueError(
                f"scene reconstruction requires Bx1xHxW SAR images, got {values.shape}"
            )
        return np.clip((values[:, 0] * 0.5 + 0.5) * 255.0, 0, 255)

    def add_batch(self, names, logits, images_a, images_b, labels):
        if isinstance(names, str):
            names = [names]
        names = [str(name) for name in names]

        probabilities = torch.softmax(logits.detach(), dim=1).cpu().numpy()
        image_a_values = self._images_to_uint8(images_a)
        image_b_values = self._images_to_uint8(images_b)
        label_values = labels.detach().cpu().numpy()
        if label_values.ndim == 4 and label_values.shape[1] == 1:
            label_values = label_values[:, 0]
        if label_values.ndim != 3:
            raise ValueError(f"expected Bx1xHxW labels, got {labels.shape}")
        label_values = (label_values > 0).astype(np.int8)

        batch_size = probabilities.shape[0]
        if not all(
            len(values) == batch_size
            for values in (names, image_a_values, image_b_values, label_values)
        ):
            raise ValueError("batch names, logits, images, and labels have different sizes")

        for index, name in enumerate(names):
            if name not in self.rows_by_name:
                raise KeyError(f"test patch is absent from metadata: {name}")
            if name in self.seen_names:
                raise ValueError(f"test patch was evaluated more than once: {name}")

            row = self.rows_by_name[name]
            key = (row["domain"], row["scene"])
            if key not in self.buffers:
                self.buffers[key] = self._create_buffer(row)
            buffer = self.buffers[key]

            left = int(row["left"])
            top = int(row["top"])
            right = int(row["right"])
            bottom = int(row["bottom"])
            patch_shape = (bottom - top, right - left)
            if probabilities[index].shape[1:] != patch_shape:
                raise ValueError(
                    f"prediction size for {name} is {probabilities[index].shape[1:]}, "
                    f"metadata expects {patch_shape}"
                )
            if image_a_values[index].shape != patch_shape:
                raise ValueError(f"image size does not match metadata for {name}")
            if not (
                0 <= left < right <= buffer["width"]
                and 0 <= top < bottom <= buffer["height"]
            ):
                raise ValueError(f"patch coordinates are outside the scene: {name}")

            region = np.s_[top:bottom, left:right]
            existing_labels = buffer["ground_truth"][region]
            overlap = existing_labels >= 0
            if np.any(existing_labels[overlap] != label_values[index][overlap]):
                raise ValueError(f"overlapping ground-truth patches disagree: {name}")
            existing_labels[~overlap] = label_values[index][~overlap]

            buffer["probability_sum"][:, region[0], region[1]] += probabilities[index]
            buffer["image_a_sum"][region] += image_a_values[index]
            buffer["image_b_sum"][region] += image_b_values[index]
            buffer["coverage"][region] += 1
            self.seen_names.add(name)

    def _save_scene(self, buffer, prediction, probability, metrics):
        scene_dir = (
            self.output_dir
            / _safe_component(buffer["domain"])
            / _safe_component(buffer["scene"])
        )
        scene_dir.mkdir(parents=True, exist_ok=True)

        coverage = buffer["coverage"].astype(np.float32)
        image_a = _uint8_image(buffer["image_a_sum"] / coverage)
        image_b = _uint8_image(buffer["image_b_sum"] / coverage)
        ground_truth = (buffer["ground_truth"] * 255).astype(np.uint8)
        prediction_image = (prediction * 255).astype(np.uint8)
        probability_image = _uint8_image(probability * 255.0)

        Image.fromarray(image_a, mode="L").save(scene_dir / "time1.png")
        Image.fromarray(image_b, mode="L").save(scene_dir / "time2.png")
        Image.fromarray(ground_truth, mode="L").save(scene_dir / "ground_truth.png")
        Image.fromarray(prediction_image, mode="L").save(scene_dir / "prediction.png")
        Image.fromarray(probability_image, mode="L").save(
            scene_dir / "change_probability.png"
        )
        _save_overview(
            scene_dir / "overview.png",
            image_a,
            image_b,
            ground_truth,
            prediction_image,
        )
        (scene_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def finalize(self):
        missing_names = sorted(set(self.rows_by_name) - self.seen_names)
        if missing_names:
            preview = ", ".join(missing_names[:5])
            raise ValueError(
                f"{len(missing_names)} test patches were not evaluated; first: {preview}"
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        scene_rows = []
        scene_confusions = {}
        for key in sorted(self.buffers):
            buffer = self.buffers[key]
            coverage = buffer["coverage"]
            if np.any(coverage == 0):
                missing_pixels = int(np.count_nonzero(coverage == 0))
                raise ValueError(
                    f"scene {key[0]}/{key[1]} has {missing_pixels} uncovered pixels"
                )
            if np.any(buffer["ground_truth"] < 0):
                raise ValueError(f"scene {key[0]}/{key[1]} has missing labels")

            probabilities = buffer["probability_sum"] / coverage[None, ...]
            prediction = np.argmax(probabilities, axis=0).astype(np.uint8)
            ground_truth = buffer["ground_truth"].astype(np.uint8)
            confusion = confusion_matrix_binary(ground_truth, prediction)
            metrics = binary_metrics(confusion)
            metrics.update(
                {
                    "domain": buffer["domain"],
                    "scene": buffer["scene"],
                    "width": buffer["width"],
                    "height": buffer["height"],
                }
            )
            scene_rows.append(metrics)
            scene_confusions[key] = confusion
            self._save_scene(buffer, prediction, probabilities[1], metrics)

        with (self.output_dir / "scene_metrics.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows({field: row[field] for field in CSV_FIELDS} for row in scene_rows)

        domain_rows = defaultdict(list)
        domain_confusions = defaultdict(lambda: np.zeros((2, 2), dtype=np.int64))
        for row in scene_rows:
            domain_rows[row["domain"]].append(row)
            domain_confusions[row["domain"]] += scene_confusions[
                (row["domain"], row["scene"])
            ]

        domains = {}
        for domain in sorted(domain_rows):
            domains[domain] = {
                "scene_count": len(domain_rows[domain]),
                "scene_macro_mean": _mean_rates(domain_rows[domain]),
                "pixel_global": binary_metrics(domain_confusions[domain]),
            }

        global_confusion = sum(
            scene_confusions.values(), np.zeros((2, 2), dtype=np.int64)
        )
        summary = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "metadata_path": str(self.metadata_path),
            "scene_count": len(scene_rows),
            "patch_count": len(self.seen_names),
            "scene_macro_mean": _mean_rates(scene_rows),
            "pixel_global": binary_metrics(global_confusion),
            "domains": domains,
        }
        (self.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return summary
