#!/usr/bin/env python3
"""Verify SAR fold integrity, spatial isolation, and full test coverage."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


SPLITS = ("train", "val", "test")


def parse_args():
    parser = argparse.ArgumentParser(description="Verify generated SAR cross-domain folds.")
    parser.add_argument(
        "--folds-root", type=Path, default=Path("datasets/CD/SAR-CD-128/folds")
    )
    return parser.parse_args()


def read_list(path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def read_metadata(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def image_digest(paths):
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def boxes_overlap(first, second):
    return (
        int(first["left"]) < int(second["right"])
        and int(second["left"]) < int(first["right"])
        and int(first["top"]) < int(second["bottom"])
        and int(second["top"]) < int(first["bottom"])
    )


def verify_images(fold_dir, rows, patch_size):
    digests = {}
    for row in rows:
        name = row["patch_name"]
        paths = [fold_dir / folder / name for folder in ("A", "B", "label")]
        if not all(path.is_file() for path in paths):
            raise FileNotFoundError(f"missing A/B/label files for {fold_dir.name}/{name}")

        with Image.open(paths[0]) as image_a, Image.open(paths[1]) as image_b:
            if image_a.mode != "L" or image_b.mode != "L":
                raise ValueError(f"non-grayscale SAR patch: {fold_dir.name}/{name}")
            if image_a.size != (patch_size, patch_size) or image_b.size != image_a.size:
                raise ValueError(f"invalid image size: {fold_dir.name}/{name}")
        with Image.open(paths[2]) as label_image:
            if label_image.size != (patch_size, patch_size):
                raise ValueError(f"invalid label size: {fold_dir.name}/{name}")
            values = set(np.unique(np.asarray(label_image)).tolist())
            if not values <= {0, 255}:
                raise ValueError(
                    f"non-binary label in {fold_dir.name}/{name}: {sorted(values)}"
                )

        digest = image_digest(paths)
        split = row["split"]
        if digest in digests and digests[digest] != split:
            raise ValueError(
                f"identical patch content crosses splits in {fold_dir.name}: {name}"
            )
        digests[digest] = split


def verify_spatial_isolation(fold_dir, rows):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[(row["domain"], row["scene"])][row["split"]].append(row)
    for (domain, scene), scene_rows in grouped.items():
        for train_row in scene_rows["train"]:
            for val_row in scene_rows["val"]:
                if boxes_overlap(train_row, val_row):
                    raise ValueError(
                        f"train/val spatial overlap in {fold_dir.name}/{scene}: "
                        f"{domain}/{train_row['patch_name']} vs {val_row['patch_name']}"
                    )


def verify_test_coverage(fold_dir, rows):
    grouped = defaultdict(list)
    for row in rows:
        if row["split"] == "test":
            grouped[(row["domain"], row["scene"])].append(row)
    if not grouped:
        raise ValueError(f"no test patches in {fold_dir}")

    for (domain, scene), scene_rows in grouped.items():
        height = int(scene_rows[0]["scene_height"])
        width = int(scene_rows[0]["scene_width"])
        coverage = np.zeros((height, width), dtype=np.uint16)
        for row in scene_rows:
            coverage[
                int(row["top"]):int(row["bottom"]),
                int(row["left"]):int(row["right"]),
            ] += 1
        if np.any(coverage == 0):
            missing = int(np.count_nonzero(coverage == 0))
            raise ValueError(
                f"test scene is not fully covered in {fold_dir.name}/{domain}/{scene}: "
                f"{missing} pixels"
            )


def verify_fold(fold_dir):
    protocol_path = fold_dir / "protocol.json"
    metadata_path = fold_dir / "metadata.csv"
    if not protocol_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"missing protocol or metadata in {fold_dir}")

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    rows = read_metadata(metadata_path)
    if not rows:
        raise ValueError(f"empty metadata: {metadata_path}")
    invalid_splits = sorted({row["split"] for row in rows} - set(SPLITS))
    if invalid_splits:
        raise ValueError(f"invalid metadata splits in {fold_dir.name}: {invalid_splits}")

    patch_size = int(protocol["patch_size"])
    rows_by_split = {split: [row for row in rows if row["split"] == split] for split in SPLITS}
    metadata_names = {split: {row["patch_name"] for row in split_rows}
                      for split, split_rows in rows_by_split.items()}

    for split in SPLITS:
        list_names = read_list(fold_dir / "list" / f"{split}.txt")
        if len(list_names) != len(set(list_names)):
            raise ValueError(f"duplicate names in {fold_dir.name}/{split}.txt")
        if set(list_names) != metadata_names[split]:
            raise ValueError(f"list and metadata differ in {fold_dir.name}/{split}")

    if any(metadata_names[a] & metadata_names[b]
           for index, a in enumerate(SPLITS) for b in SPLITS[index + 1:]):
        raise ValueError(f"patch names cross splits in {fold_dir.name}")

    test_domain = protocol["test_domain"]
    expected_train_domains = set(protocol["train_domains"])
    actual_test_domains = {row["domain"] for row in rows_by_split["test"]}
    actual_train_domains = {
        row["domain"]
        for split in ("train", "val")
        for row in rows_by_split[split]
    }
    if actual_test_domains != {test_domain}:
        raise ValueError(
            f"test domains differ from protocol in {fold_dir.name}: {actual_test_domains}"
        )
    if actual_train_domains != expected_train_domains:
        raise ValueError(
            f"training domains differ from protocol in {fold_dir.name}: "
            f"{actual_train_domains}"
        )
    for row in rows_by_split["test"]:
        if row["domain"] != test_domain:
            raise ValueError(f"non-test domain in test split: {fold_dir.name}")
    for split in ("train", "val"):
        for row in rows_by_split[split]:
            if row["domain"] == test_domain:
                raise ValueError(f"test domain leaked into {split}: {fold_dir.name}")

    actual_counts = {split: len(rows_by_split[split]) for split in SPLITS}
    if actual_counts != protocol["counts"]:
        raise ValueError(
            f"metadata counts differ from protocol in {fold_dir.name}: {actual_counts}"
        )

    verify_images(fold_dir, rows, patch_size)
    verify_spatial_isolation(fold_dir, rows)
    verify_test_coverage(fold_dir, rows)
    return actual_counts


def verify_dataset(folds_root):
    fold_dirs = sorted(path for path in folds_root.glob("fold_*") if path.is_dir())
    if len(fold_dirs) != 4:
        raise ValueError(f"expected four folds in {folds_root}, found {len(fold_dirs)}")
    results = {}
    for fold_dir in fold_dirs:
        results[fold_dir.name] = verify_fold(fold_dir)
    return results


def main():
    args = parse_args()
    results = verify_dataset(args.folds_root)
    for fold, counts in results.items():
        print(
            f"{fold}: train={counts['train']} val={counts['val']} "
            f"test={counts['test']} verified"
        )


if __name__ == "__main__":
    main()
