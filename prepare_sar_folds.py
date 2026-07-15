#!/usr/bin/env python3
"""Generate four leakage-aware cross-domain SAR change-detection folds."""

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from verify_sar_folds import verify_dataset


SPLITS = ("train", "val", "test")
METADATA_FIELDS = (
    "patch_name",
    "fold",
    "split",
    "domain",
    "scene",
    "region",
    "left",
    "top",
    "right",
    "bottom",
    "region_left",
    "region_top",
    "region_right",
    "region_bottom",
    "scene_width",
    "scene_height",
    "change_pixels",
    "total_pixels",
    "change_ratio",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Build the four small-scene SAR folds.")
    parser.add_argument(
        "--domains-root",
        type=Path,
        default=Path("datasets/CD/SAR-CD-128/domains"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("datasets/CD/SAR-CD-128/folds"),
    )
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_scenes(domains_root):
    scenes = []
    for domain_path in sorted(path for path in domains_root.iterdir() if path.is_dir()):
        domain_config = json.loads((domain_path / "domain.json").read_text(encoding="utf-8"))
        for scene_name in domain_config["scenes"]:
            scene_dir = domain_path / "scenes" / scene_name
            images = {}
            for name in ("A", "B", "label"):
                with Image.open(scene_dir / f"{name}.png") as image:
                    images[name] = image.copy()
            sizes = {image.size for image in images.values()}
            if len(sizes) != 1:
                raise ValueError(f"scene size mismatch: {domain_path.name}/{scene_name}")
            label = np.asarray(images["label"])
            if not set(np.unique(label).tolist()) <= {0, 255}:
                raise ValueError(f"non-binary scene label: {domain_path.name}/{scene_name}")
            scenes.append(
                {
                    "domain": domain_path.name,
                    "scene": scene_name,
                    "outer_test_fold": int(domain_config["outer_test_fold"]),
                    "images": images,
                    "label": label >= 128,
                }
            )
    return scenes


def quadrant_boxes(width, height):
    middle_x = width // 2
    middle_y = height // 2
    return {
        "top-left": (0, 0, middle_x, middle_y),
        "top-right": (middle_x, 0, width, middle_y),
        "bottom-left": (0, middle_y, middle_x, height),
        "bottom-right": (middle_x, middle_y, width, height),
    }


def choose_validation_quadrant(label):
    height, width = label.shape
    global_ratio = float(label.mean())
    candidates = []
    for index, (name, box) in enumerate(quadrant_boxes(width, height).items()):
        left, top, right, bottom = box
        region = label[top:bottom, left:right]
        changed = int(np.count_nonzero(region))
        ratio = float(region.mean())
        candidates.append((changed == 0, abs(ratio - global_ratio), index, name))
    return min(candidates)[3]


def sliding_positions(start, end, patch_size, stride):
    length = end - start
    if length < patch_size:
        raise ValueError(
            f"region length {length} is smaller than patch size {patch_size}"
        )
    last = end - patch_size
    positions = list(range(start, last + 1, stride))
    if positions[-1] != last:
        positions.append(last)
    return positions


def iter_patch_boxes(region, patch_size, stride):
    left, top, right, bottom = region
    for patch_top in sliding_positions(top, bottom, patch_size, stride):
        for patch_left in sliding_positions(left, right, patch_size, stride):
            yield (
                patch_left,
                patch_top,
                patch_left + patch_size,
                patch_top + patch_size,
            )


def patch_name(fold, split, domain, scene, left, top):
    safe_domain = domain.lower().replace("-", "_")
    safe_scene = scene.lower().replace("-", "_")
    return (
        f"f{fold}_{split}_{safe_domain}_{safe_scene}_"
        f"y{top:04d}_x{left:04d}.png"
    )


def save_patch(fold_dir, scene, fold, split, region_name, region, box):
    left, top, right, bottom = box
    name = patch_name(
        fold, split, scene["domain"], scene["scene"], left, top
    )
    for image_name in ("A", "B", "label"):
        scene["images"][image_name].crop(box).save(
            fold_dir / image_name / name, format="PNG", compress_level=1
        )

    label_patch = scene["label"][top:bottom, left:right]
    changed_pixels = int(np.count_nonzero(label_patch))
    total_pixels = int(label_patch.size)
    return {
        "patch_name": name,
        "fold": fold,
        "split": split,
        "domain": scene["domain"],
        "scene": scene["scene"],
        "region": region_name,
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
        "region_left": region[0],
        "region_top": region[1],
        "region_right": region[2],
        "region_bottom": region[3],
        "scene_width": scene["images"]["A"].size[0],
        "scene_height": scene["images"]["A"].size[1],
        "change_pixels": changed_pixels,
        "total_pixels": total_pixels,
        "change_ratio": changed_pixels / total_pixels,
    }


def build_fold(output_root, scenes, fold, patch_size, stride):
    fold_dir = output_root / f"fold_{fold}"
    for folder in ("A", "B", "label", "list"):
        (fold_dir / folder).mkdir(parents=True, exist_ok=False)

    test_domains = {scene["domain"] for scene in scenes if scene["outer_test_fold"] == fold}
    if len(test_domains) != 1:
        raise ValueError(f"fold {fold} must have exactly one test domain: {test_domains}")
    test_domain = next(iter(test_domains))
    rows = []
    validation_regions = {}

    for scene in scenes:
        width, height = scene["images"]["A"].size
        if scene["domain"] == test_domain:
            region = (0, 0, width, height)
            for box in iter_patch_boxes(region, patch_size, stride):
                rows.append(
                    save_patch(fold_dir, scene, fold, "test", "full", region, box)
                )
            continue

        quadrants = quadrant_boxes(width, height)
        val_quadrant = choose_validation_quadrant(scene["label"])
        validation_regions[f"{scene['domain']}/{scene['scene']}"] = val_quadrant
        for region_name, region in quadrants.items():
            split = "val" if region_name == val_quadrant else "train"
            for box in iter_patch_boxes(region, patch_size, stride):
                rows.append(
                    save_patch(
                        fold_dir, scene, fold, split, region_name, region, box
                    )
                )

    rows_by_split = {
        split: sorted(
            (row for row in rows if row["split"] == split),
            key=lambda row: row["patch_name"],
        )
        for split in SPLITS
    }
    for split, split_rows in rows_by_split.items():
        names = [row["patch_name"] for row in split_rows]
        (fold_dir / "list" / f"{split}.txt").write_text(
            "\n".join(names) + "\n", encoding="utf-8"
        )

    with (fold_dir / "metadata.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    counts = {split: len(split_rows) for split, split_rows in rows_by_split.items()}
    protocol = {
        "fold": fold,
        "test_domain": test_domain,
        "train_domains": sorted(
            {scene["domain"] for scene in scenes if scene["domain"] != test_domain}
        ),
        "patch_size": patch_size,
        "stride": stride,
        "validation_strategy": (
            "One contiguous 2x2 spatial quadrant per training scene; choose a "
            "non-empty quadrant whose change ratio is closest to the full scene."
        ),
        "validation_quadrants": validation_regions,
        "counts": counts,
    }
    (fold_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    return protocol


def main():
    args = parse_args()
    if args.patch_size <= 0 or args.stride <= 0:
        raise ValueError("patch size and stride must be positive")
    if not args.domains_root.is_dir():
        raise FileNotFoundError(f"domains root does not exist: {args.domains_root}")
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"output already exists: {args.output_root}; use --overwrite to replace it"
            )
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True)

    scenes = load_scenes(args.domains_root)
    folds = [
        build_fold(args.output_root, scenes, fold, args.patch_size, args.stride)
        for fold in range(1, 5)
    ]
    summary = {
        "domains": sorted({scene["domain"] for scene in scenes}),
        "scenes": sorted(scene["scene"] for scene in scenes),
        "fold_count": len(folds),
        "patch_size": args.patch_size,
        "stride": args.stride,
    }
    (args.output_root / "folds_manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )

    results = verify_dataset(args.output_root)
    for fold_name, counts in results.items():
        print(
            f"{fold_name}: train={counts['train']} val={counts['val']} "
            f"test={counts['test']} verified"
        )


if __name__ == "__main__":
    main()
