#!/usr/bin/env python3
"""Build a provenance-aware scene registry for the small SAR CD datasets."""

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


SCENES = (
    {
        "domain": "Ottawa",
        "scene": "Ottawa",
        "fold": 1,
        "a": "1997.05.bmp",
        "b": "1997.08.bmp",
        "label": "reference1.bmp",
        "source": "Local collection; original provenance must be cited separately.",
    },
    {
        "domain": "Bern",
        "scene": "Bern",
        "fold": 2,
        "a": "1999.04.bmp",
        "b": "1999.05.bmp",
        "label": "reference.bmp",
        "source": "Local collection; original provenance must be cited separately.",
    },
    {
        "domain": "SanFrancisco",
        "scene": "SanFrancisco",
        "fold": 3,
        "a": "san_1.bmp",
        "b": "san_2.bmp",
        "label": "san_gt.bmp",
        "source": "Local collection; original provenance must be cited separately.",
    },
    {
        "domain": "YellowRiver",
        "scene": "Yellow-A",
        "fold": 4,
        "a": "im11.bmp",
        "b": "im22.bmp",
        "label": "im33_rf.bmp",
        "source": "https://github.com/yulisun/INLPG/tree/master/datasets",
    },
    {
        "domain": "YellowRiver",
        "scene": "Yellow-B",
        "fold": 4,
        "a": "img1.bmp",
        "b": "img2.bmp",
        "label": "img_rf.bmp",
        "source": "https://github.com/yulisun/INLPG/tree/master/datasets",
    },
    {
        "domain": "YellowRiver",
        "scene": "Yellow-C",
        "fold": 4,
        "a": "Yellow River1.bmp",
        "b": "Yellow River2.bmp",
        "label": "Yellow River_rf.bmp",
        "source": "https://github.com/yulisun/INLPG/tree/master/datasets",
    },
    {
        "domain": "YellowRiver",
        "scene": "Yellow-D",
        "fold": 4,
        "a": "img11.bmp",
        "b": "img22.bmp",
        "label": "img33_rf.bmp",
        "source": "https://github.com/yulisun/INLPG/tree/master/datasets",
    },
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert the verified small SAR scenes into domain directories."
    )
    parser.add_argument("--source", type=Path, default=Path("datasets/CD/sar"))
    parser.add_argument(
        "--output", type=Path, default=Path("datasets/CD/SAR-CD-128/domains")
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing output directory"
    )
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_grayscale(path):
    with Image.open(path) as image:
        array = np.asarray(image)
        if array.ndim == 3:
            if array.shape[2] < 3:
                raise ValueError(f"unsupported channel layout: {path} {array.shape}")
            if not (
                np.array_equal(array[..., 0], array[..., 1])
                and np.array_equal(array[..., 1], array[..., 2])
            ):
                raise ValueError(f"expected grayscale SAR stored as RGB: {path}")
            array = array[..., 0]
        if array.ndim != 2:
            raise ValueError(f"expected a 2-D image: {path} {array.shape}")
        return array.astype(np.uint8, copy=False)


def save_png(array, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="L").save(path, format="PNG", optimize=True)


def build_scene(source_root, output_root, scene_config):
    source_paths = {
        key: source_root / scene_config[key] for key in ("a", "b", "label")
    }
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing source files: " + ", ".join(missing))

    image_a = read_grayscale(source_paths["a"])
    image_b = read_grayscale(source_paths["b"])
    label_source = read_grayscale(source_paths["label"])
    if image_a.shape != image_b.shape or image_a.shape != label_source.shape:
        raise ValueError(
            f"shape mismatch for {scene_config['scene']}: "
            f"A={image_a.shape}, B={image_b.shape}, label={label_source.shape}"
        )

    label = (label_source >= 128).astype(np.uint8) * 255
    scene_dir = (
        output_root
        / scene_config["domain"]
        / "scenes"
        / scene_config["scene"]
    )
    save_png(image_a, scene_dir / "A.png")
    save_png(image_b, scene_dir / "B.png")
    save_png(label, scene_dir / "label.png")

    changed_pixels = int(np.count_nonzero(label))
    total_pixels = int(label.size)
    metadata = {
        "domain": scene_config["domain"],
        "scene": scene_config["scene"],
        "outer_test_fold": scene_config["fold"],
        "width": int(image_a.shape[1]),
        "height": int(image_a.shape[0]),
        "changed_pixels": changed_pixels,
        "total_pixels": total_pixels,
        "change_ratio": changed_pixels / total_pixels,
        "output_format": "8-bit single-channel PNG",
        "label_values": [0, 255],
        "label_threshold": 128,
        "source_reference": scene_config["source"],
        "source_files": {
            key: {
                "name": path.name,
                "sha256": sha256(path),
            }
            for key, path in source_paths.items()
        },
    }
    with (scene_dir / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=True)
        stream.write("\n")
    return metadata


def write_domain_metadata(output_root, rows):
    domains = {}
    for row in rows:
        domains.setdefault(row["domain"], []).append(row["scene"])
    for domain, scenes in domains.items():
        payload = {
            "domain": domain,
            "outer_test_fold": next(
                row["outer_test_fold"] for row in rows if row["domain"] == domain
            ),
            "scenes": scenes,
            "scene_count": len(scenes),
        }
        path = output_root / domain / "domain.json"
        with path.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=True)
            stream.write("\n")


def write_manifest(output_root, rows):
    fieldnames = [
        "domain",
        "scene",
        "outer_test_fold",
        "width",
        "height",
        "changed_pixels",
        "total_pixels",
        "change_ratio",
        "source_reference",
    ]
    with (output_root / "manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fieldnames} for row in rows)


def main():
    args = parse_args()
    if not args.source.is_dir():
        raise FileNotFoundError(f"source directory does not exist: {args.source}")
    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"output already exists: {args.output}; use --overwrite to replace it"
            )
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)

    rows = [build_scene(args.source, args.output, config) for config in SCENES]
    write_domain_metadata(args.output, rows)
    write_manifest(args.output, rows)
    print(f"Built {len(rows)} scenes in {args.output}")
    for row in rows:
        print(
            f"{row['domain']:12s} {row['scene']:12s} "
            f"{row['width']}x{row['height']} change={row['change_ratio']:.4%}"
        )


if __name__ == "__main__":
    main()
