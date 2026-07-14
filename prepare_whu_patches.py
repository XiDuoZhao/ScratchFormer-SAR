"""Create the 256x256 WHU-CD patches and paper train/val/test split."""

from argparse import ArgumentParser
from pathlib import Path
from random import Random
import subprocess

import numpy as np
from PIL import Image
import tifffile


AREAS = ('train', 'test')
EXPECTED_SPLIT_COUNTS = {'train': 5947, 'val': 743, 'test': 744}

# WHU-CD source GeoTIFFs exceed Pillow's conservative decompression-bomb limit.
Image.MAX_IMAGE_PIXELS = None


def get_source_paths(source_root, area):
    image_root = source_root / '1. The two-period image data'
    return {
        'A': image_root / '2012' / 'whole_image' / area / 'image' / f'2012_{area}.tif',
        'B': image_root / '2016' / 'whole_image' / area / 'image' / f'2016_{area}.tif',
        'label': image_root / 'change_label' / area / 'change_label.tif',
    }


def save_source_patches(source_path, output_dir, area, patch_size, rgb):
    """Decode a tiled LZW GeoTIFF without ever loading the whole image."""
    with tifffile.TiffFile(source_path) as tif:
        page = tif.pages[0]
        height, width = page.shape[:2]
        channels = 3 if rgb else 1
        if (page.tilewidth, page.tilelength) != (128, 128):
            raise ValueError(f'Unexpected TIFF tile size: {page.tilewidth}x{page.tilelength}')

        usable_width = width - width % patch_size
        usable_height = height - height % patch_size
        pending = {}
        # A single decoder keeps peak RAM small on AutoDL instances.
        for data, index, _ in page.segments(maxworkers=1):
            top, left = index[2], index[3]
            if top + page.tilelength > usable_height or left + page.tilewidth > usable_width:
                continue

            patch_top = top - top % patch_size
            patch_left = left - left % patch_size
            key = (patch_top, patch_left)
            patch, tiles = pending.get(
                key, (np.zeros((patch_size, patch_size, channels), dtype=np.uint8), 0)
            )
            tile = data[0]
            if rgb:
                patch[top % patch_size:top % patch_size + page.tilelength,
                      left % patch_size:left % patch_size + page.tilewidth] = tile
            else:
                patch[top % patch_size:top % patch_size + page.tilelength,
                      left % patch_size:left % patch_size + page.tilewidth, 0] = tile[..., 0] * 255
            tiles += 1
            if tiles == 4:
                name = f'whu_{area}_{patch_top:05d}_{patch_left:05d}.png'
                image = Image.fromarray(patch if rgb else patch[..., 0], mode='RGB' if rgb else 'L')
                image.save(output_dir / name, compress_level=1)
                del pending[key]
            else:
                pending[key] = (patch, tiles)
        if pending:
            raise ValueError(f'Incomplete 256x256 patches remain for {source_path}')

        return [
            f'whu_{area}_{top:05d}_{left:05d}.png'
            for top in range(0, usable_height, patch_size)
            for left in range(0, usable_width, patch_size)
        ]


def build_converter():
    source = Path(__file__).with_name('whu_tiff_to_png.c')
    executable = Path('/tmp/whu_tiff_to_png')
    subprocess.run([
        'gcc', '-O3', '-o', str(executable), str(source), '-ltiff', '-lpng16', '-lz'
    ], check=True)
    return executable


def save_area_patches(source_root, output_root, area, patch_size, converter):
    paths = get_source_paths(source_root, area)
    names = None
    for source_name, rgb in (('A', True), ('B', True), ('label', False)):
        subprocess.run([
            str(converter), str(paths[source_name]), str(output_root / source_name), area,
            'rgb' if rgb else 'label'
        ], check=True)
        with tifffile.TiffFile(paths[source_name]) as tif:
            height, width = tif.pages[0].shape[:2]
        source_names = [
            f'whu_{area}_{top:05d}_{left:05d}.png'
            for top in range(0, height - height % patch_size, patch_size)
            for left in range(0, width - width % patch_size, patch_size)
        ]
        if names is None:
            names = source_names
        elif names != source_names:
            raise ValueError(f'Mismatched patch coordinates for {area}')
    return names


def main():
    parser = ArgumentParser()
    parser.add_argument('--source_root', default='./datasets/whu-cd')
    parser.add_argument('--output_root', default='./datasets/CD/WHU-CD-256-patches')
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.patch_size != 256:
        raise ValueError('The paper reproduction protocol requires --patch_size 256.')

    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    if output_root.exists():
        raise FileExistsError(f'Output directory already exists: {output_root}')

    for name in ('A', 'B', 'label', 'list'):
        (output_root / name).mkdir(parents=True, exist_ok=False)

    converter = build_converter()
    all_patch_names = []
    for area in AREAS:
        patch_names = save_area_patches(source_root, output_root, area, args.patch_size, converter)
        all_patch_names.extend(patch_names)
        print(f'{area} source area: {len(patch_names)} patches')

    expected_total = sum(EXPECTED_SPLIT_COUNTS.values())
    if len(all_patch_names) != expected_total:
        raise ValueError(f'Created {len(all_patch_names)} patches; expected {expected_total}')

    Random(args.seed).shuffle(all_patch_names)
    start = 0
    for split, count in EXPECTED_SPLIT_COUNTS.items():
        names = all_patch_names[start:start + count]
        (output_root / 'list' / f'{split}.txt').write_text('\n'.join(names) + '\n')
        print(f'{split}: {len(names)} patches')
        start += count


if __name__ == '__main__':
    main()
