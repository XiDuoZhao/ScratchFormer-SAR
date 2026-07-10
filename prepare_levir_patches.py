"""Create the non-overlapping 256x256 LEVIR-CD patches used in the paper."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image


def read_split(path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def create_patches(task):
    source_root, output_root, patch_size, image_name = task
    source_root = Path(source_root)
    output_root = Path(output_root)
    paths = {name: source_root / name / image_name for name in ('A', 'B', 'label')}
    images = {name: Image.open(path).copy() for name, path in paths.items()}
    sizes = {image.size for image in images.values()}
    if len(sizes) != 1:
        raise ValueError(f'Image and label sizes differ for {image_name}: {sizes}')

    width, height = next(iter(sizes))
    if width % patch_size or height % patch_size:
        raise ValueError(f'Image size is not divisible by {patch_size}: {image_name} ({width}x{height})')

    stem = Path(image_name).stem
    suffix = Path(image_name).suffix
    patch_names = []
    for top in range(0, height, patch_size):
        for left in range(0, width, patch_size):
            patch_name = f'{stem}_{top:04d}_{left:04d}{suffix}'
            patch_names.append(patch_name)
            box = (left, top, left + patch_size, top + patch_size)
            for name, image in images.items():
                image.crop(box).save(output_root / name / patch_name, compress_level=1)
    return patch_names


def main():
    parser = ArgumentParser()
    parser.add_argument('--source_root', default='./datasets/CD/LEVIR-CD-256')
    parser.add_argument('--output_root', default='./datasets/CD/LEVIR-CD-256-patches')
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    if output_root.exists():
        raise FileExistsError(f'Output directory already exists: {output_root}')

    expected_per_split = {'train': 7120, 'val': 1024, 'test': 2048}
    output_dirs = {name: output_root / name for name in ('A', 'B', 'label', 'list')}
    for directory in output_dirs.values():
        directory.mkdir(parents=True, exist_ok=False)

    for split, expected_count in expected_per_split.items():
        image_names = read_split(source_root / 'list' / f'{split}.txt')
        patch_names = []

        tasks = [(str(source_root), str(output_root), args.patch_size, image_name)
                 for image_name in image_names]
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for image_patch_names in executor.map(create_patches, tasks):
                patch_names.extend(image_patch_names)

        if len(patch_names) != expected_count:
            raise ValueError(f'{split} produced {len(patch_names)} patches; expected {expected_count}')
        (output_dirs['list'] / f'{split}.txt').write_text('\n'.join(patch_names) + '\n')
        print(f'{split}: {len(patch_names)} patches')


if __name__ == '__main__':
    main()
