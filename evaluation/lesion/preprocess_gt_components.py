#!/usr/bin/env python3
"""Create derived tracks after removing GT components smaller than 5 px.

The source dataset is never modified. Images are linked/copied unchanged and
cleaned masks are written into a new output root with the same track layout.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


TRACKS = {512: "Size_512", 224: "Size_224_filtered"}
STRUCTURE_8 = np.ones((3, 3), dtype=np.uint8)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", choices=(224, 512), default=[512, 224])
    parser.add_argument(
        "--splits", nargs="+", choices=("test", "val", "train"),
        default=["train", "val", "test"],
        help="SAM checkpoint entry points inspect all three splits even in eval-only mode.",
    )
    parser.add_argument("--min-component-area", type=int, default=5)
    parser.add_argument("--image-mode", choices=("symlink", "hardlink", "copy"), default="symlink")
    return parser.parse_args()


def files(directory: Path):
    result = {path.name: path for path in directory.iterdir()
              if path.is_file() and path.suffix.lower() == ".png"}
    if not result:
        raise RuntimeError(f"No PNG files found: {directory}")
    return result


def link_image(source: Path, destination: Path, mode: str):
    if mode == "symlink":
        destination.symlink_to(source.resolve())
    elif mode == "hardlink":
        os.link(source, destination)
    else:
        shutil.copy2(source, destination)


def digest_tree(paths):
    value = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        value.update(path.name.encode() + b"\0")
        value.update(path.read_bytes())
    return value.hexdigest()


def main():
    args = parse_args()
    source_root = args.data_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"Output already exists; refusing overwrite: {output_root}")
    if args.min_component_area < 1:
        raise ValueError("--min-component-area must be >= 1")
    output_root.mkdir(parents=True)
    incomplete_marker = output_root / ".incomplete"
    incomplete_marker.write_text(
        "GT component preprocessing did not complete yet.\n", encoding="utf-8"
    )
    rows = []
    for size in args.sizes:
        for split in args.splits:
            source = source_root / TRACKS[size] / split
            images, masks = files(source / "images"), files(source / "masks")
            if images.keys() != masks.keys():
                raise RuntimeError(f"Image/mask PNG identifiers differ: {source}")
            target_images = output_root / TRACKS[size] / split / "images"
            target_masks = output_root / TRACKS[size] / split / "masks"
            target_images.mkdir(parents=True)
            target_masks.mkdir(parents=True)
            for name in sorted(images):
                with Image.open(images[name]) as image_handle:
                    if image_handle.size != (size, size):
                        raise ValueError(
                            f"Expected {size}x{size} image, got {image_handle.size}: {images[name]}"
                        )
                link_image(images[name], target_images / name, args.image_mode)
                with Image.open(masks[name]) as handle:
                    if handle.size != (size, size):
                        raise ValueError(
                            f"Expected {size}x{size} mask, got {handle.size}: {masks[name]}"
                        )
                    raw = np.asarray(handle.convert("L"))
                values = set(int(value) for value in np.unique(raw))
                if not values.issubset({0, 1}):
                    raise ValueError(
                        f"Expected binary 0/1 GT, found {sorted(values)}: {masks[name]}"
                    )
                foreground = raw > 0
                labels, count = ndimage.label(foreground, STRUCTURE_8)
                areas = np.bincount(labels.ravel(), minlength=count + 1)
                remove_ids = np.flatnonzero((areas < args.min_component_area) &
                                            (np.arange(len(areas)) > 0))
                cleaned = foreground & ~np.isin(labels, remove_ids)
                _, retained = ndimage.label(cleaned, STRUCTURE_8)
                # Benchmark contract is 0=background, 1=consolidation.
                Image.fromarray(cleaned.astype(np.uint8), mode="L").save(
                    target_masks / name
                )
                rows.append({
                    "size": size, "split": split, "image_id": Path(name).stem,
                    "original_foreground_pixels": int(foreground.sum()),
                    "cleaned_foreground_pixels": int(cleaned.sum()),
                    "removed_pixels": int(foreground.sum() - cleaned.sum()),
                    "original_components": int(count),
                    "retained_components": int(retained),
                    "removed_components": int(len(remove_ids)),
                })
    with (output_root / "gt_component_cleanup.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    manifest = {
        "source_root": str(source_root), "output_root": str(output_root),
        "sizes": args.sizes, "splits": args.splits,
        "connectivity": 8, "remove_rule": f"component_area < {args.min_component_area}",
        "image_mode": args.image_mode,
        "images": len(rows), "images_changed": sum(row["removed_components"] > 0 for row in rows),
        "components_removed": sum(row["removed_components"] for row in rows),
        "pixels_removed": sum(row["removed_pixels"] for row in rows),
        "cleaned_mask_tree_sha256": digest_tree(
            path for size in args.sizes for split in args.splits
            for path in (output_root / TRACKS[size] / split / "masks").glob("*.png")
        ),
    }
    (output_root / "gt_component_cleanup.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    incomplete_marker.unlink()
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
