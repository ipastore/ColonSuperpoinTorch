#!/usr/bin/env python3
"""Visualize warped pairs and masks from a training configuration."""

from __future__ import annotations

import argparse
import logging
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import torch
import yaml
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from utils.loader import dataLoader  # noqa: E402


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r") as stream:
        return yaml.safe_load(stream)


def _prepare_config(config: Dict[str, Any], samples: int) -> Dict[str, Any]:
    cfg = deepcopy(config)
    model_cfg = cfg.setdefault("model", {})
    model_cfg["batch_size"] = 1
    model_cfg["eval_batch_size"] = 1

    training_cfg = cfg.setdefault("training", {})
    training_cfg["workers_train"] = 0
    training_cfg["workers_val"] = 0

    data_cfg = cfg.setdefault("data", {})
    data_cfg["truncate"] = samples

    return cfg


def _tensor_to_numpy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value


def _extract_batch_item(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value[0]
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def _prepare_image(array: np.ndarray) -> np.ndarray:
    img = np.asarray(array)
    if img.ndim == 3:
        img = img.squeeze(axis=0)
    return img.astype(np.float32, copy=False)


def _prepare_mask(mask: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.ones(target_shape, dtype=np.float32)
    mask_array = np.asarray(mask, dtype=np.float32)
    if mask_array.ndim == 3:
        mask_array = mask_array.squeeze(axis=0)
    if mask_array.shape != target_shape:
        mask_array = cv2.resize(
            mask_array,
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return mask_array


def _extract_keypoints(label_map: np.ndarray) -> np.ndarray:
    if label_map is None:
        return np.empty((0, 2), dtype=np.int32)
    labels = np.asarray(label_map)
    if labels.ndim == 3:
        labels = labels.squeeze(axis=0)
    coords = np.argwhere(labels > 0)
    if coords.size == 0:
        return np.empty((0, 2), dtype=np.int32)
    return coords[:, [1, 0]].astype(np.int32)


def _render_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    keypoints: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    img = _prepare_image(image)
    if img.max() <= 1.0:
        img = (img * 255.0).clip(0.0, 255.0)
    img_uint8 = img.astype(np.uint8)
    color = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)

    mask_prepared = _prepare_mask(mask, color.shape[:2])
    invalid = mask_prepared < 0.5

    overlay = color.copy()
    red = np.zeros_like(color, dtype=np.uint8)
    red[..., 2] = 255
    overlay[invalid] = (
        overlay[invalid].astype(np.float32) * (1.0 - alpha)
        + red[invalid].astype(np.float32) * alpha
    )
    overlay = overlay.astype(np.uint8)

    for point in keypoints:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < overlay.shape[1] and 0 <= y < overlay.shape[0]:
            if mask_prepared[y, x] >= 0.5:
                cv2.circle(overlay, (x, y), 2, (0, 255, 0), thickness=-1)

    return overlay


def _compose_side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    height = max(left.shape[0], right.shape[0])
    width = left.shape[1] + right.shape[1]
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[: left.shape[0], : left.shape[1]] = left
    canvas[: right.shape[0], left.shape[1] : left.shape[1] + right.shape[1]] = right
    return canvas


def visualize_pairs(
    config_path: Path,
    output_dir: Path,
    split: str,
    num_samples: int,
) -> None:
    base_config = _load_config(config_path)
    config = _prepare_config(base_config, num_samples)

    dataset_name = config.get("data", {}).get("dataset", "Colon")
    logging.info("Dataset: %s | split: %s", dataset_name, split)

    data = dataLoader(config, dataset=dataset_name)
    loader_key = f"{split}_loader"
    if loader_key not in data:
        raise ValueError(f"Split '{split}' not available in config.")
    loader = data[loader_key]

    output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for index, batch in enumerate(loader):
        if saved >= num_samples:
            break
        sample = {k: _extract_batch_item(v) for k, v in batch.items()}
        sample_np = {k: _tensor_to_numpy(v) for k, v in sample.items()}

        source_image = sample_np.get("image_2D", sample_np.get("image"))
        source_mask = sample_np.get("image_2D_valid_mask", sample_np.get("valid_mask"))
        source_labels = sample_np.get("labels_2D")

        warped_image = sample_np.get("warped_img")
        if warped_image is None:
            warped_image = sample_np.get("warped_image", sample_np.get("image"))
        warped_mask = sample_np.get("warped_valid_mask", sample_np.get("valid_mask"))
        warped_labels = sample_np.get("warped_labels")

        if source_image is None or warped_image is None:
            logging.warning("Sample %d missing warped pair fields; skipping", index)
            continue

        source_kpts = _extract_keypoints(source_labels)
        warped_kpts = _extract_keypoints(warped_labels)

        source_overlay = _render_overlay(source_image, source_mask, source_kpts)
        warped_overlay = _render_overlay(warped_image, warped_mask, warped_kpts)

        combined = _compose_side_by_side(source_overlay, warped_overlay)

        sample_name = sample_np.get("name", f"sample_{index}")
        if isinstance(sample_name, (list, tuple)):
            sample_name = sample_name[0]
        sample_name = str(sample_name).replace(" ", "_")

        filename = output_dir / f"{index:03d}_{sample_name}.png"
        cv2.imwrite(str(filename), combined)
        logging.info("Saved %s", filename)
        saved += 1

    if saved < num_samples:
        logging.info(
            "Requested %d samples but only %d were available for split '%s'.",
            num_samples,
            saved,
            split,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Warped pair visualization tool")
    parser.add_argument("config", type=Path, help="Path to training config YAML")
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        default=None,
        help="Optional output directory",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val"],
        default="train",
        help="Dataset split to visualize (default: train)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to visualize (default: 10)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG if args.debug else logging.INFO,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_dir = (
        Path("logs/visualizations")
        / "warped_pairs"
        / args.config.stem
        / timestamp
    )
    output_dir = args.output_dir or default_dir

    logging.info("Saving visualizations to %s", output_dir)
    visualize_pairs(args.config, output_dir, args.split, args.num_samples)


if __name__ == "__main__":
    main()
