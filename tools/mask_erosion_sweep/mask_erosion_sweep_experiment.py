#!/usr/bin/env python3
"""Run configurable sweeps over colon mask export parameters."""

import argparse
import json
import logging
import os
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from export import export_detector_homoAdapt_gpu  # noqa: E402


def _load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r") as handle:
        return yaml.safe_load(handle)


def _dataset_name_from_config(config: Dict[str, Any]) -> str:
    data_cfg = config.get("data", {})
    images_path = data_cfg.get("images_path")
    if images_path:
        candidate = Path(str(images_path).rstrip("/")).name
        if candidate:
            return candidate
    dataset_field = data_cfg.get("dataset")
    if dataset_field:
        return str(dataset_field).lower()
    return "unknown_dataset"


def _make_args(exper_name: str, output_img: bool) -> argparse.Namespace:
    """Create a lightweight argparse namespace for export calls."""
    return argparse.Namespace(
        command="export_detector_homoAdapt_gpu",
        exper_name=exper_name,
        outputImg=output_img,
    )


class MaskErosionSweep:
    """Run export sweeps across configurable mask parameters."""

    def __init__(
        self,
        config_path: Path,
        output_dir: Optional[Path],
        erode_camera_max: Optional[int],
        erode_specular_max: Optional[int],
        border_margin_max: Optional[int],
        white_threshold_max: Optional[float],
        white_threshold_min: float,
        white_threshold_step: float,
        output_img: bool,
    ) -> None:
        self.config_path = config_path
        self.base_config = _load_config(config_path)
        self.dataset_name = _dataset_name_from_config(self.base_config)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_output_dir = output_dir or Path(
            f"logs/export/{self.dataset_name}/mask_erosion_sweep_{timestamp}"
        )
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

        data_cfg = self.base_config.setdefault("data", {})
        self.camera_values = self._build_int_range(
            erode_camera_max,
            int(data_cfg.get("erode_camera_mask", 0)),
            "erode_camera_mask",
        )
        self.specular_values = self._build_int_range(
            erode_specular_max,
            int(data_cfg.get("erode_specular_mask", 0)),
            "erode_specular_mask",
        )
        border_base = int(
            data_cfg.get("homography_adaptation", {}).get("valid_border_margin", 0)
        )
        self.border_values = self._build_int_range(
            border_margin_max,
            border_base,
            "valid_border_margin",
        )
        self.white_threshold_values = self._build_float_range(
            white_threshold_max,
            white_threshold_min,
            white_threshold_step,
            float(data_cfg.get("specular_white_threshold", 0.75)),
        )

        self.output_img = output_img
        self.results: List[Dict[str, Any]] = []

    def run(self) -> None:
        total_runs = (
            len(self.camera_values)
            * len(self.specular_values)
            * len(self.border_values)
            * len(self.white_threshold_values)
        )
        logging.info(
            "Running sweep with %d camera × %d specular × %d border × %d white values => %d runs",
            len(self.camera_values),
            len(self.specular_values),
            len(self.border_values),
            len(self.white_threshold_values),
            total_runs,
        )
        run_index = 0
        for camera_margin in self.camera_values:
            for specular_margin in self.specular_values:
                for border_margin in self.border_values:
                    for white_threshold in self.white_threshold_values:
                        run_index += 1
                        white_tag = f"{white_threshold:.3f}".replace(".", "p")
                        run_name = (
                            f"cam{camera_margin}_spec{specular_margin}_"
                            f"border{border_margin}_white{white_tag}"
                        )
                        output_dir = self.base_output_dir / run_name
                        output_dir.mkdir(parents=True, exist_ok=True)

                        config = deepcopy(self.base_config)
                        data_cfg = config.setdefault("data", {})
                        data_cfg["erode_camera_mask"] = camera_margin
                        data_cfg["erode_specular_mask"] = specular_margin
                        homography_cfg = data_cfg.setdefault("homography_adaptation", {})
                        homography_cfg["valid_border_margin"] = border_margin
                        data_cfg["specular_white_threshold"] = white_threshold

                        exper_name = f"{self.dataset_name}_{run_name}"
                        args = _make_args(exper_name=exper_name, output_img=self.output_img)

                        logging.info(
                            "[%d/%d] Exporting with camera=%d, specular=%d, border=%d, white=%.3f",
                            run_index,
                            total_runs,
                            camera_margin,
                            specular_margin,
                            border_margin,
                            white_threshold,
                        )

                        status = "success"
                        error_message = None
                        try:
                            export_detector_homoAdapt_gpu(config, str(output_dir), args)
                        except Exception as exc:  # noqa: BLE001
                            logging.exception(
                                "Export failed for camera=%d specular=%d border=%d white=%.3f",
                                camera_margin,
                                specular_margin,
                                border_margin,
                                white_threshold,
                            )
                            status = "error"
                            error_message = str(exc)

                        self.results.append(
                            {
                                "camera_margin": camera_margin,
                                "specular_margin": specular_margin,
                                "border_margin": border_margin,
                                "white_threshold": white_threshold,
                                "status": status,
                                "output_dir": str(output_dir),
                                "exper_name": exper_name,
                                "error": error_message,
                            }
                        )
        self._write_results()

    @staticmethod
    def _build_int_range(
        max_value: Optional[int],
        base_value: int,
        label: str,
    ) -> List[int]:
        if max_value is None:
            return [base_value]
        if max_value < 0:
            raise ValueError(f"{label} maximum must be non-negative.")
        return list(range(0, max_value + 1))

    @staticmethod
    def _build_float_range(
        max_value: Optional[float],
        min_value: float,
        step: float,
        base_value: float,
    ) -> List[float]:
        if max_value is None:
            return [base_value]
        if step <= 0:
            raise ValueError("white_threshold step must be positive.")
        if max_value < min_value:
            raise ValueError("white_threshold max must be >= min.")
        values = np.arange(min_value, max_value + step / 2.0, step)
        rounded = [float(f"{v:.6f}") for v in values]
        return rounded if rounded else [base_value]

    def _write_results(self) -> None:
        summary_path = self.base_output_dir / "mask_erosion_sweep_results.json"
        csv_path = self.base_output_dir / "mask_erosion_sweep_results.csv"
        with summary_path.open("w") as handle:
            json.dump(self.results, handle, indent=2)
        with csv_path.open("w") as handle:
            handle.write(
                "camera_margin,specular_margin,border_margin,white_threshold,status,output_dir,exper_name,error\n"
            )
            for entry in self.results:
                error = entry["error"] or ""
                handle.write(
                    f"{entry['camera_margin']},{entry['specular_margin']},{entry['border_margin']},"
                    f"{entry['white_threshold']},{entry['status']},{entry['output_dir']},{entry['exper_name']},{error}\n"
                )
        logging.info("Saved sweep summary to %s", summary_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mask erosion sweep runner")
    parser.add_argument("config", type=Path, help="Path to base config file")
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        default=None,
        help="Optional output directory (default auto-generated)",
    )
    parser.add_argument(
        "--erode_camera",
        type=int,
        default=None,
        help="Sweep erode_camera_mask from 0 to this value (inclusive).",
    )
    parser.add_argument(
        "--erode_specular",
        type=int,
        default=None,
        help="Sweep erode_specular_mask from 0 to this value (inclusive).",
    )
    parser.add_argument(
        "--valid_border_margin",
        type=int,
        default=None,
        help="Sweep valid_border_margin from 0 to this value (inclusive).",
    )
    parser.add_argument(
        "--white_threshold",
        type=float,
        default=None,
        help="Sweep specular_white_threshold from white_threshold_min to this value (inclusive).",
    )
    parser.add_argument(
        "--white_threshold_min",
        type=float,
        default=0.0,
        help="Lower bound for specular_white_threshold sweep (default: 0.0).",
    )
    parser.add_argument(
        "--white_threshold_step",
        type=float,
        default=0.05,
        help="Step size for specular_white_threshold sweep (default: 0.05).",
    )
    parser.add_argument(
        "--outputImg",
        action="store_true",
        help="Enable writing debug images (mirrors export.py flag)",
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
        level=logging.DEBUG if args.debug else logging.INFO,
        format="[%(asctime)s %(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sweep = MaskErosionSweep(
        config_path=args.config,
        output_dir=args.output_dir,
        erode_camera_max=args.erode_camera,
        erode_specular_max=args.erode_specular,
        border_margin_max=args.valid_border_margin,
        white_threshold_max=args.white_threshold,
        white_threshold_min=args.white_threshold_min,
        white_threshold_step=args.white_threshold_step,
        output_img=args.outputImg,
    )
    sweep.run()


if __name__ == "__main__":
    main()
