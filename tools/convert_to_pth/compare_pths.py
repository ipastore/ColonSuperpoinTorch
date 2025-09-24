"""Compare weights between a .pth.tar checkpoint and a raw .pth state_dict.

Minimal CLI: loads both files on CPU, extracts model weights, optionally strips
DataParallel's 'module.' prefix, and reports key/shape/value differences.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch


def _infer_state_dict(
    obj: object, prefer_key: str | None = None
) -> Dict[str, torch.Tensor]:
    """Extract a state_dict from a checkpoint-like object.

    Tries (in order): explicit key, 'model_state_dict', 'state_dict', or treats
    the whole mapping as a state_dict if values are tensors.
    """
    if isinstance(obj, dict):
        if prefer_key and prefer_key in obj:
            return obj[prefer_key]
        for k in ("model_state_dict", "state_dict"):
            if k in obj:
                return obj[k]
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj  # type: ignore[return-value]
    raise KeyError(
        "Could not infer state_dict. Available: "
        f"{sorted(obj.keys()) if isinstance(obj, dict) else type(obj)}"
    )


def _strip_module(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(k.startswith("module.") for k in sd.keys()):
        return {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd


def _load_state(path: Path, key: str | None, strip: bool) -> Dict[str, torch.Tensor]:
    obj = torch.load(str(path), map_location="cpu")
    sd = _infer_state_dict(obj, prefer_key=key) if not isinstance(obj, dict) else _infer_state_dict(obj, prefer_key=key)
    return _strip_module(sd) if strip else sd


def main() -> None:
    p = argparse.ArgumentParser(description="Compare .pth.tar and .pth weights")
    p.add_argument("tar", type=Path, help="Path to .pth.tar (or .pth) checkpoint")
    p.add_argument("pth", type=Path, help="Path to raw .pth state_dict")
    p.add_argument("--tar-key", type=str, default=None, help="Key inside tar")
    p.add_argument("--pth-key", type=str, default=None, help="Key inside pth if not raw")
    p.add_argument("--no-strip-module", action="store_true", help="Keep 'module.'")
    p.add_argument("--rtol", type=float, default=0.0, help="Relative tol")
    p.add_argument("--atol", type=float, default=1e-7, help="Absolute tol")
    args = p.parse_args()

    strip = not args.no_strip_module
    sd_tar = _load_state(args.tar, key=args.tar_key, strip=strip)
    sd_pth = _load_state(args.pth, key=args.pth_key, strip=strip)

    keys_tar = set(sd_tar.keys())
    keys_pth = set(sd_pth.keys())
    only_tar = sorted(keys_tar - keys_pth)
    only_pth = sorted(keys_pth - keys_tar)
    inter = sorted(keys_tar & keys_pth)

    mismatches = []
    worst = (None, 0.0)
    for k in inter:
        a, b = sd_tar[k], sd_pth[k]
        if a.shape != b.shape:
            mismatches.append((k, f"shape {tuple(a.shape)} != {tuple(b.shape)}"))
            continue
        diff = torch.max(torch.abs(a - b)).item()
        # Allow small tolerance if requested
        if not torch.allclose(a, b, rtol=args.rtol, atol=args.atol):
            mismatches.append((k, f"max_abs_diff={diff:.3e}"))
        if diff > worst[1]:
            worst = (k, diff)

    print(f"extra in tar: {len(only_tar)}")
    if only_tar:
        print("  e.g.:", only_tar[:5])
    print(f"extra in pth: {len(only_pth)}")
    if only_pth:
        print("  e.g.:", only_pth[:5])
    print(f"compared params: {len(inter)}")
    print(f"mismatches: {len(mismatches)}")
    if mismatches:
        for k, msg in mismatches[:10]:
            print(f"  {k}: {msg}")
        if worst[0] is not None:
            print(f"worst: {worst[0]} max_abs_diff={worst[1]:.3e}")

    # Exit non-zero on any difference
    import sys

    sys.exit(0 if not only_tar and not only_pth and not mismatches else 1)


if __name__ == "__main__":
    main()

