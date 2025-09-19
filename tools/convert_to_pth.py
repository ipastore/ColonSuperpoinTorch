"""Convert a training checkpoint (.pth.tar) to a raw state dict (.pth).

Why: Many demos (e.g., MagicLeap SuperPoint) expect a plain PyTorch
state_dict file. Training often saves full checkpoints containing optimizer
state, iteration counters, etc. This CLI extracts only model weights.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch


def _infer_state_dict(
    ckpt: Dict, prefer_key: str | None = None
) -> Dict[str, torch.Tensor]:
    """Return the model state_dict from a checkpoint-like object.

    Tries (in order): explicit key, 'model_state_dict', 'state_dict',
    or interprets the whole object as a state_dict if values look like tensors.
    """

    if prefer_key and prefer_key in ckpt:
        return ckpt[prefer_key]

    for k in ("model_state_dict", "state_dict"):
        if k in ckpt:
            return ckpt[k]

    # Heuristic: plain state_dict if values are tensors
    if isinstance(ckpt, dict) and all(
        isinstance(v, torch.Tensor) for v in ckpt.values()
    ):
        return ckpt  # type: ignore[return-value]

    raise KeyError(
        "Could not infer model state_dict. Available keys: "
        f"{sorted(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}"
    )


def _strip_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove a leading 'module.' (from DataParallel) from parameter names."""

    needs_strip = any(k.startswith("module.") for k in sd.keys())
    if not needs_strip:
        return sd
    return {k.replace("module.", "", 1): v for k, v in sd.items()}


def convert(
    inp: Path, out: Path | None, key: str | None, strip_module: bool
) -> Path:
    """Load checkpoint at `inp`, extract weights, save to `out` path.

    Args:
        inp: Input checkpoint path (.pth/.pth.tar).
        out: Output file path (.pth). If None, derive from input.
        key: Optional key name to extract inside checkpoint.
        strip_module: If True, remove leading 'module.' prefixes.

    Returns:
        Path to the written .pth file.
    """

    ckpt = torch.load(str(inp), map_location="cpu")
    state = _infer_state_dict(ckpt, prefer_key=key)
    if strip_module:
        state = _strip_module_prefix(state)

    out_path = out or Path(str(inp).replace(".pth.tar", ".pth"))
    if out_path.suffix == "":
        out_path = out_path.with_suffix(".pth")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(out_path))
    print(f"Wrote raw state_dict: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a training checkpoint (.pth/.pth.tar) into a raw state_dict (.pth)."
        )
    )
    parser.add_argument("input", type=Path, help="Path to input checkpoint")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output .pth path (defaults to replacing .pth.tar with .pth)",
    )
    parser.add_argument(
        "--key",
        type=str,
        default=None,
        help="Explicit key inside checkpoint (e.g., model_state_dict)",
    )
    parser.add_argument(
        "--no-strip-module",
        action="store_true",
        help="Do not strip leading 'module.' prefixes",
    )

    args = parser.parse_args()
    convert(
        inp=args.input,
        out=args.output,
        key=args.key,
        strip_module=not args.no_strip_module,
    )


if __name__ == "__main__":
    main()

