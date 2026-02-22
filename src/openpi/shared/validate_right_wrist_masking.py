#!/usr/bin/env python3
"""Validate that pi0_ebots_cart uses masked (black) right wrist for episode_index < 548
and real right wrist for episode_index >= 548.

Usage (from repo root):
  python openpi/shared/visualize_right_wrist_masking.py

Outputs:
  - right_wrist_ep_early.png  (should be black)
  - right_wrist_ep_late.png    (should show real camera)
  - Prints episode indices, mean pixel value, and mask for each.
"""

import pathlib
import sys

import numpy as np

# Add repo root so openpi is importable
_repo_root = pathlib.Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as _transforms


def _compose(transforms):
    def fn(data):
        for t in transforms:
            data = t(data)
        return data
    return fn


def main():
    config = _config.get_config("pi0_ebots_cart")
    data_config = config.data.create(config.assets_dirs, config.model)

    # Build dataset (same as training: with EpisodeIndexWrapper so samples have episode_index)
    dataset = _data_loader.create_torch_dataset(
        data_config,
        action_horizon=config.model.action_horizon,
        model_config=config.model,
    )

    repack = _compose(data_config.repack_transforms.inputs)
    ebots_inputs = data_config.data_transforms.inputs[0]  # EbotsInputs
    transform_to_ebots = _compose([*data_config.repack_transforms.inputs, ebots_inputs])

    # Find one frame from episode < 548 and one from episode >= 548
    idx_early = None
    idx_late = None
    n = len(dataset)
    # Scan a subset if dataset is huge
    step = max(1, n // 10000) if n > 20000 else 1
    for i in range(0, n, step):
        raw = dataset[i]
        packed = repack(raw)
        ep = int(np.asarray(packed.get("episode_index", -1)).item())
        if ep < 548 and idx_early is None:
            idx_early = i
        if ep >= 548 and idx_late is None:
            idx_late = i
        if idx_early is not None and idx_late is not None:
            break
    if idx_early is None:
        idx_early = 0
    if idx_late is None:
        # Try last quarter of dataset
        idx_late = min(n - 1, (3 * n) // 4)

    out_dir = pathlib.Path(".").resolve()
    try:
        from PIL import Image
    except ImportError:
        Image = None

    for label, idx in [("early (ep < 548)", idx_early), ("late (ep >= 548)", idx_late)]:
        raw = dataset[idx]
        packed = repack(raw)
        ep = int(np.asarray(packed.get("episode_index", -1)).item())
        out = transform_to_ebots(raw)
        img = np.asarray(out["image"]["right_wrist_0_rgb"])
        mask = out["image_mask"]["right_wrist_0_rgb"]
        mean_val = float(np.mean(img))
        is_zeros = np.all(img == 0)

        print(f"\n--- Sample index {idx} ({label}), episode_index={ep} ---")
        print(f"  right_wrist_0_rgb: shape={img.shape}, dtype={img.dtype}")
        print(f"  mean pixel value: {mean_val:.2f}")
        print(f"  all zeros: {is_zeros}")
        print(f"  image_mask['right_wrist_0_rgb']: {bool(mask)}")

        if Image is not None:
            # Ensure HWC uint8 for PIL
            if img.ndim == 3 and img.shape[0] in (1, 3):
                img = np.transpose(img, (1, 2, 0))
            if img.shape[-1] == 1:
                img = np.squeeze(img, axis=-1)
            img_u8 = np.clip(img, 0, 255).astype(np.uint8)
            fname = "right_wrist_ep_early.png" if "early" in label else "right_wrist_ep_late.png"
            path = out_dir / fname
            Image.fromarray(img_u8).save(path)
            print(f"  Saved: {path}")

    print("\nDone. Check the PNGs: early should be black, late should show the real right wrist.")


if __name__ == "__main__":
    main()
