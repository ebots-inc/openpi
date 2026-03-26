import dataclasses
from typing import ClassVar, Mapping

import einops
import numpy as np

from openpi import transforms


def make_ebots_example() -> dict:
    """Creates a random input example for the Ebots policy."""
    return {
        "state": np.ones((17,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_low": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }

@dataclasses.dataclass(frozen=True)
class CropSpec:
    """Crop window as fractions (0..1) on HWC images."""
    y_start: float
    y_end: float
    x_start: float
    x_end: float

    def apply(self, img: np.ndarray) -> np.ndarray:
        # img is HWC
        h, w = img.shape[:2]

        # Clamp to [0, 1]
        ys = float(np.clip(self.y_start, 0.0, 1.0))
        ye = float(np.clip(self.y_end,   0.0, 1.0))
        xs = float(np.clip(self.x_start, 0.0, 1.0))
        xe = float(np.clip(self.x_end,   0.0, 1.0))

        # Convert to pixel indices (floor start, ceil end)
        y0 = int(np.floor(ys * h))
        y1 = int(np.ceil( ye * h))
        x0 = int(np.floor(xs * w))
        x1 = int(np.ceil( xe * w))

        # Sanity & fallback
        y0 = max(0, min(y0, h))
        y1 = max(0, min(y1, h))
        x0 = max(0, min(x0, w))
        x1 = max(0, min(x1, w))
        if y1 <= y0 or x1 <= x0:
            return img  # bad window → return original

        return img[y0:y1, x0:x1, :]


@dataclasses.dataclass(frozen=True)
class EbotsInputs(transforms.DataTransformFn):
    """Inputs for the Ebots policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [up to 17]
    - actions: [action_horizon, up to 17]
    """

    ebots_action_dim: int = 17
    use_right_arm: bool = False

    # Map logical Ebots views ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    # to physical camera keys (e.g. "cam_high", "cam_left_wrist"); `None` masks out.
    camera_sources: Mapping[str, str | None] | None = None

    # Fallback routing when `training/config.py` doesn't provide `camera_sources`.
    DEFAULT_CAMERA_SOURCES: ClassVar[Mapping[str, str]] = {
        "base_0_rgb": "cam_high",
        "left_wrist_0_rgb": "cam_left_wrist",
        "right_wrist_0_rgb": "cam_right_wrist",
    }

    # The expected cameras names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")

    # Optional crop windows per logical view name (e.g., "right_wrist_0_rgb")
    crop_windows: Mapping[str, CropSpec] | None = None

    # Optional per-view rotation k for `np.rot90`.
    camera_rot90_ks: Mapping[str, int] | None = None

    def __call__(self, data: dict) -> dict:
        data = self.convert_images(data)

        in_images = data["images"]

        camera_sources = self.camera_sources or self.DEFAULT_CAMERA_SOURCES

        # Validate camera keys.
        unexpected = set(in_images) - set(self.EXPECTED_CAMERAS)
        if unexpected:
            raise ValueError(
                f"Expected images to contain only {self.EXPECTED_CAMERAS}, "
                f"got unexpected cameras: {tuple(unexpected)}"
            )

        if "base_0_rgb" not in camera_sources:
            raise ValueError("camera_sources must contain a 'base_0_rgb' entry.")
        base_source = camera_sources["base_0_rgb"]
        if base_source is None:
            raise ValueError("camera_sources['base_0_rgb'] must not be None.")
        if base_source not in in_images:
            if base_source == "cam_high":
                raise ValueError("Missing required base camera 'cam_high' in images.")
            raise ValueError(f"Missing required base camera '{base_source}' in images.")

        # Base image (assumed to always exist).
        base_image = in_images[base_source]

        # Optionally crop this base view.
        if self.crop_windows is not None and "base_0_rgb" in self.crop_windows:
            base_image = self.crop_windows["base_0_rgb"].apply(base_image)

        # Rotate base view if configured.
        if self.camera_rot90_ks is not None and "base_0_rgb" in self.camera_rot90_ks:
            base_image = np.rot90(base_image, self.camera_rot90_ks["base_0_rgb"])

        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }

        # Add the wrist images (or black placeholders).
        for dest in ("left_wrist_0_rgb", "right_wrist_0_rgb"):
            if dest not in camera_sources:
                raise ValueError(f"camera_sources must contain a '{dest}' entry.")
            source = camera_sources[dest]
            if source is not None and source in in_images:
                img = in_images[source]

                # Crop this logical view if configured.
                if self.crop_windows is not None and dest in self.crop_windows:
                    img = self.crop_windows[dest].apply(img)

                # Rotate this logical view if configured.
                if self.camera_rot90_ks is not None and dest in self.camera_rot90_ks:
                    img = np.rot90(img, self.camera_rot90_ks[dest])

                images[dest] = img
                image_masks[dest] = np.True_
            else:
                placeholder = np.zeros_like(base_image)
                if self.camera_rot90_ks is not None and dest in self.camera_rot90_ks:
                    placeholder = np.rot90(placeholder, self.camera_rot90_ks[dest])
                images[dest] = placeholder
                image_masks[dest] = np.False_

        # Decide which slice of the 17-D state/actions to use.
        if self.ebots_action_dim in (14, 17):
            start_idx, end_idx = 0, self.ebots_action_dim
        elif self.ebots_action_dim == 7 and not self.use_right_arm:
            # Left arm: first 7 dims.
            start_idx, end_idx = 0, 7
        elif self.ebots_action_dim == 7 and self.use_right_arm:
            # Right arm: middle 7 dims.
            start_idx, end_idx = 7, 14
        else:
            raise ValueError(
                f"Unsupported (ebots_action_dim={self.ebots_action_dim}, "
                f"use_right_arm={self.use_right_arm}) combination."
            )

        # Clip state to the selected slice.
        state = np.asarray(data["state"])
        state = state[..., start_idx:end_idx]

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }

        # Actions are only available during training. Keep horizon, slice feature dim.
        if "actions" in data:
            actions = np.asarray(data["actions"])
            actions = actions[..., start_idx:end_idx]
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        if False:  # debug the inputs
            import matplotlib.pyplot as plt
            for name, img in images.items():
                plt.figure()
                plt.imshow(np.asarray(img))
                plt.title(name)
                plt.axis("off")
            
            print("\n[DEBUG] State:\n", state)
            if "actions" in inputs:
                print("\n[DEBUG] Actions:\n", inputs["actions"])
            
            if "prompt" in inputs:
                print("\n[DEBUG] Prompt:\n", inputs["prompt"])
            
            plt.show(block=True)

        return inputs
    
    def convert_images(self, data: dict) -> dict:
        def convert_image(img):
            img = np.asarray(img)
            # Convert to uint8 if using float images.
            if np.issubdtype(img.dtype, np.floating):
                img = (255 * img).astype(np.uint8)
            # Only rearrange if input is CHW; if already HWC, do nothing.
            if img.ndim == 3 and img.shape[0] in (1, 3):
                img = einops.rearrange(img, "c h w -> h w c")
            return img

        images = data["images"]
        data["images"] = {name: convert_image(img) for name, img in images.items()}
        return data


@dataclasses.dataclass(frozen=True)
class EbotsOutputs(transforms.DataTransformFn):
    """Outputs for the Ebots policy."""
    ebots_action_dim: int = 17

    def __call__(self, data: dict) -> dict:
        # Only return the first ebots_action_dim dims.
        actions = np.asarray(data["actions"])
        actions = actions[..., : self.ebots_action_dim]
        return {"actions": actions}
