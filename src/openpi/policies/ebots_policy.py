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
    dual_wrist_camera: bool = False

    # The expected cameras names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")

    # Optional crop windows per logical view name (e.g., "right_wrist_0_rgb")
    crop_windows: Mapping[str, CropSpec] | None = None

    def __call__(self, data: dict) -> dict:
        data = self.convert_images(data)

        in_images = data["images"]

        # Validate camera keys.
        unexpected = set(in_images) - set(self.EXPECTED_CAMERAS)
        if unexpected:
            raise ValueError(
                f"Expected images to contain only {self.EXPECTED_CAMERAS}, "
                f"got unexpected cameras: {tuple(unexpected)}"
            )

        if "cam_high" not in in_images:
            raise ValueError("Missing required base camera 'cam_high' in images.")

        # Base image (assumed to always exist).
        base_image = in_images["cam_high"]

        # Optionally crop this base view.
        if self.crop_windows is not None and "base_0_rgb" in self.crop_windows:
            base_image = self.crop_windows["base_0_rgb"].apply(base_image)

        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }

        # Determine extra wrist cameras.
        # Map logical names ("*_wrist_0_rgb") to physical camera keys or None (for masked-out views).
        extra_image_names: dict[str, str | None] = {}

        if not self.dual_wrist_camera:
            if self.ebots_action_dim in (14, 17):
                extra_image_names = {
                    "left_wrist_0_rgb": "cam_left_wrist",
                    "right_wrist_0_rgb": "cam_right_wrist",
                }
            elif self.ebots_action_dim == 7:
                if self.use_right_arm:
                    extra_image_names = {
                        "left_wrist_0_rgb": None,               # will become black + mask False
                        "right_wrist_0_rgb": "cam_right_wrist",
                    }
                else:
                    extra_image_names = {
                        "left_wrist_0_rgb": "cam_left_wrist",
                        "right_wrist_0_rgb": None,              # will become black + mask False
                    }
            else:
                raise ValueError(
                    f"Unsupported ebots_action_dim for single wrist: {self.ebots_action_dim}"
                )
        else:
            # Dual-wrist mode: both logical wrists map to the active arm’s camera.
            source = "cam_right_wrist" if self.use_right_arm else "cam_left_wrist"
            extra_image_names = {
                "left_wrist_0_rgb": source,
                "right_wrist_0_rgb": source,
            }

        # Add the extra images (or black placeholders).
        for dest, source in extra_image_names.items():
            if source is not None and source in in_images:
                img = in_images[source]

                # Optionally crop this logical view.
                if self.crop_windows is not None and dest in self.crop_windows:
                    img = self.crop_windows[dest].apply(img)

                images[dest] = img
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
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
