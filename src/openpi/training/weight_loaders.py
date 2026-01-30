import dataclasses
import logging
import math
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np
from scipy import ndimage

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


def _interpolate_pos_embedding(
    loaded: np.ndarray, expected_shape: tuple[int, ...]
) -> np.ndarray:
    """Resize learned position embeddings from checkpoint resolution to current resolution.

    Used when fine-tuning at higher image resolution than the checkpoint (e.g. checkpoint
    at 224x224 with 256 patches, current model at 1120x1120 with 6400 patches).

    This is load-time interpolation for the JAX training path. It is distinct from the
    PyTorch runtime flag `interpolate_pos_encoding` (see modeling_siglip.py), which
    interpolates on every forward when input resolution differs from the stored embedding
    size. The JAX SigLIP module has a fixed pos_embedding shape at init, so we resize
    the checkpoint once here instead of at forward time.
    """
    if loaded.shape == expected_shape:
        return loaded
    # Expected (1, num_patches, dim); interpret num_patches as H*W and interpolate 2D.
    _, seq_old, dim = loaded.shape
    _, seq_new, _ = expected_shape
    size_old = int(math.isqrt(seq_old))
    size_new = int(math.isqrt(seq_new))
    if size_old * size_old != seq_old or size_new * size_new != seq_new:
        raise ValueError(
            f"pos_embedding seq lengths must be perfect squares: got {seq_old} and {seq_new}"
        )
    # (1, S, D) -> (1, H, W, D)
    grid = loaded.reshape(1, size_old, size_old, dim)
    zoom = (1, size_new / size_old, size_new / size_old, 1)
    resized = ndimage.zoom(grid, zoom, order=1)
    return resized.reshape(1, seq_new, dim).astype(loaded.dtype)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            ref = flat_ref[k]
            ref_shape = ref.shape if hasattr(ref, "shape") else getattr(ref, "shape", None)
            if (
                ref_shape is not None
                and "pos_embedding" in k
                and hasattr(v, "shape")
                and v.shape != ref_shape
                and v.ndim == 3
                and v.shape[0] == 1
            ):
                old_shape = v.shape
                v = _interpolate_pos_embedding(np.asarray(v), ref_shape)
                logger.info(
                    "Interpolated %s from %s to %s for higher resolution",
                    k,
                    old_shape,
                    ref_shape,
                )
            result[k] = v.astype(ref.dtype) if hasattr(ref, "dtype") and v.dtype != ref.dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")
