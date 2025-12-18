from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
            # NOTE: If you want RTC for PyTorch-backed models, implement analogous PyTorch routine.
            self._rtc_sample_actions = getattr(model, "realtime_sample_actions", None)
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            # JIT-wrap realtime sampler if present on the model
            if hasattr(model, "realtime_sample_actions"):
                self._rtc_sample_actions = nnx_utils.module_jit(
                    model.realtime_sample_actions,
                    static_argnames=(
                        "num_steps",
                        "inference_delay",
                        "prefix_attention_horizon",
                        "prefix_attention_schedule",
                        "max_guidance_weight",
                        "guidance_r",
                    ),
                )
            else:
                self._rtc_sample_actions = None
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @override    
    def rtc_infer(
        self,
        obs: dict,
        *,
        # RTC-specific kwargs forwarded to the model.realtime_sample_actions
        num_steps: int | None = None,
        noise: np.ndarray | None = None,
        prev_chunk: np.ndarray | None = None,
        inference_delay: int = 0,
        prefix_attention_horizon: int | None = None,
        prefix_attention_schedule: str = "exp",
        max_guidance_weight: float = 5.0,
        guidance_r: float = 1.0,
    ) -> dict:  # type: ignore[misc]
        """Run the RTC-guided sampler (calls model.realtime_sample_actions).

        Returns the same shaped dict as infer().
        """
        if self._rtc_sample_actions is None:
            raise RuntimeError("RTC sampler not available on this Policy's model.")

        # prepare inputs same as infer()
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)

        if not self._is_pytorch_model:
            # JAX: batch and jax arrays
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # PyTorch path: we currently don't provide a PyTorch RTC wrapper here
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # assemble kwargs to pass to the model rtc sampler
        rtc_kwargs = {}
        if num_steps is not None:
            rtc_kwargs["num_steps"] = num_steps
        if noise is not None:
            noise_arr = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)
            if noise_arr.ndim == 2:
                noise_arr = noise_arr[None, ...]
            rtc_kwargs["noise"] = noise_arr

        if prev_chunk is not None:
            if self._is_pytorch_model:
                pc = torch.from_numpy(prev_chunk) if not isinstance(prev_chunk, torch.Tensor) else prev_chunk
                # ensure batch dim
                if pc.ndim == 2:
                    pc = pc[None, ...]
                pc = pc.to(self._pytorch_device)
                rtc_kwargs["prev_chunk"] = pc
            else:
                pc = jnp.asarray(prev_chunk)
                if pc.ndim == 2:
                    pc = pc[None, ...]
                rtc_kwargs["prev_chunk"] = pc

        rtc_kwargs["inference_delay"] = int(inference_delay)
        rtc_kwargs["prefix_attention_horizon"] = prefix_attention_horizon
        rtc_kwargs["prefix_attention_schedule"] = prefix_attention_schedule
        rtc_kwargs["max_guidance_weight"] = float(max_guidance_weight)
        rtc_kwargs["guidance_r"] = float(guidance_r)

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()

        # call the RTC sampler
        if self._is_pytorch_model:
            if self._rtc_sample_actions is None:
                raise RuntimeError("RTC sampler not implemented for PyTorch models.")
            actions = self._rtc_sample_actions(sample_rng_or_pytorch_device, observation, **rtc_kwargs)
        else:
            actions = self._rtc_sample_actions(sample_rng_or_pytorch_device, observation, **rtc_kwargs)

        model_time = time.monotonic() - start_time

        # convert outputs back to numpy and unbatch
        outputs = {
            "state": inputs["state"],
            "actions": actions,
        }
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {"infer_ms": model_time * 1000}
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
    
    @override
    def rtc_infer(self, obs: dict, **kwargs) -> dict:  # type: ignore[misc]
        results = self._policy.rtc_infer(obs, **kwargs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}_rtc"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results

