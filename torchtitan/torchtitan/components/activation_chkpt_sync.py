import sys

# mypy: allow-untyped-defs
import weakref
from collections import defaultdict

from typing import Any, Dict, List, Optional, Optional, Union, Union

import torch
from torch.autograd.graph import saved_tensors_hooks
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map
from torch.utils.checkpoint import (
    _is_compiling,
    _maybe_detach,
    _policy_from_bool,
    _VersionWrapper,
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
    SAC_IGNORED_OPS,
    SelectiveCheckpointContext,
)
from torchtitan.tools.logging import logger


def check_nan_inf(tensor, name, rank=0):
    """Check for NaN or Inf values in tensor"""
    if torch.isnan(tensor).any():
        print(f"[RANK {rank}] NaN detected in {name}, shape: {tensor.shape}")
        print(f"[RANK {rank}] NaN locations: {torch.isnan(tensor).sum().item()}")
        return True
    if torch.isinf(tensor).any():
        print(f"[RANK {rank}] Inf detected in {name}, shape: {tensor.shape}")
        print(f"[RANK {rank}] Inf locations: {torch.isinf(tensor).sum().item()}")
        return True
    return False


class _AsyncOffloadWrapper:
    """
    Wrapper that handles asynchronous CPU offloading and GPU prefetching
    for selective activation checkpointing.
    """

    def __init__(
        self,
        val,
        device,
        offload_stream=None,
        prefetch_stream=None,
        is_last_layer=False,
        layer_id=None,
    ):
        self.original_device = device
        self.version = (
            [val[i]._version for i in range(len(val))]
            if isinstance(val, tuple)
            else val._version
        )
        self.offload_stream = offload_stream
        self.prefetch_stream = prefetch_stream
        self.is_last_layer = is_last_layer
        self.layer_id = layer_id
        # State management
        self._cpu_tensor: Optional[torch.Tensor] | tuple[
            torch.Tensor, torch.Tensor
        ] = None
        self._gpu_tensor: Optional[torch.Tensor] | tuple[
            torch.Tensor, torch.Tensor
        ] = None
        self._is_offloaded = False
        self._is_prefetching = False
        self._prefetch_event: Optional[torch.cuda.Event] = None
        self.offload_event = None

        # Immediately start offloading to CPU
        if not self.is_last_layer:
            if (isinstance(val, tuple) and any(v.is_cuda for v in val)) or (
                isinstance(val, torch.Tensor) and val.is_cuda
            ):
                self._start_cpu_offload(val)
            else:
                # For non-CUDA tensors, just store directly
                self._cpu_tensor = val
                self._is_offloaded = True
        else:
            self._gpu_tensor = val

    def _start_cpu_offload(self, gpu_tensor):
        """Asynchronously offload tensor to CPU"""
        if isinstance(gpu_tensor, tuple):
            self._cpu_tensor = [
                torch.empty(
                    gpu_tensor[i].shape,
                    dtype=gpu_tensor[i].dtype,
                    device="cpu",
                    pin_memory=True,
                )
                for i in range(len(gpu_tensor))
            ]
            for i in range(len(gpu_tensor)):
                self._cpu_tensor[i].copy_(gpu_tensor[i], non_blocking=True)
            self._gpu_tensor = None  # Free GPU memory
            self._is_offloaded = True
        elif isinstance(gpu_tensor, torch.Tensor):
            self._cpu_tensor = torch.empty(
                gpu_tensor.shape, dtype=gpu_tensor.dtype, device="cpu", pin_memory=True
            )
            self._cpu_tensor.copy_(gpu_tensor, non_blocking=True)
            self._gpu_tensor = None  # Free GPU memory
            self._is_offloaded = True

    def start_gpu_prefetch(self):
        """Start asynchronous prefetching to GPU before it's needed"""
        if self.is_last_layer:
            return

        if self.offload_event is not None:
            self.offload_event.wait()
            self.offload_event = None
            self._is_offloaded = True

        if isinstance(self._cpu_tensor, list):
            self._gpu_tensor = [
                torch.empty(
                    self._cpu_tensor[i].shape,
                    dtype=self._cpu_tensor[i].dtype,
                    device=self.original_device,
                )
                for i in range(len(self._cpu_tensor))
            ]
            for i in range(len(self._cpu_tensor)):
                self._gpu_tensor[i].copy_(self._cpu_tensor[i], non_blocking=True)
        else:
            self._gpu_tensor = self._cpu_tensor.to(
                self.original_device, non_blocking=True
            )

    def get_val(self, allow_cache_entry_mutation=True):
        """Get the tensor value, handling async transfers"""
        if self.is_last_layer:
            assert (
                self._gpu_tensor is not None
            ), "GPU tensor should be available for last layer"
            return self._gpu_tensor

        # If we have a GPU tensor ready, use it
        if self._gpu_tensor is not None:
            if self._prefetch_event is not None:
                self._prefetch_event = None
                if isinstance(self._gpu_tensor, list) or isinstance(
                    self._gpu_tensor, tuple
                ):
                    assert allow_cache_entry_mutation or all(
                        self._gpu_tensor[i]._version == self.version[i]
                        for i in range(len(self._gpu_tensor))
                    ), "GPU tensor version mismatch"
                else:
                    assert (
                        allow_cache_entry_mutation
                        or self._gpu_tensor._version == self.version
                    ), "GPU tensor version mismatch"
            return self._gpu_tensor

        raise RuntimeError("Tensor not available - this should not happen")


class _AsyncOffloadCachingTorchDispatchMode(TorchDispatchMode):
    """
    Enhanced caching mode with asynchronous CPU offloading
    """

    def __init__(
        self,
        policy_fn,
        storage,
        offload_threshold_mb=50,
        offload_stream=None,
        prefetch_stream=None,
        is_last_layer=False,
        layer_id=None,
    ):
        self.policy_fn = policy_fn
        self.storage = storage
        self.offload_threshold_mb = offload_threshold_mb
        self.offload_stream = offload_stream
        self.prefetch_stream = prefetch_stream
        self.is_last_layer = is_last_layer
        self.layer_id = layer_id

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func in SAC_IGNORED_OPS:
            return func(*args, **kwargs)

        kwargs = {} if kwargs is None else kwargs
        policy = self.policy_fn(
            SelectiveCheckpointContext(is_recompute=False), func, *args, **kwargs
        )

        if isinstance(policy, bool):
            policy = _policy_from_bool(policy)

        is_compiling = _is_compiling(func, args, kwargs)
        out = func(*args, **kwargs)

        # Determine if we should cache this operation
        if (
            policy in (CheckpointPolicy.MUST_SAVE, CheckpointPolicy.PREFER_SAVE)
            or is_compiling
        ):

            # Check if tensors are large enough to warrant offloading
            def should_offload(x):
                if isinstance(x, torch.Tensor) and x.is_cuda:
                    size_mb = x.numel() * x.element_size() / (1024 * 1024)
                    return size_mb > self.offload_threshold_mb
                elif isinstance(x, tuple):
                    return any(should_offload(y) for y in x)
                return False

            # Use our async offload wrapper
            def wrap_with_offload(x):
                any_ret_has_alias_info = self._check_alias_info(func)
                detached = _maybe_detach(x, any_ret_has_alias_info)
                device = (
                    detached.device
                    if isinstance(detached, torch.Tensor)
                    else detached[0].device
                )

                if should_offload(detached):
                    return _AsyncOffloadWrapper(
                        detached,
                        device,
                        self.offload_stream,
                        self.prefetch_stream,
                        self.is_last_layer,
                        self.layer_id,
                    )
                else:
                    # For small tensors, use regular version wrapper
                    return _VersionWrapper(detached)

            self.storage[func].append(wrap_with_offload(out))

        return out

    def _check_alias_info(self, func):
        """Check if function has alias info (same as original)"""
        if isinstance(func, torch._ops.HigherOrderOperator):
            return False
        else:
            return any(ret.alias_info is not None for ret in func._schema.returns)


class _AsyncOffloadCachedTorchDispatchMode(TorchDispatchMode):
    """
    Enhanced cached mode with intelligent prefetching
    """

    def __init__(
        self,
        policy_fn,
        storage,
        allow_cache_entry_mutation,
        prefetch_ahead=1,
        layer_id=None,
    ):
        self.policy_fn = policy_fn
        self.storage = storage
        self.allow_cache_entry_mutation = allow_cache_entry_mutation
        self.prefetch_ahead = prefetch_ahead
        self.layer_id = layer_id

        # Track operation sequence for prefetching
        self.operation_sequence = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func in SAC_IGNORED_OPS:
            return func(*args, **kwargs)

        kwargs = {} if kwargs is None else kwargs
        policy = self.policy_fn(
            SelectiveCheckpointContext(is_recompute=True), func, *args, **kwargs
        )

        if isinstance(policy, bool):
            policy = _policy_from_bool(policy)

        is_compiling = _is_compiling(func, args, kwargs)

        if (
            policy in (CheckpointPolicy.MUST_SAVE, CheckpointPolicy.PREFER_SAVE)
            or is_compiling
        ):
            storage = self.storage.get(func)
            if storage is None:
                raise RuntimeError(
                    f"{func} encountered during backward, but not found in storage"
                )
            if len(storage) == 0:
                raise RuntimeError("Trying to backward an extra time...")

            # Start prefetching future operations
            self._prefetch_future_ops(func)

            # Get the cached value (may involve GPU transfer)
            cached_result = storage.pop(0)
            out = tree_map(
                lambda x: x.get_val(self.allow_cache_entry_mutation), cached_result
            )
        else:
            out = func(*args, **kwargs)

        return out

    def _prefetch_future_ops(self, current_func):
        """Prefetch tensors for upcoming operations"""
        # Look ahead in the storage to start GPU prefetching for all operations
        future_ops = list(
            self.storage.keys()
        )  

        # for future_func in future_ops:
        future_storage = self.storage[current_func]
        for cached_item in future_storage[:1]:  # Prefetch first item
            cached_item.start_gpu_prefetch()


def create_sync_offload_checkpoint_contexts(
    policy_fn_or_list,
    allow_cache_entry_mutation=False,
    offload_threshold_mb=1,
    prefetch_ahead=1,
    offload_stream=None,
    prefetch_stream=None,
    is_last_layer=False,
    layer_id=None,
):
    """
    Create selective checkpoint contexts with asynchronous CPU offloading.

    Args:
        policy_fn_or_list: Same as create_selective_checkpoint_contexts
        allow_cache_entry_mutation: Same as create_selective_checkpoint_contexts
        offload_threshold_mb: Minimum tensor size (MB) to trigger CPU offloading
        prefetch_ahead: Number of operations to prefetch ahead during recompute

    Returns:
        Tuple of (forward_context, recompute_context) with async offloading
    """

    # Handle policy function setup (same as original)
    if isinstance(policy_fn_or_list, list):
        ops_to_save = policy_fn_or_list

        def policy_fn(ctx, op, *args, **kwargs):
            if op in ops_to_save:
                return CheckpointPolicy.MUST_SAVE
            else:
                return CheckpointPolicy.PREFER_RECOMPUTE

    elif callable(policy_fn_or_list):
        policy_fn = policy_fn_or_list
    else:
        raise TypeError("policy_fn_or_list must be either a function or a list of ops.")

    storage: Dict[Any, List[Any]] = defaultdict(list)

    return (
        _AsyncOffloadCachingTorchDispatchMode(
            policy_fn,
            storage,
            offload_threshold_mb,
            offload_stream,
            prefetch_stream,
            is_last_layer,
            layer_id,
        ),
        _AsyncOffloadCachedTorchDispatchMode(
            policy_fn, storage, allow_cache_entry_mutation, prefetch_ahead, layer_id
        ),
    )


class async_save_on_cpu(saved_tensors_hooks):
    def __init__(
        self,
        pin_memory: bool = True,
        device_type: str = "cuda",
        offload_stream: torch.cuda.Stream = None,
        prefetch_stream: torch.cuda.Stream = None,
    ) -> None:
        device_module = getattr(torch, device_type, torch.cuda)
        self.offload_stream = offload_stream
        self.prefetch_stream = prefetch_stream

        def pack_to_cpu(tensor: torch.Tensor) -> tuple[torch.device, torch.Tensor]:
            packed = torch.empty(
                tensor.size(),
                dtype=tensor.dtype,
                layout=tensor.layout,
                pin_memory=(device_module.is_available() and not tensor.is_sparse),
            )
            self.offload_stream.wait_stream(torch.cuda.current_stream())
            # self.offload_stream.synchronize()
            with torch.cuda.stream(self.offload_stream):
                packed.copy_(tensor, non_blocking=True)
            return (tensor.device, packed)

        def unpack_from_cpu(packed: tuple[torch.device, torch.Tensor]) -> torch.Tensor:
            device, tensor = packed
            self.prefetch_stream.wait_stream(torch.cuda.current_stream())
            self.prefetch_stream.wait_stream(self.offload_stream)
            with torch.cuda.stream(self.prefetch_stream):
                tensor = tensor.to(device, non_blocking=True)
            return tensor

        super().__init__(pack_to_cpu, unpack_from_cpu)
