import contextlib
import os
from typing import Union
from warnings import warn

PIN_MEMORY = True if os.environ.get("PIN_MEMORY", "True") == "True" else False

import sys

# mypy: allow-untyped-defs
import weakref
from collections import defaultdict

from typing import Any, Dict, List, Optional, Optional, Union, Union

import psutil

import torch
import torchao
from torch import nn
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
from torchao.dtypes.nf4tensor import NF4Tensor
from torchtitan.tools.logging import logger

from torchtune.modules import TiedLinear
from torchtune.utils import get_logger

CURRENT_DEVICE = torch.cuda.current_device()


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
        event_registry=None,
        offload_stream=None,
        prefetch_stream=None,
        two_streams=None,
        is_last_layer=False,
        layer_id=None,
    ):
        self.original_device = device
        if isinstance(val, list):
            val = tuple(val)
        self.version = (
            [val[i]._version for i in range(len(val))]
            if isinstance(val, tuple)
            else val._version
        )
        self.offload_stream = offload_stream
        self.prefetch_stream = prefetch_stream
        self.two_streams = two_streams
        self.is_last_layer = is_last_layer
        self.layer_id = layer_id if isinstance(layer_id, int) else int(layer_id)
        # State management
        self._cpu_tensor: Optional[torch.Tensor] | tuple[
            torch.Tensor, torch.Tensor
        ] = None
        self._gpu_tensor: Optional[torch.Tensor] | tuple[
            torch.Tensor, torch.Tensor
        ] = None
        self.event_registry = event_registry
        self._is_offloaded = False
        self.offload_event = None
        self.prefetch_event = None

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
        # first clean up the event registry
        self._gpu_tensor = gpu_tensor
        
        # Asynchronously offload tensor to CPU
        if isinstance(gpu_tensor, tuple):
            self._cpu_tensor = [
                torch.empty(
                    gpu_tensor[i].shape,
                    dtype=gpu_tensor[i].dtype,
                    device="cpu",
                    pin_memory=PIN_MEMORY,
                )
                for i in range(len(gpu_tensor))
            ]
            self.offload_stream.wait_stream(torch.cuda.default_stream())
            if self.two_streams is not None:
                for s in self.two_streams:
                    self.offload_stream.wait_stream(s)
            with torch.cuda.stream(self.offload_stream):
                for i in range(len(gpu_tensor)):
                    self._cpu_tensor[i].copy_(gpu_tensor[i], non_blocking=True)
                self.offload_event = self.offload_stream.record_event()

        elif isinstance(gpu_tensor, torch.Tensor):
            self._cpu_tensor = torch.empty(
                gpu_tensor.shape,
                dtype=gpu_tensor.dtype,
                device="cpu",
                pin_memory=PIN_MEMORY,
            )
            self.offload_stream.wait_stream(torch.cuda.default_stream())
            if self.two_streams is not None:
                for s in self.two_streams:
                    self.offload_stream.wait_stream(s)
            with torch.cuda.stream(self.offload_stream):
                self._cpu_tensor.copy_(gpu_tensor, non_blocking=True)
                self.offload_event = self.offload_stream.record_event()

    def start_gpu_prefetch(self):
        """Start asynchronous prefetching to GPU before it's needed"""
        if self.is_last_layer:
            assert (
                self._gpu_tensor is not None
            ), "GPU tensor should be available for last layer"
            return

        self.prefetch_stream.wait_stream(torch.cuda.default_stream())
        if self.two_streams is not None:
            for s in self.two_streams:
                self.prefetch_stream.wait_stream(s)
        with torch.cuda.stream(self.prefetch_stream):
            self.offload_event.wait()
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
            self.prefetch_event = self.prefetch_stream.record_event()

    def get_val(self, allow_cache_entry_mutation=True):
        """Get the tensor value, handling async transfers"""
        if self.prefetch_event is not None:
            self.prefetch_event.wait()
            self.offload_event.wait()
            
        assert (
            self._gpu_tensor is not None
        ), "GPU tensor should be available already for layer " + str(self.layer_id)

        # If we have a GPU tensor ready, use it
        if isinstance(self._gpu_tensor, list) or isinstance(self._gpu_tensor, tuple):
            assert allow_cache_entry_mutation or all(
                self._gpu_tensor[i]._version == self.version[i]
                for i in range(len(self._gpu_tensor))
            ), "GPU tensor version mismatch"
        else:
            assert (
                allow_cache_entry_mutation or self._gpu_tensor._version == self.version
            ), "GPU tensor version mismatch"
        return self._gpu_tensor



class _AsyncOffloadCachingTorchDispatchMode(TorchDispatchMode):
    """
    Enhanced caching mode with asynchronous CPU offloading
    """

    def __init__(
        self,
        policy_fn,
        storage,
        offload_threshold_mb=50,
        event_registry=None,
        offload_stream=None,
        prefetch_stream=None,
        two_streams=None,
        is_last_layer=False,
        layer_id=None,
    ):
        self.policy_fn = policy_fn
        self.storage = storage
        self.offload_threshold_mb = offload_threshold_mb
        self.offload_stream = offload_stream
        self.prefetch_stream = prefetch_stream
        self.two_streams = two_streams
        self.is_last_layer = is_last_layer
        self.layer_id = layer_id if isinstance(layer_id, int) else int(layer_id)
        self.event_registry = event_registry

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

        for key in self.event_registry.keys():
            for id in range(self.layer_id):
                if id in self.event_registry[key]:
                    for wrapper in self.event_registry[key][id]:
                        if wrapper.offload_event.query():
                            wrapper._gpu_tensor = None
                            wrapper._is_offloaded = True

        # Determine if we should cache this operation
        if (
            policy in (CheckpointPolicy.MUST_SAVE, CheckpointPolicy.PREFER_SAVE)
            or is_compiling
        ):

            
            any_ret_has_alias_info = self._check_alias_info(func)
            out = _maybe_detach(out, any_ret_has_alias_info)
            device = out.device if isinstance(out, torch.Tensor) else out[0].device

            if func in self.event_registry:
                offload_wrapper = _AsyncOffloadWrapper(
                    out,
                    device,
                    self.event_registry[func],
                    self.offload_stream,
                    self.prefetch_stream,
                    self.two_streams,
                    self.is_last_layer,
                    self.layer_id,
                )
            else:
                self.event_registry[func] = {}
                offload_wrapper = _AsyncOffloadWrapper(
                    out,
                    device,
                    self.event_registry[func],
                    self.offload_stream,
                    self.prefetch_stream,
                    self.two_streams,
                    self.is_last_layer,
                    self.layer_id,
                )

            if self.layer_id in self.event_registry[func]:
                self.event_registry[func][self.layer_id].append(offload_wrapper)
            else:
                self.event_registry[func][self.layer_id] = [offload_wrapper]

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
        event_registry=None,
        layer_id=None,
    ):
        self.policy_fn = policy_fn
        self.storage = storage
        self.allow_cache_entry_mutation = allow_cache_entry_mutation
        self.prefetch_ahead = prefetch_ahead
        self.layer_id = layer_id if isinstance(layer_id, int) else int(layer_id)
        self.event_registry = event_registry

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

            # Start prefetching future operations
            if self.layer_id + 1 in self.event_registry[func]:
                self.event_registry[func][self.layer_id + 1] = []
            self._prefetch_future_ops(func)

            stored_wrapper = self.event_registry[func][self.layer_id].pop(0)
            out = stored_wrapper.get_val(self.allow_cache_entry_mutation)
        else:
            out = func(*args, **kwargs)

        return out

    def _prefetch_future_ops(self, current_func):
        """Prefetch tensors for upcoming operations"""
        # Look ahead in the storage to start GPU prefetching for all operations

        if self.layer_id - 1 in self.event_registry[current_func]:
            for cached_item in self.event_registry[current_func][self.layer_id - 1]:
                cached_item.start_gpu_prefetch()


def create_async_offload_checkpoint_contexts(
    policy_fn_or_list,
    allow_cache_entry_mutation=False,
    offload_threshold_mb=1,
    event_registry=None,
    prefetch_ahead=1,
    offload_stream=None,
    prefetch_stream=None,
    two_streams=None,
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
            event_registry,
            offload_stream,
            prefetch_stream,
            two_streams,
            is_last_layer,
            layer_id,
        ),
        _AsyncOffloadCachedTorchDispatchMode(
            policy_fn,
            storage,
            allow_cache_entry_mutation,
            prefetch_ahead,
            event_registry,
            layer_id,
        ),
    )

class async_save_on_cpu(saved_tensors_hooks):
    """Context manager under which activation tensors created in the forward pass will be offloaded.

    Enable the memory efficiency technique of activation offloading, where activations bigger than
    min_offload_size bytes will be offloaded to CPU in the forward and brought back in the backward.
    This is in contrast to maintaining the activation on GPU VRAM throughout the program.

    This manager contains the option of using one additional CUDA stream to handle the communication
    between CUDA and CPU, which is intended to overlap with the default computation stream to improve
    runtime. We designed synchronization with a few heuristics for optimizing the tradeoff between
    runtime vs memory usage.

    Args:
        use_pin_memory (bool): Whether or not the offloaded Tensor will be placed in pinned
            memory on the CPU. Pinned memory allows the Tensor to be moved back onto GPU more quickly
            but is a limited resource. Default: True.

        use_streams (bool): Whether or not to use streams for performance optimization where
            the communications get overlapped with the computation. Requires a torch build
            after torch-2.5.0.]. Default: True.

        max_fwd_stash_size (int): The maximum size of the forward stash, or the maximum number of
            consecutive activations to keep alive during the forward pass. This number must be at
            least 1. Keeping alive more activations will potentially allow more overlap between the
            communication and compute streams at the cost of increasing memory usage. Keeping alive
            fewer activations will conserve memory, but may cause poor overlap between the streams,
            increasing runtime. Default: 5.

        min_offload_size (int): The minimum number of bytes a Tensor must be in order to qualify
            for offloading. If the tensor is too small, we do not want to waste bandwidth and resources
            moving it to CPU and back. Default: 1024 bytes.

    Raises:
        ValueError: if max_fwd_stash_size is not at least 1.

    Example:
        >>> with OffloadActivations():
        >>>     logits = model(inputs)
        >>> loss = ...
        >>> loss.backward()
    """

    def __init__(
        self,
        use_pin_memory: bool = True,
        use_streams: bool = True,
        max_fwd_stash_size: int = 1,
        min_offload_size: int = 5,
        offload_stream: torch.cuda.Stream = None,
        prefetch_stream: torch.cuda.Stream = None,
        two_streams: list[torch.cuda.Stream] = None,
    ) -> None:

        self.use_streams: bool = use_streams

        self.min_tensor_size_bytes = (
            min_offload_size  # we don't want to bother with small tensors
        )
        self.tracker = (
            {}
        ) 
        self.tensor_id: int = 0
        self.is_first_forward_call = True
        self.is_first_backward_call = True
        self.is_first_forward_pass = True

        # managing cpu memory
        self.use_pin_memory: bool = use_pin_memory
        self.virtual_memory_safe_pct = (
            60  # we should not exceed this percentage of memory
        )

        self.s0 = torch.cuda.default_stream()  # comp stream
        self.two_streams = two_streams

        # for streaming
        if self.use_streams:
            self.s1 = offload_stream  # torch.cuda.Stream()  # comms stream
            self.fwd_stash = {}  # tensor_id => (activation, ev1)
            if max_fwd_stash_size < 1:
                raise ValueError(
                    f"max_fwd_stash_size should be at least 1 but is {max_fwd_stash_size}"
                )
            self.max_fwd_stash_size = max_fwd_stash_size
            self.bwd_tensor_stash = {}  # tensor_id => activation
            self.bwd_ev_stash = {}  # tensor_id => ev0
            self.curr_graph_id = None
            self.curr_autograd_node = None

        # -------- platform util functions -------- #
        def verify_sufficient_virtual_memory():
            curr_pct = get_cpu_ram_pct()
            if curr_pct > self.virtual_memory_safe_pct:
                warn(
                    f"***** WARNING: {curr_pct=}% > {self.virtual_memory_safe_pct=}% of virtual memory used"
                )

        def get_cpu_ram_pct() -> float:
            # get the percentage of memory used by the system
            return psutil.virtual_memory().percent

        def get_tensor_id() -> int:
            # create a unique id for each tensor we are managing
            self.tensor_id += 1
            return self.tensor_id

        def get_num_bytes_tensor(x: torch.Tensor) -> int:
            # get the number of bytes in a tensor, for memory management purposes
            return (
                x.element_size() * x.nelement()
            ) 

        # -------- core pack / unpack work -------- #
        def pack_tensor(activation: torch.Tensor) -> int:
            # activations are passed in during forward pass - from here we take over and return a unique id
            if self.is_first_forward_call:
                assert (
                    len(self.tracker) == 0
                ), "backward pass should have cleared tracker of all tensors"

                # set training phase trackers
                self.is_first_forward_call = False
                self.is_first_backward_call = True

            # query for basic tensor info
            num_bytes = get_num_bytes_tensor(activation)
            tensor_id = get_tensor_id()

            # only offload hefty bois if they're activations (our heuristic for that is to
            # check if they're not params or buffers)!
            if num_bytes >= self.min_tensor_size_bytes and (
                not isinstance(activation, torch.nn.Parameter)
                and not isinstance(activation, torch.nn.Buffer)
            ):
                if self.use_streams:
                    # First, sync back and dereference previously offloaded tensors
                    # as the offloading should be done sufficiently long ago.
                    for id in [k for k in self.fwd_stash.keys()]:
                        if id <= tensor_id - self.max_fwd_stash_size:
                            _, ev = self.fwd_stash[id]
                            self.s0.wait_event(ev)
                            del self.fwd_stash[id]
                        else:
                            break

                    # Sync in, offload, and add an event to sync back later
                    self.s1.wait_stream(self.s0)
                    if self.two_streams is not None:
                        self.s1.wait_stream(self.two_streams[0])
                        self.s1.wait_stream(self.two_streams[1])

                stream = self.s1 if self.use_streams else self.s0
                with torch.cuda.stream(stream):
                    try:
                        cpu_tensor = torch.empty_like(
                            activation, pin_memory=self.use_pin_memory, device="cpu"
                        )
                    except NotImplementedError as e:
                        if (
                            isinstance(activation, NF4Tensor)
                            and torchao.__version__ < "0.6.0.dev20240917"
                        ):
                            raise RuntimeError(
                                "Offloading NF4Tensors requires torchao-0.6.0.dev20240917 or later"
                            ) from e
                        raise e
                    cpu_tensor.copy_(activation, non_blocking=True)
                    self.tracker[tensor_id] = (
                        cpu_tensor,
                        True,
                    )

                if self.use_streams:
                    event = self.s1.record_event()

                    # Stash to keep activation alive til s1 is done
                    self.fwd_stash[tensor_id] = (activation, event)
            else:
                self.tracker[tensor_id] = (
                    activation,
                    False,
                )

            return tensor_id

        def unpack_tensor_single_stream(unpack_tensor_id: int) -> torch.Tensor:
            # backward pass - we are called with the tensor_id, which
            # we will use to retrieve the saved/offloaded tensor
            if self.is_first_backward_call:
                if self.is_first_forward_pass:
                    self.is_first_forward_pass = False
                    if self.use_pin_memory:
                        verify_sufficient_virtual_memory()

                self.is_first_backward_call = False
                self.is_first_forward_call = True

            assert (
                unpack_tensor_id in self.tracker
            ), f"untracked tensor with id {unpack_tensor_id}"

            maybe_gpu_tensor, modified = self.tracker[unpack_tensor_id]
            if modified:
                gpu_tensor = maybe_gpu_tensor.to(
                    torch.device("cuda", CURRENT_DEVICE), non_blocking=True
                )
                maybe_gpu_tensor = gpu_tensor

            # clear tensor from tracking
            del self.tracker[unpack_tensor_id]
            return maybe_gpu_tensor

        def unpack_tensor_with_streams(unpack_tensor_id: int) -> torch.Tensor:
            # backward pass - we are called with the tensor_id, which
            # we will use to retrieve the saved/offloaded tensor
            if self.is_first_backward_call:
                self.curr_graph_id = torch._C._current_graph_task_id()

                def wait_and_del_remaining_references() -> None:
                    for id in [k for k in self.bwd_tensor_stash.keys()]:
                        event = self.bwd_ev_stash[id]
                        self.s1.wait_event(event)
                        del self.bwd_tensor_stash[id]

                # Register a callback to the end of autograd to clean everything up
                torch.autograd.variable.Variable._execution_engine.queue_callback(
                    wait_and_del_remaining_references
                )

                if self.is_first_forward_pass:
                    self.is_first_forward_pass = False
                    if self.use_pin_memory:
                        verify_sufficient_virtual_memory()

                self.is_first_backward_call = False
                self.is_first_forward_call = True

            assert (
                unpack_tensor_id in self.tracker
            ), f"untracked tensor with id {unpack_tensor_id}"

            maybe_gpu_tensor, modified = self.tracker[unpack_tensor_id]
            if modified:
                # Get data on the current autograd node
                graph_id = torch._C._current_graph_task_id()
                node = torch._C._current_autograd_node()
                prev_node_ids = []

                # If we're on a new node, mark prev node's tensors to be freed later
                if graph_id == self.curr_graph_id and self.curr_autograd_node != node:
                    self.curr_autograd_node = node
                    prev_node_ids = [id for id in self.bwd_tensor_stash.keys()]

                brought_back_from_cpu = True
                if unpack_tensor_id in self.fwd_stash.keys():
                    self.s1.wait_event(self.fwd_stash[unpack_tensor_id][1])
                    
                # Kick off the process to bring tensors back
                self.s1.wait_stream(self.s0)
                with torch.cuda.stream(self.s1):
                    gpu_tensor = maybe_gpu_tensor.to(
                        torch.device("cuda", CURRENT_DEVICE), non_blocking=True
                    )
                    maybe_gpu_tensor = gpu_tensor

                # Tell comp stream to wait for the info to be loaded before executing
                self.s0.wait_stream(self.s1)
                if self.two_streams is not None:
                    self.s0.wait_stream(self.two_streams[0])
                    self.s0.wait_stream(self.two_streams[1])

                # Stash the tensor to keep memory alive until compute stream is complete
                self.bwd_tensor_stash[unpack_tensor_id] = maybe_gpu_tensor

                # Note: [Track views of the unpacked]
                # Why do we get the use count of the unpacked tensor here? We want an
                # initial count to compare to later, during the post-hook of the
                # backward node, when we need to decide whether we're allowed to free
                # the tensor yet. In what obscure cases must we delay freeing the
                # tensor (and thus call record_stream)?
                # 1. Any of the outputs of the backward node is a view of the unpacked
                #    tensor.
                # 2. In the case that this unpacked tensor will be used in a
                #    checkpointed region, if one of the recomputed saved tensors ends
                #    up as a view of the unpacked tensor.
                # 3. The user abuses the system somehow and manually relies on the
                #    unpacked tensor to exist after the backward node has executed.
                storage_refcount = torch._C._storage_Use_Count(
                    maybe_gpu_tensor.untyped_storage()._cdata
                )

                def hook(outputs, inputs):
                    # create events for the current node inputs/outputs if they were streamed in
                    if brought_back_from_cpu:
                        # See Note: [Track views of the unpacked]
                        # IF any of the outputs is a view of the tensor, OR if a view of
                        # the tensor has been saved as a part of checkpoint's recompute
                        # process, OR the user has abusedly incurred a reference on the
                        # unpacked tensor, THEN the tensor might be used later and we
                        # cannot presume to delete it after only the current node is
                        # done! So we use our frenemy, record_stream, to ensure the
                        # Tensor stays unmessed with until it's done getting used in the
                        # compute stream (s0 here). Note that the con here is we introduce
                        # non-deterministic (thus higher) memory usage, but this case
                        # should not happen often.
                        unpacked_tensor = self.bwd_tensor_stash[unpack_tensor_id]
                        if (
                            torch._C._storage_Use_Count(
                                unpacked_tensor.untyped_storage()._cdata
                            )
                            > storage_refcount
                        ):
                            unpacked_tensor.record_stream(self.s0)
                            del self.bwd_tensor_stash[unpack_tensor_id]
                        else:
                            event = self.s0.record_event()
                            self.bwd_ev_stash[unpack_tensor_id] = event

                    # if there are still things in the fwd_stash, get rid of them as we're in bwd now
                    for id in [k for k in self.fwd_stash.keys()]:
                        _, ev = self.fwd_stash[id]
                        self.s0.wait_event(ev)
                        if self.two_streams is not None:
                            self.two_streams[0].wait_event(ev)
                            self.two_streams[1].wait_event(ev)
                        del self.fwd_stash[id]
                    
                    # wait on prev node's events and del those
                    for id in prev_node_ids:
                        event = self.bwd_ev_stash[id]
                        self.s1.wait_event(event)
                        del self.bwd_tensor_stash[id]

                    return outputs

                node.register_hook(hook)

            # clear tensor from tracking
            del self.tracker[unpack_tensor_id]
            return maybe_gpu_tensor

        unpack_tensor = (
            unpack_tensor_with_streams
            if self.use_streams
            else unpack_tensor_single_stream
        )
        super().__init__(pack_tensor, unpack_tensor)
