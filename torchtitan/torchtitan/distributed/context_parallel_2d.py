# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import itertools
from typing import Any, Optional, Set

import torch
import torch.distributed._functional_collectives as ft_c
import torch.distributed.tensor as dist_tensor
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor._op_schema import OpInfo
from torch.distributed.tensor.experimental._attention import (
    _cp_options,
    _RotateMethod,
)

from torchtitan.tools.logging import logger


class _SequentialSharder2D:
    """
    A 2D-aware sequential sharder that chunks buffers along the ring dimension.
    """

    @classmethod
    def shard(
        cls, buffer: torch.Tensor, mesh: DeviceMesh, seq_dim: int
    ) -> torch.Tensor:
        # For 2D mesh, we shard along both ring and ulysses dimensions
        # Mesh layout: dim 0 = cp_ring (outer, cross-node), dim 1 = cp_ulysses (inner, intra-node)
        if mesh.ndim == 2:
            ring_size = mesh.size(0)  # cp_ring dimension
            ulysses_size = mesh.size(1)  # cp_ulysses dimension
            total_size = ring_size * ulysses_size  # Total size across both dimensions

            ring_rank = torch.distributed.get_rank(mesh.get_group(0))
            ulysses_rank = torch.distributed.get_rank(mesh.get_group(1))
            rank = (
                ring_rank * ulysses_size + ulysses_rank
            )  # rank in the overall ring (ulysses varies fastest as inner dim)

            logger.debug(
                f"2D mesh sharding: overall ring_size={total_size}, ring_rank={rank}, buffer_shape={buffer.shape}"
            )
            assert buffer.size()[seq_dim] % total_size == 0
            return buffer.chunk(total_size, dim=seq_dim)[rank]
        else:
            # 1D mesh - use standard sharding
            logger.debug(
                f"1D mesh sharding: mesh_size={mesh.size()}, buffer_shape={buffer.shape}"
            )
            assert buffer.size()[seq_dim] % mesh.size() == 0
            return buffer.chunk(mesh.size(), dim=seq_dim)[mesh.get_local_rank()]

    @classmethod
    def unshard(
        cls, buffer: torch.Tensor, mesh: DeviceMesh, seq_dim: int
    ) -> torch.Tensor:
        buffer = buffer.contiguous()
        if mesh.ndim == 2:
            # For 2D mesh, gather across both ring and ulysses dimensions
            ring_size = mesh.size(0)  # cp_ring dimension
            ulysses_size = mesh.size(1)  # cp_ulysses dimension
            total_size = ring_size * ulysses_size  # Total size across both dimensions
            all_buffers = [torch.empty_like(buffer) for _ in range(total_size)]
            ft_c.all_gather_inplace(
                all_buffers, buffer, mesh
            )  # Use full mesh instead of just ring_mesh
        else:
            # 1D mesh
            all_buffers = [torch.empty_like(buffer) for _ in range(mesh.size())]
            ft_c.all_gather_inplace(all_buffers, buffer, mesh)
        return torch.cat(all_buffers, dim=seq_dim)


class _RoundRobinLoadBalancer2D:
    """
    A 2D-aware round-robin load balancer that chunks buffers for load balancing.
    """

    ROUND_ROBIN_CYCLE = 2

    @classmethod
    def shard(
        cls, buffer: torch.Tensor, mesh: DeviceMesh, seq_dim: int
    ) -> torch.Tensor:
        assert cls.ROUND_ROBIN_CYCLE == 2

        if mesh.ndim == 2:
            ring_size = mesh.size(0)  # cp_ring dimension
            ulysses_size = mesh.size(1)  # cp_ulysses dimension
            total_size = ring_size * ulysses_size  # Total size across both dimensions

            ring_rank = torch.distributed.get_rank(mesh.get_group(0))
            ulysses_rank = torch.distributed.get_rank(mesh.get_group(1))
            rank = ulysses_size * ring_rank + ulysses_rank  # rank in the overall ring

            assert (
                buffer.size()[seq_dim] % (total_size * 2) == 0
            ), f"buffer.size()[seq_dim]={buffer.size()[seq_dim]}, total_size={total_size}, seq_dim={seq_dim}"
            if ulysses_size == 1 or ring_size > 1:
                chunks = buffer.chunk(total_size * 2, dim=seq_dim)
                try:
                    return torch.cat(
                        (chunks[rank], chunks[total_size * 2 - rank - 1]),
                        dim=seq_dim,
                    )
                except Exception as e:
                    logger.error(f"Error in _RoundRobinLoadBalancer2D.shard: {e}")
                    assert (
                        False
                    ), f"Ring size: {ring_size}, ulysses size: {ulysses_size}, total size: {total_size}, rank: {rank} Chunks size: {len(chunks)}"
                    raise e
            else:
                return buffer.chunk(ulysses_size, dim=seq_dim)[ulysses_rank]

        else:
            # 1D mesh
            cp_world_size = mesh.size()
            cp_rank = mesh.get_local_rank()
            assert buffer.size()[seq_dim] % (cp_world_size * 2) == 0
            chunks = buffer.chunk(cp_world_size * 2, dim=seq_dim)
            return torch.cat(
                (chunks[cp_rank], chunks[cp_world_size * 2 - cp_rank - 1]),
                dim=seq_dim,
            )

    @classmethod
    def unshard(
        cls, buffer: torch.Tensor, mesh: DeviceMesh, seq_dim: int
    ) -> torch.Tensor:
        assert cls.ROUND_ROBIN_CYCLE == 2
        buffer = buffer.contiguous()

        if mesh.ndim == 2:
            ring_size = mesh.size(0)  # cp_ring dimension
            ulysses_size = mesh.size(1)  # cp_ulysses dimension
            total_size = ring_size * ulysses_size
            all_buffers = [torch.empty_like(buffer) for _ in range(total_size)]
            ft_c.all_gather_inplace(all_buffers, buffer, mesh)
        else:
            # 1D mesh
            cp_world_size = mesh.size()
            all_buffers = [torch.empty_like(buffer) for _ in range(cp_world_size)]
            ft_c.all_gather_inplace(all_buffers, buffer, mesh)

        # Reorder buffers
        sliced_buffers = [sb for b in all_buffers for sb in b.chunk(2, dim=seq_dim)]
        ordered_buffers = list(sliced_buffers)
        size = len(all_buffers)
        for i, b in enumerate(sliced_buffers):
            if i % 2 == 0:
                ordered_buffers[i // 2] = b
            else:
                ordered_buffers[size * 2 - (i // 2) - 1] = b
        return torch.cat(ordered_buffers, dim=seq_dim)


def _context_parallel_buffers_2d(
    mesh: DeviceMesh,
    buffers: list[torch.Tensor],
    buffer_seq_dims: list[int],
) -> list[torch.Tensor]:
    """Shard the buffers along the sequence dimensions according to CP rules, supporting 2D meshes."""
    new_buffers = []
    sharder = (
        _RoundRobinLoadBalancer2D
        if _cp_options.enable_load_balance
        else _SequentialSharder2D
    )
    for buffer, seq_dim in zip(buffers, buffer_seq_dims):
        new_buffers.append(sharder.shard(buffer, mesh, seq_dim))
    return new_buffers


# Store the global 2D mesh for the custom dispatcher
_global_2d_mesh = None


def _set_global_2d_mesh(mesh: Optional[DeviceMesh]) -> None:
    """Set the global 2D mesh for attention operations."""
    global _global_2d_mesh
    _global_2d_mesh = mesh


def _get_global_2d_mesh() -> Optional[DeviceMesh]:
    """Get the global 2D mesh for attention operations."""
    return _global_2d_mesh


@contextlib.contextmanager
def _context_parallel_2d(seq_dim: int, mesh: DeviceMesh):
    """Replace SDPA with the CP-wrapped version and enable DTensor CP dispatcher for 2D meshes."""

    def attention_input_fn(
        mesh: DeviceMesh, *args: tuple[Any, ...], **kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        # For 2D mesh, we need to specify placements for both dimensions
        if mesh.ndim == 2:
            # dim 0 = cp_ring (shard sequence), dim 1 = cp_ulysses (replicate)
            placement = [Shard(seq_dim), Replicate()]
        else:
            # 1D mesh - standard sharding
            placement = [Shard(seq_dim)]

        all_args = []

        for arg in itertools.chain(args, kwargs.values()):
            if isinstance(arg, torch.Tensor) and not isinstance(arg, DTensor):
                arg = DTensor.from_local(arg, mesh, placement, run_check=False)

            all_args.append(arg)

        new_args = tuple(all_args[0 : len(args)])
        new_kwargs = dict(zip(kwargs.keys(), all_args[len(args) :]))
        return new_args, new_kwargs

    def attention_output_fn(mesh: DeviceMesh, outputs: Any) -> Any:
        new_outputs = []
        for output in [outputs] if isinstance(outputs, torch.Tensor) else outputs:
            output = output.to_local() if isinstance(output, DTensor) else output
            new_outputs.append(output)

        if isinstance(outputs, torch.Tensor):
            return new_outputs[0]

        return tuple(new_outputs)

    # Use monkey patching approach
    from torch.distributed.tensor.experimental._attention import (
        _distribute_function,
        _restore_function,
    )

    _distribute_function(
        F.scaled_dot_product_attention,
        F,
        mesh,
        attention_input_fn,
        attention_output_fn,
    )
    with _enable_cp_dispatcher_2d():
        yield
    _restore_function(F.scaled_dot_product_attention, F)


@contextlib.contextmanager
def _enable_cp_dispatcher_2d():
    """Enable CP dispatcher with 2D mesh support by using custom handlers."""
    # Import needed references
    from torch.distributed.tensor.experimental._attention import aten

    # Custom handlers that use 2D mesh wrapper
    customized_ops_2d = {
        # aten._scaled_dot_product_flash_attention.default: _sdpa_handler_2d,
        # aten._scaled_dot_product_efficient_attention.default: _sdpa_handler_2d,
        # aten._scaled_dot_product_cudnn_attention.default: _sdpa_handler_2d,
    }

    old_handlers = DTensor._op_dispatcher._custom_op_handlers
    DTensor._op_dispatcher._custom_op_handlers = {**old_handlers, **customized_ops_2d}

    try:
        yield
    finally:
        DTensor._op_dispatcher._custom_op_handlers = old_handlers


@contextlib.contextmanager
@torch.no_grad()
def context_parallel_2d(
    mesh: DeviceMesh,
    *,
    buffers: Optional[list[torch.Tensor]] = None,
    buffer_seq_dims: Optional[list[int]] = None,
    no_restore_buffers: Optional[Set[torch.Tensor]] = None,
) -> contextlib.AbstractContextManager[None]:
    """
    A 2D-aware context parallel implementation that supports both ring and Ulysses dimensions.

    This implementation shards buffers along the ring dimension while preserving the full
    2D mesh for attention operations to enable Ulysses-style head sharding.
    """

    buffers = [] if buffers is None else buffers
    buffer_seq_dims = [] if buffer_seq_dims is None else buffer_seq_dims
    no_restore_buffers = set() if no_restore_buffers is None else no_restore_buffers

    # Validate mesh dimensions for 2D context parallelism
    # dim 0 = cp_ring (outer, cross-node), dim 1 = cp_ulysses (inner, intra-node)
    if mesh.ndim == 2:
        if mesh.mesh_dim_names != ("cp_ring", "cp_ulysses"):
            raise ValueError(
                f"2D context parallel mesh must have dimensions ('cp_ring', 'cp_ulysses'), "
                f"but got {mesh.mesh_dim_names}"
            )

    if len(buffers) != len(buffer_seq_dims):
        raise ValueError(
            "`seq_dims` must have the same number of elements as `buffers`."
        )

    for buffer in no_restore_buffers:
        if not any(b is buffer for b in buffers):
            raise ValueError("`no_restore_buffers` must be a subset of `buffers`.")

    # Clone buffers that need to be restored
    original_buffers = [None if b in no_restore_buffers else b.clone() for b in buffers]

    # Shard buffers along ring dimension
    chunks = _context_parallel_buffers_2d(mesh, buffers, buffer_seq_dims)
    for buffer, chunk in zip(buffers, chunks):
        chunk = chunk.clone()
        buffer.resize_(chunk.shape)
        buffer.copy_(chunk)

    # Set the global 2D mesh for attention operations
    _set_global_2d_mesh(mesh)

    # Use our custom 2D context parallel implementation
    with _context_parallel_2d(seq_dim=2, mesh=mesh):
        yield

    # Clear the global mesh
    _set_global_2d_mesh(None)

    # Restore buffers
    for buffer, original_buffer in zip(buffers, original_buffers):
        if original_buffer is not None:
            buffer.resize_(original_buffer.shape)
            buffer.copy_(original_buffer)


@torch.no_grad()
def context_parallel_unshard_2d(
    mesh: DeviceMesh,
    buffers: list[torch.Tensor],
    seq_dims: list[int],
) -> list[torch.Tensor]:
    """
    Unshard the tensors that are sharded due to 2D context parallelism.
    """
    raise NotImplementedError("context_parallel_unshard_2d is not implemented")
    

class _DeviceMesh2DWrapper:
    """
    A wrapper around 2D DeviceMesh that provides the expected interface for _templated_ring_attention.
    """

    def __init__(self, mesh: DeviceMesh):
        if mesh.ndim != 2:
            raise ValueError(f"Expected 2D mesh, got {mesh.ndim}D mesh")
        if mesh.mesh_dim_names != ("cp_ring", "cp_ulysses"):
            raise ValueError(
                f"Expected mesh dimensions ('cp_ring', 'cp_ulysses'), got {mesh.mesh_dim_names}"
            )
        self.mesh = mesh
        self._ring_pg = mesh.get_group(0)  # cp_ring dimension (outer, cross-node)
        self._ulysses_pg = mesh.get_group(1)  # cp_ulysses dimension (inner, intra-node)

    def get_group(self):
        """Return tuple of (ring_pg, ulysses_pg) as expected by _templated_ring_attention."""
        return (self._ring_pg, self._ulysses_pg)

    def __getattr__(self, name):
        """Forward other attributes to the underlying mesh."""
        return getattr(self.mesh, name)


def _sdpa_handler_2d(
    op_call: torch._ops.OpOverload,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    """Custom SDPA handler that uses the 2D mesh wrapper."""

    # Import here to avoid circular imports
    from torch.distributed.tensor.experimental._attention import (
        _scaled_dot_product_ring_cudnn_attention,
        _scaled_dot_product_ring_efficient_attention,
        _scaled_dot_product_ring_flash_attention,
    )

    # Get the 2D mesh
    mesh = _get_global_2d_mesh()
    if mesh is None:
        raise RuntimeError("2D mesh not set for context parallel operations")

    # Wrap the mesh to provide the expected interface
    wrapped_mesh = _DeviceMesh2DWrapper(mesh)

    # Extract local tensor and sharding infos to a OpInfo
    op_info = DTensor._op_dispatcher.unwrap_to_op_info(op_call, args, kwargs)

    # Sharding propagation - this was missing
    DTensor._op_dispatcher.sharding_propagator.propagate(op_info)
    output_sharding = op_info.output_sharding
    assert output_sharding is not None, "output sharding should not be None"
    assert not output_sharding.needs_redistribute, "inputs need to be redistributed"

    # Use our wrapped 2D mesh
    if op_call == torch.ops.aten._scaled_dot_product_flash_attention.default:
        local_results = _scaled_dot_product_ring_flash_attention(
            wrapped_mesh,  # Use our wrapped 2D mesh
            *op_info.local_args,  # type: ignore[arg-type]
            **op_info.local_kwargs,  # type: ignore[arg-type]
        )
    elif op_call == torch.ops.aten._scaled_dot_product_efficient_attention.default:
        local_results = _scaled_dot_product_ring_efficient_attention(
            wrapped_mesh,  # Use our wrapped 2D mesh
            *op_info.local_args,  # type: ignore[arg-type]
            **op_info.local_kwargs,  # type: ignore[arg-type]
        )
    elif op_call == torch.ops.aten._scaled_dot_product_cudnn_attention.default:
        local_results = _scaled_dot_product_ring_cudnn_attention(
            wrapped_mesh,  # Use our wrapped 2D mesh
            *op_info.local_args,  # type: ignore[arg-type]
            **op_info.local_kwargs,  # type: ignore[arg-type]
        )
    else:
        raise NotImplementedError(
            "CP only supports flash attention and memory efficient attention now."
        )

    # Wrap results back to DTensor
    return DTensor._op_dispatcher.wrap(local_results, output_sharding.output_spec)


def set_rotate_method_2d(rotate_method: str) -> None:
    """
    Context Parallel SDPA requires the rotation of kv shards. Users can call this
    API to specify which rotation method to use. "alltoall" shuffles the kv shards
    using all-to-all collective. While "allgather" gathers the kv shards using
    all-gather collective after the first sub-SDPA computation. If this API has not
    been called, the default rotate method is "allgather".

    Args:
        rotate_method (str): the rotate method to use. Currently only supports
        "allgather" and "alltoall". If a different string other than these two
        is passed in, the function will raise an error.

    Returns:
        None
    """
    if rotate_method == "allgather":
        _cp_options.rotate_method = _RotateMethod.ALL_GATHER
    elif rotate_method == "alltoall":
        _cp_options.rotate_method = _RotateMethod.ALL_TO_ALL
    else:
        raise NotImplementedError(
            "Context Parallel does not support "
            f"using {rotate_method} for kv shards rotation"
        )


def set_load_balance_2d(load_balance: str) -> None:
    """
    Context Parallel SDPA requires the load balance of kv shards. Users can call this
    API to specify which load balance method to use. "round_robin" uses a round robin
    approach to achieve load balancing. "sequential" uses a sequential approach to
    achieve load balancing. If this API has not been called, the default load balance
    method is "round_robin".
    """
    _cp_options.enable_load_balance = "basic" not in load_balance
