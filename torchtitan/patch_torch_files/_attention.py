# Copyright (c) Meta Platforms, Inc. and affiliates

# path: pytorch/torch/distributed/tensor/experimental/_attention.py

import contextlib
import itertools
import logging
import types
import weakref
from abc import ABC, abstractmethod
from collections.abc import Generator
from dataclasses import dataclass
from enum import auto, Enum
from typing import Any, Callable, Optional, Protocol, Union

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as ft_c
import torch.nn.functional as F
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import distribute_module, DTensor, Replicate, Shard
from torch.distributed.tensor.parallel.style import ParallelStyle
from torch.overrides import TorchFunctionMode

# import os

from yunchang.comm.all_to_all import SeqAllToAll4D

__all__ = ["context_parallel", "set_rotate_method"]


class _CausalBehavior(Enum):
    SKIP = None
    NOT_IS_CAUSAL = False
    IS_CAUSAL = True


class _RotateMethod(Enum):
    ALL_TO_ALL = auto()
    ALL_GATHER = auto()


aten = torch.ops.aten
logger = logging.getLogger(__name__)


class _DispatchMode(Enum):
    MONKEY_PATCH = auto()
    TORCH_FUNCTION = auto()
    TORCH_DISPATCH = auto()


_dispatch_mode: _DispatchMode = _DispatchMode.MONKEY_PATCH


@dataclass
class _ContextParallelOptions:
    # Whether to upcast parameters and gradients to float32 to avoid accumulation
    # errors. It is likely this is always True but we currently keep this variable
    # for the experimental purpose.
    convert_to_f32: bool = True
    enable_load_balance = True
    rotate_method: _RotateMethod = (
        _RotateMethod.ALL_TO_ALL
    )  # _RotateMethod.ALL_GATHER #


_cp_options = _ContextParallelOptions()


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        torch.unsqueeze(x, dim=2)
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )


def _is_causal_behavior(
    rank: int, world_size: int, i: int, is_causal: bool
) -> _CausalBehavior:
    """
    Calculate is_causal behavior for each KV block. The attention can either be
    calculated in full, not at all or with the causal mask applied.
    """
    if not is_causal:
        return _CausalBehavior.NOT_IS_CAUSAL

    if i == 0:
        return _CausalBehavior.IS_CAUSAL

    source_rank = (rank - i) % world_size
    if source_rank < rank or _cp_options.enable_load_balance:
        return _CausalBehavior.NOT_IS_CAUSAL
    else:
        return _CausalBehavior.SKIP


def _maybe_wait(tensor: torch.Tensor) -> torch.Tensor:
    """
    When tracing the code, the result tensor is not an AsyncCollectiveTensor,
    so we cannot call ``wait()``.
    """
    if isinstance(tensor, ft_c.AsyncCollectiveTensor):
        return tensor.wait()
    return tensor


def _partial_update(
    original: torch.Tensor,
    new: torch.Tensor,
    dim: int,
    n_chunks: int,
    idx: int,
    add: bool,
) -> torch.Tensor:
    """
    This API partially update a chunk of ``original`` tensor. The ``original``
    tensor will be first chunked along ``dim`` dimension then the ``idx`` chunk
    will be updated with ``new``. If ``add`` is True, the chunk will be added
    with ``new``, otherwise the chunk with be replaced by ``add``.

    The result is a tensor that is the same size as ``original``.
    """
    # debug_on_rank(0)
    chunks = list(original.chunk(n_chunks, dim=dim))
    assert chunks[idx].shape == new.shape, (original.shape, new.shape, idx)
    if add:
        chunks[idx] += new
    else:
        chunks[idx] = new
    return torch.cat(chunks, dim=dim)


class _SDPAMerger:
    """A class to help to merge the local SDPA result."""

    def __init__(self, convert_to_f32: bool, seq_dim: int):
        self._seq_dim = seq_dim
        self._out: Optional[torch.Tensor] = None
        self._lse: Optional[torch.Tensor] = None
        self._convert_to_f32 = convert_to_f32
        self._out_dtype = torch.float32
        self._lse_dtype = torch.float32

    def _merge_one(
        self, block_out: torch.Tensor, block_lse: torch.Tensor, partial: bool
    ) -> None:
        block_lse = block_lse.unsqueeze(dim=-1)
        if self._lse is None:
            self._lse = block_lse
            self._out = block_out
        else:
            ROUND_ROBIN_CYCLE = 2
            assert self._lse is not None
            assert self._out is not None
            lse = (
                self._lse.chunk(ROUND_ROBIN_CYCLE, dim=self._seq_dim)[1]
                if partial
                else self._lse
            )
            out = (
                self._out.chunk(ROUND_ROBIN_CYCLE, dim=self._seq_dim)[1]
                if partial
                else self._out
            )

            # The algorithm from
            # github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795
            # gives a relatively stable result.
            out = out - F.sigmoid(block_lse - lse) * (out - block_out)
            lse = lse - F.logsigmoid(lse - block_lse)
            if partial:
                self._lse = _partial_update(
                    self._lse,
                    lse,
                    dim=self._seq_dim,
                    n_chunks=ROUND_ROBIN_CYCLE,
                    idx=1,
                    add=False,
                )
                self._out = _partial_update(
                    self._out,
                    out,
                    dim=self._seq_dim,
                    n_chunks=ROUND_ROBIN_CYCLE,
                    idx=1,
                    add=False,
                )
            else:
                self._lse = lse
                self._out = out

    def step(self, out: torch.Tensor, lse: torch.Tensor, partial: bool) -> None:
        self._out_dtype = out.dtype
        self._lse_dtype = lse.dtype

        if self._convert_to_f32:
            out = out.to(torch.float32)
            lse = lse.to(torch.float32)

        self._merge_one(out, lse, partial)

    def results(self) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._out is not None
        assert self._lse is not None
        out, lse = self._out, self._lse.squeeze(-1)
        return out.to(self._out_dtype), lse.to(self._lse_dtype)


def _scaled_dot_product_ring_flash_attention(
    mesh: DeviceMesh,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    *,
    scale: Optional[float] = None,
) -> tuple[torch.Tensor, ...]:
    if return_debug_mask:
        raise NotImplementedError("return_debug_mask is not supported yet")

    seq_dim = 2
    return _templated_ring_attention(
        mesh,
        seq_dim,
        aten._scaled_dot_product_flash_attention,
        query=query,
        key=key,
        value=value,
        is_causal=is_causal,
        dropout_p=dropout_p,
        scale=scale,
    )


def _scaled_dot_product_ring_efficient_attention(
    mesh: DeviceMesh,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: Optional[torch.Tensor] = None,
    compute_log_sumexp: bool = True,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: Optional[float] = None,
) -> tuple[torch.Tensor, ...]:
    if attn_bias is not None:
        raise NotImplementedError("attn_bias is not supported yet")

    if not compute_log_sumexp:
        # CP requires compute_log_sumexp to be True because it always merges LSE
        compute_log_sumexp = True

    seq_dim = 2
    return _templated_ring_attention(
        mesh,
        seq_dim,
        aten._scaled_dot_product_efficient_attention,
        query=query,
        key=key,
        value=value,
        is_causal=is_causal,
        attn_bias=attn_bias,
        dropout_p=dropout_p,
        scale=scale,
        compute_log_sumexp=compute_log_sumexp,
    )


def _scaled_dot_product_ring_cudnn_attention(
    mesh: DeviceMesh,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: Optional[torch.Tensor] = None,
    compute_log_sumexp: bool = True,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    *,
    scale: Optional[float] = None,
) -> tuple[torch.Tensor, ...]:
    if attn_bias is not None:
        raise NotImplementedError("attn_bias is not supported yet")

    if not compute_log_sumexp:
        # CP requires compute_log_sumexp to be True because it always merges LSE
        compute_log_sumexp = True

    seq_dim = 2
    return _templated_ring_attention(
        mesh,
        seq_dim,
        aten._scaled_dot_product_cudnn_attention,
        query=query,
        key=key,
        value=value,
        attn_bias=attn_bias,
        compute_log_sumexp=compute_log_sumexp,
        dropout_p=dropout_p,
        is_causal=is_causal,
        return_debug_mask=return_debug_mask,
        scale=scale,
    )


class _AttentionOp(Protocol):
    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        **kwargs: object,
    ) -> tuple[torch.Tensor, ...]:
        ...


class _RingRotater(ABC):
    @abstractmethod
    def __init__(self, pg: dist.ProcessGroup, seq_dim: int) -> None:
        ...

    @abstractmethod
    def exchange_buffers(self, curr_buffer: torch.Tensor) -> None:
        ...

    @abstractmethod
    def next_buffer(self) -> torch.Tensor:
        ...


class _AllToAllRotater(_RingRotater):
    """Use all_to_all to send the kv to the next rank"""

    def __init__(self, pg: dist.ProcessGroup, seq_dim: int) -> None:
        self._pg = pg
        self._seq_dim = seq_dim
        self._buffer: Optional[torch.Tensor] = None

    def exchange_buffers(self, curr_buffer: torch.Tensor) -> None:
        curr_buffer = curr_buffer.contiguous()
        size = dist.get_world_size(self._pg)
        dsts = list(range(1, size)) + [0]
        self._buffer = ft_c.permute_tensor(curr_buffer, dsts, self._pg)

    def next_buffer(self) -> torch.Tensor:
        assert self._buffer is not None
        return _maybe_wait(self._buffer)


class _AllGatherRotater(_RingRotater):
    """
    Allgather the kv and return the only the requried kv.
    Only one communication will be done.
    """

    def __init__(self, pg: dist.ProcessGroup, seq_dim: int) -> None:
        self._pg = pg
        self._seq_dim = seq_dim
        self._aggregated_buffer: Optional[torch.Tensor] = None
        self._idx = 0

    def exchange_buffers(self, curr_buffer: torch.Tensor) -> None:
        # We only need to perform the allgather once.
        self._idx += 1
        if self._aggregated_buffer is None:
            self._aggregated_buffer = ft_c.all_gather_tensor(
                curr_buffer.contiguous(), gather_dim=0, group=self._pg
            )

    def next_buffer(self) -> torch.Tensor:
        rank = dist.get_rank(self._pg)
        idx = rank - self._idx

        assert self._aggregated_buffer is not None
        self._aggregated_buffer = _maybe_wait(self._aggregated_buffer)
        return self._aggregated_buffer.chunk(dist.get_world_size(self._pg))[idx]


def _create_rotater(
    pg: dist.ProcessGroup, seq_dim: int, method: Optional[_RotateMethod] = None
) -> _RingRotater:
    if method is None:
        method = _cp_options.rotate_method

    if method == _RotateMethod.ALL_TO_ALL:
        return _AllToAllRotater(pg, seq_dim)
    elif method == _RotateMethod.ALL_GATHER:
        return _AllGatherRotater(pg, seq_dim)
    else:
        raise NotImplementedError(f"Unkonwn method {method}")


def _ring_rotate(
    block: torch.Tensor, pg: dist.ProcessGroup, send_to_next: bool
) -> torch.Tensor:
    block = block.contiguous()
    size = dist.get_world_size(pg)
    dsts = (
        list(range(1, size)) + [0]
        if send_to_next
        else [size - 1] + list(range(0, size - 1))
    )
    return ft_c.permute_tensor(block, dsts, pg)


def _ulysses_all_to_all_chunked(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    ulysses_pg: dist.ProcessGroup,
    ulysses_size: int,
    chunk_size: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Ultra memory-efficient Ulysses all-to-all that processes heads in chunks.
    This reduces peak memory usage from 2x to approximately 1.25x.

    Args:
        query, key, value: Input tensors of shape [B, H, S/(U*R), D]
        ulysses_pg: Ulysses process group
        ulysses_size: Number of Ulysses ranks
        chunk_size: Number of heads to process at once

    Returns:
        Resharded tensors of shape [B, H/U, S/R, D]
    """
    B, H, S_shard, D = query.shape
    assert (
        H % ulysses_size == 0
    ), f"H={H} must be divisible by ulysses_size={ulysses_size}"
    assert H % chunk_size == 0, f"H={H} must be divisible by chunk_size={chunk_size}"

    H_per_rank = H // ulysses_size
    num_chunks = H // chunk_size

    # Process heads in chunks to minimize peak memory
    query_chunks_out = []
    key_chunks_out = []
    value_chunks_out = []

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = (chunk_idx + 1) * chunk_size

        # Extract chunk
        q_chunk = query[:, start_idx:end_idx].contiguous()
        k_chunk = key[:, start_idx:end_idx].contiguous()
        v_chunk = value[:, start_idx:end_idx].contiguous()

        # Process this chunk using standard all-to-all
        chunk_per_rank = chunk_size // ulysses_size

        # Reshape chunk for all-to-all
        q_chunk_reshaped = q_chunk.view(B, ulysses_size, chunk_per_rank, S_shard, D)
        k_chunk_reshaped = k_chunk.view(B, ulysses_size, chunk_per_rank, S_shard, D)
        v_chunk_reshaped = v_chunk.view(B, ulysses_size, chunk_per_rank, S_shard, D)

        # Split by Ulysses ranks
        q_splits = [s.contiguous() for s in q_chunk_reshaped.unbind(1)]
        k_splits = [s.contiguous() for s in k_chunk_reshaped.unbind(1)]
        v_splits = [s.contiguous() for s in v_chunk_reshaped.unbind(1)]

        # Pre-allocate output buffers for this chunk
        q_gathered = [torch.empty_like(q_splits[0]) for _ in range(ulysses_size)]
        k_gathered = [torch.empty_like(k_splits[0]) for _ in range(ulysses_size)]
        v_gathered = [torch.empty_like(v_splits[0]) for _ in range(ulysses_size)]

        # Perform all-to-all
        dist.all_to_all(q_gathered, q_splits, group=ulysses_pg)
        dist.all_to_all(k_gathered, k_splits, group=ulysses_pg)
        dist.all_to_all(v_gathered, v_splits, group=ulysses_pg)

        # Free splits immediately
        del q_splits, k_splits, v_splits
        del q_chunk_reshaped, k_chunk_reshaped, v_chunk_reshaped

        # Concatenate sequence dimension for this chunk
        q_chunk_out = torch.cat(q_gathered, dim=2)  # [B, chunk_per_rank, S/R, D]
        k_chunk_out = torch.cat(k_gathered, dim=2)
        v_chunk_out = torch.cat(v_gathered, dim=2)

        # Free gathered lists
        del q_gathered, k_gathered, v_gathered

        query_chunks_out.append(q_chunk_out)
        key_chunks_out.append(k_chunk_out)
        value_chunks_out.append(v_chunk_out)

        # Free chunk memory
        del q_chunk, k_chunk, v_chunk

    # Concatenate all chunks along head dimension
    query_final = torch.cat(query_chunks_out, dim=1)  # [B, H/U, S/R, D]
    key_final = torch.cat(key_chunks_out, dim=1)
    value_final = torch.cat(value_chunks_out, dim=1)

    # Free chunk lists
    del query_chunks_out, key_chunks_out, value_chunks_out

    return query_final, key_final, value_final


def _templated_ring_attention(
    mesh: DeviceMesh,
    seq_dim: int,
    op: _AttentionOp,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = False,
    **kwargs: object,
) -> tuple[torch.Tensor, ...]:
    """
    This is a generalized ring attention implementation that can support multiple attention ops.

    Note [Context parallelism load balance algorithm for causal masking]
    =====================
    This explanation uses an example to illustrate the CP algorithm with causal
    masking.

    Consider a scenario where the sequence length of q, k, and v is 4 (e.g.,
    q = (q0, q1, q2, q3)), and there are two ranks. For simplicity, we will discuss
    only q and k, as v follows the same pattern as k.

    The diagram below represents a complete QK^T operation without parallelism.
    The `****` entries indicate that the result is not required due to causal
    masking (e.g., q0k1 is marked as `****`).

    +----+------------------------+
    |    |  k0    k1   k2     k3  |
    +----+------------------------+
    | q0 | q0k0, ****, ****, **** |
    | q1 | q1k0, q1k1, ****, **** |
    | q2 | q2k0, q2k1, q2k2, **** |
    | q3 | q3k0, q3k1, q3k2, q3k3 |
    +----+------------------------+

    ### No Load Balance:

    In this scenario, each rank owns a local chunk of q, k, and v, with each chunk
    containing two elements. Rank0 is responsible for managing (q0, q1) and (k0, k1),
    while rank1 manages (q2, q3) and (k2, k3).

    First Iteration: Both rank0 and rank1 perform SDPA with their local qkv pairs.
    Causal masking is enabled as some results are not required (e.g., q0k1).

    Second Iteration: Local queries remain the same, but local kv pairs are exchanged.
    Rank0 now has (q0, q1) and (k2, k3); rank1 has (q2, q3) and (k0, k1). Rank0 performs
    no computation, while rank1 computes locally without causal masking since all results
    (q2k0, q2k1, q3k0, q3k1) are needed.

    ### Round-robin Load Balance:

    In this setup, each rank owns two local chunks of q, k, and v, with each chunk
    containing one element. Rank0 manages (q0, q3) and (k0, k3); Rank1 manages (q1, q2)
    and (k1, k2). Although the local chunks are not consecutive, they are concatenated to
    enable SDPA to be performed in a single call for each step. Consequently, the chunk()
    function may be required to prepare the correct q, k, and v configurations.

    First Iteration: Both ranks perform SDPA with their local qkv pairs, similar to the
    no-load-balance case. This iteration corresponds to the `if` of the
    (`if, `elif`, `else`) in the implemementation.

    Second Iteration: Rank0 now has (q0, q3) and (k1, k2); rank1 has (q1, q2) and
    (k0, k3). For rank0, no computation is needed for q0. However, computations for
    q3k1 and q3k2 are required, so only q3 is used for SDPA. This corresponds to the
    `else` of the (`if`, `elif`, `else`) in the implemementation.
    For rank1, k0 is not needed for q1 and q2, so only k3 is used for SDPA. This
    corresponds to the `elif` of (`if`, `elif`, `else`) in the implementation.

    Parameters
    ----------
    op:
        The attention op to use
    *args:
        additional args are passed to the op
    **kwargs:
        additional kwargs are passed to the op

    Returns
    -------
    out:
        The merged attention output
    softmax_lse:
        The logsumexp of the merged attention output
    """
    if is_causal and (query.size(2) != key.size(2)):
        raise NotImplementedError(
            "is_causal requires the same query and context sequence lengths"
        )
    if not is_causal and _cp_options.enable_load_balance:
        raise RuntimeError("Load balancing requires `is_causal=True`.")

    if isinstance(mesh, dist.ProcessGroup):
        pg: Union[dist.ProcessGroup, list[dist.ProcessGroup]] = mesh
    else:
        pg = mesh.get_group()
    assert isinstance(pg, dist.ProcessGroup), "process group must be single dimension"
    rank = dist.get_rank(pg)
    size = dist.get_world_size(pg)

    next_kv = None

    # Without making key and value contiguous(), the lose curve is bad.
    # TODO(fegin): figure out why this is a requirement since SDPA does not have
    # this requirement.
    key = key.contiguous()
    value = value.contiguous()

    sdpa_merger = _SDPAMerger(_cp_options.convert_to_f32, seq_dim=seq_dim)

    rest: list[Any]
    out: torch.Tensor
    logsumexp: torch.Tensor

    rotater = _create_rotater(pg, 2)

    for i in range(size):
        if i > 0:
            # Wait for the kv from the (cp_rank - 1) rank.
            next_kv = rotater.next_buffer()
            key = next_kv[: key.numel()].reshape(key.shape)
            value = next_kv[key.numel() :].reshape(value.shape)

        if i < (size - 1):
            # Send the k, v to the next rank
            next_kv = torch.cat([key.flatten(), value.flatten()])
            next_kv = rotater.exchange_buffers(next_kv)

        is_causal_behavior = _is_causal_behavior(
            rank=rank, world_size=size, i=i, is_causal=is_causal
        )

        # For a detailed understanding of the load balancing algorithm, see
        # Note [Context parallelism load balance algorithm for causal masking]
        if is_causal_behavior == _CausalBehavior.SKIP:
            # If i > rank and load balancing is not turned on.
            continue

        if i == 0 or (not _cp_options.enable_load_balance or not is_causal):
            # When local balance is enabled, we still need to do SDPA with
            # the both local chunks of q, k, v for the first iteration.
            q, k, v, partial = (query, key, value, False)
        elif i <= rank:
            # Round-robin load balancing case, and i <= rank.
            # We need to do SPDA, with only the first local chunk of the k, v.
            # Note that q, k, v, each contains two local chunks.
            ROUND_ROBIN_CYCLE = 2
            q, k, v, partial = (
                query,
                key.chunk(ROUND_ROBIN_CYCLE, dim=2)[0],
                value.chunk(ROUND_ROBIN_CYCLE, dim=2)[0],
                False,
            )
        else:
            # Round-robin load balancing case, and i > rank.
            # We need to do SPDA with only the second half of the q, and update
            # only the the second part of  logsumexp. So partial is True.
            # Note that q, k, v, each contains two chunks.
            q, k, v, partial = query.chunk(2, dim=2)[1], key, value, True

        # See https://github.com/pytorch/pytorch/blob/release/2.4/aten/src/ATen/native/native_functions.yaml#L14695
        # for the SDPA kernel definitions.

        out, logsumexp, *rest = op(
            q,
            k,
            v,
            is_causal=is_causal_behavior.value,
            **kwargs,
        )
        sdpa_merger.step(out, logsumexp, partial)

    return *sdpa_merger.results(), *rest


def _templated_ring_attention_ulysses(
    mesh: DeviceMesh,
    seq_dim: int,
    op: _AttentionOp,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = False,
    **kwargs: object,
) -> tuple[torch.Tensor, ...]:
    """
    This is a generalized ring attention implementation that can support multiple attention ops.

    Note [Context parallelism load balance algorithm for causal masking]
    =====================
    This explanation uses an example to illustrate the CP algorithm with causal
    masking.

    Consider a scenario where the sequence length of q, k, and v is 4 (e.g.,
    q = (q0, q1, q2, q3)), and there are two ranks. For simplicity, we will discuss
    only q and k, as v follows the same pattern as k.

    The diagram below represents a complete QK^T operation without parallelism.
    The `****` entries indicate that the result is not required due to causal
    masking (e.g., q0k1 is marked as `****`).

    +----+------------------------+
    |    |  k0    k1   k2     k3  |
    +----+------------------------+
    | q0 | q0k0, ****, ****, **** |
    | q1 | q1k0, q1k1, ****, **** |
    | q2 | q2k0, q2k1, q2k2, **** |
    | q3 | q3k0, q3k1, q3k2, q3k3 |
    +----+------------------------+

    ### No Load Balance:

    In this scenario, each rank owns a local chunk of q, k, and v, with each chunk
    containing two elements. Rank0 is responsible for managing (q0, q1) and (k0, k1),
    while rank1 manages (q2, q3) and (k2, k3).

    First Iteration: Both rank0 and rank1 perform SDPA with their local qkv pairs.
    Causal masking is enabled as some results are not required (e.g., q0k1).

    Second Iteration: Local queries remain the same, but local kv pairs are exchanged.
    Rank0 now has (q0, q1) and (k2, k3); rank1 has (q2, q3) and (k0, k1). Rank0 performs
    no computation, while rank1 computes locally without causal masking since all results
    (q2k0, q2k1, q3k0, q3k1) are needed.

    ### Round-robin Load Balance:

    In this setup, each rank owns two local chunks of q, k, and v, with each chunk
    containing one element. Rank0 manages (q0, q3) and (k0, k3); Rank1 manages (q1, q2)
    and (k1, k2). Although the local chunks are not consecutive, they are concatenated to
    enable SDPA to be performed in a single call for each step. Consequently, the chunk()
    function may be required to prepare the correct q, k, and v configurations.

    First Iteration: Both ranks perform SDPA with their local qkv pairs, similar to the
    no-load-balance case. This iteration corresponds to the `if` of the
    (`if, `elif`, `else`) in the implemementation.

    Second Iteration: Rank0 now has (q0, q3) and (k1, k2); rank1 has (q1, q2) and
    (k0, k3). For rank0, no computation is needed for q0. However, computations for
    q3k1 and q3k2 are required, so only q3 is used for SDPA. This corresponds to the
    `else` of the (`if`, `elif`, `else`) in the implemementation.
    For rank1, k0 is not needed for q1 and q2, so only k3 is used for SDPA. This
    corresponds to the `elif` of (`if`, `elif`, `else`) in the implementation.

    Parameters
    ----------
    op:
        The attention op to use
    *args:
        additional args are passed to the op
    **kwargs:
        additional kwargs are passed to the op

    Returns
    -------
    out:
        The merged attention output
    softmax_lse:
        The logsumexp of the merged attention output
    """
    if is_causal and (query.size(2) != key.size(2)):
        raise NotImplementedError(
            "is_causal requires the same query and context sequence lengths"
        )
    if not is_causal and _cp_options.enable_load_balance:
        raise RuntimeError("Load balancing requires `is_causal=True`.")

    if isinstance(mesh, dist.ProcessGroup):
        pg: Union[dist.ProcessGroup, list[dist.ProcessGroup]] = mesh
        ring_pg = pg
        ulysses_pg = None
    else:
        # Handle both 1D and 2D DeviceMesh
        pg_result = mesh.get_group()
        if isinstance(pg_result, tuple):
            # 2D mesh case - returns (ring_pg, ulysses_pg)
            ring_pg, ulysses_pg = pg_result
        else:
            # 1D mesh case - returns single ProcessGroup
            ring_pg = pg_result
            ulysses_pg = None
    assert isinstance(
        ring_pg, dist.ProcessGroup
    ), "Ring process group must be single dimension"

    # Handle Ulysses parallelism if applicable
    if ulysses_pg is not None:
        assert isinstance(
            ulysses_pg, dist.ProcessGroup
        ), "Ulysses process group must be single dimension"
        ulysses_rank = dist.get_rank(ulysses_pg)
        ulysses_size = dist.get_world_size(ulysses_pg)

        # Check if we should use memory-efficient chunked processing
        # This can be controlled by an environment variable
        use_chunked_ulysses = os.environ.get("TORCH_ULYSSES_CHUNKED", "0") == "1"
        use_yunchang_ulysses = os.environ.get("TORCH_ULYSSES_YUNCHANG", "0") == "1"
        use_async_a2a = os.environ.get("TORCH_ULYSSES_ASYNC_A2A", "0") == "1"
        chunk_size = int(os.environ.get("TORCH_ULYSSES_CHUNK_SIZE", "8"))

        if use_chunked_ulysses and query.size(1) >= chunk_size:
            # Ultra memory-efficient chunked processing
            logger.info(
                f"Using chunked Ulysses all-to-all with chunk_size={chunk_size}"
            )
            query, key, value = _ulysses_all_to_all_chunked(
                query, key, value, ulysses_pg, ulysses_size, chunk_size
            )
        elif use_yunchang_ulysses:
            # debug_on_rank(0)  # Break only on rank 0
            # Use Yunchang's all-to-all
            query = SeqAllToAll4D.apply(ulysses_pg, query, 1, 2, False)
            key = SeqAllToAll4D.apply(ulysses_pg, key, 1, 2, False)
            value = SeqAllToAll4D.apply(ulysses_pg, value, 1, 2, False)
        elif use_async_a2a:
            # expects bs, seq, hc, hs

            logger.info(f"Using Async All-to-All...\n")

            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)

            ulysses_degree = dist.get_world_size(ulysses_pg)
            a2a_stream = torch.cuda.Stream()

            bs, shard_seqlen, hc, hs = query.shape
            bs, shard_seqlen, hc_kv, hs = key.shape
            seq_len = shard_seqlen * ulysses_degree
            un = hc // ulysses_degree
            un_kv = hc_kv // ulysses_degree

            assert un_kv == un, f"un_kv {un_kv} un {un}"

            qkv = torch.cat([query, key, value]).contiguous()
            # (3*bs, seqlen/P, hc, hs) -> (hc, seqlen/P, 3*bs, hs) -> (un, ud, seqlen/P, 3*bs, hs), where hc = un*ud
            qkv_list = torch.unbind(
                qkv.transpose(0, 2)
                .contiguous()
                .reshape(un, ulysses_degree, shard_seqlen, 3 * bs, hs)
            )
            # 3xall-to-all output buffer
            qkv_trans_list = [
                torch.zeros(
                    ulysses_degree,
                    1,
                    shard_seqlen,
                    3 * bs,
                    hs,
                    dtype=query.dtype,
                    device=query.device,
                )
                for i in range(len(qkv_list))
            ]

            # last all-to-all buffter
            context_layer_list = [
                torch.zeros(
                    ulysses_degree,
                    1,
                    shard_seqlen,
                    bs,
                    hs,
                    dtype=query.dtype,
                    device=query.device,
                )
                for i in range(len(qkv_list))
            ]

            comm_handle_list = []

            # un * (ud, shard_seqlen, 3*bs, hs)
            for i, qkv in enumerate(qkv_list):
                with torch.cuda.stream(a2a_stream):
                    ret = dist.all_to_all_single(
                        qkv_trans_list[i],
                        qkv,
                        group=ulysses_pg,
                        async_op=True,
                    )
                comm_handle_list.append(ret)

            last_comm_handle_list = []
            for i_head, qkv_trans in enumerate(qkv_trans_list):
                if comm_handle_list[i_head] is not None:
                    comm_handle_list[i_head].wait()
                qkv_trans = (
                    qkv_trans.reshape(seq_len, 3 * bs, 1, hs)
                    .transpose(0, 1)
                    .contiguous()
                    .reshape(3 * bs, seq_len, 1, hs)
                )

                # qkv_trans = all_to_all_4D_async(qkv, qkv_trans_list[i], self.scatter_idx, self.gather_idx, self.ulysses_pg)
                qkv_trans = torch.chunk(qkv_trans, 3, dim=0)

                query_chunks = qkv_trans[0].transpose(1, 2)
                key_chunks = qkv_trans[1].transpose(1, 2)
                value_chunks = qkv_trans[2].transpose(1, 2)

                # Now perform ring attention on the resharded tensors
                rank = dist.get_rank(ring_pg)
                size = dist.get_world_size(ring_pg)

                next_kv = None

                # Without making key and value contiguous(), the lose curve is bad.
                # TODO(fegin): figure out why this is a requirement since SDPA does not have
                # this requirement.
                key_chunks = key_chunks.contiguous()
                value_chunks = value_chunks.contiguous()

                sdpa_merger = _SDPAMerger(
                    _cp_options.convert_to_f32, seq_dim=seq_dim
                )  # _cp_options.convert_to_f32,

                rest: list[Any]
                out: torch.Tensor
                logsumexp: torch.Tensor

                rotater = _create_rotater(ring_pg, 2)

                for i in range(size):
                    if i > 0:
                        # Wait for the kv from the (cp_rank - 1) rank.
                        next_kv = rotater.next_buffer()
                        key_chunks = next_kv[: key_chunks.numel()].reshape(
                            key_chunks.shape
                        )
                        value_chunks = next_kv[key_chunks.numel() :].reshape(
                            value_chunks.shape
                        )

                    if i < (size - 1):
                        # Send the k, v to the next rank
                        next_kv = torch.cat(
                            [key_chunks.flatten(), value_chunks.flatten()]
                        )
                        next_kv = rotater.exchange_buffers(next_kv)

                    is_causal_behavior = _is_causal_behavior(
                        rank=rank, world_size=size, i=i, is_causal=is_causal
                    )

                    # For a detailed understanding of the load balancing algorithm, see
                    # Note [Context parallelism load balance algorithm for causal masking]
                    if is_causal_behavior == _CausalBehavior.SKIP:
                        # If i > rank and load balancing is not turned on.
                        continue

                    if i == 0 or (not _cp_options.enable_load_balance or not is_causal):
                        # When local balance is enabled, we still need to do SDPA with
                        # the both local chunks of q, k, v for the first iteration.
                        q, k, v, partial = (
                            query_chunks,
                            key_chunks,
                            value_chunks,
                            False,
                        )
                    elif i <= rank:
                        # Round-robin load balancing case, and i <= rank.
                        # We need to do SPDA, with only the first local chunk of the k, v.
                        # Note that q, k, v, each contains two local chunks.
                        ROUND_ROBIN_CYCLE = 2
                        q, k, v, partial = (
                            query,
                            key_chunks.chunk(ROUND_ROBIN_CYCLE, dim=2)[0],
                            value_chunks.chunk(ROUND_ROBIN_CYCLE, dim=2)[0],
                            False,
                        )
                    else:
                        # Round-robin load balancing case, and i > rank.
                        # We need to do SPDA with only the second half of the q, and update
                        # only the the second part of  logsumexp. So partial is True.
                        # Note that q, k, v, each contains two chunks.
                        q, k, v, partial = (
                            query_chunks.chunk(2, dim=2)[1],
                            key_chunks,
                            value_chunks,
                            True,
                        )

                    # See https://github.com/pytorch/pytorch/blob/release/2.4/aten/src/ATen/native/native_functions.yaml#L14695
                    # for the SDPA kernel definitions.
                    # debug_on_rank(0)  # Break only on rank 0
                    out, logsumexp, *rest = op(
                        q,
                        k,
                        v,
                        is_causal=is_causal_behavior.value,
                        **kwargs,
                    )
                    sdpa_merger.step(out, logsumexp, partial)

                # Get merged results
                out, logsumexp = sdpa_merger.results()

                # debug_on_rank(0)

                context_layer = out.transpose(1, 2)
                # (bs, seq_len, head_cnt/N, head_size) -> (bs, seq_len/N, head_cnt, head_size)
                # scatter 1, gather 2

                context_layer = (
                    context_layer.reshape(bs, ulysses_degree, shard_seqlen, 1, hs)
                    .transpose(0, 3)
                    .transpose(0, 1)
                    .contiguous()
                    .reshape(ulysses_degree, 1, shard_seqlen, bs, hs)
                )
                with torch.cuda.stream(a2a_stream):
                    ret = dist.all_to_all_single(
                        context_layer_list[i_head],
                        context_layer,
                        group=ulysses_pg,
                        async_op=True,
                    )
                last_comm_handle_list.append(ret)

            # hc = un * P
            # un x (hc = P, seq_len/P, bs, hs) -> (bs, seq_len, hc = P, hs)
            for i_last, ret in enumerate(last_comm_handle_list):
                if ret is not None:
                    ret.wait()
                context_layer_list[i_last] = (
                    context_layer_list[i_last]
                    .reshape(ulysses_degree, shard_seqlen, bs, hs)
                    .transpose(0, 2)
                    .contiguous()
                    .reshape(bs, shard_seqlen, ulysses_degree, hs)
                )

            out = torch.cat(context_layer_list, dim=2).transpose(1, 2)
            # debug_on_rank(0)
            return out, logsumexp, *rest

        else:
            # Standard all-to-all processing
            # Perform all-to-all to reshard from [B, S/(U*R), H, D] to [B, S/R, H/U, D]
            # where U = ulysses_size and R = ring_size
            head_dim = 2  # Assuming head dimension is at position 2

            # Reshape tensors for all-to-all: split heads across Ulysses dimension
            # Original: [B, S/(UR), H, D]
            # After chunking: [B, S/(UR), U, H/U, D]
            B, H, S_shard, D = query.shape
            assert (
                H % ulysses_size == 0
            ), f"Number of heads {H} must be divisible by Ulysses size {ulysses_size}"

            # Chunk heads for all-to-all
            query_chunks = query.view(B, H // ulysses_size, ulysses_size, S_shard, D)
            key_chunks = key.view(B, H // ulysses_size, ulysses_size, S_shard, D)
            value_chunks = value.view(B, H // ulysses_size, ulysses_size, S_shard, D)

            # Memory-efficient all-to-all using in-place operations
            # Option 1: Use all_to_all_single which can be more memory efficient
            if hasattr(dist, "all_to_all_single"):  # TODO needs testing
                # Flatten for all_to_all_single
                # query_flat = torch.cat(query_splits, dim=0)
                # key_flat = torch.cat(key_splits, dim=0)
                # value_flat = torch.cat(value_splits, dim=0)

                query_flat = query_chunks.transpose(0, 2).contiguous()
                key_flat = key_chunks.transpose(0, 2).contiguous()
                value_flat = value_chunks.transpose(0, 2).contiguous()

                # Pre-allocate output buffers
                query_out = torch.empty_like(query_flat)
                key_out = torch.empty_like(key_flat)
                value_out = torch.empty_like(value_flat)

                dist.all_to_all_single(query_out, query_flat, group=ulysses_pg)
                dist.all_to_all_single(key_out, key_flat, group=ulysses_pg)
                dist.all_to_all_single(value_out, value_flat, group=ulysses_pg)

                # Reshape back
                # chunk_size = query_out.size(0) // ulysses_size
                # query_gathered = list(query_out.chunk(ulysses_size, dim=0))
                # key_gathered = list(key_out.chunk(ulysses_size, dim=0))
                # value_gathered = list(value_out.chunk(ulysses_size, dim=0))
                query = (
                    query_out.reshape(S_shard * ulysses_size, B, H // ulysses_size, D)
                    .transpose(0, 1)
                    .contiguous()
                    .reshape(B, H // ulysses_size, S_shard * ulysses_size, D)
                )
                key = (
                    key_out.reshape(S_shard * ulysses_size, B, H // ulysses_size, D)
                    .transpose(0, 1)
                    .contiguous()
                    .reshape(B, H // ulysses_size, S_shard * ulysses_size, D)
                )
                value = (
                    value_out.reshape(S_shard * ulysses_size, B, H // ulysses_size, D)
                    .transpose(0, 1)
                    .contiguous()
                    .reshape(B, H // ulysses_size, S_shard * ulysses_size, D)
                )
            else:
                # Option 2: Standard all_to_all but with memory optimization

                # Split tensors by Ulysses ranks (each rank will get a different chunk of heads)
                query_splits = list(
                    query_chunks.unbind(2)
                )  # List of [B, S/(UR), U, H/U, D]
                key_splits = list(key_chunks.unbind(2))
                value_splits = list(value_chunks.unbind(2))

                # Make tensors contiguous for all_to_all
                query_splits = [q.contiguous() for q in query_splits]
                key_splits = [k.contiguous() for k in key_splits]
                value_splits = [v.contiguous() for v in value_splits]

                # Pre-allocate output buffers
                query_gathered = [
                    torch.empty_like(query_splits[0]) for _ in range(ulysses_size)
                ]
                key_gathered = [
                    torch.empty_like(key_splits[0]) for _ in range(ulysses_size)
                ]
                value_gathered = [
                    torch.empty_like(value_splits[0]) for _ in range(ulysses_size)
                ]

                # Perform all-to-all
                dist.all_to_all(
                    query_gathered, query_splits, group=ulysses_pg, async_op=True
                )
                dist.all_to_all(
                    key_gathered, key_splits, group=ulysses_pg, async_op=True
                )
                dist.all_to_all(
                    value_gathered, value_splits, group=ulysses_pg, async_op=True
                )

                # Free the splits to reduce memory pressure
                del query_splits, key_splits, value_splits
                del query_chunks, key_chunks, value_chunks

                # Concatenate sequence shards from different Ulysses ranks
                # query_gathered[i] contains sequence shard i with head chunk for current rank
                query = torch.cat(query_gathered, dim=2)  # [B, H/U, S/R, D]
                key = torch.cat(key_gathered, dim=2)
                value = torch.cat(value_gathered, dim=2)

    # Now perform ring attention on the resharded tensors
    rank = dist.get_rank(ring_pg)
    size = dist.get_world_size(ring_pg)

    next_kv = None

    # Without making key and value contiguous(), the lose curve is bad.
    # TODO(fegin): figure out why this is a requirement since SDPA does not have
    # this requirement.
    key = key.contiguous()
    value = value.contiguous()

    sdpa_merger = _SDPAMerger(_cp_options.convert_to_f32, seq_dim=seq_dim)

    rest: list[Any]
    out: torch.Tensor
    logsumexp: torch.Tensor

    rotater = _create_rotater(ring_pg, 2)

    for i in range(size):
        if i > 0:
            # Wait for the kv from the (cp_rank - 1) rank.
            next_kv = rotater.next_buffer()
            key = next_kv[: key.numel()].reshape(key.shape)
            value = next_kv[key.numel() :].reshape(value.shape)

        if i < (size - 1):
            # Send the k, v to the next rank
            next_kv = torch.cat([key.flatten(), value.flatten()])
            next_kv = rotater.exchange_buffers(next_kv)

        is_causal_behavior = _is_causal_behavior(
            rank=rank, world_size=size, i=i, is_causal=is_causal
        )

        # For a detailed understanding of the load balancing algorithm, see
        # Note [Context parallelism load balance algorithm for causal masking]
        if is_causal_behavior == _CausalBehavior.SKIP:
            # If i > rank and load balancing is not turned on.
            continue

        if i == 0 or (not _cp_options.enable_load_balance or not is_causal):
            # When local balance is enabled, we still need to do SDPA with
            # the both local chunks of q, k, v for the first iteration.
            q, k, v, partial = (query, key, value, False)
        elif i <= rank:
            # Round-robin load balancing case, and i <= rank.
            # We need to do SPDA, with only the first local chunk of the k, v.
            # Note that q, k, v, each contains two local chunks.
            ROUND_ROBIN_CYCLE = 2
            q, k, v, partial = (
                query,
                key.chunk(ROUND_ROBIN_CYCLE, dim=2)[0],
                value.chunk(ROUND_ROBIN_CYCLE, dim=2)[0],
                False,
            )
        else:
            # Round-robin load balancing case, and i > rank.
            # We need to do SPDA with only the second half of the q, and update
            # only the the second part of  logsumexp. So partial is True.
            # Note that q, k, v, each contains two chunks.
            q, k, v, partial = query.chunk(2, dim=2)[1], key, value, True

        # See https://github.com/pytorch/pytorch/blob/release/2.4/aten/src/ATen/native/native_functions.yaml#L14695
        # for the SDPA kernel definitions.
        # debug_on_rank(0)  # Break only on rank 0
        out, logsumexp, *rest = op(
            q,
            k,
            v,
            is_causal=is_causal_behavior.value,
            **kwargs,
        )
        sdpa_merger.step(out, logsumexp, partial)

    # Get merged results
    out, logsumexp = sdpa_merger.results()
    # debug_on_rank(0)
    # If we performed Ulysses resharding, we need to undo it
    if ulysses_pg is not None:
        # Current shape: [B, H/U, S/R, D]
        # Target shape: [B, H, S/(U*R), D]

        B, H_shard, S_full, D = out.shape

        # Check if we should use chunked processing for reverse all-to-all
        use_chunked_ulysses = os.environ.get("TORCH_ULYSSES_CHUNKED", "0") == "1"
        use_yunchang_ulysses = os.environ.get("TORCH_ULYSSES_YUNCHANG", "0") == "1"
        chunk_size = int(os.environ.get("TORCH_ULYSSES_CHUNK_SIZE", "8"))

        if use_chunked_ulysses and H_shard >= chunk_size // ulysses_size:
            # Chunked reverse all-to-all
            out_chunks = []
            logsumexp_chunks = []

            chunk_per_rank = chunk_size // ulysses_size
            num_chunks = H_shard // chunk_per_rank

            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * chunk_per_rank
                end_idx = (chunk_idx + 1) * chunk_per_rank

                # Extract chunk
                out_chunk = out[:, start_idx:end_idx].contiguous()

                # Split sequence back for this chunk
                out_seq_chunks = list(out_chunk.chunk(ulysses_size, dim=2))
                out_seq_chunks = [chunk.contiguous() for chunk in out_seq_chunks]

                # Pre-allocate for all-to-all
                out_gathered = [
                    torch.empty_like(out_seq_chunks[0]) for _ in range(ulysses_size)
                ]
                dist.all_to_all(out_gathered, out_seq_chunks, group=ulysses_pg)

                # Free intermediates
                del out_seq_chunks

                # Stack the head chunks
                out_stacked = torch.stack(
                    out_gathered, dim=1
                )  # [B, U, chunk_per_rank, S/(U*R), D]
                out_chunk_final = out_stacked.reshape(
                    B, chunk_size, -1, D
                )  # [B, chunk_size, S/(U*R), D]

                out_chunks.append(out_chunk_final)
                del out_gathered, out_stacked

                # Handle logsumexp if needed
                if logsumexp.dim() == 3:
                    lse_chunk = logsumexp[:, start_idx:end_idx].contiguous()
                    lse_seq_chunks = list(lse_chunk.chunk(ulysses_size, dim=2))
                    lse_seq_chunks = [chunk.contiguous() for chunk in lse_seq_chunks]

                    lse_gathered = [
                        torch.empty_like(lse_seq_chunks[0]) for _ in range(ulysses_size)
                    ]
                    dist.all_to_all(lse_gathered, lse_seq_chunks, group=ulysses_pg)

                    del lse_seq_chunks
                    lse_stacked = torch.stack(lse_gathered, dim=1)
                    lse_chunk_final = lse_stacked.reshape(B, chunk_size, -1)

                    logsumexp_chunks.append(lse_chunk_final)
                    del lse_gathered, lse_stacked

            # Concatenate all chunks
            out = torch.cat(out_chunks, dim=1)
            if logsumexp.dim() == 3:
                logsumexp = torch.cat(logsumexp_chunks, dim=1)
        elif use_yunchang_ulysses:

            out = SeqAllToAll4D.apply(ulysses_pg, out, 2, 1, False)
            logsumexp = SeqAllToAll4D.apply(
                ulysses_pg, logsumexp.unsqueeze(-1), 2, 1, False
            )
        else:
            # Standard reverse all-to-all
            # First, we need to chunk the sequence dimension back to per-Ulysses-rank portions
            # Split the full sequence back into ulysses_size chunks
            out_chunks = list(
                out.chunk(ulysses_size, dim=2)
            )  # List of [B, H/U, S/(U*R), D]

            # Make chunks contiguous for all_to_all
            out_chunks = [chunk.contiguous() for chunk in out_chunks]

            # Perform all-to-all to reverse the resharding
            # Each rank sends out_chunks[i] to rank i and receives head chunks from all ranks
            out_gathered = [
                torch.empty_like(out_chunks[0]) for _ in range(ulysses_size)
            ]
            dist.all_to_all(out_gathered, out_chunks, group=ulysses_pg)

            # Stack the head chunks back together
            # out_gathered[i] contains head chunk i with sequence shard for current rank
            out_stacked = torch.stack(out_gathered, dim=1)  # [B, U, H/U, S/(U*R), D]
            out = out_stacked.reshape(
                B, H_shard * ulysses_size, -1, D
            )  # [B, H, S/(U*R), D]

            # Handle logsumexp similarly if it needs resharding
            # Note: logsumexp typically has shape [B, H/U, S/R] (no D dimension)
            # We need to check if logsumexp needs similar treatment
            if logsumexp.dim() == 3:  # [B, H/U, S/R]
                logsumexp_chunks = list(
                    logsumexp.chunk(ulysses_size, dim=2)
                )  # List of [B, H/U, S/(U*R)]
                # Make chunks contiguous for all_to_all
                logsumexp_chunks = [chunk.contiguous() for chunk in logsumexp_chunks]
                logsumexp_gathered = [
                    torch.empty_like(logsumexp_chunks[0]) for _ in range(ulysses_size)
                ]
                dist.all_to_all(logsumexp_gathered, logsumexp_chunks, group=ulysses_pg)
                logsumexp_stacked = torch.stack(
                    logsumexp_gathered, dim=1
                )  # [B, U, H/U, S/(U*R)]
                logsumexp = logsumexp_stacked.reshape(
                    B, -1, logsumexp_stacked.size(-1)
                )  # [B, H, S/(U*R)]

    return out, logsumexp, *rest


def _templated_ring_attention_ulysses_async(
    mesh: DeviceMesh,
    seq_dim: int,
    op: _AttentionOp,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = False,
    **kwargs: object,
) -> tuple[torch.Tensor, ...]:
    """
    This is a generalized ring attention implementation that can support multiple attention ops.

    Note [Context parallelism load balance algorithm for causal masking]
    =====================
    This explanation uses an example to illustrate the CP algorithm with causal
    masking.

    Consider a scenario where the sequence length of q, k, and v is 4 (e.g.,
    q = (q0, q1, q2, q3)), and there are two ranks. For simplicity, we will discuss
    only q and k, as v follows the same pattern as k.

    The diagram below represents a complete QK^T operation without parallelism.
    The `****` entries indicate that the result is not required due to causal
    masking (e.g., q0k1 is marked as `****`).

    +----+------------------------+
    |    |  k0    k1   k2     k3  |
    +----+------------------------+
    | q0 | q0k0, ****, ****, **** |
    | q1 | q1k0, q1k1, ****, **** |
    | q2 | q2k0, q2k1, q2k2, **** |
    | q3 | q3k0, q3k1, q3k2, q3k3 |
    +----+------------------------+

    ### No Load Balance:

    In this scenario, each rank owns a local chunk of q, k, and v, with each chunk
    containing two elements. Rank0 is responsible for managing (q0, q1) and (k0, k1),
    while rank1 manages (q2, q3) and (k2, k3).

    First Iteration: Both rank0 and rank1 perform SDPA with their local qkv pairs.
    Causal masking is enabled as some results are not required (e.g., q0k1).

    Second Iteration: Local queries remain the same, but local kv pairs are exchanged.
    Rank0 now has (q0, q1) and (k2, k3); rank1 has (q2, q3) and (k0, k1). Rank0 performs
    no computation, while rank1 computes locally without causal masking since all results
    (q2k0, q2k1, q3k0, q3k1) are needed.

    ### Round-robin Load Balance:

    In this setup, each rank owns two local chunks of q, k, and v, with each chunk
    containing one element. Rank0 manages (q0, q3) and (k0, k3); Rank1 manages (q1, q2)
    and (k1, k2). Although the local chunks are not consecutive, they are concatenated to
    enable SDPA to be performed in a single call for each step. Consequently, the chunk()
    function may be required to prepare the correct q, k, and v configurations.

    First Iteration: Both ranks perform SDPA with their local qkv pairs, similar to the
    no-load-balance case. This iteration corresponds to the `if` of the
    (`if, `elif`, `else`) in the implemementation.

    Second Iteration: Rank0 now has (q0, q3) and (k1, k2); rank1 has (q1, q2) and
    (k0, k3). For rank0, no computation is needed for q0. However, computations for
    q3k1 and q3k2 are required, so only q3 is used for SDPA. This corresponds to the
    `else` of the (`if`, `elif`, `else`) in the implemementation.
    For rank1, k0 is not needed for q1 and q2, so only k3 is used for SDPA. This
    corresponds to the `elif` of (`if`, `elif`, `else`) in the implementation.

    Parameters
    ----------
    op:
        The attention op to use
    *args:
        additional args are passed to the op
    **kwargs:
        additional kwargs are passed to the op

    Returns
    -------
    out:
        The merged attention output
    softmax_lse:
        The logsumexp of the merged attention output
    """

    if is_causal and (query.size(2) != key.size(2)):
        raise NotImplementedError(
            "is_causal requires the same query and context sequence lengths"
        )
    if not is_causal and _cp_options.enable_load_balance:
        raise RuntimeError("Load balancing requires `is_causal=True`.")

    if isinstance(mesh, dist.ProcessGroup):
        pg: Union[dist.ProcessGroup, list[dist.ProcessGroup]] = mesh
        ring_pg = pg
        ulysses_pg = None
    else:
        # Handle both 1D and 2D DeviceMesh
        pg_result = mesh.get_group()
        if isinstance(pg_result, tuple):
            # 2D mesh case - returns (ulysses_pg, ring_pg)
            ulysses_pg, ring_pg = pg_result
        else:
            # 1D mesh case - returns single ProcessGroup
            ring_pg = pg_result
            ulysses_pg = None
    assert isinstance(
        ring_pg, dist.ProcessGroup
    ), "Ring process group must be single dimension"

    # Handle Ulysses parallelism if applicable
    if ulysses_pg is not None:
        assert isinstance(
            ulysses_pg, dist.ProcessGroup
        ), "Ulysses process group must be single dimension"
        ulysses_rank = dist.get_rank(ulysses_pg)
        ulysses_size = dist.get_world_size(ulysses_pg)

        # Check if we should use memory-efficient chunked processing
        # This can be controlled by an environment variable
        use_chunked_ulysses = 0  # os.environ.get("TORCH_ULYSSES_CHUNKED", "0") == "1"
        use_yunchang_ulysses = 1  # os.environ.get("TORCH_ULYSSES_YUNCHANG", "0") == "1"
        use_async_a2a = 0  # os.environ.get("TORCH_ULYSSES_ASYNC_A2A", "0") == "1"
        chunk_size = 8  # int(os.environ.get("TORCH_ULYSSES_CHUNK_SIZE", "8"))

        if use_chunked_ulysses and query.size(1) >= chunk_size:
            # Ultra memory-efficient chunked processing
            logger.info(
                f"Using chunked Ulysses all-to-all with chunk_size={chunk_size}"
            )
            query, key, value = _ulysses_all_to_all_chunked(
                query, key, value, ulysses_pg, ulysses_size, chunk_size
            )
        elif use_yunchang_ulysses:
            # debug_on_rank(0)  # Break only on rank 0
            # Use Yunchang's all-to-all
            query = SeqAllToAll4D.apply(ulysses_pg, query, 1, 2, False)
            key = SeqAllToAll4D.apply(ulysses_pg, key, 1, 2, False)
            value = SeqAllToAll4D.apply(ulysses_pg, value, 1, 2, False)
        elif use_async_a2a:
            # expects bs, seq, hc, hs

            logger.info(f"Using Async All-to-All...\n")

            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)

            ulysses_degree = dist.get_world_size(ulysses_pg)
            a2a_stream = torch.cuda.Stream()

            bs, shard_seqlen, hc, hs = query.shape
            bs, shard_seqlen, hc_kv, hs = key.shape
            seq_len = shard_seqlen * ulysses_degree
            un = hc // ulysses_degree
            un_kv = hc_kv // ulysses_degree

            assert un_kv == un, f"un_kv {un_kv} un {un}"

            qkv = torch.cat([query, key, value]).contiguous()
            # (3*bs, seqlen/P, hc, hs) -> (hc, seqlen/P, 3*bs, hs) -> (un, ud, seqlen/P, 3*bs, hs), where hc = un*ud
            qkv_list = torch.unbind(
                qkv.transpose(0, 2)
                .contiguous()
                .reshape(un, ulysses_degree, shard_seqlen, 3 * bs, hs)
            )
            # 3xall-to-all output buffer
            qkv_trans_list = [
                torch.zeros(
                    ulysses_degree,
                    1,
                    shard_seqlen,
                    3 * bs,
                    hs,
                    dtype=query.dtype,
                    device=query.device,
                )
                for i in range(len(qkv_list))
            ]

            # last all-to-all buffter
            context_layer_list = [
                torch.zeros(
                    ulysses_degree,
                    1,
                    shard_seqlen,
                    bs,
                    hs,
                    dtype=query.dtype,
                    device=query.device,
                )
                for i in range(len(qkv_list))
            ]

            comm_handle_list = []

            # un * (ud, shard_seqlen, 3*bs, hs)
            for i, qkv in enumerate(qkv_list):
                with torch.cuda.stream(a2a_stream):
                    ret = dist.all_to_all_single(
                        qkv_trans_list[i],
                        qkv,
                        group=ulysses_pg,
                        async_op=True,
                    )
                comm_handle_list.append(ret)

            last_comm_handle_list = []
            for i_head, qkv_trans in enumerate(qkv_trans_list):
                if comm_handle_list[i_head] is not None:
                    comm_handle_list[i_head].wait()
                qkv_trans = (
                    qkv_trans.reshape(seq_len, 3 * bs, 1, hs)
                    .transpose(0, 1)
                    .contiguous()
                    .reshape(3 * bs, seq_len, 1, hs)
                )

                # qkv_trans = all_to_all_4D_async(qkv, qkv_trans_list[i], self.scatter_idx, self.gather_idx, self.ulysses_pg)
                qkv_trans = torch.chunk(qkv_trans, 3, dim=0)

                query_chunks = qkv_trans[0].transpose(1, 2)
                key_chunks = qkv_trans[1].transpose(1, 2)
                value_chunks = qkv_trans[2].transpose(1, 2)

                # Now perform ring attention on the resharded tensors
                out, logsumexp, *rest = _templated_ring_attention(
                    ring_pg,
                    seq_dim,
                    op,
                    query_chunks,
                    key_chunks,
                    value_chunks,
                    is_causal,
                    **kwargs,
                )

                context_layer = out.transpose(1, 2)
                # (bs, seq_len, head_cnt/N, head_size) -> (bs, seq_len/N, head_cnt, head_size)
                # scatter 1, gather 2

                context_layer = (
                    context_layer.reshape(bs, ulysses_degree, shard_seqlen, 1, hs)
                    .transpose(0, 3)
                    .transpose(0, 1)
                    .contiguous()
                    .reshape(ulysses_degree, 1, shard_seqlen, bs, hs)
                )
                with torch.cuda.stream(a2a_stream):
                    ret = dist.all_to_all_single(
                        context_layer_list[i_head],
                        context_layer,
                        group=ulysses_pg,
                        async_op=True,
                    )
                last_comm_handle_list.append(ret)

            # hc = un * P
            # un x (hc = P, seq_len/P, bs, hs) -> (bs, seq_len, hc = P, hs)
            for i_last, ret in enumerate(last_comm_handle_list):
                if ret is not None:
                    ret.wait()
                context_layer_list[i_last] = (
                    context_layer_list[i_last]
                    .reshape(ulysses_degree, shard_seqlen, bs, hs)
                    .transpose(0, 2)
                    .contiguous()
                    .reshape(bs, shard_seqlen, ulysses_degree, hs)
                )

            out = torch.cat(context_layer_list, dim=2).transpose(1, 2)
            # debug_on_rank(0)
            return out, logsumexp, *rest
        else:
            raise NotImplementedError(
                "Only Yunchang and Async All-to-All are supported"
            )
    else:
        return _templated_ring_attention(
            ring_pg, seq_dim, op, query, key, value, is_causal, **kwargs
        )

    out, logsumexp, *rest = _templated_ring_attention(
        ring_pg, seq_dim, op, query, key, value, is_causal, **kwargs
    )

    if ulysses_pg is not None:
        # Current shape: [B, H/U, S/R, D]
        # Target shape: [B, H, S/(U*R), D]

        B, H_shard, S_full, D = out.shape

        if use_chunked_ulysses and H_shard >= chunk_size // ulysses_size:
            # Chunked reverse all-to-all
            out_chunks = []
            logsumexp_chunks = []

            chunk_per_rank = chunk_size // ulysses_size
            num_chunks = H_shard // chunk_per_rank

            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * chunk_per_rank
                end_idx = (chunk_idx + 1) * chunk_per_rank

                # Extract chunk
                out_chunk = out[:, start_idx:end_idx].contiguous()

                # Split sequence back for this chunk
                out_seq_chunks = list(out_chunk.chunk(ulysses_size, dim=2))
                out_seq_chunks = [chunk.contiguous() for chunk in out_seq_chunks]

                # Pre-allocate for all-to-all
                out_gathered = [
                    torch.empty_like(out_seq_chunks[0]) for _ in range(ulysses_size)
                ]
                dist.all_to_all(out_gathered, out_seq_chunks, group=ulysses_pg)

                # Free intermediates
                del out_seq_chunks

                # Stack the head chunks
                out_stacked = torch.stack(
                    out_gathered, dim=1
                )  # [B, U, chunk_per_rank, S/(U*R), D]
                out_chunk_final = out_stacked.reshape(
                    B, chunk_size, -1, D
                )  # [B, chunk_size, S/(U*R), D]

                out_chunks.append(out_chunk_final)
                del out_gathered, out_stacked

                # Handle logsumexp if needed
                if logsumexp.dim() == 3:
                    lse_chunk = logsumexp[:, start_idx:end_idx].contiguous()
                    lse_seq_chunks = list(lse_chunk.chunk(ulysses_size, dim=2))
                    lse_seq_chunks = [chunk.contiguous() for chunk in lse_seq_chunks]

                    lse_gathered = [
                        torch.empty_like(lse_seq_chunks[0]) for _ in range(ulysses_size)
                    ]
                    dist.all_to_all(lse_gathered, lse_seq_chunks, group=ulysses_pg)

                    del lse_seq_chunks
                    lse_stacked = torch.stack(lse_gathered, dim=1)
                    lse_chunk_final = lse_stacked.reshape(B, chunk_size, -1)

                    logsumexp_chunks.append(lse_chunk_final)
                    del lse_gathered, lse_stacked

            # Concatenate all chunks
            out = torch.cat(out_chunks, dim=1)
            if logsumexp.dim() == 3:
                logsumexp = torch.cat(logsumexp_chunks, dim=1)
        elif use_yunchang_ulysses:

            out = SeqAllToAll4D.apply(ulysses_pg, out, 2, 1, False)
            logsumexp = SeqAllToAll4D.apply(
                ulysses_pg, logsumexp.unsqueeze(-1), 2, 1, False
            )
        else:
            # Standard reverse all-to-all
            # First, we need to chunk the sequence dimension back to per-Ulysses-rank portions
            # Split the full sequence back into ulysses_size chunks
            out_chunks = list(
                out.chunk(ulysses_size, dim=2)
            )  # List of [B, H/U, S/(U*R), D]

            # Make chunks contiguous for all_to_all
            out_chunks = [chunk.contiguous() for chunk in out_chunks]

            # Perform all-to-all to reverse the resharding
            # Each rank sends out_chunks[i] to rank i and receives head chunks from all ranks
            out_gathered = [
                torch.empty_like(out_chunks[0]) for _ in range(ulysses_size)
            ]
            dist.all_to_all(out_gathered, out_chunks, group=ulysses_pg)

            # Stack the head chunks back together
            # out_gathered[i] contains head chunk i with sequence shard for current rank
            out_stacked = torch.stack(out_gathered, dim=1)  # [B, U, H/U, S/(U*R), D]
            out = out_stacked.reshape(
                B, H_shard * ulysses_size, -1, D
            )  # [B, H, S/(U*R), D]

            # Handle logsumexp similarly if it needs resharding
            # Note: logsumexp typically has shape [B, H/U, S/R] (no D dimension)
            # We need to check if logsumexp needs similar treatment
            if logsumexp.dim() == 3:  # [B, H/U, S/R]
                logsumexp_chunks = list(
                    logsumexp.chunk(ulysses_size, dim=2)
                )  # List of [B, H/U, S/(U*R)]
                # Make chunks contiguous for all_to_all
                logsumexp_chunks = [chunk.contiguous() for chunk in logsumexp_chunks]
                logsumexp_gathered = [
                    torch.empty_like(logsumexp_chunks[0]) for _ in range(ulysses_size)
                ]
                dist.all_to_all(logsumexp_gathered, logsumexp_chunks, group=ulysses_pg)
                logsumexp_stacked = torch.stack(
                    logsumexp_gathered, dim=1
                )  # [B, U, H/U, S/(U*R)]
                logsumexp = logsumexp_stacked.reshape(
                    B, -1, logsumexp_stacked.size(-1)
                )  # [B, H, S/(U*R)]

    return out, logsumexp, *rest


def _sdpa_handler(
    op_call: torch._ops.OpOverload,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    # extract local tensor and sharding infos to a OpInfo
    op_info = DTensor._op_dispatcher.unwrap_to_op_info(op_call, args, kwargs)
    logger.debug("Dispatching op_call: %s", op_info.schema)

    # sharding propagation
    # TODO: remove the context parallel strategy from the default propagation
    # rule. Either figure out how to dynamically enable it or just don't call
    # propagate.
    DTensor._op_dispatcher.sharding_propagator.propagate(op_info)
    output_sharding = op_info.output_sharding
    assert output_sharding is not None, "output sharding should not be None"
    assert not output_sharding.needs_redistribute, "inputs need to be redistributed"

    if op_call == aten._scaled_dot_product_flash_attention.default:
        local_results = _scaled_dot_product_ring_flash_attention(
            op_info.compute_mesh,
            *op_info.local_args,  # type: ignore[arg-type]
            **op_info.local_kwargs,  # type: ignore[arg-type]
        )
    elif op_call == aten._scaled_dot_product_efficient_attention.default:
        local_results = _scaled_dot_product_ring_efficient_attention(
            op_info.compute_mesh,
            *op_info.local_args,  # type: ignore[arg-type]
            **op_info.local_kwargs,  # type: ignore[arg-type]
        )
    elif op_call == aten._scaled_dot_product_cudnn_attention.default:
        local_results = _scaled_dot_product_ring_cudnn_attention(
            op_info.compute_mesh,
            *op_info.local_args,  # type: ignore[arg-type]
            **op_info.local_kwargs,  # type: ignore[arg-type]
        )
    else:
        raise NotImplementedError(
            "CP only supports flash attention and memory efficient attention now."
        )

    return DTensor._op_dispatcher.wrap(local_results, output_sharding.output_spec)


def _sdpa_backward_handler(
    op_call: torch._ops.OpOverload,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    # Redistribute grad_output tensor to the same placement as output tensor
    args = list(args)
    args = tuple(args)

    # extract local tensor and sharding infos to a OpInfo
    op_info = DTensor._op_dispatcher.unwrap_to_op_info(op_call, args, kwargs)
    logger.debug("Dispatching op_call: %s", op_info.schema)

    # sharding propagation
    DTensor._op_dispatcher.sharding_propagator.propagate(op_info)
    output_sharding = op_info.output_sharding
    assert output_sharding is not None, "output sharding should not be None"
    assert not output_sharding.needs_redistribute, "inputs need to be redistributed"

    if op_call == aten._scaled_dot_product_flash_attention_backward.default:
        local_results = _scaled_dot_product_ring_flash_attention_backward(
            op_info.compute_mesh,
            *op_info.local_args,  # type: ignore[arg-type]
            **op_info.local_kwargs,  # type: ignore[arg-type]
        )
    elif op_call == aten._scaled_dot_product_efficient_attention_backward.default:
        local_results = _scaled_dot_product_ring_efficient_attention_backward(
            op_info.compute_mesh,
            *op_info.local_args,  # type: ignore[arg-type]
            **op_info.local_kwargs,  # type: ignore[arg-type]
        )
    elif op_call == aten._scaled_dot_product_cudnn_attention_backward.default:
        local_results = _scaled_dot_product_ring_cudnn_attention_backward(
            op_info.compute_mesh,
            *op_info.local_args,  # type: ignore[arg-type]
            **op_info.local_kwargs,  # type: ignore[arg-type]
        )
    else:
        raise NotImplementedError(f"{op_call=}")

    return DTensor._op_dispatcher.wrap(local_results, output_sharding.output_spec)


def _templated_ring_attention_backward(
    mesh: DeviceMesh,
    seq_dim: int,
    op: _AttentionOp,
    grad_out: torch.Tensor,
    grad_out_name: str,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    is_causal: bool,
    **kwargs: Any,
) -> tuple[torch.Tensor, ...]:
    """This API implements the backward of the ring attention."""
    if not is_causal and _cp_options.enable_load_balance:
        raise RuntimeError("Load balancing requires `is_causal=True`.")
    pg = mesh.get_group()
    # debug_on_rank(0)
    assert isinstance(pg, dist.ProcessGroup), "must be single dimension"
    rank = dist.get_rank(pg)
    # if rank == 0:
    #     print(f"rank={rank}, size={size} Ring Attention Backward\n\n\n\n\n!\n!!!!!!!!!\n!!!!!!!!!!")
    size = dist.get_world_size(pg)
    next_kv = None
    next_grad_kv = None
    rest: list[Any]
    grad_query_, grad_key_, grad_value_ = None, None, None

    accum_dtype = torch.float32 if _cp_options.convert_to_f32 else query.dtype
    grad_query = torch.zeros_like(query, dtype=accum_dtype)
    grad_key = torch.zeros_like(key, dtype=accum_dtype)
    grad_value = torch.zeros_like(value, dtype=accum_dtype)

    key = key.contiguous()
    value = value.contiguous()
    kv_rotater = _create_rotater(pg, 2)
    dkv_rotater = _create_rotater(pg, 2, method=_RotateMethod.ALL_TO_ALL)
    for i in range(size):
        if i > 0:
            # Wait for the kv from the (cp_rank - 1) rank.
            buffer = kv_rotater.next_buffer()
            pointer = 0
            key = buffer[pointer : pointer + key.numel()].reshape(key.shape)
            pointer += key.numel()
            value = buffer[pointer : pointer + value.numel()].reshape(value.shape)
            pointer += value.numel()

        if i != size - 1:
            # Send the kv to the next rank.
            next_kv = torch.cat([key.flatten(), value.flatten()])
            kv_rotater.exchange_buffers(next_kv)

        is_causal_behavior = _is_causal_behavior(
            rank=rank, world_size=size, i=i, is_causal=is_causal
        )

        if is_causal_behavior != _CausalBehavior.SKIP:
            if i == 0 or (not _cp_options.enable_load_balance or not is_causal):
                # We need to do SDPA with the full local q, k, v.
                q, k, v, out_, dout, lse = (query, key, value, out, grad_out, logsumexp)
            elif i <= rank:
                # Round-robin load balancing case, and i <= rank.
                # We need to do SPDA with only the first half of the k, v.
                # Note that q, k, v, each contains two chunks.
                q, k, v, out_, dout, lse = (
                    query,
                    key.chunk(2, dim=seq_dim)[0],
                    value.chunk(2, dim=seq_dim)[0],
                    out,
                    grad_out,
                    logsumexp,
                )
            else:
                # Round-robin load balancing case, and i > rank.
                # We need to do SPDA with only the second half of the q
                # Note that q, k, v, each contains two chunks.
                q, k, v, out_, dout, lse = (
                    query.chunk(2, dim=seq_dim)[1],
                    key,
                    value,
                    out.chunk(2, dim=seq_dim)[1],
                    grad_out.chunk(2, dim=seq_dim)[1],
                    # Need to make logsumexp contiguous, otherwise there will
                    # be numerical error.
                    logsumexp.chunk(2, dim=seq_dim)[1].contiguous(),
                )

            kwargs[grad_out_name] = dout
            # See https://github.com/pytorch/pytorch/blob/release/2.4/aten/src/ATen/native/native_functions.yaml#L14695
            # for the SDPA kernel definitions.
            # if q.shape[1] != k.shape[1]:
            #     bs, n_kv_heads, slen, head_dim = k.shape
            #     n_rep = q.shape[1] // k.shape[1]
            #     k = torch.unsqueeze(k, dim=2).expand(bs, n_kv_heads, n_rep, slen, head_dim).reshape(bs, n_kv_heads * n_rep, slen, head_dim)
            #     v = torch.unsqueeze(v, dim=2).expand(bs, n_kv_heads, n_rep, slen, head_dim).reshape(bs, n_kv_heads * n_rep, slen, head_dim)
            grad_query_, grad_key_, grad_value_, *rest = op(
                query=q,
                key=k,
                value=v,
                out=out_,
                logsumexp=lse,
                is_causal=is_causal_behavior.value,
                **kwargs,
            )
        else:
            grad_query_ = torch.zeros_like(query, dtype=accum_dtype)
            grad_key_ = torch.zeros_like(key, dtype=accum_dtype)
            grad_value_ = torch.zeros_like(value, dtype=accum_dtype)

        ROUND_ROBIN_CYCLE = 2
        if i == 0:
            grad_key += grad_key_
            grad_value += grad_value_
        else:
            pointer = 0
            # Wait for the kv gradient from (cp_rank - 1) rank.
            next_grad_kv = dkv_rotater.next_buffer()
            grad_key = next_grad_kv[pointer : pointer + grad_key.numel()].reshape(
                grad_key.shape
            )
            pointer += grad_key.numel()
            grad_value = next_grad_kv[pointer : pointer + grad_value.numel()].reshape(
                grad_value.shape
            )

            if i <= rank and _cp_options.enable_load_balance:
                grad_key = _partial_update(
                    grad_key,
                    grad_key_,
                    dim=seq_dim,
                    n_chunks=ROUND_ROBIN_CYCLE,
                    idx=0,
                    add=True,
                )
                grad_value = _partial_update(
                    grad_value,
                    grad_value_,
                    dim=seq_dim,
                    n_chunks=ROUND_ROBIN_CYCLE,
                    idx=0,
                    add=True,
                )
            else:
                grad_key += grad_key_
                grad_value += grad_value_

        next_grad_kv = torch.cat([grad_key.flatten(), grad_value.flatten()])
        # Send the grad key, and grad value to the next rank.
        dkv_rotater.exchange_buffers(next_grad_kv)

        if i <= rank or not _cp_options.enable_load_balance:
            grad_query += grad_query_
        else:
            grad_query = _partial_update(
                grad_query,
                grad_query_,
                dim=seq_dim,
                n_chunks=ROUND_ROBIN_CYCLE,
                idx=1,
                add=True,
            )

    assert grad_key_ is not None
    assert grad_value_ is not None
    grad_query = grad_query.to(query.dtype)
    next_grad_kv = dkv_rotater.next_buffer().to(key.dtype)
    grad_key = next_grad_kv[: grad_key.numel()].reshape(grad_key.shape)
    grad_value = next_grad_kv[grad_key.numel() :].reshape(grad_value.shape)
    return (
        grad_query,
        grad_key,
        grad_value,
        *rest,
    )


def _scaled_dot_product_ring_flash_attention_backward(
    mesh: DeviceMesh,
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    cum_seq_q: torch.Tensor,
    cum_seq_k: torch.Tensor,
    max_q: int,
    max_k: int,
    dropout_p: float,
    is_causal: bool,
    philox_seed: torch.Tensor,
    philox_offset: torch.Tensor,
    *,
    scale: Optional[float] = None,
) -> tuple[torch.Tensor, ...]:
    seq_dim = 2
    return _templated_ring_attention_backward(
        mesh,
        seq_dim,
        aten._scaled_dot_product_flash_attention_backward.default,
        grad_out=grad_out,
        grad_out_name="grad_out",
        query=query,
        key=key,
        value=value,
        out=out,
        logsumexp=logsumexp,
        is_causal=is_causal,
        cum_seq_q=cum_seq_q,
        cum_seq_k=cum_seq_k,
        max_q=max_q,
        max_k=max_k,
        dropout_p=dropout_p,
        philox_seed=philox_seed,
        philox_offset=philox_offset,
        scale=scale,
    )


def _scaled_dot_product_ring_efficient_attention_backward(
    mesh: DeviceMesh,
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    philox_seed: torch.Tensor,
    philox_offset: torch.Tensor,
    dropout_p: float,
    grad_input_mask: tuple[bool, ...],
    is_causal: bool = False,
    *,
    scale: Optional[float] = None,
) -> tuple[torch.Tensor, ...]:
    seq_dim = 2
    return _templated_ring_attention_backward(
        mesh,
        seq_dim,
        aten._scaled_dot_product_efficient_attention_backward.default,
        grad_out=grad_out,
        grad_out_name="grad_out_",
        query=query,
        key=key,
        value=value,
        attn_bias=bias,
        out=out,
        logsumexp=logsumexp,
        philox_seed=philox_seed,
        philox_offset=philox_offset,
        dropout_p=dropout_p,
        grad_input_mask=grad_input_mask,
        is_causal=is_causal,
        scale=scale,
    )


def _scaled_dot_product_ring_cudnn_attention_backward(
    mesh: DeviceMesh,
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    philox_seed: torch.Tensor,
    philox_offset: torch.Tensor,
    attn_bias: torch.Tensor,
    cum_seq_q: torch.Tensor,
    cum_seq_k: torch.Tensor,
    max_q: int,
    max_k: int,
    dropout_p: float,
    is_causal: bool,
    *,
    scale: Optional[float] = None,
) -> tuple[torch.Tensor, ...]:
    seq_dim = 2
    return _templated_ring_attention_backward(
        mesh,
        seq_dim,
        aten._scaled_dot_product_cudnn_attention_backward.default,
        grad_out=grad_out,
        grad_out_name="grad_out",
        query=query,
        key=key,
        value=value,
        out=out,
        logsumexp=logsumexp,
        philox_seed=philox_seed,
        philox_offset=philox_offset,
        attn_bias=attn_bias,
        cum_seq_q=cum_seq_q,
        cum_seq_k=cum_seq_k,
        max_q=max_q,
        max_k=max_k,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )


customized_ops = {
    aten._scaled_dot_product_flash_attention.default: _sdpa_handler,
    aten._scaled_dot_product_flash_attention_backward.default: _sdpa_backward_handler,
    aten._scaled_dot_product_efficient_attention.default: _sdpa_handler,
    aten._scaled_dot_product_efficient_attention_backward.default: _sdpa_backward_handler,
    aten._scaled_dot_product_cudnn_attention.default: _sdpa_handler,
    aten._scaled_dot_product_cudnn_attention_backward.default: _sdpa_backward_handler,
}


_replaced_functions: dict[Callable, tuple[str, Callable]] = {}


def _distribute_function(
    fn: Callable,
    fn_module: types.ModuleType,
    device_mesh: DeviceMesh,
    input_fn: Optional[Callable] = None,
    output_fn: Optional[Callable] = None,
) -> None:
    """
    ``distribute_function`` is an experimental API that allows users to "distribute"
    the inputs and outputs of a function. Similar to ``distribute_module``, this API
    installs hooks to the ``fn`` to convert the inputs and outputs. There are two
    major differences between ``distribute_function`` and ``distribute_module``.
    First, a function does not have parammeters and buffers, as a result,
    ``distribute_function`` itself won't convert any parameters/buffers but simply
    install the input and output hooks.  The tensor conversion will happen in the hooks.
    Another difference is an nn.Module subclass can have several instances and each
    instance be fed into ``distribute_module`` independently with affecting other
    instance. On the other hand, function is a singleton object. So if a function
    is distributed by ``distribute_function`` all subsequent calls to the function
    will invoke the installed hooks.

    Args:
        fn (Callable): the function to be distributed.
        fn_module (types.ModuleType): the Python module that the function is declared.
            e.g., if ``fn`` is ``torch.nn.functional.scaled_dot_product_attention``,
            ``fn_module`` is ``torch.nn.functional``.
        device_mesh (:class:`DeviceMesh`): the device mesh that will be used by the
            input and output hooks to distribute the tensors.
        input_fn (Optioinal[Callable]): the hook to distribute or convert the input
            arguments of ``fn``.
        output_fn (Optioinal[Callable]): the hook to distribute or convert the output
            arguments of ``fn``.
    """

    def wrapper(
        target_fn: Callable, input_fn: Optional[Callable], output_fn: Optional[Callable]
    ) -> Callable:
        def inner_fn(*args: tuple[Any, ...], **kwargs: dict[str, Any]) -> Any:
            if input_fn is not None:
                args, kwargs = input_fn(device_mesh, *args, **kwargs)
            output = target_fn(*args, **kwargs)
            if output_fn is not None:
                output = output_fn(device_mesh, output)
            return output

        return inner_fn

    global _replaced_functions

    if fn in _replaced_functions:
        return

    wrapper_fn = wrapper(fn, input_fn, output_fn)
    setattr(fn_module, fn.__name__, wrapper_fn)
    _replaced_functions[wrapper_fn] = (fn.__name__, fn)


def _restore_function(fn: Callable, fn_module: types.ModuleType) -> None:
    """Restore the function that is replaced by _distribute_function."""
    global _original_functions
    global _wrapper_functions

    if fn not in _replaced_functions:
        return

    original_name, original_fn = _replaced_functions[fn]
    setattr(fn_module, original_name, original_fn)


@contextlib.contextmanager
def _enable_cp_dispatcher() -> Generator[None, None, None]:
    """Enables DTensor dispatcher to dispatch SDPA to CP."""
    old_handlers = DTensor._op_dispatcher._custom_op_handlers
    DTensor._op_dispatcher._custom_op_handlers = {**old_handlers, **customized_ops}

    yield

    DTensor._op_dispatcher._custom_op_handlers = old_handlers


@contextlib.contextmanager
def _enable_cp_dispatcher_yunchang() -> Generator[None, None, None]:
    """Just a dummy dispatcher, since yunchang still needs parts of context parallel manager for shard distribution"""
    old_handlers = DTensor._op_dispatcher._custom_op_handlers
    DTensor._op_dispatcher._custom_op_handlers = old_handlers

    yield

    DTensor._op_dispatcher._custom_op_handlers = old_handlers


class _AttentionContextParallel(ParallelStyle):
    """
    Applies context parallel optimizations to the attention layer.

    This will work for nn.MultiHeadedAttention and custom attention layers that
    call F.scaled_dotproduct_attention with a simliar signature.

    This expects the `forward` method consumes either:

    * a single tensor for self attention
    * one argument for each of: query, key, value

    This currently only supports ring attention and the
    SDPBackend.FLASH_ATTENTION backend. See sdpa_kernel.

    Non-flash attention backends will result in incorrect results.
    """

    # use a weakref dictionary to store context managers for each nn.Module
    _CONTEXT_MANAGERS: "weakref.WeakKeyDictionary[nn.Module, Any]" = (
        weakref.WeakKeyDictionary()
    )

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        if not isinstance(device_mesh, DeviceMesh):
            raise ValueError(
                f"{type(device_mesh)} is not supported by {type(self)} yet."
            )

        if not device_mesh.ndim == 1:
            raise ValueError

        return distribute_module(
            module,
            device_mesh,
            input_fn=self._input_fn,  # type: ignore[arg-type]
            output_fn=self._output_fn,  # type: ignore[arg-type]
        )

    @classmethod
    def _input_fn(
        cls,
        module: nn.Module,
        inputs: tuple[Union[torch.Tensor, int, float], ...],
        device_mesh: DeviceMesh,
    ) -> tuple[Union[torch.Tensor, int, float], ...]:
        # TODO(d4l3k); this should be Shard(2), need to fix Linear layer rules
        placement = [Replicate()]

        def backward_hook(grad: torch.Tensor) -> None:
            if module in cls._CONTEXT_MANAGERS:
                cls._CONTEXT_MANAGERS[module].__exit__(None, None, None)
                del cls._CONTEXT_MANAGERS[module]

        # convert inputs to DTensor
        inp = []
        for input in inputs:
            if isinstance(input, torch.Tensor) and not isinstance(input, DTensor):
                input = DTensor.from_local(
                    input.contiguous(), device_mesh, placement, run_check=False
                )

            if isinstance(input, torch.Tensor) and input.requires_grad:
                input.register_hook(backward_hook)

            inp.append(input)

        manager = _enable_cp_dispatcher()
        manager.__enter__()
        cls._CONTEXT_MANAGERS[module] = manager

        return tuple(inp)

    @classmethod
    def _output_fn(
        cls,
        module: nn.Module,
        outputs: Union[torch.Tensor, tuple[Union[torch.Tensor, int, float], ...]],
        device_mesh: DeviceMesh,
    ) -> Union[
        Union[torch.Tensor, int, float], tuple[Union[torch.Tensor, int, float], ...]
    ]:
        cls._CONTEXT_MANAGERS[module].__exit__(None, None, None)
        del cls._CONTEXT_MANAGERS[module]

        def backward_hook(grad: torch.Tensor) -> None:
            if module not in cls._CONTEXT_MANAGERS:
                manager = _enable_cp_dispatcher()
                manager.__enter__()
                cls._CONTEXT_MANAGERS[module] = manager

        # back to local tensor
        out = []
        for output in [outputs] if isinstance(outputs, torch.Tensor) else outputs:
            output = output.to_local() if isinstance(output, DTensor) else output

            if isinstance(output, torch.Tensor) and output.requires_grad:
                output.register_hook(backward_hook)

            out.append(output)

        if isinstance(outputs, torch.Tensor):
            return out[0]

        return tuple(out)


@contextlib.contextmanager
def _context_parallel(
    seq_dim: int, mesh: DeviceMesh, impl: str = "yunchang"
) -> Generator[None, None, None]:
    """Replace SDPA with the CP-wrapped version and enable DTensor CP dispatcher."""

    def attention_input_fn(
        mesh: DeviceMesh, *args: tuple[Any, ...], **kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
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

    class DistributeFunction(TorchFunctionMode):
        def __init__(
            self,
            fn: Callable,
            device_mesh: DeviceMesh,
            input_fn: Optional[Callable] = None,
            output_fn: Optional[Callable] = None,
        ):
            self._device_mesh = device_mesh
            self._input_fn = input_fn
            self._output_fn = output_fn
            self._fn = fn

        def __torch_function__(
            self,
            func: Callable,
            types: Any,
            args: tuple[Any, ...] = (),
            kwargs: Optional[dict[str, Any]] = None,
        ) -> Any:
            kwargs = kwargs or {}

            if func != self._fn:
                return func(*args, **kwargs)

            if self._input_fn is not None:
                args, kwargs = self._input_fn(self._device_mesh, *args, **kwargs)
            output = func(*args, **kwargs)
            if self._output_fn is not None:
                output = self._output_fn(self._device_mesh, output)
            return output

    if _dispatch_mode == _DispatchMode.MONKEY_PATCH:
        _distribute_function(
            F.scaled_dot_product_attention,
            F,
            mesh,
            attention_input_fn,
            attention_output_fn,
        )
        cp_dispatcher = (
            _enable_cp_dispatcher_yunchang()
            if ("usp_" in impl or "upipe_" in impl)
            else _enable_cp_dispatcher()
        )
        with cp_dispatcher:
            yield
        _restore_function(F.scaled_dot_product_attention, F)
    elif _dispatch_mode == _DispatchMode.TORCH_FUNCTION:
        with DistributeFunction(
            F.scaled_dot_product_attention,
            mesh,
            attention_input_fn,
            attention_output_fn,
        ):
            cp_dispatcher = (
                _enable_cp_dispatcher_yunchang()
                if ("usp_" in impl or "upipe_" in impl)
                else _enable_cp_dispatcher()
            )
            with cp_dispatcher:
                yield
    else:
        raise NotImplementedError("torch dispatch mode is not supported yet.")


class _LoadBalancer(ABC):
    @classmethod
    @abstractmethod
    def shard(
        cls, buffer: torch.Tensor, mesh: DeviceMesh, seq_dim: int
    ) -> torch.Tensor:
        ...

    @classmethod
    @abstractmethod
    def unshard(
        cls, buffer: torch.Tensor, mesh: DeviceMesh, seq_dim: int
    ) -> torch.Tensor:
        ...


class _SequentialSharder(_LoadBalancer):
    """
    This load balancer chunks the buffer into cp_world_size and rank0 gets
    0th shard, rank1 gets 1st shard, ...
    So this doesn't have any load balancing effect when using the causal masking.
    """

    @classmethod
    def shard(
        cls, buffer: torch.Tensor, mesh: DeviceMesh, seq_dim: int
    ) -> torch.Tensor:
        assert buffer.size()[seq_dim] % mesh.size() == 0
        return buffer.chunk(mesh.size(), dim=seq_dim)[mesh.get_local_rank()]

    @classmethod
    def unshard(
        cls, buffer: torch.Tensor, mesh: DeviceMesh, seq_dim: int
    ) -> torch.Tensor:
        buffer = buffer.contiguous()
        all_buffers = [torch.empty_like(buffer) for _ in range(mesh.size())]
        ft_c.all_gather_inplace(all_buffers, buffer, mesh)
        return torch.cat(all_buffers, dim=seq_dim)


class _RoundRobinLoadBalancer(_LoadBalancer):
    """
    This load balancer chunk the buffer into cp_world_size * ROUND_ROBIN_CYCLE
    shards, and uses a round robin approach to achieve load balancing.
    Since ROUND_ROBIN_CYCLE being 2 will achieve perfect load balancing for
    causal masking, we assume ROUND_ROBIN_CYCLE is always 2 to simplify the
    implementation.
    """

    ROUND_ROBIN_CYCLE = 2

    @classmethod
    def shard(
        cls, buffer: torch.Tensor, mesh: DeviceMesh, seq_dim: int
    ) -> torch.Tensor:
        assert (
            cls.ROUND_ROBIN_CYCLE == 2
        ), "The current implementation only works if ROUND_ROBIN_CYCLE is 2."
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
        assert (
            cls.ROUND_ROBIN_CYCLE == 2
        ), "The current implementation only works if ROUND_ROBIN_CYCLE is 2."
        buffer = buffer.contiguous()
        cp_world_size = mesh.size()

        all_buffers = [torch.empty_like(buffer) for _ in range(cp_world_size)]
        ft_c.all_gather_inplace(all_buffers, buffer, mesh)
        sliced_buffers = [sb for b in all_buffers for sb in b.chunk(2, dim=seq_dim)]
        ordered_buffers = list(sliced_buffers)
        for i, b in enumerate(sliced_buffers):
            if i % 2 == 0:
                ordered_buffers[i // 2] = b
            else:
                ordered_buffers[cp_world_size * 2 - (i // 2) - 1] = b
        return torch.cat(ordered_buffers, dim=seq_dim)


def _context_parallel_buffers(
    mesh: DeviceMesh,
    buffers: list[torch.Tensor],
    buffer_seq_dims: list[int],
) -> list[torch.Tensor]:
    """Shard the buffers along the sequence dimensions according to CP rules."""
    # debug_on_rank(0)
    new_buffers = []
    sharder = (
        _RoundRobinLoadBalancer
        if _cp_options.enable_load_balance
        else _SequentialSharder
    )
    for buffer, seq_dim in zip(buffers, buffer_seq_dims):
        new_buffers.append(sharder.shard(buffer, mesh, seq_dim))

    return new_buffers


@contextlib.contextmanager
@torch.no_grad()
def context_parallel(
    mesh: DeviceMesh,
    *,
    buffers: Optional[list[torch.Tensor]] = None,
    buffer_seq_dims: Optional[list[int]] = None,
    no_restore_buffers: Optional[set[torch.Tensor]] = None,
    impl: Optional[str] = "yunchang",
) -> Generator[None, None, None]:
    """

    ``context_parallel`` is an experimental API to enable context
    parallelism (CP). This API performs two actions: 1) patch the SDPA
    (``torch.nn.functional.scaled_dot_product_attention``) with the CP-enabled
    one, 2) shard ``buffers`` along the sequence dimension and each rank will
    preserve the corresponding shard according ``mesh``.

    Args:
        mesh (:class:`DeviceMesh`): the device mesh for the context parallelism.
        buffers (Optional[List[torch.Tensor]]): buffers that the usage depend
            on the sequence dimension. Examples are input batch, labels and
            positional embedding buffers. These buffers must be sharded along
            the sequence dimension to ensure the accuracy. The sharding will
            happen in-place, the buffer's shape will change within the context.
            The buffers will be restored after the context finishes.
            ``no_restore_buffers`` can be used to specify which buffers don't
            need to be restored. Note that ``buffers`` should not contain any
            nn.Parameter.
        buffer_seq_dims (Optional[List[int]]): the sequence dimensions of ``buffers``.
        no_restore_buffers (Optional[Set[torch.Tensor]]): buffers in these set
            won't be restored after the context exits. This set must be a subset
            of ``buffers``. If the buffers won't be used after the context exits,
            these buffers can be put in this list to avoid extra restore time.

    .. warning::
        `torch.distributed.tensor.experimental.context_parallel` is a
        prototype feature in PyTorch. The API is subject to change.
    """
    buffers = [] if buffers is None else buffers
    buffer_seq_dims = [] if buffer_seq_dims is None else buffer_seq_dims
    no_restore_buffers = set() if no_restore_buffers is None else no_restore_buffers

    if len(buffers) != len(buffer_seq_dims):
        raise ValueError(
            "`seq_dims` must have the same number of elements as `buffers`."
        )

    for buffer in no_restore_buffers:
        # Cannot use `if not buffer in buffers` which will incur tensor comparison.
        if not any(b is buffer for b in buffers):
            raise ValueError("`no_restore_buffers` must be a subset of `buffers`.")

    original_buffers = [None if b in no_restore_buffers else b.clone() for b in buffers]
    chunks = _context_parallel_buffers(mesh, buffers, buffer_seq_dims)
    for buffer, chunk in zip(buffers, chunks):
        chunk = (
            chunk.clone()
        )  # TODO: potentially remove this clone - PyTorch PR - doesn't really matter
        buffer.resize_(chunk.shape)
        buffer.copy_(chunk)

    with _context_parallel(seq_dim=2, mesh=mesh, impl=impl):
        yield

    for buffer, original_buffer in zip(buffers, original_buffers):
        if original_buffer is not None:
            buffer.resize_(original_buffer.shape)
            buffer.copy_(original_buffer)


@torch.no_grad()
def context_parallel_unshard(
    mesh: DeviceMesh,
    buffers: list[torch.Tensor],
    seq_dims: list[int],
) -> list[torch.Tensor]:
    """
    Unshard the tensors (e.g., output) that are sharded due to context parallelism.

    Args:
        mesh (:class:`DeviceMesh`): the device mesh for the context parallelism.
        buffers (List[torch.Tensor]): the buffers to be unsharded.
        seq_dims (List[int]): the sequence dimensions of ``buffers``. This list
            must have the same length as ``buffers``.

    Returns:
        List[torch.Tensor]: the unsharded buffers.
    """
    sharder = (
        _RoundRobinLoadBalancer
        if _cp_options.enable_load_balance
        else _SequentialSharder
    )
    return [sharder.unshard(b, mesh, dim) for b, dim in zip(buffers, seq_dims)]


def set_rotate_method(rotate_method: str) -> None:
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


def set_load_balance(load_balance: str) -> None:
    """
    Context Parallel SDPA requires the load balance of kv shards. Users can call this
    API to specify which load balance method to use. "round_robin" uses a round robin
    approach to achieve load balancing. "sequential" uses a sequential approach to
    achieve load balancing. If this API has not been called, the default load balance
    method is "round_robin".
    """
    _cp_options.enable_load_balance = "basic" not in load_balance
