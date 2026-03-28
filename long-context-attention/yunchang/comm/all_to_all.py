# Copyright (c) Microsoft Corporation and Jiarui Fang
# SPDX-License-Identifier: Apache-2.0


from typing import Any, Tuple

import torch

import torch.distributed as dist
from torch import Tensor
from torch.nn import Module

from yunchang.globals import (  # , symm_streams
    channel_dict,
    hdl,
    local_inp_buf,
    PROCESS_GROUP,
    set_o_handle,
    set_u_handle,
)

# global group


@torch.library.custom_op(
    "yunchang::_all_to_all_4D", mutates_args=(), device_types="cuda"
)
def all_to_all_4D(
    input: torch.Tensor,
    scatter_idx: int = 2,
    gather_idx: int = 1,
    use_sync: bool = False,
    async_op: bool = False,
    output: torch.Tensor = None,
) -> torch.Tensor:
    """
    all-to-all for QKV

    Args:
        input (torch.tensor): a tensor sharded along dim scatter dim
        scatter_idx (int): default 1
        gather_idx (int): default 2
        group : torch process group
        use_sync (bool): whether to synchronize after all-to-all

    Returns:
        torch.tensor: resharded tensor (bs, seqlen/P, hc, hs)
    """
    # global group
    group = PROCESS_GROUP.ULYSSES_PG

    assert (
        input.dim() == 4
    ), f"input must be 4D tensor, got {input.dim()} and shape {input.shape}"

    seq_world_size = dist.get_world_size(group)

    if scatter_idx == 2 and gather_idx == 1:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen/P, hc, hs) output: (bs, seqlen, hc/P, hs)
        bs, shard_seqlen, hc, hs = input.shape
        seqlen = shard_seqlen * seq_world_size
        shard_hc = hc // seq_world_size

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen/P, hc, hs) -reshape-> (bs, seq_len/P, P, hc/P, hs) -transpose(0,2)-> (P, seq_len/P, bs, hc/P, hs)
        input_t = (
            input.reshape(bs, shard_seqlen, seq_world_size, shard_hc, hs)
            .transpose(0, 2)
            .contiguous()
        )

        if output is None:
            output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, seq_len/P, bs, hc/P, hs) scatter seqlen -all2all-> (P, seq_len/P, bs, hc/P, hs) scatter head

        if seq_world_size > 1:
            if async_op:
                # output = torch.zeros_like(input_t)
                u_handle = dist.all_to_all_single(
                    output, input_t, group=group, async_op=async_op
                )
                set_u_handle(u_handle)
            else:
                dist.all_to_all_single(output, input_t, group=group)
            if use_sync:
                torch.cuda.synchronize()
        else:
            output = input_t

        if not async_op:
            # if scattering the seq-dim, transpose the heads back to the original dimension
            output = output.reshape(seqlen, bs, shard_hc, hs)

            # (seq_len, bs, hc/P, hs) -reshape-> (bs, seq_len, hc/P, hs)
            output = (
                output.transpose(0, 1).contiguous().reshape(bs, seqlen, shard_hc, hs)
            )

        return output

    elif scatter_idx == 1 and gather_idx == 2:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen, hc/P, hs) output: (bs, seqlen/P, hc, hs)
        bs, seqlen, shard_hc, hs = input.shape
        hc = shard_hc * seq_world_size
        shard_seqlen = seqlen // seq_world_size
        seq_world_size = dist.get_world_size(group)

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen, hc/P, hs) -reshape-> (bs, P, seq_len/P, hc/P, hs) -transpose(0, 3)-> (hc/P, P, seqlen/P, bs, hs) -transpose(0, 1) -> (P, hc/P, seqlen/P, bs, hs)
        input_t = (
            input.reshape(bs, seq_world_size, shard_seqlen, shard_hc, hs)
            .transpose(0, 3)
            .transpose(0, 1)
            .contiguous()
            .reshape(seq_world_size, shard_hc, shard_seqlen, bs, hs)
        )

        if output is None:
            output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, bs x hc/P, seqlen/P, hs) scatter seqlen -all2all-> (P, bs x seq_len/P, hc/P, hs) scatter head
        if seq_world_size > 1:
            if async_op:
                # output = torch.zeros_like(input_t)
                u_handle = dist.all_to_all_single(
                    output, input_t, group=group, async_op=async_op
                )
                set_o_handle(u_handle)
            else:
                dist.all_to_all_single(output, input_t, group=group)
            if use_sync:
                torch.cuda.synchronize()
        else:
            output = input_t

        if not async_op:
            # if scattering the seq-dim, transpose the heads back to the original dimension
            output = output.reshape(hc, shard_seqlen, bs, hs)

            # (hc, seqlen/N, bs, hs) -tranpose(0,2)-> (bs, seqlen/N, hc, hs)
            output = (
                output.transpose(0, 2).contiguous().reshape(bs, shard_seqlen, hc, hs)
            )

        return output

    else:
        raise RuntimeError("scatter_idx must be 1 or 2 and gather_idx must be 1 or 2")


@all_to_all_4D.register_fake
def _(
    input: torch.Tensor,
    scatter_idx: int = 2,
    gather_idx: int = 1,
    use_sync: bool = False,
    async_op: bool = False,
    output: torch.Tensor = None,
) -> torch.Tensor:
    # global group
    group = PROCESS_GROUP.ULYSSES_PG
    if scatter_idx == 2 and gather_idx == 1:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen/P, hc, hs) output: (bs, seqlen, hc/P, hs)
        seq_world_size = dist.get_world_size(group)
        bs, shard_seqlen, hc, hs = input.shape
        seqlen = shard_seqlen * seq_world_size
        shard_hc = hc // seq_world_size
        return torch.empty([bs, seqlen, shard_hc, hs], device=input.device)
    elif scatter_idx == 1 and gather_idx == 2:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen, hc/P, hs) output: (bs, seqlen/P, hc, hs)
        seq_world_size = dist.get_world_size(group)
        bs, seqlen, shard_hc, hs = input.shape
        hc = shard_hc * seq_world_size
        shard_seqlen = seqlen // seq_world_size
        return torch.empty([bs, shard_seqlen, hc, hs], device=input.device)
    else:
        raise RuntimeError("scatter_idx must be 1 or 2 and gather_idx must be 1 or 2")


class SeqAllToAll4D(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        ulysses_group: dist.ProcessGroup,
        input: Tensor,
        scatter_idx: int,
        gather_idx: int,
        use_sync: bool = False,
        async_op: bool = False,
    ) -> Tensor:

        # global group
        # group = ulysses_group
        ctx.group = ulysses_group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        ctx.use_sync = use_sync
        ctx.async_op = async_op
        if dist.get_world_size(ulysses_group) > 1:
            output = all_to_all_4D(
                input, scatter_idx, gather_idx, use_sync=use_sync, async_op=async_op
            )
            return output
        else:
            return input

    @staticmethod
    def backward(
        ctx: Any, *grad_output: Tensor
    ) -> Tuple[None, Tensor, None, None, None, None]:

        if dist.get_world_size(ctx.group) == 1:
            return None, grad_output[0], None, None, None, None

        assert (
            len(grad_output) == 1
        ), "SeqAllToAll4D backward expects exactly one gradient output"
        if grad_output[0].dim() == 5:
            grad_output = grad_output[0]
            if ctx.scatter_idx == 2:
                seq_world_size, shard_seqlen, bs, shard_hc, hs = grad_output.shape
                seqlen = shard_seqlen * seq_world_size
                grad_output = (
                    grad_output.reshape(seqlen, bs, shard_hc, hs)
                    .transpose(0, 1)
                    .contiguous()
                    .reshape(bs, seqlen, shard_hc, hs)
                )
            if ctx.scatter_idx == 1:
                seq_world_size, shard_hc, shard_seqlen, bs, hs = grad_output.shape
                hc = shard_hc * seq_world_size
                grad_output = (
                    grad_output.reshape(hc, shard_seqlen, bs, hs)
                    .transpose(0, 2)
                    .contiguous()
                    .reshape(bs, shard_seqlen, hc, hs)
                )
            # else:
            #     raise RuntimeError("scatter_idx must be 1 or 2")
            grad_output = (grad_output,)

        return (
            None,
            SeqAllToAll4D.apply(
                ctx.group,
                *grad_output,
                ctx.gather_idx,
                ctx.scatter_idx,
                ctx.use_sync,
                False,
            ),
            None,
            None,
            None,
            None,
        )


def all_to_all_5D(
    input: torch.tensor,
    scatter_idx: int = 3,
    gather_idx: int = 1,
    group=None,
    use_sync: bool = False,
) -> torch.tensor:
    """
    all-to-all for QKV
    forward (bs, seqlen/N, 3, hc, hs) -> (bs, seqlen, 3, hc/N, hs)

    Args:
        input (torch.tensor): a tensor sharded along dim scatter dim
        scatter_idx (int): default 1
        gather_idx (int): default 2
        group : torch process group
        use_sync: whether to synchronize after all-to-all

    Returns:
        torch.tensor: resharded tensor (bs, seqlen/P, 3, hc, hs)
    """
    assert (
        input.dim() == 5
    ), f"input must be 5D tensor, got {input.dim()} and shape {input.shape}"

    seq_world_size = dist.get_world_size(group)

    if scatter_idx == 3 and gather_idx == 1:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen/P, 3, hc, hs) output: (bs, seqlen, 3, hc/P, hs)
        bs, shard_seqlen, t_cnt, hc, hs = input.shape

        assert t_cnt == 3
        seqlen = shard_seqlen * seq_world_size
        shard_hc = hc // seq_world_size

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen/P, 3, hc, hs) -reshape-> (bs, seq_len/P, 3, P, hc/P, hs) -transpose(0,3)-> (P, seq_len/P, 3, bs, hc/P, hs)
        input_t = (
            input.reshape(bs, shard_seqlen, 3, seq_world_size, shard_hc, hs)
            .transpose(0, 3)
            .contiguous()
        )

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, seq_len/P, 3, bs, hc/P, hs) scatter seqlen -all2all-> (P, seq_len/P, 3, bs, hc/P, hs) scatter head
        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
            if use_sync:
                torch.cuda.synchronize()
        else:
            output = input_t

        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(seqlen, 3, bs, shard_hc, hs)

        # (seq_len, 3, bs, hc/P, hs) -trans-> (bs, seq_len, 3, hc/P, hs)
        output = output.transpose(0, 2).transpose(1, 2).contiguous()

        return output.reshape(bs, seqlen, 3, shard_hc, hs).contiguous()
    elif scatter_idx == 1 and gather_idx == 3:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen, hc/P, hs) output: (bs, seqlen/P, hc, hs)
        bs, seqlen, _, shard_hc, hs = input.shape
        hc = shard_hc * seq_world_size
        shard_seqlen = seqlen // seq_world_size
        seq_world_size = dist.get_world_size(group)

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen, 3, hc/P, hs) -reshape-> (bs, P, seq_len/P, 3, hc/P, hs) -transpose(0, 4)-> (hc/P, P, seqlen/P, 3, bs, hs) -transpose(0, 1) -> (P, hc/P, seqlen/P, 3, bs, hs)
        input_t = (
            input.reshape(bs, seq_world_size, shard_seqlen, 3, shard_hc, hs)
            .transpose(0, 4)
            .transpose(0, 1)
            .contiguous()
            .reshape(seq_world_size, shard_hc, shard_seqlen, 3, bs, hs)
        )

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, bs x hc/P, seqlen/P, hs) scatter seqlen -all2all-> (P, bs x seq_len/P, hc/P, hs) scatter head
        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
            if use_sync:
                torch.cuda.synchronize()
        else:
            output = input_t

        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(hc, shard_seqlen, 3, bs, hs)

        # (hc, seqlen/N, bs, hs) -tranpose(0,2)-> (bs, seqlen/N, hc, hs)
        output = output.transpose(0, 3).contiguous()

        return output.reshape(bs, shard_seqlen, 3, hc, hs).contiguous()
    else:
        raise RuntimeError("scatter_idx must be 1 or 3 and gather_idx must be 1 or 3")


class SeqAllToAll5D(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        input: Tensor,
        scatter_idx: int = 3,
        gather_idx: int = 1,
        use_sync: bool = False,
    ) -> Tensor:

        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        ctx.use_sync = use_sync

        return all_to_all_5D(
            input, scatter_idx, gather_idx, group=group, use_sync=use_sync
        )

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor, None, None]:
        return (
            None,
            SeqAllToAll5D.apply(
                ctx.group, *grad_output, ctx.gather_idx, ctx.scatter_idx, ctx.use_sync
            ),
            None,
            None,
            None,
        )


@torch.no_grad()
def vanilla_all_to_all_4D(
    input: torch.Tensor,
    scatter_idx: int = 2,
    gather_idx: int = 1,
    use_sync: bool = False,
    async_op: bool = False,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    all-to-all for QKV

    Args:
        input (torch.tensor): a tensor sharded along dim scatter dim
        scatter_idx (int): default 1
        gather_idx (int): default 2
        group : torch process group
        use_sync (bool): whether to synchronize after all-to-all

    Returns:
        torch.tensor: resharded tensor (bs, seqlen/P, hc, hs)
    """
    # global group
    group = PROCESS_GROUP.ULYSSES_PG

    assert (
        input.dim() == 4
    ), f"input must be 4D tensor, got {input.dim()} and shape {input.shape}"

    seq_world_size = dist.get_world_size(group)

    if scatter_idx == 2 and gather_idx == 1:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen/P, hc, hs) output: (bs, seqlen, hc/P, hs)
        bs, shard_seqlen, hc, hs = input.shape
        seqlen = shard_seqlen * seq_world_size
        shard_hc = hc // seq_world_size

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen/P, hc, hs) -reshape-> (bs, seq_len/P, P, hc/P, hs) -transpose(0,2)-> (P, seq_len/P, bs, hc/P, hs)

        # assert input.requires_grad, f"input requires_grad must be True"
        input_t = (
            input.reshape(bs, shard_seqlen, seq_world_size, shard_hc, hs)
            .transpose(0, 2)
            .contiguous()
        )
        # input_t = (
        #     input.reshape(bs, shard_seqlen, seq_world_size, shard_hc, hs)
        #     .transpose(0, 2)
        #     .clone(memory_format=torch.contiguous_format)
        # )

        if output is None:
            output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, seq_len/P, bs, hc/P, hs) scatter seqlen -all2all-> (P, seq_len/P, bs, hc/P, hs) scatter head

        if seq_world_size > 1:
            if async_op:
                # output = torch.zeros_like(input_t)
                u_handle = dist.all_to_all_single(
                    output, input_t, group=group, async_op=async_op
                )
                set_u_handle(u_handle)
            else:
                dist.all_to_all_single(output, input_t, group=group)
                # if input_t.requires_grad:
                #     assert output.requires_grad, f"output requires_grad must be True"
            if use_sync:
                torch.cuda.synchronize()
        else:
            output = input_t

        if not async_op:
            # if scattering the seq-dim, transpose the heads back to the original dimension
            output = output.reshape(seqlen, bs, shard_hc, hs)

            # (seq_len, bs, hc/P, hs) -reshape-> (bs, seq_len, hc/P, hs)
            output = output.transpose(
                0, 1
            )  # .contiguous().reshape(bs, seqlen, shard_hc, hs)

        return output

    elif scatter_idx == 1 and gather_idx == 2:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen, hc/P, hs) output: (bs, seqlen/P, hc, hs)
        bs, seqlen, shard_hc, hs = input.shape
        hc = shard_hc * seq_world_size
        shard_seqlen = seqlen // seq_world_size
        seq_world_size = dist.get_world_size(group)

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen, hc/P, hs) -reshape-> (bs, P, seq_len/P, hc/P, hs) -transpose(0, 3)-> (hc/P, P, seqlen/P, bs, hs) -transpose(0, 1) -> (P, hc/P, seqlen/P, bs, hs)
        input_t = (
            input.reshape(bs, seq_world_size, shard_seqlen, shard_hc, hs)
            .transpose(0, 3)
            .transpose(0, 1)
            .contiguous()
            .reshape(seq_world_size, shard_hc, shard_seqlen, bs, hs)
        )

        if output is None:
            output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, bs x hc/P, seqlen/P, hs) scatter seqlen -all2all-> (P, bs x seq_len/P, hc/P, hs) scatter head
        if seq_world_size > 1:
            if async_op:
                # output = torch.zeros_like(input_t)
                u_handle = dist.all_to_all_single(
                    output, input_t, group=group, async_op=async_op
                )
                set_o_handle(u_handle)
            else:
                dist.all_to_all_single(output, input_t, group=group)
            if use_sync:
                torch.cuda.synchronize()
        else:
            output = input_t

        if not async_op:
            # if scattering the seq-dim, transpose the heads back to the original dimension
            output = output.reshape(hc, shard_seqlen, bs, hs)

            # (hc, seqlen/N, bs, hs) -tranpose(0,2)-> (bs, seqlen/N, hc, hs)
            output = (
                output.transpose(0, 2).contiguous().reshape(bs, shard_seqlen, hc, hs)
            )  # for 1 2, we need contiguous dq/dk/dv for viewing as (bs*seqlen, -1)

        return output

    else:
        raise RuntimeError("scatter_idx must be 1 or 2 and gather_idx must be 1 or 2")


from typing import Optional

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem


def symm_all_to_all_equal(
    out: Optional[torch.Tensor],
    input_tensor: torch.Tensor,
    group=dist.group.WORLD,
    split_dim: int = 0,
    channel: int = 0,
) -> torch.Tensor:
    """
    Symmetric-memory based all_to_all for equal splits along `split_dim`.
    """

    assert input_tensor.is_cuda, "symmetric memory currently expects CUDA tensors"
    world_size = dist.get_world_size(group)
    if world_size == 1:
        if out is None:
            return input_tensor.clone()
        else:
            return out.copy_(input_tensor)

    assert split_dim == 0, "split_dim must be 0 for this helper"

    dim_size = input_tensor.size(split_dim)
    assert (
        dim_size % world_size == 0
    ), "size along split_dim must be divisible by world_size"
    per_rank = dim_size // world_size

    # chunks and shapes
    inp_chunks = input_tensor.chunk(world_size, dim=split_dim)
    assert len(inp_chunks) == world_size
    chunk_shape = list(input_tensor.shape)
    chunk_shape[split_dim] = per_rank
    chunk_numel = inp_chunks[0].numel()

    # allocate out if needed
    if out is None:
        out = torch.empty_like(input_tensor)

    out_bufs = list(out.chunk(world_size, dim=split_dim))
    # use a cache keyed by shape/dtype/device (make sure these globals exist)
    global local_inp_buf, hdl, channel_dict
    key = (tuple(input_tensor.shape), str(input_tensor.dtype), str(input_tensor.device))

    my_rank = dist.get_rank(group)

    if key not in local_inp_buf:
        local_inp_buf[key] = symm_mem.empty(
            input_tensor.shape, device=input_tensor.device, dtype=input_tensor.dtype
        )
        hdl[key] = symm_mem.rendezvous(local_inp_buf[key], group)

    # write my local tensor into symmetric buffer
    local_inp_buf[key].copy_(input_tensor)

    # get current stream id
    current_stream_id = torch.cuda.current_stream().cuda_stream

    # use channel corresponding to current stream id
    if current_stream_id not in channel_dict:
        channel_dict[current_stream_id] = len(channel_dict)
    channel = channel_dict[current_stream_id]

    # barrier to ensure all writers have written
    hdl[key].barrier(channel=channel)

    # read all chunks from each rank's buffer
    # For rank src, fetch buffer slice corresponding to chunk index i (typically i)
    # Storage offset is chunk_numel * i (i.e. chunk i within that remote buffer)
    for src in range(world_size):
        # get_buffer(src_rank, shape, dtype, storage_offset=...)
        remote_slice = hdl[key].get_buffer(
            src,
            tuple(chunk_shape),
            input_tensor.dtype,
            storage_offset=chunk_numel * my_rank,
        )
        # copy into our preallocated out buffer chunk for src
        out_bufs[src].copy_(remote_slice)  # , non_blocking=True)

    # ensure copies finished (simple approach: barrier)
    hdl[key].barrier(channel=channel)

    # out is already written in-place; return it
    return out


# def symm_all_to_all_equal(
#     out: torch.Tensor,
#     input_tensor: torch.Tensor,
#     group = dist.group.WORLD,
#     split_dim: int = 0,
#     channel: int = 0,
# ) -> torch.Tensor:
#     """
#     Symmetric-memory based all_to_all for equal splits along `split_dim`.

#     Semantics:
#       - Each rank splits `input_tensor` equally into `world_size` chunks along `split_dim`.
#       - Rank r writes its chunk destined for dst into dst's symmetric buffer at index [r].
#       - After a barrier, each rank concatenates local buffer slices [0..world_size-1] along `split_dim`.

#     Args:
#       input_tensor: tensor to exchange; size along `split_dim` must be divisible by world_size
#       split_dim: dimension to split/concat
#       group: torch distributed process group
#       channel: symmetric-memory barrier channel to use
#       out: optional preallocated output tensor with same shape as `input_tensor`

#     Returns:
#       Tensor with the same shape as `input_tensor` after the all_to_all permutation.
#     """
#     # return input_tensor
#     # torch.cuda.synchronize()
#     global local_inp_buf, hdl#, symm_streams

#     assert input_tensor.is_cuda, "symmetric memory currently expects CUDA tensors"
#     world_size = dist.get_world_size(group)
#     if world_size == 1:
#         return input_tensor if out is None else out.copy_(input_tensor)

#     dim_size = input_tensor.size(split_dim)
#     assert dim_size % world_size == 0, "size along split_dim must be divisible by world_size"
#     per_rank = dim_size // world_size

#     assert split_dim == 0, "split_dim must be 0"
#     inp_chunks = input_tensor.chunk(world_size, dim=split_dim)
#     assert len(inp_chunks) == world_size, f"expected {world_size} chunks, got {len(inp_chunks)}"

#     # Shape of one received chunk
#     chunk_shape = list(input_tensor.shape)
#     chunk_shape[split_dim] = per_rank

#     out_bufs = list(torch.chunk(out, world_size, dim=split_dim))

#     my_rank = dist.get_rank(group)

#     if input_tensor.shape not in local_inp_buf:
#         local_inp_buf[input_tensor.shape] = symm_mem.empty(input_tensor.shape, device=input_tensor.device, dtype=input_tensor.dtype)
#         hdl[input_tensor.shape] = symm_mem.rendezvous(local_inp_buf[input_tensor.shape], group)
#     # if input_tensor.shape not in symm_streams:
#     #     symm_streams[input_tensor.shape] = [torch.cuda.Stream() for _ in range(world_size)]
#     # else:
#     #     assert len(symm_streams[input_tensor.shape]) == world_size, f"expected {world_size} streams, got {len(symm_streams[input_tensor.shape])}"

#     local_inp_buf[input_tensor.shape].copy_(input_tensor)
#     hdl[input_tensor.shape].barrier(channel=channel)

#     remote_buf = [None] * world_size

#     # for i in range(world_size):
#     #     symm_streams[input_tensor.shape][i].wait_stream(torch.cuda.current_stream())

#     for i in range(world_size):
#         # with torch.cuda.stream(symm_streams[input_tensor.shape][i]):
#         # if False and i == my_rank: # do not copy the local buffer
#         #     out[i*per_rank:(i+1)*per_rank] = inp_chunks[i].clone()
#         # else:
#         remote_buf[i] = hdl[input_tensor.shape].get_buffer(i, tuple(chunk_shape), input_tensor.dtype, storage_offset=inp_chunks[0].numel()*my_rank)
#         out_bufs[i].copy_(remote_buf[i], non_blocking=True)
#         # out[i*per_rank:(i+1)*per_rank].copy_(remote_buf[i])#, non_blocking=True)
#         # hdl[input_tensor.shape].barrier(channel=i)
#             # out_bufs[i] = hdl[input_tensor.shape].get_buffer(i, tuple(chunk_shape), input_tensor.dtype, storage_offset=inp_chunks[0].numel()*my_rank)

#     # for i in range(world_size):
#     #     torch.cuda.current_stream().wait_stream(symm_streams[input_tensor.shape][i])

#         hdl[input_tensor.shape].barrier(channel=channel)


#     out = torch.cat(out_bufs, dim=split_dim)
#     return out


@torch.no_grad()
def symm_all_to_all_4D(
    input: torch.Tensor,
    scatter_idx: int = 2,
    gather_idx: int = 1,
    use_sync: bool = False,
    async_op: bool = False,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    all-to-all for QKV

    Args:
        input (torch.tensor): a tensor sharded along dim scatter dim
        scatter_idx (int): default 1
        gather_idx (int): default 2
        group : torch process group
        use_sync (bool): whether to synchronize after all-to-all

    Returns:
        torch.tensor: resharded tensor (bs, seqlen/P, hc, hs)
    """
    # global group
    group = PROCESS_GROUP.ULYSSES_PG

    assert (
        input.dim() == 4
    ), f"input must be 4D tensor, got {input.dim()} and shape {input.shape}"

    seq_world_size = dist.get_world_size(group)

    if scatter_idx == 2 and gather_idx == 1:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen/P, hc, hs) output: (bs, seqlen, hc/P, hs)
        bs, shard_seqlen, hc, hs = input.shape
        seqlen = shard_seqlen * seq_world_size
        shard_hc = hc // seq_world_size

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen/P, hc, hs) -reshape-> (bs, seq_len/P, P, hc/P, hs) -transpose(0,2)-> (P, seq_len/P, bs, hc/P, hs)

        # assert input.requires_grad, f"input requires_grad must be True"
        # input_t = (
        #     input.reshape(bs, shard_seqlen, seq_world_size, shard_hc, hs)
        #     .transpose(0, 2)
        #     .contiguous()
        # )
        input_t = (
            input.reshape(bs, shard_seqlen, seq_world_size, shard_hc, hs)
            .transpose(0, 2)
            .clone(memory_format=torch.contiguous_format)
        )

        if output is None:
            output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, seq_len/P, bs, hc/P, hs) scatter seqlen -all2all-> (P, seq_len/P, bs, hc/P, hs) scatter head

        if seq_world_size > 1:
            if async_op:
                # output = torch.zeros_like(input_t)
                raise NotImplementedError("async_op not supported")
            else:
                symm_all_to_all_equal(output, input_t, group=group)
            if use_sync:
                torch.cuda.synchronize()
        else:
            output = input_t

        if not async_op:
            # if scattering the seq-dim, transpose the heads back to the original dimension
            output = output.reshape(seqlen, bs, shard_hc, hs)

            # (seq_len, bs, hc/P, hs) -reshape-> (bs, seq_len, hc/P, hs)
            output = (
                output.transpose(0, 1).contiguous().reshape(bs, seqlen, shard_hc, hs)
            )

        return output

    elif scatter_idx == 1 and gather_idx == 2:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen, hc/P, hs) output: (bs, seqlen/P, hc, hs)
        bs, seqlen, shard_hc, hs = input.shape
        hc = shard_hc * seq_world_size
        shard_seqlen = seqlen // seq_world_size
        seq_world_size = dist.get_world_size(group)

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen, hc/P, hs) -reshape-> (bs, P, seq_len/P, hc/P, hs) -transpose(0, 3)-> (hc/P, P, seqlen/P, bs, hs) -transpose(0, 1) -> (P, hc/P, seqlen/P, bs, hs)
        input_t = (
            input.reshape(bs, seq_world_size, shard_seqlen, shard_hc, hs)
            .transpose(0, 3)
            .transpose(0, 1)
            .contiguous()
            .reshape(seq_world_size, shard_hc, shard_seqlen, bs, hs)
        )

        if output is None:
            output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, bs x hc/P, seqlen/P, hs) scatter seqlen -all2all-> (P, bs x seq_len/P, hc/P, hs) scatter head
        if seq_world_size > 1:
            if async_op:
                # output = torch.zeros_like(input_t)
                raise NotImplementedError("async_op not supported")
            else:
                symm_all_to_all_equal(output, input_t, group=group)
            if use_sync:
                torch.cuda.synchronize()
        else:
            output = input_t

        if not async_op:
            # if scattering the seq-dim, transpose the heads back to the original dimension
            output = output.reshape(hc, shard_seqlen, bs, hs)

            # (hc, seqlen/N, bs, hs) -tranpose(0,2)-> (bs, seqlen/N, hc, hs)
            output = (
                output.transpose(0, 2).contiguous().reshape(bs, shard_seqlen, hc, hs)
            )

        return output

    else:
        raise RuntimeError("scatter_idx must be 1 or 2 and gather_idx must be 1 or 2")


# Cache for symm_all_to_all_latest buffers
_symm_latest_inp_buf: dict = {}
_symm_latest_out_buf: dict = {}
_symm_latest_hdl: dict = {}
_symm_latest_splits_cache: dict = {}


def _get_group_name(group) -> str:
    """Get the group name string from a ProcessGroup for use with symm_mem ops."""
    if group is None or group == dist.group.WORLD:
        group = dist.distributed_c10d._get_default_group()
    # ProcessGroup objects have a name() method in recent PyTorch versions
    if hasattr(group, "group_name"):
        return group.group_name
    elif hasattr(group, "name"):
        return group.name()
    else:
        # Fallback: try to get from internal registry
        return dist.distributed_c10d._get_pg_default_device(group).type


def symm_all_to_all_latest(
    out: Optional[torch.Tensor],
    input_tensor: torch.Tensor,
    group=dist.group.WORLD,
    split_dim: int = 0,
) -> torch.Tensor:
    """
    Symmetric-memory based all_to_all using torch.ops.symm_mem.all_to_all_vdev (NVSHMEM).

    This function performs an equal-split all-to-all operation where each rank
    sends/receives the same amount of data to/from every other rank.

    Args:
        out: Optional output tensor. If None, will be allocated.
        input_tensor: Input tensor to perform all-to-all on.
        group: Process group for the collective operation.
        split_dim: Dimension along which to split (must be 0 for this implementation).

    Returns:
        Output tensor after all-to-all permutation.
    """
    global _symm_latest_inp_buf, _symm_latest_out_buf, _symm_latest_hdl, _symm_latest_splits_cache

    # Get group name for symmetric memory registration
    group_name = _get_group_name(group)

    # Enable symmetric memory for this group and set NVSHMEM backend (must be done before any symm_mem.empty calls)
    if symm_mem.is_nvshmem_available():
        symm_mem.enable_symm_mem_for_group(group_name)
        # Also enable symm_mem for the default group because symm_mem.empty() uses it internally
        default_group = dist.distributed_c10d._get_default_group()
        default_group_name = (
            default_group.group_name if hasattr(default_group, "group_name") else "0"
        )
        if default_group_name != group_name:
            symm_mem.enable_symm_mem_for_group(default_group_name)
        symm_mem.set_backend("NVSHMEM")

    assert input_tensor.is_cuda, "symmetric memory currently expects CUDA tensors"
    assert split_dim == 0, "split_dim must be 0 for this implementation"

    world_size = dist.get_world_size(group)
    if world_size == 1:
        if out is None:
            return input_tensor.clone()
        else:
            out.copy_(input_tensor)
            return out

    dim_size = input_tensor.size(split_dim)
    assert (
        dim_size % world_size == 0
    ), "size along split_dim must be divisible by world_size"
    per_rank = dim_size // world_size

    # Allocate output if needed
    if out is None:
        out = torch.empty_like(input_tensor)

    # Cache key based on shape, dtype, device
    key = (tuple(input_tensor.shape), str(input_tensor.dtype), str(input_tensor.device))

    # Allocate/retrieve symmetric buffers
    if key not in _symm_latest_inp_buf:
        _symm_latest_inp_buf[key] = symm_mem.empty(
            input_tensor.shape, device=input_tensor.device, dtype=input_tensor.dtype
        )
        _symm_latest_out_buf[key] = symm_mem.empty(
            input_tensor.shape, device=input_tensor.device, dtype=input_tensor.dtype
        )
        _symm_latest_hdl[key] = symm_mem.rendezvous(_symm_latest_inp_buf[key], group)
        # Also rendezvous the output buffer (needed for symmetric memory)
        symm_mem.rendezvous(_symm_latest_out_buf[key], group)

    # Build splits tensors as symmetric memory (cached per world_size, per_rank, and group)
    # Note: in_splits and out_splits_offsets must be symmetric tensors per the API
    splits_key = (world_size, per_rank, str(input_tensor.device), id(group))
    if splits_key not in _symm_latest_splits_cache:
        # in_splits: [per_rank, per_rank, ...] of shape (world_size,)
        # Allocate as symmetric and fill with values
        in_splits_symm = symm_mem.empty(
            (world_size,), dtype=torch.int64, device=input_tensor.device
        )
        in_splits_symm.fill_(per_rank)
        symm_mem.rendezvous(in_splits_symm, group)

        # out_splits_offsets: shape (2, world_size)
        # Row 0: output splits [per_rank, per_rank, ...]
        # Row 1: output offsets [0, per_rank, 2*per_rank, ...]
        out_splits_offsets_symm = symm_mem.empty(
            (2, world_size), dtype=torch.int64, device=input_tensor.device
        )
        out_splits_offsets_symm[0].fill_(per_rank)  # splits row
        out_splits_offsets_symm[1].copy_(
            torch.arange(
                0,
                world_size * per_rank,
                per_rank,
                dtype=torch.int64,
                device=input_tensor.device,
            )
        )  # offsets row
        symm_mem.rendezvous(out_splits_offsets_symm, group)

        _symm_latest_splits_cache[splits_key] = (
            in_splits_symm,
            out_splits_offsets_symm,
        )

    in_splits, out_splits_offsets = _symm_latest_splits_cache[splits_key]

    # Copy input to symmetric buffer
    _symm_latest_inp_buf[key].copy_(input_tensor)

    # Get group name for the symm_mem op
    group_name = _get_group_name(group)

    # Perform the all-to-all using NVSHMEM
    torch.ops.symm_mem.all_to_all_vdev(
        _symm_latest_inp_buf[key],
        _symm_latest_out_buf[key],
        in_splits,
        out_splits_offsets,
        group_name,
    )

    # Copy result to output
    out.copy_(_symm_latest_out_buf[key])

    return out


@torch.no_grad()
def symm_all_to_all_latest_4D(
    input: torch.Tensor,
    scatter_idx: int = 2,
    gather_idx: int = 1,
    use_sync: bool = False,
    async_op: bool = False,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    4D all-to-all using torch.ops.symm_mem.all_to_all_vdev (NVSHMEM).

    Performs all-to-all for QKV tensors, exchanging heads for sequence positions
    or vice versa across the Ulysses process group.

    Args:
        input: A 4D tensor sharded along the scatter dimension.
        scatter_idx: Dimension to scatter (1 or 2). Default 2.
        gather_idx: Dimension to gather (1 or 2). Default 1.
        use_sync: Whether to synchronize after all-to-all.
        async_op: Not supported for this implementation.
        output: Optional pre-allocated output tensor.

    Returns:
        Resharded tensor:
        - If scatter_idx=2, gather_idx=1: (bs, seqlen/P, hc, hs) -> (bs, seqlen, hc/P, hs)
        - If scatter_idx=1, gather_idx=2: (bs, seqlen, hc/P, hs) -> (bs, seqlen/P, hc, hs)
    """
    group = PROCESS_GROUP.ULYSSES_PG

    assert (
        input.dim() == 4
    ), f"input must be 4D tensor, got {input.dim()} and shape {input.shape}"

    seq_world_size = dist.get_world_size(group)

    if scatter_idx == 2 and gather_idx == 1:
        # input: (bs, seqlen/P, hc, hs) -> output: (bs, seqlen, hc/P, hs)
        bs, shard_seqlen, hc, hs = input.shape
        seqlen = shard_seqlen * seq_world_size
        shard_hc = hc // seq_world_size

        # Reshape and transpose to prepare for all-to-all
        # (bs, seqlen/P, hc, hs) -> (bs, seqlen/P, P, hc/P, hs) -> (P, seqlen/P, bs, hc/P, hs)
        input_t = (
            input.reshape(bs, shard_seqlen, seq_world_size, shard_hc, hs)
            .transpose(0, 2)
            .clone(memory_format=torch.contiguous_format)
        )

        if output is None:
            output = torch.empty_like(input_t)

        if seq_world_size > 1:
            if async_op:
                raise NotImplementedError(
                    "async_op not supported for symm_all_to_all_latest_4D"
                )
            else:
                symm_all_to_all_latest(output, input_t, group=group)
            if use_sync:
                torch.cuda.synchronize()
        else:
            output = input_t

        if not async_op:
            # Transpose back: (P, seqlen/P, bs, hc/P, hs) -> (seqlen, bs, hc/P, hs)
            output = output.reshape(seqlen, bs, shard_hc, hs)
            # -> (bs, seqlen, hc/P, hs)
            output = (
                output.transpose(0, 1).contiguous().reshape(bs, seqlen, shard_hc, hs)
            )

        return output

    elif scatter_idx == 1 and gather_idx == 2:
        # input: (bs, seqlen, hc/P, hs) -> output: (bs, seqlen/P, hc, hs)
        bs, seqlen, shard_hc, hs = input.shape
        hc = shard_hc * seq_world_size
        shard_seqlen = seqlen // seq_world_size

        # Reshape and transpose to prepare for all-to-all
        # (bs, seqlen, hc/P, hs) -> (bs, P, seqlen/P, hc/P, hs)
        # -> (hc/P, P, seqlen/P, bs, hs) -> (P, hc/P, seqlen/P, bs, hs)
        input_t = (
            input.reshape(bs, seq_world_size, shard_seqlen, shard_hc, hs)
            .transpose(0, 3)
            .transpose(0, 1)
            .contiguous()
            .reshape(seq_world_size, shard_hc, shard_seqlen, bs, hs)
        )

        if output is None:
            output = torch.empty_like(input_t)

        if seq_world_size > 1:
            if async_op:
                raise NotImplementedError(
                    "async_op not supported for symm_all_to_all_latest_4D"
                )
            else:
                symm_all_to_all_latest(output, input_t, group=group)
            if use_sync:
                torch.cuda.synchronize()
        else:
            output = input_t

        if not async_op:
            # Transpose back: (P, hc/P, seqlen/P, bs, hs) -> (hc, seqlen/P, bs, hs)
            output = output.reshape(hc, shard_seqlen, bs, hs)
            # -> (bs, seqlen/P, hc, hs)
            output = (
                output.transpose(0, 2).contiguous().reshape(bs, shard_seqlen, hc, hs)
            )

        return output

    else:
        raise RuntimeError("scatter_idx must be 1 or 2 and gather_idx must be 1 or 2")
