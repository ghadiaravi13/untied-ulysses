"""All-to-all communication primitives for sequence parallelism."""

import torch
import torch.distributed as dist

from yunchang.globals import PROCESS_GROUP, U_HANDLE as HANDLE


def set_u_handle(handle):
    """Set Ulysses forward handle for async all-to-all."""
    HANDLE.HANDLE.append(handle)


def set_o_handle(handle):
    """Set output handle for async all-to-all."""
    HANDLE.O_HANDLE.append(handle)


def wait_u_handle():
    """Wait for Ulysses forward handles."""
    for h in HANDLE.HANDLE:
        h.wait()
    HANDLE.HANDLE.clear()


def wait_o_handle():
    """Wait for output handles."""
    for h in HANDLE.O_HANDLE:
        h.wait()
    HANDLE.O_HANDLE.clear()


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
    All-to-all for 4D QKV tensors.

    Args:
        input: Input tensor sharded along scatter_idx dimension
        scatter_idx: Dimension to scatter (1 or 2)
        gather_idx: Dimension to gather (1 or 2)
        use_sync: Whether to synchronize after all-to-all
        async_op: Whether to run asynchronously
        output: Optional pre-allocated output buffer

    Returns:
        Resharded tensor (or input for async_op=True before wait)
    """
    group = PROCESS_GROUP.ULYSSES_PG
    assert input.dim() == 4, f"input must be 4D tensor, got {input.dim()}"

    seq_world_size = dist.get_world_size(group)

    if scatter_idx == 2 and gather_idx == 1:
        # (bs, seqlen/P, hc, hs) -> (bs, seqlen, hc/P, hs)
        bs, shard_seqlen, hc, hs = input.shape
        seqlen = shard_seqlen * seq_world_size
        shard_hc = hc // seq_world_size

        # (bs, seqlen/P, hc, hs) -> (bs, seqlen/P, P, hc/P, hs) -> (P, seqlen/P, bs, hc/P, hs)
        input_t = (
            input.reshape(bs, shard_seqlen, seq_world_size, shard_hc, hs)
            .transpose(0, 2)
            .contiguous()
        )

        if output is None:
            output = torch.empty_like(input_t)

        if seq_world_size > 1:
            if async_op:
                handle = dist.all_to_all_single(
                    output, input_t, group=group, async_op=True
                )
                set_u_handle(handle)
            else:
                dist.all_to_all_single(output, input_t, group=group)
            if use_sync:
                torch.cuda.synchronize()
        else:
            output = input_t

        if not async_op:
            output = output.reshape(seqlen, bs, shard_hc, hs)
            output = output.transpose(0, 1)

        return output

    elif scatter_idx == 1 and gather_idx == 2:
        # (bs, seqlen, hc/P, hs) -> (bs, seqlen/P, hc, hs)
        bs, seqlen, shard_hc, hs = input.shape
        hc = shard_hc * seq_world_size
        shard_seqlen = seqlen // seq_world_size

        # (bs, seqlen, hc/P, hs) -> (bs, P, seqlen/P, hc/P, hs) -> (P, hc/P, seqlen/P, bs, hs)
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
                handle = dist.all_to_all_single(
                    output, input_t, group=group, async_op=True
                )
                set_o_handle(handle)
            else:
                dist.all_to_all_single(output, input_t, group=group)
            if use_sync:
                torch.cuda.synchronize()
        else:
            output = input_t

        if not async_op:
            output = output.reshape(hc, shard_seqlen, bs, hs)
            output = (
                output.transpose(0, 2).contiguous().reshape(bs, shard_seqlen, hc, hs)
            )

        return output

    else:
        raise RuntimeError("scatter_idx must be 1 or 2 and gather_idx must be 1 or 2")
