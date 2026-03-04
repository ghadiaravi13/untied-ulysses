"""
Upipe Attention with Untied Ulysses Sequence Parallelism.
"""

import os
import sys

# Add the directory containing this file to path for relative imports
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from all_to_all import vanilla_all_to_all_4D as all_to_all_4D

from flash_attn_utils import (
    zigzag_ring_flash_attn_backward,
    zigzag_ring_flash_attn_forward,
)

from rotary import apply_rotary_emb_flash
from yunchang.globals import PROCESS_GROUP


@torch.library.custom_op(
    "upipe::_upipe_attn_gqa_forward", mutates_args=(), device_types="cuda"
)
def upipe_attn_gqa_forward(
    ulysses_group_name: str,
    ring_group_name: str,
    x: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    freqs_cis: torch.Tensor,
    head_dim: int,
    dropout_p: float = 0.0,
    softmax_scale: float = 0.0,
    causal: bool = True,
    attn_type: str = "fa2",
) -> list[torch.Tensor]:
    """
    Forward pass for upipe GQA attention.


    Args:
        ulysses_group_name: Process group name for Ulysses parallelism
        ring_group_name: Process group name for ring attention
        x: Input tensor [batch, seqlen, hidden_dim]
        wq, wk, wv: Weight matrices for Q, K, V projections
        freqs_cis: Rotary embedding frequencies
        head_dim: Head dimension
        dropout_p: Dropout probability
        softmax_scale: Softmax scale (0 means auto-compute)
        causal: Whether to use causal attention
        attn_type: "fa2" or "fa3"

    Returns:
        List of [final_out, *lse_per_stage]
    """
    ulysses_group = dist.distributed_c10d._resolve_process_group(ulysses_group_name)
    ring_group = dist.distributed_c10d._resolve_process_group(ring_group_name)

    freqs_cis = freqs_cis.to(x.device)

    bs, shard_seqlen, hid_dim = x.shape
    # Derive n_heads from wq weight shape, not from hid_dim, to support
    # head_dim != dim // n_heads (e.g. Qwen3-32B: dim=5120, head_dim=128, n_heads=64)
    n_heads = wq.shape[0] // head_dim
    n_kv_heads_may_be_replicated = wk.shape[0] // head_dim
    gqa_ratio = n_heads // n_kv_heads_may_be_replicated

    ulysses_degree = dist.get_world_size(ulysses_group)
    pipe_degree = n_heads // ulysses_degree

    assert (
        n_kv_heads_may_be_replicated % ulysses_degree == 0
    ), f"n_kv_heads_may_be_replicated ({n_kv_heads_may_be_replicated}) must be divisible by ulysses_degree ({ulysses_degree})"

    if softmax_scale == 0.0:
        softmax_scale = head_dim ** (-0.5)

    wq_chunks = torch.chunk(wq, pipe_degree, dim=0)
    wk_chunks = torch.chunk(wk, pipe_degree // gqa_ratio, dim=0)
    wv_chunks = torch.chunk(wv, pipe_degree // gqa_ratio, dim=0)

    lse_list = []
    final_out = torch.empty(
        [bs, shard_seqlen, n_heads, head_dim], device=x.device, dtype=x.dtype
    )

    k_out = None
    v_out = None

    for stage in range(pipe_degree):
        q_proj = F.linear(x, wq_chunks[stage])
        q_proj = q_proj.view(bs, shard_seqlen, -1, head_dim)

        if stage % gqa_ratio == 0:
            k_proj = F.linear(x, wk_chunks[stage // gqa_ratio])
            v_proj = F.linear(x, wv_chunks[stage // gqa_ratio])

            k_proj = k_proj.view(bs, shard_seqlen, -1, head_dim)
            v_proj = v_proj.view(bs, shard_seqlen, -1, head_dim)

            q_proj, k_proj = apply_rotary_emb_flash(
                xq=q_proj, xk=k_proj, freqs_cis=freqs_cis
            )

            q_out = all_to_all_4D(q_proj, scatter_idx=2, gather_idx=1)
            k_out = all_to_all_4D(k_proj, scatter_idx=2, gather_idx=1)
            v_out = all_to_all_4D(v_proj, scatter_idx=2, gather_idx=1)

            # deleting the inp to all_to_all to avoid memory leaks
            del q_proj, k_proj, v_proj
            
        else:
            q_proj = apply_rotary_emb_flash(xq=q_proj, xk=None, freqs_cis=freqs_cis)
            q_out = all_to_all_4D(q_proj, scatter_idx=2, gather_idx=1)

            # deleting the inp to all_to_all to avoid memory leaks
            del q_proj

        attn_out, lse = zigzag_ring_flash_attn_forward(
            ring_group,
            q_out,
            k_out,
            v_out,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            attn_type=attn_type,
        )
        lse_list.append(lse)

        # deleting the inp to attn_forward to avoid memory leaks
        del q_out
        if (stage+1)//gqa_ratio != stage//gqa_ratio:
            del k_out, v_out

        out_local = all_to_all_4D(attn_out, scatter_idx=1, gather_idx=2)

        # deleting the inp to all_to_all to avoid memory leaks
        del attn_out

        head_start = stage * ulysses_degree
        head_end = head_start + ulysses_degree
        final_out[:, :, head_start:head_end, :] = out_local

        # deleting the output of all_to_all to avoid memory leaks
        del out_local

    return [final_out] + lse_list


@torch.library.custom_op(
    "upipe::_upipe_attn_gqa_backward", mutates_args=(), device_types="cuda"
)
def upipe_attn_gqa_backward(
    ulysses_group_name: str,
    ring_group_name: str,
    dout: torch.Tensor,
    x: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    freqs_cis: torch.Tensor,
    final_out: torch.Tensor,
    lse_list: list[torch.Tensor],
    head_dim: int,
    n_kv_heads: int,
    dropout_p: float = 0.0,
    softmax_scale: float = 0.0,
    causal: bool = True,
    attn_type: str = "fa2",
    deterministic: bool = False,
) -> list[torch.Tensor]:
    """
    Backward pass for upipe GQA attention.

    Args:
        ulysses_group_name: Process group name for Ulysses parallelism
        ring_group_name: Process group name for ring attention
        dout: Gradient of output [batch, seqlen, n_heads, head_dim]
        x: Input tensor from forward
        wq, wk, wv: Weight matrices (wk/wv may be replicated)
        freqs_cis: Rotary embedding frequencies
        final_out: Output from forward [batch, seqlen, n_heads, head_dim]
        lse_list: LSE tensors from forward (one per stage)
        head_dim: Head dimension
        n_kv_heads: Original (pre-replication) number of KV heads
        dropout_p: Dropout probability
        softmax_scale: Softmax scale
        causal: Whether causal attention was used
        attn_type: "fa2" or "fa3"
        deterministic: Use deterministic algorithms

    Returns:
        List of [dx, dwq, dwk, dwv]
    """
    ulysses_group = dist.distributed_c10d._resolve_process_group(ulysses_group_name)
    ring_group = dist.distributed_c10d._resolve_process_group(ring_group_name)

    freqs_cis = freqs_cis.to(x.device)
    freqs_cis_conj = torch.conj(freqs_cis)

    bs, shard_seqlen, hid_dim = x.shape
    # Derive n_heads from wq weight shape, not from hid_dim (see forward fix)
    n_heads = wq.shape[0] // head_dim
    # wk may be replicated; use n_kv_heads_may_be_replicated for chunking/indexing,
    # but pass the original n_kv_heads to _reduce_gqa_gradients.
    n_kv_heads_may_be_replicated = wk.shape[0] // head_dim
    gqa_ratio = n_heads // n_kv_heads_may_be_replicated

    ulysses_degree = dist.get_world_size(ulysses_group)
    pipe_degree = n_heads // ulysses_degree

    if softmax_scale == 0.0:
        softmax_scale = head_dim ** (-0.5)

    wq_chunks = torch.chunk(wq, pipe_degree, dim=0)
    wk_chunks = torch.chunk(wk, pipe_degree // gqa_ratio, dim=0)
    wv_chunks = torch.chunk(wv, pipe_degree // gqa_ratio, dim=0)

    final_out_chunks = list(torch.chunk(final_out, pipe_degree, dim=2))
    dout_chunks = list(torch.chunk(dout, pipe_degree, dim=2))

    dx = None
    dwq = torch.zeros_like(wq)
    dwk = torch.zeros_like(wk)
    dwv = torch.zeros_like(wv)

    dk_accum = [None for _ in range(pipe_degree // gqa_ratio)]
    dv_accum = [None for _ in range(pipe_degree // gqa_ratio)]

    k_out = None
    v_out = None

    x_flat = x.view(bs * shard_seqlen, -1)

    for stage in range(pipe_degree):
        q_proj = F.linear(x, wq_chunks[stage])
        q_proj = q_proj.view(bs, shard_seqlen, -1, head_dim)

        if stage % gqa_ratio == 0:
            k_proj = F.linear(x, wk_chunks[stage // gqa_ratio])
            v_proj = F.linear(x, wv_chunks[stage // gqa_ratio])

            k_proj = k_proj.view(bs, shard_seqlen, -1, head_dim)
            v_proj = v_proj.view(bs, shard_seqlen, -1, head_dim)

            q_proj, k_proj = apply_rotary_emb_flash(
                xq=q_proj, xk=k_proj, freqs_cis=freqs_cis
            )

            q_out = all_to_all_4D(q_proj, scatter_idx=2, gather_idx=1)
            k_out = all_to_all_4D(k_proj, scatter_idx=2, gather_idx=1)
            v_out = all_to_all_4D(v_proj, scatter_idx=2, gather_idx=1)
            del (
                q_proj,
                k_proj,
                v_proj,
            )  # deleting the inp to all_to_all to avoid memory leaks
        else:
            q_proj = apply_rotary_emb_flash(xq=q_proj, xk=None, freqs_cis=freqs_cis)
            q_out = all_to_all_4D(q_proj, scatter_idx=2, gather_idx=1)
            del q_proj  # deleting the inp to all_to_all to avoid memory leaks

        out_a2a = all_to_all_4D(final_out_chunks[stage], scatter_idx=2, gather_idx=1)
        dout_a2a = all_to_all_4D(dout_chunks[stage], scatter_idx=2, gather_idx=1)

        # deleting the inp to all_to_all to avoid memory leaks
        final_out_chunks[stage] = None
        dout_chunks[stage] = None

        attn_dq, attn_dk, attn_dv = zigzag_ring_flash_attn_backward(
            ring_group,
            dout_a2a,
            q_out,
            k_out,
            v_out,
            out_a2a,
            lse_list[stage],
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            deterministic=deterministic,
            attn_type=attn_type,
        )

        # deleting the inp to attn_backward to avoid memory leaks
        lse_list[stage] = None
        del dout_a2a, q_out, out_a2a
        if (
            stage + 1
        ) // gqa_ratio != stage // gqa_ratio:  # clear k_out / v_out if they are not reused in the next stage
            del k_out, v_out

        if dk_accum[stage // gqa_ratio] is None:
            dk_accum[stage // gqa_ratio] = attn_dk
            dv_accum[stage // gqa_ratio] = attn_dv
        else:
            dk_accum[stage // gqa_ratio].add_(attn_dk)
            dv_accum[stage // gqa_ratio].add_(attn_dv)

        dq_local = all_to_all_4D(attn_dq, scatter_idx=1, gather_idx=2)
        dq_local = apply_rotary_emb_flash(dq_local, freqs_cis=freqs_cis_conj)
        dq_flat = dq_local.view(bs * shard_seqlen, -1)

        # deleting the output of attn_backward to avoid memory leaks
        del attn_dq, attn_dk, attn_dv

        if dx is None:
            dx = dq_flat @ wq_chunks[stage]
        else:
            dx.addmm_(dq_flat, wq_chunks[stage])

        head_start = stage * (head_dim * ulysses_degree)
        head_end = (stage + 1) * (head_dim * ulysses_degree)
        dwq[head_start:head_end, :] = dq_flat.T @ x_flat

        # deleting to avoid memory leaks
        del dq_local, dq_flat

        if (stage + 1) % gqa_ratio == 0 or stage == pipe_degree - 1:
            kv_idx = stage // gqa_ratio

            dk_local = all_to_all_4D(dk_accum[kv_idx], scatter_idx=1, gather_idx=2)
            dv_local = all_to_all_4D(dv_accum[kv_idx], scatter_idx=1, gather_idx=2)

            dk_accum[kv_idx] = None
            dv_accum[kv_idx] = None

            dk_local = apply_rotary_emb_flash(dk_local, freqs_cis=freqs_cis_conj)
            dk_flat = dk_local.view(bs * shard_seqlen, -1)
            dv_flat = dv_local.view(bs * shard_seqlen, -1)

            dx.addmm_(dk_flat, wk_chunks[kv_idx])
            dx.addmm_(dv_flat, wv_chunks[kv_idx])

            kv_head_start = kv_idx * (head_dim * ulysses_degree)
            kv_head_end = (kv_idx + 1) * (head_dim * ulysses_degree)
            dwk[kv_head_start:kv_head_end, :] = dk_flat.T @ x_flat
            dwv[kv_head_start:kv_head_end, :] = dv_flat.T @ x_flat

            # deleting the inp/out of all_to_all to avoid memory leaks
            del dk_local, dv_local, dk_flat, dv_flat

    # Handle replicated KV weights case (when n_kv_heads < ulysses_degree)
    dwk = _reduce_gqa_gradients(dwk, n_kv_heads, head_dim, hid_dim)
    dwv = _reduce_gqa_gradients(dwv, n_kv_heads, head_dim, hid_dim)

    return [dx.view(bs, shard_seqlen, -1).to(x.dtype), dwq, dwk, dwv]


def _reduce_gqa_gradients(
    dw: torch.Tensor,
    n_kv_heads: int,
    head_dim: int,
    hidden_dim: int,
) -> torch.Tensor:
    """
    Reduce gradients for replicated KV weights.

    When n_kv_heads < ulysses_degree, KV weights are replicated.
    This sums gradients from replicated slots and copies back to all slots.
    """
    if dw.shape[0] // head_dim > n_kv_heads:
        n_rep = (dw.shape[0] // head_dim) // n_kv_heads
        dw = dw.view(n_kv_heads, n_rep, head_dim, hidden_dim)
        dw_sum = dw.sum(dim=1, keepdim=True)
        dw = dw_sum.expand(-1, n_rep, -1, -1).reshape(-1, hidden_dim)
    return dw


class UpipeAttnGQAFunc(torch.autograd.Function):
    """Autograd function for upipe GQA attention with Untied Ulysses parallelism."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        wq: torch.Tensor,
        wk: torch.Tensor,
        wv: torch.Tensor,
        freqs_cis: torch.Tensor,
        head_dim: int,
        n_kv_heads: int,
        dropout_p: float,
        softmax_scale: Optional[float],
        causal: bool,
        ulysses_group_name: str,
        ring_group_name: str,
        attn_type: str,
        deterministic: bool,
    ) -> torch.Tensor:
        """Forward pass."""
        bs, seqlen, hid_dim = x.shape

        if softmax_scale is None:
            softmax_scale = head_dim ** (-0.5)

        with torch.no_grad():
            outputs = torch.ops.upipe._upipe_attn_gqa_forward(
                ulysses_group_name,
                ring_group_name,
                x,
                wq,
                wk,
                wv,
                freqs_cis,
                head_dim,
                dropout_p,
                softmax_scale,
                causal,
                attn_type,
            )
            final_out = outputs[0]
            lse_list = outputs[1:]

        ctx.save_for_backward(x, wq, wk, wv, freqs_cis, final_out, *lse_list)
        ctx.ulysses_group_name = ulysses_group_name
        ctx.ring_group_name = ring_group_name
        ctx.head_dim = head_dim
        ctx.n_kv_heads = n_kv_heads
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.attn_type = attn_type
        ctx.deterministic = deterministic

        # Use -1 instead of hid_dim: when head_dim != dim//n_heads,
        # n_heads*head_dim differs from the input hidden dim
        return final_out.view(bs, seqlen, -1)

    @staticmethod
    def backward(ctx, dout: torch.Tensor) -> tuple:
        """Backward pass."""
        saved = ctx.saved_tensors
        x, wq, wk, wv, freqs_cis, final_out = saved[:6]
        lse_list = list(saved[6:])

        bs, seqlen, hid_dim = x.shape
        # Derive n_heads from wq weight shape, not from hid_dim (see forward fix)
        n_heads = wq.shape[0] // ctx.head_dim

        dout = dout.view(bs, seqlen, n_heads, ctx.head_dim)

        with torch.no_grad():
            dx, dwq, dwk, dwv = torch.ops.upipe._upipe_attn_gqa_backward(
                ctx.ulysses_group_name,
                ctx.ring_group_name,
                dout,
                x,
                wq,
                wk,
                wv,
                freqs_cis,
                final_out,
                lse_list,
                ctx.head_dim,
                ctx.n_kv_heads,
                ctx.dropout_p,
                ctx.softmax_scale,
                ctx.causal,
                ctx.attn_type,
                ctx.deterministic,
            )

        return (
            dx,
            dwq,
            dwk,
            dwv,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def upipe_attn_gqa(
    x: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    freqs_cis: torch.Tensor,
    head_dim: int,
    n_kv_heads: int,
    ulysses_group_name: str,
    ring_group_name: str,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    attn_type: str = "fa2",
    deterministic: bool = False,
) -> torch.Tensor:
    """
    Upipe GQA attention with Untied Ulysses sequence parallelism.

    Args:
        x: Input tensor [batch, seqlen, hidden_dim]
        wq, wk, wv: Weight matrices for Q, K, V projections (wk/wv may be replicated)
        freqs_cis: Rotary embedding frequencies
        head_dim: Head dimension
        n_kv_heads: Original (pre-replication) number of KV heads
        ulysses_group_name: Process group name for Ulysses parallelism
        ring_group_name: Process group name for ring attention
        dropout_p: Dropout probability
        softmax_scale: Softmax scale (default: 1/sqrt(head_dim))
        causal: Whether to use causal attention
        attn_type: "fa2" or "fa3"
        deterministic: Use deterministic algorithms in backward

    Returns:
        Output tensor [batch, seqlen, hidden_dim]
    """
    return UpipeAttnGQAFunc.apply(
        x,
        wq,
        wk,
        wv,
        freqs_cis,
        head_dim,
        n_kv_heads,
        dropout_p,
        softmax_scale,
        causal,
        ulysses_group_name,
        ring_group_name,
        attn_type,
        deterministic,
    )


class UpipeAttention(torch.nn.Module):
    """
    Upipe Attention Layer with Untied Ulysses Sequence Parallelism.

    This is a drop-in replacement for FullyFusedLongContextAttention that uses
    the Untied Ulysses parallelism strategy.

    Args:
        ulysses_pg: Process group for Ulysses parallelism (default: from globals)
        ring_pg: Process group for ring attention (default: from globals)
        attn_type: "fa2" or "fa3" for FlashAttention version
        layer_id: Layer identifier for debugging
        deterministic: Use deterministic algorithms in backward
    """

    def __init__(
        self,
        ulysses_pg=None,
        ring_pg=None,
        attn_type: str = "fa2",
        layer_id: int = None,
        deterministic: bool = False,
        ring_impl_type: str = None,
        use_pack_qkv: bool = False,
        offload_stream=None,
        fetch_stream=None,
        two_streams=None,
    ):
        super().__init__()

        if ulysses_pg is None:
            ulysses_pg = PROCESS_GROUP.ULYSSES_PG
        if ring_pg is None:
            ring_pg = PROCESS_GROUP.RING_PG

        self.ulysses_pg = ulysses_pg
        self.ring_pg = ring_pg

        assert (
            self.ulysses_pg is not None
        ), "Ulysses process group not set. Call set_seq_parallel_pg() first."

        if hasattr(attn_type, "name"):
            self.attn_type = "fa3" if "FA3" in attn_type.name else "fa2"
        else:
            self.attn_type = attn_type

        self.layer_id = layer_id
        self.deterministic = deterministic

    def forward(
        self,
        x: torch.Tensor,
        wq: torch.Tensor,
        wk: torch.Tensor,
        wv: torch.Tensor,
        freqs_cis: torch.Tensor,
        head_dim: int,
        n_kv_heads: Optional[int] = None,
        dropout_p: float = 0.0,
        softmax_scale: Optional[float] = None,
        causal: bool = True,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        deterministic=None,
        return_attn_probs=False,
        fused_attn_type=None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, seqlen, hidden_dim]
            wq, wk, wv: Weight matrices for Q, K, V projections (wk/wv may be replicated)
            freqs_cis: Rotary embedding frequencies
            head_dim: Head dimension
            n_kv_heads: Original (pre-replication) number of KV heads.
                        If None, derived from wk.shape (no replication assumed).
            dropout_p: Dropout probability
            softmax_scale: Softmax scale (default: 1/sqrt(head_dim))
            causal: Whether to use causal attention

        Returns:
            Output tensor [batch, seqlen, n_heads, head_dim]
        """
        if deterministic is None:
            deterministic = self.deterministic

        # If n_kv_heads not provided, assume no replication (derive from wk shape)
        if n_kv_heads is None:
            n_kv_heads = wk.shape[0] // head_dim

        ulysses_group_name = self.ulysses_pg.group_name
        ring_group_name = self.ring_pg.group_name if self.ring_pg is not None else ""

        output = UpipeAttnGQAFunc.apply(
            x,
            wq,
            wk,
            wv,
            freqs_cis,
            head_dim,
            n_kv_heads,
            dropout_p,
            softmax_scale,
            causal,
            ulysses_group_name,
            ring_group_name,
            self.attn_type,
            deterministic,
        )

        bs, seqlen, _attn_dim = output.shape
        # Derive n_heads from wq weight shape for correctness when head_dim != dim//n_heads
        n_heads = wq.shape[0] // head_dim
        return output.view(bs, seqlen, n_heads, head_dim)
