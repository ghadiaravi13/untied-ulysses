import os

from typing import List, Tuple

import torch

from yunchang.comm.all_to_all import vanilla_all_to_all_4D as all_to_all_4D
from yunchang.kernels import AttnType, select_flash_attn_impl
from .utils import RingComm, update_out_and_lse

global process_group, attn_type, alibi_slopes, window_size


@torch.library.custom_op(
    "yunchang::_zigzag_ring_flash_attn_forward", mutates_args=(), device_types="cuda"
)
def zigzag_ring_flash_attn_forward(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
    softmax_scale: float,
    dropout_p: float = 0,
    causal: bool = True,
    softcap: float = 0.0,
    deterministic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:

    global process_group, attn_type, alibi_slopes, window_size

    assert causal == True, "zigzag ring is meaningless for causal=False"
    comm = RingComm(process_group)

    q = all_to_all_4D(q_in, 2, 1, False, False)
    k = all_to_all_4D(k_in, 2, 1, False, False)
    v = all_to_all_4D(v_in, 2, 1, False, False)

    block_seq_len = q.shape[1] // 2
    q1 = q[:, block_seq_len:]

    out = None
    lse = None
    next_k, next_v = None, None

    def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
        return x
        bs, slen, n_kv_heads, head_dim = x.shape
        if n_rep == 1:
            return x
        return (
            torch.unsqueeze(x, dim=3)
            .expand(bs, slen, n_kv_heads, n_rep, head_dim)
            .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
        )

    def forward(q, k, v, causal):
        fn = select_flash_attn_impl(attn_type, stage="fwd-only")
        # assert q.shape[2]%k.shape[2] == 0, f"q.shape[2] {q.shape[2]} must be divisible by k.shape[2] {k.shape[2]}"
        # if k.shape[2] != q.shape[2]:
        #     k = repeat_kv(k, q.shape[2]//k.shape[2])
        #     v = repeat_kv(v, q.shape[2]//v.shape[2])
        block_out, block_lse = fn(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=True and dropout_p > 0,
        )
        return block_out, block_lse

    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k: torch.Tensor = comm.send_recv(k)
            next_v: torch.Tensor = comm.send_recv(v)
            comm.commit()

        if step == 0:
            block_out, block_lse = forward(q, k, v, causal=True)
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        elif step <= comm.rank:
            k0 = k[:, :block_seq_len]
            v0 = v[:, :block_seq_len]
            block_out, block_lse = forward(q, k0, v0, causal=False)
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        else:
            block_out, block_lse = forward(q1, k, v, causal=False)
            out, lse = update_out_and_lse(
                out,
                lse,
                block_out,
                block_lse,
                slice_=(slice(None), slice(block_seq_len, None)),
            )

        if step + 1 != comm.world_size:
            comm.wait()
            k = next_k
            v = next_v

    out = out.to(q.dtype)
    lse = lse.squeeze(dim=-1).transpose(1, 2)

    out_new = all_to_all_4D(out, 1, 2, False, False)
    return out_new, lse


@zigzag_ring_flash_attn_forward.register_fake
def _(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    dropout_p: float = 0,
    causal: bool = True,
    softcap: float = 0.0,
    deterministic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bs, sl, nh, d = q.shape
    out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    lse = torch.empty([bs, sl, nh], dtype=q.dtype, device=q.device)
    return out, lse


# @torch.library.custom_op("yunchang::_zigzag_ring_flash_attn_forward_op", mutates_args=(), device_types="cuda")
# def zigzag_ring_flash_attn_forward_op(out: torch.Tensor, softmax_lse: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#     return out.clone().detach().requires_grad_(True), softmax_lse.clone().detach().requires_grad_(True)

# @zigzag_ring_flash_attn_forward_op.register_fake
# def _(out: torch.Tensor, softmax_lse: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#     return out.clone().detach().requires_grad_(True), softmax_lse.clone().detach().requires_grad_(True)


def zigzag_ring_flash_attn_backward(
    process_group,
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    attn_type: AttnType = AttnType.FA,
):
    assert causal == True, "zigzag ring is meaningless for causal=False"
    kv_comm = RingComm(process_group)
    d_kv_comm = RingComm(process_group)
    dq, dk, dv = None, None, None
    next_dk, next_dv = None, None
    next_k, next_v = None, None
    dk_comm_buffer, dv_comm_buffer = None, None

    dout = all_to_all_4D(dout, 2, 1, False, False)
    out = all_to_all_4D(out, 2, 1, False, False)
    q = all_to_all_4D(q, 2, 1, False, False)
    k = all_to_all_4D(k, 2, 1, False, False)
    v = all_to_all_4D(v, 2, 1, False, False)

    dout1 = dout.chunk(2, dim=1)[1]
    q1 = q.chunk(2, dim=1)[1]
    out1 = out.chunk(2, dim=1)[1]
    softmax_lse1 = softmax_lse.chunk(2, dim=2)[1].contiguous()
    block_seq_len = q.shape[1] // 2

    # repeatly allocating buffer may be slow...
    # if k.shape[2] == q.shape[2]:
    dq_buffer = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    dk_buffer = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    dv_buffer = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    # else:
    #     dq_buffer = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    #     dk_buffer = torch.empty((k.shape[0], k.shape[1], q.shape[2], k.shape[3]), dtype=k.dtype, device=k.device)
    #     dv_buffer = torch.empty((v.shape[0], v.shape[1], q.shape[2], v.shape[3]), dtype=v.dtype, device=v.device)

    def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
        return x
        bs, slen, n_kv_heads, head_dim = x.shape
        if n_rep == 1:
            return x
        return (
            torch.unsqueeze(x, dim=3)
            .expand(bs, slen, n_kv_heads, n_rep, head_dim)
            .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
        )

    def backward(dout, q, k, v, out, softmax_lse, causal):
        seqlen_q = q.shape[1]
        seqlen_kv = k.shape[1]
        fn = select_flash_attn_impl(attn_type, stage="bwd-only")

        # if k.shape[2] != q.shape[2]:
        #     k = repeat_kv(k, q.shape[2]//k.shape[2])
        #     v = repeat_kv(v, q.shape[2]//v.shape[2])

        fn(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq_buffer[:, :seqlen_q],
            dk_buffer[:, :seqlen_kv],
            dv_buffer[:, :seqlen_kv],
            dropout_p,
            softmax_scale,
            causal,
            window_size,
            softcap,
            alibi_slopes,
            deterministic,
            rng_state=None,
        )

    for step in range(kv_comm.world_size):
        if step + 1 != kv_comm.world_size:
            next_k = kv_comm.send_recv(k)
            next_v = kv_comm.send_recv(v)
            kv_comm.commit()

        if step == 0:
            backward(dout, q, k, v, out, softmax_lse, causal=True)
            if (
                os.environ.get("DEBUG_MODE", "0") == "1"
                and torch.distributed.get_rank() == 0
            ):
                breakpoint()
            if os.environ.get("DEBUG_MODE", "0") == "1":
                torch.distributed.barrier()
            if kv_comm.world_size == 1:
                dq_buffer_new = all_to_all_4D(dq_buffer, 1, 2, False, False)
                dk_buffer_new = all_to_all_4D(dk_buffer, 1, 2, False, False)
                dv_buffer_new = all_to_all_4D(dv_buffer, 1, 2, False, False)
                return dq_buffer_new, dk_buffer_new, dv_buffer_new
            dq = dq_buffer.to(torch.float32)
            dk = dk_buffer.to(torch.float32)
            dv = dv_buffer.to(torch.float32)
        else:
            if step <= kv_comm.rank:
                k0 = k[:, :block_seq_len]
                v0 = v[:, :block_seq_len]
                backward(dout, q, k0, v0, out, softmax_lse, causal=False)
                dq += dq_buffer
            else:
                backward(dout1, q1, k, v, out1, softmax_lse1, causal=False)
                # always use the first half in dq_buffer.
                dq[:, block_seq_len:] += dq_buffer[:, :block_seq_len]

            d_kv_comm.wait()
            dk_comm_buffer, dv_comm_buffer = dk, dv
            dk, dv = next_dk, next_dv

            if step <= kv_comm.rank:
                dk[:, :block_seq_len] += dk_buffer[:, :block_seq_len]
                dv[:, :block_seq_len] += dv_buffer[:, :block_seq_len]
            else:
                dk += dk_buffer
                dv += dv_buffer

        if step + 1 != kv_comm.world_size:
            kv_comm.wait()
            k = next_k
            v = next_v

        next_dk = d_kv_comm.send_recv(dk, dk_comm_buffer)
        next_dv = d_kv_comm.send_recv(dv, dv_comm_buffer)
        d_kv_comm.commit()

    d_kv_comm.wait()

    # if k.shape[2] != q.shape[2]:
    #     bs, slen, nqh, hdim = q.shape
    #     return dq.to(q.dtype), next_dk.view(bs, slen, k.shape[2], (q.shape[2]//k.shape[2]), hdim).sum(dim=3).to(q.dtype), next_dv.view(bs, slen, v.shape[2], (q.shape[2]//v.shape[2]), hdim).sum(dim=3).to(q.dtype)
    # else:
    dq = dq.to(q.dtype)
    next_dk = next_dk.to(q.dtype)
    next_dv = next_dv.to(q.dtype)

    dq_new = all_to_all_4D(dq, 1, 2, False, False)
    next_dk_new = all_to_all_4D(next_dk, 1, 2, False, False)
    next_dv_new = all_to_all_4D(next_dv, 1, 2, False, False)

    return dq_new, next_dk_new, next_dv_new


class ZigZagRingFlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_softmax,
        group,
        attn_type,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        assert alibi_slopes is None
        k = k.contiguous()
        v = v.contiguous()

        # global variables
        global process_group
        import sys

        current_module = sys.modules[__name__]
        current_module.process_group = group
        current_module.attn_type = attn_type
        current_module.alibi_slopes = alibi_slopes
        current_module.window_size = window_size

        out, softmax_lse = zigzag_ring_flash_attn_forward(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            softcap=softcap,
            deterministic=False,
        )

        # out, softmax_lse = zigzag_ring_flash_attn_forward_op(out_temp, softmax_lse_temp)

        # this should be out_padded
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.group = group
        ctx.attn_type = attn_type
        return out if not return_softmax else (out, softmax_lse, None)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse = ctx.saved_tensors
        dq, dk, dv = zigzag_ring_flash_attn_backward(
            ctx.group,
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            softmax_scale=ctx.softmax_scale,
            dropout_p=ctx.dropout_p,
            causal=ctx.causal,
            window_size=ctx.window_size,
            softcap=ctx.softcap,
            alibi_slopes=ctx.alibi_slopes,
            deterministic=ctx.deterministic,
            attn_type=ctx.attn_type,
        )
        if (
            os.environ.get("DEBUG_MODE", "0") == "1"
            and torch.distributed.get_rank() == 0
        ):
            breakpoint()
        if os.environ.get("DEBUG_MODE", "0") == "1":
            torch.distributed.barrier()
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None


def zigzag_ring_flash_attn_qkvpacked_func(
    qkv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
    attn_type: AttnType = AttnType.FA,
):
    return ZigZagRingFlashAttnFunc.apply(
        qkv[:, :, 0],
        qkv[:, :, 1],
        qkv[:, :, 2],
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
        attn_type,
    )


def zigzag_ring_flash_attn_kvpacked_func(
    q,
    kv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
    attn_type: AttnType = AttnType.FA,
):
    return ZigZagRingFlashAttnFunc.apply(
        q,
        kv[:, :, 0],
        kv[:, :, 1],
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
        attn_type,
    )


def zigzag_ring_flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
    attn_type: AttnType = AttnType.FA,
    attn_processor=None,
):
    return ZigZagRingFlashAttnFunc.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
        attn_type,
    )
