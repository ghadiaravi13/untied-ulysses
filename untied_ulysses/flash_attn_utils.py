"""
Flash Attention utilities with Ring Attention support.

This module provides ring attention implementations that work with both
FlashAttention 2 and FlashAttention 3 backends.
"""

from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

try:
    from flash_attn.flash_attn_interface import (
        _flash_attn_backward as _flash_attn_backward_fa2,
        _flash_attn_forward as _flash_attn_forward_fa2,
    )

    HAS_FA2 = True
except ImportError:
    _flash_attn_forward_fa2 = None
    _flash_attn_backward_fa2 = None
    HAS_FA2 = False

try:
    from flash_attn_interface import (
        _flash_attn_backward as _flash_attn_backward_fa3,
        _flash_attn_forward as _flash_attn_forward_fa3,
    )

    HAS_FA3 = True
except ImportError:
    _flash_attn_forward_fa3 = None
    _flash_attn_backward_fa3 = None
    HAS_FA3 = False


def flash_attn_forward_fa2(
    q,
    k,
    v,
    dropout_p,
    softmax_scale,
    causal=False,
    window_size_left=-1,
    window_size_right=-1,
    softcap=0.0,
    alibi_slopes=None,
    return_softmax=False,
):
    """Unified FA2 forward wrapper."""
    return _flash_attn_forward_fa2(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        softcap,
        alibi_slopes,
        return_softmax,
    )


def flash_attn_forward_fa3(
    q,
    k,
    v,
    dropout_p,
    softmax_scale,
    causal=False,
    window_size_left=-1,
    window_size_right=-1,
    softcap=0.0,
    alibi_slopes=None,
    return_softmax=False,
):
    """Unified FA3 forward wrapper. Note: FA3 doesn't support dropout or alibi."""
    result = _flash_attn_forward_fa3(
        q,
        k,
        v,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        softcap=softcap,
    )
    # FA3 returns (out, lse, S_dmask, rng_state) or just (out, lse)
    out, lse = result[0], result[1]
    return out, lse


def flash_attn_backward_fa2(
    dout,
    q,
    k,
    v,
    out,
    lse,
    dq,
    dk,
    dv,
    dropout_p,
    softmax_scale,
    causal,
    window_size_left,
    window_size_right,
    softcap,
    alibi_slopes,
    deterministic,
    rng_state=None,
):
    """Unified FA2 backward wrapper."""
    return _flash_attn_backward_fa2(
        dout,
        q,
        k,
        v,
        out,
        lse,
        dq,
        dk,
        dv,
        dropout_p,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        softcap,
        alibi_slopes,
        deterministic,
        rng_state,
    )


def flash_attn_backward_fa3(
    dout,
    q,
    k,
    v,
    out,
    lse,
    dq,
    dk,
    dv,
    dropout_p,
    softmax_scale,
    causal,
    window_size_left,
    window_size_right,
    softcap,
    alibi_slopes,
    deterministic,
    rng_state=None,
):
    """Unified FA3 backward wrapper. Note: FA3 doesn't support dropout or alibi."""
    _flash_attn_backward_fa3(
        dout,
        q,
        k,
        v,
        out,
        lse,
        dq=dq,
        dk=dk,
        dv=dv,
        softmax_scale=softmax_scale,
        is_causal=causal,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        softcap=softcap,
        deterministic=deterministic,
    )


def select_flash_attn_impl(attn_type: str, stage: str = "fwd-only"):
    """
    Select flash attention implementation based on type and stage.

    Args:
        attn_type: "fa2" or "fa3"
        stage: "fwd-only" or "bwd-only"

    Returns:
        The appropriate forward or backward function (with unified API)
    """
    if attn_type == "fa2":
        assert HAS_FA2, "FlashAttention 2 is not available"
        return (
            flash_attn_forward_fa2 if stage == "fwd-only" else flash_attn_backward_fa2
        )
    elif attn_type == "fa3":
        assert HAS_FA3, "FlashAttention 3 is not available"
        return (
            flash_attn_forward_fa3 if stage == "fwd-only" else flash_attn_backward_fa3
        )
    else:
        raise ValueError(f"Unknown attn_type: {attn_type}. Use 'fa2' or 'fa3'.")


class RingComm:
    """
    Ring communication helper for P2P tensor exchange.

    Implements a ring topology for passing tensors. Direction depends on pass_kv:
    - pass_kv=True: send to (rank+1), recv from (rank-1) - for passing K/V forward
    - pass_kv=False: send to (rank-1), recv from (rank+1) - for passing Q backward
    """

    def __init__(
        self,
        process_group: dist.ProcessGroup,
        recv_buffer: Optional[torch.Tensor] = None,
        pass_kv: bool = True,
    ):
        self._process_group = process_group
        self._ops = []
        self._reqs = None
        self.recv_buffer = recv_buffer

        self.rank = dist.get_rank(self._process_group)
        self.world_size = dist.get_world_size(self._process_group)

        self.send_rank = (self.rank + 1) % self.world_size
        self.recv_rank = (self.rank - 1) % self.world_size

        # Convert to global ranks for the process group
        if process_group is not None:
            self.send_rank = dist.get_global_rank(self._process_group, self.send_rank)
            self.recv_rank = dist.get_global_rank(self._process_group, self.recv_rank)

    def send_recv(
        self,
        to_send: torch.Tensor,
        recv_tensor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Initiate async send/recv of tensor around the ring.

        Args:
            to_send: Tensor to send to next rank
            recv_tensor: Optional pre-allocated receive buffer

        Returns:
            The receive buffer (will contain data after wait())
        """
        if recv_tensor is None:
            res = (
                self.recv_buffer
                if self.recv_buffer is not None
                else torch.empty_like(to_send)
            )
        else:
            res = recv_tensor

        send_op = dist.P2POp(
            dist.isend, to_send, self.send_rank, group=self._process_group
        )
        recv_op = dist.P2POp(dist.irecv, res, self.recv_rank, group=self._process_group)
        self._ops.append(send_op)
        self._ops.append(recv_op)

        return res

    def commit(self):
        """Start the P2P operations."""
        if self._reqs is not None:
            raise RuntimeError("commit called twice")
        self._reqs = dist.batch_isend_irecv(self._ops)

    def wait(self):
        """Wait for all P2P operations to complete."""
        if self._reqs is None:
            raise RuntimeError("wait called before commit")

        for req in self._reqs:
            req.wait()

        dist.barrier(group=self._process_group)

        self._reqs.clear()
        self._reqs = None
        self._ops.clear()
        self._ops = []


@torch.jit.script
def _update_out_and_lse(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    JIT-compiled LSE update using sigmoid for numerical stability.

    Uses the identity: softmax(a, b) = sigmoid(a-b) * a + sigmoid(b-a) * b
    which avoids explicit exp() calls that can overflow.

    Reference: https://github.com/zhuzilin/ring-flash-attention/pull/34
    """
    block_out = block_out.to(torch.float32)
    if block_lse.ndim == 3:
        block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)

    # Numerically stable update using sigmoid
    out = out - F.sigmoid(block_lse - lse) * (out - block_out)
    lse = lse - F.logsigmoid(lse - block_lse)

    return out, lse


def update_out_and_lse(
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    slice_: Optional[Tuple] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Merge attention outputs using log-sum-exp for numerical stability.

    Args:
        out: Accumulated output tensor, or None if first block
        lse: Accumulated log-sum-exp, or None if first block
        block_out: Output from current attention block
        block_lse: Log-sum-exp from current attention block
        slice_: Optional slice to apply for partial updates

    Returns:
        Updated (out, lse) tuple
    """
    if out is None:
        if slice_ is not None:
            raise RuntimeError("first update_out_and_lse should not pass slice_ args")
        out = block_out.to(torch.float32)
        lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
    elif slice_ is not None:
        slice_out, slice_lse = out[slice_], lse[slice_]
        slice_out, slice_lse = _update_out_and_lse(
            slice_out, slice_lse, block_out, block_lse
        )
        out[slice_], lse[slice_] = slice_out, slice_lse
    else:
        out, lse = _update_out_and_lse(out, lse, block_out, block_lse)

    return out, lse


# ============================================================================
# Zigzag Ring Flash Attention Forward
# ============================================================================


def zigzag_ring_flash_attn_forward(
    process_group: dist.ProcessGroup,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    dropout_p: float = 0.0,
    causal: bool = True,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    attn_type: str = "fa2",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Zigzag ring attention forward pass.

    Implements ring attention with zigzag scheduling for causal attention.
    Query tensors are passed around the ring while K/V stay local.

    Args:
        process_group: Ring process group for communication
        q: Query tensor, shape (batch, seq_len, heads, head_dim)
        k: Key tensor, shape (batch, seq_len, heads, head_dim)
        v: Value tensor, shape (batch, seq_len, heads, head_dim)
        softmax_scale: Softmax scaling factor
        dropout_p: Dropout probability
        causal: Must be True for zigzag ring attention
        window_size: Sliding window size (left, right)
        softcap: Soft cap for attention logits
        alibi_slopes: ALiBi slopes if using ALiBi
        attn_type: "fa2" or "fa3"

    Returns:
        Tuple of (output, lse) where:
        - output: Attention output, same shape as q
        - lse: Log-sum-exp, shape (batch, heads, seq_len)
    """
    assert causal, "Zigzag ring attention requires causal=True"

    comm = RingComm(process_group, recv_buffer=torch.empty_like(q), pass_kv=False)

    block_seq_len = q.shape[1] // 2
    k_first_half = k[:, :block_seq_len]
    v_first_half = v[:, :block_seq_len]

    out = None
    lse = None
    next_q = None

    def forward(q_in, k_in, v_in, is_causal):
        """Inner attention forward."""
        fn = select_flash_attn_impl(attn_type, stage="fwd-only")
        out = fn(
            q_in,
            k_in,
            v_in,
            dropout_p,
            softmax_scale,
            causal=is_causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=(dropout_p > 0),
        )
        block_out, block_lse = out[0], out[1]
        return block_out, block_lse

    for step in range(comm.world_size):
        # Start async communication for next step (except last)
        if step + 1 < comm.world_size:
            next_q = comm.send_recv(q)
            comm.commit()

        # Compute attention for this step
        if step == 0:
            # First step: full causal attention with local Q, K, V
            block_out, block_lse = forward(q, k, v, is_causal=True)
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        elif step <= comm.rank:
            # Q comes from a rank earlier in the global sequence
            # Only the second half of Q needs to attend to our K, V
            q_second_half = q[:, block_seq_len:]
            block_out, block_lse = forward(q_second_half, k, v, is_causal=False)
            out, lse = update_out_and_lse(
                out,
                lse,
                block_out,
                block_lse,
                slice_=(slice(None), slice(block_seq_len, None)),
            )
        else:
            # Q comes from a rank later in the global sequence
            # Full Q attends to first half of our K, V
            block_out, block_lse = forward(
                q, k_first_half, v_first_half, is_causal=False
            )
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)

        # Wait for communication and update q
        if step + 1 < comm.world_size:
            comm.wait()
            q = next_q

    del comm, next_q, k_first_half, v_first_half

    out = out.to(q.dtype)
    lse = lse.squeeze(dim=-1).transpose(1, 2)

    return out, lse


def zigzag_ring_flash_attn_backward(
    process_group: dist.ProcessGroup,
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    softmax_scale: float,
    dropout_p: float = 0.0,
    causal: bool = True,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    attn_type: str = "fa2",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Zigzag ring attention backward pass.

    Passes K/V around the ring while accumulating gradients.

    Args:
        process_group: Ring process group for communication
        dout: Gradient of output, same shape as out
        q, k, v: Query, key, value tensors from forward
        out: Output from forward
        softmax_lse: Log-sum-exp from forward, shape (batch, heads, seq_len)
        softmax_scale: Softmax scaling factor
        dropout_p: Dropout probability
        causal: Must be True for zigzag ring attention
        window_size: Sliding window size
        softcap: Soft cap for attention logits
        alibi_slopes: ALiBi slopes
        deterministic: Use deterministic algorithms
        attn_type: "fa2" or "fa3"

    Returns:
        Tuple of (dq, dk, dv) gradients
    """
    assert causal, "Zigzag ring attention requires causal=True"

    kv_comm = RingComm(process_group, pass_kv=True)
    dkv_comm = RingComm(process_group, pass_kv=True)

    # Prepare second-half slices for zigzag pattern
    block_seq_len = q.shape[1] // 2
    dout_second = dout[:, block_seq_len:]
    q_second = q[:, block_seq_len:]
    out_second = out[:, block_seq_len:]
    lse_second = softmax_lse[:, :, block_seq_len:].contiguous()

    dq_buffer = torch.empty_like(q)
    dk_buffer = torch.empty_like(k)
    dv_buffer = torch.empty_like(v)

    dq, dk, dv = None, None, None
    next_dk, next_dv = None, None
    next_k, next_v = None, None
    dk_comm_buffer, dv_comm_buffer = None, None

    def backward(dout_in, q_in, k_in, v_in, out_in, lse_in, is_causal):
        """Inner attention backward."""
        seqlen_q = q_in.shape[1]
        seqlen_kv = k_in.shape[1]
        fn = select_flash_attn_impl(attn_type, stage="bwd-only")

        fn(
            dout_in,
            q_in,
            k_in,
            v_in,
            out_in,
            lse_in,
            dq_buffer[:, :seqlen_q],
            dk_buffer[:, :seqlen_kv],
            dv_buffer[:, :seqlen_kv],
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=is_causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            rng_state=None,
        )

    for step in range(kv_comm.world_size):
        # Start async KV communication for next step (except last)
        if step + 1 < kv_comm.world_size:
            next_k = kv_comm.send_recv(k)
            next_v = kv_comm.send_recv(v)
            kv_comm.commit()

        if step == 0:
            # First step: full causal backward
            backward(
                dout.contiguous(),
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                out.contiguous(),
                softmax_lse.contiguous(),
                is_causal=True,
            )

            # Single GPU case - return immediately
            if kv_comm.world_size == 1:
                return dq_buffer, dk_buffer, dv_buffer

            # Initialize accumulated gradients
            dq = dq_buffer.to(torch.float32)
            dk = dk_buffer.to(torch.float32)
            dv = dv_buffer.to(torch.float32)
        else:
            if step <= kv_comm.rank:
                # K/V from rank earlier in sequence - use first half of K/V
                k_first = k[:, :block_seq_len]
                v_first = v[:, :block_seq_len]
                backward(dout, q, k_first, v_first, out, softmax_lse, is_causal=False)
                dq += dq_buffer
            else:
                # K/V from rank later in sequence - use second half of Q
                backward(
                    dout_second, q_second, k, v, out_second, lse_second, is_causal=False
                )
                # Gradient goes to second half of dq
                dq[:, block_seq_len:] += dq_buffer[:, :block_seq_len]

            # Wait for gradient communication from previous step
            dkv_comm.wait()
            dk_comm_buffer, dv_comm_buffer = dk, dv
            dk, dv = next_dk, next_dv

            # Accumulate gradients based on position
            if step <= kv_comm.rank:
                dk[:, :block_seq_len] += dk_buffer[:, :block_seq_len]
                dv[:, :block_seq_len] += dv_buffer[:, :block_seq_len]
            else:
                dk += dk_buffer
                dv += dv_buffer

        # Wait for KV communication
        if step + 1 < kv_comm.world_size:
            kv_comm.wait()
            k = next_k
            v = next_v

        # Start async gradient communication
        next_dk = dkv_comm.send_recv(dk, dk_comm_buffer)
        next_dv = dkv_comm.send_recv(dv, dv_comm_buffer)
        dkv_comm.commit()

    dkv_comm.wait()

    orig_dtype = q.dtype

    return (
        dq.to(orig_dtype).detach(),
        next_dk.to(orig_dtype).detach(),
        next_dv.to(orig_dtype).detach(),
    )


class ZigzagRingFlashAttnFunc(torch.autograd.Function):
    """Autograd function for zigzag ring flash attention."""

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        process_group: dist.ProcessGroup,
        softmax_scale: float,
        dropout_p: float,
        causal: bool,
        window_size: Tuple[int, int],
        softcap: float,
        alibi_slopes: Optional[torch.Tensor],
        deterministic: bool,
        attn_type: str,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        out, softmax_lse = zigzag_ring_flash_attn_forward(
            process_group,
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            attn_type=attn_type,
        )

        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.process_group = process_group
        ctx.softmax_scale = softmax_scale
        ctx.dropout_p = dropout_p
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.attn_type = attn_type

        return out, softmax_lse

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse = ctx.saved_tensors

        dq, dk, dv = zigzag_ring_flash_attn_backward(
            ctx.process_group,
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

        return dq, dk, dv, None, None, None, None, None, None, None, None, None


def zigzag_ring_flash_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    process_group: dist.ProcessGroup,
    softmax_scale: Optional[float] = None,
    dropout_p: float = 0.0,
    causal: bool = True,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    attn_type: str = "fa2",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    High-level API for zigzag ring flash attention with autograd support.

    Args:
        q: Query tensor, shape (batch, seq_len, heads, head_dim)
        k: Key tensor, shape (batch, seq_len, heads, head_dim)
        v: Value tensor, shape (batch, seq_len, heads, head_dim)
        process_group: Ring process group
        softmax_scale: Softmax scaling factor (default: 1/sqrt(head_dim))
        dropout_p: Dropout probability
        causal: Use causal attention mask
        window_size: Sliding window size (left, right)
        softcap: Soft cap for attention logits
        alibi_slopes: ALiBi slopes
        deterministic: Use deterministic algorithms
        attn_type: "fa2" or "fa3"

    Returns:
        Tuple of (output, lse)
    """
    pg = process_group

    if pg is None or dist.get_world_size(pg) == 1:
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        fn = select_flash_attn_impl(attn_type, stage="fwd-only")
        out, lse = fn(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=False,
        )
        return out.to(q.dtype), lse

    return ZigzagRingFlashAttnFunc.apply(
        q,
        k,
        v,
        pg,
        softmax_scale,
        dropout_p,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        attn_type,
    )
