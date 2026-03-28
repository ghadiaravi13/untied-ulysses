from typing import Any, Tuple

import torch

import torch.distributed as dist
from torch import Tensor

from yunchang.comm.all_to_all import SeqAllToAll4D, SeqAllToAll5D
from yunchang.globals import HAS_SPARSE_SAGE_ATTENTION, PROCESS_GROUP
from yunchang.kernels import AttnType
from .utils import RING_IMPL_DICT, RING_IMPL_QKVPACKED_DICT


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        torch.unsqueeze(x, dim=3)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class LongContextAttention(torch.nn.Module):
    """Initialization.

    Arguments:
        ulysses_pg (ProcessGroup): ulysses process group
        ring_pg (ProcessGroup): ring process group
        scatter_idx (int): scatter_idx for all2all comm
        gather_idx (int): gather_idx for all2all comm
        use_sync (bool): whether to synchronize after all-to-all
    """

    def __init__(
        self,
        scatter_idx: int = 2,
        gather_idx: int = 1,
        ring_impl_type: str = "basic",
        use_pack_qkv: bool = False,
        use_sync: bool = False,
        attn_type: AttnType = AttnType.FA,
        attn_processor: torch.nn.Module = None,
    ) -> None:

        super(LongContextAttention, self).__init__()
        self.ring_pg = PROCESS_GROUP.RING_PG
        self.ulysses_pg = PROCESS_GROUP.ULYSSES_PG

        self.use_pack_qkv = use_pack_qkv
        self.use_sync = use_sync
        self.attn_type = attn_type
        assert (
            self.ulysses_pg is not None or self.ring_pg is not None
        ), f"use set_seq_parallel_pg() first. Now ulysses pg {self.ulysses_pg} and ring pg {self.ring_pg}"
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx
        self.attn_processor = attn_processor
        self.ring_attn_fn = RING_IMPL_DICT[ring_impl_type]
        self.ring_impl_type = ring_impl_type

        if HAS_SPARSE_SAGE_ATTENTION:
            from spas_sage_attn.autotune import SparseAttentionMeansim

            if (
                isinstance(attn_processor, SparseAttentionMeansim)
                and dist.get_world_size(self.ring_pg) > 1
            ):
                raise RuntimeError(
                    "Sparse Sage attention does not support ring degree > 1."
                )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        *args: Any,
    ) -> Tensor:
        """forward

        Arguments:
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            args: other args

        Returns:
            * output (Tensor): context output
        """

        # 3 X (bs, seq_len/N, head_cnt, head_size) -> 3 X (bs, seq_len, head_cnt/N, head_size)
        # scatter 2, gather 1
        if self.use_pack_qkv:
            # dist expects qkv in (b, h, s, d) format, so we need to transpose them here.
            if "dist" in self.ring_impl_type:
                print(
                    f"[DEBUG] Rank {dist.get_rank()}: About to call SeqAllToAll4D.apply for dist",
                    flush=True,
                )
                # (3*bs, seq_len/N, head_cnt, head_size)
                bs, head_cnt, seq_len, head_size = query.shape
                if key.shape[1] != query.shape[1]:
                    query = query.reshape(-1, key.shape[1], seq_len, head_size)
                qkv = torch.cat([query, key, value]).contiguous()
                q_chunks = query.shape[0]
                # (3*bs, seq_len, head_cnt/N, head_size)
                qkv = SeqAllToAll4D.apply(
                    self.ulysses_pg,
                    qkv,
                    self.gather_idx,
                    self.scatter_idx,
                    self.use_sync,
                )
                qkv = torch.chunk(qkv, q_chunks + 2, dim=0)

                q = (
                    torch.stack(qkv[:q_chunks], dim=0).reshape(
                        bs, head_cnt, seq_len, head_size
                    )
                    if q_chunks > 1
                    else qkv[q_chunks - 1]
                )
                k = qkv[q_chunks]
                v = qkv[q_chunks + 1]
                assert (
                    q.shape[1] % k.shape[1] == 0
                ), f"q.shape[1] {q.shape[1]} must be divisible by k.shape[1] {k.shape[1]}"
                out = self.ring_attn_fn(
                    q,
                    k,
                    v,
                    dropout_p=dropout_p,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=window_size,
                    softcap=softcap,
                    alibi_slopes=alibi_slopes,
                    deterministic=deterministic,
                    return_attn_probs=return_attn_probs,
                    group=self.ring_pg,
                    attn_type=self.attn_type,
                    attn_processor=self.attn_processor,
                )

                if type(out) == tuple:
                    context_layer, _, _ = out
                else:
                    context_layer = out

                # (bs, head_cnt/N, seq_len, head_size) -> (bs, head_cnt, seq_len/N, head_size)
                # scatter 2, gather 1
                print(
                    f"[DEBUG] Rank {dist.get_rank()}: About to call SeqAllToAll4D.apply ",
                    flush=True,
                )
                output = SeqAllToAll4D.apply(
                    self.ulysses_pg,
                    context_layer,
                    self.scatter_idx,
                    self.gather_idx,
                    self.use_sync,
                )
                print(
                    f"[DEBUG] Rank {dist.get_rank()}: SeqAllToAll4D.apply completed {output.shape}",
                    flush=True,
                )
                # out e.g., [s/p::h]
                return output

            # zigzag expects qkv in (b, s, h, d) format
            else:
                # (3*bs, seq_len/N, head_cnt, head_size)
                bs, seq_len, head_cnt, head_size = query.shape
                if key.shape[2] != query.shape[2]:
                    query = (
                        query.transpose(1, 2)
                        .reshape(-1, key.shape[2], seq_len, head_size)
                        .transpose(1, 2)
                    )
                qkv = torch.cat([query, key, value]).contiguous()

                q_chunks = query.shape[0]
                # (3*bs, seq_len, head_cnt/N, head_size)
                qkv = SeqAllToAll4D.apply(
                    self.ulysses_pg,
                    qkv,
                    self.scatter_idx,
                    self.gather_idx,
                    self.use_sync,
                )
                qkv = torch.chunk(qkv, q_chunks + 2, dim=0)

                q = (
                    torch.stack(qkv[:q_chunks], dim=0)
                    .transpose(1, 2)
                    .reshape(bs, head_cnt, seq_len, head_size)
                    .transpose(1, 2)
                    if q_chunks > 1
                    else qkv[q_chunks - 1]
                )
                k = qkv[q_chunks]
                v = qkv[q_chunks + 1]
                assert (
                    q.shape[2] % k.shape[2] == 0
                ), f"q.shape[2] {q.shape[2]} must be divisible by k.shape[2] {k.shape[2]}"
                out = self.ring_attn_fn(
                    q,
                    k,
                    v,
                    dropout_p=dropout_p,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=window_size,
                    softcap=softcap,
                    alibi_slopes=alibi_slopes,
                    deterministic=deterministic,
                    return_attn_probs=return_attn_probs,
                    group=self.ring_pg,
                    attn_type=self.attn_type,
                    attn_processor=self.attn_processor,
                )

        else:
            world_size = dist.get_world_size(self.ulysses_pg)

            if key.shape[self.scatter_idx] < world_size:
                assert (
                    world_size % key.shape[self.scatter_idx] == 0
                ), f"world_size {world_size} must be divisible by key head count {key.shape[self.scatter_idx]}"
                key = repeat_kv(key, world_size // key.shape[self.scatter_idx])
                value = repeat_kv(value, world_size // value.shape[self.scatter_idx])

            out = self.ring_attn_fn(
                query,
                key,
                value,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                softcap=softcap,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=return_attn_probs,
                group=self.ring_pg,
                attn_type=self.attn_type,
                attn_processor=self.attn_processor,
            )

        if type(out) == tuple:
            context_layer, _, _ = out
        else:
            context_layer = out

        # (bs, seq_len, head_cnt/N, head_size) -> (bs, seq_len/N, head_cnt, head_size)
        # scatter 1, gather 2

        output = context_layer

        return output


class LongContextAttentionQKVPacked(torch.nn.Module):
    """Initialization.

    Arguments:
        ulysses_pg (ProcessGroup): ulysses process group
        ring_pg (ProcessGroup): ring process group
        scatter_idx (int): scatter_idx for all2all comm
        gather_idx (int): gather_idx for all2all comm
        use_sync (bool): whether to synchronize after all-to-all
    """

    def __init__(
        self,
        scatter_idx: int = 3,
        gather_idx: int = 1,
        ring_impl_type: str = "basic",
        use_sync: bool = False,
        attn_type: AttnType = AttnType.FA,
    ) -> None:

        super(LongContextAttentionQKVPacked, self).__init__()

        self.ring_pg = PROCESS_GROUP.RING_PG
        self.ulysses_pg = PROCESS_GROUP.ULYSSES_PG

        assert (
            self.ulysses_pg is not None or self.ring_pg is not None
        ), f"use set_seq_parallel_pg() first. Now ulysses pg {self.ulysses_pg} and ring pg {self.ring_pg}"
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx
        self.use_sync = use_sync
        self.ring_attn_fn = RING_IMPL_QKVPACKED_DICT[ring_impl_type]
        self.attn_type = attn_type

    def forward(
        self,
        qkv,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        *args: Any,
    ) -> Tensor:
        """forward

        Arguments:
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            args: other args

        Returns:
            * output (Tensor): context output
        """

        world_size = dist.get_world_size(self.ulysses_pg)

        if world_size > 1:
            qkv = SeqAllToAll5D.apply(
                self.ulysses_pg, qkv, self.scatter_idx, self.gather_idx, self.use_sync
            )

        out = self.ring_attn_fn(
            qkv,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=return_attn_probs,
            group=self.ring_pg,
            attn_type=self.attn_type,
        )

        if type(out) == tuple:
            out = out[0]

        # (bs, seq_len, head_cnt/N, head_size) -> (bs, seq_len/N, head_cnt, head_size)
        # scatter 1, gather 2

        if world_size > 1:
            out = SeqAllToAll4D.apply(
                self.ulysses_pg,
                out,
                self.gather_idx,
                self.scatter_idx - 1,
                self.use_sync,
            )
        return out
