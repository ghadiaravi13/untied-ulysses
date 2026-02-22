# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.


import os

import sys
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

from yunchang import LongContextAttention
from yunchang.comm.all_to_all import SeqAllToAll4D
from yunchang.globals import PROCESS_GROUP

from torchtitan.components.tokenizer import Tokenizer
from torchtitan.config_manager import JobConfig
from torchtitan.models.attention import build_attention, init_attention_mask
from torchtitan.protocols.train_spec import BaseModelArgs, ModelProtocol

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "untied_ulysses")
)
import torch.distributed as dist
from fully_fused_attn import UpipeAttention

from liger_kernel.transformers.fused_linear_cross_entropy import (
    LigerFusedLinearCrossEntropyLoss,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import math

from collections import namedtuple

from patch_torch_files.ulysses_sp import TiledMLP

from yunchang.kernels import AttnType

from torchtitan.components.tiled_rms_norm import tiled_rms_norm_forward

from torchtitan.tools.logging import logger

DEBUG_MODE = os.environ.get("DEBUG_MODE", "0") == "1"


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


@dataclass
class TransformerModelArgs(BaseModelArgs):
    dim: int = 4096
    n_layers: int = 1
    n_heads: int = 32
    n_kv_heads: int | None = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: float | None = None
    norm_eps: float = 1e-5
    rope_theta: float = 10000
    head_dim: int = 128

    max_seq_len: int = 2048
    batch_size: int = 1
    # If `True`, then each transformer block init uses its layer ID, and if
    # `False`, each uses the total number of transformer blocks
    depth_init: bool = True

    use_flex_attn: bool = False
    attn_mask_type: str = "causal"
    eos_id: int = 0
    attn_impl: str = "torch_ring"
    ring_comm_heads: str = "mha"
    use_pipelined_ff: bool = False

    def update_from_config(self, job_config: JobConfig, tokenizer: Tokenizer) -> None:
        self.vocab_size = tokenizer.n_words
        self.max_seq_len = job_config.training.seq_len
        self.batch_size = job_config.training.batch_size
        self.eos_id = tokenizer.eos_id
        self.attn_impl = job_config.model.attn_impl
        self.ring_comm_heads = job_config.model.ring_comm_heads
        self.chunked_loss = job_config.training.chunked_loss

        if job_config.activation_checkpoint.mode == "selective" and self.use_flex_attn:
            raise ValueError(
                "FlexAttention is not compatible with selective AC yet. "
                "See https://github.com/pytorch/pytorch/issues/147879"
            )

        if job_config.parallelism.context_parallel_degree > 1 and self.use_flex_attn:
            raise ValueError(
                "FlexAttention is not compatible with CP yet. "
                "We are still working on this."
            )

    def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
        nparams = sum(p.numel() for p in model.parameters())
        nparams_embedding = sum(
            sum(p.numel() for p in m.parameters())
            for m in model.children()
            if isinstance(m, nn.Embedding)
        )

        l, h, q, t = (
            self.n_layers,
            self.n_heads,
            self.head_dim if self.head_dim is not None else self.dim // self.n_heads,
            seq_len,
        )
        # Reasoning behind the factor of 12 for the self-attention part of the formula:
        # 1. each self-attention has 2 matmul in the forward and 4 in the backward (6)
        # 2. the flash attention does 1 more matmul recomputation in the backward
        #    but recomputation should not be counted in calculating MFU           (+0)
        # 3. each matmul performs 1 multiplication and 1 addition                 (*2)
        # 4. we follow the convention and do not account for sparsity in causal attention
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t

        return nparams, num_flops_per_token


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float | None): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    freqs = 1.0 / (
        theta
        ** (
            torch.arange(0, dim, 2, device=torch.cuda.current_device())[
                : (dim // 2)
            ].float()
            / dim
        )
    )
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    The input freqs_cis tensor is assumed to be of shape (max_seqlen, dim),
    and the first seqlen elements will be sliced, but dim must match x.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.
    """
    ndim = x.ndim
    assert ndim > 1
    seqlen = x.shape[1]
    freqs_cis = freqs_cis[0:seqlen]
    assert freqs_cis.shape == (
        seqlen,
        x.shape[-1],
    ), f"freqs_cis.shape: {freqs_cis.shape} != (seqlen, x.shape[-1]): {(seqlen, x.shape[-1])}"
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


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


def repeat_wkv(
    w: torch.Tensor, n_kv_heads: int, head_dim: int, n_rep: int
) -> torch.Tensor:
    """Repeat the weight tensor for the number of times specified by n_rep"""
    inp_dim, out_dim = w.shape
    if n_rep == 1:
        return w
    return (
        w.unsqueeze(0)
        .view(-1, head_dim, out_dim)
        .repeat_interleave(n_rep, dim=0)
        .reshape(inp_dim * n_rep, out_dim)
    )


class Attention(nn.Module):
    """
    Multi-head attention module.

    Args:
        model_args (TransformerModelArgs): Model configuration arguments.

    Attributes:
        n_kv_heads (int): Number of key and value heads.
        n_heads (int): Number of query heads.
        n_rep (int): Number of repetitions for local heads.
        head_dim (int): Dimension size of each attention head.
        wq (Linear): Linear transformation for queries.
        wk (Linear): Linear transformation for keys.
        wv (Linear): Linear transformation for values.
        wo (Linear): Linear transformation for output.

    """

    def __init__(
        self,
        model_args: TransformerModelArgs,
        layer_id: int = -1,
        offload_stream: torch.cuda.Stream = None,
        fetch_stream: torch.cuda.Stream = None,
        two_streams: list[torch.cuda.Stream] = None,
    ):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.model_args = model_args
        self.n_kv_heads = (
            model_args.n_heads
            if model_args.n_kv_heads is None
            else model_args.n_kv_heads
        )
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = model_args.head_dim if model_args.head_dim is not None else model_args.dim // model_args.n_heads

        self.wq = nn.Linear(
            model_args.dim, model_args.n_heads * self.head_dim, bias=False
        )
        if "upipe" in model_args.attn_impl and "mha" in model_args.ring_comm_heads:
            self.wk = nn.Linear(
                model_args.dim, self.n_heads * self.head_dim, bias=False
            )
            self.wv = nn.Linear(
                model_args.dim, self.n_heads * self.head_dim, bias=False
            )
        else:
            self.wk = nn.Linear(
                model_args.dim, self.n_kv_heads * self.head_dim, bias=False
            )
            self.wv = nn.Linear(
                model_args.dim, self.n_kv_heads * self.head_dim, bias=False
            )
        self.wo = nn.Linear(
            model_args.n_heads * self.head_dim, model_args.dim, bias=False
        )

        self.two_streams = two_streams
        self.offload_stream = offload_stream
        self.fetch_stream = fetch_stream
        self.wq_init = False

        self.attn_impl = model_args.attn_impl
        self.ring_comm_heads = model_args.ring_comm_heads
        self.layer_id = layer_id
        self.pack_qkv = "qkvpacked" in model_args.attn_impl
        self.attn_type = (
            AttnType.FA3
            if "fa3" in model_args.attn_impl
            else (AttnType.TORCH_FLASH if "torch" in model_args.attn_impl else AttnType.FA)
        )

        if "upipe" in model_args.attn_impl:
            self.sdpa = UpipeAttention(attn_type=self.attn_type, layer_id=self.layer_id)
        elif "usp" in model_args.attn_impl:
            self.sdpa = LongContextAttention(
                ring_impl_type="zigzag",
                attn_type=self.attn_type,
                use_pack_qkv=self.pack_qkv,
            )
        elif "torch" in model_args.attn_impl:
            self.sdpa = build_attention(
                model_args.use_flex_attn, model_args.attn_mask_type
            )
        else:
            raise ValueError(
                f"Unknown attn_impl: {model_args.attn_impl}. Supported: upipe, usp, torch"
            )

    def init_weights(self, init_std: float):
        for linear in (self.wq, self.wk, self.wv):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.0001)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)
        if hasattr(self, "attention_norm"):
            self.attention_norm.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        bs, seqlen, _ = x.shape

        if "upipe" in self.attn_impl:
            if "mha" in self.ring_comm_heads:
                output = self.sdpa(
                    x,
                    self.wq.weight,
                    self.wk.weight,
                    self.wv.weight,
                    freqs_cis,
                    self.head_dim,
                    n_kv_heads=self.n_heads,
                    causal=True,
                    fused_attn_type="mha_upipe",
                )
            else:
                if self.wk.weight.shape[0] // self.head_dim < dist.get_world_size(
                    PROCESS_GROUP.ULYSSES_PG
                ):
                    self.wk.weight.data = repeat_wkv(
                        self.wk.weight.data,
                        self.n_kv_heads,
                        self.head_dim,
                        dist.get_world_size(PROCESS_GROUP.ULYSSES_PG)
                        // self.n_kv_heads,
                    )
                    self.wv.weight.data = repeat_wkv(
                        self.wv.weight.data,
                        self.n_kv_heads,
                        self.head_dim,
                        dist.get_world_size(PROCESS_GROUP.ULYSSES_PG)
                        // self.n_kv_heads,
                    )
                output = self.sdpa(
                    x,
                    self.wq.weight,
                    self.wk.weight,
                    self.wv.weight,
                    freqs_cis,
                    self.head_dim,
                    n_kv_heads=self.n_kv_heads,
                    causal=True,
                )

            # reindex output for GQA rearrangement
            final_out_idx = []
            ulysses_degree = dist.get_world_size(PROCESS_GROUP.ULYSSES_PG)
            pipe_degree = self.n_heads // ulysses_degree
            gqa_ratio = self.n_heads // self.n_kv_heads
            for stage in range(pipe_degree):
                if stage == 0 or stage // gqa_ratio != (stage - 1) // gqa_ratio:
                    stage_idx = [(stage + i) * gqa_ratio for i in range(ulysses_degree)]
                else:
                    stage_idx = [idx + 1 for idx in stage_idx]
                final_out_idx.extend(stage_idx)
            sorted_final_out_idx = torch.argsort(torch.tensor(final_out_idx))
            output = output[:, :, sorted_final_out_idx, :]
            wo_in = output.view(bs, seqlen, -1)

            result = self.wo(wo_in)

            return result
        else:
            xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

            # Use -1 instead of `n_heads` (or `n_kv_heads`) to infer the actual
            # local heads from sizes of xq, xk, and xv as TP may have sharded them
            # after the above linear ops.
            xq = xq.view(bs, seqlen, -1, self.head_dim)
            xk = xk.view(bs, seqlen, -1, self.head_dim)
            xv = xv.view(bs, seqlen, -1, self.head_dim)

            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

            if self.ring_comm_heads == "mha_kv" or "torch" in self.attn_impl:
                # repeat k/v heads if n_kv_heads < n_heads
                keys = repeat_kv(
                    xk, self.n_rep
                )  # (bs, seqlen, n_local_heads, head_dim)
                values = repeat_kv(
                    xv, self.n_rep
                )  # (bs, seqlen, n_local_heads, head_dim)
            else:
                keys = xk
                values = xv

            if "usp" in self.attn_impl:
                xq = xq
                xk = keys
                xv = values
                output = self.sdpa(xq, xk, xv, causal=True)
                assert output.requires_grad, f"output requires_grad must be True"
            else:
                xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
                xk = keys.transpose(1, 2)
                xv = values.transpose(1, 2)
                output = self.sdpa(xq, xk, xv)

                output = output.transpose(
                    1, 2
                ).contiguous()  # (bs, seqlen, n_local_heads, head_dim)

            output = output.contiguous().view(bs, seqlen, -1)
            return self.wo(output)


class FeedForward(nn.Module):
    """
    FeedForward module

    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension of the feedforward layer.
        multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
        ffn_dim_multiplier (float | None): Custom multiplier for hidden dimension. Defaults to None.

    Attributes:
        w1 (Linear): Linear transformation for the first layer.
        w2 (Linear): Linear transformation for the second layer.
        w3 (Linear): Linear transformation for the third layer.

    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: float | None,
        use_tiled_mlp: bool = False,
        use_triton_ffn: bool = False,
        triton_out_of_place: bool = True,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.use_tiled_mlp = use_tiled_mlp
        self.use_triton_ffn = use_triton_ffn
        self.triton_out_of_place = triton_out_of_place

    def forward(self, x):
        if self.use_tiled_mlp:

            def mlp_forward(self, x):
                return self.w2(F.silu(self.w1(x)) * self.w3(x))

            bs, seqlen, hidden = x.shape
            num_shards = math.ceil(seqlen * bs / hidden)
            compute_params = [self.w2.weight, self.w1.weight, self.w3.weight]
            return TiledMLP.apply(
                mlp_forward,
                self,
                x,
                num_shards,
                compute_params,
            )
        else:
            return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


class TiledRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.norm = nn.RMSNorm(dim, eps=eps)
        self.weight = self.norm.weight
        self.eps = self.norm.eps

    def forward(self, x_in):
        def rms_norm_forward(self, x):
            return self.norm(x)

        bs, seqlen, hidden = x_in.shape
        num_shards = math.ceil(seqlen * bs / hidden)
        compute_params = [self.weight]
        return TiledMLP.apply(
            rms_norm_forward,
            self,
            x_in,
            num_shards,
            compute_params,
        )

    def reset_parameters(self):
        self.norm.reset_parameters()


class TransformerBlock(nn.Module):
    """
    TransformerBlock Module

    Args:
        layer_id (int): Identifier for the layer.
        model_args (TransformerModelArgs): Model configuration arguments.

    Attributes:
        n_heads (int): Number of attention heads.
        dim (int): Dimension size of the model.
        head_dim (int): Dimension size of each attention head.
        attention (Attention): Attention module.
        feed_forward (FeedForward): FeedForward module.
        layer_id (int): Identifier for the layer.
        attention_norm (RMSNorm): Layer normalization for attention output.
        ffn_norm (RMSNorm): Layer normalization for feedforward output.

    """

    def __init__(
        self,
        layer_id: int,
        model_args: TransformerModelArgs,
        offload_stream: torch.cuda.Stream = None,
        fetch_stream: torch.cuda.Stream = None,
        two_streams: list[torch.cuda.Stream] = None,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.model_args = model_args
        self.n_heads = model_args.n_heads
        self.dim = model_args.dim
        self.attention = Attention(
            model_args, layer_id, offload_stream, fetch_stream, two_streams
        )
        self.graph_built = False
        self.attn_impl = model_args.attn_impl

        self.feed_forward = FeedForward(
            model_args.dim,
            4 * model_args.dim,
            model_args.multiple_of,
            model_args.ffn_dim_multiplier,
            use_tiled_mlp="tiled_mlp" in model_args.attn_impl,
        )

        if "tiled_mlp" in model_args.attn_impl:
            self.attention_norm = TiledRMSNorm(model_args.dim, eps=model_args.norm_eps)
            self.ffn_norm = TiledRMSNorm(model_args.dim, eps=model_args.norm_eps)
        else:
            self.attention_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
            self.ffn_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)

        if model_args.depth_init:
            self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * model_args.n_layers) ** 0.5

        # Optional debug hooks for NaN/Inf grads across residual/norm/ffn
        if os.getenv("TT_DEBUG_NAN_GRADS", "0") == "1":

            def _reg_hook(tensor: torch.Tensor, tag: str):
                if tensor is None or not isinstance(tensor, torch.Tensor):
                    return
                try:
                    tensor.register_hook(
                        lambda g: (_ for _ in ()).throw(
                            RuntimeError(f"NaN/Inf grad at {tag} layer{self.layer_id}")
                        )
                        if g is not None
                        and (torch.isnan(g).any() or torch.isinf(g).any())
                        else None
                    )
                except Exception:
                    pass

            # Wrap forward to tap intermediate tensors
            orig_forward = self.forward

            def wrapped_forward(x: torch.Tensor, freqs_cis: torch.Tensor):
                x_attn_norm = self.attention_norm(x)
                _reg_hook(x_attn_norm, "attn_norm.out")
                attn_out = self.attention(x_attn_norm, freqs_cis)
                _reg_hook(attn_out, "attention.out")
                h = x + attn_out
                _reg_hook(h, "residual.attn")
                h_norm = self.ffn_norm(h)
                _reg_hook(h_norm, "ffn_norm.out")
                ff_out = self.feed_forward(h_norm)
                _reg_hook(ff_out, "ffn.out")
                out = h + ff_out
                _reg_hook(out, "residual.ffn")
                return out

            self.forward = wrapped_forward

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """

        h = x + self.attention(self.attention_norm(x), freqs_cis)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def init_weights(self):
        for norm in (self.attention_norm, self.ffn_norm):
            if norm is not None:
                norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        self.feed_forward.init_weights(self.weight_init_std)


class Transformer(nn.Module, ModelProtocol):
    """
    Transformer Module

    Args:
        model_args (TransformerModelArgs): Model configuration arguments.

    Attributes:
        model_args (TransformerModelArgs): Model configuration arguments.
        vocab_size (int): Vocabulary size.
        n_layers (int): Number of layers in the model.
        tok_embeddings (ParallelEmbedding): Token embeddings.
        layers (torch.nn.ModuleList): List of Transformer blocks.
        norm (RMSNorm): Layer normalization for the model output.
        output (ColumnParallelLinear): Linear layer for final output.
        freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

    """

    def __init__(self, model_args: TransformerModelArgs):
        super().__init__()
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers
        self.eos_id = model_args.eos_id

        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)

        # TODO persistent should be set to false, since this buffer can be recomputed.
        # however, we set it to true for 2 reasons.  (1) due to pytorch/pytorch#123411,
        # compile or pipeline-tracer will not correctly handle non-persistent buffers,
        # so we need to fix that.  (2) if we initialize pipeline-parallel models from
        # a seed checkpoint rather than calling init_weights, we need freqs_cis to be
        # initialized by the checkpoint, or we need to add a separate initializer for
        # just the non-persistent buffers that is called after loading checkpoints.
        self.register_buffer(
            "freqs_cis", self._precompute_freqs_cis(), persistent=False
        )

        self.vocab_size = model_args.vocab_size
        self.dim = model_args.dim

        if "offload" in model_args.attn_impl:
            self.offload_stream = torch.cuda.Stream()
            self.fetch_stream = torch.cuda.Stream()
        else:
            self.offload_stream = None
            self.fetch_stream = None

        self.two_streams = None

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(
                layer_id,
                model_args,
                self.offload_stream,
                self.fetch_stream,
                self.two_streams,
            )

        if "tiled_mlp" in model_args.attn_impl:
            self.norm = TiledRMSNorm(model_args.dim, eps=model_args.norm_eps)
        else:
            self.norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)

        self.event_registry = defaultdict(dict)

        self.loss_fn = LigerFusedLinearCrossEntropyLoss(
            ignore_index=-100, reduction="sum"
        )
        self.init_weights()

    def init_weights(
        self,
        buffer_device: torch.device | None = None,
    ):
        """
        [Note: On ``init_weights`` vs. ``reset_parameters``]
        Modules may define ``reset_parameters`` to initialize parameter values.
        ``reset_parameters`` is meant to only initialize directly owned
        parameters/buffers, not those of their child modules, and it can be
        used to give the initial values for these tensors.
        Separately, users may want custom initialization for their modules,
        different from that in ``reset_parameters``. For this, we define
        ``init_weights``. We only call it in the constructor of this
        ``Transformer`` root module to avoid reinitializing tensors.
        """

        buffer_device = buffer_device or self.freqs_cis.device
        with torch.device(buffer_device):
            self.freqs_cis = self._precompute_freqs_cis()
        if self.tok_embeddings is not None:
            nn.init.normal_(self.tok_embeddings.weight)
        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights()
        if self.norm is not None:
            self.norm.reset_parameters()
        final_out_std = self.model_args.dim**-0.5
        cutoff_factor = 3
        if self.output is not None:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=final_out_std,
                a=-cutoff_factor * final_out_std,
                b=cutoff_factor * final_out_std,
            )

    def _precompute_freqs_cis(self) -> torch.Tensor:
        return precompute_freqs_cis(
            self.model_args.head_dim if self.model_args.head_dim is not None else self.model_args.dim // self.model_args.n_heads,
            # Need to compute until at least the max token limit for generation
            # TODO: explain in docs/composability.md why we removed the 2x
            # relaxing in our CP enablement PR
            self.model_args.max_seq_len,
            self.model_args.rope_theta,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        input_batch: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices if pipeline parallelism is not enabled.
                If pipeline parallelism is enabled, this will be the input token indices
                for the ranks on the first pipeline stage. This will be the activation of the
                previous pipeline stage if the current rank is not on the first stage.
            input_batch (torch.Tensor): The input batch read from the dataloader.
                This will always be the input batch regardless of the pipeline stage.
                This field is required for non-first PP stages to perform document
                masking attention (to analyze the boundary of the document).

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.

        """
        self.event_registry = defaultdict(dict)
        if self.model_args.use_flex_attn:
            init_attention_mask(
                input_batch if input_batch is not None else tokens, eos_id=self.eos_id
            )

        # passthrough for nonexistent layers, allows easy configuration of pipeline parallel stages
        h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens

        for layer in self.layers.values():
            h = layer(h, self.freqs_cis)

        h = self.norm(h) if self.norm else h
        orig_h_shape = h.shape

        if self.model_args.chunked_loss:
            # Ensure weight tensor has proper storage for distributed training
            output_weight = self.output.weight
            if output_weight.storage().size() == 0 or not output_weight.is_contiguous():
                assert (
                    False
                ), "Output weight tensor has empty storage or is not contiguous, falling back to standard PyTorch cross entropy"
            loss = self.loss_fn(
                output_weight, h.reshape(-1, self.dim), labels.reshape(-1)
            )
            non_ignored_tokens = (labels != -100).sum()

            if non_ignored_tokens > 0:
                loss = loss / non_ignored_tokens

            if os.environ.get("CLEAR_CUDA_CACHE", "0") == "1":
                torch.cuda.empty_cache()

            return loss
        else:
            output = self.output(h) if self.output else h
            return output

    @classmethod
    def from_model_args(cls, model_args: TransformerModelArgs) -> "Transformer":
        """
        Initialize a Transformer model from a TransformerModelArgs object.

        Args:
            model_args (TransformerModelArgs): Model configuration arguments.

        Returns:
            Transformer: Transformer model.

        """
        return cls(model_args)
