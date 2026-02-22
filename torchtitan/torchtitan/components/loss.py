# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, List, TypeAlias

import torch
import torch.nn.functional as F

from liger_kernel.transformers.fused_linear_cross_entropy import (
    LigerFusedLinearCrossEntropyLoss,
)


class CEWithChunkedOutputLoss(torch.nn.Module):
    """
    Cross-entropy with chunked outputs that saves memory by only upcasting one chunk at a time.

    Whenever the model is trained with bf16, before running CE, we have to upcast
    it to fp32 for better accuracy and stability. When upcasting happens, the memory usage doubles.
    Models like llama3 have large vocabulary size and, therefore, have a large output
    tensor of shape ``(bsz, num_tokens, vocab_size)``. If we chunk on the token level, you can still compute
    the cross entropy normally, but upcasting only one chunk at a time saves considerable memory.

    The CE and upcasting have to be compiled together for better performance.
    When using this class, we recommend using :func:`torch.compile` only on the method ``compute_cross_entropy``.
    The gains from chunking won't be realized if you compile the entire class.

    For more details, please refer to: https://github.com/pytorch/torchtune/pull/1390
    """

    def __init__(self, num_output_chunks: int = 8, ignore_index: int = -100):
        super().__init__()
        self.num_output_chunks = num_output_chunks
        self.ignore_index = ignore_index

    @torch.compile
    def compute_cross_entropy(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Upcast logits to fp32 and compute cross entropy loss.
        """
        return F.cross_entropy(
            logits.float(), labels, ignore_index=self.ignore_index, reduction="sum"
        )

    def forward(self, logits: List[torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (List[torch.Tensor]): List of chunked logits of length
                ``self.num_output_chunks``, where each chunk has shape
                ``(batch_size, num_tokens / num_output_chunks, vocab_size)``.
            labels (torch.Tensor): Ground truth labels of shape ``(batch_size, num_tokens)``.

        Returns:
            torch.Tensor: Cross entropy loss of shape (1,).

        Example:
            >>> loss_fn = ChunkedCrossEntropyLoss()
            >>>
            >>> h = torch.tensor([bsz, num_tokens, dim])
            >>> output_chunks = [model.output(chunk) for chunk in h.chunk(num_chunks, dim=1)]
            >>>
            >>> labels = torch.tensor([bsz, num_tokens])
            >>> loss = loss_fn(output_chunks, labels)
        """

        total_elements = (labels != self.ignore_index).sum()

        # chunk and reshape labels (bsz, num_tokens, vocab) -> [(bsz*num_tokens/num_chunks, vocab)]
        labels = [
            target_chunk.reshape(-1)
            for target_chunk in labels.chunk(self.num_output_chunks, dim=1)
        ]
        # reshape logits [(bsz, num_tokens/num_chunks, vocab)] -> [(bsz*num_tokens/num_chunks, vocab)]
        logits = [
            logit_chunk.reshape(-1, logit_chunk.size(-1)) for logit_chunk in logits
        ]

        # compute one chunk at a time
        total_loss = 0.0
        for logits_chunk, labels_chunk in zip(logits, labels):
            total_loss += self.compute_cross_entropy(logits_chunk, labels_chunk)

        return total_loss / total_elements


from torchtitan.config_manager import JobConfig
from torchtitan.tools.logging import logger

LossFunction: TypeAlias = Callable[..., torch.Tensor]


def cross_entropy_loss(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Common cross-entropy loss function for Transformer models training."""
    # if torch.all(labels == -100):
    #     # Create a zero loss that requires gradients but has no computational dependencies
    #     # This ensures devices with all masked labels contribute 0 to loss without NaN issues
    #     return torch.tensor(0.0, device=pred.device, dtype=pred.dtype, requires_grad=True)

    return torch.nn.functional.cross_entropy(
        pred.flatten(0, 1).float(), labels.flatten(0, 1)
    )


def chunked_ce_loss(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Common cross-entropy loss function for Transformer models training."""
    num_output_chunks = 64
    chunked_loss = CEWithChunkedOutputLoss(
        num_output_chunks=num_output_chunks
    )  # default chunks
    return chunked_loss(pred.chunk(num_output_chunks, dim=1), labels)


def liger_loss(
    output_weights: torch.Tensor, hidden_states: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """
    Liger loss function for Transformer models training.
    """
    # Ensure all tensors are on the same device and have compatible dtypes
    device = hidden_states.device

    # Handle potential storage issues with output_weights in distributed training
    if output_weights.storage().size() == 0:
        print(f"Warning: Output weights has empty storage - likely distributed tensor")
        print(f"Tensor shape: {output_weights.shape}, device: {output_weights.device}")
        print(f"Is distributed tensor: {hasattr(output_weights, '_spec')}")
        print(f"Tensor type: {type(output_weights)}")

        # Try to get the actual data if this is a distributed tensor
        if hasattr(output_weights, "to_local"):
            print("Converting DTensor to local tensor...")
            output_weights = output_weights.to_local()
        elif hasattr(output_weights, "_local_tensor"):
            print("Extracting local tensor from distributed tensor...")
            output_weights = output_weights._local_tensor
        else:
            # If we can't get local data, fall back to standard cross entropy
            print(
                "Cannot extract local data from distributed tensor, using standard cross entropy fallback"
            )
            try:
                logits = torch.matmul(hidden_states, output_weights.t())
                return torch.nn.functional.cross_entropy(
                    logits, labels, ignore_index=-100, reduction="mean"
                )
            except Exception as fallback_error:
                print(f"Fallback matmul also failed: {fallback_error}")
                print(
                    "This suggests tensor parallelism is incompatible with chunked_loss mode"
                )
                raise RuntimeError(
                    "Cannot use chunked_loss with this distributed tensor configuration. "
                    "Please disable chunked_loss in your config."
                )

    # Ensure weight tensor is properly configured for gradients
    if not output_weights.requires_grad:
        output_weights.requires_grad_(True)

    # Ensure tensors are contiguous and on the same device
    output_weights = output_weights.contiguous().to(device)
    hidden_states = hidden_states.contiguous().to(device)
    labels = labels.contiguous().to(device)

    # Validate tensor shapes
    batch_seq_len, hidden_dim = hidden_states.shape
    vocab_size, weight_hidden_dim = output_weights.shape

    assert (
        weight_hidden_dim == hidden_dim
    ), f"Hidden dimension mismatch: {weight_hidden_dim} vs {hidden_dim}"
    assert (
        labels.numel() == batch_seq_len
    ), f"Labels shape mismatch: {labels.numel()} vs {batch_seq_len}"

    try:
        # For very large vocabularies (>100k), use more conservative settings
        if vocab_size > 100000:
            # Use smaller chunks to reduce memory pressure
            loss_fn = LigerFusedLinearCrossEntropyLoss(
                ignore_index=-100,
                reduction="sum",  # Use sum reduction and normalize manually
            )
            loss = loss_fn(output_weights, hidden_states, labels)
            # Manually normalize by non-ignored tokens for mean reduction
            non_ignored_tokens = (labels != -100).sum()
            if non_ignored_tokens > 0:
                loss = loss / non_ignored_tokens
            return loss
        else:
            loss_fn = LigerFusedLinearCrossEntropyLoss(
                ignore_index=-100, reduction="mean"
            )
            return loss_fn(output_weights, hidden_states, labels)
    except Exception as e:
        # Fallback to standard cross entropy if Liger fails
        print(f"Liger loss failed with error: {e}")
        print(f"Falling back to standard PyTorch cross entropy")
        print(
            f"Tensor shapes - weights: {output_weights.shape}, hidden: {hidden_states.shape}, labels: {labels.shape}"
        )

        # Fallback using standard matrix multiplication and cross entropy
        logits = torch.matmul(hidden_states, output_weights.t())
        return torch.nn.functional.cross_entropy(
            logits, labels, ignore_index=-100, reduction="mean"
        )


def build_cross_entropy_loss(job_config: JobConfig):
    loss_fn = cross_entropy_loss
    if job_config.training.chunked_loss:
        # loss_fn = chunked_ce_loss
        loss_fn = liger_loss
    if job_config.training.compile:
        logger.info("Compiling the loss function with torch.compile")
        loss_fn = torch.compile(loss_fn)
    return loss_fn
