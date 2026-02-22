import torch
from triton_patch_rotary import apply_rotary as _apply_rotary_emb_flash


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
    xk: torch.Tensor = None,
    freqs_cis: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
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
    assert xq.device != torch.device("cpu"), "xq must be on GPU"
    assert freqs_cis.device != torch.device("cpu"), "freqs_cis must be on GPU"

    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    if xk is not None:
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
        return xq_out.type_as(xq), xk_out.type_as(xk)
    else:
        return xq_out.type_as(xq)


def apply_rotary_emb_flash(
    xq: torch.Tensor,
    xk: torch.Tensor = None,
    freqs_cis: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.
    """
    cos = freqs_cis.real
    sin = freqs_cis.imag

    if xk is not None:
        _apply_rotary_emb_flash(xq, cos, sin, interleaved=True, inplace=True)
        _apply_rotary_emb_flash(xk, cos, sin, interleaved=True, inplace=True)
        return xq, xk
    else:
        _apply_rotary_emb_flash(xq, cos, sin, interleaved=True, inplace=True)
        return xq
