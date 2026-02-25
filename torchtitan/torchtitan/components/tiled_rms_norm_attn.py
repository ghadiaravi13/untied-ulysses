import torch
import torch.nn as nn

# Register operator
lib = torch.library.Library("my_ops", "DEF")
lib.define("rms_norm(Tensor x, Tensor weight, float eps) -> (Tensor, Tensor)")

# torch.ops.my_ops.rms_norm


def rms_norm_impl(x: torch.Tensor, weight: torch.Tensor, eps: float):
    mean_square = x.pow(2).mean(dim=-1, keepdim=True)
    rms = torch.sqrt(mean_square + eps)
    norm_x = x / rms
    out = norm_x * weight
    return out, norm_x


lib.impl("rms_norm", rms_norm_impl, "CPU")
lib.impl("rms_norm", rms_norm_impl, "CUDA")


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


class _RMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps, chunked=False):
        if chunked:
            bs, seqlen, hid_dim = x.shape
            num_chunks = seqlen // hid_dim
            chunks = torch.chunk(x, num_chunks, dim=1)
            out = torch.empty_like(x)
            norm_x = torch.empty_like(x)
            for i in range(num_chunks):
                (
                    out[:, i * hid_dim : (i + 1) * hid_dim],
                    norm_x[:, i * hid_dim : (i + 1) * hid_dim],
                ) = torch.ops.my_ops.rms_norm(chunks[i], weight, eps)
        else:
            out, norm_x = torch.ops.my_ops.rms_norm(x, weight, eps)

        # out = norm_x * weight
        ctx.save_for_backward(x, weight, norm_x)
        ctx.eps = eps
        ctx.chunked = chunked
        assert not check_nan_inf(out, "out"), "NaN detected in out"
        assert not check_nan_inf(norm_x, "norm_x"), "NaN detected in norm_x"
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight, norm_x = ctx.saved_tensors
        eps = ctx.eps
        chunked = ctx.chunked
        dim = x.shape[-1]

        grad_weight = torch.zeros(
            weight.shape, dtype=weight.dtype, device=weight.device
        )  # (grad_out * norm_x).sum(dim=tuple(range(grad_out.dim()-1)))
        grad_x_list = []

        if chunked:
            bs, seqlen, hid_dim = x.shape
            num_chunks = seqlen // hid_dim
            grad_out_chunks = torch.chunk(grad_out, num_chunks, dim=1)
            x_chunks = torch.chunk(x, num_chunks, dim=1)
            norm_x_chunks = torch.chunk(norm_x, num_chunks, dim=1)

            for go, xc, nc in zip(grad_out_chunks, x_chunks, norm_x_chunks):

                # grad_weight
                grad_weight += (go * nc).sum(dim=tuple(range(go.dim() - 1)))

                # grad_x
                grad_norm_x_c = go * weight
                rms_c = xc / nc
                dot_c = (xc * grad_norm_x_c).sum(dim=-1, keepdim=True) / dim
                grad_x_c = (grad_norm_x_c / rms_c) - (xc * dot_c / (rms_c**3))
                grad_x_list.append(grad_x_c)

            grad_x = torch.cat(grad_x_list, dim=1)
        else:
            grad_norm_x = grad_out * weight
            rms = x / norm_x
            dot = (x * grad_norm_x).sum(dim=-1, keepdim=True) / dim
            grad_x = (grad_norm_x / rms) - (x * dot / (rms**3))

        return grad_x, grad_weight, None, None


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, chunked=False):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.chunked = chunked

    def forward(self, x):
        return _RMSNormFn.apply(x, self.weight, self.eps, self.chunked)

    def reset_parameters(self):
        nn.init.ones_(self.weight)
