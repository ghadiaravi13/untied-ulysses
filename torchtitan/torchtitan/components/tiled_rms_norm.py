import torch
import torch.nn as nn
import torch.nn.functional as F

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


@torch.library.custom_op(
    "yunchang::_tiled_rms_norm_operator", mutates_args=(), device_types="cuda"
)
def tiled_rms_norm_operator(
    w: torch.Tensor, x: torch.Tensor, shards: int, eps: float = 1e-6
) -> torch.Tensor:

    x_shards = list(torch.chunk(x, chunks=shards, dim=1))
    with torch.no_grad():
        y_shards = []
        for xs in x_shards:
            # Ensure weight is on the same device/dtype as xs to mirror torch semantics
            if w.device != xs.device or w.dtype != xs.dtype:
                w = w.to(dtype=xs.dtype, device=xs.device)
            y_shards.append(F.rms_norm(xs, [xs.shape[-1]], weight=w, eps=eps))
    y = torch.cat(y_shards, dim=1)

    return y


@tiled_rms_norm_operator.register_fake
def _(w: torch.Tensor, x: torch.Tensor, shards: int, eps: float = 1e-6) -> torch.Tensor:
    return torch.empty_like(x)


class TiledRMSNorm(torch.autograd.Function):
    """
    Tiled RMSNorm using autograd replay per shard, similar to DeepSpeed's TiledMLP.

    - Forward: shards the sequence dimension and computes F.rms_norm with no grad.
    - Backward: replays per shard under enable_grad and calls torch.autograd.backward,
      accumulating x.grad slices and parameter gradients on self.weight.

    This mirrors PyTorch rms_norm behavior and avoids manual gradient formulas.
    """

    @staticmethod
    def forward(
        ctx, self_mod: nn.Module, x: torch.Tensor, shards: int | None, eps: float | None
    ):
        ctx.self_mod = self_mod
        ctx.shards = shards
        ctx.eps = eps if eps is not None else self_mod.eps
        # Save x and a handle to the parameter to detect device/dtype changes

        bs, seqlen, hidden = x.shape
        if shards is None:
            # Match TiledMLP heuristic: aim for shard length ~= hidden
            shards = max(1, (seqlen + hidden - 1) // hidden)
        ctx.shards = shards

        with torch.no_grad():
            y = tiled_rms_norm_operator(self_mod.weight, x, shards, self_mod.eps)

        ctx.save_for_backward(x, self_mod.weight)
        return y

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (x, saved_w) = ctx.saved_tensors
        self_mod = ctx.self_mod
        shards = ctx.shards
        eps = ctx.eps

        x_requires_grad = x.requires_grad
        x = x.detach()
        x.requires_grad_(x_requires_grad)

        bs, seqlen, hidden = x.shape

        # Flatten (bs, seqlen) to avoid stride issues when narrowing
        x_flat = x.view(-1, hidden)
        g_flat = grad_out.view(-1, hidden)
        x_grad_flat = torch.zeros_like(x_flat)

        x_shards = list(torch.chunk(x_flat, chunks=shards, dim=0))
        current_offset = 0
        for i, x_shard in enumerate(x_shards):
            # If using ZeRO, coordinate gradient readiness flag as in TiledMLP
            if hasattr(self_mod.weight, "ds_grad_is_ready"):
                if i + 1 < shards:
                    self_mod.weight.ds_grad_is_ready = False
                else:
                    self_mod.weight.ds_grad_is_ready = True

            shard_step = x_shard.shape[0]
            shard_offset = current_offset

            x_shard.requires_grad_(x_requires_grad)
            # Route autograd to write directly into the appropriate x_grad slice
            x_shard.grad = x_grad_flat.narrow(0, shard_offset, shard_step).view_as(
                x_shard
            )
            incoming_grad_shard = g_flat.narrow(0, shard_offset, shard_step).view_as(
                x_shard
            )

            with torch.enable_grad():
                w = self_mod.weight
                if w.device != x_shard.device or w.dtype != x_shard.dtype:
                    # cast a view for compute but keep parameter for grad accumulation
                    w = w.to(dtype=x_shard.dtype, device=x_shard.device)
                y = F.rms_norm(x_shard, [hidden], weight=w, eps=eps)
            torch.autograd.backward(y, incoming_grad_shard)

            current_offset += shard_step

        # Unflatten
        x_grad = x_grad_flat.view(bs, -1, hidden) if x_requires_grad else None

        # Gradients for inputs returned; parameter gradients are accumulated on self_mod.weight
        return (None, x_grad, None, None)


def tiled_rms_norm_forward(
    self_mod: nn.Module,
    x: torch.Tensor,
    shards: int | None = None,
    eps: float | None = None,
) -> torch.Tensor:
    """Convenience wrapper to apply TiledRMSNorm on an RMSNorm-like module with a .weight and .eps"""
    return TiledRMSNorm.apply(self_mod, x, shards, eps)
