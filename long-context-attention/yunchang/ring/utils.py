import gc
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

__all__ = ["update_out_and_lse", "RingComm"]


@torch.jit.script
def _update_out_and_lse(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    block_out = block_out.to(torch.float32)
    if block_lse.ndim == 3:
        block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)

    # new_lse = lse + torch.log(1 + torch.exp(block_lse - lse))
    # torch.exp(lse - new_lse) * out + torch.exp(block_lse - new_lse) * block_out
    # For additional context and discussion, please refer to:
    # https://github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795
    # assert False, f"out.shape: {out.shape}, lse.shape: {lse.shape}, block_out.shape: {block_out.shape}, block_lse.shape: {block_lse.shape}"
    out = out - F.sigmoid(block_lse - lse) * (out - block_out)
    lse = lse - F.logsigmoid(lse - block_lse)

    return out, lse


def update_out_and_lse(
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    slice_=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
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


@torch.jit.script
def flatten_varlen_lse(lse, cu_seqlens):
    new_lse = []
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        new_lse.append(lse[i, :, : end - start])
    return torch.cat(new_lse, dim=1)


@torch.jit.script
def unflatten_varlen_lse(lse, cu_seqlens, max_seqlen: int):
    num_seq = len(cu_seqlens) - 1
    num_head = lse.shape[-2]
    new_lse = torch.empty(
        (num_seq, max_seqlen, num_head, 1), dtype=torch.float32, device=lse.device
    )
    for i in range(num_seq):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        new_lse[i, : end - start] = lse[start:end]
    return new_lse.squeeze(dim=-1).transpose(1, 2).contiguous()


class RingComm:
    def __init__(
        self,
        process_group: dist.ProcessGroup,
        recv_buffer: Optional[torch.Tensor] = None,
        pass_kv: bool = True,
    ):
        self._process_group = process_group
        self._ops = []
        self.rank = dist.get_rank(self._process_group)
        self.world_size = dist.get_world_size(self._process_group)
        self._reqs = None
        self.recv_buffer = recv_buffer

        # for pass KV
        if pass_kv:
            self.send_rank = (self.rank + 1) % self.world_size
            self.recv_rank = (self.rank - 1) % self.world_size

        # for pass query
        else:
            self.send_rank = (self.rank - 1) % self.world_size
            self.recv_rank = (self.rank + 1) % self.world_size

        if process_group is not None:
            self.send_rank = dist.get_global_rank(self._process_group, self.send_rank)
            self.recv_rank = dist.get_global_rank(self._process_group, self.recv_rank)

    def send_recv(
        self,
        to_send: torch.Tensor,
        recv_tensor: Optional[torch.Tensor] = None,
        send_rank: Optional[int] = None,
        recv_rank: Optional[int] = None,
    ) -> torch.Tensor:
        if recv_tensor is None:
            res = (
                self.recv_buffer
                if self.recv_buffer is not None
                else torch.empty_like(to_send)
            )
            # print(f"send_recv: empty_like {to_send.shape}")
        else:
            res = recv_tensor

        send_rank = send_rank if send_rank is not None else self.send_rank
        recv_rank = recv_rank if recv_rank is not None else self.recv_rank

        send_op = dist.P2POp(dist.isend, to_send, send_rank, group=self._process_group)
        recv_op = dist.P2POp(dist.irecv, res, recv_rank, group=self._process_group)
        self._ops.append(send_op)
        self._ops.append(recv_op)
        return res

    def commit(self):
        if self._reqs is not None:
            raise RuntimeError("commit called twice")
        self._reqs = dist.batch_isend_irecv(self._ops)

    def wait(self):
        if self._reqs is None:
            raise RuntimeError("wait called before commit")
        for req in self._reqs:
            req.wait()
        # self._reqs = None
        # self._ops = []

        # # Force CUDA to complete all pending operations
        # torch.cuda.synchronize()
        torch.distributed.barrier(group=self._process_group)

        # Clear references
        self._reqs.clear()
        self._reqs = None
        self._ops.clear()
        self._ops = []

        # Force Python GC to collect Work objects
        # gc.collect()


class SplitRingComm:
    def __init__(self, process_group: dist.ProcessGroup):
        self._process_group = process_group
        self._ops = []
        self.rank = dist.get_rank(self._process_group)
        self.world_size = dist.get_world_size(self._process_group)
        self._reqs = None

        self.send_rank = (self.rank + 1) % self.world_size
        self.recv_rank = (self.rank - 1) % self.world_size

        if process_group is not None:
            self.send_rank = dist.get_global_rank(self._process_group, self.send_rank)
            self.recv_rank = dist.get_global_rank(self._process_group, self.recv_rank)

    def send_recv(
        self, to_send: torch.Tensor, recv_tensor: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if recv_tensor is None:
            res = torch.empty_like(to_send)
            # print(f"send_recv: empty_like {to_send.shape}")
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
        if self._reqs is not None:
            raise RuntimeError("commit called twice")
        self._reqs = dist.batch_isend_irecv(self._ops)

    def recv_wait(self):
        if self._reqs is None:
            raise RuntimeError("recv wait called before commit")
        self._reqs[1].wait()
        self._reqs[1] = None
        self._ops = self._ops[:1]  # remove the recv op

    def send_wait(self):
        if self._reqs is None:
            raise RuntimeError("send wait called before commit")
        self._reqs[0].wait()
        self._reqs[0] = None
        self._ops = []  # remove the send op
