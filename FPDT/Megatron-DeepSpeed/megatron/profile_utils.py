"""Profiling utilities for Megatron-DeepSpeed training.

This module provides:
  - ``create_torch_profiler``: creates a ``torch.profiler.profile`` context
    manager for activity tracing (CPU + CUDA chrome traces).
  - ``MemoryProfiler``: continuously records CUDA memory allocation history
    via ``torch.cuda.memory._record_memory_history()`` and dumps ``.pickle``
    snapshots periodically or on OOM.  Visualise the snapshots with
    https://pytorch.org/memory_viz .

Constants:
  - ``MAX_MEMORY_SNAPSHOT_ENTRIES``: max allocation/free events kept in the
    ring-buffer (default 1 000 000).
  - ``MEMORY_SNAPSHOT_DUMP_FREQ``: dump a snapshot every N training steps
    under normal operation (default 3).
"""

import os
import pickle
import time

import torch


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_MEMORY_SNAPSHOT_ENTRIES = 1_000_000
MEMORY_SNAPSHOT_DUMP_FREQ = 3  # dump a memory snapshot every N steps (unless OOM)


# ---------------------------------------------------------------------------
# Chrome-trace helpers
# ---------------------------------------------------------------------------
def _create_trace_handler(trace_dir: str):
    """Return an ``on_trace_ready`` callback that exports chrome traces."""
    def trace_handler(prof):
        curr_trace_dir_name = "iteration_" + str(prof.step_num)
        curr_trace_dir = os.path.join(trace_dir, curr_trace_dir_name)
        if not os.path.exists(curr_trace_dir):
            os.makedirs(curr_trace_dir, exist_ok=True)

        device_num = torch.distributed.get_rank()
        prof.export_chrome_trace(f"{curr_trace_dir}/rank{device_num}_trace.json")
    return trace_handler


def create_torch_profiler(args):
    """Create a ``torch.profiler.profile`` if enabled via *args*, else ``None``.

    Memory snapshot recording is handled separately by :class:`MemoryProfiler`.
    """
    if not getattr(args, 'torch_profiler_enable', False):
        return None

    trace_dir = getattr(args, 'torch_profiler_trace_dir', None)
    if trace_dir is None:
        trace_dir = os.path.join(os.getcwd(), "profiler_traces")

    return torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=0, active=1),
        with_stack=True,
        on_trace_ready=_create_trace_handler(trace_dir),
    )


# ---------------------------------------------------------------------------
# Memory snapshot profiler
# ---------------------------------------------------------------------------
class MemoryProfiler:
    """Continuously records CUDA memory history and dumps snapshots periodically.

    Inspired by torchtitan's ``MemoryProfiler``.  Key design choices:

    * Recording is started **once** at construction and is **never stopped**
      between steps, so the snapshot always contains the full allocation
      history across all prior steps.
    * Periodic dumps happen every *freq* steps.
    * On OOM, call ``step(exit_ctx=True)`` to force-dump the accumulated
      history (the most valuable artifact for debugging memory pressure).
    """

    def __init__(
        self,
        snapshot_dir: str,
        step_num: int = 0,
        freq: int = MEMORY_SNAPSHOT_DUMP_FREQ,
    ):
        self.snapshot_dir = snapshot_dir
        self.step_num = step_num
        self.freq = freq
        self.rank = torch.distributed.get_rank()

        # Start recording — never stopped until OOM dump or process exit
        torch.cuda.memory._record_memory_history(
            max_entries=MAX_MEMORY_SNAPSHOT_ENTRIES
        )
        os.makedirs(snapshot_dir, exist_ok=True)

    def step(self, exit_ctx: bool = False):
        """Call after each training step (or from an OOM handler with ``exit_ctx=True``)."""
        self.step_num += 1

        if not exit_ctx and self.step_num % self.freq != 0:
            return

        if not exit_ctx:
            curr_step = self.step_num
            dir_name = f"iteration_{curr_step}"
        else:
            # OOM happened during this step; label with the step that failed
            curr_step = self.step_num - 1
            dir_name = f"iteration_{curr_step}_oom"

        curr_snapshot_dir = os.path.join(self.snapshot_dir, dir_name)
        os.makedirs(curr_snapshot_dir, exist_ok=True)
        snapshot_path = os.path.join(
            curr_snapshot_dir, f"rank{self.rank}_memory_snapshot.pickle"
        )

        print(f"[Rank {self.rank}] Dumping memory snapshot at step {curr_step}")
        begin = time.monotonic()
        try:
            with open(snapshot_path, "wb") as f:
                pickle.dump(torch.cuda.memory._snapshot(), f)
            elapsed = time.monotonic() - begin
            print(
                f"[Rank {self.rank}] Memory snapshot saved to {snapshot_path} "
                f"in {elapsed:.2f}s"
            )
        except Exception as e:
            print(f"[Rank {self.rank}] Failed to dump memory snapshot: {e}")

        if exit_ctx:
            # After an OOM dump, stop recording to free internal buffers
            torch.cuda.memory._record_memory_history(enabled=None)
