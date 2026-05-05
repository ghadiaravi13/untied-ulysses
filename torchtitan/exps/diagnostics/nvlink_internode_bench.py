"""
Internode NCCL benchmark, launched by torchrun.

Three probes, all run inside one process group (NCCL backend):

  1. Pairwise sendrecv across nodes
       For every pair of nodes (a, b) and every local-rank l in [0..7]:
         rank_a = a*8 + l, rank_b = b*8 + l
       isend/irecv 1 GiB bf16 round-trip; reports per-link bandwidth.
       This is what catches a single sick EFA NIC.

  2. AllReduce sweep over the full world (e.g. 32 ranks).
       1 MiB -> 8 GiB, x2.  busBW is what FSDP grad-sync sees.

  3. AllToAll sweep over the full world.
       1 MiB -> 1 GiB total tensor, x2.  Maps to Ulysses CP across nodes.

All ranks dump a per-rank JSON; rank 0 additionally writes summary CSV/MD.

The bandwidth math follows the standard NCCL test conventions:
  AllReduce      busBW = 2 * (n - 1) / n * size / time
  AllToAll       busBW = (n - 1) / n     * size / time
  Pairwise SR    busBW =                   size / time      (per direction;
                                                             we report the
                                                             bidirectional
                                                             pair's mean BW)
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import socket
from pathlib import Path
from typing import Iterable

import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def env_int(key: str, default: int) -> int:
    v = os.environ.get(key)
    return int(v) if v is not None else default


def parse_size(s: str) -> int:
    s = s.strip().upper()
    if not s:
        return 0
    mult = 1
    if s.endswith("K"):
        mult, s = 1024, s[:-1]
    elif s.endswith("M"):
        mult, s = 1024 ** 2, s[:-1]
    elif s.endswith("G"):
        mult, s = 1024 ** 3, s[:-1]
    return int(float(s) * mult)


def fmt_bytes(n: int) -> str:
    x = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if x < 1024 or unit == "TiB":
            return f"{x:.0f}{unit}" if unit == "B" else f"{x:.2f}{unit}"
        x /= 1024
    return f"{x:.2f}TiB"


def time_kernel(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    return ms / 1000.0  # seconds per iter


def default_sizes() -> list[int]:
    # 64 KiB ... 8 GiB (covers latency- and bandwidth-bound regimes)
    return [1 << k for k in (16, 18, 20, 22, 24, 26, 28, 30, 32, 33)]


# ---------------------------------------------------------------------------
# probes
# ---------------------------------------------------------------------------

def run_pairwise_sendrecv(
    iters: int,
    warmup: int,
    log_dir: Path,
    profile_tag: str,
) -> list[dict]:
    """Cross-node sendrecv across every (node_a, node_b, local_rank) triple."""
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = env_int("LOCAL_RANK", rank % torch.cuda.device_count())
    nproc = env_int("LOCAL_WORLD_SIZE", torch.cuda.device_count())
    nnodes = world // nproc
    node_id = rank // nproc

    if rank == 0:
        print(f"\n[pairwise sendrecv] world={world} nnodes={nnodes} nproc={nproc}",
              flush=True)

    sz = 1 << 30  # 1 GiB
    nelem = sz // 2  # bf16
    send_buf = torch.empty(nelem, dtype=torch.bfloat16, device="cuda")
    recv_buf = torch.empty(nelem, dtype=torch.bfloat16, device="cuda")
    send_buf.fill_(1.0)

    results: list[dict] = []

    for a in range(nnodes):
        for b in range(a + 1, nnodes):
            for l in range(nproc):
                rank_a = a * nproc + l
                rank_b = b * nproc + l

                involved = rank in (rank_a, rank_b)
                # Everyone synchronizes on the barrier so timings are comparable
                dist.barrier()

                if involved:
                    peer = rank_b if rank == rank_a else rank_a

                    def _step():
                        send_op = dist.P2POp(dist.isend, send_buf, peer)
                        recv_op = dist.P2POp(dist.irecv, recv_buf, peer)
                        reqs = dist.batch_isend_irecv([send_op, recv_op])
                        for r in reqs:
                            r.wait()

                    sec = time_kernel(_step, iters=iters, warmup=warmup)
                    # Each direction moves `sz` bytes; the round-trip pair
                    # moved 2*sz bytes in `sec`.  Per-direction BW = sz/sec.
                    bw = sz / sec
                    results.append({
                        "kind": "pairwise_sendrecv",
                        "profile": profile_tag,
                        "node_a": a, "node_b": b, "local_rank": l,
                        "rank_a": rank_a, "rank_b": rank_b,
                        "size_bytes": sz, "iters": iters,
                        "sec_per_iter": sec, "bw_per_dir_Bps": bw,
                    })
                    if rank == rank_a:
                        print(f"  [SR] node {a} <-> {b} (local_rank={l}) "
                              f"BW(per-dir) = {bw / 1e9:.2f} GB/s "
                              f"({sec * 1e3:.2f} ms / iter)", flush=True)
                else:
                    # uninvolved ranks just wait at the next barrier
                    pass

                dist.barrier()

    return results


def run_collective_sweep(
    coll_name: str,
    sizes: Iterable[int],
    iters: int,
    warmup: int,
    profile_tag: str,
) -> list[dict]:
    rank = dist.get_rank()
    world = dist.get_world_size()
    results: list[dict] = []

    if rank == 0:
        print(f"\n[{coll_name}] world={world}  sizes={[fmt_bytes(s) for s in sizes]}",
              flush=True)

    for sz in sizes:
        nelem = max(sz // 2, world)         # bf16
        nelem = (nelem // world) * world    # divisible by world

        try:
            if coll_name == "all_reduce":
                buf = torch.empty(nelem, dtype=torch.bfloat16, device="cuda")
                buf.fill_(1.0)
                fn = lambda: dist.all_reduce(buf, op=dist.ReduceOp.SUM)
                # NCCL convention: busBW = 2*(n-1)/n * size / time
                bw_factor = 2.0 * (world - 1) / world

            elif coll_name == "all_gather":
                shard = torch.empty(nelem // world, dtype=torch.bfloat16,
                                    device="cuda")
                shard.fill_(1.0)
                buf = torch.empty(nelem, dtype=torch.bfloat16, device="cuda")
                fn = lambda: dist.all_gather_into_tensor(buf, shard)
                bw_factor = (world - 1) / world

            elif coll_name == "reduce_scatter":
                buf = torch.empty(nelem, dtype=torch.bfloat16, device="cuda")
                buf.fill_(1.0)
                shard = torch.empty(nelem // world, dtype=torch.bfloat16,
                                    device="cuda")
                fn = lambda: dist.reduce_scatter_tensor(shard, buf,
                                                       op=dist.ReduceOp.SUM)
                bw_factor = (world - 1) / world

            elif coll_name == "all_to_all":
                buf_in = torch.empty(nelem, dtype=torch.bfloat16, device="cuda")
                buf_out = torch.empty(nelem, dtype=torch.bfloat16, device="cuda")
                buf_in.fill_(1.0)
                fn = lambda: dist.all_to_all_single(buf_out, buf_in)
                bw_factor = (world - 1) / world

            else:
                raise ValueError(coll_name)

            dist.barrier()
            sec = time_kernel(fn, iters=iters, warmup=warmup)
            algbw = (nelem * 2) / sec        # bytes/sec moved by tensor
            busbw = bw_factor * algbw

            results.append({
                "kind": coll_name,
                "profile": profile_tag,
                "size_bytes": nelem * 2,
                "world_size": world,
                "iters": iters,
                "sec_per_iter": sec,
                "algbw_Bps": algbw,
                "busbw_Bps": busbw,
            })

            if rank == 0:
                print(f"  size={fmt_bytes(nelem * 2):>10}  "
                      f"time={sec * 1e3:8.3f} ms  "
                      f"algBW={algbw / 1e9:7.2f} GB/s  "
                      f"busBW={busbw / 1e9:7.2f} GB/s",
                      flush=True)

            if coll_name == "all_to_all":
                del buf_in, buf_out
            else:
                del buf
                if coll_name in ("all_gather", "reduce_scatter"):
                    del shard
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError as e:
            if rank == 0:
                print(f"  size={fmt_bytes(sz):>10}  OOM  ({e})", flush=True)
            torch.cuda.empty_cache()
            break

    return results


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--profile-tag", default="iso_train",
                        help="Label for this run; written into the JSON output.")
    parser.add_argument("--sizes", nargs="*", default=None,
                        help="Override collective sizes (e.g. 1M 16M 1G).")
    args = parser.parse_args()

    args.log_dir.mkdir(parents=True, exist_ok=True)

    # torchrun sets these
    rank = env_int("RANK", 0)
    world = env_int("WORLD_SIZE", 1)
    local_rank = env_int("LOCAL_RANK", 0)

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl",
                            rank=rank, world_size=world,
                            device_id=torch.device("cuda", local_rank),
                            timeout=datetime.timedelta(seconds=1800))

    if rank == 0:
        print(f"[init] rank={rank}/{world} host={socket.gethostname()} "
              f"local_rank={local_rank} device={torch.cuda.current_device()} "
              f"cuda={torch.version.cuda} torch={torch.__version__} "
              f"nccl={torch.cuda.nccl.version()}", flush=True)

    if args.sizes:
        sizes = [parse_size(s) for s in args.sizes]
    else:
        sizes = default_sizes()

    all_results: list[dict] = []

    # 1. pairwise cross-node sendrecv (use a smaller iter count by default
    #    because we run nnodes*(nnodes-1)/2 * nproc pairs serially)
    pw_iters = max(args.iters // 2, 10)
    pw_warmup = max(args.warmup // 2, 5)
    all_results += run_pairwise_sendrecv(
        iters=pw_iters, warmup=pw_warmup,
        log_dir=args.log_dir, profile_tag=args.profile_tag,
    )

    # 2-4. collective sweeps
    for coll in ("all_reduce", "all_to_all", "reduce_scatter", "all_gather"):
        all_results += run_collective_sweep(
            coll_name=coll, sizes=sizes,
            iters=args.iters, warmup=args.warmup,
            profile_tag=args.profile_tag,
        )

    # per-rank JSON
    rank_path = args.log_dir / f"rank_{rank:03d}_{socket.gethostname()}.json"
    rank_path.write_text(json.dumps({
        "rank": rank, "world": world, "host": socket.gethostname(),
        "local_rank": local_rank, "profile": args.profile_tag,
        "results": all_results,
    }, indent=2))

    dist.barrier()

    # rank-0 summary tables
    if rank == 0:
        write_summary(args.log_dir, args.profile_tag, all_results, world)

    dist.destroy_process_group()


def write_summary(log_dir: Path, profile_tag: str,
                  results: list[dict], world: int) -> None:
    md_path = log_dir / f"_summary_{profile_tag}.md"
    csv_path = log_dir / f"_summary_{profile_tag}.csv"

    csv_lines = ["kind,size_bytes,world,sec_per_iter,algbw_GBps,busbw_GBps,extra"]
    md_lines: list[str] = []
    md_lines.append(f"# Internode NCCL bench  ({profile_tag})\n")
    md_lines.append(f"World size: {world}\n")

    # pairwise
    pw = [r for r in results if r["kind"] == "pairwise_sendrecv"]
    if pw:
        md_lines.append("## Pairwise cross-node sendrecv (per-direction BW)\n")
        md_lines.append("| node_a | node_b | local_rank | size | sec/iter | BW/dir GB/s |")
        md_lines.append("|--------|--------|------------|------|----------|-------------|")
        for r in pw:
            md_lines.append(
                f"| {r['node_a']} | {r['node_b']} | {r['local_rank']} | "
                f"{fmt_bytes(r['size_bytes'])} | {r['sec_per_iter']*1e3:.3f} ms | "
                f"{r['bw_per_dir_Bps']/1e9:.2f} |"
            )
            csv_lines.append(
                f"pairwise_sendrecv,{r['size_bytes']},2,"
                f"{r['sec_per_iter']:.6f},"
                f"{r['bw_per_dir_Bps']/1e9:.4f},,"
                f"a={r['node_a']};b={r['node_b']};l={r['local_rank']}"
            )

    # collectives
    for coll in ("all_reduce", "all_to_all", "reduce_scatter", "all_gather"):
        rs = [r for r in results if r["kind"] == coll]
        if not rs:
            continue
        md_lines.append(f"\n## {coll}  (world={world})\n")
        md_lines.append("| size | sec/iter | algBW GB/s | busBW GB/s |")
        md_lines.append("|------|----------|-----------|-----------|")
        for r in rs:
            md_lines.append(
                f"| {fmt_bytes(r['size_bytes'])} | "
                f"{r['sec_per_iter']*1e3:.3f} ms | "
                f"{r['algbw_Bps']/1e9:.2f} | "
                f"{r['busbw_Bps']/1e9:.2f} |"
            )
            csv_lines.append(
                f"{coll},{r['size_bytes']},{world},"
                f"{r['sec_per_iter']:.6f},"
                f"{r['algbw_Bps']/1e9:.4f},"
                f"{r['busbw_Bps']/1e9:.4f},"
            )

    md_path.write_text("\n".join(md_lines) + "\n")
    csv_path.write_text("\n".join(csv_lines) + "\n")
    print(f"\n[rank 0] wrote {md_path}", flush=True)
    print(f"[rank 0] wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
