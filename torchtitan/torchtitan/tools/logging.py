# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os


logger = logging.getLogger()


def init_logger(save_traces_folder=None, dump_folder=None):
    # Check if we're in a distributed environment
    rank = 0
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
    elif "LOCAL_RANK" in os.environ:
        rank = int(os.environ["LOCAL_RANK"])

    # Only rank 0 gets full INFO logging to console
    # Other ranks get WARNING level to reduce log noise
    if rank == 0:
        logger.setLevel(logging.INFO)

        # Console handler (existing behavior)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "[titan] %(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File handler - save logs to the traces folder if provided
        if save_traces_folder and dump_folder:
            log_dir = os.path.join(dump_folder, save_traces_folder)
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f"training_rank{rank}.log")

            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

            # Log that we're saving to file
            logger.info(f"Training logs will be saved to: {log_file}")
    else:
        # Other ranks: only show warnings/errors, no console handler
        logger.setLevel(logging.WARNING)

        # For other ranks, still save to file if requested (but only WARNING+)
        if save_traces_folder and dump_folder:
            log_dir = os.path.join(dump_folder, save_traces_folder)
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f"training_rank{rank}.log")

            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.WARNING)
            formatter = logging.Formatter(
                "[titan] %(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    # suppress verbose torch.profiler logging
    os.environ["KINETO_LOG_LEVEL"] = "5"


def get_rank():
    """Get the current rank from environment variables."""
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    elif "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    return 0


def rank_print(*args, rank=0, **kwargs):
    """Print only on the specified rank (default: rank 0)."""
    if get_rank() == rank:
        print(*args, **kwargs)
