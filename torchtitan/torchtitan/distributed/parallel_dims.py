# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property

from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from torchtitan.tools.logging import logger


__all__ = ["ParallelDims"]


@dataclass
class ParallelDims:
    dp_replicate: int
    dp_shard: int
    cp_ring: int
    cp_ulysses: int
    tp: int
    pp: int
    world_size: int
    enable_loss_parallel: bool

    def __post_init__(self):
        self._validate()

    def _validate(self):
        dp_replicate, dp_shard, cp_ring, cp_ulysses, tp, pp = (
            self.dp_replicate,
            self.dp_shard,
            self.cp_ring,
            self.cp_ulysses,
            self.tp,
            self.pp,
        )
        for d in (dp_replicate, cp_ring, cp_ulysses, tp, pp):
            assert d >= 1, "Parallelism degree should be >= 1, except for dp_shard"

        assert dp_shard == -1 or dp_shard >= 1, " dp_shard must -1 or >=1."
        if dp_shard < 0:
            self.dp_shard = dp_shard = self.world_size // (
                dp_replicate * cp_ring * cp_ulysses * tp * pp
            )
        assert dp_shard >= 1

        assert (
            dp_replicate * dp_shard * cp_ring * cp_ulysses * tp * pp == self.world_size
        ), (
            f"Invalid parallel dims: dp_replicate({dp_replicate}) * dp_shard({dp_shard}) * "
            f"cp_ring({cp_ring}) * cp_ulysses({cp_ulysses}) * tp({tp}) * pp({pp}) != WORLD_SIZE({self.world_size})"
        )

    def build_mesh(self, device_type: str) -> DeviceMesh:
        dims = []
        names = []
        for d, name in zip(
            [
                self.pp,
                self.dp_replicate,
                self.dp_shard,
                self.cp_ring,
                self.cp_ulysses,
                self.tp,
            ],
            ["pp", "dp_replicate", "dp_shard", "cp_ring", "cp_ulysses", "tp"],
        ):
            if d > 1 or (d == 1 and "cp_ring" in name):
                dims.append(d)
                names.append(name)

        return self._build_mesh(device_type, dims, names, init_device_mesh)

    def _build_mesh(
        self,
        device_type: str,
        dims: list[int],
        names: list[str],
        init_device_mesh_fn: Callable,
    ) -> DeviceMesh:
        logger.info(f"Building {len(dims)}-D device mesh with {names}, {dims}")
        mesh = init_device_mesh_fn(device_type, dims, mesh_dim_names=names)

        # Create all the submesh here to ensure all required process groups are
        # initialized:
        # Mesh for data loading (no communication on this mesh)
        dp_mesh_dim_names = []
        # Mesh for param sharding
        dp_shard_cp_mesh_dim_names = []
        # Mesh for loss all-reduce
        dp_cp_mesh_dim_names = []
        # Mesh for context parallelism
        cp_mesh_dim_names = []

        if self.dp_replicate_enabled:
            dp_mesh_dim_names.append("dp_replicate")
            dp_cp_mesh_dim_names.append("dp_replicate")
        if self.dp_shard_enabled:
            dp_mesh_dim_names.append("dp_shard")
            dp_shard_cp_mesh_dim_names.append("dp_shard")
            dp_cp_mesh_dim_names.append("dp_shard")
        if self.cp_ring >= 1:
            dp_shard_cp_mesh_dim_names.append("cp_ring")
            dp_cp_mesh_dim_names.append("cp_ring")
            cp_mesh_dim_names.append("cp_ring")
        if self.cp_ulysses > 1:
            dp_shard_cp_mesh_dim_names.append("cp_ulysses")
            cp_mesh_dim_names.append("cp_ulysses")

        if dp_mesh_dim_names != []:
            mesh[tuple(dp_mesh_dim_names)]._flatten(mesh_dim_name="dp")
        if dp_shard_cp_mesh_dim_names != []:
            mesh[tuple(dp_shard_cp_mesh_dim_names)]._flatten(
                mesh_dim_name="dp_shard_cp"
            )
        if dp_cp_mesh_dim_names != []:
            mesh[tuple(dp_cp_mesh_dim_names)]._flatten(mesh_dim_name="dp_cp")
        if cp_mesh_dim_names != []:
            # For CP mesh, create a flattened version for single dimension
            # or keep as multi-dimensional for ring + ulysses
            if len(cp_mesh_dim_names) == 1:
                # Single dimension CP, create a flattened mesh
                logger.info(f"Creating 1D CP mesh with {cp_mesh_dim_names[0]}")
                mesh[tuple(cp_mesh_dim_names)]._flatten(mesh_dim_name="cp")
            else:
                # Multi-dimensional CP (ring + ulysses)
                # The submesh is already accessible via mesh["cp_ring", "cp_ulysses"] (ring=dim0, ulysses=dim1)
                # But we also want it accessible via mesh["cp"]
                # Since DeviceMesh doesn't have a direct API for this, we need to
                # ensure that the 2D submesh can be accessed
                logger.info(f"Creating 2D CP mesh with dimensions {cp_mesh_dim_names}")
                pass

        return mesh

    @property
    def dp_enabled(self):
        return self.dp_replicate > 1 or self.dp_shard > 1

    @property
    def dp_replicate_enabled(self):
        return self.dp_replicate > 1

    @property
    def dp_shard_enabled(self):
        return self.dp_shard > 1

    @property
    def cp_enabled(self):
        return self.cp_ring > 1 or self.cp_ulysses > 1

    @property
    def tp_enabled(self):
        return self.tp > 1

    @property
    def pp_enabled(self):
        return self.pp > 1

    @property
    def loss_parallel_enabled(self):
        return self.tp > 1 and self.enable_loss_parallel

    @cached_property
    def non_data_parallel_size(self):
        return self.cp_ring * self.cp_ulysses * self.tp * self.pp
