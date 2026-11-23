"""The ucla-roms ``partition`` global attribute on per-rank files.

ucla-roms writes ``partition = (node, nnodes, xi_start, eta_start)`` on every per-rank
file it creates (``src/roms_read_write.F90``, ``src/instant_output.F90``), and so does
its reference partitioner ``Tools-Roms/partit.F``; ``xi_start``/``eta_start`` are the
1-based indices, in the whole grid's rho points, of the tile's first rho point
(``iSW_corn + iwest`` on a western-edge tile, ``iSW_corn + iwest + 1`` otherwise).
ucla-roms' ``ncjoin`` places tiles by it. Without it the tile layout has to be
inferred from dimension sizes, which is ambiguous for files with only rho points.
"""

import numpy as np
import pytest
import xarray as xr
import xarray.testing as xrt

from roms_tools.tiling.join import join_datasets
from roms_tools.tiling.partition import partition


def _rho_only(nx, ny):
    rng = np.random.default_rng(nx * 100 + ny)
    return xr.Dataset({"h": (("eta_rho", "xi_rho"), rng.random((ny + 2, nx + 2)))})


def test_partition_writes_the_ucla_roms_partition_attribute():
    # 10 interior points on 3 tiles: interior widths 3, 4, 3 (mpi_setup), so the
    # tiles start at rho points 1, 5 and 9 (1-based, ghost column included).
    _, parts = partition(_rho_only(10, 4), np_eta=2, np_xi=3)
    got = [tuple(int(v) for v in p.attrs["partition"]) for p in parts]
    assert got == [
        (0, 6, 1, 1), (1, 6, 5, 1), (2, 6, 9, 1),
        (3, 6, 1, 4), (4, 6, 5, 4), (5, 6, 9, 4),
    ]


def test_single_tile_is_left_unchanged():
    ds = _rho_only(10, 4)
    _, [part] = partition(ds, np_eta=1, np_xi=1)
    xrt.assert_identical(part, ds)


@pytest.mark.parametrize(
    ("nx", "ny", "np_xi", "np_eta"),
    [(1, 2, 1, 2), (4, 4, 1, 2), (10, 10, 2, 2), (10, 9, 3, 2), (6, 6, 2, 3)],
)
def test_rho_only_partitions_join_back(nx, ny, np_xi, np_eta):
    # Each of these failed before: all tiles of a column had the same eta size, so
    # the layout inferred from sizes was wrong.
    ds = _rho_only(nx, ny)
    _, parts = partition(ds, np_eta=np_eta, np_xi=np_xi)
    xrt.assert_identical(join_datasets(parts), ds)


def test_join_places_tiles_by_the_attribute_in_any_order():
    ds = _rho_only(10, 9)
    _, parts = partition(ds, np_eta=2, np_xi=3)
    shuffled = [parts[i] for i in (4, 0, 5, 2, 1, 3)]
    xrt.assert_identical(join_datasets(shuffled), ds)


def test_join_without_the_attribute_still_infers_the_layout():
    # Files from older roms-tools releases carry no attribute.
    ds = xr.Dataset(
        {
            "h": (("eta_rho", "xi_rho"), np.arange(12 * 12.0).reshape(12, 12)),
            "v": (("eta_v", "xi_rho"), np.arange(11 * 12.0).reshape(11, 12)),
        }
    )
    _, parts = partition(ds, np_eta=2, np_xi=3)
    for p in parts:
        del p.attrs["partition"]
    xrt.assert_identical(join_datasets(parts), ds)
