"""Uneven tilings, as ucla-roms splits them.

ucla-roms ``mpi_setup`` (``src/roms_mpi.F90``) and its reference partitioner
``partit`` (``Tools-Roms/partit.F``) give each of ``n`` tiles
``w = ceil(points / n)`` interior points and remove the surplus
``s = n * w - points`` from the first tile (``s // 2``) and the last tile
(``(s + 1) // 2``). ``init_coarse`` (``src/roms_read_write.F90``) applies the same
rule to the coarse grid. The expected sizes below were produced by compiling those
Fortran lines of ucla-roms 0.16.0 unchanged.
"""

import numpy as np
import pytest
import xarray.testing as xrt

from roms_tools import Grid
from roms_tools.tiling.join import join_datasets
from roms_tools.tiling.partition import partition


@pytest.fixture(scope="module")
def grid10():
    # 10 x 10 interior rho points: xi_rho = 12, xi_u = 11, xi_coarse = 7
    return Grid(
        nx=10, ny=10, size_x=500, size_y=500, center_lon=-20, center_lat=0, rot=0
    )


def test_ten_points_on_three_tiles_matches_ucla_roms(grid10):
    # ucla-roms 0.16.0 mpi_setup, gnx = 10, npx = 3: interior widths 3, 4, 3;
    # xi_rho file sizes 4, 4, 4; xi_u file sizes 3, 4, 4.
    # init_coarse, gnxc = 5, np_xi = 3: coarse widths 2, 2, 1 -> xi_coarse 3, 2, 2.
    _, parts = partition(grid10.ds, np_eta=1, np_xi=3)
    assert [p.sizes["xi_rho"] for p in parts] == [4, 4, 4]
    assert [p.sizes["xi_u"] for p in parts] == [3, 4, 4]
    assert [p.sizes["xi_coarse"] for p in parts] == [3, 2, 2]
    assert [p.sizes["eta_rho"] for p in parts] == [12, 12, 12]


def test_uneven_tiles_start_where_ucla_roms_starts_them(grid10):
    # mpi_setup iSW_corn for gnx = 10, npx = 3 is 0, 3, 7: tile i holds rho columns
    # iSW_corn + 1 ... iSW_corn + width (plus the boundary ghost column).
    _, parts = partition(grid10.ds, np_eta=1, np_xi=3)
    lon = grid10.ds["lon_rho"].values
    np.testing.assert_array_equal(parts[0]["lon_rho"].values, lon[:, 0:4])
    np.testing.assert_array_equal(parts[1]["lon_rho"].values, lon[:, 4:8])
    np.testing.assert_array_equal(parts[2]["lon_rho"].values, lon[:, 8:12])


@pytest.mark.parametrize(
    ("np_eta", "np_xi", "include_coarse_dims"),
    [
        (1, 3, True),
        (3, 1, True),
        (2, 5, True),
        (5, 2, True),
        (3, 3, True),
        # 5 coarse interior points on 4 tiles leave an empty coarse tile (see
        # test_refusal_names_the_interior_point_count), so these keep coarse dims whole.
        (3, 4, False),
        (4, 3, False),
        (4, 4, False),
    ],
)
def test_uneven_partition_joins_back_to_the_original(
    grid10, np_eta, np_xi, include_coarse_dims
):
    _, parts = partition(
        grid10.ds, np_eta=np_eta, np_xi=np_xi, include_coarse_dims=include_coarse_dims
    )
    joined = join_datasets(parts)
    names = [
        v
        for v in grid10.ds.data_vars
        if include_coarse_dims or not any("coarse" in d for d in grid10.ds[v].dims)
    ]
    xrt.assert_identical(joined[names], grid10.ds[names])


def test_even_partition_unchanged(grid10):
    # 10 points on 2 tiles divides evenly: sizes are those of the previous rule.
    _, parts = partition(grid10.ds, np_eta=1, np_xi=2, include_coarse_dims=False)
    assert [p.sizes["xi_rho"] for p in parts] == [6, 6]
    assert [p.sizes["xi_u"] for p in parts] == [5, 6]


def test_tiling_with_an_empty_tile_is_refused(grid10):
    # 10 points on 8 tiles: w = 2, s = 6, first and last tile get 2 - 3 = -1 points.
    with pytest.raises(ValueError, match="leaves a tile with no points"):
        partition(grid10.ds, np_eta=1, np_xi=8, include_coarse_dims=False)


def test_refusal_names_the_interior_point_count(grid10):
    # xi_coarse has 7 points of which 5 are interior; 5 points on 4 tiles leave
    # an empty tile (w = 2, s = 3, last tile 2 - 2 = 0).
    with pytest.raises(ValueError, match="5 interior points of dimension 'xi_coarse'"):
        partition(grid10.ds, np_eta=1, np_xi=4)
