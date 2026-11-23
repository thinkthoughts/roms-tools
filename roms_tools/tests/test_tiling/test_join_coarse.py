"""Joining tiles whose coarse dimensions are whole, as ucla-roms ``partit`` writes them."""

import numpy as np
import pytest
import xarray as xr

from roms_tools.tiling.join import join_datasets
from roms_tools.tiling.partition import partition


def _grid(n_xi: int, n_eta: int) -> xr.Dataset:
    """A grid-like dataset with ``n_xi`` x ``n_eta`` interior rho points and the
    coarse grid ROMS-Tools derives from it, every value distinct.
    """
    rng = np.random.default_rng(0)
    xr_, er = n_xi + 2, n_eta + 2
    xc, ec = (xr_ - 2) // 2 + 2, (er - 2) // 2 + 2
    return xr.Dataset(
        {
            "h": (("eta_rho", "xi_rho"), rng.random((er, xr_))),
            "pm": (("eta_rho", "xi_u"), rng.random((er, xr_ - 1))),
            "pn": (("eta_v", "xi_rho"), rng.random((er - 1, xr_))),
            "angle_coarse": (("eta_coarse", "xi_coarse"), rng.random((ec, xc))),
            "mask_coarse": (("eta_coarse", "xi_coarse"), np.ones((ec, xc), np.int32)),
        }
    )


def _assert_same(a: xr.Dataset, b: xr.Dataset):
    assert set(a.data_vars) == set(b.data_vars)
    for v in a.data_vars:
        assert a[v].dims == b[v].dims, v
        np.testing.assert_array_equal(a[v].values, b[v].values, err_msg=v)


@pytest.mark.parametrize(
    ("n_xi", "n_eta", "np_xi", "np_eta"),
    [
        (36, 22, 2, 2),  # even split: every coarse tile would have the same size
        (37, 23, 3, 2),  # uneven, as on a ROMS-Tools grid checked against partit
        (37, 23, 5, 4),
        (37, 23, 1, 6),
        (37, 23, 8, 1),
    ],
)
def test_whole_coarse_dims_are_copied_not_joined(n_xi, n_eta, np_xi, np_eta):
    ds = _grid(n_xi, n_eta)
    _, tiles = partition(ds, np_eta=np_eta, np_xi=np_xi, include_coarse_dims=False)
    assert all(t.sizes["xi_coarse"] == ds.sizes["xi_coarse"] for t in tiles)
    _assert_same(join_datasets(tiles), ds)


@pytest.mark.parametrize(("np_xi", "np_eta"), [(2, 2), (3, 2), (2, 1), (1, 2)])
def test_split_coarse_dims_are_still_joined(np_xi, np_eta):
    ds = _grid(36, 22)
    _, tiles = partition(ds, np_eta=np_eta, np_xi=np_xi, include_coarse_dims=True)
    assert (
        all(t.sizes["xi_coarse"] < ds.sizes["xi_coarse"] for t in tiles) or np_xi == 1
    )
    _assert_same(join_datasets(tiles), ds)


def test_tiles_copied_by_hand_join_to_the_source():
    # Independent of `partition`: slice the fine grid by hand (ucla-roms split of 37
    # interior points into 3: 12, 13, 12) and give every tile the whole coarse grid.
    ds = _grid(37, 23)
    edges = [0, 13, 26, 39]  # rho columns incl. boundary: 1+12, 13, 12+1
    tiles = []
    for k in range(3):
        a, b = edges[k], edges[k + 1]
        tile = ds.isel(
            xi_rho=slice(a, b), xi_u=slice(max(a - 1, 0), b - 1 if k < 2 else 38)
        )
        tiles.append(tile.assign_attrs(partition=np.array([k, 3, a + 1, 1], np.int32)))
    assert [t.sizes["xi_u"] for t in tiles] == [12, 13, 13]
    _assert_same(join_datasets(tiles), ds)
