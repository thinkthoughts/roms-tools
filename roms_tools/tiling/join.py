from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

import numpy as np
import xarray as xr

from roms_tools.utils import (
    DEFAULT_NETCDF_FORMAT,
    NETCDF_WRITE_ENGINE,
    FilePaths,
    NetCDFFormat,
    _path_list_from_input,
)


def open_partitions(files: FilePaths) -> xr.Dataset:
    """
    Open partitioned ROMS netCDF files as a single dataset.

    Parameters
    ----------
    files: str | List[str | Path]
        List or wildcard pattern describing files to join,
        e.g. "roms_rst.20121209133435.*.nc"

    Returns
    -------
    xarray.Dataset
        Dataset containing unified partitioned datasets
    """
    filepaths = _path_list_from_input(files)
    datasets = [xr.open_dataset(p, decode_timedelta=True) for p in sorted(filepaths)]
    joined = join_datasets(datasets)
    return joined


def join_netcdf(
    files: FilePaths,
    output_path: Path | None = None,
    format: NetCDFFormat = DEFAULT_NETCDF_FORMAT,
) -> Path:
    """
    Join partitioned NetCDFs into a single dataset.

    Parameters
    ----------
    files : str | List[str | Path]
        List or wildcard pattern describing files to join,
        e.g. "roms_rst.20121209133435.*.nc"

    output_path : Path, optional
        If provided, the joined dataset will be saved to this path.
        Otherwise, the common base of pattern (e.g. roms_rst.20121209133435.nc)
        will be used.
    format : {"NETCDF4", "NETCDF3_CLASSIC", "NETCDF3_64BIT_OFFSET", "NETCDF3_64BIT_DATA"}, optional
        NetCDF file format. Defaults to ``"NETCDF4"``.

    Returns
    -------
    Path
        The path of the saved file
    """
    filepaths = _path_list_from_input(files)
    # Determine output path if not provided
    if output_path is None:
        # e.g. roms_rst.20120101120000.023.nc -> roms_rst.20120101120000.nc
        output_path = filepaths[0].with_suffix("").with_suffix(".nc")

    joined = open_partitions(cast(FilePaths, filepaths))
    joined.to_netcdf(output_path, format=format, engine=NETCDF_WRITE_ENGINE)
    print(f"Saved joined dataset to: {output_path}")

    return output_path


def _find_transitions(dim_sizes: list[int]) -> list[int]:
    """Finds the indices of all transitions in a list of dimension sizes.

    A transition is a point where the dimension size changes from the previous one.
    This function is used to determine the number of partitions (e.g., np_eta or np_xi).

    Parameters
    ----------
    dim_sizes : list[int]
        A list of integer sizes for a given dimension across multiple datasets.

    Returns
    -------
    List[int]
        A list of indices where a transition was detected.
    """
    transitions: list[int] = []
    if len(dim_sizes) < 2:
        return transitions

    for i in range(1, len(dim_sizes)):
        if dim_sizes[i] != dim_sizes[i - 1]:
            transitions.append(i)
    return transitions


def _find_common_dims(
    direction: Literal["xi", "eta"], datasets: Sequence[xr.Dataset]
) -> list[str]:
    """Finds all common dimensions along the xi or eta direction amongst a list of Datasets.

    Parameters
    ----------
    direction: str ("xi" or "eta")
        The direction in which to seek a common dimension
    datasets: Sequence[xr.Dataset]:
        The datasets in which to look

    Returns
    -------
    common_dim: list[str]
        The dimensions common to all specified datasets along 'direction'
    """
    if direction not in ["xi", "eta"]:
        raise ValueError("'direction' must be 'xi' or 'eta'")
    dims = []
    for point in ["rho", "u", "v"]:
        if all(f"{direction}_{point}" in d.dims for d in datasets):
            dims.append(f"{direction}_{point}")
    if not dims:
        raise ValueError(f"No common point found along direction {direction}")
    return dims


def _infer_partition_layout_from_datasets(
    datasets: Sequence[xr.Dataset],
) -> tuple[int, int]:
    """Infer np_eta, np_xi from datasets."""
    nd = len(datasets)
    if nd == 1:
        return 1, 1

    eta_dims = _find_common_dims("eta", datasets)
    first_eta_transition = nd

    for eta_dim in eta_dims:
        dim_sizes = [ds.sizes.get(eta_dim, 0) for ds in datasets]
        eta_transitions = _find_transitions(dim_sizes)
        if eta_transitions and (min(eta_transitions) < first_eta_transition):
            first_eta_transition = min(eta_transitions)
    if first_eta_transition < nd:
        np_xi = first_eta_transition
        np_eta = nd // np_xi
        return np_xi, np_eta
    # If we did not successfully find np_xi,np_eta using eta points
    # then we have a single-column grid:

    return nd, 1


def _layout_from_partition_attribute(
    datasets: Sequence[xr.Dataset],
) -> tuple[int, int, list[xr.Dataset]] | None:
    """np_xi, np_eta and the datasets in file order, from ucla-roms' ``partition``
    global attribute ``(node, nnodes, xi_start, eta_start)``.

    ucla-roms writes it on every per-rank file, as does its reference partitioner
    ``Tools-Roms/partit.F`` and :func:`roms_tools.tiling.partition.partition`.
    ``None`` unless every dataset carries a consistent attribute: the same ``nnodes``,
    equal to the number of datasets, and one tile per (xi_start, eta_start) pair.
    """
    entries = []
    for ds in datasets:
        value = ds.attrs.get("partition")
        if value is None:
            return None
        flat = np.asarray(value).ravel()
        if flat.size != 4 or not np.issubdtype(flat.dtype, np.integer):
            return None
        entries.append(tuple(int(v) for v in flat))
    nd = len(datasets)
    if {e[1] for e in entries} != {nd}:
        return None
    xi_starts = sorted({e[2] for e in entries})
    eta_starts = sorted({e[3] for e in entries})
    np_xi, np_eta = len(xi_starts), len(eta_starts)
    positions = {(e[3], e[2]) for e in entries}
    if np_xi * np_eta != nd or len(positions) != nd:
        return None
    order = sorted(range(nd), key=lambda k: (entries[k][3], entries[k][2]))
    return np_xi, np_eta, [datasets[k] for k in order]


def _coarse_is_whole(
    dim: str, slices: Sequence[xr.DataArray], fine_rho_points: int | None
) -> bool:
    """Whether every tile holds the whole coarse dimension ``dim``.

    ucla-roms ``partit`` (``Tools-Roms/partit.F``) and
    :func:`roms_tools.tiling.partition.partition` with ``include_coarse_dims=False``
    leave ``xi_coarse`` and ``eta_coarse`` unsplit, so every tile holds all
    ``(n - 2) // 2 + 2`` coarse points of a fine grid of ``n`` rho points
    (:func:`roms_tools.setup.grid._f2c_xdir`); ``ncjoin`` copies such a dimension
    rather than joining it. Split coarse tiles add up to that size instead, so with
    more than one tile the two cases are told apart by size.
    """
    if not dim.endswith("_coarse") or len(slices) < 2 or fine_rho_points is None:
        return False
    whole = (fine_rho_points - 2) // 2 + 2
    return all(s.sizes[dim] == whole for s in slices)


def _fine_rho_points(tiles: Sequence[xr.Dataset], axis: str) -> int | None:
    """Rho points of the joined fine grid along ``axis`` ("xi" or "eta"), from one
    row (xi) or column (eta) of tiles; ``None`` if the tiles carry no rho or
    u/v dimension along it."""
    staggered = "xi_u" if axis == "xi" else "eta_v"
    if all(f"{axis}_rho" in t.sizes for t in tiles):
        return sum(t.sizes[f"{axis}_rho"] for t in tiles)
    if all(staggered in t.sizes for t in tiles):
        return sum(t.sizes[staggered] for t in tiles) + 1
    return None


def join_datasets(datasets: Sequence[xr.Dataset]) -> xr.Dataset:
    """Take a sequence of partitioned Datasets and return a joined Dataset.

    Tiles are placed by ucla-roms' ``partition`` global attribute when every dataset
    carries it, otherwise by the layout inferred from their dimension sizes. Coarse
    dimensions that every tile holds whole, as ucla-roms ``partit`` writes them, are
    copied rather than joined, as ``ncjoin`` does.
    """
    if (layout := _layout_from_partition_attribute(datasets)) is not None:
        np_xi, np_eta, datasets = layout
    else:
        np_xi, np_eta = _infer_partition_layout_from_datasets(datasets)

    # Arrange into grid
    grid = [[datasets[j + i * np_xi] for j in range(np_xi)] for i in range(np_eta)]

    fine_xi = _fine_rho_points(grid[0], "xi")
    fine_eta = _fine_rho_points([row[0] for row in grid], "eta")

    # Join each row (along xi_*)
    rows_joined = []
    for row in grid:
        all_vars = set().union(*(ds.data_vars for ds in row))
        row_dataset = xr.Dataset()

        for varname in all_vars:
            var_slices = [ds[varname] for ds in row if varname in ds]
            xi_dims = [dim for dim in var_slices[0].dims if dim.startswith("xi_")]

            if not xi_dims or _coarse_is_whole(xi_dims[0], var_slices, fine_xi):
                row_dataset[varname] = var_slices[0]
            else:
                xi_dim = xi_dims[0]
                row_dataset[varname] = xr.concat(
                    var_slices, dim=xi_dim, combine_attrs="override"
                )

        rows_joined.append(row_dataset)

    # Join all rows (along eta_*)
    final_dataset = xr.Dataset()
    all_vars = set().union(*(ds.data_vars for ds in rows_joined))

    for varname in all_vars:
        var_slices = [ds[varname] for ds in rows_joined if varname in ds]
        eta_dims = [dim for dim in var_slices[0].dims if dim.startswith("eta_")]

        if not eta_dims or _coarse_is_whole(eta_dims[0], var_slices, fine_eta):
            final_dataset[varname] = var_slices[0]
        else:
            eta_dim = eta_dims[0]
            final_dataset[varname] = xr.concat(
                var_slices, dim=eta_dim, combine_attrs="override"
            )
    # Copy attributes from first dataset; the tile's position does not apply to the
    # joined dataset
    final_dataset.attrs = {
        k: v for k, v in datasets[0].attrs.items() if k != "partition"
    }

    return final_dataset
