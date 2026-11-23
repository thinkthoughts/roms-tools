from collections.abc import Sequence
from numbers import Integral
from pathlib import Path

import numpy as np
import xarray as xr

from roms_tools.utils import DEFAULT_NETCDF_FORMAT, NetCDFFormat, save_datasets

DIM_INFO = {
    # eta-direction
    "eta_rho": dict(axis="eta", ghost=2, edge="both"),
    "eta_u": dict(axis="eta", ghost=2, edge="both"),
    "eta_v": dict(axis="eta", ghost=1, edge="upper"),
    "eta_psi": dict(axis="eta", ghost=3, edge="both"),
    "eta_coarse": dict(axis="eta", ghost=2, edge="both"),
    # xi-direction
    "xi_rho": dict(axis="xi", ghost=2, edge="both"),
    "xi_u": dict(axis="xi", ghost=1, edge="upper"),
    "xi_v": dict(axis="xi", ghost=2, edge="both"),
    "xi_psi": dict(axis="xi", ghost=3, edge="both"),
    "xi_coarse": dict(axis="xi", ghost=2, edge="both"),
}


def _interior_widths(points: int, nparts: int, dim: str) -> list[int]:
    """Interior points of each tile, split the way ucla-roms splits them.

    ucla-roms ``mpi_setup`` (``src/roms_mpi.F90``), its reference partitioner
    ``partit`` (``Tools-Roms/partit.F``) and ``init_coarse`` (coarse grid,
    ``src/roms_read_write.F90``) give every tile ``ceil(points / nparts)`` points and
    take the surplus from the first tile (half, rounded down) and the last tile (the
    rest). When ``nparts`` divides ``points`` every tile gets ``points / nparts``.
    """
    width = -(-points // nparts)
    surplus = nparts * width - points
    widths = [width] * nparts
    widths[0] -= surplus // 2
    widths[-1] -= (surplus + 1) // 2
    if min(widths) < 1:
        raise ValueError(
            f"Splitting the {points} interior points of dimension '{dim}' into "
            f"{nparts} partitions leaves a tile with no points (ucla-roms gives every "
            f"tile {width} and takes the surplus of {surplus} from the first and last "
            "tile)."
        )
    return widths


def _compute_partition_sizes(
    total_size: int,
    nparts: int,
    ghost: int,
    edge: str,
    dim: str,
) -> list[int]:
    """Compute per-tile sizes including ghost cells."""
    if nparts == 1:
        return [total_size]

    sizes = _interior_widths(total_size - ghost, nparts, dim)

    if edge == "both":
        sizes[0] += ghost // 2
        sizes[-1] += ghost - ghost // 2
    elif edge == "upper":
        sizes[-1] += ghost
    else:
        raise ValueError(f"Unknown edge rule '{edge}' for {dim}")

    return sizes


def _rho_starts(partitioned_sizes: dict[str, list[int]], axis: str, nparts: int) -> list[int] | None:
    """1-based index, in the whole grid's rho points, of each tile's first rho point
    along ``axis``, as ucla-roms records it in the ``partition`` attribute
    (``iSW_corn + iwest`` on the first tile, ``iSW_corn + iwest + 1`` on the others).

    Read off the first fine-grid dimension of ``axis`` that is partitioned: a
    dimension with a ghost point at both ends starts its tiles on rho points, one with
    a ghost point at the upper end only (u or v points) one point earlier. ``None`` if
    ``axis`` is split but has no partitioned fine-grid dimension.
    """
    if nparts == 1:
        return [1]
    for dim, sizes in partitioned_sizes.items():
        info = DIM_INFO[dim]
        if info["axis"] != axis or "coarse" in dim:
            continue
        bounds = _cumsum(sizes)
        offset = 1 if info["edge"] == "both" else 2
        return [1] + [int(b) + offset for b in bounds[1:-1]]
    return None


def _cumsum(sizes: list[int]) -> np.ndarray:
    out = np.zeros(len(sizes) + 1, dtype=int)
    out[1:] = np.cumsum(sizes)
    return out


def partition(
    ds: xr.Dataset, np_eta: int = 1, np_xi: int = 1, include_coarse_dims: bool = True
) -> tuple[list[int], list[xr.Dataset]]:
    """Partition a ROMS (Regional Ocean Modeling System) dataset into smaller spatial
    tiles.

    This function divides the input dataset into `np_eta` by `np_xi` tiles, where each tile
    represents a subdomain of the original dataset. The partitioning is performed along
    the spatial dimensions `eta_rho`, `eta_u`, `eta_v`, `eta_coarse`, `xi_rho`, `xi_u`, `xi_v`,
    and `xi_coarse`, depending on which dimensions are present in the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The input ROMS dataset that is to be partitioned.

    np_eta : int, optional
        The number of partitions along the `eta` direction. Must be a positive integer. Default is 1.

    np_xi : int, optional
        The number of partitions along the `xi` direction. Must be a positive integer. Default is 1.

    include_coarse_dims : bool, optional
        Whether to include coarse grid dimensions (`eta_coarse`, `xi_coarse`) in the partitioning.
        If False, these dimensions will not be split. Relevant if none of the coarse resolution variables are actually used by ROMS.
        Default is True.

    Returns
    -------
    tuple[list[int], list[xr.Dataset]]
        A tuple containing two elements:

        - A list of integers representing the file numbers associated with each partition.
        - A list of `xarray.Dataset` objects, each representing a partitioned subdomain of the original dataset.

    Raises
    ------
    ValueError
        If `np_eta` or `np_xi` is not a positive integer, or if splitting a dimension into the
        specified number of tiles would leave a tile with no interior points.

    Notes
    -----
    When there is more than one tile, each tile carries ucla-roms' global attribute
    ``partition = (node, nnodes, xi_start, eta_start)``, as ucla-roms and its reference
    partitioner write it: the tile's file number, the number of tiles, and the 1-based
    indices, in the whole grid's rho points, of the tile's first rho point. ``join_datasets``
    and ucla-roms' ``ncjoin`` place tiles by it.

    Tiles follow ucla-roms' domain decomposition (``mpi_setup``): along each dimension every
    tile gets ``ceil(points / n)`` interior points, and the surplus is removed from the first
    tile (half, rounded down) and the last tile (the rest), so tilings that do not divide the
    grid evenly are supported. When ``n`` divides the number of interior points all tiles have
    the same interior size.


    Example
    -------
    >>> partitioned_file_numbers, partitioned_datasets = partition(
    ...     ds, np_eta=2, np_xi=3
    ... )
    >>> print(partitioned_file_numbers)
    [0, 1, 2, 3, 4, 5]
    >>> print([ds.sizes for ds in partitioned_datasets])
    [{'eta_rho': 50, 'xi_rho': 50}, {'eta_rho': 50, 'xi_rho': 50}, ...]

    This example partitions the dataset into 2 tiles along the `eta` direction and 3 tiles
    along the `xi` direction, resulting in a total of 6 partitions.
    """
    if (
        not isinstance(np_eta, Integral)
        or np_eta < 1
        or not isinstance(np_xi, Integral)
        or np_xi < 1
    ):
        raise ValueError("np_eta and np_xi must be positive integers")

    # Select applicable dimensions
    dims_to_partition = [
        d
        for d in DIM_INFO
        if d in ds.dims and (include_coarse_dims or "coarse" not in d)
    ]

    partitioned_sizes = {}

    for dim in dims_to_partition:
        info = DIM_INFO[dim]
        nparts = np_eta if info["axis"] == "eta" else np_xi

        partitioned_sizes[dim] = _compute_partition_sizes(
            total_size=ds.sizes[dim],
            nparts=nparts,
            ghost=info["ghost"],
            edge=info["edge"],
            dim=dim,
        )

    file_numbers = []
    partitioned_datasets = []
    nnodes = np_eta * np_xi
    xi_starts = _rho_starts(partitioned_sizes, "xi", np_xi)
    eta_starts = _rho_starts(partitioned_sizes, "eta", np_eta)

    for i in range(np_eta):
        for j in range(np_xi):
            file_numbers.append(j + i * np_xi)
            indexers = {}

            for dim, sizes in partitioned_sizes.items():
                info = DIM_INFO[dim]
                idx = i if info["axis"] == "eta" else j
                bounds = _cumsum(sizes)
                indexers[dim] = slice(bounds[idx], bounds[idx + 1])

            tile = ds.isel(**indexers)
            if nnodes > 1 and xi_starts is not None and eta_starts is not None:
                # ucla-roms' partition attribute (as Tools-Roms/partit.F writes it),
                # which identifies the tile's place in the processor grid.
                tile.attrs = {
                    **tile.attrs,
                    "partition": np.array(
                        [j + i * np_xi, nnodes, xi_starts[j], eta_starts[i]],
                        dtype=np.int32,
                    ),
                }
            partitioned_datasets.append(tile)

    return file_numbers, partitioned_datasets


def partition_netcdf(
    filepath: str | Path | Sequence[str | Path],
    np_eta: int = 1,
    np_xi: int = 1,
    output_dir: str | Path | None = None,
    include_coarse_dims: bool = True,
    format: NetCDFFormat = DEFAULT_NETCDF_FORMAT,
) -> list[Path]:
    """Partition one or more ROMS NetCDF files into smaller spatial tiles and save them to disk.

    This function divides each dataset into `np_eta` by `np_xi` tiles.
    Each tile is saved as a separate NetCDF file.

    Parameters
    ----------
    filepath : str | Path | Sequence[str | Path]
        A path or list of paths to input NetCDF files.

    np_eta : int, optional
        The number of partitions along the `eta` direction. Must be a positive integer. Default is 1.

    np_xi : int, optional
        The number of partitions along the `xi` direction. Must be a positive integer. Default is 1.

    output_dir : str | Path | None, optional
        Directory or base path to save partitioned files.
        If None, files are saved alongside the input file.

    include_coarse_dims : bool, optional
        Whether to include coarse grid dimensions (`eta_coarse`, `xi_coarse`) in the partitioning.
        If False, these dimensions will not be split. Relevant if none of the coarse resolution variables are actually used by ROMS.
        Default is True.
    format : {"NETCDF4", "NETCDF3_CLASSIC", "NETCDF3_64BIT_OFFSET", "NETCDF3_64BIT_DATA"}, optional
        NetCDF file format. Defaults to ``"NETCDF4"``.

    Returns
    -------
    list[Path]
        A list of Path objects for the filenames that were saved.
    """
    if isinstance(filepath, str | Path):
        filepaths = [Path(filepath)]
    else:
        filepaths = [Path(fp) for fp in filepath]

    all_saved_filenames = []

    for fp in filepaths:
        input_file = fp.with_suffix(".nc")
        ds = xr.open_dataset(input_file, decode_timedelta=False)

        file_numbers, partitioned_datasets = partition(
            ds, np_eta=np_eta, np_xi=np_xi, include_coarse_dims=include_coarse_dims
        )

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            base_filepath = output_dir / fp.stem
        else:
            base_filepath = fp.with_suffix("")

        ndigits = len(str(max(file_numbers)))
        paths_to_partitioned_files = [
            Path(f"{base_filepath}.{num:0{ndigits}d}") for num in file_numbers
        ]

        saved = save_datasets(
            partitioned_datasets,
            paths_to_partitioned_files,
            verbose=False,
            format=format,
        )
        all_saved_filenames.extend(saved)

    return all_saved_filenames
