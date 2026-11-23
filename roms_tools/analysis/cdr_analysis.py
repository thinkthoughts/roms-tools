import logging
import re
import typing as t

import numpy as np
import xarray as xr

# Molar mass of CO2 (kg mol^-1): 44.0095 g mol^-1 = 12.0107 + 2 * 15.9994, the standard
# atomic weights of carbon and oxygen in the IUPAC 2007 table (Pure Appl. Chem. 81,
# 2131-2156, 2009). CIAAW's current abridged values (C 12.011 +- 0.002, O 15.999 +- 0.001)
# give 44.009 +- 0.004, which includes 44.0095.
_MOLAR_MASS_CO2_KG_PER_MOL = 44.0095e-3

# "release" is the CDR intervention itself; "transfer" is a parent domain receiving
# the child's boundary perturbation (ucla-roms ``upscale_output.F90`` writes
# ``ALK_add = ALK_rate - ALK_alt_rate``, same for DIC, so transfer sources
# legitimately take both signs).
SourceRole = t.Literal["release", "transfer"]
SOURCE_ROLES: t.Final[tuple[str, ...]] = t.get_args(SourceRole)

# The outermost rho row and column on each side are boundary points whose values the
# boundary condition sets. What reaches them has left the domain: ucla-roms exports
# it to an upscaled parent across the faces inside them, so the parent counts it.
_INTERIOR: t.Final[dict[str, slice]] = {
    "eta_rho": slice(1, -1),
    "xi_rho": slice(1, -1),
}


def compute_cdr_metrics(
    ds: xr.Dataset, grid_ds: xr.Dataset, source_role: SourceRole = "release"
) -> xr.Dataset:
    """
    Compute Carbon Dioxide Removal (CDR) metrics from model output.

    Calculates CDR uptake efficiency and total carbon uptake using two methods:
      1. Flux-based: from area-integrated CO2 flux differences (time-cumulative
         for uptake).
      2. DIC difference-based: from volume-integrated DIC differences

    Both integrals cover the interior rho points only. The outermost row and
    column on each side are boundary points set by the boundary condition; a
    perturbation there has already left the domain (and, in a nested experiment,
    has been exported to the parent, which counts it).

    Copies selected tracer and flux variables and computes grid cell areas
    and averaging window durations.

    Parameters
    ----------
    ds : xr.Dataset
        Model output with required variables:
        'avg_begin_time', 'avg_end_time', 'ALK_source', 'DIC_source',
        'FG_CO2', 'FG_ALT_CO2', 'hDIC', 'hDIC_ALT_CO2'.

    grid_ds : xr.Dataset
        Grid dataset with 'pm', 'pn' (inverse grid spacing).

    source_role : {"release", "transfer"}, optional
        Role of ``ALK_source``/``DIC_source`` in this domain. ``"release"`` (default)
        is the CDR intervention itself: the source must satisfy
        ``ALK_source >= 0`` and ``DIC_source <= 0``, and efficiencies are the uptake
        normalised by the cumulative source. ``"transfer"`` marks a parent domain
        that receives the child's boundary perturbation re-injected as a source
        (signed, not a new release): the sign checks are skipped and the efficiency
        variables are all-NaN, because the domain's source is not a release. Uptake
        (tonnes CO2) is computed identically for both roles.

    Returns
    -------
    ds_cdr : xr.Dataset
        Dataset containing:
        - 'area', 'window_length'
        - copied flux/tracer variables
        - 'cdr_efficiency_from_flux' and 'cdr_efficiency_from_DIC_difference'
          (dimensionless; all-NaN for ``source_role="transfer"``)
        - 'cdr_cumulative_source' (time-cumulative volume-integrated
          ``ALK_source - DIC_source``; the efficiency denominator for a release,
          the net perturbation received for a transfer)
        - 'cdr_carbon_uptake_from_flux' (cumulative flux-based uptake, tonnes CO2)
        - 'cdr_carbon_uptake_from_DIC_difference' (DIC-difference-based uptake, tonnes CO2)

        Both uptake variables carry a ``tonnes_co2_per_native_unit`` attribute (the
        scale applied to the native integrated carbon amount), and
        ``ds_cdr.attrs["cdr_source_role"]`` records `source_role`, so the dataset
        is self-describing for downstream combination.

    Notes
    -----
    Units contract. ``FG_CO2`` and ``FG_ALT_CO2`` must be an amount of carbon per area
    per second, ``hDIC`` and ``hDIC_ALT_CO2`` an amount per area (ucla-roms writes
    ``meters mmol/m^3``), ``DIC_source`` (and ``DIC_ALT_source``) an amount per second
    and ``ALK_source`` (and ``ALK_ALT_source``) equivalents per second, with lengths in
    metres; mol, mmol, umol and nmol (eq, meq, ueq, neq) are converted, so variables may
    use different prefixes. Other units, such as ``umol/kg``, CGS units or days, raise a
    ``ValueError``. Missing units are taken as the ucla-roms milli-units (with a warning
    for ``FG_CO2`` and ``hDIC``). One equivalent of alkalinity is counted as one mole in
    the cumulative source.

    Raises
    ------
    ValueError
        If `source_role` is not one of ``SOURCE_ROLES``, if ``source_role="release"``
        and the source violates the release sign constraints, or if a variable's
        ``units`` attribute breaks the units contract below.
    KeyError
        If required variables are missing from `ds` or `grid_ds`.
    """
    if source_role not in SOURCE_ROLES:
        raise ValueError(
            f"source_role must be one of {SOURCE_ROLES}, got {source_role!r}"
        )

    # Define required variables
    ds_vars = [
        "avg_begin_time",
        "avg_end_time",
        "ALK_source",
        "DIC_source",
        "FG_CO2",
        "FG_ALT_CO2",
        "hDIC",
        "hDIC_ALT_CO2",
    ]
    grid_vars = ["pm", "pn"]

    # Check that all required variables exist
    missing_ds = [var for var in ds_vars if var not in ds]
    missing_grid = [var for var in grid_vars if var not in grid_ds]

    if missing_ds:
        raise KeyError(f"Missing required variables in ds: {missing_ds}")
    if missing_grid:
        raise KeyError(f"Missing required variables in grid_ds: {missing_grid}")

    ds_cdr = xr.Dataset()

    # Copy relevant variables
    vars_to_copy = ["FG_CO2", "FG_ALT_CO2", "hDIC", "hDIC_ALT_CO2"]
    for var_name in vars_to_copy:
        ds_cdr[var_name] = ds[var_name]

    # Grid cell area
    ds_cdr["area"] = 1 / (grid_ds["pm"] * grid_ds["pn"])
    ds_cdr["area"].attrs.update(
        long_name="Grid cell area",
        units="m^2",
    )

    # Duration of each averaging window
    ds_cdr["window_length"] = ds["avg_end_time"] - ds["avg_begin_time"]
    ds_cdr["window_length"].attrs.update(
        long_name="Duration of each averaging window",
        units="s",
    )

    _validate_source(ds, source_role)

    # Units contract (see _UNIT_CONTRACT): moles (or equivalents) per native unit of
    # each variable. Quantities are combined in the native units of FG_CO2, hDIC and
    # ALK_source, so with ucla-roms' units every ratio below is exactly 1.
    mol_flux = _native_dic_amount_to_mol_multiplier(
        ds["FG_CO2"].attrs.get("units"), _FLUX_DIMS, "FG_CO2"
    )
    mol_flux_alt = _contract_factor(ds, "FG_ALT_CO2", mol_flux)
    mol_hdic = _native_dic_amount_to_mol_multiplier(
        ds["hDIC"].attrs.get("units"), _INVENTORY_DIMS, "hDIC"
    )
    mol_hdic_alt = _contract_factor(ds, "hDIC_ALT_CO2", mol_hdic)
    eq_alk = _contract_factor(ds, "ALK_source")
    mol_dic_source = _contract_factor(ds, "DIC_source")

    # Cumulative alkalinity source, in the units of ALK_source (one equivalent of
    # alkalinity counted as one mole, as in the efficiency's definition)
    source = (
        (
            (ds["ALK_source"] - ds["DIC_source"] * (mol_dic_source / eq_alk)).sum(
                dim=["s_rho", "eta_rho", "xi_rho"]
            )
            * ds_cdr["window_length"]
        )
        .cumsum(dim="time")
        .compute()
    )

    # Cumulative flux-based uptake (Method 1)
    flux = (
        (
            ((ds["FG_CO2"] - ds["FG_ALT_CO2"] * (mol_flux_alt / mol_flux)) * ds_cdr["area"])
            .isel(_INTERIOR)
            .sum(dim=["eta_rho", "xi_rho"])
            * ds_cdr["window_length"]
        )
        .cumsum(dim="time")
        .compute()
    )

    # DIC difference-based uptake (Method 2)
    diff_DIC = (
        ((ds["hDIC"] - ds["hDIC_ALT_CO2"] * (mol_hdic_alt / mol_hdic)) * ds_cdr["area"])
        .isel(_INTERIOR)
        .sum(dim=["s_rho", "eta_rho", "xi_rho"])
        .compute()
    )

    if source_role == "release":
        # Normalize by cumulative source with safe division (NaN where source=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_flux = (flux / source) * (mol_flux / eq_alk)
            ratio_diff = (diff_DIC / source) * (mol_hdic / eq_alk)
            uptake_efficiency_flux = ratio_flux.where(np.isfinite(ratio_flux))
            uptake_efficiency_diff = ratio_diff.where(np.isfinite(ratio_diff))

        _validate_uptake_efficiency(uptake_efficiency_flux, uptake_efficiency_diff)
        description_suffix = ""
    else:
        uptake_efficiency_flux = xr.full_like(source, np.nan, dtype=float)
        uptake_efficiency_diff = xr.full_like(source, np.nan, dtype=float)
        description_suffix = (
            " (not defined for source_role='transfer': the domain's source is a "
            "transfer, not a release)"
        )

    flux_scale = mol_flux * _MOLAR_MASS_CO2_KG_PER_MOL / 1000.0
    dic_scale = mol_hdic * _MOLAR_MASS_CO2_KG_PER_MOL / 1000.0
    flux_tonnes_co2 = flux * flux_scale
    diff_dic_tonnes_co2 = diff_DIC * dic_scale

    source_units = ds["ALK_source"].attrs.get("units") or "mmol"
    source_units = source_units.removesuffix("/s")

    # Store results with metadata
    ds_cdr["cdr_cumulative_source"] = source
    ds_cdr["cdr_cumulative_source"].attrs.update(
        long_name="Cumulative CDR source",
        units=source_units,
        description=(
            "Time-cumulative volume-integrated (ALK_source - DIC_source); for "
            "source_role='release' this is the denominator of the efficiency, for "
            "'transfer' it is the net perturbation received from a child domain"
        ),
    )

    ds_cdr["cdr_efficiency_from_flux"] = uptake_efficiency_flux
    ds_cdr["cdr_efficiency_from_flux"].attrs.update(
        long_name="CDR uptake efficiency (from flux differences)",
        units="nondimensional",
        description="Carbon Dioxide Removal efficiency computed using area-integrated CO2 flux differences"
        + description_suffix,
    )
    ds_cdr["cdr_efficiency_from_DIC_difference"] = uptake_efficiency_diff
    ds_cdr["cdr_efficiency_from_DIC_difference"].attrs.update(
        long_name="CDR uptake efficiency (from DIC differences)",
        units="nondimensional",
        description="Carbon Dioxide Removal efficiency computed using volume-integrated DIC differences"
        + description_suffix,
    )

    ds_cdr["cdr_carbon_uptake_from_flux"] = flux_tonnes_co2
    ds_cdr["cdr_carbon_uptake_from_flux"].attrs.update(
        long_name="CDR carbon uptake (cumulative flux differences)",
        units="tonnes CO2",
        description="Time-cumulative CO2 uptake mass from area-integrated CO2 flux differences",
        tonnes_co2_per_native_unit=flux_scale,
    )

    ds_cdr["cdr_carbon_uptake_from_DIC_difference"] = diff_dic_tonnes_co2
    ds_cdr["cdr_carbon_uptake_from_DIC_difference"].attrs.update(
        long_name="CDR carbon uptake (from DIC differences)",
        units="tonnes CO2",
        description=("CO2-equivalent mass from the volume-integrated DIC difference"),
        tonnes_co2_per_native_unit=dic_scale,
    )

    ds_cdr.attrs["cdr_source_role"] = source_role

    return ds_cdr


def _validate_uptake_efficiency(
    uptake_efficiency_flux: xr.DataArray,
    uptake_efficiency_diff: xr.DataArray,
) -> float:
    """
    Compute and log the maximum absolute difference between two uptake efficiency estimates.

    Parameters
    ----------
    uptake_efficiency_flux : xr.DataArray
        Uptake computed from fluxes.
    uptake_efficiency_diff : xr.DataArray
        Uptake computed from DIC differences.

    Returns
    -------
    max_abs_diff : float
        Maximum absolute difference between uptake_flux and uptake_diff.
    """
    abs_diff = np.abs(uptake_efficiency_flux - uptake_efficiency_diff)
    max_abs_diff = float(abs_diff.max())

    logging.info(
        "Max absolute difference between flux-based and DIC-based uptake efficiency: %.3e",
        max_abs_diff,
    )

    return max_abs_diff


def _validate_source(ds: xr.Dataset, source_role: SourceRole):
    """
    Validate ALK_source and DIC_source in a ROMS dataset for the given source role.

    The presence of both variables is always checked. For ``source_role="release"``
    the release sign constraints are also enforced:

    - 'ALK_source' must be non-negative (≥ 0).
    - 'DIC_source' must be non-positive (≤ 0).

    A ``"transfer"`` source is signed, so no sign check is applied.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset expected to contain 'ALK_source' and 'DIC_source'.
    source_role : {"release", "transfer"}
        Role of the source; the sign constraints apply only to ``"release"``.

    Raises
    ------
    KeyError
        If 'ALK_source' or 'DIC_source' are missing from the dataset.
    ValueError
        If ``source_role="release"`` and 'ALK_source' or 'DIC_source' violate the
        release constraints.
    """
    constraints = {
        "ALK_source": lambda x: x >= 0,
        "DIC_source": lambda x: x <= 0,
    }

    for var, check in constraints.items():
        if var not in ds.data_vars:
            raise KeyError(f"Dataset is missing required variable '{var}'.")
        if source_role == "release" and not check(ds[var]).all():
            sign = "negative" if var == "ALK_source" else "positive"
            raise ValueError(
                f"'{var}' contains {sign} values, which violates release constraints."
            )


def _native_carbon_amount_to_tonnes_co2_scale(
    units: str | None, expected: tuple[str, ...] | None = None
) -> float:
    """
    Scale factor from integrated ROMS-style DIC carbon amounts to tonnes CO2.

    For each mole of DIC carbon counted in the flux or inventory difference, we
    assign one mole of CO2 (air-sea CO2 exchange bookkeeping), then convert to
    mass using the molar mass of CO2:

        tonnes CO2 = n_C (mol) * M_CO2 (kg mol^-1) / (1000 kg tonne^-1)

    ``n_C`` is obtained from the native amount using the units contract of
    :func:`_native_dic_amount_to_mol_multiplier`; missing units default to mmol.
    """
    mol_per_native = _native_dic_amount_to_mol_multiplier(units, expected)
    kg_co2_per_native = mol_per_native * _MOLAR_MASS_CO2_KG_PER_MOL
    return kg_co2_per_native / 1000.0


# Units contract of the CDR metrics. ucla-roms (cdr_output.F90) writes MARBL's MKS
# units (marbl_driver.F90 sets unit_system_opt = 'mks'): FG_CO2 and FG_ALT_CO2 in
# "mmol/m^2/s", hDIC and hDIC_ALT_CO2 in "meters mmol/m^3", ALK_source and
# ALK_ALT_source in "meq/s", DIC_source and DIC_ALT_source in "mmol/s". The metrics
# integrate fluxes over grid areas in m^2 (1 / (pm * pn)) and windows in seconds, and
# inventories over areas in m^2, so each variable must have exactly the dimensions
# below; any amount prefix is converted, nothing else is.
_AMOUNT_UNITS = {"mol": 1.0, "mmol": 1e-3, "umol": 1e-6, "nmol": 1e-9}
_EQUIVALENT_UNITS = {"eq": 1.0, "meq": 1e-3, "ueq": 1e-6, "neq": 1e-9}
_LENGTH_UNITS = {"m": "m", "meter": "m", "meters": "m", "metre": "m", "metres": "m"}
_TIME_UNITS = {"s": "s", "sec": "s", "second": "s", "seconds": "s"}
# Recognised so that mass-normalised units (e.g. umol/kg) are reported as having the
# wrong dimensions; no contract variable accepts mass.
_MASS_UNITS = {"kg", "g"}
# Symbols are case-sensitive, as in UDUNITS ("M" is mega, so "Mmol" is a megamole, not
# a millimole). The one other spelling accepted is the one ucla-roms writes for its own
# passive and CDR tracers (src/tracers.F90: 'mMol m-3').
_SYMBOL_ALIASES = {"mMol": "mmol"}
# Dimensions as sorted (base, exponent) pairs; N: amount of carbon, E: equivalents
# of alkalinity, L: length, T: time, M: mass.
_FLUX_DIMS = (("L", -2), ("N", 1), ("T", -1))
_INVENTORY_DIMS = (("L", -2), ("N", 1))
_RATE_DIMS = (("N", 1), ("T", -1))
_ALK_RATE_DIMS = (("E", 1), ("T", -1))
_UNIT_CONTRACT: dict[str, tuple] = {
    "FG_CO2": _FLUX_DIMS,
    "FG_ALT_CO2": _FLUX_DIMS,
    "hDIC": _INVENTORY_DIMS,
    "hDIC_ALT_CO2": _INVENTORY_DIMS,
    "DIC_source": _RATE_DIMS,
    "DIC_ALT_source": _RATE_DIMS,
    "ALK_source": _ALK_RATE_DIMS,
    "ALK_ALT_source": _ALK_RATE_DIMS,
}
_UNIT_TOKEN = re.compile(r"^([A-Za-z]+)(?:\^|\*\*)?(-?\d+)?$")
_DIMENSION_NAMES = {
    _FLUX_DIMS: "amount per area per time (e.g. mmol/m^2/s)",
    _INVENTORY_DIMS: "amount per area (e.g. meters mmol/m^3 or mmol/m^2)",
    _RATE_DIMS: "amount per time (e.g. mmol/s)",
    _ALK_RATE_DIMS: "equivalents per time (e.g. meq/s)",
}


def _parse_units(units: str) -> tuple[float, tuple]:
    """Factor to mol (or eq), and dimensions, of a units string.

    Grammar: factors separated by spaces, ``*`` or ``.``; ``/`` divides by the next
    factor only (``mmol/m^2/s`` = mmol m^-2 s^-1; ``mmol/m^3 cm/s`` = mmol m^-3 cm s^-1);
    a factor is a unit symbol with an optional integer exponent (``m^2``, ``m**2``,
    ``m2``, ``m-2``). Supported symbols: mol, mmol, umol, nmol, eq, meq, ueq, neq,
    m (meter, metre and plurals), s (sec, second and plurals), and ``mMol`` for mmol.
    Symbols are case-sensitive: ``Mmol`` (megamole) and ``MMOL`` are refused.

    Raises
    ------
    ValueError
        On an empty or malformed string, an unsupported symbol (centimetres, mass,
        hours, days, ...), or more than one amount factor.
    """
    u = units.strip().replace("µ", "u").replace("μ", "u").replace("micromol", "umol")
    u = u.replace("**", "^").replace("·", " ").replace("*", " ")
    if not u or "(" in u or ")" in u or "//" in u or u.startswith("/") or u.endswith("/"):
        raise ValueError(f"malformed units {units!r}")
    tokens = re.findall(r"/|[^\s/]+", u)
    factor = 1.0
    dims: dict[str, int] = {}
    amounts = 0
    sign = 1
    for tok in tokens:
        if tok == "/":
            if sign == -1:
                raise ValueError(f"malformed units {units!r}")
            sign = -1
            continue
        for part in tok.split("."):
            if not part:
                raise ValueError(f"malformed units {units!r}")
            m = _UNIT_TOKEN.match(part)
            if m is None:
                raise ValueError(f"malformed units {units!r} (at {part!r})")
            symbol, exp = m.group(1), int(m.group(2) or 1)
            symbol = _SYMBOL_ALIASES.get(symbol, symbol)
            exp *= sign
            if symbol in _AMOUNT_UNITS or symbol in _EQUIVALENT_UNITS:
                if exp != 1:
                    raise ValueError(f"units {units!r} have an amount to a power")
                amounts += 1
                base = "N" if symbol in _AMOUNT_UNITS else "E"
                factor *= _AMOUNT_UNITS.get(symbol) or _EQUIVALENT_UNITS[symbol]
            elif symbol in _LENGTH_UNITS:
                base = "L"
            elif symbol in _TIME_UNITS:
                base = "T"
            elif symbol in _MASS_UNITS:
                base = "M"
            else:
                hint = (
                    "; unit symbols are case-sensitive"
                    if symbol.lower() in {*_AMOUNT_UNITS, *_EQUIVALENT_UNITS, *_LENGTH_UNITS, *_TIME_UNITS}
                    else ""
                )
                raise ValueError(
                    f"units {units!r} contain {part!r}, which is not a supported unit "
                    f"(mol, mmol, umol, nmol, eq, meq, ueq, neq, m, s){hint}"
                )
            dims[base] = dims.get(base, 0) + exp
        sign = 1
    if sign == -1:
        raise ValueError(f"malformed units {units!r}")
    if amounts != 1:
        raise ValueError(f"units {units!r} must name exactly one amount of substance")
    return factor, tuple(sorted((k, v) for k, v in dims.items() if v != 0))


def _native_dic_amount_to_mol_multiplier(
    units: str | None, expected: tuple[str, ...] | None = None, var: str = "DIC carbon"
) -> float:
    """Moles (or equivalents) per native unit, after checking the units' dimensions.

    ``expected`` is one or more of the dimension tuples of ``_UNIT_CONTRACT``; by
    default a carbon flux or inventory (``_FLUX_DIMS`` or ``_INVENTORY_DIMS``). Missing
    or blank units are taken as mmol, the ucla-roms convention, with a warning.

    Raises
    ------
    ValueError
        If the units are malformed, use an unsupported unit, or have other dimensions
        than expected (for example ``umol/kg``, mass-normalised, where an amount per
        area is expected).
    """
    if expected is None:
        expected = (_FLUX_DIMS, _INVENTORY_DIMS)
    elif expected and isinstance(expected[0][0], str):
        expected = (expected,)
    if units is None or not str(units).strip():
        logging.warning(
            "%s amount units are missing; assuming mmol for conversion to tonnes CO2.", var
        )
        return 1e-3
    try:
        factor, dims = _parse_units(str(units))
    except ValueError as e:
        raise ValueError(f"{var}: {e}") from None
    if dims not in expected:
        wanted = " or ".join(_DIMENSION_NAMES.get(d, str(d)) for d in expected)
        raise ValueError(f"{var} units {units!r} are not {wanted}.")
    return factor


def _contract_factor(ds: xr.Dataset, var: str, default: float = 1e-3) -> float:
    """Factor to mol (or eq) of ``ds[var]`` under ``_UNIT_CONTRACT``. Missing units
    give ``default`` without a warning (source terms: the ucla-roms milli-units)."""
    units = ds[var].attrs.get("units")
    if units is None or not str(units).strip():
        return default
    return _native_dic_amount_to_mol_multiplier(units, _UNIT_CONTRACT[var], var)
