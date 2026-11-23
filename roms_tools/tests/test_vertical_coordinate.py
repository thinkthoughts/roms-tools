import numpy as np
import pytest
import xarray as xr

from roms_tools.constants import UPPER_BOUND_THETA_B, UPPER_BOUND_THETA_S
from roms_tools.vertical_coordinate import (
    compute_cs,
    compute_depth,
    compute_depth_coordinates,
    sigma_stretch,
)


def test_compute_cs():
    sigma = np.linspace(-1, 0, 10)
    theta_s, theta_b = 5, 2
    cs = compute_cs(sigma, theta_s, theta_b)
    assert cs.shape == sigma.shape
    assert np.all(cs <= 0) and np.all(cs >= -1)

    with pytest.raises(
        ValueError, match=f"theta_s must be between 0 and {UPPER_BOUND_THETA_S}"
    ):
        compute_cs(sigma, UPPER_BOUND_THETA_S + 1, theta_b)

    with pytest.raises(
        ValueError, match=f"theta_b must be between 0 and {UPPER_BOUND_THETA_B}"
    ):
        compute_cs(sigma, theta_s, UPPER_BOUND_THETA_B + 1)


def test_sigma_stretch():
    theta_s, theta_b, N = 5, 2, 10
    cs, sigma = sigma_stretch(theta_s, theta_b, N, "r")
    assert cs.shape == sigma.shape
    assert isinstance(cs, xr.DataArray)
    assert isinstance(sigma, xr.DataArray)

    with pytest.raises(
        ValueError,
        match="Type must be either 'w' for vertical velocity points or 'r' for rho-points.",
    ):
        sigma_stretch(theta_s, theta_b, N, "invalid")


def test_compute_depth():
    zeta = xr.DataArray(0.5)
    h = xr.DataArray(10.0)
    hc = 5.0
    cs = xr.DataArray(np.linspace(-1, 0, 10), dims="s_rho")
    sigma = xr.DataArray(np.linspace(-1, 0, 10), dims="s_rho")

    depth = compute_depth(zeta, h, hc, cs, sigma)
    assert depth.shape == sigma.shape
    assert isinstance(depth, xr.DataArray)


def test_compute_depth_coordinates():
    grid_ds = xr.Dataset(
        {
            "h": xr.DataArray([[10, 20], [30, 40]], dims=("eta_rho", "xi_rho")),
            "Cs_r": xr.DataArray(np.linspace(-1, 0, 10), dims="s_rho"),
            "sigma_r": xr.DataArray(np.linspace(-1, 0, 10), dims="s_rho"),
            "Cs_w": xr.DataArray(np.linspace(-1, 0, 11), dims="s_w"),
            "sigma_w": xr.DataArray(np.linspace(-1, 0, 11), dims="s_w"),
        },
        attrs={"hc": 5.0},
    )

    depth = compute_depth_coordinates(grid_ds, depth_type="layer", location="rho")
    assert isinstance(depth, xr.DataArray)
    assert "eta_rho" in depth.dims and "xi_rho" in depth.dims

    with pytest.raises(ValueError, match="Invalid depth_type"):
        compute_depth_coordinates(grid_ds, depth_type="invalid")

    with pytest.raises(ValueError, match="Invalid location"):
        compute_depth_coordinates(grid_ds, location="invalid")


# CSF of ucla-roms 0.16.0 (src/set_scoord_mod.F90, compiled unchanged) at
# sigma = -0.9 and -0.5, for (theta_s, theta_b).
UCLA_ROMS_CSF = {
    (0.0, 0.0): (-8.10000000000000053e-01, -2.50000000000000000e-01),
    (0.0, 2.0): (-9.27644305782258205e-01, -4.55054233923411267e-01),
    (5.0, 0.0): (-6.01204085447860526e-01, -7.01037165451081495e-02),
    (5.0, 2.0): (-8.09019067166472428e-01, -1.51298043077170458e-01),
    (10.0, 10.0): (-9.74776354271068568e-01, -6.43218211624036751e-02),
}


@pytest.mark.parametrize(("theta_s", "theta_b"), list(UCLA_ROMS_CSF))
def test_compute_cs_matches_ucla_roms_csf(theta_s, theta_b):
    """ucla-roms takes theta_s = 0 as csrf = -sigma**2 and theta_b = 0 as no bottom
    refinement (CSF = csrf); both are the theta -> 0 limits of the stretching.
    """
    cs = compute_cs(np.array([-0.9, -0.5]), theta_s, theta_b)
    np.testing.assert_allclose(
        cs, UCLA_ROMS_CSF[(theta_s, theta_b)], rtol=1e-14, atol=0
    )


@pytest.mark.parametrize(("theta_s", "theta_b"), [(0.0, 0.0), (0.0, 2.0), (5.0, 0.0)])
def test_compute_cs_endpoints_with_zero_theta(theta_s, theta_b):
    assert compute_cs(np.float64(0.0), theta_s, theta_b) == 0.0
    assert compute_cs(np.float64(-1.0), theta_s, theta_b) == -1.0


def test_compute_cs_is_continuous_at_zero_theta():
    # theta_s -> 0: (1 - cosh(t s)) / (cosh t - 1) = -s**2 + O(t**2).
    # theta_b -> 0: (exp(t C) - 1) / (1 - exp(-t)) = C + O(t).
    sigma = np.linspace(-1, 0, 41)
    np.testing.assert_allclose(
        compute_cs(sigma, 1e-6, 2.0), compute_cs(sigma, 0.0, 2.0), rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        compute_cs(sigma, 5.0, 1e-9), compute_cs(sigma, 5.0, 0.0), rtol=0, atol=1e-9
    )


# The stretching at sigma = -0.9 and -0.5 to 18 digits, from the defining formulas
# evaluated in 50-digit arithmetic (mpmath; RLT-CSTAR-007 f011/csf_accuracy.py). The
# float64 formulas lose up to 4e-4 at theta_s = 1e-6 and give NaN at theta_s = 1e-8.
CSF_50_DIGITS = {
    (1e-08, 0.0): (-0.810000000000000039, -0.249999999999999998),
    (1e-06, 2.0): (-0.927644305782252332, -0.455054233923389324),
    (0.0001, 1e-10): (-0.80999999987944504, -0.249999999853125),
    (5.0, 1e-10): (-0.601204085459848563, -0.0701037165483676162),
    (10.0, 1e-06): (-0.367822160611032127, -0.00664805997272070456),
}


@pytest.mark.parametrize(("theta_s", "theta_b"), list(CSF_50_DIGITS))
def test_compute_cs_is_accurate_for_small_theta(theta_s, theta_b):
    cs = compute_cs(np.array([-0.9, -0.5]), theta_s, theta_b)
    np.testing.assert_allclose(cs, CSF_50_DIGITS[(theta_s, theta_b)], rtol=0, atol=1e-15)


@pytest.mark.parametrize(("theta_s", "theta_b"), [(-0.1, 2.0), (5.0, -0.1)])
def test_compute_cs_refuses_negative_theta(theta_s, theta_b):
    with pytest.raises(ValueError, match="must be between 0 and"):
        compute_cs(np.array([-0.5]), theta_s, theta_b)
