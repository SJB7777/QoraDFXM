"""Ring profile — checked against analytic truth, not against itself."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import RING_AXES, RING_K, RING_SIGMA, ellipse_geom, ellipse_points

from qoradfxm.core import QoraDFXMDataset, ring_profile
from qoradfxm.core.profile import k_axis


def ramanujan_perimeter(a: float, b: float) -> float:
    return np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))


@pytest.fixture
def profile(ring_image):
    return ring_profile(ring_image, ellipse_geom(), k=(0.5, 2.0, 0.005))


def test_peak_lands_on_the_true_ring_radius(profile):
    k_peak, intensity = profile.peak()
    assert k_peak == pytest.approx(RING_K, abs=0.002)
    assert intensity == pytest.approx(100.0 + 5.0, rel=0.02)


def test_fwhm_matches_the_gaussian_width(profile):
    assert profile.fwhm() == pytest.approx(2.3548 * RING_SIGMA, rel=0.05)


def test_perimeter_matches_ramanujans_formula(profile):
    a, b = RING_AXES
    at_one = int(np.argmin(np.abs(profile.k - 1.0)))
    assert profile.perimeter[at_one] == pytest.approx(
        ramanujan_perimeter(a, b), rel=1e-4
    )


def test_perimeter_scales_linearly_with_k(profile):
    assert profile.perimeter[-1] / profile.perimeter[0] == pytest.approx(
        profile.k[-1] / profile.k[0], rel=1e-9
    )


def test_total_is_mean_times_perimeter(profile):
    assert np.allclose(profile.total, profile.mean * profile.perimeter)


def test_arc_length_weighting_corrects_the_equal_angle_bias():
    """Equal steps in the angle parameter crowd the high-curvature ends.

    A blob sitting at the end of the major axis — where the contour moves
    slowest per unit angle — is over-represented by a naive mean over t. The
    arc-length-weighted mean must come out visibly lower.
    """
    a, b = 120.0, 30.0  # eccentric on purpose: the bias grows with a/b
    cx, cy = 200.0, 150.0
    yy, xx = np.mgrid[0:300, 0:400]
    img = np.exp(-(((xx - (cx + a)) ** 2 + (yy - cy) ** 2) / (2 * 15.0**2)))
    geom = {
        "center_x": cx,
        "center_y": cy,
        "semi_major_axis": a,
        "semi_minor_axis": b,
        "angle_major_from_x_deg": 0.0,
    }
    prof = ring_profile(
        img.astype(np.float32), geom, k=(1.0, 1.0, 1.0), n_theta=2000, keep_map=True
    )
    weighted = float(prof.mean[0])
    naive = float(np.nanmean(prof.map[0]))
    assert naive > weighted * 1.5


def test_off_image_samples_are_reported_not_silently_dropped(ring_image):
    prof = ring_profile(ring_image, ellipse_geom(), k=(0.5, 6.0, 0.5))
    assert prof.valid_frac[0] == pytest.approx(1.0)
    assert prof.valid_frac[-1] < 0.5
    assert np.isfinite(prof.mean[0])


def test_keep_map_returns_the_unrolled_ring(ring_image):
    prof = ring_profile(
        ring_image, ellipse_geom(), k=(0.8, 1.6, 0.01), n_theta=360, keep_map=True
    )
    assert prof.map.shape == (len(prof), 360)
    assert prof.theta.shape == (360,)


def test_width_averages_sub_rings_and_broadens_the_peak(ring_image):
    thin = ring_profile(ring_image, ellipse_geom(), k=(1.0, 1.6, 0.005))
    thick = ring_profile(
        ring_image, ellipse_geom(), k=(1.0, 1.6, 0.005), width=12.0, n_sub=5
    )
    assert thick.fwhm() > thin.fwhm()
    assert thick.peak()[0] == pytest.approx(thin.peak()[0], abs=0.01)


def test_accepts_a_fit_result_and_a_master_row(ring_image):
    from qoradfxm.core import FitResult

    fit = FitResult.from_points(ellipse_points())
    from_fit = ring_profile(ring_image, fit, k=(1.0, 1.6, 0.01))
    from_row = ring_profile(ring_image, fit.to_row(), k=(1.0, 1.6, 0.01))
    assert from_fit.peak()[0] == pytest.approx(from_row.peak()[0], abs=1e-9)


def test_rejects_a_degenerate_ellipse(ring_image):
    bad = ellipse_geom() | {"semi_minor_axis": 0.0}
    with pytest.raises(ValueError, match="semi-axis"):
        ring_profile(ring_image, bad)


def test_rejects_a_non_2d_image():
    with pytest.raises(ValueError, match="2-D"):
        ring_profile(np.zeros((2, 3, 4), dtype=np.float32), ellipse_geom())


def test_k_axis_is_inclusive():
    ks = k_axis(0.5, 1.5, 0.25)
    assert ks[0] == pytest.approx(0.5) and ks[-1] == pytest.approx(1.5)
    assert len(ks) == 5


def test_dataset_measures_through_linear_view(ring_image):
    """log/gamma on the display must not change a quantitative measurement."""
    plain = QoraDFXMDataset.from_array(ring_image).fit_ellipse(ellipse_points())
    logged = plain.apply_log().gamma(0.4)
    a = plain.ring_profile(k=(1.0, 1.6, 0.01))
    b = logged.ring_profile(k=(1.0, 1.6, 0.01))
    assert np.allclose(a.mean, b.mean)


def test_dataframe_round_trips_to_csv(tmp_path, profile):
    import pandas as pd

    out = tmp_path / "ring.csv"
    profile.to_csv(out)
    df = pd.read_csv(out, encoding="utf-8-sig")
    assert list(df.columns) == list(profile.to_dataframe().columns)
    assert len(df) == len(profile)
