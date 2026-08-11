"""Core engine: geometric ops, the recipe, and the intensity transforms."""

from __future__ import annotations

import numpy as np
import pytest

from dfxm.core import DFXMDataset, warp
from dfxm.core.history import History
from dfxm.core.ops import GEOMETRIC_KINDS, NONLINEAR_KINDS, Operation, apply_op
from dfxm.core.transform import adaptive_log


@pytest.fixture
def img():
    return np.arange(60, dtype=np.float32).reshape(6, 10)


# ------------------------------------------------------------------ warp
def test_scale_changes_each_axis_independently(img):
    assert warp.scale(img, 2.0, 0.5).shape == (3, 20)
    assert warp.scale(img, 1.5).shape == (9, 15)  # sy defaults to sx


def test_scale_identity_is_a_no_op(img):
    assert np.array_equal(warp.scale(img, 1.0), img)


@pytest.mark.parametrize("factor", [0.0, -1.0])
def test_scale_rejects_non_positive_factors(img, factor):
    with pytest.raises(ValueError, match="must be > 0"):
        warp.scale(img, factor)


def test_scale_never_produces_an_empty_image(img):
    assert warp.scale(img, 0.001).shape == (1, 1)


def test_rotate_expands_the_canvas_to_keep_the_content(img):
    assert warp.rotate(img, 90).shape == (10, 6)
    grown = warp.rotate(img, 30)
    assert grown.shape[0] > img.shape[0] and grown.shape[1] > img.shape[1]


def test_rotate_without_expand_keeps_the_shape(img):
    assert warp.rotate(img, 30, expand=False).shape == img.shape


def test_rotate_by_a_full_turn_is_a_no_op(img):
    assert np.array_equal(warp.rotate(img, 360), img)


def test_flip_mirrors_the_expected_axis(img):
    assert np.array_equal(warp.flip(img, "h"), img[:, ::-1])
    assert np.array_equal(warp.flip(img, "v"), img[::-1, :])
    assert np.array_equal(warp.flip(img, "both"), img[::-1, ::-1])


def test_flip_rejects_an_unknown_axis(img):
    with pytest.raises(ValueError, match="flip axis"):
        warp.flip(img, "diagonal")


def test_rotate_fill_value_lands_in_the_corners(img):
    out = warp.rotate(img, 45, fill=-1.0)
    assert out[0, 0] == pytest.approx(-1.0)


# --------------------------------------------------------------- recipe
def test_dataset_ops_are_immutable_and_ordered(img):
    ds = DFXMDataset.from_array(img)
    grown = ds.scale(2.0, 0.5).rotate(15).flip("v")
    assert len(ds.history) == 0, "the original dataset must not change"
    assert [op.kind for op in grown.history] == ["scale", "rotate", "flip"]


def test_geometric_ops_survive_serialization(img):
    ds = DFXMDataset.from_array(img).scale(2.0, 0.5).rotate(30).flip("h")
    rebuilt = DFXMDataset.from_dict(ds.to_dict(), raw=ds.raw)
    assert rebuilt.image.shape == ds.image.shape
    assert np.allclose(rebuilt.image, ds.image, equal_nan=True)


def test_geometric_kinds_are_all_registered(img):
    for kind in GEOMETRIC_KINDS:
        assert apply_op(Operation(kind, {}), img) is not None


def test_op_labels_show_their_parameters():
    assert "2 × 0.5" in Operation("scale", {"sx": 2.0, "sy": 0.5}).label()
    assert "+30" in Operation("rotate", {"angle": 30.0}).label()
    assert "0.5" in Operation("gamma", {"gamma": 0.5}).label()


def test_history_pop_is_undo(img):
    ds = DFXMDataset.from_array(img).sqrt().rotate(10)
    assert [op.kind for op in ds.undo().history] == ["sqrt"]


def test_replace_history_recomputes_the_image(img):
    ds = DFXMDataset.from_array(img).scale(2.0)
    plain = ds.set_history(History())
    assert plain.image.shape == img.shape


# ------------------------------------------------------- linear_view
def test_linear_view_drops_only_the_nonlinear_ops(img):
    ds = DFXMDataset.from_array(img).scale(2.0).apply_log().gamma(0.5).normalize()
    kinds = [op.kind for op in ds.linear_view().history]
    assert kinds == ["scale", "normalize"]
    assert all(k not in kinds for k in NONLINEAR_KINDS)


def test_linear_view_keeps_the_geometry_so_coordinates_still_match(img):
    ds = DFXMDataset.from_array(img).scale(2.0, 0.5).apply_log()
    assert ds.linear_view().image.shape == ds.image.shape


def test_ring_profile_requires_a_fit(img):
    with pytest.raises(ValueError, match="no ellipse fit"):
        DFXMDataset.from_array(img).ring_profile()


# ----------------------------------------------------------- transform
def test_adaptive_log_survives_nan_corners():
    a = np.array([[1.0, 2.0], [np.nan, 4.0]], dtype=np.float32)
    out = adaptive_log(a)
    assert np.isfinite(out[0, 0]) and np.isnan(out[1, 0])
    assert out[1, 1] == pytest.approx(1.0)  # the max maps to 1


def test_adaptive_log_of_all_nan_does_not_raise():
    out = adaptive_log(np.full((2, 2), np.nan, dtype=np.float32))
    assert np.isnan(out).all()
