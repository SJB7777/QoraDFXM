"""CLI: the op grammar in both directions, and the commands end to end."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import DETECTOR_PATH

from dfxm.cli import main
from dfxm.cli.spec import (
    SpecError,
    argv_for_dataset,
    format_op,
    parse_op,
    parse_ops,
)
from dfxm.core import DFXMDataset


# ------------------------------------------------------------- grammar
@pytest.mark.parametrize(
    "spec",
    [
        "log",
        "pure_log",
        "sqrt",
        "normalize",
        "gamma:0.4",
        "sub_bg:/dark",
        "divide:flat.tif",
        "scale:1.5",
        "scale:2x0.5",
        "rotate:30",
        "rotate:-12.5",
        "flip:h",
        "flip:both",
    ],
)
def test_op_specs_round_trip(spec):
    assert format_op(parse_op(spec)) == spec


def test_uniform_scale_formats_without_the_second_factor():
    assert format_op(parse_op("scale:2x2")) == "scale:2"


def test_dataset_reference_vs_file_reference():
    assert parse_op("sub_bg:/dark").params == {"dataset_path": "/dark"}
    assert parse_op("divide:/mnt/flat.tif").params == {"file_path": "/mnt/flat.tif"}
    assert parse_op(r"divide:C:\ref\flat.tif").params == {
        "file_path": r"C:\ref\flat.tif"
    }


@pytest.mark.parametrize(
    "spec, message",
    [
        ("nope", "unknown op"),
        ("scale:0", "must be > 0"),
        ("scale:1x2x3", "sx"),
        ("scale:abc", "numbers"),
        ("flip:diagonal", "flip axis"),
        ("sub_bg", "needs a source"),
        ("gamma:abc", "needs a number"),
        ("log:2", "takes no argument"),
    ],
)
def test_bad_specs_are_rejected_with_a_useful_message(spec, message):
    with pytest.raises(SpecError, match=message):
        parse_op(spec)


def test_parse_ops_keeps_the_order():
    ops = parse_ops(["scale:2", "sub_bg:/dark", "log"])
    assert [o.kind for o in ops] == ["scale", "sub_bg", "log"]


def test_argv_for_dataset_reproduces_the_recipe(tmp_path):
    ds = (
        DFXMDataset.from_array(
            np.ones((4, 4), np.float32), source_path=tmp_path / "a.tif"
        )
        .scale(2.0, 0.5)
        .sub_bg(dataset_path="/dark")
        .apply_log()
    )
    argv = argv_for_dataset(ds, out="r.csv")
    assert argv[:2] == ["fit", str(tmp_path / "a.tif")]
    assert argv.count("--op") == 3
    assert "scale:2x0.5" in argv and "sub_bg:/dark" in argv and "log" in argv
    assert argv[-2:] == ["--out", "r.csv"]


def test_argv_needs_a_source_path():
    with pytest.raises(SpecError, match="source_path"):
        argv_for_dataset(DFXMDataset.from_array(np.ones((2, 2), np.float32)))


# ------------------------------------------------------------ commands
def test_info_lists_frames_and_shapes(ring_h5, capsys):
    assert main(["info", str(ring_h5)]) == 0
    out = capsys.readouterr().out
    assert "eh1hama_img" in out and "(300, 400)" in out


def test_fit_without_points_reports_image_stats(ring_tif, capsys):
    assert main(["fit", str(ring_tif), "--op", "scale:0.5"]) == 0
    out = capsys.readouterr().out
    assert "shape=(150, 200)" in out


def test_fit_writes_a_master_row(ring_tif, points_json, tmp_path, capsys):
    out_csv = tmp_path / "master.csv"
    assert (
        main(
            ["fit", str(ring_tif), "--points", str(points_json), "--out", str(out_csv)]
        )
        == 0
    )
    import pandas as pd

    df = pd.read_csv(out_csv, encoding="utf-8-sig")
    assert len(df) == 1
    assert df.loc[0, "major_axis"] == pytest.approx(240.0, rel=1e-3)


def test_unknown_op_exits_with_code_2(ring_tif, capsys):
    assert main(["fit", str(ring_tif), "--op", "nope"]) == 2
    assert "unknown op" in capsys.readouterr().err


def test_ring_finds_the_true_radius(ring_tif, points_json, tmp_path, capsys):
    out_csv = tmp_path / "ring.csv"
    code = main(
        [
            "ring",
            str(ring_tif),
            "--points",
            str(points_json),
            "--k",
            "0.5:2.0:0.005",
            "--out",
            str(out_csv),
        ]
    )
    assert code == 0
    assert "k=1.30" in capsys.readouterr().out
    assert out_csv.exists()


def test_ring_strips_log_unless_asked(ring_tif, points_json, capsys):
    main(["ring", str(ring_tif), "--points", str(points_json), "--k", "1.0:1.6:0.01"])
    linear = capsys.readouterr().out
    main(
        [
            "ring",
            str(ring_tif),
            "--points",
            str(points_json),
            "--k",
            "1.0:1.6:0.01",
            "--op",
            "log",
        ]
    )
    stripped = capsys.readouterr().out
    assert linear == stripped  # the log op was dropped for the measurement

    main(
        [
            "ring",
            str(ring_tif),
            "--points",
            str(points_json),
            "--k",
            "1.0:1.6:0.01",
            "--op",
            "log",
            "--keep-log",
        ]
    )
    assert capsys.readouterr().out != linear


def test_ring_can_reuse_an_ellipse_from_a_master_csv(
    ring_tif, points_json, tmp_path, capsys
):
    master = tmp_path / "master.csv"
    main(["fit", str(ring_tif), "--points", str(points_json), "--out", str(master)])
    capsys.readouterr()
    assert (
        main(["ring", str(ring_tif), "--from-csv", str(master), "--k", "1.0:1.6:0.005"])
        == 0
    )
    assert "k=1.30" in capsys.readouterr().out


def test_ring_writes_the_unrolled_map(ring_tif, points_json, tmp_path):
    import tifffile

    out_map = tmp_path / "map.tif"
    main(
        [
            "ring",
            str(ring_tif),
            "--points",
            str(points_json),
            "--k",
            "1.0:1.5:0.01",
            "--angles",
            "180",
            "--map",
            str(out_map),
        ]
    )
    assert tifffile.imread(out_map).shape == (51, 180)


def test_convert_exports_a_tif_per_h5(ring_h5, tmp_path, capsys):
    out_dir = tmp_path / "tif"
    assert main(["convert", str(ring_h5), "-o", str(out_dir)]) == 0
    assert list(out_dir.glob("*.tif"))


def test_convert_reports_when_nothing_matches(tmp_path, capsys):
    assert main(["convert", str(tmp_path), "-o", str(tmp_path / "out")]) == 1
    assert "no .h5" in capsys.readouterr().err


def test_ring_needs_an_ellipse_source(ring_tif):
    with pytest.raises(SystemExit, match="--points"):
        main(["ring", str(ring_tif)])


def test_dataset_flag_selects_the_hdf5_path(ring_h5, capsys):
    assert main(["fit", str(ring_h5), "--dataset", DETECTOR_PATH]) == 0
    assert "shape=(300, 400)" in capsys.readouterr().out


def test_cli_never_imports_qt(ring_tif):
    import subprocess
    import sys

    code = "import sys, dfxm.cli;sys.exit(1 if 'PySide6' in sys.modules else 0)"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
