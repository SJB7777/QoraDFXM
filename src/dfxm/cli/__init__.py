"""DFXM command line — the headless front-end over the Core engine.

This layer never imports Qt. It runs in a plain (GUI-less) install:

    pip install dfxm            # engine + CLI
    pip install dfxm[gui]       # + the PySide6 desktop app

Commands::

    dfxm fit shot.h5 --dataset /run/scan00001/det/d/data \
        --op sub_bg:/dark --op log --points pts.json --out results.csv
    dfxm convert dat/ --out-dir tif/          # h5 → tif export
    dfxm info shot.h5                         # frames / structure
    dfxm gui shot.h5                          # launch the desktop app

Ops run in the order given. See :mod:`dfxm.cli.spec` for the ``KIND[:SRC]``
grammar; the GUI reuses it to submit its own recipes as CLI jobs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..core import MASTER_COLUMNS, DFXMDataset, FitResult, ResultsFrame, ring_profile
from ..core import io as core_io
from .spec import OP_ARG_HELP, SpecError, parse_ops

__all__ = ["build_parser", "main"]


# --------------------------------------------------------------- helpers
def _load_dataset(path: Path, dataset_path: str | None) -> DFXMDataset:
    if dataset_path:
        return DFXMDataset.from_h5(path, dataset_path)
    if path.suffix.lower() in core_io.H5_SUFFIXES:
        return DFXMDataset.from_frame(path)  # first frame
    return DFXMDataset.from_image_file(path)


def _apply_ops(ds: DFXMDataset, op_specs) -> DFXMDataset:
    for op in parse_ops(op_specs):
        ds = ds.add_op(op.kind, **op.params)
    return ds


# -------------------------------------------------------------- commands
def cmd_fit(args) -> int:
    ds = _load_dataset(Path(args.file), args.dataset)
    ds = _apply_ops(ds, args.op)

    if not args.points:
        # No fit — just report the processed image stats.
        img = ds.image
        print(f"shot_id : {ds.meta.get('shot_id')}")
        print(f"history : {[op.kind for op in ds.history]}")
        print(f"image   : shape={img.shape} min={img.min():.4g} max={img.max():.4g}")
        return 0

    pts = json.loads(Path(args.points).read_text(encoding="utf-8"))
    ds = ds.fit_ellipse(pts)
    rec = ds.to_record()
    if args.out:
        rf = ResultsFrame()
        rf.add_row(rec)
        rf.to_csv(args.out)
        print(f"wrote {args.out} ({len(rf)} row)")
    else:
        print(json.dumps(rec, ensure_ascii=False, indent=2))
    return 0


def _parse_k(text: str) -> tuple[float, float, float]:
    """``"0.2:2.0:0.01"`` → (start, stop, step)."""
    parts = text.split(":")
    if len(parts) != 3:
        raise SystemExit(f"--k must be START:STOP:STEP (got '{text}')")
    try:
        return tuple(float(p) for p in parts)  # type: ignore[return-value]
    except ValueError:
        raise SystemExit(f"--k values must be numbers (got '{text}')") from None


def _ellipse_from_csv(csv_path: str, shot_id: str | None) -> dict:
    """Reuse an ellipse the GUI already fitted, straight from the Master CSV."""
    rf = ResultsFrame.from_csv(csv_path)
    if not len(rf):
        raise SystemExit(f"{csv_path} has no rows")
    r = 0 if shot_id is None else rf.index_of(shot_id)
    if r is None:
        raise SystemExit(f"shot_id '{shot_id}' not found in {csv_path}")
    return {c: rf.get_cell(r, c) for c in MASTER_COLUMNS}


def cmd_ring(args) -> int:
    """Mean brightness along the fitted ellipse, swept over its scale k."""
    ds = _load_dataset(Path(args.file), args.dataset)
    ds = _apply_ops(ds, args.op)
    if not args.keep_log:
        ds = ds.linear_view()  # quantitative measurement → linear intensity

    if args.from_csv:
        fit = _ellipse_from_csv(args.from_csv, args.shot_id)
    elif args.points:
        pts = json.loads(Path(args.points).read_text(encoding="utf-8"))
        fit = FitResult.from_points(pts)
    else:
        raise SystemExit("need --points PTS.json or --from-csv MASTER.csv")

    prof = ring_profile(
        ds.image,
        fit,
        k=_parse_k(args.k),
        n_theta=args.angles,
        width=args.width,
        width_unit=args.width_unit,
        n_sub=args.n_sub,
        keep_map=bool(args.map),
    )

    kp, ip = prof.peak()
    print(f"shot_id  : {ds.meta.get('shot_id')}")
    print(
        f"ellipse  : a={prof.params['semi_major_axis']:.2f} "
        f"b={prof.params['semi_minor_axis']:.2f} px, "
        f"angle={prof.params['angle_deg']:.2f}°"
    )
    print(f"k range  : {prof.k[0]:g} … {prof.k[-1]:g}  ({len(prof)} steps)")
    print(
        f"peak     : k={kp:.4f} (a·k={kp * prof.params['semi_major_axis']:.2f} px)  "
        f"I={ip:.6g}  FWHM={prof.fwhm():.4f} k"
    )
    if (worst := float(prof.valid_frac.min())) < 1.0:
        print(f"note     : some rings leave the image (min valid fraction {worst:.2f})")

    if args.out:
        prof.to_csv(args.out)
        print(f"wrote {args.out} ({len(prof)} rows)")
    if args.map:
        core_io.save_tif(prof.map, args.map)
        print(
            f"wrote {args.map} (I(k, theta), {prof.map.shape[0]}×{prof.map.shape[1]})"
        )
    return 0


def cmd_convert(args) -> int:
    """Batch-export detector frames to TIFF (the old h5totif.py script)."""
    files = core_io.iter_h5_files(args.src, recursive=not args.no_recursive)
    if not files:
        print(f"no .h5 files under {args.src}", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir)
    src_root = Path(args.src)
    failed = 0
    for f in files:
        stem = f.stem if src_root.is_file() else f"{f.parent.name}_{f.stem}"
        out_tif = out_dir / f"{stem}.tif"
        try:
            core_io.h5_to_tif(f, out_tif, dataset_path=args.dataset)
        except (OSError, ValueError, KeyError) as exc:
            print(f"FAIL {f}: {exc}", file=sys.stderr)
            failed += 1
            continue
        print(f"{f} -> {out_tif}")
    print(f"converted {len(files) - failed}/{len(files)} file(s)")
    return 1 if failed else 0


def cmd_info(args) -> int:
    path = Path(args.file)
    if path.suffix.lower() not in core_io.H5_SUFFIXES:
        img = core_io.load_image_file(path)
        print(f"{path.name}: shape={img.shape} dtype={img.dtype}")
        return 0

    frames = core_io.list_frames(path)
    print(f"{path.name}: {len(frames)} frame(s)")
    for fr in frames:
        shape = core_io.dataset_shape(path, fr.dataset_path)
        flag = "" if len(shape) == 2 else "   (not a 2-D image)"
        print(f"  {fr}  {shape}  ->  /{fr.dataset_path}{flag}")
    if args.tree:
        _print_tree(core_io.read_structure(path))
    return 0


def _print_tree(node, indent: int = 0) -> None:
    pad = "  " * indent
    if node.is_group:
        print(f"{pad}{node.name}/")
        for child in node.children or []:
            _print_tree(child, indent + 1)
    else:
        print(f"{pad}{node.name}  {node.shape} {node.dtype}")


def cmd_gui(args) -> int:
    """Launch the desktop app. Qt is imported here and nowhere else."""
    try:
        from ..gui.__main__ import main as gui_main

        return gui_main(args.files)
    except ImportError as exc:  # Qt is imported lazily inside gui_main too
        print(
            f"GUI dependencies are not installed ({exc}).\n"
            "Install them with:  pip install 'dfxm[gui]'  (or: uv sync --extra gui)",
            file=sys.stderr,
        )
        return 1


# ---------------------------------------------------------------- parser
def _version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("dfxm")
    except PackageNotFoundError:  # running from a source checkout
        return "0.0.0+source"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dfxm", description="DFXM batch analysis")
    p.add_argument("-V", "--version", action="version", version=f"dfxm {_version()}")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit", help="preprocess (+ optionally ellipse-fit) a shot")
    f.add_argument("file", help="HDF5 or image file")
    f.add_argument("--dataset", help="HDF5 dataset path (default: first frame)")
    f.add_argument("--op", action="append", metavar="KIND[:SRC]", help=OP_ARG_HELP)
    f.add_argument("--points", help="JSON file of [[x,y],...] to fit an ellipse")
    f.add_argument("--out", help="write the result row to this CSV")
    f.set_defaults(func=cmd_fit)

    r = sub.add_parser(
        "ring", help="brightness along the fitted ellipse vs. its scale k"
    )
    r.add_argument("file", help="HDF5 or image file")
    r.add_argument("--dataset", help="HDF5 dataset path (default: first frame)")
    r.add_argument("--op", action="append", metavar="KIND[:SRC]", help=OP_ARG_HELP)
    r.add_argument("--points", help="JSON file of [[x,y],...] to fit the ellipse from")
    r.add_argument("--from-csv", help="reuse an ellipse from a Master CSV instead")
    r.add_argument("--shot-id", help="which row of --from-csv (default: first)")
    r.add_argument("--k", default="0.2:2.0:0.01", metavar="START:STOP:STEP")
    r.add_argument("--angles", type=int, default=720, help="samples around the ring")
    r.add_argument("--width", type=float, default=0.0, help="ring thickness (0 = line)")
    r.add_argument("--width-unit", choices=("px", "k"), default="px")
    r.add_argument("--n-sub", type=int, default=3, help="sub-rings across --width")
    r.add_argument(
        "--keep-log",
        action="store_true",
        help="measure on the nonlinear (log/gamma) image — off by default because "
        "mean(log I) != log(mean I)",
    )
    r.add_argument("--out", help="write the profile to this CSV")
    r.add_argument("--map", help="write the unrolled I(k, theta) image to this TIFF")
    r.set_defaults(func=cmd_ring)

    c = sub.add_parser("convert", help="export HDF5 detector frames to TIFF")
    c.add_argument("src", help="an .h5 file or a directory of them")
    c.add_argument("-o", "--out-dir", default=".", help="output directory")
    c.add_argument("--dataset", help="HDF5 dataset path (default: first frame)")
    c.add_argument(
        "--no-recursive", action="store_true", help="do not descend into subfolders"
    )
    c.set_defaults(func=cmd_convert)

    i = sub.add_parser("info", help="list the frames (and structure) of a file")
    i.add_argument("file")
    i.add_argument("--tree", action="store_true", help="print the full HDF5 tree")
    i.set_defaults(func=cmd_info)

    g = sub.add_parser("gui", help="launch the desktop app (needs the gui extra)")
    g.add_argument("files", nargs="*", help="files to open on startup")
    g.set_defaults(func=cmd_gui)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except SpecError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
