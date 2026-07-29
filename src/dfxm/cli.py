"""DFXM batch CLI — the terminal front-end over the Core engine.

Same fluent pipeline as the GUI, scripted:

    dfxm fit shot.h5 --dataset /run/s1/det/d/data \
        --op sub_bg:/dark --op log \
        --points pts.json --out results.csv

Ops run in the order given. A source after ':' starting with '/' is a dataset
in the SAME h5; anything else is treated as an external file path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core import DFXMDataset, ResultsFrame
from .core import io as core_io


def _load_dataset(args) -> DFXMDataset:
    path = Path(args.file)
    if args.dataset:
        return DFXMDataset.from_h5(path, args.dataset)
    if path.suffix.lower() in core_io.H5_SUFFIXES:
        return DFXMDataset.from_frame(path)  # first frame
    return DFXMDataset.from_image_file(path)


def _apply_ops(ds: DFXMDataset, op_specs) -> DFXMDataset:
    for spec in op_specs or []:
        kind, _, src = spec.partition(":")
        if kind in ("log", "pure_log", "sqrt", "normalize"):
            ds = ds.add_op(kind)
        elif kind == "gamma":
            ds = ds.add_op("gamma", gamma=float(src) if src else 0.5)
        elif kind in ("sub_bg", "divide"):
            if not src:
                raise SystemExit(f"op '{kind}' needs a source, e.g. {kind}:/dark")
            key = "dataset_path" if src.startswith("/") else "file_path"
            ds = ds.add_op(kind, **{key: src})
        else:
            raise SystemExit(f"unknown op: {kind}")
    return ds


def cmd_fit(args) -> int:
    ds = _load_dataset(args)
    ds = _apply_ops(ds, args.op)

    if args.points:
        pts = json.loads(Path(args.points).read_text())
        ds = ds.fit_ellipse(pts)
        rec = ds.to_record()
        rf = ResultsFrame()
        rf.add_row(rec)
        if args.out:
            rf.to_csv(args.out)
            print(f"wrote {args.out} ({len(rf)} row)")
        else:
            print(json.dumps(rec, ensure_ascii=False, indent=2))
    else:
        # No fit — just report the processed image stats.
        img = ds.image
        print(f"shot_id : {ds.meta.get('shot_id')}")
        print(f"history : {[op.kind for op in ds.history]}")
        print(f"image   : shape={img.shape} min={img.min():.4g} max={img.max():.4g}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dfxm", description="DFXM batch analysis")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit", help="preprocess (+ optionally ellipse-fit) a shot")
    f.add_argument("file", help="HDF5 or image file")
    f.add_argument("--dataset", help="HDF5 dataset path (default: first frame)")
    f.add_argument(
        "--op",
        action="append",
        metavar="KIND[:SRC]",
        help="pipeline op, repeatable & ordered: sub_bg:/dark, divide:flat.tif, "
        "log, pure_log, sqrt, gamma:0.5, normalize",
    )
    f.add_argument("--points", help="JSON file of [[x,y],...] to fit an ellipse")
    f.add_argument("--out", help="write the result row to this CSV")
    f.set_defaults(func=cmd_fit)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
