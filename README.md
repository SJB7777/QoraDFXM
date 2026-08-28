# QoraDFXM

Dark-Field X-ray Microscopy detector-image analysis: a pure Python engine with
two front-ends — a headless CLI and a PySide6 desktop app.

```
core/  pure engine (no Qt)  ←  cli/  headless front-end  ←  gui/  desktop app
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Install

```bash
uv sync                     # dev: engine + CLI + GUI (the dev group pulls qoradfxm[gui])
uv sync --no-dev            # headless: engine + CLI only, no Qt
pip install 'qoradfxm[gui]'     # desktop app
pip install qoradfxm            # batch / beamline machine — no Qt
```

`cupy` (GPU) is an opt-in extra: `uv sync --extra cuda`.

## CLI

```bash
qoradfxm info shot.h5 [--tree]                    # frames, shapes, HDF5 structure
qoradfxm convert dat/ -o tif/                     # export detector frames to TIFF
qoradfxm fit shot.h5 \
    --dataset /run/scan00001/det/eh1hama_img/data \
    --op sub_bg:/dark --op divide:flat.tif --op gamma:0.5 --op log \
    --points pts.json --out results.csv
qoradfxm ring shot.h5 --points pts.json \
    --k 0.5:2.0:0.005 --width 3 --out ring.csv --map ring_map.tif
qoradfxm gui shot.h5                              # launch the desktop app
```

`ring` sweeps the fitted ellipse's scale `k` and reports the mean brightness
along its contour (intensity per unit length), with `peak k` / FWHM printed and
the curve written to CSV. The ellipse comes from `--points` or from a Master CSV
(`--from-csv results.csv --shot-id …`). It measures on **linear** intensity —
log/gamma ops are stripped unless you pass `--keep-log`.

Ops apply in the order given:

| op | meaning |
|---|---|
| `sub_bg:SRC` / `divide:SRC` | dark subtract / flat-field divide |
| `log`, `pure_log`, `sqrt`, `gamma:0.5`, `normalize` | intensity transforms |
| `scale:1.5`, `scale:2x0.5` | resize; `sxXsy` changes the aspect ratio |
| `rotate:30` | rotate degrees, + = counter-clockwise (canvas expands) |
| `flip:h` / `flip:v` / `flip:both` | mirror |

`SRC` starting with `/` (and not an image suffix) is a dataset inside the same
HDF5; anything else is a file path. Geometric ops change the image shape — put
them before a fit, since points picked earlier no longer line up.
Without `--points`, `fit` only reports the processed image stats.

Module form (no install step): `python -m qoradfxm.cli ...`, `python -m qoradfxm.gui`.

## Python

```python
from qoradfxm.core import QoraDFXMDataset

ds = (QoraDFXMDataset.from_h5("shot.h5", "/run/scan00001/det/eh1hama_img/data")
        .sub_bg(dataset_path="/dark")
        .apply_log()
        .fit_ellipse(points))
row = ds.to_record()   # one Master-table row
```

## Dev

```bash
uv run task gui     # launch the app
uv run task cli     # qoradfxm CLI
uv run task lint    # ruff check + format --check
```
