# DFXM

Dark-Field X-ray Microscopy detector-image analysis: a pure Python engine with
two front-ends — a headless CLI and a PySide6 desktop app.

```
core/  pure engine (no Qt)  ←  cli/  headless front-end  ←  gui/  desktop app
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Install

```bash
uv sync                     # dev: engine + CLI + GUI (the dev group pulls dfxm[gui])
uv sync --no-dev            # headless: engine + CLI only, no Qt
pip install 'dfxm[gui]'     # desktop app
pip install dfxm            # batch / beamline machine — no Qt
```

`cupy` (GPU) is an opt-in extra: `uv sync --extra cuda`.

## CLI

```bash
dfxm info shot.h5 [--tree]                    # frames, shapes, HDF5 structure
dfxm convert dat/ -o tif/                     # export detector frames to TIFF
dfxm fit shot.h5 \
    --dataset /run/scan00001/det/eh1hama_img/data \
    --op sub_bg:/dark --op divide:flat.tif --op gamma:0.5 --op log \
    --points pts.json --out results.csv
dfxm gui shot.h5                              # launch the desktop app
```

Ops apply in the order given: `sub_bg:SRC`, `divide:SRC`, `log`, `pure_log`,
`sqrt`, `gamma:0.5`, `normalize`. `SRC` starting with `/` (and not an image
suffix) is a dataset inside the same HDF5; anything else is a file path.
Without `--points`, `fit` only reports the processed image stats.

Module form (no install step): `python -m dfxm.cli ...`, `python -m dfxm.gui`.

## Python

```python
from dfxm.core import DFXMDataset

ds = (DFXMDataset.from_h5("shot.h5", "/run/scan00001/det/eh1hama_img/data")
        .sub_bg(dataset_path="/dark")
        .apply_log()
        .fit_ellipse(points))
row = ds.to_record()   # one Master-table row
```

## Dev

```bash
uv run task gui     # launch the app
uv run task cli     # dfxm CLI
uv run task lint    # ruff check + format --check
```
