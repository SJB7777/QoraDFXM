# DFXM — Engine-First Architecture

## Layers

```
src/dfxm/
  core/            ← Pure engine. ZERO Qt/GUI imports. numpy/pandas/h5py/tifffile only.
    io.py          h5 / image loading + TIFF export, FramePath, H5Node
    transform.py   pixel transforms (adaptive_log)
    fitting.py     ellipse least-squares (Halir–Flusser) + geometry + Sampson error
    ops.py         Operation (kind+params) + OP_REGISTRY (pure funcs) + apply_op
    history.py     History — immutable ordered op sequence, JSON (de)serialize
    results.py     FitResult + ResultsFrame (pandas Master table, no Qt)
    dataset.py     DFXMDataset — fluent immutable domain object
  cli/             ← Headless front-end. Imports Core; never imports Qt.
    __init__.py    subcommands: fit / convert / info / gui, build_parser(), main()
    spec.py        recipe ⇄ argv translation (`--op sub_bg:/dark`), pure
    __main__.py    python -m dfxm.cli
  gui/             ← View only. Calls Core, reflects results. No pixel math.
    __main__.py    launcher; `--cli` routes to dfxm.cli (single frozen binary)
    cli_bridge.py  CliJob — runs `dfxm ...` as a QProcess, streams output
    models.py      MasterTableModel (QAbstractTableModel) over Core ResultsFrame
    image_view.py  dumb display of a processed ndarray
    main_window.py orchestration; DocumentSession.ds holds a DFXMDataset
```

Dependency direction: `core ← cli ← gui`. Nothing points back.
Verified invariants: `import dfxm.cli` pulls in **no** PySide6; the base install
has no Qt at all (Qt lives in the `gui` extra).

## Two front-ends, one engine

| | install | entry point |
|---|---|---|
| headless / beamline / batch | `pip install dfxm` | `dfxm fit\|convert\|info` |
| desktop | `pip install dfxm[gui]` | `dfxm gui`, `dfxm-gui`, `python -m dfxm.gui` |

`dfxm gui` imports Qt lazily inside the command, so a Qt-less install still
parses every other subcommand and fails with an install hint only if the GUI is
actually asked for.

## GUI → CLI

Batch work started from the GUI goes out through the same command line a user
could have typed, as a child process (`gui/cli_bridge.py`):

```python
from .cli_bridge import CliJob, argv_for_dataset
job = CliJob(self)
job.line.connect(self._log)                    # stdout/stderr, line by line
job.finished_ok.connect(self._on_job_done)     # exit code
job.start(argv_for_dataset(doc.ds, points_file=pts, out="results.csv"))
```

`cli.spec` owns both directions of the translation (`parse_op` /
`format_op`), so "what the GUI runs" and "what the CLI accepts" cannot drift.
A subprocess (rather than an in-process call) keeps the event loop free, makes
a job cancellable, contains crashes, and yields a printable, reproducible
command (`command_line(argv)`).

## DFXMDataset — fluent, immutable

Every op returns a NEW dataset sharing the (never-mutated) `raw` array and a
grown `History`. The recipe is *data, not code* → enables Replay, Undo, and
SQLite session storage.

```python
ds = (
    DFXMDataset.from_h5(path, "/run/s1/det/d/data")
    .sub_bg(dataset_path="/dark")  # append op, return new ds
    .divide(file_path="flat.tif")
    .apply_log()
    .fit_ellipse(points)
)  # ds.fit = FitResult
img = ds.image  # raw + history replayed (memoized per instance)
row = ds.to_record()  # one Master-table row
d = ds.to_dict()  # JSON-serializable recipe+result (→ SQLite, Phase 3)
```

- **Reference data is a location, not an array.** Ops store `dataset_path`
  (same h5) or `file_path`; arrays resolve lazily at `.image` time via
  `source_path`. Keeps `History` serializable.
- `bg_applied` / `log_scale` are derived from the history.
- `.undo()` → dataset with the last op dropped.

## GUI ↔ Core contract

- `DocumentSession.ds` is the single data model per open image.
- Preprocessing chain = `doc.ds.history`, shown/managed in the **오브젝트 tab**
  (`⚙ 전처리` group). Dark/Flat sources chosen from the 구조/파일 sidebar
  context menus. Log is just a `log` op.
- On any recipe change the GUI calls `view.update_processed(doc.ds.image)` —
  Core computes, the view only displays.
- Fit: `doc.ds = doc.ds.fit_ellipse(pts)`; the Master row is `doc.ds.to_record()`.
- `MasterTableModel` is a Qt adapter; the data lives in Core `ResultsFrame`.

## Master schema (source of truth = `core.results.MASTER_COLUMNS`)

`shot_id, status, center_x, center_y, major_axis, minor_axis, angle_deg,
fit_error, points_json, bg_applied, log_scale`

## Roadmap fit

- Phase 1 (done): viewer, preproc chain, ellipse fit, Master table, CSV.
- Phase 2 (done): shot ↔ row 1:1 (`ResultsFrame.upsert_row`/`index_of` keyed by
  shot_id; `MainWindow._shot_map` maps shot_id → file+frame). Populate shots,
  row-click switches shot (`_goto_shot`), fit auto-updates the row in place,
  EXCLUDE flag (`_toggle_exclude`) survives re-fitting and dims the row.
- Phase 3: SQLite `.dfxm_proj` autosave/restore via `DFXMDataset.to_dict()` /
  `ResultsFrame`; ROI-drag ↔ table live sync. Undo/Redo already supported by the
  immutable history. Session persistence belongs in `core/` (a `.dfxm_proj` must
  be readable headlessly) — the old empty `dfxm/session/` stub was removed.
- Next for GUI → CLI: a "batch this recipe over N shots" panel on top of
  `CliJob` (queue, progress, cancel) — the plumbing is there, the UI is not.

Preproc ops: sub_bg, divide, log (adaptive), pure_log, sqrt, gamma(γ), normalize.
