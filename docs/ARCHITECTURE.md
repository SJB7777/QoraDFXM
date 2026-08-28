# QoraDFXM — Engine-First Architecture

## Layers

```
src/qoradfxm/
  core/            ← Pure engine. ZERO Qt/GUI imports. numpy/pandas/h5py/tifffile only.
    io.py          h5 / image loading + TIFF export, FramePath, H5Node
    transform.py   intensity transforms (adaptive_log)
    warp.py        geometric transforms (scale / rotate / flip) via OpenCV
    fitting.py     ellipse least-squares (Halir–Flusser) + geometry + Sampson error
    ops.py         Operation (kind+params) + OP_REGISTRY (pure funcs) + apply_op
    history.py     History — immutable ordered op sequence, JSON (de)serialize
    results.py     FitResult + ResultsFrame (pandas Master table, no Qt)
    profile.py     ring_profile — brightness along the fitted ellipse vs. scale k
    dataset.py     QoraDFXMDataset — fluent immutable domain object
  cli/             ← Headless front-end. Imports Core; never imports Qt.
    __init__.py    subcommands: fit / convert / info / gui, build_parser(), main()
    spec.py        recipe ⇄ argv translation (`--op sub_bg:/dark`), pure
    __main__.py    python -m qoradfxm.cli
  gui/             ← View only. Calls Core, reflects results. No pixel math.
    __main__.py    launcher; `--cli` routes to qoradfxm.cli (single frozen binary)
    cli_bridge.py  CliJob — runs `qoradfxm ...` as a QProcess, streams output
    ring_panel.py  링 프로파일 tab — live I(k) plot driven by the ellipse ROI
    models.py      MasterTableModel (QAbstractTableModel) over Core ResultsFrame
    image_view.py  dumb display of a processed ndarray
    main_window.py orchestration; DocumentSession.ds holds a QoraDFXMDataset
```

Dependency direction: `core ← cli ← gui`. Nothing points back.
Verified invariants: `import qoradfxm.cli` pulls in **no** PySide6; the base install
has no Qt at all (Qt lives in the `gui` extra).

## Two front-ends, one engine

| | install | entry point |
|---|---|---|
| headless / beamline / batch | `pip install qoradfxm` | `qoradfxm fit\|convert\|info` |
| desktop | `pip install qoradfxm[gui]` | `qoradfxm gui`, `qoradfxm-gui`, `python -m qoradfxm.gui` |

`qoradfxm gui` imports Qt lazily inside the command, so a Qt-less install still
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

## QoraDFXMDataset — fluent, immutable

Every op returns a NEW dataset sharing the (never-mutated) `raw` array and a
grown `History`. The recipe is *data, not code* → enables Replay, Undo, and
SQLite session storage.

```python
ds = (
    QoraDFXMDataset.from_h5(path, "/run/s1/det/d/data")
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

## Who owns which state

Every display bug so far came from the same shape: one setting living in a
widget, a QSettings key, a document dict, and the recipe at once. Three tiers
now, and nothing is duplicated across them:

| tier | lives in | examples |
|---|---|---|
| document, Core-owned | `doc.ds.history` | log, gamma, sub_bg, scale/rotate/flip |
| document, view-owned | `doc.prefs` (`gui/prefs.py` `ViewPrefs`) | colormap, over-max, scale bar, tool |
| app-scoped defaults | `MainWindow._app_prefs` → QSettings | what the *next* opened document starts with |

Consequences worth remembering:

- A panel widget is never a store. Handlers call `_edit_prefs(**changes)`,
  which updates the app defaults *and* the current document, then push the
  change to the view. `_sync_panel_from(doc.prefs)` re-reads on tab change.
- The Log checkbox is a *view* of `ds.log_scale`. Because it is a
  cross-file preference stored per document, `_make_dataset` applies the op to
  every newly created dataset — otherwise the box says Log and the pixels are
  linear (the bug that motivated this split).
- `ViewPrefs.apply_to(view)` is the only path from prefs to an `ImageView`.

## Tests

`tests/` runs head-less (`QT_QPA_PLATFORM=offscreen`, set in `conftest.py`
before Qt loads) against synthetic data — a Gaussian ring on a known ellipse —
so nothing depends on beamline files. `uv run task test`.

Ground truths worth keeping: the ring peak/FWHM/perimeter versus analytic
values, the arc-length weighting actually correcting the equal-angle bias, the
CLI op grammar round-tripping, and the GUI regressions (log across files,
arrow-key preview, per-document prefs). GUI tests wipe QSettings per window —
`MainWindow` restores `last_file`, which otherwise leaks between tests.

## GUI ↔ Core contract

- `DocumentSession.ds` is the single data model per open image.
- Preprocessing chain = `doc.ds.history`, shown/managed in the **오브젝트 tab**
  (`⚙ 전처리` group). Dark/Flat sources chosen from the 구조/파일 sidebar
  context menus. Log is just a `log` op.
- On any recipe change the GUI calls `view.update_processed(doc.ds.image)` —
  Core computes, the view only displays.
- Fit: `doc.ds = doc.ds.fit_ellipse(pts)`; the Master row is `doc.ds.to_record()`.
- `MasterTableModel` is a Qt adapter; the data lives in Core `ResultsFrame`.

## Ring profile — I(k)

`core.profile.ring_profile(img, fit, k=(0.2, 2.0, 0.01), n_theta=720, width=…)`
scales the fitted ellipse by `k` about its centre and returns the mean
brightness along the contour — the intensity *per unit contour length*.

- **Arc-length weighting.** Equal steps in the parameter `t` are not equal steps
  in arc length, so each sample is weighted by `ds/dt`. Verified against
  Ramanujan's perimeter (3e-6 relative) and a synthetic Gaussian ring
  (peak `k` exact, FWHM 0.1185 vs analytic 0.1177).
- **Linear intensity only.** `mean(log I) ≠ log(mean I)`, so both the CLI and
  the GUI measure through `QoraDFXMDataset.linear_view()`, which drops
  `log/pure_log/sqrt/gamma` (`core.ops.NONLINEAR_KINDS`) but keeps geometry and
  background correction — the coordinates still match what the user picked.
- Sampling is bilinear (`cv2.remap`); samples off the image become NaN and are
  reported as `valid_frac`. `width` (px or k) averages `n_sub` sub-rings.
- `RingProfile` carries k / mean / std / total / perimeter / valid_frac plus
  `peak()` (parabola-refined) and `fwhm()`, and `keep_map=True` also returns the
  unrolled `I(k, θ)` array.
- Surfaces: `qoradfxm ring …` (CSV + optional `--map` TIFF, ellipse from `--points`
  or `--from-csv` a Master row) and the GUI 링 프로파일 tab, which recomputes on
  ellipse-ROI drag through a 120 ms coalescing timer.

## Master schema (source of truth = `core.results.MASTER_COLUMNS`)

`shot_id, status, center_x, center_y, major_axis, minor_axis, angle_deg,
fit_error, points_json, bg_applied, log_scale`

## Roadmap fit

- Phase 1 (done): viewer, preproc chain, ellipse fit, Master table, CSV.
- Phase 2 (done): shot ↔ row 1:1 (`ResultsFrame.upsert_row`/`index_of` keyed by
  shot_id; `MainWindow._shot_map` maps shot_id → file+frame). Populate shots,
  row-click switches shot (`_goto_shot`), fit auto-updates the row in place,
  EXCLUDE flag (`_toggle_exclude`) survives re-fitting and dims the row.
- Phase 3: SQLite `.qoradfxm_proj` autosave/restore via `QoraDFXMDataset.to_dict()` /
  `ResultsFrame`; ROI-drag ↔ table live sync. Undo/Redo already supported by the
  immutable history. Session persistence belongs in `core/` (a `.qoradfxm_proj` must
  be readable headlessly) — the old empty `qoradfxm/session/` stub was removed.
- Next for GUI → CLI: a "batch this recipe over N shots" panel on top of
  `CliJob` (queue, progress, cancel) — the plumbing is there, the UI is not.

Preproc ops
- intensity: sub_bg, divide, log (adaptive), pure_log, sqrt, gamma(γ), normalize
- geometric (`core.ops.GEOMETRIC_KINDS`): scale (sx, sy — aspect ratio),
  rotate (angle, expand), flip (h/v/both)

Geometric ops resample the grid, so they change the image shape: a `sub_bg`
whose reference no longer matches is skipped by the shape guard, and points
picked before them are stale (the GUI warns). Put geometry first in the recipe.
