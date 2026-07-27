"""
Interactive ellipse fitting from hand-picked points on an image.

Usage
-----
    python ellipse_fit.py path/to/image.png
    python ellipse_fit.py path/to/image.png --npoints 8 --outdir results

Workflow
--------
    1. The image opens in a matplotlib window.
    2. Left-click N points (default 8) along the ellipse edge.
       - Backspace / right-click removes the last point.
       - The fit runs automatically once N points are placed.
       - Press 'r' to reset and start over, 'q' to quit.
    3. The fitted ellipse, its axes and center are drawn over the image.
    4. Results are written to <outdir>/ as .json, .txt and an annotated .png.

Coordinate convention
---------------------
    Pixel coordinates: x to the right, y DOWNWARD (standard image convention).
    "Tilt" is reported as the angle of the MAJOR axis away from the vertical
    axis of the image, in degrees, positive = clockwise on screen,
    in the range (-90, +90].

Requires: numpy, matplotlib  (pillow for most image formats)
Needs an interactive matplotlib backend (TkAgg, QtAgg, macosx...).
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


# ----------------------------------------------------------------------
# CONFIG  --  for Spyder / "Run file" users
# ----------------------------------------------------------------------
# Put your image path here and just press Run. Leave it as "" to get a
# file-picker dialog instead. A path given on the command line still wins.
#
# On Windows, use a raw string (the r"...") so backslashes aren't a problem:
#     IMAGE_PATH = r"C:\Users\USER\Desktop\연구\DFXM\untitled1.png"
IMAGE_PATH = ""

NPOINTS = 8            # how many points to pick
OUTDIR = "ellipse_results"

# Kept alive at module scope so the interactive picker's event callbacks
# are not garbage-collected after main() returns (matters in Spyder/IPython,
# where plt.show() does not block).
_PICKER = None


# ----------------------------------------------------------------------
# Core fitting
# ----------------------------------------------------------------------

def fit_ellipse(x, y):
    """
    Direct least-squares ellipse fit (Halir & Flusser 1998), a numerically
    stable reformulation of Fitzgibbon-Pilu-Fisher.

    The constraint 4ac - b^2 = 1 forces the returned conic to be an ellipse:
    a hyperbola or parabola can never be returned, no matter how the points
    are arranged.

    Returns
    -------
    coeffs : ndarray, shape (6,)
        (a, b, c, d, e, f) for  a x^2 + b x y + c y^2 + d x + e y + f = 0
        expressed in the ORIGINAL pixel coordinates.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size < 5:
        raise ValueError("Need at least 5 points to fit a conic.")

    # --- Normalize: shift to centroid, scale to RMS radius ~ sqrt(2).
    # Without this the x^2 / xy entries dominate the design matrix and the
    # eigenproblem becomes badly conditioned for typical pixel magnitudes.
    mx, my = x.mean(), y.mean()
    s = np.sqrt(((x - mx) ** 2 + (y - my) ** 2).mean())
    if s <= 0:
        raise ValueError("All points are identical.")
    u = (x - mx) / s
    v = (y - my) / s

    D1 = np.column_stack([u * u, u * v, v * v])          # quadratic part
    D2 = np.column_stack([u, v, np.ones_like(u)])        # linear part

    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2

    T = -np.linalg.solve(S3, S2.T)          # 3x3
    M = S1 + S2 @ T                         # reduced scatter matrix

    # Pre-multiply by inv(C1) where C1 is the constraint matrix; this is
    # just a fixed row shuffle, done explicitly for stability.
    M = np.array([M[2] / 2.0, -M[1], M[0] / 2.0])

    evals, evecs = np.linalg.eig(M)

    # Pick the eigenvector satisfying the ellipse condition 4ac - b^2 > 0.
    cond = 4.0 * evecs[0] * evecs[2] - evecs[1] ** 2
    idx = np.argmax(cond)
    if cond[idx] <= 0:
        raise RuntimeError(
            "No ellipse solution found. The points are probably collinear "
            "or nearly so."
        )
    a1 = np.real(evecs[:, idx])
    a2 = T @ a1
    an, bn, cn, dn, en, fn = np.concatenate([a1, a2])

    # --- Un-normalize back to pixel coordinates.
    # Substitute u = (X - mx)/s, v = (Y - my)/s and expand.
    A = an / s**2
    B = bn / s**2
    C = cn / s**2
    D = (-2.0 * an * mx - bn * my) / s**2 + dn / s
    E = (-2.0 * cn * my - bn * mx) / s**2 + en / s
    F = ((an * mx**2 + bn * mx * my + cn * my**2) / s**2
         - (dn * mx + en * my) / s + fn)

    coeffs = np.array([A, B, C, D, E, F], dtype=float)
    return coeffs / np.linalg.norm(coeffs)   # scale is arbitrary; fix it


def conic_to_geometry(coeffs):
    """
    Convert a x^2 + b x y + c y^2 + d x + e y + f = 0 into geometric
    ellipse parameters, using the eigen-decomposition of the quadratic form
    (more robust than the closed-form radical expressions).

    Returns a dict with center, semi-axes, full axes and orientation angles.
    """
    a, b, c, d, e, f = coeffs

    disc = b * b - 4.0 * a * c
    if disc >= 0:
        raise ValueError("Conic is not an ellipse (b^2 - 4ac >= 0).")

    # Center: solve grad(Q) = 0
    x0 = (2.0 * c * d - b * e) / disc
    y0 = (2.0 * a * e - b * d) / disc

    # Constant term after translating the origin to the center
    f0 = a * x0**2 + b * x0 * y0 + c * y0**2 + d * x0 + e * y0 + f

    # Quadratic form matrix
    Mq = np.array([[a, b / 2.0],
                   [b / 2.0, c]])
    evals, evecs = np.linalg.eigh(Mq)        # symmetric -> real, orthonormal

    # Centered equation:  lambda_i * w_i^2 = -f0   =>   semi-axis = sqrt(-f0/lambda_i)
    radii = np.sqrt(-f0 / evals)
    order = np.argsort(radii)[::-1]          # major first
    semi_major, semi_minor = radii[order]
    major_vec = evecs[:, order[0]]

    # Angle of the major axis measured from +x, in pixel coords (y down).
    ang_x = np.degrees(np.arctan2(major_vec[1], major_vec[0]))
    ang_x = (ang_x + 180.0) % 180.0          # axis is undirected -> [0,180)

    # Tilt away from the vertical axis, positive = clockwise on screen.
    tilt = 90.0 - ang_x
    if tilt <= -90.0:
        tilt += 180.0
    elif tilt > 90.0:
        tilt -= 180.0

    ecc = np.sqrt(1.0 - (semi_minor / semi_major) ** 2)

    return {
        "center_x": float(x0),
        "center_y": float(y0),
        "semi_major_axis": float(semi_major),
        "semi_minor_axis": float(semi_minor),
        "major_diameter": float(2.0 * semi_major),
        "minor_diameter": float(2.0 * semi_minor),
        "axis_ratio": float(semi_minor / semi_major),
        "eccentricity": float(ecc),
        "angle_major_from_x_deg": float(ang_x),
        "tilt_from_vertical_deg": float(tilt),
        "area": float(np.pi * semi_major * semi_minor),
    }


def sampson_residuals(coeffs, x, y):
    """
    First-order (Sampson) approximation of the perpendicular distance from
    each point to the conic. A quick sanity check on fit quality, in pixels.
    """
    a, b, c, d, e, f = coeffs
    Q = a * x**2 + b * x * y + c * y**2 + d * x + e * y + f
    gx = 2 * a * x + b * y + d
    gy = 2 * c * y + b * x + e
    grad = np.sqrt(gx**2 + gy**2)
    grad[grad < 1e-12] = 1e-12
    return np.abs(Q) / grad


def ellipse_polyline(geom, n=400):
    """Sample points along the fitted ellipse for plotting."""
    t = np.linspace(0.0, 2.0 * np.pi, n)
    th = np.radians(geom["angle_major_from_x_deg"])
    ct, st = np.cos(th), np.sin(th)
    a, b = geom["semi_major_axis"], geom["semi_minor_axis"]
    xs = geom["center_x"] + a * np.cos(t) * ct - b * np.sin(t) * st
    ys = geom["center_y"] + a * np.cos(t) * st + b * np.sin(t) * ct
    return xs, ys


def format_equation(coeffs):
    a, b, c, d, e, f = coeffs
    return (f"{a:+.6e} x^2 {b:+.6e} xy {c:+.6e} y^2 "
            f"{d:+.6e} x {e:+.6e} y {f:+.6e} = 0")


def format_report(coeffs, geom, pts, resid):
    L = []
    L.append("ELLIPSE FIT RESULT")
    L.append("=" * 62)
    L.append("")
    L.append("Picked points (pixel coordinates, y downward):")
    for i, (px, py) in enumerate(pts, 1):
        L.append(f"  {i:2d}.  x = {px:9.3f}   y = {py:9.3f}   "
                 f"residual = {resid[i-1]:7.3f} px")
    L.append("")
    L.append("General conic equation (unit-norm coefficients):")
    L.append("  " + format_equation(coeffs))
    L.append("")
    L.append(f"  a = {coeffs[0]:+.9e}")
    L.append(f"  b = {coeffs[1]:+.9e}")
    L.append(f"  c = {coeffs[2]:+.9e}")
    L.append(f"  d = {coeffs[3]:+.9e}")
    L.append(f"  e = {coeffs[4]:+.9e}")
    L.append(f"  f = {coeffs[5]:+.9e}")
    L.append(f"  discriminant b^2-4ac = {coeffs[1]**2 - 4*coeffs[0]*coeffs[2]:+.6e}"
             "   (< 0 confirms an ellipse)")
    L.append("")
    L.append("Geometric parameters:")
    L.append(f"  Center                 : ({geom['center_x']:.3f}, "
             f"{geom['center_y']:.3f}) px")
    L.append(f"  Long (major) diameter  : {geom['major_diameter']:.3f} px")
    L.append(f"  Short (minor) diameter : {geom['minor_diameter']:.3f} px")
    L.append(f"  Semi-major a           : {geom['semi_major_axis']:.3f} px")
    L.append(f"  Semi-minor b           : {geom['semi_minor_axis']:.3f} px")
    L.append(f"  Axis ratio b/a         : {geom['axis_ratio']:.5f}")
    L.append(f"  Eccentricity           : {geom['eccentricity']:.5f}")
    L.append(f"  Area                   : {geom['area']:.2f} px^2")
    L.append(f"  Tilt from VERTICAL     : {geom['tilt_from_vertical_deg']:+.3f} deg"
             "   (+ = clockwise on screen)")
    L.append(f"  Major axis from +x     : {geom['angle_major_from_x_deg']:.3f} deg")
    L.append("")
    L.append("Fit quality (Sampson distance, approx. perpendicular error):")
    L.append(f"  RMS  = {np.sqrt((resid**2).mean()):.4f} px")
    L.append(f"  max  = {resid.max():.4f} px")
    L.append("")
    return "\n".join(L)


# ----------------------------------------------------------------------
# Analysis + plotting
# ----------------------------------------------------------------------

def analyze(image, pts, outdir, basename, show=True):
    pts = np.asarray(pts, dtype=float)
    x, y = pts[:, 0], pts[:, 1]

    coeffs = fit_ellipse(x, y)
    geom = conic_to_geometry(coeffs)
    resid = sampson_residuals(coeffs, x, y)

    report = format_report(coeffs, geom, pts, resid)
    print("\n" + report)

    os.makedirs(outdir, exist_ok=True)
    txt_path = os.path.join(outdir, basename + "_ellipse.txt")
    json_path = os.path.join(outdir, basename + "_ellipse.json")
    png_path = os.path.join(outdir, basename + "_ellipse.png")

    with open(txt_path, "w") as fh:
        fh.write(report)

    payload = {
        "points": pts.tolist(),
        "conic_coefficients": {
            k: float(v) for k, v in zip("abcdef", coeffs)
        },
        "equation": format_equation(coeffs),
        "geometry": geom,
        "residuals_px": resid.tolist(),
        "rms_residual_px": float(np.sqrt((resid**2).mean())),
        "notes": "Pixel coordinates, y axis downward. tilt_from_vertical_deg "
                 "is positive clockwise on screen.",
    }
    with open(json_path, "w") as fh:
        json.dump(payload, fh, indent=2)

    # --- Overlay figure
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(image, cmap="gray" if image.ndim == 2 else None)

    ex, ey = ellipse_polyline(geom)
    ax.plot(ex, ey, "-", color="#ff2d55", lw=2.0, label="fitted ellipse")
    ax.plot(x, y, "o", mfc="none", mec="#00e5ff", mew=2.0, ms=10,
            label="picked points")
    ax.plot(geom["center_x"], geom["center_y"], "+", color="#ffe600",
            ms=14, mew=2.5, label="center")

    th = np.radians(geom["angle_major_from_x_deg"])
    for r, vec, col in ((geom["semi_major_axis"], (np.cos(th), np.sin(th)), "#ffe600"),
                        (geom["semi_minor_axis"], (-np.sin(th), np.cos(th)), "#7cff5c")):
        ax.plot([geom["center_x"] - r * vec[0], geom["center_x"] + r * vec[0]],
                [geom["center_y"] - r * vec[1], geom["center_y"] + r * vec[1]],
                "--", color=col, lw=1.4)

    ax.set_title(
        f"major D = {geom['major_diameter']:.2f} px   "
        f"minor D = {geom['minor_diameter']:.2f} px   "
        f"tilt = {geom['tilt_from_vertical_deg']:+.2f}$^\\circ$ from vertical\n"
        f"RMS residual = {np.sqrt((resid**2).mean()):.3f} px",
        fontsize=11)
    ax.legend(loc="upper right", framealpha=0.85)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)

    print(f"Saved:\n  {txt_path}\n  {json_path}\n  {png_path}\n")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return coeffs, geom


# ----------------------------------------------------------------------
# Interactive picker
# ----------------------------------------------------------------------

class PointPicker:
    def __init__(self, image, npoints, on_done):
        self.image = image
        self.npoints = npoints
        self.on_done = on_done
        self.pts = []

        self.fig, self.ax = plt.subplots(figsize=(9, 9))
        self.ax.imshow(image, cmap="gray" if image.ndim == 2 else None)
        self.scatter, = self.ax.plot([], [], "o", mfc="none", mec="#00e5ff",
                                     mew=2.0, ms=10)
        self.labels = []
        self._update_title()

        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def _update_title(self):
        self.ax.set_title(
            f"Click {self.npoints} points on the ellipse edge "
            f"({len(self.pts)}/{self.npoints})\n"
            "right-click or Backspace = undo,  r = reset,  q = quit",
            fontsize=11)
        self.fig.canvas.draw_idle()

    def _redraw(self):
        if self.pts:
            arr = np.array(self.pts)
            self.scatter.set_data(arr[:, 0], arr[:, 1])
        else:
            self.scatter.set_data([], [])
        for t in self.labels:
            t.remove()
        self.labels = [
            self.ax.text(px + 6, py - 6, str(i + 1), color="#00e5ff",
                         fontsize=11, fontweight="bold")
            for i, (px, py) in enumerate(self.pts)
        ]
        self._update_title()

    def on_click(self, event):
        if event.inaxes is not self.ax:
            return
        if event.button == 3:                      # right-click = undo
            if self.pts:
                self.pts.pop()
                self._redraw()
            return
        if event.button != 1 or len(self.pts) >= self.npoints:
            return
        self.pts.append((event.xdata, event.ydata))
        self._redraw()
        if len(self.pts) == self.npoints:
            plt.close(self.fig)
            self.on_done(self.pts)

    def on_key(self, event):
        if event.key in ("backspace", "delete"):
            if self.pts:
                self.pts.pop()
                self._redraw()
        elif event.key == "r":
            self.pts = []
            self._redraw()
        elif event.key == "q":
            plt.close(self.fig)


# ----------------------------------------------------------------------

def _ask_for_image():
    """Open a file-picker dialog. Returns a path string, or '' if cancelled."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return ""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Select image to analyze",
        filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.gif"),
                   ("All files", "*.*")])
    root.destroy()
    return path or ""


def _warn_if_noninteractive():
    """Point-picking needs an interactive backend; warn if we don't have one."""
    backend = plt.get_backend().lower()
    if "inline" in backend or "agg" in backend:
        print("\n" + "!" * 64)
        print("matplotlib is using a non-interactive backend:", plt.get_backend())
        print("Clicking on the image will NOT work until you switch it.")
        print("In the Spyder console, run this once, then re-run the script:")
        print("    %matplotlib qt")
        print("(or: Tools > Preferences > IPython console > Graphics >")
        print(" Graphics backend > set to 'Qt5' or 'Automatic', then restart)")
        print("!" * 64 + "\n")


def main():
    ap = argparse.ArgumentParser(description="Fit an ellipse to hand-picked "
                                             "points on an image.")
    ap.add_argument("image", nargs="?", default=None,
                    help="path to the image file (optional; falls back to "
                         "IMAGE_PATH at the top of the file, then a dialog)")
    ap.add_argument("-n", "--npoints", type=int, default=NPOINTS,
                    help="number of points to pick (default 8, minimum 5)")
    ap.add_argument("-o", "--outdir", default=OUTDIR,
                    help="output directory (default: ellipse_results)")
    ap.add_argument("--points", nargs="+", type=float, default=None,
                    help="skip picking; supply x1 y1 x2 y2 ... directly")
    ap.add_argument("--no-show", action="store_true",
                    help="save figures without opening a window")
    # parse_known_args (not parse_args) so Spyder's own injected arguments
    # don't cause a SystemExit.
    args, _unknown = ap.parse_known_args()

    if args.npoints < 5:
        ap.error("need at least 5 points to define a conic")

    # Resolve the image path: command line > IMAGE_PATH constant > dialog.
    image_path = args.image or (IMAGE_PATH.strip() or None)
    if image_path is None:
        image_path = _ask_for_image()
    if not image_path:
        print("No image selected. Set IMAGE_PATH at the top of the file, "
              "or pick a file in the dialog.")
        return
    if not os.path.isfile(image_path):
        print(f"File not found: {image_path}")
        return

    _warn_if_noninteractive()

    image = plt.imread(image_path)
    basename = os.path.splitext(os.path.basename(image_path))[0]

    if args.points is not None:
        vals = args.points
        if len(vals) % 2 != 0:
            ap.error("--points needs an even number of values")
        pts = np.array(vals).reshape(-1, 2)
        analyze(image, pts, args.outdir, basename, show=not args.no_show)
        return

    def done(pts):
        analyze(image, pts, args.outdir, basename, show=not args.no_show)

    global _PICKER
    _PICKER = PointPicker(image, args.npoints, done)
    plt.show()


if __name__ == "__main__":
    main()