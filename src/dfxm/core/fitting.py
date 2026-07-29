"""Direct least-squares ellipse fitting from picked points. Pure NumPy.

Coordinate convention: pixel coordinates, x right, y DOWNWARD. Moved out of the
old interactive ``ellipse_fit.py`` so the math has zero matplotlib / GUI deps.
"""

from __future__ import annotations

import numpy as np


def fit_ellipse(x, y):
    """
    Direct least-squares ellipse fit (Halir & Flusser 1998), a numerically
    stable reformulation of Fitzgibbon-Pilu-Fisher.

    Returns
    -------
    coeffs : ndarray, shape (6,)
        (a, b, c, d, e, f) for  a x^2 + b x y + c y^2 + d x + e y + f = 0
        in the ORIGINAL pixel coordinates.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size < 5:
        raise ValueError("Need at least 5 points to fit a conic.")

    mx, my = x.mean(), y.mean()
    s = np.sqrt(((x - mx) ** 2 + (y - my) ** 2).mean())
    if s <= 0:
        raise ValueError("All points are identical.")
    u = (x - mx) / s
    v = (y - my) / s

    D1 = np.column_stack([u * u, u * v, v * v])
    D2 = np.column_stack([u, v, np.ones_like(u)])

    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2

    T = -np.linalg.solve(S3, S2.T)
    M = S1 + S2 @ T
    M = np.array([M[2] / 2.0, -M[1], M[0] / 2.0])

    evals, evecs = np.linalg.eig(M)

    cond = 4.0 * evecs[0] * evecs[2] - evecs[1] ** 2
    idx = np.argmax(cond)
    if cond[idx] <= 0:
        raise RuntimeError(
            "No ellipse solution found. The points are probably collinear or nearly so."
        )
    a1 = np.real(evecs[:, idx])
    a2 = T @ a1
    an, bn, cn, dn, en, fn = np.concatenate([a1, a2])

    A = an / s**2
    B = bn / s**2
    C = cn / s**2
    D = (-2.0 * an * mx - bn * my) / s**2 + dn / s
    E = (-2.0 * cn * my - bn * mx) / s**2 + en / s
    F = (an * mx**2 + bn * mx * my + cn * my**2) / s**2 - (dn * mx + en * my) / s + fn

    coeffs = np.array([A, B, C, D, E, F], dtype=float)
    return coeffs / np.linalg.norm(coeffs)


def conic_to_geometry(coeffs):
    """Convert conic coefficients into geometric ellipse parameters."""
    a, b, c, d, e, f = coeffs

    disc = b * b - 4.0 * a * c
    if disc >= 0:
        raise ValueError("Conic is not an ellipse (b^2 - 4ac >= 0).")

    x0 = (2.0 * c * d - b * e) / disc
    y0 = (2.0 * a * e - b * d) / disc

    f0 = a * x0**2 + b * x0 * y0 + c * y0**2 + d * x0 + e * y0 + f

    Mq = np.array([[a, b / 2.0], [b / 2.0, c]])
    evals, evecs = np.linalg.eigh(Mq)

    radii = np.sqrt(-f0 / evals)
    order = np.argsort(radii)[::-1]
    semi_major, semi_minor = radii[order]
    major_vec = evecs[:, order[0]]

    ang_x = np.degrees(np.arctan2(major_vec[1], major_vec[0]))
    ang_x = (ang_x + 180.0) % 180.0

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
    """First-order (Sampson) approx of perpendicular distance, in pixels."""
    a, b, c, d, e, f = coeffs
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
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


def rms_error(coeffs, x, y) -> float:
    """RMS Sampson residual — the single-number fit-quality score."""
    r = sampson_residuals(coeffs, x, y)
    return float(np.sqrt(np.mean(r**2)))


def format_equation(coeffs):
    a, b, c, d, e, f = coeffs
    return f"{a:+.6e} x^2 {b:+.6e} xy {c:+.6e} y^2 {d:+.6e} x {e:+.6e} y {f:+.6e} = 0"


def format_report(coeffs, geom, pts, resid):
    L = []
    L.append("ELLIPSE FIT RESULT")
    L.append("=" * 62)
    L.append("")
    L.append("Picked points (pixel coordinates, y downward):")
    for i, (px, py) in enumerate(pts, 1):
        L.append(
            f"  {i:2d}.  x = {px:9.3f}   y = {py:9.3f}   "
            f"residual = {resid[i - 1]:7.3f} px"
        )
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
    L.append(
        f"  discriminant b^2-4ac = {coeffs[1] ** 2 - 4 * coeffs[0] * coeffs[2]:+.6e}"
        "   (< 0 confirms an ellipse)"
    )
    L.append("")
    L.append("Geometric parameters:")
    L.append(
        f"  Center                 : ({geom['center_x']:.3f}, "
        f"{geom['center_y']:.3f}) px"
    )
    L.append(f"  Long (major) diameter  : {geom['major_diameter']:.3f} px")
    L.append(f"  Short (minor) diameter : {geom['minor_diameter']:.3f} px")
    L.append(f"  Semi-major a           : {geom['semi_major_axis']:.3f} px")
    L.append(f"  Semi-minor b           : {geom['semi_minor_axis']:.3f} px")
    L.append(f"  Axis ratio b/a         : {geom['axis_ratio']:.5f}")
    L.append(f"  Eccentricity           : {geom['eccentricity']:.5f}")
    L.append(f"  Area                   : {geom['area']:.2f} px^2")
    L.append(
        f"  Tilt from VERTICAL     : {geom['tilt_from_vertical_deg']:+.3f} deg"
        "   (+ = clockwise on screen)"
    )
    L.append(f"  Major axis from +x     : {geom['angle_major_from_x_deg']:.3f} deg")
    L.append("")
    L.append("Fit quality (Sampson distance, approx. perpendicular error):")
    L.append(f"  RMS  = {np.sqrt((resid**2).mean()):.4f} px")
    L.append(f"  max  = {resid.max():.4f} px")
    L.append("")
    return "\n".join(L)
