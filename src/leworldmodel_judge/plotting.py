"""Shared plotting layer: optional matplotlib plus SVG-fallback primitives.

Matplotlib is an optional extra (``viz``). This module owns the single
import guard for it: on import it tries ``matplotlib`` with the ``Agg``
backend and records the outcome in the module globals
:data:`MATPLOTLIB_AVAILABLE` and :data:`plt`. Renderers (``report``,
``demo``) must read those globals through this module at call time — never
copy them at import time — so tests can monkeypatch them and the SVG
fallback path stays exercisable with matplotlib installed.

The SVG primitives are deliberately tiny: enough to draw honest line charts
without any dependency, not a plotting library.
"""

from __future__ import annotations

from typing import Any

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as _pyplot

    plt: Any = _pyplot
    MATPLOTLIB_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - exercised in environments without matplotlib
    plt = None
    MATPLOTLIB_AVAILABLE = False


def svg_polyline(points: list[tuple[float, float]], color: str) -> str:
    """Serialize points into one SVG ``<polyline>`` element."""
    serialised = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{serialised}" />'


def line_points(
    values: list[float], width: int = 760, height: int = 420, padding: int = 48
) -> list[tuple[float, float]]:
    """Map a [0, 1] value series onto evenly spaced SVG plot coordinates.

    Values are clipped to [0, 1]; a single value is centred horizontally.
    """
    if not values:
        return []
    usable_width = width - (2 * padding)
    usable_height = height - (2 * padding)
    if len(values) == 1:
        x_positions = [padding + usable_width / 2]
    else:
        x_positions = [
            padding + usable_width * idx / (len(values) - 1) for idx in range(len(values))
        ]
    return [
        (x_positions[idx], padding + usable_height * (1.0 - max(0.0, min(1.0, values[idx]))))
        for idx in range(len(values))
    ]
