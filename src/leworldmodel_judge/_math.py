"""Private numeric helpers shared across pipeline modules.

Kept deliberately tiny: only helpers whose duplication across modules risked
semantic drift live here. Absence semantics stay in the calling modules —
``prefixes._mean`` returns ``None`` on empty input (absence is not zero),
``latents._mean_or_zero`` returns ``0.0`` (a latent gap of nothing is zero).
"""

from __future__ import annotations


def clip01(value: float) -> float:
    """Clip ``value`` to the closed interval [0.0, 1.0]."""
    return max(0.0, min(1.0, float(value)))
