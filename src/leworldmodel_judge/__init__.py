"""A world model as a judge: early, auditable verdicts on partial rollouts.

Public API of the ``leworldmodel_judge`` package. The pipeline stages compose
as plain functions over dict rows and dataclass records:

collect (:func:`collect_synthetic` / :func:`collect_metaworld`) →
:func:`build_prefixes` → :func:`score_prefixes` / :func:`build_latent_cache` →
:func:`run_judge` → :func:`summarize`.

The core is stdlib-only; plotting and Meta-World capture are optional extras
(``viz``, ``metaworld``).
"""

from __future__ import annotations

from .baselines import score_prefix, score_prefixes
from .collect import (
    POLICY_FAMILIES,
    collect_metaworld,
    collect_synthetic,
    derive_task_seed,
)
from .io import read_jsonl, write_json, write_jsonl
from .judge import (
    DEFAULT_COMPOSITE_WEIGHTS,
    DEFAULT_HYBRID_WEIGHTS,
    JUDGE_MODES,
    CompositeWeights,
    HybridWeights,
    dummy_score,
    heuristic_surprise_score,
    hybrid_surprise_score,
    run_judge,
)
from .labels import label_prefix
from .latents import LATENT_CACHE_VERSION, build_latent_cache
from .metrics import family_name, summarize
from .prefixes import DEFAULT_FRACTIONS, build_prefixes, group_by_episode
from .schema import PrefixRecord, RolloutStep, prefix_key
from .tasks import LOCKED_TASKS, resolve_tasks

__version__ = "0.2.0"

__all__ = [
    "DEFAULT_COMPOSITE_WEIGHTS",
    "DEFAULT_FRACTIONS",
    "DEFAULT_HYBRID_WEIGHTS",
    "JUDGE_MODES",
    "LATENT_CACHE_VERSION",
    "LOCKED_TASKS",
    "POLICY_FAMILIES",
    "CompositeWeights",
    "HybridWeights",
    "PrefixRecord",
    "RolloutStep",
    "__version__",
    "build_latent_cache",
    "build_prefixes",
    "collect_metaworld",
    "collect_synthetic",
    "derive_task_seed",
    "dummy_score",
    "family_name",
    "group_by_episode",
    "heuristic_surprise_score",
    "hybrid_surprise_score",
    "label_prefix",
    "prefix_key",
    "read_jsonl",
    "resolve_tasks",
    "run_judge",
    "score_prefix",
    "score_prefixes",
    "summarize",
    "write_json",
    "write_jsonl",
]
