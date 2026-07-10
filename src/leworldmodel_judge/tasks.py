"""The locked benchmark task set and task-argument resolution.

``LOCKED_TASKS`` is the benchmark contract (docs/benchmark.md): exactly these
three Meta-World tasks, in this order. Adding a task is a benchmark change,
not a convenience edit.
"""

from __future__ import annotations

LOCKED_TASKS = ("reach-v3", "push-v3", "pick-place-v3")


def resolve_tasks(task_arg: str) -> list[str]:
    """Resolve a task argument to an ordered, de-duplicated task list.

    Accepts a single task id, a comma-separated subset of ``LOCKED_TASKS``,
    or the literal ``"all"`` (the full locked set in canonical order).
    Unknown task ids raise ``ValueError`` — there is no silent fallback.
    """
    task_arg = task_arg.strip()
    if not task_arg:
        raise ValueError("task argument must not be empty")
    if task_arg == "all":
        return list(LOCKED_TASKS)

    tasks = [part.strip() for part in task_arg.split(",") if part.strip()]
    invalid = sorted(set(tasks) - set(LOCKED_TASKS))
    if invalid:
        raise ValueError(
            f"task must be one of {list(LOCKED_TASKS)} or a comma-separated subset; got invalid {invalid}"
        )
    if not tasks:
        raise ValueError("task argument resolved to an empty task list")

    seen: set[str] = set()
    ordered: list[str] = []
    for task in tasks:
        if task in seen:
            continue
        seen.add(task)
        ordered.append(task)
    return ordered
