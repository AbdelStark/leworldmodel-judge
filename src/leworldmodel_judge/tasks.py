from __future__ import annotations

LOCKED_TASKS = ("reach-v3", "push-v3", "pick-place-v3")


def resolve_tasks(task_arg: str) -> list[str]:
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
