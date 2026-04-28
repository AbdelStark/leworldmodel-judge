from __future__ import annotations

from collections import defaultdict

DEFAULT_THRESHOLDS = {
    "judge_failure_threshold": 0.5,
    "progress_failure_threshold": 0.2,
}


def _blank_bucket() -> dict[str, float | list[tuple[bool, float]]]:
    return {
        "count": 0,
        "failure_labels": 0,
        "non_failure_labels": 0,
        "judge_failure_hits": 0,
        "judge_false_positives": 0,
        "baseline_sparse_absence_hits": 0,
        "baseline_sparse_absence_false_positives": 0,
        "baseline_progress_hits": 0,
        "baseline_progress_false_positives": 0,
        "_judge_pairs": [],
        "_baseline_sparse_absence_pairs": [],
        "_baseline_progress_pairs": [],
    }


def _rate(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 6)


def _pairwise_accuracy(pairs: list[tuple[bool, float]]) -> float | None:
    failures = [score for label, score in pairs if label]
    non_failures = [score for label, score in pairs if not label]
    if not failures or not non_failures:
        return None

    wins = 0.0
    total = 0
    for fail_score in failures:
        for non_fail_score in non_failures:
            total += 1
            if fail_score > non_fail_score:
                wins += 1.0
            elif fail_score == non_fail_score:
                wins += 0.5
    return round(wins / total, 6)


def _average_precision(pairs: list[tuple[bool, float]]) -> float | None:
    positives = sum(1 for label, _ in pairs if label)
    negatives = sum(1 for label, _ in pairs if not label)
    if positives == 0 or negatives == 0:
        return None

    ranked = sorted(pairs, key=lambda pair: pair[1], reverse=True)
    average_precision = 0.0
    true_positives = 0
    false_positives = 0
    index = 0
    while index < len(ranked):
        score = ranked[index][1]
        group = []
        while index < len(ranked) and ranked[index][1] == score:
            group.append(ranked[index])
            index += 1
        positives_in_group = sum(1 for label, _ in group if label)
        negatives_in_group = len(group) - positives_in_group
        true_positives += positives_in_group
        false_positives += negatives_in_group
        if positives_in_group:
            precision = true_positives / (true_positives + false_positives)
            recall_delta = positives_in_group / positives
            average_precision += precision * recall_delta
    return round(average_precision, 6)


def _classification_stats(
    pairs: list[tuple[bool, float]], threshold: float
) -> dict[str, float | None]:
    failures = [score for label, score in pairs if label]
    non_failures = [score for label, score in pairs if not label]
    if not failures or not non_failures:
        return {
            "hit_rate": None,
            "false_positive_rate": None,
            "balanced_accuracy": None,
        }

    tp = sum(1 for score in failures if score >= threshold)
    fp = sum(1 for score in non_failures if score >= threshold)
    hit_rate = tp / len(failures)
    false_positive_rate = fp / len(non_failures)
    balanced_accuracy = 0.5 * (hit_rate + (1.0 - false_positive_rate))
    return {
        "hit_rate": round(hit_rate, 6),
        "false_positive_rate": round(false_positive_rate, 6),
        "balanced_accuracy": round(balanced_accuracy, 6),
    }


def _candidate_thresholds(pairs: list[tuple[bool, float]], fallback: float) -> list[float]:
    scores = sorted({round(float(score), 6) for _, score in pairs}, reverse=True)
    if not scores:
        return [round(float(fallback), 6)]
    candidates = scores + [round(float(max(scores) + 1e-6), 6), round(float(min(scores) - 1e-6), 6)]
    return sorted(set(candidates), reverse=True)


def _recommend_threshold(
    pairs: list[tuple[bool, float]], fallback: float, *, mode: str
) -> dict[str, float | None | str | dict[str, float | None]]:
    failures = [score for label, score in pairs if label]
    non_failures = [score for label, score in pairs if not label]
    if not failures or not non_failures:
        return {
            "recommended_threshold": round(float(fallback), 6),
            "hit_rate": None,
            "false_positive_rate": None,
            "balanced_accuracy": None,
            "mode": "fallback",
            "cohort_stats": _cohort_stats(pairs),
        }

    best_threshold = round(float(fallback), 6)
    best_stats = _classification_stats(pairs, best_threshold)
    best_key = (
        best_stats["balanced_accuracy"] if best_stats["balanced_accuracy"] is not None else -1.0,
        -(
            best_stats["false_positive_rate"]
            if best_stats["false_positive_rate"] is not None
            else 1.0
        ),
        best_stats["hit_rate"] if best_stats["hit_rate"] is not None else -1.0,
        best_threshold,
    )

    for threshold in _candidate_thresholds(pairs, fallback):
        stats = _classification_stats(pairs, threshold)
        key = (
            stats["balanced_accuracy"] if stats["balanced_accuracy"] is not None else -1.0,
            -(stats["false_positive_rate"] if stats["false_positive_rate"] is not None else 1.0),
            stats["hit_rate"] if stats["hit_rate"] is not None else -1.0,
            threshold,
        )
        if key > best_key:
            best_threshold = threshold
            best_stats = stats
            best_key = key

    return {
        "recommended_threshold": round(float(best_threshold), 6),
        "hit_rate": best_stats["hit_rate"],
        "false_positive_rate": best_stats["false_positive_rate"],
        "balanced_accuracy": best_stats["balanced_accuracy"],
        "mode": mode,
        "cohort_stats": _cohort_stats(pairs),
    }


def _family_name(record: dict) -> str:
    family = record.get("policy_family")
    if family:
        return str(family)
    episode_id = str(record.get("episode_id", ""))
    task_id = str(record.get("task_id", ""))
    prefix = f"{task_id}-"
    if episode_id.startswith(prefix) and "-ep-" in episode_id:
        return episode_id[len(prefix) :].split("-ep-")[0]
    return "unknown"


def _cohort_stats(pairs: list[tuple[bool, float]]) -> dict[str, int | float | None]:
    failures = [score for label, score in pairs if label]
    non_failures = [score for label, score in pairs if not label]
    return {
        "count": len(pairs),
        "failure_labels": len(failures),
        "non_failure_labels": len(non_failures),
        "pairwise_accuracy": _pairwise_accuracy(pairs),
        "auroc": _pairwise_accuracy(pairs),
        "average_precision": _average_precision(pairs),
    }


def _finalize_bucket(bucket: dict[str, float | list[tuple[bool, float]]]) -> dict[str, float | None]:
    failures = float(bucket["failure_labels"])
    non_failures = float(bucket["non_failure_labels"])
    finalized = {k: v for k, v in bucket.items() if not k.startswith("_")}
    finalized["judge_failure_hit_rate"] = _rate(float(bucket["judge_failure_hits"]), failures)
    finalized["judge_false_positive_rate"] = _rate(
        float(bucket["judge_false_positives"]), non_failures
    )
    finalized["baseline_sparse_absence_hit_rate"] = _rate(
        float(bucket["baseline_sparse_absence_hits"]), failures
    )
    finalized["baseline_sparse_absence_false_positive_rate"] = _rate(
        float(bucket["baseline_sparse_absence_false_positives"]), non_failures
    )
    finalized["baseline_progress_hit_rate"] = _rate(
        float(bucket["baseline_progress_hits"]), failures
    )
    finalized["baseline_progress_false_positive_rate"] = _rate(
        float(bucket["baseline_progress_false_positives"]), non_failures
    )
    finalized["failure_label_coverage"] = _rate(failures, float(bucket["count"]))

    judge_pairwise = _pairwise_accuracy(bucket["_judge_pairs"])
    sparse_pairwise = _pairwise_accuracy(bucket["_baseline_sparse_absence_pairs"])
    progress_pairwise = _pairwise_accuracy(bucket["_baseline_progress_pairs"])

    finalized["judge_pairwise_accuracy"] = judge_pairwise
    finalized["judge_auroc"] = judge_pairwise
    finalized["judge_average_precision"] = _average_precision(bucket["_judge_pairs"])
    finalized["baseline_sparse_absence_pairwise_accuracy"] = sparse_pairwise
    finalized["baseline_sparse_absence_auroc"] = sparse_pairwise
    finalized["baseline_sparse_absence_average_precision"] = _average_precision(
        bucket["_baseline_sparse_absence_pairs"]
    )
    finalized["baseline_progress_pairwise_accuracy"] = progress_pairwise
    finalized["baseline_progress_auroc"] = progress_pairwise
    finalized["baseline_progress_average_precision"] = _average_precision(
        bucket["_baseline_progress_pairs"]
    )
    return finalized


def summarize(
    prefixes: list[dict],
    baselines: list[dict],
    judge: list[dict],
    thresholds: dict[str, float] | None = None,
    *,
    calibration_families: list[str] | None = None,
    evaluation_families: list[str] | None = None,
) -> dict:
    baseline_map = {(r["task_id"], r["episode_id"], r["prefix_fraction"]): r for r in baselines}
    judge_map = {(r["task_id"], r["episode_id"], r["prefix_fraction"]): r for r in judge}
    calibration_family_set = set(calibration_families or [])
    evaluation_family_set = set(evaluation_families or [])

    all_rows: list[dict] = []
    for prefix in prefixes:
        key = (prefix["task_id"], prefix["episode_id"], prefix["prefix_fraction"])
        b = baseline_map.get(key, {})
        j = judge_map.get(key, {})
        family = _family_name(prefix)
        all_rows.append(
            {
                "task": prefix["task_id"],
                "family": family,
                "label": bool(prefix.get("prefix_failure_label")),
                "judge_score": float(j.get("failure_score", 0.0)),
                "progress_failure_score": 1.0 - float(b.get("progress_proxy_score", 0.0)),
                "sparse_absence_score": 1.0
                if float(b.get("sparse_reward_score", 0.0)) <= 0.0
                else 0.0,
            }
        )

    evaluation_rows = [
        row for row in all_rows if not evaluation_family_set or row["family"] in evaluation_family_set
    ]
    calibration_rows = [
        row for row in all_rows if not calibration_family_set or row["family"] in calibration_family_set
    ]

    evaluation_families_present = {row["family"] for row in evaluation_rows}
    calibration_families_present = {row["family"] for row in calibration_rows}
    family_overlap = bool(calibration_families_present & evaluation_families_present)
    judge_mode = "in_slice_balanced_accuracy"
    if (
        calibration_family_set
        and calibration_families_present
        and evaluation_rows
        and not family_overlap
    ):
        judge_mode = "held_out_family_split"

    chosen_thresholds = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        chosen_thresholds.update(thresholds)

    calibration_pairs = [(row["label"], row["judge_score"]) for row in calibration_rows]
    evaluation_pairs = [(row["label"], row["judge_score"]) for row in evaluation_rows]
    calibration = {
        "judge": _recommend_threshold(
            calibration_pairs,
            chosen_thresholds["judge_failure_threshold"],
            mode=judge_mode,
        ),
        "progress": {
            "recommended_threshold": round(
                float(1.0 - chosen_thresholds["progress_failure_threshold"]), 6
            ),
            "hit_rate": None,
            "false_positive_rate": None,
            "balanced_accuracy": None,
            "mode": "fixed_progress_baseline",
        },
        "provenance": {
            "calibration_families": sorted(calibration_families_present)
            if calibration_family_set
            else ["all"],
            "evaluation_families": sorted(evaluation_families_present)
            if evaluation_family_set
            else ["all"],
            "family_overlap": family_overlap,
            "calibration_count": len(calibration_rows),
            "evaluation_count": len(evaluation_rows),
            "calibration_failure_labels": sum(1 for row in calibration_rows if row["label"]),
            "calibration_non_failure_labels": sum(1 for row in calibration_rows if not row["label"]),
            "evaluation_failure_labels": sum(1 for row in evaluation_rows if row["label"]),
            "evaluation_non_failure_labels": sum(1 for row in evaluation_rows if not row["label"]),
        },
    }
    calibration["judge"]["evaluation_stats"] = _classification_stats(
        evaluation_pairs,
        float(calibration["judge"]["recommended_threshold"]),
    )
    calibration["judge"]["evaluation_cohort"] = _cohort_stats(evaluation_pairs)
    chosen_thresholds["judge_failure_threshold"] = float(
        calibration["judge"]["recommended_threshold"]
    )

    per_task: dict[str, dict[str, float | list[tuple[bool, float]]]] = defaultdict(_blank_bucket)
    per_family: dict[str, dict[str, float | list[tuple[bool, float]]]] = defaultdict(_blank_bucket)
    overall = _blank_bucket()

    for row in evaluation_rows:
        for bucket in (overall, per_task[row["task"]], per_family[row["family"]]):
            bucket["count"] += 1
            if row["label"]:
                bucket["failure_labels"] += 1
            else:
                bucket["non_failure_labels"] += 1
            bucket["_judge_pairs"].append((row["label"], row["judge_score"]))
            bucket["_baseline_progress_pairs"].append(
                (row["label"], row["progress_failure_score"])
            )
            bucket["_baseline_sparse_absence_pairs"].append(
                (row["label"], row["sparse_absence_score"])
            )

        judge_predicts_failure = row["judge_score"] >= chosen_thresholds["judge_failure_threshold"]
        progress_predicts_failure = row["progress_failure_score"] >= (
            1.0 - chosen_thresholds["progress_failure_threshold"]
        )
        sparse_signal_absent = row["sparse_absence_score"] >= 1.0
        buckets = (overall, per_task[row["task"]], per_family[row["family"]])
        if row["label"]:
            if judge_predicts_failure:
                for bucket in buckets:
                    bucket["judge_failure_hits"] += 1
            if sparse_signal_absent:
                for bucket in buckets:
                    bucket["baseline_sparse_absence_hits"] += 1
            if progress_predicts_failure:
                for bucket in buckets:
                    bucket["baseline_progress_hits"] += 1
        else:
            if judge_predicts_failure:
                for bucket in buckets:
                    bucket["judge_false_positives"] += 1
            if sparse_signal_absent:
                for bucket in buckets:
                    bucket["baseline_sparse_absence_false_positives"] += 1
            if progress_predicts_failure:
                for bucket in buckets:
                    bucket["baseline_progress_false_positives"] += 1

    return {
        "thresholds": {
            "judge_failure_threshold": round(
                float(chosen_thresholds["judge_failure_threshold"]), 6
            ),
            "progress_failure_threshold": round(
                float(chosen_thresholds["progress_failure_threshold"]), 6
            ),
        },
        "calibration": calibration,
        "overall": _finalize_bucket(overall),
        "tasks": {task: _finalize_bucket(bucket) for task, bucket in per_task.items()},
        "families": {family: _finalize_bucket(bucket) for family, bucket in per_family.items()},
    }
