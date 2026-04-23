from pathlib import Path

from leworldmodel_judge.data import build_prefixes
from leworldmodel_judge.evaluate import summarize
from leworldmodel_judge.judge import heuristic_surprise_score


def test_pick_place_late_grasp_without_transport_is_labeled_doomed():
    steps = []
    distances = [0.34, 0.31, 0.30, 0.30]
    for t, distance in enumerate(distances):
        steps.append({
            'episode_id': 'pick-place-doomed-1',
            'task_id': 'pick-place-v3',
            'timestep': t,
            'episode_horizon': len(distances),
            'observation': [0.0, 0.0],
            'action': [0.0],
            'reward': 0.0,
            'done': t == len(distances) - 1,
            'success_label': False,
            'info': {
                'near_object': 1.0 if t >= 1 else 0.0,
                'obj_to_target': distance,
                'in_place_reward': 0.12,
                'grasp_success': 0.0,
                'grasp_reward': 1.0 if t >= 1 else 0.0,
                'success': 0.0,
                'unscaled_reward': 0.08,
            },
        })

    prefix = build_prefixes(steps, (0.75,))[0].to_dict()

    assert prefix['prefix_failure_label'] is True
    assert prefix['prefix_recoverability_label'] == 'doomed'


def test_judge_is_more_patient_on_early_engaged_prefixes_than_late_stalled_ones():
    early = {
        'episode_id': 'ep-early',
        'task_id': 'pick-place-v3',
        'prefix_fraction': 0.25,
        'progress_proxy': 0.0,
        'sparse_reward_prefix': 0.0,
        'distance_progress': 0.0,
        'target_distance_last': 0.30,
        'target_distance_best': 0.30,
        'in_place_score': 0.14,
        'near_object_score': 1.0,
        'success_signal_peak': 0.0,
        'grasp_signal_peak': 1.0,
        'reward_density': 0.08,
        'stall_score': 1.0,
    }
    late = dict(early)
    late['episode_id'] = 'ep-late'
    late['prefix_fraction'] = 0.75
    late['target_distance_last'] = 0.42
    late['target_distance_best'] = 0.28

    early_score = heuristic_surprise_score(early)['failure_score']
    late_score = heuristic_surprise_score(late)['failure_score']

    assert early_score < 0.5
    assert late_score > early_score


def test_summary_reports_calibrated_thresholds_and_family_slices(tmp_path):
    prefixes = [
        {
            'episode_id': 'push-v3-weak-ep-0',
            'task_id': 'push-v3',
            'policy_family': 'weak',
            'prefix_fraction': 0.75,
            'prefix_failure_label': False,
        },
        {
            'episode_id': 'push-v3-doomed-ep-0',
            'task_id': 'push-v3',
            'policy_family': 'doomed',
            'prefix_fraction': 0.75,
            'prefix_failure_label': True,
        },
    ]
    baselines = [
        {
            'episode_id': 'push-v3-weak-ep-0',
            'task_id': 'push-v3',
            'policy_family': 'weak',
            'prefix_fraction': 0.75,
            'sparse_reward_score': 0.0,
            'progress_proxy_score': 0.45,
            'terminal_success_score': 1.0,
        },
        {
            'episode_id': 'push-v3-doomed-ep-0',
            'task_id': 'push-v3',
            'policy_family': 'doomed',
            'prefix_fraction': 0.75,
            'sparse_reward_score': 0.0,
            'progress_proxy_score': 0.35,
            'terminal_success_score': 0.0,
        },
    ]
    judge = [
        {
            'episode_id': 'push-v3-weak-ep-0',
            'task_id': 'push-v3',
            'policy_family': 'weak',
            'prefix_fraction': 0.75,
            'failure_score': 0.31,
        },
        {
            'episode_id': 'push-v3-doomed-ep-0',
            'task_id': 'push-v3',
            'policy_family': 'doomed',
            'prefix_fraction': 0.75,
            'failure_score': 0.72,
        },
    ]

    summary = summarize(prefixes, baselines, judge)

    assert summary['calibration']['judge']['recommended_threshold'] == 0.72
    assert summary['families']['doomed']['failure_labels'] == 1
    assert summary['families']['weak']['non_failure_labels'] == 1

    report_dir = tmp_path / 'report'
    report_dir.mkdir()
    from scripts.render_family_report import render_family_report

    outputs = render_family_report(summary, report_dir)
    assert Path(outputs['markdown']).exists()
    assert Path(outputs['plot']).exists()