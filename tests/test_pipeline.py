from leworldmodel_judge.data import build_prefixes
from leworldmodel_judge.baselines import score_prefix
from leworldmodel_judge.judge import heuristic_surprise_score


def test_prefix_pipeline_smoke():
    steps = []
    for t in range(4):
        steps.append({
            'episode_id': 'ep1',
            'task_id': 'reach-v3',
            'timestep': t,
            'episode_horizon': 4,
            'observation': [1.0 - t * 0.2, 0.0],
            'action': [0.0],
            'reward': 0.0,
            'done': t == 3,
            'success_label': True,
        })
    prefixes = build_prefixes(steps, (0.5,))
    assert len(prefixes) == 1
    baseline = score_prefix(prefixes[0].to_dict())
    judge = heuristic_surprise_score(prefixes[0].to_dict())
    assert baseline['task_id'] == 'reach-v3'
    assert 'failure_score' in judge
