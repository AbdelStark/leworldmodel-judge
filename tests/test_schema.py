from leworldmodel_judge.schema import RolloutStep, PrefixRecord


def test_schema_to_dict():
    step = RolloutStep(
        episode_id='ep1',
        task_id='reach-v2',
        timestep=0,
        episode_horizon=20,
        observation=[0.1, 0.2],
        action=[0.0],
        reward=0.0,
        done=False,
        success_label=False,
    )
    prefix = PrefixRecord(
        episode_id='ep1',
        task_id='reach-v2',
        prefix_index=5,
        prefix_fraction=0.25,
        final_success_label=False,
        prefix_failure_label=True,
        prefix_recoverability_label='doomed',
        sparse_reward_prefix=-0.1,
    )
    assert step.to_dict()['task_id'] == 'reach-v2'
    assert prefix.to_dict()['prefix_recoverability_label'] == 'doomed'
