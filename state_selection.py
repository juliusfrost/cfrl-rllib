"""
State selection methods
Fill in necessary method parameters as needed
TODO: Implement
"""


def random_state(data, num_states, **kwargs):
    state_indices = None
    return state_indices


def user_state(data, num_states, user, **kwargs):
    state_indices = user.query(data)
    return state_indices


def critical_state(data, num_states, policy, **kwargs):
    state_indices = []
    for trajectory in data.all_trajectories:
        for obs_index in trajectory.observation_range:
            obs = data.all_observations[obs_index]
            logits, _ = policy.model.from_batch({"obs": obs})
            action_dist = policy.dist_class(logits, policy.model)
            entropy = action_dist.entropy()

            if len(state_indices) < num_states:
                state_indices.append(entropy, obs_index)
            else:
                last = state_indices[-1]
                if entropy < last[0]:
                    state_indices[len(state_indices) - 1] = (entropy, obs_index)
                    state_indices.sort()

    # Strip out the entropies
    state_indices = [state for entropy, state in state_indices]
    return state_indices


def low_reward_state(data, num_states, **kwargs):
    state_indices = None
    return state_indices
