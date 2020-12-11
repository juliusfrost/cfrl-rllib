"""
State selection methods
Fill in necessary method parameters as needed
"""
import random
import torch
import numpy as np

from explanations.data import Data


def random_state(data: Data, num_states, policy, min_dist=20, **kwargs):
    state_indices = list(zip(data.all_time_steps, data.all_trajectories))
    random.seed(kwargs.get('seed', None))
    random.shuffle(state_indices)
    selected_states = filter(state_indices, num_states, min_dist)
    return selected_states


def filter(state_indices, num_states, min_dist):
    """
    Filter states down to a subset which aren't too close to each other
    :param state_indices: list of tuples (value, traj_id), sorted from most to least important.
    :param num_states: number of states to select
    :param min_dist: minimum number of timesteps between selected states
    :return: list of state indices of length num_states
    """
    selected_states = []
    for state, traj_id in state_indices:
        if len(selected_states) == num_states:  # Done
            break
        too_close = False
        for other_state, other_traj_id in selected_states:
            # Too close to another selected state
            if traj_id == other_traj_id and np.abs(state - other_state) < min_dist:
                too_close = True
                break
        if not too_close:
            selected_states.append((state, traj_id))
    if len(selected_states) < num_states:
        num_states_remaining = num_states - len(selected_states)
        print(f"Warning: You requested {num_states} non-overlapping states. "
              f"We were only able to generate {len(selected_states)}, "
              f"so the remaining {num_states_remaining} may overlap.")
        # This time through, include states which are less than min_dist apart, so long as we haven't selected them yet.
        for state_tuple in state_indices:
            if len(selected_states) == num_states:
                break
            if not state_tuple in selected_states:
                selected_states.append(state_tuple)
    return [state for state, traj_id in selected_states]


def user_state(data: Data, num_states, user, **kwargs):
    # state_indices = user.query(data)
    # return state_indices
    raise NotImplementedError


def critical_state(data: Data, num_states, policy, min_dist=20, **kwargs):
    state_indices = []
    for obs_index, (obs, traj_id) in enumerate(zip(data.all_observations, data.all_trajectories)):
        with torch.no_grad():
            logits, _ = policy.model.from_batch({"obs": torch.FloatTensor(obs).unsqueeze(0).to(policy.device)})
            action_dist = policy.dist_class(logits, policy.model)
            entropy = action_dist.entropy()[0].item()
        state_indices.append((entropy, obs_index, traj_id))

    state_indices.sort(key=lambda x: x[0])
    state_indices = [(state, traj_id) for entropy, state, traj_id in state_indices]
    # Filter out states which are too near other states
    selected_states = filter(state_indices, num_states, min_dist)
    # Strip out the entropies
    return selected_states


def low_reward_state(data, num_states, **kwargs):
    # state_indices = None
    # return state_indices
    raise NotImplementedError
