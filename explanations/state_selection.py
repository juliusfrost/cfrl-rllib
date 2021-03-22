"""
State selection methods
Fill in necessary method parameters as needed
"""
import random
import torch
import numpy as np

from explanations.data import Data


def random_state(data: Data, num_states, policy, min_dist=20, **kwargs):
    indices = np.arange(len(data.all_time_steps))[~data.all_dones]  # don't include done states
    state_indices = list(zip(indices, data.all_trajectories))
    random.seed(kwargs.get('seed', None))
    random.shuffle(state_indices)
    selected_states = filter(state_indices, num_states, min_dist)
    print(selected_states)
    return selected_states


def filter(state_indices, num_states, min_dist):
    """
    Filter states down to a subset which aren't too close to each other
    :param state_indices: list of tuples (value, traj_id), sorted from most to least important.
    :param num_states: number of states to select
    :param min_dist: minimum number of timesteps between selected states
    :return: list of state indices of length num_states
    """
    blocked = [False] * len(state_indices)
    selected_states = []
    for i, (state, traj_id) in enumerate(state_indices):
        if len(selected_states) >= num_states:
            break
        if blocked[i]:
            continue
        for j in range(i + 1, len(state_indices)):
            if blocked[j]:
                continue
            other_state, other_traj_id = state_indices[j]
            if traj_id == other_traj_id and np.abs(state - other_state) < min_dist:
                blocked[j] = True
        selected_states.append(state)
    if len(selected_states) < num_states:
        num_states_remaining = num_states - len(selected_states)
        print(f"Warning: You requested {num_states} non-overlapping states. "
              f"We were only able to generate {len(selected_states)}, "
              f"so the remaining {num_states_remaining} may overlap.")
        for i in range(len(state_indices)):
            if len(selected_states) >= num_states:
                break
            if blocked[i]:
                selected_states.append(state_indices[i][0])
    return selected_states


def user_state(data: Data, num_states, user, **kwargs):
    # state_indices = user.query(data)
    # return state_indices
    raise NotImplementedError


def critical_state(data: Data, num_states, policy, min_dist=20, **kwargs):
    state_indices = []
    for obs_index, (obs, traj_id) in enumerate(zip(data.all_observations, data.all_trajectories)):
        # don't include done states
        if data.all_dones[obs_index]:
            continue
        with torch.no_grad():
            logits, _ = policy.model.from_batch({"obs": torch.FloatTensor(obs).unsqueeze(0).to(policy.device)})
            action_dist = policy.dist_class(logits, policy.model)
            entropy = action_dist.entropy()[0].item()
        state_indices.append((entropy, obs_index, traj_id))

    state_indices.sort(key=lambda x: x[0])
    state_indices = [(state, traj_id) for entropy, state, traj_id in state_indices]
    # Filter out states which are too near other states
    selected_states = filter(state_indices, num_states, min_dist)
    print(selected_states)
    return selected_states


def low_reward_state(data, num_states, **kwargs):
    # state_indices = None
    # return state_indices
    raise NotImplementedError
