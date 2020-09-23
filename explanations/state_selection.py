"""
State selection methods
Fill in necessary method parameters as needed
"""
import torch

from explanations.data import Data


def random_state(data: Data, num_states, policy, **kwargs):
    traj_timesteps = [data.get_trajectory(traj).get_time_steps()[0].time_step_id for traj in data.all_trajectory_ids]
    state_indices = traj_timesteps
    return state_indices[:num_states]


def user_state(data: Data, num_states, user, **kwargs):
    state_indices = user.query(data)
    return state_indices


def critical_state(data: Data, num_states, policy, **kwargs):
    state_indices = []
    for obs_index, obs in enumerate(data.all_observations):
        with torch.no_grad():
            logits, _ = policy.model.from_batch({"obs": torch.FloatTensor(obs).unsqueeze(0)})
            action_dist = policy.dist_class(logits, policy.model)
            entropy = action_dist.entropy()[0].item()
        state_indices.append((entropy, obs_index))

    state_indices.sort(key=lambda x: x[0])
    # Strip out the entropies
    state_indices = [state for entropy, state in state_indices[:num_states]]
    return state_indices


def low_reward_state(data, num_states, **kwargs):
    state_indices = None
    return state_indices
