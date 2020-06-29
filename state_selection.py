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
    state_indices = None
    return state_indices


def low_reward_state(data, num_states, **kwargs):
    state_indices = None
    return state_indices
