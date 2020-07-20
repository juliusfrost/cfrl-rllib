"""
Action selection methods
Fill in necessary method parameters as needed
TODO: Implement
"""


def random_action(action_space, **kwargs):
    action = action_space.sample()
    return action


def policy_action(observation, policy, **kwargs):
    action = policy(observation)
    return action


def user_action(observation, user, **kwargs):
    action = user(observation)
    return action
