"""
Action selection methods
Fill in necessary method parameters as needed
"""
from collections import namedtuple


def policy_action(observation, policy, **kwargs):
    action = policy(observation)
    return action


def user_action(observation, user, **kwargs):
    action = user(observation)
    return action


Policy = namedtuple("Policy", "action_space")

class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space
        self.policy = Policy(action_space=action_space)

    def compute_action(self, obs, **kwargs):
        return self.action_space.sample()


def constant_generator(n_timesteps):
    for i in range(n_timesteps):
        yield False
    yield True

def make_handoff_func(n_timesteps):
    gen = constant_generator(n_timesteps)
    def handoff(state, action):
        return next(gen)
    return handoff