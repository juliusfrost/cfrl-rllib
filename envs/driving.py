import gym
import gym_minigrid
import random

from ray.tune.registry import register_env


from envs.driving_env.ple_env import PLEEnv

NUM_STEPS = 5

def env_creator(**kwargs):
    env = PLEEnv(
        game_name=kwargs.get('game_name', 'Driving'),
        display_screen=False,
        num_steps=NUM_STEPS,
        switch_prob=kwargs.get('switch_prob', 0.0),
        switch_duration = kwargs.get('switch_duration', 50),
        action_dim=kwargs.get('action_dim', 2)
    )
    return env


def register(**kwargs):
    switch_prob = kwargs.get('switch_prob', 0.0)
    switch_duration = kwargs.get('switch_duration', 50)
    env_name = 'DrivingAccPLE-v0'
    env_func = lambda _: env_creator(game_name="Driving", action_dim=2, switch_prob=switch_prob, swtich_duration=switch_duration)
    register_env(env_name, env_func)

    env_name = 'DrivingPLE-v0'
    env_func = lambda _: env_creator(game_name="Driving")
    register_env(env_name, env_func)