import gym
import gym_minigrid
import random

from ray.tune.registry import register_env

ENV_IDS = [
    'MiniGrid-FourRooms-v0',
]


class SmallFourRoomsEnv(gym_minigrid.envs.FourRoomsEnv):
    def __init__(self, agent_pos=None, goal_pos=None, grid_size=9, max_steps=100):
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos
        gym_minigrid.envs.MiniGridEnv.__init__(self, grid_size=grid_size, max_steps=max_steps)


class MiniGridObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = self.observation_space['image']

    def observation(self, observation):
        return observation['image']


class MiniGridActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(3)

    def action(self, action):
        return action

    def reverse_action(self, action):
        if action >= 3:
            raise NotImplementedError
        return action


def env_creator(**kwargs):
    env = SmallFourRoomsEnv(
        agent_pos=kwargs.get('agent_pos', (2, 2)),
        goal_pos=kwargs.get('goal_pos', (2, 6)),
        grid_size=kwargs.get('grid_size', 9),
        max_steps=kwargs.get('max_steps', 100),
    )
    env = gym_minigrid.wrappers.FullyObsWrapper(env)
    env = MiniGridObservationWrapper(env)
    env = MiniGridActionWrapper(env)
    env.seed(kwargs.get('seed', random.randint(0, 1000)))
    return env


def register(**kwargs):
    for env_name in ENV_IDS:
        def f(env_config):
            env_config['env_name'] = env_name
            return env_creator(**env_config)

        register_env(env_name, f)
