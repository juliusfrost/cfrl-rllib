import copy
import random

import gym
from gym.spaces import Box
import gym_minigrid
from gym_minigrid.envs import MiniGridEnv, FourRoomsEnv
from ray.tune.registry import register_env

from envs.save import SimulatorStateWrapper

ENV_IDS = [
    'MiniGrid-FourRooms-v0',
]


class SmallFourRoomsEnv(FourRoomsEnv):
    def __init__(self, agent_pos=None, goal_pos=None, grid_size=9, max_steps=100):
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos
        MiniGridEnv.__init__(self, grid_size=grid_size, max_steps=max_steps)


class MiniGridObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = self.observation_space['image']

    def observation(self, observation):
        return observation['image']

class MiniGridFlatWrapper(MiniGridObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Existing observation space is a box (9, 9, 3)
        self.observation_space = Box(low=self.observation_space.low.flatten(),
                                     high=self.observation_space.high.flatten())

    def observation(self, observation):
        return observation['image'].flatten()


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


class MiniGridSimulatorStateWrapper(SimulatorStateWrapper):
    def get_simulator_state(self) -> MiniGridEnv:
        return copy.deepcopy(self.env)

    def load_simulator_state(self, state: MiniGridEnv) -> bool:
        try:
            self.env = None
            self.env = copy.deepcopy(state)
            success = True
        except Exception as e:
            print(e)
            success = False
        return success


def env_creator(normalize=False, normalize_constant=10, flat_obs=False, **kwargs):
    env = SmallFourRoomsEnv(
        agent_pos=kwargs.get('agent_pos', (2, 2)),
        goal_pos=kwargs.get('goal_pos', (2, 6)),
        grid_size=kwargs.get('grid_size', 9),
        max_steps=kwargs.get('max_steps', 100),
    )
    env = MiniGridSimulatorStateWrapper(env)
    env = gym_minigrid.wrappers.FullyObsWrapper(env)
    if flat_obs:
        env = MiniGridFlatWrapper(env)
    else:
        env = MiniGridObservationWrapper(env)
    env = MiniGridActionWrapper(env)
    if normalize:
        env = gym.wrappers.TransformObservation(env, lambda obs: obs / normalize_constant)
    env.seed(kwargs.get('seed', random.randint(0, 1000)))
    return env


def register():
    for env_name in ENV_IDS:
        def f(env_config):
            env_config['env_name'] = env_name
            return env_creator(**env_config)

        register_env(env_name, f)
