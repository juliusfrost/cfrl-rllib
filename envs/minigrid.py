import copy
import random

import numpy as np
import gym
import gym_minigrid
from gym_minigrid.envs import MiniGridEnv, Grid, Goal
from ray.tune.registry import register_env

from envs.save import SimulatorStateWrapper

ENV_IDS = [
    'MiniGrid-FourRooms-v0',
]


class ModifiedFourRoomsEnv(MiniGridEnv):
    """
    Classic 4 rooms gridworld environment.
    Can specify agent and goal position, if not it set at random.
    """

    def __init__(self, agent_pos=None, goal_pos=None, deterministic_rooms=False, **kwargs):
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos
        self.deterministic_rooms = deterministic_rooms
        super().__init__(**kwargs)

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        room_w = width // 2
        room_h = height // 2

        # For each row of rooms
        for j in range(0, 2):

            # For each column
            for i in range(0, 2):
                xL = i * room_w
                yT = j * room_h
                xR = xL + room_w
                yB = yT + room_h

                # Bottom wall and door
                if i + 1 < 2:
                    self.grid.vert_wall(xR, yT, room_h)
                    if self.deterministic_rooms:
                        pos = (xR, (yT + 1 + yB) // 2)
                    else:
                        pos = (xR, self._rand_int(yT + 1, yB))
                    self.grid.set(*pos, None)

                # Bottom wall and door
                if j + 1 < 2:
                    self.grid.horz_wall(xL, yB, room_w)
                    if self.deterministic_rooms:
                        pos = ((xL + 1 + xR) // 2, yB)
                    else:
                        pos = (self._rand_int(xL + 1, xR), yB)
                    self.grid.set(*pos, None)

        # Randomize the player start position and orientation
        if self._agent_default_pos is not None:
            self.agent_pos = self._agent_default_pos
            self.grid.set(*self._agent_default_pos, None)
            self.agent_dir = self._rand_int(0, 4)  # assuming random start direction
        else:
            self.place_agent()

        if self._goal_default_pos is not None:
            goal = Goal()
            self.put_obj(goal, *self._goal_default_pos)
            goal.init_pos, goal.cur_pos = self._goal_default_pos
        else:
            self.place_obj(Goal())

        self.mission = 'Reach the goal'

    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)
        return obs, reward, done, info


class SmallModifiedFourRoomsEnv(ModifiedFourRoomsEnv):
    def __init__(self, agent_pos=None, goal_pos=None, grid_size=9, max_steps=100, deterministic_rooms=False, **kwargs):
        super().__init__(agent_pos, goal_pos, deterministic_rooms, grid_size=grid_size, max_steps=max_steps, **kwargs)


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


class MiniGridResetSeedWrapper(gym.Wrapper):
    def __init__(self, env, seed):
        super().__init__(env)
        self._reset_seed = seed

    def reset(self, **kwargs):
        self.seed(self._reset_seed)
        return super().reset(**kwargs)


def env_creator(normalize=False, normalize_constant=10, **kwargs):
    env = SmallModifiedFourRoomsEnv(
        agent_pos=kwargs.get('agent_pos', (2, 2)),
        goal_pos=kwargs.get('goal_pos', (2, 6)),
        grid_size=kwargs.get('grid_size', 9),
        max_steps=kwargs.get('max_steps', 100),
        deterministic_rooms=kwargs.get('deterministic_rooms', False),
    )
    env = MiniGridSimulatorStateWrapper(env)
    env = gym_minigrid.wrappers.FullyObsWrapper(env)
    env = MiniGridObservationWrapper(env)
    env = MiniGridActionWrapper(env)
    if normalize:
        env = gym.wrappers.TransformObservation(env, lambda obs: obs / normalize_constant)
    if 'reset_seed' in kwargs:
        env = MiniGridResetSeedWrapper(env, kwargs.get('reset_seed'))
    env.seed(kwargs.get('seed', random.randint(0, 1000)))
    return env


def register():
    for env_name in ENV_IDS:
        def f(env_config):
            env_config['env_name'] = env_name
            return env_creator(**env_config)

        register_env(env_name, f)
