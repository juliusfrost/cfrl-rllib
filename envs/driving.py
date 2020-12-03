from typing import Any

import gym
import numpy as np
from ray.tune.registry import register_env

from envs.driving_env.ple_env import PLEEnv
from envs.save import SimulatorStateWrapper


class DrivingSimulatorStateWrapper(SimulatorStateWrapper):
    def get_simulator_state(self) -> Any:
        inner_state = self.env.game_state.game.getGameStateSave()
        wrapper_state = (
            self.env.game_state.previous_score, self.env.game_state.last_action, self.env.game_state.action)
        return (inner_state, wrapper_state)

    def load_simulator_state(self, state: Any) -> bool:
        try:
            inner_state, (previous_score, last_action, action) = state
            self.env.game_state.game.setGameState(*inner_state)
            self.env.game_state.previous_score = previous_score
            self.env.game_state.last_action = last_action
            self.env.game_state.action = action
            success = True
        except Exception as e:
            print(e)
            success = False
        return success


class NoisyActionWrapper(gym.ActionWrapper):
    def __init__(self, env, action_noise=0.):
        super().__init__(env)
        self._action_noise = action_noise

    def action(self, action):
        noise = np.random.randn(*action.shape)
        return action + self._action_noise * noise

    def reverse_action(self, action):
        return action


def driving_creator(
        # if greater than 0, adds random normal distributed multiplied by this constant to actions
        action_noise=0.,
        # other kwargs for the PLE Driving environment
        **kwargs):
    env = PLEEnv(
        # cannot change
        game_name='Driving',
        # display screen on monitor
        display_screen=kwargs.get('display_screen', False),
        # whether to use image observations
        img_input=kwargs.get('image_observations', False),
        # whether to use continuous actions
        continuous_actions=kwargs.get('continuous_actions', True),
        # number of action repeat (frame skip)
        num_steps=kwargs.get('action_repeat', 5),
        # applicable only if using image observations
        obs_width=kwargs.get('obs_width', 84),
        # applicable only if using image observations
        obs_height=kwargs.get('obs_height', 84),
        # applicable only if using image observations
        n_frames=kwargs.get('n_frames', 4),
        # probability of cars switching lanes
        switch_prob=kwargs.get('switch_prob', 0.),
        # time for cars to switch lanes
        switch_duration=kwargs.get('switch_duration', 50),
        # dimension of actions
        action_dim=2,
        # reward penalty for collision
        collision_penalty=kwargs.get('collision_penalty', -100),
        # proportion to use per time step reward bonus
        time_reward_proportion=kwargs.get('time_reward_proportion', 1),
        # normalize rewards by dividing by constant
        normalize_reward=kwargs.get('normalize_reward', True),
        # environment time step limit.
        # agent time steps = environment time steps / action repeat
        time_limit=kwargs.get('time_limit', 500),
        # weights for the reward feature vector
        # list of 6 floating-point numbers
        # [ft_lanes, ft_speed, ft_carnear, ft_turn, ft_forward, ft_sharpturn]
        # ft_lanes: Dist from center of lane
        # ft_speed: Not going over speed limit
        # ft_carnear: Distance from other cars
        # ft_turn: Something about turning
        # ft_forward: Making progress forward
        # ft_sharpturn: Does it make sharp turns?
        theta=kwargs.get('reward_feature_weights', [-1., 0., -10., -1., 1., -0.01, 0., 0., 0., 0., 0.]),
        # probability of generating a car in a new time step
        prob_car=kwargs.get('prob_car', 0.5),
        # steering resistance is the scalar that divides the steering angle to reduce sharp angles
        steering_resistance=kwargs.get('steering_resistance', 100.),
        # Speed multiplier of k sets the target speed for the agent to k * robot car speed
        speed_multiplier=kwargs.get('speed_multiplier', 1.0),
        # speed ratio controls the car's speed
        players_speed_ratio_max=kwargs.get('players_speed_ratio_max', 0.2),
        players_speed_ratio_min=kwargs.get('players_speed_ratio_min', 0.1),
        cpu_speed_ratio_max=kwargs.get('cpu_speed_ratio_max', 0.1),
        cpu_speed_ratio_min=kwargs.get('cpu_speed_ratio_min', 0.05),
        # Friction coefficient
        # if > 0, the car slows down to the minimum speed if no acceleration is applied.
        friction_coefficient=kwargs.get('friction_coefficient', 0.),
    )
    env = DrivingSimulatorStateWrapper(env)
    if action_noise > 0.:
        env = NoisyActionWrapper(env, action_noise)
    return env


def register():
    register_env('DrivingPLE-v0', lambda env_config: driving_creator(**env_config))
