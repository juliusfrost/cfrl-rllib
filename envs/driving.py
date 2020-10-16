from typing import Any

from ray.tune.registry import register_env

from envs.driving_env.ple_env import PLEEnv
from envs.save import SimulatorStateWrapper

NUM_STEPS = 5


class DrivingSimulatorStateWrapper(SimulatorStateWrapper):
    def get_simulator_state(self) -> Any:
        inner_state = self.env.game_state.game.getGameStateSave()
        wrapper_state = (self.env.game_state.previous_score, self.env.game_state.last_action, self.env.game_state.action)
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


def driving_creator(**kwargs):
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
        num_steps=kwargs.get('action_repeat', NUM_STEPS),
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
        action_dim=kwargs.get('action_dim', 2),
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
        # TODO: figure out what each feature exactly means
        theta=kwargs.get('reward_feature_weights', [-1., 0., -10., -1., 1., -0.01])
    )
    env = DrivingSimulatorStateWrapper(env)
    return env


def register():
    register_env('DrivingPLE-v0', lambda env_config: driving_creator(**env_config))
