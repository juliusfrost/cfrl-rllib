from typing import Any

from ray.tune.registry import register_env

from envs.driving_env.ple_env import PLEEnv
from envs.save import SimulatorStateWrapper

NUM_STEPS = 5


class DrivingSimulatorStateWrapper(SimulatorStateWrapper):
    def get_simulator_state(self) -> Any:
        return self.env.game_state.game.getGameStateSave()

    def load_simulator_state(self, state: Any) -> bool:
        try:
            self.env.game_state.game.setGameState(*state)
            success = True
        except Exception as e:
            print(e)
            success = False
        return success


def env_creator(**kwargs):
    env = PLEEnv(
        game_name=kwargs.get('game_name', 'Driving'),
        display_screen=False,
        num_steps=NUM_STEPS,
        switch_prob=kwargs.get('switch_prob', 0.),
        switch_duration=kwargs.get('switch_duration', 50),
        action_dim=kwargs.get('action_dim', 2),
        collision_penalty=kwargs.get('collision_penalty', -100),
        time_reward_proportion=kwargs.get('time_reward_proportion', 1),
        normalize_reward=kwargs.get('normalize_reward', True),
    )
    env = DrivingSimulatorStateWrapper(env)
    return env


def register():
    def driving_acc_env_factory(env_config):
        return env_creator(game_name="Driving",
                           action_dim=env_config.get('action_dim', 2),
                           switch_prob=env_config.get('switch_prob', 0.),
                           switch_duration=env_config.get('switch_duration', 50),
                           collision_penalty=env_config.get('collision_penalty', -100),
                           time_reward_proportion=env_config.get('time_reward_proportion', 1),
                           normalize_reward=env_config.get('normalize_reward', True),
                           )

    register_env('DrivingPLE-v0', driving_acc_env_factory)
