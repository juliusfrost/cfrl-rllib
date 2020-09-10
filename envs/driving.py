from ray.tune.registry import register_env

from envs.driving_env.ple_env import PLEEnv

NUM_STEPS = 5


def env_creator(**kwargs):
    env = PLEEnv(
        game_name=kwargs.get('game_name', 'Driving'),
        display_screen=False,
        num_steps=NUM_STEPS,
        switch_prob=kwargs.get('switch_prob', 0.0),
        switch_duration=kwargs.get('switch_duration', 50),
        action_dim=kwargs.get('action_dim', 2)
    )
    return env


def register():

    def driving_acc_env_factory(env_config):
        env_creator(game_name="Driving",
                    action_dim=env_config.get('action_dim', 2),
                    switch_prob=env_config.get('switch_prob', 50),
                    switch_duration=env_config.get('switch_duration', 50))

    register_env('DrivingAccPLE-v0', driving_acc_env_factory)

    def driving_env_factory(env_config):
        return env_creator(game_name="Driving")

    register_env('DrivingPLE-v0', driving_env_factory)
