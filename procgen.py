import gym, ray
from gym import spaces
from ray.rllib.agents import ppo
import cv2
import numpy as np
import copy
import argparse

class ProcgenEnv(gym.Env):
    def __init__(self, env_config):
        env_name = env_config["env_name"]
        env_config_copy = copy.deepcopy(env_config)
        del env_config_copy["env_name"]
        self.env = gym.make(env_name, **env_config_copy)
        self.action_space = self.env.action_space
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)

    def reset(self):
        o = self.env.reset()
        return cv2.resize(o, (84, 84))

    def step(self, action):
        o, r, d, i = self.env.step(action)
        return cv2.resize(o, (84, 84)), r, d, i

parser = argparse.ArgumentParser()
parser.add_argument('--num_levels', type=int, default=0,
                    help='number of unique levels to train on (0 means unlimited)')
parser.add_argument('--env', type=str, required=True, help='environment name')
parser.add_argument('--difficulty', type=str, choices=['easy', 'hard'], default='easy')
args = parser.parse_args()


filters = filters_84x84 = [
        [32, [8, 8], 4],
        [256, [4, 4], 2],
        [512, [4, 4], 1],
        [1024, [11, 11], 1],
    ]


ray.init()
trainer = ppo.PPOTrainer(env=ProcgenEnv, config={
    "entropy_coeff": 0.01,
    "lambda": .95,
    "clip_param": 0.2,
    "gamma": 0.999,
    "horizon": 256,
    "num_workers": 1,
    "num_envs_per_worker": 64,
    "lr": 5e-4,
    "model": {
        "conv_filters": filters
    },
    "train_batch_size": 512,
    "env_config": {
        "env_name": f'procgen:procgen-{args.env}-v0',
        "num_levels": args.num_levels,
        "distribution_mode": args.difficulty,
    },
})
trainer = ppo.PPOTrainer(env=ProcgenEnv, "env_config": {
        "env_name": f'procgen:procgen-{args.env}-v0',
        "num_levels": args.num_levels,
        "distribution_mode": args.difficulty,
    })



print("Training!!!!!!")
while True:
    print(trainer.train())
