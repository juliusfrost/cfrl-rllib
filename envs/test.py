import os
import sys
import time
import argparse
import pickle as pkl
import ray
from ray.tune.registry import get_trainable_cls
import torch
import envs

sys.path.append('.')
ray.init()
envs.register()
parser = argparse.ArgumentParser()
parser.add_argument('--policy', default=None, help='policy to run')
parser.add_argument('--algo', default='PPO', help='algorithm to run')
args = parser.parse_args()

# Load Policy
if args.policy is not None:
    config_dir = os.path.dirname(args.policy)
    config_path = os.path.join(config_dir, "params.pkl")
    # Try parent directory.
    if not os.path.exists(config_path):
        config_path = os.path.join(config_dir, "../params.pkl")
    print("CONFIG PATH", config_path)
    with open(config_path, "rb") as f:
        config = pkl.load(f)
    cls = get_trainable_cls(args.algo)
    agent = cls(env=config["env"], config=config)
    # Load state from checkpoint.
    agent.restore(args.policy)
    policy = agent.get_policy()
else:
    policy = None


env = envs.driving.driving_creator(switch_prob=0.0)
obs = env.reset()
env.render()
done = False
while not done:
    # action = env.action_space.sample()
    action = [0,0]
    a = input()
    # action[0] = +left -right, action[1] = +forward, -backward
    if 'a' in a:
        action[0] += 1
    if 'd' in a:
        action[0] -= 1
    if 'w' in a:
        action[1] += 1
    if 's' in a:
        action[1] -= 1
    if 'v' in a:
        saved_state, time_steps, sp, rng = env.game_state.game.getGameStateSave()
    if 'r' in a:
        env.game_state.game.setGameState(saved_state, time_steps, sp, rng)
    if 'p' in a:
        if policy is None:
            print("No policy found!")
        else:
            logits, _ = policy.model.from_batch({"obs": torch.FloatTensor(obs).unsqueeze(0).to(policy.device)})
            action_dist = policy.dist_class(logits, policy.model)
            action = action_dist.sample().detach().cpu().numpy()[0]
    print(action)
    obs, _, done, _ = env.step(action)
    env.render()
