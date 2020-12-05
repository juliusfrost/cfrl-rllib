import os
import sys
import time
import argparse
import numpy as np
import pathlib
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
parser.add_argument('--load_path', default=None, help='Load env from this path')
parser.add_argument('--load_index', default=0, help='Which env to loead within load_path')
parser.add_argument('--save_path', default='.', help='if we save the trajectory, store it here.')
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

img_list = []
save_index = 0
done = False
env = envs.driving.driving_creator(switch_prob=0.0)
if args.load_path is not None:
    load_index = args.load_index
    with open(pathlib.Path(args.load_path).joinpath(f'env_state_{load_index}.pkl'), 'rb') as f:
        env_state = pkl.load(f)
        env.game_state.game.setGameState(*env_state)
    with open(pathlib.Path(args.load_path).joinpath(f'obs_{load_index}.pkl'), 'rb') as f:
        obs = pkl.load(f)
    driving_env = env.env.game_state.game
    driving_env.backdrop.update(driving_env.ydiff)
    driving_env.backdrop.draw_background(driving_env.screen)
    driving_env.cars_group.draw(driving_env.screen)
else:
    obs = env.reset()
env.render()
while not done:
    img_list.append(env.render(mode='rgb_array'))
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
        # Take the action the loaded policy would have taken.
        if policy is None:
            print("No policy found!")
        else:
            logits, _ = policy.model.from_batch({"obs": torch.FloatTensor(obs).unsqueeze(0).to(policy.device)})
            action_dist = policy.dist_class(logits, policy.model)
            action = action_dist.sample().detach().cpu().numpy()[0]
    if 'o' in a:
        # Save the frames up to this state, the current state, and the env
        save_path = pathlib.Path(args.save_path)
        if not save_path.exists():
            save_path.mkdir()
        with open(save_path.joinpath(f'env_state_{save_index}.pkl'), 'wb') as f:
            env_state = env.game_state.game.getGameStateSave()
            pkl.dump(env_state, f)
        with open(save_path.joinpath(f'obs_{save_index}.pkl'), 'wb') as f:
            pkl.dump(obs, f)
        with open(save_path.joinpath(f'traj_{save_index}.pkl'), 'wb') as f:
            pkl.dump(np.stack(img_list), f)
        save_index += 1
        save_name = save_path.joinpath(f'obs_{save_index}.pkl')
        print(f"Saved current state to {save_name}")
    print(action)
    if action is not None:
        obs, _, done, _ = env.step(action)
    env.render()
