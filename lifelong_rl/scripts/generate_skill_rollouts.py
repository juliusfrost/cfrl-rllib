# Imports
import pathlib
import argparse
import sys
import torch
import cv2
import numpy as np
import imageio
sys.path.append('lifelong_rl')
from lifelong_rl.samplers.utils.rollout_functions import rollout  # TODO: rollout with latent?
from lifelong_rl.envs.env_processor import make_env

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_resets', type=int, default=2,
                        help='number of different env configurations to test')
    parser.add_argument('--num_skills', type=int, default=5,
                        help='number of different skills to test in each env configuration')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='number of different trials with each skill')
    parser.add_argument('--save_path', type=str, default='.',
                        help='where to save generated videos')
    parser.add_argument('--agent_path', type=str, default=None,
                        help='path to agent checkpoint')
    parser.add_argument('--env', type=str, default='MiniGrid',
                        help='which env to run')
    return parser

def make_video(traj_dict, save_path):
    # frames = [cv2.resize(img, dsize=(w, h), interpolation=cv2.INTER_CUBIC) for img in traj_dict['observations']]
    # TODO: don't hardcode for minigrid
    # obs = traj_dict['observations']
    # frames, obs_size = obs.shape
    # depth = 3
    # w = h = int(np.sqrt(obs_size / depth))
    # obs = obs.reshape(frames, h, w, depth)
    obs = traj_dict['images']
    # Add some darkness on the end so we know when it resets
    t, h, w, d = obs.shape
    obs = np.concatenate([obs, np.zeros((5, h, w, d), dtype=np.uint8)])
    imageio.mimwrite(save_path, obs, fps=5)

def load_agent(path):
    with open(path, 'rb') as f:
        snapshot = torch.load(f, map_location='cpu')
    return snapshot['exploration/policy']

def collect(num_resets, num_skills, num_samples, save_path):
    # Collect
    print("Saving to", save_path)
    for env_i in range(num_resets):
        print(f"Collecting {env_i + 1} of {num_resets}")
        # TODO: wrap env in a wrapper so it resets the same way each time. Either that or just choose a version of minigrid where it resets each time anyway.
        for trial_j in range(num_skills):
            agent.sample_latent()
            last_states = []
            for sample_k in range(num_samples):
                traj_dict = rollout(
                    env,
                    agent,
                    max_path_length=15,
                    render=True,
                    render_kwargs={'mode': 'rgb_array'},
                    save_render=True,
                )
                make_video(traj_dict, save_path.joinpath(f'env_{env_i}_trial_{trial_j}_sample{sample_k}.gif'))
                last_states.append(traj_dict['images'][-1])
            final_img = np.max(np.stack(last_states), axis=0)
            imageio.imwrite(save_path.joinpath(f'final_states_env_{env_i}_trial_{trial_j}.gif'), final_img)


class RandomAgent:
    def reset(self):
        pass

    def sample_latent(self):
        pass

    def get_action(self, *args, **kwargs):
        import random
        return random.choice([0, 1, 2, 3]), None

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    save_path = pathlib.Path(args.save_path)
    if not save_path.exists():
        save_path.mkdir()
    # Load trained agent
    if args.agent_path is None:
        print("No agent found, using random agent")
        agent = RandomAgent()
    else:
        agent = load_agent(args.agent_path)
    env, _ = make_env(args.env)

    collect(args.num_resets, args.num_skills, args.num_samples, save_path)