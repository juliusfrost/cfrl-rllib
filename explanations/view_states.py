import pickle as pkl
import os
import numpy as npy
import argparse
from state_selection import random_state, critical_state, low_reward_state
from matplotlib import pyplot as plt
from envs import register

register()
from envs.driving import register as registerD
registerD()

# Example usage:
# python explanations/view_states.py --dataset_dir  saved_dataset/testing.pkl --num_states 4 --save_name "found_states" --state_selection_method critical

def select_states(args):
    with open(args.dataset_dir, "rb") as f:
        dataset = pkl.load(f)
    policy = dataset.policy.get_policy()
    state_selection_dict = {
        "critical": critical_state,
        "random": random_state,
        "low_reward": low_reward_state,
    }
    state_selection_fn = state_selection_dict[args.state_selection_method]
    state_indices = state_selection_fn(dataset, args.num_states, policy)
    if args.save_name is not None:
        if not os.path.exists(args.save_name):
            os.makedirs(args.save_name)
        for i in state_indices:
            img = dataset.get_timestep(i).image_observation
            plt.imshow(img)
            plt.savefig(args.save_name + f"/state{i}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True, help='pkl file containing the dataset')
    parser.add_argument('--num_states', type=int, default=10, help='Number of states to select.')
    parser.add_argument('--save_name', type=str, help='Place to save states found.')
    parser.add_argument('--state_selection_method', type=str, help='State selection method.',
                        choices=['critical', 'random', 'low_reward'], default='critical')
    args = parser.parse_args()
    select_states(args)
