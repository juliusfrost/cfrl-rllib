import argparse
import os
import pickle as pkl

import cv2

from envs import register
from explanations.data import Data
from explanations.state_selection import random_state, critical_state, low_reward_state

register()


def write_video(frames, filename, image_shape, fps=5):
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    writer = cv2.VideoWriter(filename, fourcc, fps, image_shape)
    for img in frames:
        writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def select_states(args):
    with open(args.dataset_file, "rb") as f:
        dataset: Data = pkl.load(f)
    policy = dataset.policy.get_policy()
    state_selection_dict = {
        "critical": critical_state,
        "random": random_state,
        "low_reward": low_reward_state,
    }
    state_selection_fn = state_selection_dict[args.state_selection_method]
    state_indices = state_selection_fn(dataset, args.num_states, policy)
    if args.save_path is not None:
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        for i in state_indices:
            timestep = dataset.get_timestep(i)
            trajectory = dataset.get_trajectory(timestep.trajectory_id)
            split = i - trajectory.timestep_range_start
            imgs = trajectory.image_observation_range

            img_shape = (imgs.shape[2], imgs.shape[1])
            context_file = os.path.join(args.save_path, f'{trajectory.trajectory_id}_{i}_context.avi')
            explanation_file = os.path.join(args.save_path, f'{trajectory.trajectory_id}_{i}_explanation.avi')

            write_video(imgs[:split], context_file, img_shape)
            write_video(imgs[split:], explanation_file, img_shape)


def save_trajectories(args):
    with open(args.dataset_file, "rb") as f:
        dataset: Data = pkl.load(f)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    for t in dataset.all_trajectories:
        trajectory = dataset.get_trajectory(t)
        file = os.path.join(args.save_path, f'{trajectory.trajectory_id}.avi')
        frames = trajectory.image_observation_range
        img_shape = (frames.shape[2], frames.shape[1])
        write_video(frames, file, img_shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-file', type=str, required=True, help='pkl file containing the dataset')
    parser.add_argument('--num-states', type=int, default=10, help='Number of states to select.')
    parser.add_argument('--save-path', type=str, default='videos', help='Place to save states found.')
    parser.add_argument('--save-trajectories', action='store_true')
    parser.add_argument('--state-selection-method', type=str, help='State selection method.',
                        choices=['critical', 'random', 'low_reward'], default='critical')
    args = parser.parse_args()
    if args.save_trajectories:
        save_trajectories(args)
    else:
        select_states(args)
