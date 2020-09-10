import argparse
import os
import pickle as pkl
from collections import namedtuple

import cv2
import numpy as np

from envs import register
from envs.driving import env_creator
from explanations.action_selection import RandomAgent, make_handoff_func, until_end_handoff
from explanations.create_dataset import create_dataset
from explanations.data import Data
from explanations.rollout import RolloutSaver, rollout_env
from explanations.state_selection import random_state, critical_state, low_reward_state

register()


def write_video(frames, filename, image_shape, fps=5):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(filename, fourcc, fps, image_shape)
    for img in frames:
        writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


DatasetArgs = namedtuple("DatasetArgs", ["out", "env", "run", "checkpoint"])


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

    exploration_rollout_saver = RolloutSaver(
        outfile=args.save_path + "/exploration.pkl",
        target_steps=None,
        target_episodes=args.num_states,
        save_info=True)
    # Neither of these seem to be used, just required to save dataset
    exploration_args = DatasetArgs(out=args.save_path + "/exploration.pkl", env="DrivingPLE-v0", run="PPO",
                                   checkpoint=1)
    exploration_policy_config = {}

    counterfactual_rollout_saver = RolloutSaver(
        outfile=args.save_path + "/counterfactual.pkl",
        target_steps=None,
        target_episodes=args.num_states,
        save_info=True)
    # Neither of these seem to be used, just required to save dataset
    counterfactual_args = DatasetArgs(out=args.save_path + "/counterfactual.pkl", env="DrivingPLE-v0", run="PPO",
                                      checkpoint=1)
    counterfactual_policy_config = {}

    env = env_creator()
    env.reset()
    cf_to_exp_index = {}
    cf_count = 0
    if args.save_path is not None:
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        for exp_index, i in enumerate(state_indices):
            timestep = dataset.get_timestep(i)
            simulator_state = timestep.simulator_state
            obs = timestep.observation
            env.game_state.game.setGameState(*simulator_state)
            random_agent = RandomAgent(env.action_space)
            handoff_func = make_handoff_func(args.timesteps)

            with exploration_rollout_saver as saver:
                exp_env, env_obs, env_done = rollout_env(random_agent, env, handoff_func, obs, saver=saver,
                                                         no_render=False)

            if not env_done:
                cf_to_exp_index[cf_count] = exp_index
                cf_count += 1
                with counterfactual_rollout_saver as saver:
                    rollout_env(random_agent, exp_env, until_end_handoff, env_obs, saver=saver, no_render=False)

        exploration_dataset = create_dataset(exploration_args, exploration_policy_config)
        counterfactual_dataset = create_dataset(counterfactual_args, counterfactual_policy_config)

        for cf_i in range(len(counterfactual_dataset.all_trajectory_ids)):
            cf_id = counterfactual_dataset.all_trajectory_ids[cf_i]
            i = cf_to_exp_index[cf_id]

            exp_id = exploration_dataset.all_trajectory_ids[i]
            state_index = state_indices[i]
            pre_timestep = dataset.get_timestep(state_index)

            original_trajectory = dataset.get_trajectory(pre_timestep.trajectory_id)
            exp_trajectory = exploration_dataset.get_trajectory(exp_id)
            cf_trajectory = counterfactual_dataset.get_trajectory(cf_id)
            original_imgs = original_trajectory.image_observation_range
            exp_imgs = exp_trajectory.image_observation_range
            cf_imgs = cf_trajectory.image_observation_range
            img_shape = (exp_imgs.shape[2], exp_imgs.shape[1])

            pre_trajectory_file = os.path.join(args.save_path, f'{cf_id}_pre_old.mp4')
            old_trajectory_file = os.path.join(args.save_path, f'{cf_id}_full_old.mp4')
            new_trajectory_file = os.path.join(args.save_path, f'{cf_id}_full_new.mp4')
            exploration_file = os.path.join(args.save_path, f'{cf_id}_exp.mp4')
            cf_explanation_file = os.path.join(args.save_path, f'{cf_id}_cf.mp4')

            split = state_index - original_trajectory.timestep_range_start

            write_video(original_imgs, old_trajectory_file, img_shape)
            write_video(original_imgs[:split], pre_trajectory_file, img_shape)
            write_video(exp_imgs, exploration_file, img_shape)
            write_video(cf_imgs, cf_explanation_file, img_shape)
            franken_video = np.concatenate((original_imgs[:split], exp_imgs, cf_imgs))
            write_video(franken_video, new_trajectory_file, img_shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-file', type=str, required=True, help='pkl file containing the dataset')
    parser.add_argument('--num-states', type=int, default=10, help='Number of states to select.')
    parser.add_argument('--save-path', type=str, default='videos', help='Place to save states found.')
    parser.add_argument('--state-selection-method', type=str, help='State selection method.',
                        choices=['critical', 'random', 'low_reward'], default='critical')
    parser.add_argument('--timesteps', type=int, default=3, help='Number of timesteps to run the exploration policy.')
    args = parser.parse_args()
    select_states(args)


if __name__ == "__main__":
    main()
