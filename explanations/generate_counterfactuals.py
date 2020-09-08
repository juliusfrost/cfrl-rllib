import argparse
import os
import pickle as pkl

import cv2

from envs import register
from explanations.data import Data
from explanations.state_selection import random_state, critical_state, low_reward_state
from explanations.rollout import RolloutSaver, rollout_env
from explanations.action_selection import RandomAgent, make_handoff_func

register()
from envs.driving import register as registerD
registerD()

from envs.driving import env_creator

def write_video(frames, filename, image_shape, fps=5):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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

    exploration_rollout_saver = RolloutSaver(
                 outfile=args.save_path + "exploration",
                 target_steps=None,
                 target_episodes=args.num_states,
                 save_info=True)
    counterfactual_rollout_saver = RolloutSaver(
        outfile=args.save_path + "counterfactual",
        target_steps=None,
        target_episodes=args.num_states,
        save_info=True)

    env = env_creator()
    env.reset()

    if args.save_path is not None:
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        for i in state_indices:
            timestep = dataset.get_timestep(i)
            simulator_state = timestep.simulator_state
            obs = timestep.observation
            env.game_state.game.setGameState(*simulator_state)
            random_agent = RandomAgent(env.action_space)
            handoff_func = make_handoff_func(args.timesteps)

            with exploration_rollout_saver as saver:
                rollout_env(random_agent, env, handoff_func, obs, saver=saver, no_render=False)

            trajectory = dataset.get_trajectory(timestep.trajectory_id)
            split = i - trajectory.timestep_range_start
            imgs = trajectory.image_observation_range

            img_shape = (imgs.shape[2], imgs.shape[1])
            context_file = os.path.join(args.save_path, f'{trajectory.trajectory_id}_{i}_context.mp4')
            explanation_file = os.path.join(args.save_path, f'{trajectory.trajectory_id}_{i}_explanation.mp4')
            full_file = os.path.join(args.save_path, f'{trajectory.trajectory_id}_{i}_full.mp4')

            write_video(imgs[:split], context_file, img_shape)
            write_video(imgs[split:], explanation_file, img_shape)
            write_video(imgs, full_file, img_shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-file', type=str, required=True, help='pkl file containing the dataset')
    parser.add_argument('--num-states', type=int, default=10, help='Number of states to select.')
    parser.add_argument('--save-path', type=str, default='videos', help='Place to save states found.')
    parser.add_argument('--state-selection-method', type=str, help='State selection method.',
                        choices=['critical', 'random', 'low_reward'], default='critical')
    parser.add_argument('--timesteps', type=int, default=3, help='Number of timesteps to run the exploration policy.')
    args = parser.parse_args()
    select_states(args)
