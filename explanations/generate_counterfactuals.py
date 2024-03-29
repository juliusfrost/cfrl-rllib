import argparse
import copy
import json
import os
from collections import namedtuple
from typing import List

import pickle5 as pkl
import cv2
import imageio
import numpy as np
import ray
from ray.tune.registry import _global_registry, ENV_CREATOR, get_trainable_cls

import envs
import models
from explanations.action_selection import RandomAgent, MinigridOracleAgent, make_handoff_func, until_end_handoff
from explanations.create_dataset import create_dataset
from explanations.data import Data
from explanations.rollout import RolloutSaver, rollout_env
from explanations.state_selection import random_state, critical_state, initial_state


def get_env_creator(env_name):
    return _global_registry.get(ENV_CREATOR, env_name)


def window_slice(arr, index: int, window_radius: int):
    window_low = max(0, index - window_radius)
    window_high = min(len(arr), index + window_radius)
    cropped_end = window_high < len(arr)
    return arr[window_low: window_high], cropped_end


def add_border(imgs, border_size=30, border_color=(255, 255, 255)):
    final_images = []
    for img in imgs:
        img_with_border = cv2.copyMakeBorder(
            img,
            top=border_size,
            bottom=border_size,
            left=border_size,
            right=border_size,
            borderType=cv2.BORDER_CONSTANT,
            value=border_color
        )
        final_images.append(img_with_border)
    return final_images


def add_borders(imgs, border_size=30, border_color1=(255, 255, 255), border_color2=(0, 255, 0)):
    switch_index = len(imgs) // 2
    prefix_imgs = add_border(imgs[:switch_index], border_size=border_size, border_color=border_color1)
    post_imgs = add_border(imgs[switch_index:], border_size=border_size, border_color=border_color2)
    prefix_imgs.extend(post_imgs)
    return prefix_imgs


def add_text(images, traj_start, initial_reward, traj_rewards, agent=None, show_timestep=False, show_agent_label=False,
             show_reward=False, **kwargs):
    final_images = []
    agent_name = kwargs.get('overlay_agent_name', 'Driver')
    font = kwargs.get('overlay_font', cv2.FONT_HERSHEY_SIMPLEX)
    org = tuple(kwargs.get('overlay_org', (50, 75)))
    agent_org = tuple(kwargs.get('overlay_agent_org', (50, 150)))
    rew_org = tuple(kwargs.get('overlay_rew_org', (50, 225)))
    font_scale = kwargs.get('overlay_font_scale', 1.8)
    color = tuple(kwargs.get('overlay_color', (255, 0, 0)))
    thickness = kwargs.get('overlay_thickness', 3)
    cum_reward = initial_reward
    for i, (image, reward) in enumerate(zip(images, traj_rewards)):
        cum_reward += reward
        if show_timestep:
            image = cv2.putText(image, f'Timestep: {i + traj_start}', org, font,
                                font_scale, color, thickness, cv2.LINE_AA)
        if show_reward:
            image = cv2.putText(image, f'Score: {cum_reward:.0f}', rew_org, font,
                                font_scale, color, thickness, cv2.LINE_AA)
        if show_agent_label:
            agent = agent if agent is not None else '?'
            image = cv2.putText(image, f'{agent_name}: {agent}', agent_org, font,
                                font_scale, color, thickness, cv2.LINE_AA)
        final_images.append(image)
    return final_images


def format_images(frames, start_timestep=0, trajectory_reward=None, initial_reward=0, border_size=30,
                  border_color=(255, 255, 255), driver=None, show_timestep=False, show_agent=False, show_reward=False,
                  **kwargs):
    final_images = add_text(copy.deepcopy(frames), start_timestep, initial_reward, trajectory_reward, driver,
                            show_timestep,
                            show_agent, show_reward, **kwargs)
    final_images = add_border(final_images, border_size, border_color)
    return final_images


def write_video(frames, filename, image_shape, fps=5, show_start=True, show_stop=True,
                show_blank_frames=False, downscale=2, crashed=False, **kwargs):
    w, h = image_shape
    font_scale = kwargs.get('end_font_scale', 4)
    color = tuple(kwargs.get('end_color', (255, 255, 255)))
    thickness = kwargs.get('end_thickness', 8)
    font = kwargs.get('end_font', cv2.FONT_HERSHEY_SIMPLEX)
    crashed_text = kwargs.get('crashed_text', 'CRASHED')
    done_text = kwargs.get('done_text', 'DONE')
    end_text_org = tuple(kwargs.get('end_text_org', (int(w / 2) - 41 * len(done_text), int(h / 2))))
    end_length = kwargs.get('end_length', 2)
    if show_blank_frames:
        blank_frames = np.zeros((fps, h, w, 3))
        frames = np.concatenate([frames, blank_frames]).astype(np.uint8)
    if show_stop:
        try:
            final_frame = frames[-1]
        except:
            print("uh oh")
        if crashed:
            text = crashed_text
        else:
            text = done_text
        final_frame = cv2.putText(final_frame, text, end_text_org, font, font_scale, color, thickness, cv2.LINE_AA)
        frames = np.concatenate([frames, [final_frame] * fps * end_length]).astype(np.uint8)
    if show_start:
        # Approximately center at the middle of the image
        # The -200 offset is so the text is about centered horizontally
        start_frame = np.zeros((h, w, 3)).astype(np.uint8) + 100
        bottom_left = (int(w / 2) - 310, int(h / 2))
        start_frame = cv2.putText(start_frame, 'Restarting', bottom_left, font, font_scale, color, thickness,
                                  cv2.LINE_AA)
        frames = np.concatenate([frames, [start_frame] * fps]).astype(np.uint8)
    w = w // downscale
    h = h // downscale
    frames = [cv2.resize(img, dsize=(w, h), interpolation=cv2.INTER_CUBIC) for img in frames]
    imageio.mimwrite(filename, frames, fps=fps)


DatasetArgs = namedtuple("DatasetArgs", ["out", "env", "run", "checkpoint"])


def load_other_policies(other_policies: List[dict]):  # TODO: include original policy here too!
    policies = []
    for policy in other_policies:
        assert isinstance(policy, dict)
        name, run_type, checkpoint = policy['name'], policy['run'], policy['checkpoint']
        # Load configuration from checkpoint file.
        config_dir = os.path.dirname(checkpoint)
        config_path = os.path.join(config_dir, "params.pkl")
        # Try parent directory.
        if not os.path.exists(config_path):
            config_path = os.path.join(config_dir, "../params.pkl")

        # If no pkl file found, require command line `--config`.
        if not os.path.exists(config_path):
            raise ValueError(
                "Could not find params.pkl in either the checkpoint dir or "
                "its parent directory AND no config given on command line!")

        # Load the config from pickled.
        else:
            with open(config_path, "rb") as f:
                config = pkl.load(f)
        # force num_workers 0
        # TODO: consolidate loading policies into a single method
        config['num_workers'] = 0

        # Create the Trainer from config.
        cls = get_trainable_cls(run_type)
        agent = cls(env=config["env"], config=config)
        # Load state from checkpoint.
        agent.restore(checkpoint)
        policies.append((agent, run_type, name))

    return policies


def generate_videos_cf(cf_dataset, cf_name, reward_so_far, start_timestep, args, cf_id, save_id, split, prefix_video,
                       driver, **kwargs):
    # Generate continuation video
    cf_trajectory = cf_dataset.get_trajectory(cf_id)
    cf_rewards = cf_trajectory.reward_range
    cf_imgs = format_images(cf_trajectory.image_observation_range,
                            start_timestep=start_timestep,
                            trajectory_reward=cf_rewards,
                            initial_reward=reward_so_far,
                            border_color=[0, 255, 0],
                            border_size=args.border_width,
                            driver=driver,
                            **kwargs)
    # Get failure, if it's recorded
    last_env_info = cf_trajectory.env_info_range[-1]
    crashed = last_env_info.get('failure', 0)

    img_shape = (cf_imgs[0].shape[1], cf_imgs[0].shape[0])

    # We will generate 3 videos using this continuation:
    #   (1) Continuation video alone
    #   (2) beginning video + continuation
    #   (3) Shorter version of (2) centered around the selected state.
    # f'{video_type}-{cf_name}-t_{save_id}.{args.video_format}'
    vidpath = lambda vid_type, cf_name, save_id: f'{vid_type}-{cf_name}-t_{save_id}.{args.video_format}'
    if len(prefix_video) > 0 and args.show_exploration:
        cf_video = np.concatenate((prefix_video, cf_imgs))
    else:
        cf_video = cf_imgs
        split = 0
    cf_window_video, cropped_end = window_slice(cf_video, split, args.window_len)
    window_crashed = crashed and not cropped_end
    show_start = args.video_format == 'gif'

    if args.save_all:
        # Writing video 1 == continuation video alone
        cf_explanation_file = os.path.join(args.save_path, vidpath('continuation', cf_name, save_id))
        write_video(cf_imgs, cf_explanation_file, img_shape, args.fps, show_start=show_start, show_stop=True,
                    downscale=args.downscale, crashed=crashed, **kwargs)
    if args.save_all:
        # Writing video 2 == beginning + continuation
        new_trajectory_file = os.path.join(args.save_path, vidpath('counterfactual_vid', cf_name, save_id))
        write_video(cf_video, new_trajectory_file, img_shape, args.fps, show_start=show_start, show_stop=True,
                    downscale=args.downscale, crashed=crashed, **kwargs)
    if args.save_all or not args.side_by_side:
        # Writing video 3 == shorter version of 2
        cf_window_explanation_file = os.path.join(args.save_path,
                                                  vidpath('counterfactual_window', cf_name, save_id))
        write_video(cf_window_video, cf_window_explanation_file, img_shape, args.fps, show_start=show_start,
                    show_stop=True, downscale=args.downscale, crashed=window_crashed, **kwargs)

    return (cf_imgs, crashed), (cf_video, crashed), (cf_window_video, window_crashed)


def save_joint_video(video_list, video_names, base_video_name, id, args, **kwargs):
    h, w, c = video_list[0][0][0].shape
    font_scale = kwargs.get('end_font_scale', 4)
    color = tuple(kwargs.get('end_color', (255, 255, 255)))
    thickness = kwargs.get('end_thickness', 8)
    font = kwargs.get('end_font', cv2.FONT_HERSHEY_SIMPLEX)
    crashed_text = kwargs.get('crashed_text', 'CRASHED')
    done_text = kwargs.get('done_text', 'DONE')
    cf_net_rewards = kwargs.get('cf_net_rewards', None)
    end_length = kwargs.get('end_length', 2)
    order = np.random.permutation(len(video_list))
    padded_videos = []
    max_len = max([len(v[0]) for v in video_list]) + args.fps * end_length
    for i in order.tolist():
        curr_vid, crashed = video_list[i]
        final_frame = curr_vid[-1]
        if crashed:
            text = crashed_text
        else:
            text = done_text
        bottom_left = (int(w / 2) - 41 * len(text), int(h / 2))
        final_frame = cv2.putText(final_frame, text, bottom_left, font, font_scale, color, thickness, cv2.LINE_AA)
        padded_video = np.concatenate([curr_vid, [final_frame] * (max_len - len(curr_vid))])
        padded_videos.append(padded_video.astype(np.uint8))
    combined_video = np.concatenate(padded_videos, axis=2)

    t, h, w, c = combined_video.shape
    save_file = os.path.join(args.save_path, f"{base_video_name}-t_{id}.{args.video_format}")
    answer_key_file = os.path.join(args.save_path, f"{base_video_name}-answer_key.txt")
    reward_answer_key_file = os.path.join(args.save_path, f"{base_video_name}-reward_answer_key.txt")
    show_start = args.video_format == 'gif'
    write_video(combined_video, save_file, (w, h), args.fps, show_start=show_start, show_stop=False,
                downscale=args.downscale, **kwargs)
    with open(answer_key_file, 'a') as f:
        f.write(f"{id},")
        for i in order.tolist():
            f.write(f"{video_names[i]},")
        f.write("\n")
    if cf_net_rewards is not None:
        with open(reward_answer_key_file, 'a') as f:
            f.write(f"{id},")
            for i in order.tolist():
                f.write(f"{video_names[i]},")
            f.write("\n")


def save_separate_videos(video_list, video_names, base_video_name, id, args, window_len, is_context=False, **kwargs):
    h, w, c = video_list[0][0][0].shape
    cf_net_rewards = kwargs.get('cf_net_rewards', None)
    if not is_context:
        videos_path = os.path.join(args.save_path, f"vids_{id}")
        os.mkdir(videos_path)
    else:
        videos_path = args.save_path
    if args.randomize_videos:
        order = np.random.permutation(len(video_list)).tolist()
    else:
        order = np.arange(len(video_list))
    for j, i in enumerate(order):
        curr_vid, crashed = video_list[i]
        if len(curr_vid) > window_len:
            if is_context:
                curr_vid = curr_vid[-window_len:]
            else:
                curr_vid = curr_vid[:window_len]
        save_file = os.path.join(videos_path, f"{base_video_name}-t_{id}_{j}.{args.video_format}")
        show_start = args.video_format == 'gif'
        write_video(curr_vid, save_file, (w, h), args.fps, show_start=show_start, show_stop=False,
                    downscale=args.downscale, **kwargs)

    if not is_context:
        answer_key_file = os.path.join(args.save_path, f"{base_video_name}-answer_key.txt")
        reward_answer_key_file = os.path.join(args.save_path, f"{base_video_name}-reward_answer_key.txt")

        with open(answer_key_file, 'a') as f:
            f.write(f"{id},")
            for i in order.tolist():
                f.write(f"{video_names[i]},")
            f.write("\n")
        if cf_net_rewards is not None:
            with open(reward_answer_key_file, 'a') as f:
                f.write(f"{id},")
                for i in order.tolist():
                    f.write(f"{cf_net_rewards[i]},")
                f.write("\n")


def generate_videos_counterfactual_method(original_dataset, exploration_dataset, cf_datasets, cf_to_exp_index, args,
                                          cf_names, state_indices, **kwargs):
    # Loop through the cf ids
    for cf_i in range(len(cf_datasets[0].all_trajectory_ids)):
        cf_id = cf_datasets[0].all_trajectory_ids[cf_i]
        i = cf_to_exp_index[cf_id]

        # We will generate 5 types of videos:
        #  (1) Video of the beginning (up to the critical state)
        #  (2) Exploration videos
        #  (3) Beginning + exploration
        #  (4) Baseline (window from the original trajectory, centered around the critical state)
        #  (5) A bunch of videos involving different continuation trajectories (see the function above)

        # (1) Create images for beginning of the trajectory
        state_index = state_indices[i]
        pre_timestep = original_dataset.get_timestep(state_index)

        original_trajectory = original_dataset.get_trajectory(pre_timestep.trajectory_id)
        # if args.show_exploration:
        split = state_index - original_trajectory.timestep_range_start + 1
        # else:
        #     split = state_index + 1
        original_rewards = original_trajectory.reward_range
        original_imgs = format_images(original_trajectory.image_observation_range,
                                      start_timestep=0,
                                      trajectory_reward=original_rewards,
                                      initial_reward=0,
                                      border_color=[255, 255, 255],
                                      border_size=args.border_width,
                                      driver='A',
                                      **kwargs)

        #  (2) Create images of exploration
        if exploration_dataset is None:
            has_explored = False
        else:
            has_explored = len(exploration_dataset.all_trajectory_ids) != 0
        if has_explored:
            exp_id = exploration_dataset.all_trajectory_ids[i]
            exp_trajectory = exploration_dataset.get_trajectory(exp_id)
            exp_rewards = exp_trajectory.reward_range
            exp_imgs = format_images(exp_trajectory.image_observation_range,
                                     start_timestep=split,
                                     trajectory_reward=exp_rewards,
                                     initial_reward=original_rewards[:split].sum(),
                                     border_color=[255, 0, 255],
                                     border_size=args.border_width,
                                     driver='B',
                                     **kwargs)

            #  (5) Create videos with the continuations
            initial_rewards = original_rewards[:split].sum() + exp_rewards.sum()
            start_timestep = split + len(exp_imgs)
            prefix_video = np.concatenate((original_imgs[:split], exp_imgs)) if split > 0 else exp_imgs
        else:
            initial_rewards = original_rewards[:split].sum()
            start_timestep = split
            prefix_video = original_imgs[:split]

        continuation_list = []
        cf_list = []
        window_list = []
        if args.side_by_side or args.save_separate:
            cf_driver = '?'
        else:
            cf_driver = 'A'
        cf_net_rewards = []
        for cf_dataset, cf_name in zip(cf_datasets, cf_names):
            continuation, cf, window = generate_videos_cf(cf_dataset, cf_name, initial_rewards, start_timestep, args,
                                                          cf_id, cf_id, split, prefix_video, cf_driver,
                                                          **kwargs)
            continuation_list.append(continuation)
            cf_trajectory = cf_dataset.get_trajectory(cf_id)
            cf_rewards = cf_trajectory.reward_range
            reward = np.sum(cf_rewards)
            cf_net_rewards.append(reward)
            cf_list.append(cf)
            window_list.append(window)
        if args.side_by_side:
            if args.save_all:
                continuation_file = 'continuation'
                cf_file = 'counterfactual_vid'
                save_joint_video(continuation_list, cf_names, continuation_file, i, args, **kwargs)
                save_joint_video(cf_list, cf_names, cf_file, i, args, **kwargs)
            cf_window_file = 'counterfactual_window'
            save_joint_video(window_list, cf_names, cf_window_file, cf_id, args, **kwargs)
        elif args.save_separate:
            context_file = "context_vid"
            counterfactual_file = "counterfactual_vid"
            save_separate_videos([[prefix_video, False]], ["A"], context_file, cf_id, args, is_context=True,
                                 window_len=args.window_len, **kwargs)
            save_separate_videos(continuation_list, cf_names, counterfactual_file, cf_id, args, is_context=False,
                                 cf_net_rewards=cf_net_rewards, window_len=args.window_len, **kwargs)

        # We've already generated the images; now we store them as a video
        img_shape = (original_imgs[0].shape[1], original_imgs[0].shape[0])
        show_start = args.video_format == 'gif'

        if args.save_all:
            #  (1) Beginning video
            old_trajectory_file = os.path.join(args.save_path, f'original-t_{i}.{args.video_format}')
            write_video(original_imgs, old_trajectory_file, img_shape, args.fps, show_start=show_start, show_stop=True,
                        downscale=args.downscale, crashed=False, **kwargs)

            if has_explored:
                #  (2) Exploration video
                exploration_file = os.path.join(args.save_path, f'exploration-t_{cf_id}.{args.video_format}')
                write_video(exp_imgs, exploration_file, img_shape, args.fps, show_start=show_start, show_stop=True,
                            downscale=args.downscale, crashed=False, **kwargs)

            #  (3) Beginning + Exploration
            pre_trajectory_file = os.path.join(args.save_path, f'prefix-t_{cf_id}.{args.video_format}')
            write_video(prefix_video, pre_trajectory_file, img_shape, args.fps, show_start=show_start, show_stop=True,
                        downscale=args.downscale, crashed=False, **kwargs)

        #  (4) Baseline (Critical-state-centered window)
        baseline_window_explanation_file = os.path.join(args.save_path,
                                                        f'baseline_window-trial_{i}.{args.video_format}')
        baseline_window_video, cropped_end = window_slice(original_imgs, split, args.window_len)
        # Get failure, if it's recorded
        last_env_info = original_trajectory.env_info_range[-1]
        crashed = last_env_info.get('failure', 0) and not cropped_end
        img_shape = (baseline_window_video[0].shape[1], baseline_window_video[0].shape[0])
        write_video(baseline_window_video, baseline_window_explanation_file, img_shape, args.fps,
                    show_start=show_start, show_stop=True, downscale=args.downscale, crashed=crashed, **kwargs)


def generate_videos_state_method(original_dataset, args, state_indices, **kwargs):
    for i, state_index in enumerate(state_indices):

        # We will generate 2 types of videos:
        #  (1) Video of the beginning (up to the critical state)
        #  (2) Baseline (window from the original trajectory, centered around the critical state)

        # (1) Create images for beginning of the trajectory
        pre_timestep = original_dataset.get_timestep(state_index)

        original_trajectory = original_dataset.get_trajectory(pre_timestep.trajectory_id)
        split = state_index - original_trajectory.timestep_range_start
        original_rewards = original_trajectory.reward_range
        original_imgs = format_images(original_trajectory.image_observation_range,
                                      start_timestep=0,
                                      trajectory_reward=original_rewards,
                                      initial_reward=0,
                                      border_color=[255, 255, 255],
                                      border_size=0,
                                      driver='A',
                                      **kwargs)

        # We've already generated the images; now we store them as a video
        img_shape = (original_imgs[0].shape[1], original_imgs[0].shape[0])

        if args.save_all:
            #  (1) Beginning video
            old_trajectory_file = os.path.join(args.save_path, f'original-t_{i}.{args.video_format}')
            write_video(original_imgs, old_trajectory_file, img_shape, args.fps,
                        downscale=args.downscale, crashed=False, **kwargs)

        #  (2) Baseline (Critical-state-centered window)
        baseline_window_explanation_file = os.path.join(args.save_path,
                                                        f'baseline_window-trial_{i}.{args.video_format}')
        baseline_window_video, cropped_end = window_slice(original_imgs, split, args.window_len)
        baseline_window_video = add_borders(baseline_window_video, border_size=args.border_width)
        img_shape = (baseline_window_video[0].shape[1], baseline_window_video[0].shape[0])
        last_env_info = original_trajectory.env_info_range[-1]
        crashed = last_env_info.get('failure', 0) and not cropped_end
        show_start = args.video_format == 'gif'
        write_video(baseline_window_video, baseline_window_explanation_file, img_shape, args.fps,
                    downscale=args.downscale, crashed=crashed, show_start=show_start, **kwargs)


def select_states(args):
    with open(args.dataset_file, "rb") as f:
        dataset: Data = pkl.load(f)
    agent = dataset.policy.get_agent()
    policy = agent.get_policy()
    state_selection_dict = {
        "critical": critical_state,
        "random": random_state,
        "initial": initial_state,
    }
    if args.explanation_method == "counterfactual":
        alternative_agents = load_other_policies(args.eval_policies)
        # Add the original policy in too
        if args.evaluate_behavior_policy or len(alternative_agents) == 0:
            alternative_agents.append((agent, args.run, args.policy_name))
        state_selection_fn = state_selection_dict[args.state_selection_method]
        state_indices = state_selection_fn(dataset, args.num_states + args.num_buffer_states, policy)
        # Add
        exploration_rollout_saver = RolloutSaver(
            outfile=args.save_path + "/exploration.pkl",
            target_steps=None,
            target_episodes=args.num_states,
            save_info=True)
        # Neither of these seem to be used, just required to save dataset
        exploration_args = DatasetArgs(out=args.save_path + "/exploration.pkl", env="DrivingPLE-v0", run="PPO",
                                       checkpoint=1)
        exploration_policy_config = {}
        test_rollout_savers = []
        for agent, run_type, name in alternative_agents:
            counterfactual_rollout_saver = RolloutSaver(
                outfile=args.save_path + f"/{name}_counterfactual.pkl",
                target_steps=None,
                target_episodes=args.num_states,
                save_info=True)
            # Neither of these seem to be used, just required to save dataset
            counterfactual_args = DatasetArgs(out=args.save_path + f"/{name}_counterfactual.pkl", env="DrivingPLE-v0",
                                              run=run_type, checkpoint=1)
            test_rollout_savers.append((counterfactual_rollout_saver, counterfactual_args))
        counterfactual_policy_config = {}

        env_creator = get_env_creator(args.env)
        env = env_creator(args.env_config)
        env.reset()
        cf_to_exp_index = {}
        cf_count = 0
        if args.save_path is not None:
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            successful_trajs = 0
            exp_index = 0
            while successful_trajs < args.num_states and (exp_index < len(state_indices)):
                branching_state = state_indices[exp_index]
                timestep = dataset.get_timestep(branching_state)
                simulator_state = timestep.simulator_state
                obs = timestep.observation
                env.load_simulator_state(simulator_state)
                if args.exploration_method == 'random':
                    exploration_agent = RandomAgent(env.action_space)
                elif args.exploration_method == 'minigrid_oracle':
                    exploration_agent = MinigridOracleAgent(env)
                else:
                    config_dir = os.path.dirname(args.exploration_policy['checkpoint'])
                    config_path = os.path.join(config_dir, "../params.pkl")
                    with open(config_path, "rb") as f:
                        config = pkl.load(f)
                    # force num_workers 0
                    # TODO: consolidate loading policies into a single method
                    config['num_workers'] = 0
                    cls = get_trainable_cls(args.exploration_policy['run'])
                    exploration_agent = cls(env=config["env"], config=config)
                    exploration_agent.restore(args.exploration_policy['checkpoint'])

                handoff_func = make_handoff_func(args.timesteps)

                with exploration_rollout_saver as saver:
                    exp_env, env_obs, env_done = rollout_env(exploration_agent, env, handoff_func, obs, saver=saver,
                                                             no_render=False)

                post_explore_state = exp_env.get_simulator_state()

                if not env_done:
                    cf_to_exp_index[cf_count] = exp_index
                    cf_count += 1
                    for agent_stuff, saver_stuff in zip(alternative_agents, test_rollout_savers):
                        exp_env.load_simulator_state(post_explore_state)
                        agent, run_type, name = agent_stuff
                        counterfactual_saver, counterfactual_args = saver_stuff
                        with counterfactual_saver as saver:
                            rollout_env(agent, exp_env, until_end_handoff, env_obs, saver=saver, no_render=False,
                                        save_start_img=True if args.save_separate else False)
                    successful_trajs += 1
                exp_index += 1
            if not successful_trajs == args.num_states:
                print("WARNING: We were not able to generate the requested number of trajectories. "
                      "Consider raising num_buffer_states.")

            exploration_dataset = create_dataset(exploration_args, exploration_policy_config, write_data=args.save_all)
            cf_datasets = []
            for counterfactual_saver, counterfactual_args in test_rollout_savers:
                counterfactual_dataset = create_dataset(counterfactual_args, counterfactual_policy_config,
                                                        write_data=args.save_all)
                cf_datasets.append(counterfactual_dataset)

            cf_names = [agent_stuff[2] for agent_stuff in alternative_agents]
            generate_videos_counterfactual_method(dataset, exploration_dataset, cf_datasets, cf_to_exp_index, args,
                                                  cf_names, state_indices, **args.settings_config, **args.show_text)
    else:
        if args.save_path is not None:
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
        state_selection_fn = state_selection_dict[args.explanation_method]
        state_indices = state_selection_fn(dataset, args.num_states, policy)
        generate_videos_state_method(dataset, args, state_indices, **args.settings_config, **args.show_text)


def generate_with_selected_states(args):
    with open(args.dataset_file, "rb") as f:
        dataset: Data = pkl.load(f)
    eval_policies = args.eval_policies
    if args.behavioral_policy is not None:
        eval_policies.append({'name': args.policy_name, 'run': args.run, 'checkpoint': args.behavioral_policy})
    alternative_agents = load_other_policies(args.eval_policies)
    # Add the original policy in too
    test_rollout_savers = []
    for agent, run_type, name in alternative_agents:
        counterfactual_rollout_saver = RolloutSaver(
            outfile=args.save_path + f"/{name}_counterfactual.pkl",
            target_steps=None,
            target_episodes=args.num_states,
            save_info=True)
        # Neither of these seem to be used, just required to save dataset
        counterfactual_args = DatasetArgs(out=args.save_path + f"/{name}_counterfactual.pkl", env="DrivingPLE-v0",
                                          run=run_type, checkpoint=1)
        test_rollout_savers.append((counterfactual_rollout_saver, counterfactual_args))
    counterfactual_policy_config = {}
    state_indices = [i for i in range(len(dataset.all_simulator_states)) if dataset.all_simulator_states[i] is not None]
    if not len(state_indices) >= args.num_states:
        print(f"Couldn't generate the requested {args.num_states} state since we only have {len(state_indices)} "
              "pre-selected states.")
    env_creator = get_env_creator(args.env)
    env = env_creator(args.env_config)
    env.reset()
    num_states = min(len(state_indices), args.num_states)
    cf_to_exp_index = {i: i for i in range(num_states)}
    assert args.save_path is not None
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    successful_trajs = 0
    for i in range(num_states):
        state_id = state_indices[i]
        selected_state = dataset.all_simulator_states[state_id]
        env_obs = dataset.all_observations[state_id]
        for agent_stuff, saver_stuff in zip(alternative_agents, test_rollout_savers):
            env.load_simulator_state(selected_state)
            agent, run_type, name = agent_stuff
            counterfactual_saver, counterfactual_args = saver_stuff
            with counterfactual_saver as saver:
                handoff_func = make_handoff_func(args.num_eval_steps)
                rollout_env(agent, env, handoff_func, env_obs, saver=saver, no_render=False, save_start_img=True)
            successful_trajs += 1
    cf_datasets = []
    for counterfactual_saver, counterfactual_args in test_rollout_savers:
        counterfactual_dataset = create_dataset(counterfactual_args, counterfactual_policy_config,
                                                write_data=args.save_all)
        cf_datasets.append(counterfactual_dataset)
    if len(args.alt_file_names) > 0:
        for file in args.alt_file_names:
            with open(file, "rb") as f:
                loaded_dataset = pkl.load(f)
            cf_datasets.append(loaded_dataset)
    exploration_dataset = None
    cf_names = [agent_stuff[2] for agent_stuff in alternative_agents] + [f[:-4] for f in args.alt_file_names]
    generate_videos_counterfactual_method(dataset, exploration_dataset, cf_datasets, cf_to_exp_index, args,
                                          cf_names, state_indices, **args.settings_config)


def nullable_string(val):
    if not val:
        return None
    if val == 'None':
        return None
    return val


def main(parser_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-file', type=str, required=True, help='pkl file containing the dataset')
    parser.add_argument('--env', type=str, required=True, help='name of the environment')
    parser.add_argument('--num-states', type=int, default=10, help='Number of states to select.')
    parser.add_argument('--save-path', type=str, default='videos', help='Place to save states found.')
    parser.add_argument('--window-len', type=int, default=20, help='config')
    parser.add_argument('--state-selection-method', type=str, help='State selection method.',
                        choices=['critical', 'random', 'initial', 'manual'], default='critical')
    parser.add_argument('--explanation-method', type=str, help='Explanation Method',
                        choices=['counterfactual', 'critical', 'random'], default='counterfactual')
    parser.add_argument('--eval-policies', type=json.loads, default=None,
                        help='list of evaluation policies to continue rollouts. '
                             'Policies are a tuple of (name, algorithm, checkpoint)')
    parser.add_argument('--evaluate-behavior-policy', action='store_true',
                        help='include the behavior policy in the evaluation videos')
    parser.add_argument('--timesteps', type=int, default=3, help='Number of timesteps to run the exploration policy.')
    parser.add_argument('--fps', type=int, default=5)
    parser.add_argument('--border-width', type=int, default=30)
    parser.add_argument('--settings-config', type=json.loads, default='{}', help='text configuration')
    parser.add_argument('--downscale', type=int, default=2)
    # TODO: make way env-config and alt-policy-config are handled identitcal.
    parser.add_argument('--env-config', type=json.loads, default="{}",
                        help='environment configuration')
    parser.add_argument('--policy-name', type=str, default="test_policy_name")
    parser.add_argument('--run', type=str, default="PPO")
    parser.add_argument('--behavioral-policy', type=nullable_string, default=None)
    parser.add_argument('--side-by-side', default=False, action='store_true')
    parser.add_argument('--save-separate', default=False, action='store_true')
    parser.add_argument('--save-all', action='store_true',
                        help='Save all possible combinations of videos. '
                             'Note that this will take up a lot of space!')
    parser.add_argument('--num-buffer-states', type=int, default=10, help='Number of buffer states to select.')
    parser.add_argument('--video-format', type=str, help='Video file format', choices=['mp4', 'gif'], default='gif')
    parser.add_argument('--exploration-method', type=str, help='Type of policy to use for exploration',
                        choices=['random', 'policy', 'minigrid_oracle'], default='random')
    parser.add_argument('--exploration-policy', type=json.loads, help='Checkpoint and run point for exploration policy',
                        default=None)
    parser.add_argument('--alt-file-names', type=json.loads, help='Optional filename for alt policies',
                        default=[])
    parser.add_argument('--num-eval-steps', type=int, default=20)
    parser.add_argument('--show-text', type=json.loads, default='{}')
    parser.add_argument('--show-exploration', action='store_true')
    parser.add_argument('--randomize-videos', action='store_true')
    args = parser.parse_args(parser_args)

    ray.init()
    # register environments
    envs.register()
    # register models
    models.register()
    if args.state_selection_method == 'manual':
        generate_with_selected_states(args)
    else:
        select_states(args)
    ray.shutdown()


if __name__ == "__main__":
    main()
