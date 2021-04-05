import os
import sys
import copy
import time
import argparse
import numpy as np
import pathlib
import pickle as pkl
import ray
from ray.tune.registry import get_trainable_cls
import torch
import envs
from explanations.data import Data, PolicyInfo
import matplotlib.pyplot as plt

sys.path.append('.')
ray.init()
envs.register()
parser = argparse.ArgumentParser()
parser.add_argument('--policy', default=None, help='policy to run')
parser.add_argument('--load_path', default=None, help='Load env from this path')
parser.add_argument('--load_name', default='', help='Which env to load within load_path')
parser.add_argument('--save_path', default='.', help='if we save the trajectory, store it here.')
parser.add_argument('--switch_prob', type=int, default=.005, help='car switch probability')  # TODO: add full config
parser.add_argument('--prob_car', type=int, default=.2, help='car appearance probability')  # TODO: add full config
parser.add_argument('--algo', default='PPO', help='algorithm to run')  # TODO: add full config
parser.add_argument('--env', default='MiniGrid', help='algorithm to run')
parser.add_argument('--agent_pos', default='2 2', help='agent start pos')  # Train is 2 2, test is 6 2
parser.add_argument('--goal_pos', default='2 6', help='goal pos')  # Train is 2 6, goal is 2 6
parser.add_argument('--load_start_states', default=None, help='load a dataset of start states')
args = parser.parse_args()


def load_start_state(index, env):
    with open(args.load_start_states, 'rb') as f:
        dataset = pkl.load(f)
    traj_ids = dataset.all_trajectory_ids
    if index >= len(traj_ids):
        print("ALL DONE WITH START STATES!")
        exit(0)
    traj_id = traj_ids[index]
    traj = dataset.get_trajectory(traj_id)
    simulator_state = traj.simulator_state_range[-1]
    env.load_simulator_state(simulator_state)

# Load Policy
if args.policy is not None:
    config_dir = os.path.dirname(args.policy)
    config_path = os.path.join(config_dir, "params.pkl")
    # Try parent directory.
    if not os.path.exists(config_path):
        config_path = os.path.join(config_dir, "../params.pkl")
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
if args.env == 'Driving':
    env = envs.driving.driving_creator(switch_prob=args.switch_prob, prob_car=args.prob_car, time_limit=float('inf'))
elif args.env.lower() == 'minigrid':
    agent_token = [int(x) for x in args.agent_pos if not x == ' ']
    assert len(agent_token) == 2, ("wrong format for agent_pos", agent_token)
    goal_token = [int(x) for x in args.goal_pos if not x == ' ']
    assert len(goal_token) == 2, ("wrong format for goal_pos", goal_token)
    env = envs.minigrid.env_creator(agent_pos=np.array(agent_token), goal_pos=np.array(goal_token))
load_state_index = None
if args.load_start_states is not None:
    load_start_state(0, env)
    load_state_index = 1
elif args.load_path is not None:
    load_name = args.load_name
    with open(pathlib.Path(args.load_path).joinpath(f'env_state_{load_name}.pkl'), 'rb') as f:
        env_state = pkl.load(f)
        env.load_simulator_state(env_state)
    with open(pathlib.Path(args.load_path).joinpath(f'obs_{load_name}.pkl'), 'rb') as f:
        obs = pkl.load(f)
else:
    if args.load_start_states is not None:
        load_start_state(load_state_index, env)
        load_state_index += 1
    else:
        obs = env.reset()
if args.env == 'Driving':
    env.render()
else:
    plt.imshow(env.render('rgb_array'))
    plt.show()

all_time_steps = []
trajectories = []
observations = []
image_observations = []
simulator_states = []
env_infos = []
rewards = []

save_index = 0

while not done:
    img_list.append(copy.deepcopy(env.render(mode='rgb_array')))

    window_len = 20  # TODO: make this a flag!  Also make a config that it can take in.
    if len(img_list) > window_len:
        img_list = img_list[-window_len:]
    action = [0, 0]
    a = input()
    # action[0] = +left -right, action[1] = +forward, -backward
    # if a ==
    if args.env == 'Driving':
        if 'a' in a:
            action[0] += 1
        if 'd' in a:
            action[0] -= 1
        if 'w' in a:
            action[1] += 1
        if 's' in a:
            action[1] -= 1
    elif args.env.lower() == 'minigrid':  # wasd navigation
        if 'a' in a:
            action = 0
        elif 'd' in a:
            action = 1
        elif 'w' in a:
            action = 2
    if 'v' in a:
        saved_state = env.get_simulator_state()
    if 'l' in a:
        env.load_simulator_state(saved_state)
    if 'r' in a:
        if args.load_start_states is not None:
            load_start_state(load_state_index, env)
            load_state_index += 1
        else:
            obs = env.reset()
        action = None
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
        if not save_path.suffix == '.pkl':
            save_path = save_path.parent / (save_path.name + '.pkl')
        # Store the window leading up to the save state as a trajectory in a dataset
        # Most of the fields will be None b/c we don't use them and may as well save space.
        none_list = []
        for timestep in range(len(img_list)):
            none_list.append(None)
            all_time_steps.append(timestep)
            trajectories.append(save_index)
            observation = None if timestep < len(img_list) - 1 else obs
            observations.append(observation)
            image_observations.append(copy.deepcopy(img_list[timestep]))
            rewards.append(0)
            simulator_state = None if timestep < len(img_list) - 1 else env.get_simulator_state()
            simulator_states.append(simulator_state)
            env_infos.append({})

        dataset = Data(
            all_time_steps=all_time_steps,
            all_trajectories=trajectories,
            all_observations=observations,
            all_image_observations=image_observations,
            all_actions=copy.deepcopy(none_list),
            all_rewards=rewards,
            all_dones=copy.deepcopy(none_list),
            all_env_infos=env_infos,
            policy=None,
            all_simulator_states=simulator_states
        )
        with open(save_path, "wb") as f:
            pkl.dump(dataset, f)
        save_index += 1
        print("SAVED")
        action = None

    print(action)
    if action is not None:
        print("stepping", action)
        obs, _, done, _ = env.step(action)
    if args.env == 'Driving':
        env.render()
    else:
        plt.imshow(env.render('rgb_array'))
        plt.show()
