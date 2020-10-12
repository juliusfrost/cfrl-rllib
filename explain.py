import argparse
import json
import os
import pickle
import shutil
import pathlib

import yaml
from ray.tune.utils import merge_dicts

from explanations.create_dataset import main as create_dataset_main
from explanations.generate_counterfactuals import main as generate_counterfactuals_main
from explanations.generate_forms import main as generate_forms_main

DEFAULT_CONFIG = {
    # REQUIRED
    # config for the behavior policy
    'behavior_policy_config': {
        # path to the rllib policy checkpoint file
        'checkpoint': "/home/olivia/ray_results/driving-ppo/PPO_DrivingPLE-v0_7c851_00000_0_2020-09-28_11-43-22/checkpoint_2/checkpoint-2",  # TODO: back to None
        # name of algorithm used to train the behavior policy
        'run': "PPO",
        'identifier': "policyA",
    },
    # REQUIRED
    # train environment name
    'env': "DrivingPLE-v0",
    # train environment configuration
    'env_config': {},

    # REQUIRED
    # test environment name
    'eval_env': "DrivingPLE-v0",
    # test environment configuration
    'eval_env_config': {},

    # whether to overwrite existing files (uses existing files if not set)
    'overwrite': False,
    # whether to stop at generating the videos or continue to generate forms as well
    'stop': 'video',  # [video, form]
    # number of rollouts in the train environment used to generate explanations
    'episodes': 10,
    # location to save results and logs
    'result_dir': 'experiments',
    # number of frames before and after the branching state
    'window_size': 20,
    # state selection method for the branching state
    'state_selection': 'random',  # [random, critical] (branching state for counterfactual states)
    # use counterfactual states
    'counterfactual': True,

    'counterfactual_config': {
        # policy to continue after counterfactual state
        'rollout_policy': 'behavior',  # [behavior, random]
        # number of time steps to use the counterfactual policy
        'timesteps': 3,
    },
    'video_config': {
        # directory name to store videos in result directory
        'dir_name': 'videos',
        # frames per second
        'fps': 5,
    },
    'form_config': {
        # REQUIRED
        # upload folder google drive id
        'upload_folder_id': None,
        # path to credentials json file for google api
        'credentials': 'credentials.json',
        # token file
        'token_file': 'token.pickle',
        # app script dir
        'app_script_dir': 'explanations/forms',
        # project name
        'project_name': 'cfrl',
    },
    'eval_config': {
        # number of trial iterations of explanation and evaluation
        'num_trials': 10,
        # list of evaluation policies to continue rollouts
        'eval_policies': [
            # Policies are a tuple of (name, algorithm, checkpoint)
            ("test_policy_name", "PPO", "/home/olivia/ray_results/driving-ppo/PPO_DrivingPLE-v0_7c851_00000_0_2020-09-28_11-43-22/checkpoint_2/checkpoint-2"),
        ],
        # distribution of states to pick
        'state_selection': 'random',
        # window size of the evaluation videos
        'window_size': 20,
    },
    # extra create_dataset.py arguments
    'create_dataset_arguments': ['--save-info'],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--experiment-config', type=str, required=True,
                        help='experiment config located in explanations/config/experiments')
    parser.add_argument('--overwrite', action='store_true',
                        help='whether to overwrite existing files (uses existing files if not set)')
    parser.add_argument('--stop', default=None, choices=['video', 'form'],
                        help='whether to stop at generating videos or create the user study form')
    args = parser.parse_args()
    return args


def load_config(path):
    config = yaml.safe_load(open(path))
    return merge_dicts(DEFAULT_CONFIG, config)


def load_policy_config_from_checkpoint(checkpoint_path):
    config_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(config_dir, "../params.pkl")
    with open(config_path, 'rb') as f:
        config = pickle.load(f)
    return config


def create_dataset(config, dataset_file):
    args = []
    args += [config['behavior_policy_config']['checkpoint']]
    args += ['--run', config['behavior_policy_config']['run']]
    args += ['--env', config['env']]
    args += ['--episodes', str(config['episodes'])]
    args += ['--out', dataset_file]
    args += config['create_dataset_arguments']
    create_dataset_main(args)


def generate_counterfactuals(config, dataset_file, video_dir, exp_config_path):
    exp_config_name = pathlib.Path(exp_config_path).stem
    env = config['eval_env']
    policy_id = config['behavior_policy_config']['identifier']
    video_dir = os.path.join(video_dir,
                             f"videos-environment_{env}-behavior_{policy_id}-{exp_config_name}")
    args = []
    args += ['--dataset-file', dataset_file]
    args += ['--env', env]
    args += ['--num-states', str(config['eval_config']['num_trials'])]
    args += ['--save-path', video_dir]
    args += ['--window-len', str(config['window_size'])]
    args += ['--state-selection-method', config['state_selection']]
    args += ['--timesteps', str(config['counterfactual_config']['timesteps'])]
    args += ['--fps', str(config['video_config']['fps'])]
    args += ['--env-config', json.dumps(config['env_config'])]
    args += ['--eval_policies', json.dumps(config['eval_config']['eval_policies'])]
    args += ['--policy_name', policy_id]
    args += ['--run',  config['behavior_policy_config']['run']]
    generate_counterfactuals_main(args)


def generate_forms(config, video_dir):
    args = []
    args += ['--video-dir', video_dir]
    args += ['--app-script-dir', config['form_config']['app_script_dir']]
    args += ['--token-file', config['form_config']['token_file']]
    args += ['--project-name', config['form_config']['project_name']]
    args += ['--deployment-folder-id', config['form_config']['upload_folder_id']]
    args += ['--credentials', config['form_config']['credentials']]
    args += ['--config', json.dumps(config)]
    generate_forms_main(args)


def main():
    args = parse_args()
    config = load_config(args.experiment_config)

    stop = args.stop if args.stop is not None else config['stop']
    overwrite = config['overwrite'] or args.overwrite

    # file structure of the results directory
    result_dir = os.path.abspath(config['result_dir'])
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    # the experiment name is the name of the yaml file
    experiment_name, _ = os.path.splitext(os.path.basename(args.experiment_config))
    experiment_dir = os.path.join(result_dir, experiment_name)
    print(f'saving results to {experiment_dir}')
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)
    dataset_file = os.path.join(experiment_dir, 'data.pkl')
    # create dataset
    if overwrite or not os.path.exists(dataset_file):
        create_dataset(config, dataset_file)
    else:
        print('dataset file already exists.')
        print(dataset_file)

    video_dir = os.path.join(experiment_dir, config['video_config']['dir_name'])
    if config['counterfactual']:
        generate_videos = True
        if os.path.exists(video_dir):
            if len(os.listdir(video_dir)) > 0:
                print(f'files exist at {video_dir}')
                if not overwrite:
                    generate_videos = False
                    print('skipping generating videos.')
                else:
                    print('overwriting files...')
                    shutil.rmtree(video_dir)
                    os.mkdir(video_dir)
        if generate_videos:
            generate_counterfactuals(config, dataset_file, video_dir, args.experiment_config)
    else:
        raise NotImplementedError

    if stop == 'video':
        return

    generate_forms(config, video_dir)


if __name__ == '__main__':
    main()
