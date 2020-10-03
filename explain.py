import argparse
import os
import pickle
import shutil

import yaml
from ray.tune.utils import merge_dicts

from explanations.create_dataset import main as create_dataset_main
from explanations.generate_counterfactuals import main as generate_counterfactuals_main

DEFAULT_CONFIG = {
    # REQUIRED
    # config for the behavior policy
    'behavior_policy_config': {
        # path to the rllib policy checkpoint file
        'checkpoint': None,
        # name of algorithm used to train the behavior policy
        'run': None,
    },
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
    # path to the environment configuration yaml file
    # this config defines the environment train and test split
    'env_config': None,
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
        # upload folder google drive id
        'upload_folder_id': None,
        # path to credentials json file for google api
        'credentials': None,
    },
    'eval_config': {
        # number of trial iterations of explanation and evaluation
        'num_trials': 10,
        # list of evaluation policies to continue rollouts
        'eval_policies': [],
        # distribution of states to pick
        'state_selection': 'random',
        # window size of the evaluation videos
        'window_size': 20,
    },
    # extra create_dataset.py arguments
    'create_dataset_arguments': ['--save-info'],
}

DEFAULT_ENV_CONFIG = {
    'train': {
        # environment name
        'env': None,  # required to match the behavior policy train environment
        # environment configuration
        'env_config': {}
    },
    'eval': {
        # environment name
        'env': None,  # must have the same observation and action spaces as the train environment
        # environment configuration
        'env_config': {}
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--experiment-config', type=str, required=True,
                        help='experiment config located in explanations/config/experiments')
    parser.add_argument('--overwrite', action='store_true',
                        help='whether to overwrite existing files (uses existing files if not set)')
    parser.add_argument('--result', default='video', choices=['video', 'form'],
                        help='whether to stop at generating videos or create the user study form')
    args = parser.parse_args()
    return args


def load_config(path):
    config = yaml.safe_load(open(path))
    return merge_dicts(DEFAULT_CONFIG, config)


def load_env_config(path):
    config = yaml.safe_load(open(path))
    return merge_dicts(DEFAULT_ENV_CONFIG, config)


def load_policy_config_from_checkpoint(checkpoint_path):
    config_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(config_dir, "../params.pkl")
    with open(config_path, 'rb') as f:
        config = pickle.load(f)
    return config


def create_dataset(config, dataset_file):
    behavior_policy_config = load_policy_config_from_checkpoint(config['behavior_policy_config']['checkpoint'])
    args = []
    args += [config['behavior_policy_config']['checkpoint']]
    args += ['--run', config['behavior_policy_config']['run']]
    args += ['--env', behavior_policy_config['env']]
    args += ['--episodes', str(config['sample_episodes'])]
    args += ['--out', dataset_file]
    args += config['create_dataset_arguments']
    create_dataset_main(args)


def generate_counterfactuals(config, dataset_file, video_dir):
    behavior_policy_config = load_policy_config_from_checkpoint(config['behavior_policy_config']['checkpoint'])
    args = []
    args += ['--dataset-file', dataset_file]
    args += ['--env', behavior_policy_config['env']]
    args += ['--num-states', str(config['eval_config']['num_trials'])]
    args += ['--save-path', video_dir]
    args += ['--window-len', str(config['window_size'])]
    args += ['--state-selection-method', config['state_selection']]
    args += ['--timesteps', str(config['counterfactual_config']['timesteps'])]
    args += ['--fps', str(config['video_config']['fps'])]
    generate_counterfactuals_main(args)


def main():
    args = parse_args()
    config = load_config(args.experiment_config)

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
    if args.overwrite or not os.path.exists(dataset_file):
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
                if not args.overwite:
                    generate_videos = False
                    print('skipping generating videos.')
                else:
                    print('overwriting files...')
                    shutil.rmtree(video_dir)
                    os.mkdir(video_dir)
        if generate_videos:
            generate_counterfactuals(config, dataset_file, video_dir)
    else:
        raise NotImplementedError



if __name__ == '__main__':
    main()
