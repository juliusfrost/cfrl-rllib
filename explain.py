import argparse

import yaml
from ray.tune.utils import merge_dicts

DEFAULT_CONFIG = {
    # path to the rllib policy checkpoint file
    'behavior_policy_checkpoint': None,
    # location to save results and logs
    'result_dir': 'experiments',
    # number of frames before and after the branching state
    'window_size': 20,
    # state selection method for the branching state
    'state_selection': 'random',  # [random, critical] (branching state for counterfactual states)
    # use counterfactual states
    'counterfactual': False,
    # path to the environment configuration yaml file
    # this config defines the environment train and test split
    'env_config': None,
    'counterfactual_config': {
        # policy to continue after counterfactual state
        'rollout_policy': 'behavior',  # [behavior, random]
    },
    'video_config': {
        # directory to store videos locally
        'video_dir': 'videos',
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
        # list of evaluation policies to continue rollouts
        'eval_policies': [],
        # distribution of states to pick
        'state_selection': 'random',
        # window size of the evaluation videos
        'window_size': 20,
    }
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


def main():
    args = parse_args()
    config = load_config(args.experiment_config)


if __name__ == '__main__':
    main()
