import argparse
import json
import os
import pickle
import shutil

import yaml
from ray.tune.utils import merge_dicts

from explanations.create_dataset import main as create_dataset_main
from explanations.generate_counterfactuals import main as generate_counterfactuals_main

DEFAULT_CONFIG = {
    # experiment name
    'name': 'explanation-experiment',
    # REQUIRED
    # config for the behavior policy
    'behavior_policy_config': {
        # path to the rllib policy checkpoint file
        'checkpoint': None,
        # name of algorithm used to train the behavior policy
        'run': None,
        # name of the behavior policy
        'name': "behavior",
    },
    # REQUIRED
    # train environment name
    'env': None,
    # train environment configuration
    'env_config': {},

    # REQUIRED
    # test environment name
    'eval_env': None,
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
    # What explanation method to use
    'explanation_method': ['random', 'counterfactual'],  # [counterfactual, critical, random]
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
        'border_width': 30
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
    'doc_config': {
        'form_name': 'test.docx'
    },
    'eval_config': {
        # number of trial iterations of explanation and evaluation
        'num_trials': 10,
        # number of extra branching states we compute in case the trajectory ends during exploration
        'num_buffer_trials': 10,
        # list of evaluation policies to continue rollouts
        # Policies are a dict of {'name': name, 'run': run, 'checkpoint': checkpoint}
        'eval_policies': [],
        # distribution of states to pick
        'state_selection': 'random',
        # window size of the evaluation videos
        'window_size': 20,
        # number of time steps to use the counterfactual policy
        'timesteps': 3,
    },
    # extra create_dataset.py arguments
    'create_dataset_arguments': ['--save-info'],
    # remove files with this extension
    'remove_ext': ['pkl'],
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--experiment-config', type=str,
                        help='experiment file config. see the default parameters in this file')
    parser.add_argument('--overwrite', action='store_true',
                        help='whether to overwrite existing files (uses existing files if not set)')
    parser.add_argument('--stop', default=None, choices=['video', 'form'],
                        help='whether to stop at generating videos or create the user study form')
    parser.add_argument('--config', type=json.loads, default='{}', help='use json config instead of file config')
    args = parser.parse_args(argv)
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


def create_dataset(config, dataset_file, eval=False):
    args = []
    args += [config['behavior_policy_config']['checkpoint']]
    args += ['--run', config['behavior_policy_config']['run']]
    if not eval:
        args += ['--env', config['env']]
        args += ['--config', json.dumps({'env_config': config['env_config']})]
    else:
        args += ['--env', config['eval_env']]
        args += ['--config', json.dumps({'env_config': config['eval_env_config']})]

    args += ['--episodes', str(config['episodes'])]
    args += ['--out', dataset_file]
    args += config['create_dataset_arguments']
    create_dataset_main(args)


def generate_explanation_videos(config, dataset_file, video_dir, explanation_method=None):
    args = []
    args += ['--dataset-file', dataset_file]
    args += ['--env', config['eval_env']]
    args += ['--num-states', str(config['eval_config']['num_trials'])]
    args += ['--num-buffer-states', str(config['eval_config']['num_buffer_trials'])]
    args += ['--save-path', video_dir]
    args += ['--window-len', str(config['window_size'])]
    args += ['--state-selection-method', config['state_selection']]
    if explanation_method is None:
        args += ['--explanation-method', config['explanation_method']]
    else:
        args += ['--explanation-method', explanation_method]
    args += ['--timesteps', str(config['counterfactual_config']['timesteps'])]
    args += ['--fps', str(config['video_config']['fps'])]
    args += ['--border-width', str(config['video_config']['border_width'])]
    args += ['--env-config', json.dumps(config['env_config'])]
    args += ['--eval-policies', json.dumps([])]  # no evaluation policies for explanations
    args += ['--policy-name', config['behavior_policy_config']['name']]
    args += ['--run', config['behavior_policy_config']['run']]
    generate_counterfactuals_main(args)


def generate_evaluation_videos(config, dataset_file, video_dir):
    args = []
    args += ['--dataset-file', dataset_file]
    args += ['--env', config['eval_env']]
    args += ['--num-states', str(config['eval_config']['num_trials'])]
    args += ['--num-buffer-states', str(config['eval_config']['num_buffer_trials'])]
    args += ['--save-path', video_dir]
    args += ['--window-len', str(config['eval_config']['window_size'])]
    args += ['--state-selection-method', config['eval_config']['state_selection']]
    # args += ['--explanation-method', config['explanation_method']]
    args += ['--timesteps', str(config['eval_config']['timesteps'])]
    args += ['--fps', str(config['video_config']['fps'])]
    args += ['--border-width', str(config['video_config']['border_width'])]
    args += ['--env-config', json.dumps(config['eval_env_config'])]
    args += ['--eval-policies', json.dumps(config['eval_config']['eval_policies'])]
    args += ['--policy-name', config['behavior_policy_config']['name']]
    args += ['--run', config['behavior_policy_config']['run']]
    args += ['--side-by-side']
    generate_counterfactuals_main(args)


def generate_forms(config, video_dir):
    from explanations.generate_forms import main as generate_forms_main
    args = []
    args += ['--video-dir', video_dir]
    args += ['--app-script-dir', config['form_config']['app_script_dir']]
    args += ['--token-file', config['form_config']['token_file']]
    args += ['--project-name', config['form_config']['project_name']]
    args += ['--deployment-folder-id', config['form_config']['upload_folder_id']]
    args += ['--credentials', config['form_config']['credentials']]
    args += ['--config', json.dumps(config)]
    generate_forms_main(args)


def generate_doc(config, video_dir):
    from explanations.generate_doc import main as generate_doc_main
    args = []
    args += ['--video-dir', video_dir]
    args += ['--save-dir', video_dir]
    args += ['--config', json.dumps(config)]
    generate_doc_main(args)


def remove_ext(path, ext='pkl'):
    for f in os.listdir(path):
        f_path = os.path.join(path, f)
        if os.path.isdir(f_path):
            remove_ext(f_path)
        elif os.path.isfile(f_path) and os.path.splitext(f)[1] == f'.{ext}':
            os.remove(f_path)
            print(f'removed {f_path}')


def main(argv=None):
    args = parse_args(argv)
    if args.experiment_config is not None and os.path.exists(args.experiment_config):
        config = load_config(args.experiment_config)
    else:
        config = merge_dicts(DEFAULT_CONFIG, args.config)

    # required checks
    assert 'behavior_policy_config' in config
    assert 'checkpoint' in config['behavior_policy_config']
    assert 'run' in config['behavior_policy_config']
    assert 'env' in config
    assert 'eval_env' in config
    assert config['behavior_policy_config']['checkpoint'] is not None
    assert config['behavior_policy_config']['run'] is not None
    assert config['env'] is not None
    assert config['eval_env'] is not None

    stop = args.stop if args.stop is not None else config['stop']
    overwrite = config['overwrite'] or args.overwrite

    # file structure of the results directory
    result_dir = os.path.abspath(config['result_dir'])
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    # the experiment name is the name of the yaml file
    experiment_name = config['name']
    # experiment_name, _ = os.path.splitext(os.path.basename(args.experiment_config))
    experiment_dir = os.path.join(result_dir, experiment_name)
    print(f'saving results to {experiment_dir}')
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)
    explanation_dataset = os.path.join(experiment_dir, 'explanation_dataset.pkl')
    evaluation_dataset = os.path.join(experiment_dir, 'evaluation_dataset.pkl')
    # create dataset
    if overwrite or not os.path.exists(explanation_dataset):
        create_dataset(config, explanation_dataset, eval=False)
    else:
        print('dataset file already exists.')
        print(explanation_dataset)

    if overwrite or not os.path.exists(evaluation_dataset):
        create_dataset(config, evaluation_dataset, eval=True)
    else:
        print('dataset file already exists.')
        print(evaluation_dataset)

    video_dir = os.path.join(experiment_dir, config['video_config']['dir_name'])
    # TODO: fix file structure
    env = config['eval_env']
    policy_id = config['behavior_policy_config']['name']
    # video_dir = os.path.join(video_dir,
    #                          f"videos-environment_{env}-behavior_{policy_id}-{experiment_name}")
    explain_dir = os.path.join(video_dir, 'explain')
    eval_dir = os.path.join(video_dir, 'eval')

    if config['counterfactual']:
        generate_videos = True
        # TODO: fix logic here when overwrite = False
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
            if isinstance(config['explanation_method'], list):
                for expl_method in config['explanation_method']:
                    explain_dir = os.path.join(video_dir, f'explain-{expl_method}')
                    generate_explanation_videos(config, explanation_dataset, explain_dir, expl_method)
            else:
                generate_explanation_videos(config, explanation_dataset, explain_dir)
            generate_evaluation_videos(config, evaluation_dataset, eval_dir)
    else:
        raise NotImplementedError

    if stop == 'form':
        generate_forms(config, video_dir)
    elif stop == 'doc':
        generate_doc(config, video_dir)

    for ext in config.get('remove_ext', []):
        remove_ext(experiment_dir, ext)


if __name__ == '__main__':
    main()
