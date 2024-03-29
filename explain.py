import argparse
import json
import os
import pickle
import shutil

import yaml
from ray.tune.utils import merge_dicts

from explanations.create_dataset import main as create_dataset_main
from explanations.generate_counterfactuals import main as generate_counterfactuals_main

PERFORMANCE_SELECTION = 'performance_selection'
BEHAVIOR_CONTINUATION = 'behavior_continuation'
PERFORMANCE_EVALUATION = 'performance_evaluation'

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
    # video: generate videos and exit
    # study: output config files necessary to import questionnaire to study server
    # doc: (deprecated) output word doc study
    # html: (deprecated) output html study
    'stop': 'study',  # [video, doc, html, study]
    # number of rollouts in the train environment used to generate explanations
    'episodes': 10,
    # location to save results and logs
    'result_dir': 'experiments',
    # whether to interpret paths in config as relative to the config file directory
    # can be bool or path
    'relative_config_path': False,
    # number of frames before and after the branching state
    'window_size': 20,
    # state selection method for the branching state
    'state_selection': 'random',  # [random, critical, initial] (branching state for counterfactual states)
    # What explanation method to use
    'explanation_method': ['random', 'critical', 'counterfactual'],  # [counterfactual, critical, random]
    # use counterfactual states
    'counterfactual': True,

    'counterfactual_config': {
        # policy to continue after counterfactual state
        'rollout_policy': 'behavior',  # [behavior, random]
        # number of time steps to use the counterfactual policy
        'timesteps': 10,
        # method for counterfactual exploration to get to the counterfactual state
        # random: take random actions
        # policy: use the policy specified in exploration_policy
        'exploration_method': 'random',  # [random, policy]
        # to use a custom exploration policy, set checkpoint, run, and name arguments like in the behavior_policy_config
        'exploration_policy': None,
        'show_exploration': False,
    },
    'video_config': {
        # directory name to store videos in result directory
        'dir_name': 'videos',
        # frames per second
        'fps': 3,
        # width of the colored boarder around the videos
        'border_width': 0,
        # downscaling of videos, primarily used to save space
        'downscale': 2,
        # mp4 or gif
        'format': 'mp4',
        # Overlay text on videos showing driver, timestep, and/or reward info
        'show_text': {'show_agent': True, 'show_timestep': False, 'show_reward': False},
        # settings configuration
        # this gets added to the kwargs in generate_counterfactuals.py
        # useful for configuring text settings and video settings
        'settings_config': {}
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
        # id of the document user study
        'id': None,
    },
    'study_config': {
        'id': None,
        # you can configure the text for the study builder
        'text': {},
        # passed into config yaml that's generated and used to import questionnaires
        'build_config': {
            # used by import questionnaire
            'include_context': True,
            # used by import questionnaire
            'include_continuation': True,
            # puts all the explanations at the beginning
            'batched_explanations': False,
        }
    },
    'eval_config': {
        # number of trial iterations of explanation and evaluation
        'num_trials': 10,
        # number of extra branching states we compute in case the trajectory ends during exploration
        'num_buffer_trials': 30,
        # list of evaluation policies to continue rollouts
        # Policies are a dict of {'name': name, 'run': run, 'checkpoint': checkpoint}
        'eval_policies': [],
        # whether to include the behavior policy in the evaluation videos
        'evaluate_behavior_policy': True,
        # distribution of states to pick
        # [random, critical, initial, manual]
        'state_selection': 'random',
        # if manual selection, must have path to dataset pickle file
        'evaluation_dataset': None,
        # takes effect with manual states
        # TODO: not sure what this does, possibly duplicate of timesteps?
        'num_eval_steps': 20,
        # window size of the evaluation videos
        'window_size': 20,
        # number of time steps to use the counterfactual policy
        'timesteps': 0,
        # whether to save videos side by side or separately
        'side_by_side': False,
        # anything in here overwrites video config for evaluation videos
        'video_config': {
            'show_text': {'show_agent': False, 'show_timestep': False, 'show_reward': False}
        },
        # Optional pkl files for hardcoded evaluation policies (expects list of paths)
        # these are generated with explanations/generate_eval.py
        'alt_file_names': None,
    },
    # extra create_dataset.py arguments
    'create_dataset_arguments': ['--save-info'],
    # remove files with this extension
    'remove_ext': ['pkl'],
    # which evaluation task to create
    # if PERFORMANCE_SELECTION, the behavior policies are the eval_policies
    'eval_task': BEHAVIOR_CONTINUATION,  # PERFORMANCE_SELECTION OR BEHAVIOR_CONTINUATION
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--experiment-config', type=str,
                        help='experiment file config. see the default parameters in this file')
    parser.add_argument('--overwrite', action='store_true',
                        help='whether to overwrite existing files (uses existing files if not set)')
    parser.add_argument('--stop', default=None, choices=['video', 'form', 'doc', 'study'],
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


def create_dataset(config, dataset_file, eval=False, policy=None):
    args = []
    if policy is not None:
        args += [policy['checkpoint']]
        args += ['--run', policy['run']]
    else:
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


def generate_explanation_videos(config, dataset_file, video_dir, explanation_method=None, policy_dict=None):
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
    args += ['--settings-config', json.dumps(config['video_config']['settings_config'])]
    args += ['--downscale', str(config['video_config']['downscale'])]
    args += ['--env-config', json.dumps(config['env_config'])]
    args += ['--eval-policies', json.dumps([])]  # no evaluation policies for explanations
    policy_name = config['behavior_policy_config']['name'] if policy_dict is None else policy_dict['name']
    policy_run = config['behavior_policy_config']['run'] if policy_dict is None else policy_dict['run']
    args += ['--policy-name', policy_name]
    args += ['--run', policy_run]
    if config['stop'] == 'html' and config['video_config']['format'] != 'mp4':
        print(f'When generating a html study, the video format must be mp4. '
              f'You are currently using {config["video_config"]["format"]}. '
              f'Set the video_config/format to mp4 in the configuration file. '
              f'Retroactively setting video_config/format to mp4...')
        config['video_config']['format'] = 'mp4'
    if config['stop'] == 'doc' and config['video_config']['format'] != 'gif':
        print(f'When generating a doc study, the video format must be gif. '
              f'You are currently using {config["video_config"]["format"]}. '
              f'Set the video_config/format to gif in the configuration file. '
              f'Retroactively setting video_config/format to gif...')
        config['video_config']['format'] = 'gif'
    if config['video_config']['format'] is not None:
        args += ['--video-format', config['video_config']['format']]
    args += ['--exploration-method', config['counterfactual_config']['exploration_method']]
    args += ['--exploration-policy', json.dumps(config['counterfactual_config']['exploration_policy'])]
    args += ['--show-text', json.dumps(config['video_config']['show_text'])]
    if config['counterfactual_config']['show_exploration']:
        args += ['--show-exploration']
    generate_counterfactuals_main(args)


def generate_evaluation_videos(config, dataset_file, video_dir):
    video_config = merge_dicts(config['video_config'], config['eval_config']['video_config'])
    args = []
    args += ['--dataset-file', dataset_file]
    args += ['--env', config['eval_env']]
    args += ['--num-states', str(config['eval_config']['num_trials'])]
    args += ['--num-buffer-states', str(config['eval_config']['num_buffer_trials'])]
    args += ['--save-path', video_dir]
    args += ['--window-len', str(config['eval_config']['window_size'])]
    args += ['--state-selection-method', config['eval_config']['state_selection']]
    args += ['--timesteps', str(config['eval_config']['timesteps'])]
    args += ['--fps', str(video_config['fps'])]
    args += ['--border-width', str(video_config['border_width'])]
    args += ['--settings-config', json.dumps(video_config['settings_config'])]
    args += ['--downscale', str(video_config['downscale'])]
    args += ['--env-config', json.dumps(config['eval_env_config'])]
    args += ['--eval-policies', json.dumps(config['eval_config']['eval_policies'])]
    if config['eval_config']['evaluate_behavior_policy']:
        args += ['--evaluate-behavior-policy']
    args += ['--policy-name', config['behavior_policy_config']['name']]
    args += ['--run', config['behavior_policy_config']['run']]
    args += ['--behavioral-policy', config['behavior_policy_config']['checkpoint']]
    if config['eval_config']['side_by_side']:
        args += ['--side-by-side']
    else:
        args += ['--save-separate']
    if config['stop'] == 'html' and video_config['format'] != 'mp4':
        print(f'When generating a html study, the video format must be mp4. '
              f'You are currently using {video_config["format"]}. '
              f'Set the video_config/format to mp4 in the configuration file. '
              f'Retroactively setting video_config/format to mp4...')
        video_config['format'] = 'mp4'
    if config['stop'] == 'doc' and video_config['format'] != 'gif':
        print(f'When generating a doc study, the video format must be gif. '
              f'You are currently using {video_config["format"]}. '
              f'Set the video_config/format to gif in the configuration file. '
              f'Retroactively setting video_config/format to gif...')
        video_config['format'] = 'gif'
    if video_config['format'] is not None:
        args += ['--video-format', video_config['format']]
    args += ['--exploration-method', 'random']
    args += ['--exploration-policy', json.dumps(None)]
    if config['eval_config']['alt_file_names'] is not None:
        args += ['--alt-file-names', json.dumps(config['eval_config']['alt_file_names'])]
    args += ['--num-eval-steps', str(config['eval_config']['num_eval_steps'])]
    args += ['--show-text', json.dumps(video_config['show_text'])]
    if config['eval_task'] == BEHAVIOR_CONTINUATION:
        args += ['--randomize-videos']
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


def generate_study(config, video_dir, builder='doc'):
    from explanations.generate_study import main as generate_study_main
    args = []
    args += ['--video-dir', video_dir]
    args += ['--save-dir', video_dir]
    if config['doc_config']['id'] is None:
        import random
        config['doc_config']['id'] = f'{random.randint(0, 1000):03d}'
    args += ['--config', json.dumps(config)]
    args += ['--builder', builder]
    generate_study_main(args)


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
        if isinstance(config['relative_config_path'], bool) and config['relative_config_path']:
            os.chdir(os.path.dirname(args.experiment_config))
        elif isinstance(config['relative_config_path'], str) and os.path.exists(config['relative_config_path']):
            os.chdir(config['relative_config_path'])
    else:
        config = merge_dicts(DEFAULT_CONFIG, args.config)

    # required checks
    assert 'behavior_policy_config' in config
    assert 'checkpoint' in config['behavior_policy_config']
    assert 'run' in config['behavior_policy_config']
    assert 'env' in config
    assert 'eval_env' in config
    if config['eval_task'] == BEHAVIOR_CONTINUATION:
        assert config['behavior_policy_config']['checkpoint'] is not None
        assert config['behavior_policy_config']['run'] is not None
    elif config['eval_task'] == PERFORMANCE_SELECTION:
        assert len(config['eval_config']['eval_policies']) > 1
    elif config['eval_task'] == PERFORMANCE_EVALUATION:
        assert len(config['eval_config']['eval_policies']) == 0
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
    # TODO new: 1 per policy
    if config['eval_task'] == PERFORMANCE_SELECTION:
        explanation_datasets = [os.path.join(experiment_dir, f'explanation_dataset_{d["name"]}.pkl')
                                for d in config['eval_config']['eval_policies']]
    else:
        explanation_dataset = os.path.join(experiment_dir, 'explanation_dataset.pkl')
    manual_selection = config['eval_config']['state_selection'] == 'manual'
    if manual_selection:
        evaluation_dataset = config['eval_config']['evaluation_dataset']
    else:
        # must use behavior policy to select states
        if config['behavior_policy_config']['checkpoint'] is None or config['behavior_policy_config']['run'] is None:
            raise ValueError(
                f"must define a behavior policy with state selection method {config['eval_config']['state_selection']}")
        evaluation_dataset = os.path.join(experiment_dir, 'evaluation_dataset.pkl')
    # create dataset
    if overwrite or not os.path.exists(explanation_dataset):
        if config['eval_task'] == PERFORMANCE_SELECTION:
            for policy_dict, dset in zip(config['eval_config']['eval_policies'], explanation_datasets):
                create_dataset(config, dset, eval=False, policy=policy_dict)
        else:
            create_dataset(config, explanation_dataset, eval=False)
    else:
        print('Explanation dataset file already exists.')
        print(explanation_dataset)

    if (not manual_selection) and (overwrite or not os.path.exists(evaluation_dataset)):
        create_dataset(config, evaluation_dataset, eval=True)
    else:
        print('Evaluation dataset file already exists.')
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
                    # config['new_eval_task'] = True
                    if config['eval_task'] == PERFORMANCE_SELECTION:
                        for policy_dict, dset in zip(config['eval_config']['eval_policies'], explanation_datasets):
                            explain_vid_dir = os.path.join(explain_dir, policy_dict['name'])
                            generate_explanation_videos(config, dset, explain_vid_dir, expl_method)
                    else:
                        generate_explanation_videos(config, explanation_dataset, explain_dir, expl_method)
            else:
                if config['eval_task'] == PERFORMANCE_SELECTION:
                    for policy_dict, dset in zip(config['eval_config']['eval_policies'], explanation_datasets):
                        explain_vid_dir = os.path.join(explain_dir, policy_dict['name'])
                        generate_explanation_videos(config, dset, explain_vid_dir, config['explanation_method'])
                else:
                    generate_explanation_videos(config, explanation_dataset, explain_dir)
            generate_evaluation_videos(config, evaluation_dataset, eval_dir)
    else:
        raise NotImplementedError

    if stop == 'form':
        # broken for now
        raise NotImplementedError
    generate_study(config, video_dir, stop)

    for ext in config.get('remove_ext', []):
        remove_ext(experiment_dir, ext)


if __name__ == '__main__':
    main()
