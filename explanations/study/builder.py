import json
import os

import yaml
from ray.tune.utils import merge_dicts

from explanations.study.files import get_eval_names, get_explain_name, get_solutions, get_eval_name, get_eval_rewards, \
    get_explain_names, path_verifier
from explanations.study.text import get_introduction_text, get_explain_study_text, get_eval_study_text, \
    get_title_text, get_explain_heading_text, get_eval_heading_text, get_question_text

BEHAVIOR_CONTINUATION = 'behavior_continuation'
PERFORMANCE_SELECTION = 'performance_selection'
TASKS = [
    BEHAVIOR_CONTINUATION,
    PERFORMANCE_SELECTION,
]


class StudyBuilder:
    def __init__(self, save_dir, root_dir, explanation_method, num_trials, config, name):
        """
        Parent class to aid in building studies in various formats
        @param save_dir: the directory to save the output study
        @param root_dir: the directory containing generated videos
        @param explanation_method: type of explanation method (random, critical, counterfactual)
        @param num_trials: number of trials
        @param config: same format as the config in explain.py
        @param name: the name of the study
        """
        self.save_dir = save_dir
        self.root_dir = root_dir
        self.explanation_method = explanation_method
        self.num_trials = num_trials
        self.config = config
        self.extension = self.config['video_config']['format']
        self.task = self.config['eval_task']
        self.policy_names = [p['name'] for p in self.config['eval_config']['eval_policies']]
        self.study_config = config.get('study_config')
        self.text_config = {}
        text_config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'text_configs')
        env_text_config_path = os.path.join(text_config_dir, config.get('env') + '.yaml')
        if os.path.exists(env_text_config_path):
            with open(env_text_config_path) as f:
                self.text_config = yaml.safe_load(f)
        else:
            with open(os.path.join(text_config_dir, 'default.yaml')) as f:
                self.text_config = yaml.safe_load(f)

        self.name = name
        self.build_config = {}

    def get_text(self, key, default):
        if key in self.study_config['text']:
            value = self.study_config['text'].get(key)
            assert isinstance(value, type(default))
            return value
        else:
            return default

    def build(self):
        intro_text = self.get_text('intro_text', get_introduction_text(self.text_config))
        title_text = self.get_text('title_text', get_title_text())
        self.build_intro(title_text, intro_text)
        explanation_study_text = self.get_text('explanation_study_text',
                                               get_explain_study_text(self.explanation_method, self.text_config))
        eval_study_text = self.get_text('eval_study_text', get_eval_study_text(self.text_config, self.config))
        explain_heading_text = self.get_text('explain_heading_text', get_explain_heading_text())
        eval_heading_text = self.get_text('eval_heading_text', get_eval_heading_text())
        question_text = self.get_text('question_text', get_question_text(self.text_config))
        eval_rewards, num_choices = get_eval_rewards(self.root_dir)
        if self.task == BEHAVIOR_CONTINUATION:
            solutions, _ = get_solutions(self.root_dir, self.config)
            self.build_config['solutions'] = solutions

        # save to build config
        self.build_config['name'] = self.name
        self.build_config['num_trials'] = self.num_trials
        self.build_config['intro_text'] = intro_text
        self.build_config['title_text'] = title_text
        self.build_config['explanation_study_text'] = explanation_study_text
        self.build_config['eval_study_text'] = eval_study_text
        self.build_config['explain_heading_text'] = explain_heading_text
        self.build_config['eval_heading_text'] = eval_heading_text
        self.build_config['question_text'] = question_text
        self.build_config['eval_rewards'] = eval_rewards
        self.build_config['num_choices'] = num_choices
        self.build_config['trial_heading_texts'] = []
        self.build_config['explain_video_paths'] = []
        self.build_config['eval_video_paths'] = []
        self.build_config['task_name'] = self.config['eval_task']
        self.build_config = merge_dicts(self.build_config, self.study_config['build_config'])

        for trial in range(self.num_trials):
            explanation_dir = f'explain-{self.explanation_method}'
            eval_dir = 'eval'
            trial_heading_text = f'Trial {trial + 1}'
            self.trial_heading(trial_heading_text)
            if self.task == BEHAVIOR_CONTINUATION:
                explain_video_paths = get_explain_name(self.explanation_method, self.config, explanation_dir, trial,
                                                      self.extension)
            elif self.task == PERFORMANCE_SELECTION:
                explain_video_paths = get_explain_names(self.explanation_method, self.policy_names, self.config,
                                                       explanation_dir, trial, self.extension)
            else:
                raise NotImplementedError
            if self.config['eval_config']['side_by_side']:
                eval_video_paths = get_eval_name(eval_dir, trial, extension=self.extension)
            else:
                eval_video_paths = get_eval_names(eval_dir, trial, self.build_config['num_choices'],
                                                  extension=self.extension)
            self.add_explanations(explain_video_paths, self.root_dir, explain_heading_text, explanation_study_text)
            self.add_evaluations(eval_video_paths, self.root_dir, eval_heading_text, eval_study_text, question_text)

            # save to build config
            self.build_config['trial_heading_texts'].append(trial_heading_text)
            self.build_config['explain_video_paths'].append(explain_video_paths)
            self.build_config['eval_video_paths'].append(eval_video_paths)

        # verify paths exist
        path_verifier(self.root_dir, self.build_config['explain_video_paths'])
        path_verifier(self.root_dir, self.build_config['eval_video_paths'])

        self.build_outro()
        self.save_build_config()
        self.save()

    def build_intro(self, title: str, intro_text: list):
        pass

    def build_outro(self):
        pass

    def add_explanations(self, video_path, root_dir, heading_text, body_text):
        pass

    def add_evaluations(self, video_path, root_dir, heading_text, body_text, question_text):
        pass

    def save_build_config(self):
        build_config_file = os.path.join(self.save_dir, self.name + '_config.json')
        with open(build_config_file, mode='w') as f:
            json.dump(self.build_config, f)

    def save(self):
        pass

    def trial_heading(self, trial_heading_text):
        pass
