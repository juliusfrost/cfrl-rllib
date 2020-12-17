import os
import yaml

from explanations.study.files import get_eval_name, get_explain_name
from explanations.study.text import get_introduction_text, get_explain_study_text, get_eval_study_text, get_title_text, \
    get_explain_heading_text, get_eval_heading_text, get_question_text


class StudyBuilder:
    def __init__(self, save_dir, video_dir, explanation_method, num_trials, config, name):
        self.save_dir = save_dir
        self.video_dir = video_dir
        self.explanation_method = explanation_method
        self.num_trials = num_trials
        self.config = config
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

    def build(self):
        intro_text = get_introduction_text(self.text_config)
        title_text = get_title_text()
        self.build_intro(title_text, intro_text)
        name_formula = get_explain_name(self.explanation_method, self.config,
                                        extension=self.config['video_config']['format'])
        explanation_study_text = get_explain_study_text(self.explanation_method, self.text_config)
        eval_study_text = get_eval_study_text(self.text_config, self.config)
        for trial in range(self.num_trials):
            explanation_dir = os.path.join(self.video_dir, f'explain-{self.explanation_method}')
            eval_dir = os.path.join(self.video_dir, 'eval')
            trial_heading_text = f'Trial {trial + 1}'
            self.trial_heading(trial_heading_text)
            explain_heading_text = get_explain_heading_text()
            eval_heading_text = get_eval_heading_text()
            explain_image_path = name_formula(explanation_dir, trial)
            eval_image_path = get_eval_name(eval_dir, trial, extension=self.config['video_config']['format'])
            question_text = get_question_text(self.text_config)
            self.add_explanations(explain_image_path, explain_heading_text, explanation_study_text)
            self.add_evaluations(eval_image_path, eval_heading_text, eval_study_text, question_text)
        self.build_outro()
        self.save()

    def build_intro(self, title: str, intro_text: list):
        raise NotImplementedError

    def build_outro(self):
        pass

    def add_explanations(self, video_path, heading_text, body_text):
        raise NotImplementedError

    def add_evaluations(self, video_path, heading_text, body_text, question_text):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def trial_heading(self, trial_heading_text):
        raise NotImplementedError
