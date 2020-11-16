import argparse
import json
import os
import os.path

import numpy as np
import docx
from docx.shared import Inches
from docx.text.run import Run


def parse_args(parser_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-dir', default='videos', help='directory to load the videos from')
    parser.add_argument('--save-dir', help='directory to save output docs')
    parser.add_argument('--config', type=json.loads, default="{}", help='experiment configuration')
    parser.add_argument('--doc-config', type=json.loads, default="{}", help='generated doc configuration')
    args = parser.parse_args(parser_args)
    # if not os.path.exists(args.video_dir):
    #     raise FileNotFoundError(f'Video directory does not exist: {args.video_dir}')
    return args


def get_explain_name(explanation_method, config):
    if explanation_method == 'random' or explanation_method == 'critical':
        def get_name(explanation_dir, trial):
            return os.path.join(explanation_dir, f'baseline_window-trial_{trial}.gif')
    elif explanation_method == 'counterfactual':
        def get_name(explanation_dir, trial):
            return os.path.join(explanation_dir,
                                f'counterfactual_window-'
                                f'{config["behavior_policy_config"]["name"]}-'
                                f't_{trial}.gif')
    else:
        return NotImplementedError
    return get_name


def get_eval_name(eval_dir, trial):
    return os.path.join(eval_dir, f'counterfactual_window-t_{trial}.gif')


def add_explanations(document, trial, image_path):
    p = document.add_paragraph('Explanation')
    p = document.add_paragraph('')
    r: Run = p.add_run()
    try:
        r.add_picture(image_path, height=Inches(2))
    except FileNotFoundError:
        print(f'Could not find image {image_path}')


def add_evaluations(document, trial, eval_image):
    p = document.add_paragraph('Evaluation')
    p = document.add_paragraph('')
    r = p.add_run()
    r.add_picture(eval_image, height=Inches(2))
    p = document.add_paragraph('')
    p = document.add_paragraph('Which continuation do you think came from the explained behavior policy? ')


def build_document(save_dir, video_dir, explanation_method, num_trials, config, doc_name):
    document = docx.Document()
    document.add_heading('Explainable Reinforcement Learning User Study', 0)
    p = document.add_paragraph()
    r: Run = p.add_run()
    r.add_text('Here are the explanations:\n')
    name_formula = get_explain_name(explanation_method, config)
    for trial in range(num_trials):
        explanation_dir = os.path.join(video_dir, f'explain-{explanation_method}')
        eval_dir = os.path.join(video_dir, 'eval')

        document.add_heading(f'Trial {trial}', level=1)
        add_explanations(document, trial, name_formula(explanation_dir, trial))
        add_evaluations(document, trial, get_eval_name(eval_dir, trial))

    document.save(os.path.join(save_dir, f'{doc_name}.docx'))


def read_eval_policies(video_dir):
    eval_dir = os.path.join(video_dir, 'eval')
    eval_images = {}
    for image_file in os.listdir(eval_dir):
        if 'counterfactual_window' in image_file and '.gif' in image_file:
            base_name, ext = os.path.splitext(image_file)
            vid_type, policy, trial_name = base_name.split('-')
            trial = int(trial_name.split('_')[-1])
            if trial not in eval_images:
                eval_images[trial] = {}
            eval_images[trial][policy] = os.path.join(eval_dir, image_file)
    return eval_images


def main(parser_args=None):
    args = parse_args(parser_args)

    num_trials = args.config['eval_config']['num_trials']
    explanation_methods = args.config['explanation_method']
    assert isinstance(explanation_methods, list)

    for explanation_method in explanation_methods:
        build_document(args.save_dir, args.video_dir, explanation_method, num_trials, args.config, args.doc_name)


if __name__ == '__main__':
    main()
