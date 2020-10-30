import argparse
import json
import os
import os.path
import pickle

import docx
from docx.shared import Inches

def parse_args(parser_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-dir', default='videos', help='directory to load the videos from')
    parser.add_argument('--config', type=json.loads, default="{}", help='experiment configuration')
    args = parser.parse_args(parser_args)
    # if not os.path.exists(args.video_dir):
    #     raise FileNotFoundError(f'Video directory does not exist: {args.video_dir}')
    return args

def build_document(video_dir, explanation_method, num_trials):
    document = docx.Document()
    document.add_heading('Explainable Reinforcement Learning User Study', 0)
    p = document.add_paragraph()
    r = p.add_run()
    r.add_text('Here are the explanations:\n')
    name_formula = get_explain_name(explanation_method)
    for trial in num_trials:
        explanation_dir = os.path.join(video_dir, 'explain')
        eval_dir = os.path.join(video_dir, 'eval')

        add_explanations(doc, name_formula(explanation_dir, trial))
        add_evaluations(doc, name_f)
    

    document.save('/Users/ericweiner/Documents/cfrl-rllib/demo.docx')

def main(parser_args=None):
    """Calls the Apps Script API.
    """

    args = parse_args(parser_args)
    for explanation_method in args.config['explanation_methods']:
        build_document(args.video_dir, explanation_method, args.config['eval_config']['num_trials'])



if __name__ == '__main__':
    main()
