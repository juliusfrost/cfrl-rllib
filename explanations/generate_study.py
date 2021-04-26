import argparse
import json
import os
import os.path

from explanations.study import EXPLANATION_IDS
from explanations.study.builder import StudyBuilder
from explanations.study.document_builder import DocumentStudyBuilder
from explanations.study.html_builder import HTMLStudyBuilder


def parse_args(parser_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-dir', default='videos', help='directory to load the videos from')
    parser.add_argument('--save-dir', help='directory to save output docs')
    parser.add_argument('--config', type=json.loads, default="{}", help='experiment configuration')
    parser.add_argument('--doc-id', type=str, default='000', help='id to be used as the document title')
    parser.add_argument('--builder', choices=['doc', 'html', 'study'])
    args = parser.parse_args(parser_args)
    # if not os.path.exists(args.video_dir):
    #     raise FileNotFoundError(f'Video directory does not exist: {args.video_dir}')
    return args


def main(parser_args=None):
    args = parse_args(parser_args)

    num_trials = args.config['eval_config']['num_trials']
    explanation_methods = args.config['explanation_method']
    assert isinstance(explanation_methods, list)

    doc_id = args.config['doc_config']['id']
    with open(os.path.join(args.save_dir, 'doc_ids.txt'), 'w') as f:
        for explanation_method in explanation_methods:
            method_doc_id = EXPLANATION_IDS[explanation_method]
            full_doc_id = f'{doc_id}-{method_doc_id:03d}'
            f.write(f'{explanation_method}, {full_doc_id},\n')
            if args.builder == 'doc':
                builder_cls = DocumentStudyBuilder
            elif args.builder == 'html':
                builder_cls = HTMLStudyBuilder
            elif args.builder == 'study':
                builder_cls = StudyBuilder
            else:
                raise NotImplementedError
            builder = builder_cls(args.save_dir, args.video_dir, explanation_method, num_trials, args.config,
                                  full_doc_id)
            builder.build()


if __name__ == '__main__':
    main()
