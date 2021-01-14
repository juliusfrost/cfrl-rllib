import argparse
import json

from django.core.management.base import BaseCommand
from django.db import transaction

from study.models import Questionnaire, Trial, Explanation, Evaluation


class Command(BaseCommand):
    help = 'Imports a questionnaire into the database from a local html study'

    def add_arguments(self, parser: argparse.ArgumentParser):
        parser.add_argument('config_path', nargs='+', type=str,
                            help='The path to the study config file (generated after running explain.py)')

    def handle(self, *args, **options):
        for config_path in options['config_path']:
            with open(config_path, 'r') as f:
                config = json.load(f)
            with transaction.atomic():
                questionnaire = Questionnaire(
                    name=config['name'],
                    num_trials=config['num_trials'],
                    intro_text='\n'.join(config['intro_text']),
                    title_text=config['title_text'],
                    explanation_study_text=config['explanation_study_text'],
                    eval_study_text=config['eval_study_text'],
                    explain_heading_text=config['explain_heading_text'],
                    eval_heading_text=config['eval_heading_text'],
                    question_text=config['question_text'],
                )
                questionnaire.save()
                for t in range(config['num_trials']):
                    trial_heading_text = config['trial_heading_texts'][t]
                    trial = Trial(questionnaire=questionnaire, trial_heading_text=trial_heading_text, order=t + 1)
                    trial.save()
                    # TODO: convert local files into static files
                    explain_video_path = config['explain_video_paths'][t]
                    eval_video_path = config['eval_video_paths'][t]
                    explanation = Explanation(
                        trial=trial,
                        static_path='',
                    )
                    explanation.save()
                    evaluation = Evaluation(
                        trial=trial,
                        static_path='',
                        num_choices=config['num_choices'],
                        solution=config['solutions'][t],
                    )
                    evaluation.save()
