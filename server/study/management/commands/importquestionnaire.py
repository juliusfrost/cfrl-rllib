import argparse
import os
import json
import uuid

from django.core.management.base import BaseCommand
from django.db import transaction

from study.models import Questionnaire, Trial, Explanation, Evaluation
from server.settings import STATIC_DIRS
from ._static_upload import upload_video


class Command(BaseCommand):
    help = 'Imports a questionnaire into the database from a local html study'

    def add_arguments(self, parser: argparse.ArgumentParser):
        parser.add_argument('config_path', nargs='+', type=str,
                            help='The path to the study config file (generated after running explain.py) '
                                 'or directory containing it')

    def handle(self, *args, **options):
        import_list = options['config_path']
        for config_path in import_list:
            # if the path is a directory, search for config files and add it to the import list
            if os.path.isdir(config_path):
                for file in os.listdir(config_path):
                    if 'config.json' in file:
                        import_list.append(os.path.join(config_path, file))
                continue
            # load config from json file
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
                    video_dir = os.path.dirname(config_path)
                    explain_static_path = upload_video(os.path.join(video_dir, explain_video_path), STATIC_DIRS[0],
                                                       str(uuid.uuid4()))
                    eval_static_path = upload_video(os.path.join(video_dir, eval_video_path), STATIC_DIRS[0],
                                                    str(uuid.uuid4()))
                    explanation = Explanation(
                        trial=trial,
                        static_path=explain_static_path,
                    )
                    explanation.save()
                    evaluation = Evaluation(
                        trial=trial,
                        static_path=eval_static_path,
                        num_choices=config['num_choices'],
                        solution=config['solutions'][t],
                    )
                    evaluation.save()

            self.stdout.write('successfully imported questionnaire: ' + config['name'])
