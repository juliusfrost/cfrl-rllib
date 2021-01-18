import csv
import os


def get_explain_name(explanation_method, config, extension='gif'):
    if explanation_method == 'random' or explanation_method == 'critical':
        def get_name(explanation_dir, trial):
            return os.path.join(explanation_dir, f'baseline_window-trial_{trial}.{extension}')
    elif explanation_method == 'counterfactual':
        def get_name(explanation_dir, trial):
            return os.path.join(explanation_dir,
                                f'counterfactual_window-'
                                f'{config["behavior_policy_config"]["name"]}-'
                                f't_{trial}.{extension}')
    else:
        raise NotImplementedError
    return get_name


def get_eval_name(eval_dir, trial, extension='gif'):
    return os.path.join(eval_dir, f'counterfactual_window-t_{trial}.{extension}')


def get_solutions(root_dir, config):
    solution_file = os.path.join(root_dir, 'eval', 'counterfactual_window-answer_key.txt')
    solutions = []
    num_choices = None
    with open(solution_file) as csvfile:
        for row in csv.reader(csvfile):
            policies = row[1:-1]
            solution = policies.index(config['behavior_policy_config']['name'])
            solutions.append(solution)
            if num_choices is None:
                num_choices = len(policies)
    return solutions, num_choices
