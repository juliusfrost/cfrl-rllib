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


def get_eval_names(eval_dir, trial, num_choices, extension='gif'):
    """
    Returns [[contexts], [continuations]]
    """
    contexts = []
    for i in range(num_choices):
        context_path = os.path.join(eval_dir, f'context_vid-t_{trial}_{i}.{extension}')
        if os.path.exists(context_path):
            contexts.append(context_path)
    continuations_dir = os.path.join(eval_dir, f'vids_{trial}')
    continuations = [os.path.join(continuations_dir, f'counterfactual_vid-t_{trial}_{i}.{extension}') for i in
                     range(num_choices)]
    return [contexts, continuations]


def get_solutions(root_dir, config):
    solution_file = os.path.join(root_dir, 'eval', 'counterfactual_vid-answer_key.txt')
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
