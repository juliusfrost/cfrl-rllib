import csv
import os

sep = '/'  # used to make paths cross platform


def path_verifier(root, paths):
    if isinstance(paths, list):
        for p in paths:
            path_verifier(root, p)
    elif isinstance(paths, str) and not os.path.exists(os.path.join(root, paths)):
        raise FileNotFoundError(f'Path {paths} does not exist in root directory {root}!')


def get_explain_name(explanation_method, config, explanation_dir, trial, extension='gif'):
    if explanation_method == 'random' or explanation_method == 'critical':
        return explanation_dir + sep + f'baseline_window-trial_{trial}.{extension}'
    elif explanation_method == 'counterfactual':
        return explanation_dir + sep + \
               f'counterfactual_window-{config["behavior_policy_config"]["name"]}-t_{trial}.{extension}'
    else:
        raise NotImplementedError


def get_explain_names(explanation_method, policy_names, config, explanation_dir, trial, extension='gif'):
    paths = []
    if explanation_method == 'random' or explanation_method == 'critical':
        for p in policy_names:
            paths.append(explanation_dir + sep + p + sep + f'baseline_window-trial_{trial}.{extension}')
    elif explanation_method == 'counterfactual':
        for p in policy_names:
            paths.append(explanation_dir + sep + p + sep +
                         f'counterfactual_window-{config["behavior_policy_config"]["name"]}-t_{trial}.{extension}')
    else:
        raise NotImplementedError
    return paths


def get_eval_name(eval_dir, trial, extension='gif'):
    return eval_dir + sep + f'counterfactual_window-t_{trial}.{extension}'


def get_eval_names(eval_dir, trial, num_choices, extension='gif'):
    """
    Returns [[contexts], [continuations]]
    """
    contexts = []
    for i in range(1):
        context_path = eval_dir + sep + f'context_vid-t_{trial}_{i}.{extension}'
        contexts.append(context_path)
    continuations = []
    for i in range(num_choices):
        continuation_path = eval_dir + sep + f'vids_{trial}' + sep + f'counterfactual_vid-t_{trial}_{i}.{extension}'
        continuations.append(continuation_path)
    return [contexts, continuations]


def get_solutions(root_dir, config):
    solution_file = root_dir + sep + 'eval' + sep + 'counterfactual_vid-answer_key.txt'
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


def get_eval_rewards(root_dir):
    reward_file = root_dir + sep + 'eval' + sep + 'counterfactual_vid-reward_answer_key.txt'
    rewards = []
    num_choices = None
    with open(reward_file) as csvfile:
        for row in csv.reader(csvfile):
            rewards.append(row[1:-1])
            if num_choices is None:
                num_choices = len(row[1:-1])
    return rewards, num_choices
