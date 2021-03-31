import csv

sep = '/'  # used to make paths cross platform


def get_explain_name(explanation_method, config, extension='gif'):
    if explanation_method == 'random' or explanation_method == 'critical':
        def get_name(explanation_dir, trial):
            return explanation_dir + sep + f'baseline_window-trial_{trial}.{extension}'
    elif explanation_method == 'counterfactual':
        def get_name(explanation_dir, trial):
            return explanation_dir + sep + \
                   f'counterfactual_window-{config["behavior_policy_config"]["name"]}-t_{trial}.{extension}'
    else:
        raise NotImplementedError
    return get_name


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
