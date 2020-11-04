import copy

import numpy as np
from ray.tune.utils import merge_dicts

# sample configuration options
# these keys are the values of the "sample" key
SAMPLE_CONFIG = {
    # randomly sample n items out of list of values for a single trial
    'num_samples': 1,
    # number of trials to repeat. adds a combinatorial amount of experiments.
    'num_trials': 1,
    # sample with replacement
    'replace': True,
    # list of values. must be larger or equal to num_sample
    'values': None,
    # if set will sample from a distribution rather than from values, ignores replacement
    # uniform: [low, high]
    # normal: [mean, std]
    'dist_info': None,
    # distribution
    'dist': 'uniform',  # uniform or normal
    # (optional) add probability for each item for sampling. values are renormalized for replacement
    'probs': None,
    # (optional) list key path to config value. ex. ["key1", "key2", "key3"]
    # whatever the value for the key path is not sampled from values. probabilities renormalized
    'exclude': None,
    # if set true, when num_samples = 1, the value is pulled out from the list of samples.
    'reduce_values': True,
}


def key_path_get_value(d, key_path: list):
    """
    recursively index d to get the value at the end of key path
    :param d: index-able data structure, can be either dict, list, or tuple
    :param key_path: list of keys to recursively index
    :return: value at the end of key path
    """
    if len(key_path) > 0:
        return key_path_get_value(d[key_path[0]], key_path[1:])
    else:
        return d


def key_path_set_value(d, key_path: list, value):
    """
        recursively index d to set the value at the end of key path
        :param d: index-able data structure, can be either dict, list, or tuple
        :param key_path: list of keys to recursively index
        :param value: value to set at the end of the key path
        :return: the modified data structure d
        """
    if len(key_path) > 0:
        d[key_path[0]] = key_path_set_value(d[key_path[0]], key_path[1:], value)
        return d
    else:
        return value


def sample(sample_config, exclude=None):
    """
    This method samples values as specified in the sample config
    :param sample_config: the sample config with the format specified in `SAMPLE_CONFIG`
    :param exclude: value to exclude when sampling from list of values
    :return: sampled values
    """
    if sample_config['dist_info']:
        if sample_config['dist'] == 'uniform':
            low, high = sample_config['dist_info']
            batch = np.random.rand(
                sample_config['num_trials'], sample_config['num_samples']) * (high - low) + low
            if sample_config['num_samples'] == 1:
                batch = np.squeeze(batch, axis=1)
            new_values = batch.tolist()
        elif sample_config['dist'] == 'normal':
            mean, std = sample_config['dist_info']
            batch = np.random.randn(sample_config['num_trials'], sample_config['num_samples']) * std + mean
            if sample_config['num_samples'] == 1:
                batch = np.squeeze(batch, axis=1)
            new_values = batch.tolist()
        else:
            raise NotImplementedError
    else:
        sample_values = sample_config['values']
        size = (sample_config['num_trials'], sample_config['num_samples'])
        probs = sample_config['probs']
        if exclude is not None:
            # count exclude as one instance of values
            if probs is None:
                probs = np.ones(len(sample_values))
            for i, val in enumerate(sample_values):
                if exclude == val:
                    probs[i] *= 0  # exclude by setting probability to 0
            probs /= np.sum(probs)  # renormalize

        sample_indices = np.arange(len(sample_values))
        batch = np.random.choice(sample_indices, size=size, replace=sample_config['replace'], p=probs)
        if sample_config['num_samples'] == 1 and sample_config['reduce_values']:
            batch = np.squeeze(batch, axis=1)
            new_values = []
            for trial in range(size[0]):
                trial_index = batch[trial]
                val = copy.deepcopy(sample_values[trial_index])
                new_values.append(val)
        else:
            new_values = []
            for trial in range(size[0]):
                row = []
                for i in range(size[1]):
                    trial_index = batch[trial][i]
                    val = copy.deepcopy(sample_values[trial_index])
                    row.append(val)
                new_values.append(row)

    return new_values


def traverse_config(multi_config: dict):
    """
    Go through the multi config and sample subset configs
    :param multi_config: config with "sample" statements
    :return: list of configs
    """
    config_list = [copy.deepcopy(multi_config)]
    queue = [[key] for key in multi_config.keys()]
    sample_functions = []
    while len(queue) > 0:
        key_path = queue.pop(0)
        values = key_path_get_value(multi_config, key_path)
        if key_path[-1] == 'sample':
            assert isinstance(values, dict)
            parent_key_path = key_path[:-1]
            # parent_config = key_path_get_value(multi_config, parent_key_path)
            # the value of the parent_key_path is replaced with the sample in config_list
            # this assertion can be false when the default explain config is merged into the multi config
            # assert len(parent_config) == 1
            sample_config = merge_dicts(SAMPLE_CONFIG, values)

            if sample_config['exclude'] is not None:
                def exclude_func(exclude_config: dict):
                    return key_path_get_value(exclude_config, sample_config['exclude'])
            else:
                def exclude_func(exclude_config: dict):
                    return None
            # TODO: functionality to sample together
            new_config_list = []
            for c in config_list:
                samples = sample(sample_config, exclude_func(c))
                for val in samples:
                    config = key_path_set_value(copy.deepcopy(c), parent_key_path, val)
                    new_config_list.append(config)
            config_list = new_config_list

        elif isinstance(values, dict):
            for key in values.keys():
                queue.append(key_path + [key])
        elif isinstance(values, list) or isinstance(values, tuple):
            for i in range(len(values)):
                queue.append(key_path + [i])

    return config_list
