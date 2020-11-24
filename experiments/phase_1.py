import argparse
import json
import os
import traceback

import ray
import yaml
from numpy.random import default_rng

from experiments.utils.sampling import traverse_config
from explain import main as explain_main


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--multi-config', type=str,
                        help='config with indicators for which to compare experiments')
    parser.add_argument('--config', type=json.loads, default='{}', help='optional bypass multi config file')
    parser.add_argument('--stop', default=None, choices=['video', 'form', 'doc'],
                        help='whether to stop at generating videos or create the user study form')

    args = parser.parse_args(argv)
    stop = args.stop
    if args.multi_config is not None and os.path.exists(args.multi_config):
        multi_config = yaml.safe_load(open(args.multi_config))
    else:
        multi_config = args.config

    list_configs = traverse_config(multi_config)

    rng = default_rng()
    numbers = rng.choice(1000, size=len(list_configs), replace=False)
    for doc_id, config in zip(numbers, list_configs):
        # this will be the name of the folder to save results to
        # result_dir/name
        config['name'] = 'behavior_policy_' + config['behavior_policy_config']['name'] + '-distribution_shift_' + \
                         config['eval_env_config']['name']
        args = []
        args += ['--config', json.dumps(config)]
        args += ['--stop', stop]
        args += ['--doc-id', f'{doc_id:03d}']
        try:
            explain_main(args)
        except Exception as e:
            print(e)
            print(f'errored with config {json.dumps(config)}')
            print(traceback.format_exc())


if __name__ == '__main__':
    main()
