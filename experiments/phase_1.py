import argparse
import json
import os

import yaml

from experiments.utils.sampling import traverse_config
from explain import main as explain_main


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--multi-config', type=str,
                        help='config with indicators for which to compare experiments')
    parser.add_argument('--config', type=json.loads, default='{}', help='optional bypass multi config file')

    args = parser.parse_args(argv)
    if args.multi_config is not None and os.path.exists(args.multi_config):
        multi_config = yaml.safe_load(open(args.multi_config))
    else:
        multi_config = args.config

    list_configs = traverse_config(multi_config)

    for config in list_configs:
        # this will be the name of the folder to save results to
        # result_dir/name
        config['name'] = 'explanations-' + config['behavior_policy_config']['name']
        args = []
        args += ['--config', json.dumps(config)]
        explain_main(args)


if __name__ == '__main__':
    main()
