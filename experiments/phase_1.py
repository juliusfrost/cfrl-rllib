import argparse
import json
import os

from ray.tune.utils import merge_dicts

from explain import load_config, DEFAULT_CONFIG
from explain import main as explain_main
from experiments.utils.sampling import traverse_config


def main(argv=None):
    """

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--multi-config', type=str,
                        help='config with indicators for which to compare experiments')
    parser.add_argument('--config', type=json.loads, default='{}', help='optional bypass config file')

    args = parser.parse_args(argv)
    if args.multi_config is not None and os.path.exists(args.multi_config):
        multi_config = load_config(args.multi_config)
    else:
        multi_config = merge_dicts(DEFAULT_CONFIG, args.config)

    list_configs = traverse_config(multi_config)

    for config in list_configs:
        args = []
        args += ['--config', json.dumps(config)]
        explain_main(args)


if __name__ == '__main__':
    main()
