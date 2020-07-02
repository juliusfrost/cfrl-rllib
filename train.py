import argparse
import os
import pprint

import ray
import yaml
from ray import tune

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='PongNoFrameskip-v4', help='environment name')
parser.add_argument('--config', type=str, default=None,
                    help='Config file to load algorithm from. Defaults to algorithm argument choice.')
parser.add_argument('--algo', type=str, default='ppo',
                    help='Choose algorithm from those implemented. Used if config argument not set.')
parser.add_argument('--framework', choices=['torch', 'tf', 'tfe'], default='torch')
parser.add_argument('--suite', default='atari', help='used for config location')


def main():
    args = parser.parse_args()
    ray.init()

    if args.config is not None:
        algo_config = yaml.safe_load(open(args.config))
    elif os.path.exists(os.path.join('config', args.suite, args.algo + '.yaml')):
        config_file = os.path.join('config', args.suite, args.algo + '.yaml')
        algo_config = yaml.safe_load(open(config_file))
    else:
        raise FileNotFoundError(f'Config {args.config} or algorithm {args.algo} not found!')

    print('\nArguments:')
    pprint.pprint(args)
    print('\nConfig:')
    pprint.pprint(algo_config)
    print()

    config = algo_config['config']
    config.update(dict(
        framework=args.framework,
        env=args.env,
    ))

    tune.run(
        algo_config['run'],
        name=f'{args.suite}-{args.algo}',
        local_dir='./results',
        checkpoint_freq=1000,
        checkpoint_at_end=True,
        checkpoint_score_attr='episode_reward_mean',
        keep_checkpoints_num=50,
        global_checkpoint_period=int(60 * 60 * 8),
        stop={'time_total_s': int(60 * 60 * 47.5), 'timesteps_total': 50000000},
        config=config
    )


if __name__ == '__main__':
    main()
