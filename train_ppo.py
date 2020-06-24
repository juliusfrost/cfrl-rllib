import os
import argparse
import ray
from ray import tune

parser = argparse.ArgumentParser()
parser.add_argument('env', type=str, help='environment name')
parser.add_argument('--gpus', type=int, default=0)
parser.add_argument('--workers', type=int, default=1)
parser.add_argument('--output-dir', default='./data')


def main():
    args = parser.parse_args()
    ray.init()
    tune.run(
        'PPO',
        # stop={"episode_reward_mean": 200},
        config={
            'env': args.env,
            'num_gpus': args.gpus,
            'num_workers': args.workers,
            'output': os.path.join(args.output_dir, args.env),
            'output_max_file_size': 5000000,
        },
    )


if __name__ == '__main__':
    main()
