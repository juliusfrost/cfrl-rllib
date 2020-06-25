import os
import argparse
import ray
from ray import tune

parser = argparse.ArgumentParser()
parser.add_argument('env', type=str, help='environment name')
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--workers', type=int, default=10)
# parser.add_argument('--output-dir', default='')
# parser.add_argument('--output-max-file-size', default=5000000)


def main():
    args = parser.parse_args()
    ray.init()
    tune.run(
        'PPO',
        name='PPO-Atari',
        local_dir='./results',
        checkpoint_freq=1000,
        checkpoint_at_end=True,
        checkpoint_score_attr='episode_reward_mean',
        keep_checkpoints_num=50,
        global_checkpoint_period=int(60 * 60 * 8),
        stop={"time_total_s": int(60 * 60 * 47.5)},
        config={
            'env': args.env,
            'num_gpus': args.gpus,
            'num_workers': args.workers,
            # 'output': os.path.join(args.output_dir, args.env),
            # 'output_max_file_size': args.output_max_file_size,
            'lambda': 0.95,
            'kl_coeff': 0.5,
            'clip_rewards': True,
            'clip_param': 0.1,
            'vf_clip_param': 10.0,
            'entropy_coeff': 0.01,
            'train_batch_size': 5000,
            'rollout_fragment_length': 100,
            'sgd_minibatch_size': 500,
            'num_sgd_iter': 10,
            'num_envs_per_worker': 5,
            'batch_mode': 'truncate_episodes',
            'observation_filter': 'NoFilter',
            'vf_share_layers': True,
            'use_pytorch': True
        },
    )


if __name__ == '__main__':
    main()
