import os
import argparse
import ray
from ray import tune

parser = argparse.ArgumentParser()
parser.add_argument('env', type=str, help='environment name')
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--workers', type=int, default=0)
# parser.add_argument('--output-dir', default='')
# parser.add_argument('--output-max-file-size', default=5000000)


def main():
    args = parser.parse_args()
    ray.init()
    tune.run(
        'SAC',
        name='SAC-Atari',
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
            'gamma': 0.99,
            'use_state_preprocessor': True,
            'Q_model': {
                'fcnet_activation': 'relu',
                'fcnet_hiddens': [512],
            },
            'policy_model': {
                'fcnet_activation': 'relu',
                'fcnet_hiddens': [512],
            },
            'tau': 1.0,
            'target_network_update_freq': 8000,
            'target_entropy': 'auto',
            'clip_rewards': 1.0,
            'no_done_at_end': False,
            'n_step': 1,
            'rollout_fragment_length': 1,
            'prioritized_replay': True,
            'train_batch_size': 64,
            'timesteps_per_iteration': 4,
            'learning_starts': 100000,
            'optimization': {
                'actor_learning_rate': 0.0003,
                'critic_learning_rate': 0.0003,
                'entropy_learning_rate': 0.0003,
            },
            'metrics_smoothing_episodes': 5,
            'use_pytorch': True
        },
    )


if __name__ == '__main__':
    main()
