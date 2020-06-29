import os
import argparse
import ray
from ray import tune

parser = argparse.ArgumentParser()
parser.add_argument('env', type=str, help='environment name')
parser.add_argument('--gpus', default=0.2)
parser.add_argument('--workers', type=int, default=10)
# parser.add_argument('--output-dir', default='')
# parser.add_argument('--output-max-file-size', default=5000000)


def main():
    args = parser.parse_args()
    ray.init()
    tune.run(
        'DQN',
        name='DQN-Atari',
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
            # 'num_workers': args.workers,
            # 'output': os.path.join(args.output_dir, args.env),
            # 'output_max_file_size': args.output_max_file_size,
            'double_q': False,
            'dueling': False,
            'num_atoms': 1,
            'noisy': False,
            'prioritized_replay': False,
            'n_step': 1,
            'target_network_update_freq': 8000,
            'lr': .0000625,
            'adam_epsilon': .00015,
            'hiddens': [512],
            'learning_starts': 20000,
            'buffer_size': 1000000,
            'sample_batch_size': 4,
            'train_batch_size': 32,
            'schedule_max_timesteps': 2000000,
            'exploration_final_eps': 0.01,
            'exploration_fraction': .1,
            'prioritized_replay_alpha': 0.5,
            'beta_annealing_fraction': 1.0,
            'final_prioritized_replay_beta': 1.0,
            'timesteps_per_iteration': 10000,
            'use_pytorch': True
        },
    )


if __name__ == '__main__':
    main()
