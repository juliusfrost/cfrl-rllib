import argparse
import json
import os

import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, required=True)
    args = parser.parse_args()

    list_metrics = []
    for folder in os.listdir(args.logdir):
        if os.path.isdir(os.path.join(args.logdir, folder)):
            csv_path = os.path.join(args.logdir, folder, 'progress.csv')
            try:
                df = pd.read_csv(csv_path)
                episode_reward_mean = float(df['episode_reward_mean'].iloc[len(df) - 1])
            except:
                continue

            with open(os.path.join(args.logdir, folder, 'params.json')) as f:
                params = json.load(f)
                goal_pos = params['env_config']['goal_pos']
            list_metrics.append((episode_reward_mean, goal_pos, os.path.join(args.logdir, folder)))
    for episode_reward_mean, goal_pos, path in sorted(list_metrics, key=lambda x: x[0], reverse=True):
        print(episode_reward_mean, goal_pos, path)
