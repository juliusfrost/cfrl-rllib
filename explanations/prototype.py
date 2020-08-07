import argparse
import copy
import os
import pickle
import random
import shutil
import subprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=str, help='path to the checkpoint file')
    args = parser.parse_args()

    possible_inits = []
    for x in [1, 2, 3, 5, 6, 7]:
        for y in [1, 2, 3, 5, 6, 7]:
            if not (x == 2 and y == 6):  # don't set the agent on top of the goal
                possible_inits.append([x, y])

    random.shuffle(possible_inits)

    # Load configuration from checkpoint file.
    config_dir = os.path.dirname(args.checkpoint)
    config_path = os.path.join(config_dir, "params.pkl")
    # Try parent directory.
    if not os.path.exists(config_path):
        config_path = os.path.join(config_dir, "../params.pkl")
    with open(config_path, "rb") as f:
        config = pickle.load(f)
        assert isinstance(config, dict)
    original_config = copy.deepcopy(config)

    for i, init in enumerate(possible_inits):
        if 'agent_pos' in config.get('evaluation_config', {}).get('env_config', {}):
            config['evaluation_config']['env_config']['agent_pos'] = init
        elif 'agent_pos' in config.get('env_config', {}):
            config['env_config']['agent_pos'] = init
        with open(config_path, 'wb') as f:
            pickle.dump(config, f)

        data_file = f'testing_{i}.pkl'
        command = f'python create_dataset.py ' \
                  f'--run A2C ' \
                  f'{args.checkpoint} ' \
                  f'--out {data_file} ' \
                  f'--episodes 1 ' \
                  f'--save-info'
        subprocess.run(command.split())

        video_dir = f'videos/agent-pos-{init[0]}-{init[1]}'
        command = f'python view_trajectory.py ' \
                  f'--dataset-file {data_file} ' \
                  f'--save-trajectories ' \
                  f'--save-path {video_dir}'
        subprocess.run(command.split())

        shutil.move(f'{video_dir}/0.avi', f'videos/oracle-{init[0]}-{init[1]}.avi')
        os.rmdir(video_dir)
        os.remove(data_file)

    # restore the original config
    with open(config_path, 'wb') as f:
        pickle.dump(original_config, f)
