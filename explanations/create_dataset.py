import pickle
import uuid

from ray.rllib.rollout import create_parser

from envs import register
from explanations.data import Data, PolicyInfo
from explanations.rollout import run

register()


# Example usage
# python explanations/create_dataset.py
# /home/olivia/Documents/XRL/cfrl-rllib/results/tempName/SAC_Pong-ram-v0_0_2020-08-01_12-27-152bzcg1rj/checkpoint_210/checkpoint-210
# --run SAC --env Pong-ram-v0 --out saved_dataset/testing.pkl --episodes 2 --save-info

def create_dataset(policy_config):
    with open(args.out, "rb") as f:
        time_steps = []
        observations = []
        actions = []
        image_observations = []
        rewards = []
        dones = []
        env_infos = []
        trajectories = []

        data = pickle.load(f)
        # I'm not sure why, but everything's wrapped in a list
        print(f"Saving {len(data)} trajectories")
        for trajectory_id, trajectory in enumerate(data):
            for time_step_id, timestep in enumerate(trajectory):
                # Traj is a list with either 5 or 6 elements, depending on whether env info was saved or not.
                if len(timestep) == 5:
                    obs, act, img_obs, rew, done = timestep
                    env_info = {}
                else:
                    obs, act, img_obs, rew, done, env_info = timestep
                observations.append(obs)
                actions.append(act)
                image_observations.append(img_obs)
                rewards.append(rew)
                dones.append(done)
                env_infos.append(env_info)
                trajectories.append(trajectory_id)
                time_steps.append(time_step_id)
    policy_dict = {
        "env": args.env,
        "config": policy_config,
        "run": args.run,
        "checkpoint": args.checkpoint,
    }
    policy_info = PolicyInfo(id=uuid.uuid1(), policy_info=policy_dict)

    dataset = Data(
        all_time_steps=time_steps,
        all_trajectories=trajectories,
        all_observations=observations,
        all_image_observations=image_observations,
        all_actions=actions,
        all_rewards=rewards,
        all_dones=dones,
        policy=policy_info,
    )
    with open(args.out, "wb") as f:
        pickle.dump(dataset, f)


if __name__ == "__main__":
    # Load arguments
    parser = create_parser()
    args = parser.parse_args()

    # Collect Rollouts
    policy, policy_config = run(args, parser)

    # Save them in a dataset
    create_dataset(policy_config)
