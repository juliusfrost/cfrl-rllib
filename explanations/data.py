"""
The Data Framework
"""
import numpy as np
import ray
from ray.tune.registry import get_trainable_cls


class Data:
    """
    should have:

    environment
    simulator

    The following should be in a data structure that takes up the least memory (eg. numpy array):

    all_time_steps
    all_policy_infos
    all_trajectories
    all_observations
    all_actions
    all_rewards
    all_dones
    all_policy_states
    all_simulator_states
    """

    def __init__(self,
                 all_time_steps=(),
                 all_policy_infos=(),
                 all_trajectories=(),
                 all_observations=(),
                 all_image_observations=(),  # This can be blank if observations are images
                 all_actions=(),
                 all_rewards=(),
                 all_dones=(),
                 all_policy_states=(),
                 all_simulator_states=(),
                 policy=None,
                 # TODO: it's probably better to have an index for each timestep in case diff policies were used in the same dataset
                 ):
        self.all_time_steps = np.array(all_time_steps)
        self.all_policy_infos = np.array(all_policy_infos)
        self.all_trajectories = np.array(all_trajectories)  # Trajectory Id for each timestep
        self.all_observations = np.array(all_observations)
        self.all_image_observations = np.array(all_image_observations)
        self.all_actions = np.array(all_actions)
        self.all_rewards = np.array(all_rewards)
        self.all_dones = np.array(all_dones)
        self.all_policy_states = np.array(all_policy_states)
        self.all_simulator_states = np.array(all_simulator_states)
        self.all_trajectory_ids = np.unique(self.all_trajectories)
        self.policy = policy

    def get_trajectory(self, trajectory_id):
        timesteps = [timestep for timestep, traj in enumerate(self.all_trajectories) if traj == trajectory_id]
        return Trajectory(self, trajectory_id, timesteps[0], timesteps[-1] + 1)

    def get_timestep(self, time_step_id):
        trajectory_id = self.all_trajectories[time_step_id]
        return TimeStep(self, time_step_id, trajectory_id)


class Trajectory:
    """
    Trajectories store the ranges of time steps
    These exist as an intermediary between time steps and data.
    They hold episodic information, as well as segmented trajectory information (such as counterfactuals).
    From a Trajectory one can infer all TimeStep and PolicyData pertaining to it
    They are segmented by different policies (counterfactuals) and episode beginning and end

    Should have:

    trajectory_id: A unique identifier
    episode_id
    policy_id
    observation_range
    action_range
    reward_range
    done_range
    next_observation_range
    prev_action_range
    prev_reward_range
    policy_state_range
    simulation_state_range
    """

    # TODO: add in policy ID and stuff
    def __init__(self, data, trajectory_id, range_start, range_end):
        self.data = data
        self.trajectory_id = trajectory_id
        self.timestep_range_start = range_start
        self.timestep_range_end = range_end

    @property
    def policy_info(self) -> list:
        return self.data.all_policy_infos[self.policy_id]

    def get_time_steps(self) -> list:
        timestep_range = range(self.timestep_range_start, self.timestep_range_end)
        return [TimeStep(self.data, t, self.trajectory_id) for t in timestep_range]

    @property
    def observation_range(self) -> np.ndarray:
        return self.data.all_observations[self.timestep_range_start:self.timestep_range_end]

    @property
    def image_observation_range(self) -> np.ndarray:
        if len(self.data.all_image_observations) == 0:
            arr = self.data.all_observations
        else:
            arr = self.data.all_image_observations
        return arr[self.timestep_range_start:self.timestep_range_end]

    @property
    def action_range(self) -> np.ndarray:
        return self.data.all_actions[self.timestep_range_start:self.timestep_range_end]

    @property
    def reward_range(self) -> np.ndarray:
        return self.data.all_rewards[self.timestep_range_start:self.timestep_range_end]

    @property
    def next_observation_range(self) -> np.ndarray:  # TODO: deal with edge cases
        return self.data.all_observations[self.timestep_range_start + 1:self.timestep_range_end + 1]

    @property
    def prev_action_range(self) -> np.ndarray:  # TODO: deal with edge cases
        return self.data.all_actions[self.timestep_range_start - 1:self.timestep_range_end - 1]

    @property
    def prev_reward_range(self) -> np.ndarray:  # TODO: deal with edge cases
        return self.data.all_rewards[self.timestep_range_start - 1:self.timestep_range_end - 1]

    @property
    def done_range(self) -> np.ndarray:
        return self.data.all_dones[self.timestep_range_start:self.timestep_range_end]

    @property
    def info_range(self) -> list:
        return self.data.all_policy_infos[self.timestep_range_start:self.timestep_range_end]

    @property
    def policy_state_range(self) -> list:
        return self.data.all_policy_states[self.timestep_range_start:self.timestep_range_end]

    @property
    def simulator_state_range(self) -> list:
        return self.data.all_simulator_states[self.timestep_range_start:self.timestep_range_end]


class TimeStep:
    """
    Each time step should have the following.
    These are created by the data class when relevant, and only at the last possible moment to avoid hogging memory.
    They do not store data themselves but point to the location in data.

    data: reference to the data
    time_step_id: A unique identifier
    trajectory_id
    observation
    action
    reward
    next_observation: points to next time step observation
    prev_action: points to previous time step action
    prev_reward: points to previous time step reward
    done
    info
    policy_state: points to the hidden state of the policy, typically the rnn state
    simulator_state: points to the simulator state

    for an example see self.observation below
    we use the @property keyword that only references the data when called and does not explicitly store anything
    """

    def __init__(self, data, time_step_id,
                 trajectory_id):  # TODO: should we be interacting with the trajectory more? Seems weird we can get next_ob from a different trajectory
        self.data = data
        self.time_step_id = time_step_id
        self.trajectory_id = trajectory_id

    @property
    def observation(self):
        return self.data.all_observations[self.time_step_id]

    @property
    def image_observation(self):
        if len(self.data.all_image_observations) == 0:
            arr = self.data.all_observations
        else:
            arr = self.data.all_image_observations
        return arr[self.time_step_id]

    @property
    def action(self):
        return self.data.all_actions[self.time_step_id]

    @property
    def reward(self):
        return self.data.all_rewards[self.time_step_id]

    @property
    def next_observation(self):
        return self.data.all_observations[(self.time_step_id + 1) % len(self.data.all_observations)]

    @property
    def prev_action(self):
        return self.data.all_actions[(self.time_step_id - 1) % len(self.data.all_actions)]

    @property
    def prev_reward(self):
        return self.data.all_rewards[(self.time_step_id - 1) % len(self.data.all_rewards)]

    @property
    def done(self):
        return self.data.all_dones[self.time_step_id]

    @property
    def info(self):
        return self.data.all_policy_infos[self.time_step_id]

    @property
    def policy_state(self):
        return self.data.all_policy_states[self.time_step_id]

    @property
    def simulator_state(self):
        return self.data.all_simulator_states[self.time_step_id]


class PolicyInfo:
    """

    policy_id: A unique identifier so multiple policies don't get mixed up
    policy_initialization: information about how to initialize the policy, from weights and hidden_state
    policy_weights: pointer to policy weights
    """

    def __init__(self, id, policy_info):
        self.id = id
        self.policy_info = policy_info

    def get_policy(self):
        agent = self.get_agent()
        return agent.get_policy()

    def get_agent(self):
        ray.init()
        run = self.policy_info['run']
        env = self.policy_info['env']
        config = self.policy_info['config']
        checkpoint = self.policy_info['checkpoint']
        cls = get_trainable_cls(run)
        agent = cls(env=env, config=config)
        # Load state from checkpoint.
        agent.restore(checkpoint)
        return agent
