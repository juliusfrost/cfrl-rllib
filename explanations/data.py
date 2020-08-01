"""
The Data Framework

There is a lot to detail here.
TODO: Implement
"""
import numpy as np


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
                 all_time_steps=[],
                 all_policy_infos=[],,
                 all_trajectories=[],
                 all_observations=[],
                 all_actions=[],
                 all_rewards=[],
                 all_dones=[],
                 all_policy_states=[],
                 all_simulator_states=[],
                 ):
        self.all_time_steps = np.array(all_time_steps)
        self.all_policy_infos = np.array(all_policy_infos)
        self.all_trajectories = np.array(all_trajectories)
        self.all_observations = np.array(all_observations)
        self.all_actions = np.array(all_actions)
        self.all_rewards = np.array(all_rewards)
        self.all_dones = np.array(all_dones)
        self.all_policy_states = np.array(all_policy_states)
        self.all_simulator_states = np.array(all_simulator_states)

    def add_trajectory(self):
        pass

    def add_timestep(self, observation, action, reward, done, time_step, policy_info=None, ):
        pass

    def get_trajectory(self, trajectory_id):
        pass

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
    def __init__(self, data, trajectory_id):
        self.data = data
        self.trajectory_id = trajectory_id

    @property
    def policy_info(self):
        return self.data.all_policy_infos[self.policy_id]

    def get_time_steps(self):
        pass


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
    def __init__(self, data, time_step_id, trajectory_id): # TODO: should we be interacting with the trajectory more?
        self.data = data
        self.time_step_id = time_step_id
        self.trajectory_id = trajectory_id

    @property
    def observation(self):
        return self.data.all_observations[self.time_step_id]

    @property
    def action(self):
        return self.data.all_actions[self.time_step_id]

    @property
    def reward(self):
        return self.data.all_rewards[self.time_step_id]

    @property
    def next_observation(self):
        return self.data.all_observations[self.time_step_id + 1]  # TODO: handle case where we're on the last timestep

    @property
    def prev_action(self):
        return self.data.all_actions[self.time_step_id - 1]  # TODO: handle case where we're on the first timestep

    @property
    def prev_reward(self):
        return self.data.all_rewards[self.time_step_id - 1]  # TODO: handle case where we're on the first timestep

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
    pass
