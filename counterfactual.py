"""
Counterfactuals state and counterfactual trajectory
TODO: Implement
"""


def counterfactual_state(data, num_states, simulator, state_selection_method, action_selection_method, **kwargs):
    """
    Unlike the previous state selection methods,
    we can simulate from a starting state to get to a counterfactual state
    """
    # get starting state with state selection method
    # loop for however many steps:
    #   call action selection method
    # save new trajectory to data
    state_indices = None
    return state_indices


def counterfactual_trajectory(state_indices, num_counterfactuals, simulator, policies, time_limit=None, **kwargs):
    """
    Starting from a state, we can simulate the rest of the trajectory, aka. the counterfactual trajectory
    :returns the sequence of states that form the trajectory (context + explanation)
    """
    trajectory_indices = None
    return trajectory_indices
