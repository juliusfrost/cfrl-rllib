"""
Action selection methods
Fill in necessary method parameters as needed
"""
import random
import numpy as np
from collections import namedtuple
from gym_minigrid.minigrid import Goal

def policy_action(observation, policy, **kwargs):
    action = policy(observation)
    return action


def user_action(observation, user, **kwargs):
    action = user(observation)
    return action


Policy = namedtuple("Policy", "action_space")


class MinigridOracleAgent:
    def __init__(self, env, *args, **kwargs):
        self.env = env
        self.goal_state = None
        self.path = self.compute_path_to_goal()
        self.policy = Policy(action_space=env.action_space)

    def compute_path_to_goal(self):
        open_states = []  # TODO: collect list of all states; we could hard-code this for now
        for i in range(self.env.grid.height):
            for j in range(self.env.grid.width):
                cell = self.env.grid.get(i, j)
                if cell is None or type(cell) is Goal:  # empty cell or goal cell
                    open_states.append((i, j))

        initial_states = [(*self.env.agent_pos, *self.env.dir_vec)]
        self.goal_state = random.choice(open_states)
        accept_fn = lambda pos, _: np.array_equal(pos, self.goal_state)
        ignore_blockers = True  # shouldn't really matter, since the env has no blockers anyway
        path, _, _ = self._breadth_first_search(initial_states, accept_fn, ignore_blockers)
        return path

    def _breadth_first_search(self, initial_states, accept_fn, ignore_blockers):
        """Performs breadth first search.

        This is pretty much your textbook BFS. The state space is agent's locations,
        but the current direction is also added to the queue to slightly prioritize
        going straight over turning.

        BFS code: https://github.com/mila-iqia/babyai

        """
        queue = [(state, None) for state in initial_states]
        grid = self.env.grid
        previous_pos = dict()

        while len(queue) > 0:
            state, prev_pos = queue[0]
            queue = queue[1:]
            i, j, di, dj = state

            if (i, j) in previous_pos:
                continue

            cell = grid.get(i, j)
            previous_pos[(i, j)] = prev_pos

            # If we reached a position satisfying the acceptance condition
            if accept_fn((i, j), cell):
                path = []
                pos = (i, j)
                while pos:
                    path.append(pos)
                    pos = previous_pos[pos]
                return path, (i, j), previous_pos

            if cell:
                if cell.type == 'wall':
                    continue
                # If this is a door
                elif cell.type == 'door':
                    # If the door is closed, don't visit neighbors
                    if not cell.is_open:
                        continue
                elif not ignore_blockers:
                    continue

            # Location to which the bot can get without turning
            # are put in the queue first
            for k, l in [(di, dj), (dj, di), (-dj, -di), (-di, -dj)]:
                next_pos = (i + k, j + l)
                next_dir_vec = (k, l)
                next_state = (*next_pos, *next_dir_vec)
                queue.append((next_state, (i, j)))

        # Path not found
        raise ValueError('Tried to get to invalid state')

    def compute_action(self, *args, **kwargs):
        # Path is empty. I don't think this ever gets called. In that case, spin randomly.
        if len(self.path) == 0:
            return random.choice([self.env.actions.left, self.env.actions.right])

        # Path is longer. We will go to the first cell on it
        next_pos = np.array(self.path[-1])
        curr_pos = np.array(self.env.agent_pos)

        # We're already at our goal. In this case, spin randomly.
        if len(self.path) == 1 and np.array_equal(next_pos, curr_pos):
            return random.choice([self.env.actions.left, self.env.actions.right])

        # If we're already on the path, we can skip a step
        while np.array_equal(next_pos, curr_pos):
            self.path.pop()
            next_pos = np.array(self.path[-1])

        # We'd better be one cell away fom the new cell
        assert np.linalg.norm(next_pos - curr_pos) == 1, (curr_pos, next_pos, self.path)

        # If the next pos is ahead, we can go forward and pop the cell from the path
        if np.array_equal(next_pos - curr_pos, self.env.dir_vec):
            self.path.pop()
            return self.env.actions.forward

        # If the next pos is to the left or right, we spin in that direction
        if np.array_equal(next_pos - curr_pos, self.env.right_vec):
            return self.env.actions.right
        elif np.array_equal(next_pos - curr_pos, -self.env.right_vec):
            return self.env.actions.left

        # Otherwise, the next pos is behind us, so we choose randomly which direction to go
        return random.choice([self.env.actions.left, self.env.actions.right])

    def get_policy(self):
        return self.policy


class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space
        self.policy = Policy(action_space=action_space)

    def compute_action(self, obs, **kwargs):
        return self.action_space.sample()

    def get_policy(self):
        return self.policy

def constant_generator(n_timesteps):
    for i in range(n_timesteps):
        yield False
    yield True


def make_handoff_func(n_timesteps):
    gen = constant_generator(n_timesteps)

    def handoff(state, action):
        # handoff based on state and action.
        # action will be null on the first timestep before we've stepped yet.
        return next(gen)

    return handoff


def until_end_handoff(state, action):
    return False
