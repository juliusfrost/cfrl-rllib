import os

import gym
import numpy as np
from PIL import Image
from gym import spaces

from envs.driving_env.ple import PLE


# from scipy.misc import imresize


def state_preproc(state):
    ret = sorted(state.items(), key=lambda x: x[0])
    return np.array([x[1] for x in ret])


def state_preproc_reverse(env, state_vec):
    state_dict = {}
    curr_state = env.game_state.game.getGameState()
    for i, k in enumerate(sorted(curr_state.keys())):
        state_dict[k] = state_vec[i]
    return state_dict


def add_to_sampler_dict(sampler_dict, env, state_vec):
    state_dict = state_preproc_reverse(env, state_vec)
    for k_tup in sampler_dict.keys():
        sampler_dict[k_tup].add(tuple([state_dict[k] for k in k_tup]))


def img_preproc(img, new_width, new_height):
    # RGB -> luminance
    print("IMG SHAPE", img.shape)
    img = img[:, :, 0] * 0.2126 + img[:, :, 1] * 0.7152 + img[:, :, 2] * 0.0722
    img = img.astype(np.uint8)
    # Rescale to new_width x new_height
    img = Image.fromarray(img)
    img = np.array(img.imresize((new_height, new_width), Image.BILINEAR))
    # img = imresize(img, (new_height, new_width), interp="bilinear")
    return img


class PLEEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, game_name='FlappyBird', display_screen=True, img_input=False, continuous_actions=True,
                 num_steps=1, obs_width=84, obs_height=84, n_frames=4, **kwargs):
        # img_input - if False, uses game state as input
        # continuous_actions - must be True, to be compatible with soft actor-critic
        # n_frames - number of frames in input
        # set headless mode
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        self.img_input = img_input
        self.obs_width = obs_width
        self.obs_height = obs_height
        self.n_frames = n_frames
        self.img_preproc_fn = lambda x: img_preproc(x, obs_width, obs_height)

        # open up a game state to communicate with emulator
        import importlib
        game_module_name = ('envs.driving_env.%s' % game_name).lower()
        game_module = importlib.import_module(game_module_name)
        game = getattr(game_module, game_name)(continuous_actions=continuous_actions, **kwargs)

        self.game_state = PLE(game, fps=30, display_screen=display_screen, state_preprocessor=state_preproc,
                              num_steps=num_steps)
        self.game_state.init()

        if not continuous_actions:
            self._action_set = self.game_state.getActionSet()
            self.action_space = spaces.Discrete(len(self._action_set))
        else:
            high = np.ones(self.game_state.game.action_dim)
            low = -high
            self.action_space = spaces.Box(low=low, high=high)

        self.screen_width, self.screen_height = self.game_state.getScreenDims()
        if self.img_input:
            self.observation_space = spaces.Box(low=0., high=1., shape=(self.obs_width, self.obs_height, n_frames))
        else:
            state = self.game_state.getGameState()
            high = np.inf * np.ones(len(state))
            low = -high
            self.observation_space = spaces.Box(low=low, high=high)
        self.viewer = None
        self.continuous_actions = continuous_actions

    def step(self, a):
        if not self.continuous_actions:
            reward = self.game_state.act(self._action_set[a])
        else:
            reward = self.game_state.act(a)
        state = self._get_image(img=self.img_input)
        self.frames = state
        if self.img_input:
            state = self.img_preproc_fn(state)
            self.frames = np.c_[self.frames[:, :, 1:], state[:, :, np.newaxis]]
        terminal = self.game_state.game_over()
        return np.array(self.frames), reward, terminal, {}

    def _get_image(self, img=True):
        if img:
            image_rotated = np.fliplr(np.rot90(self.game_state.getScreenRGB(), 3))
            return image_rotated
        return self.game_state.getGameState()

    @property
    def _n_actions(self):
        if not self.continuous_actions:
            return len(self._action_set)
        else:
            return self.game_state.game.action_dim

    # return: (states, observations)
    def reset(self):
        self.game_state.reset_game()
        state = self._get_image(img=self.img_input)
        self.frames = state
        if self.img_input:
            state = self.img_preproc_fn(state)
            self.frames = np.repeat(state[:, :, np.newaxis], self.n_frames, axis=2)
        return np.array(self.frames)  # .flatten()

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        img = self._get_image(img=True)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control.rendering import SimpleImageViewer
            if self.viewer is None:
                self.viewer = SimpleImageViewer()
            self.viewer.imshow(img)

    def seed(self, seed):
        rng = np.random.RandomState(seed)
        self.game_state.rng = rng
        self.game_state.game.rng = self.game_state.rng

        self.game_state.init()

    def set_state(self, state):
        # state - vector of state feature values
        state_dict = state_preproc_reverse(self, state)
        self.game_state.game.setGameState(state_dict)

    def save_image(self, fname):
        # self.game_state.saveScreen(fname)
        img = self._get_image(img=True)
        frame = Image.fromarray(img)
        frame.save(fname)

    def get_state_sampler(self, states):
        # Samples states by mix-and-match of features from the encountered states
        # (sort of like a convex hull, but not really b/c there's no interpolation)
        sampler_dict = {}
        for k in self.game_state.game.getSamplerKeys():
            sampler_dict[k] = set()
        for state in states:
            add_to_sampler_dict(sampler_dict, self, state)

        return StateSampler(sampler_dict, rng=self.game_state.rng)


class StateSampler(object):

    def __init__(self, sampler_dict, seed=24, rng=None):
        self.sampler_dict = sampler_dict
        for k_tup in self.sampler_dict.keys():
            self.sampler_dict[k_tup] = list(self.sampler_dict[k_tup])
        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.RandomState(seed)

    def sample(self, n=1):
        states = []
        for _ in range(n):
            state_dict = {}
            for k_tup in sorted(self.sampler_dict.keys()):
                idx = self.rng.randint(0, len(self.sampler_dict[k_tup]))
                vals = self.sampler_dict[k_tup][idx]
                for i, k in enumerate(k_tup):
                    state_dict[k] = vals[i]
            states.append(state_preproc(state_dict))
        return states
