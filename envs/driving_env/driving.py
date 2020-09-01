import math
import numpy as np
import os, os.path as osp

import pygame
import gym
from pygame.constants import K_w, K_s, K_a, K_d

from envs.driving_env.utils import percent_round_int
from envs.driving_env.pygamewrapper import PyGameWrapper
from envs.driving_env.driving_ft import get_state_ft, get_reward_ft, out_of_bound, collision_exists, get_car_states_from_ft, get_n_cpu_cars_from_ft
from envs.driving_env.driving_car import Car, Backdrop

import random

FILEDIR = osp.dirname(osp.realpath(__file__))


def init_robot_car(img, state, speed_limit, speed_min, controls=[(0, 0)],
                   ydiff=0, **kwargs):
    width = kwargs["car_width"]
    height = kwargs["car_height"]
    return Car(img, state, width, height, speed_limit, speed_min,
               controls=controls, stationary=True, ydiff=ydiff, **kwargs)


def init_nonrobot_car(img, state, speed_limit, speed_min, controls=[(0, 0)], **kwargs):
    width = kwargs["car_width"]
    height = kwargs["car_height"]
    if "replace_img" in kwargs:
        img = kwargs["replace_img"]

    return Car(img, state, width, height, speed_limit, speed_min,
               controls=controls, stationary=False, **kwargs)


def robot_car_from_state(game, car_state, **kwargs):
    kwargs.update(game.constants)
    return init_robot_car(game.images["agent"], car_state,
                          game.players_speed_ratio * game.height,
                          game.cpu_speed_ratio_max * game.height,
                          **kwargs)


def nonrobot_car_from_state(game, car_state, **kwargs):
    kwargs.update(game.constants)
    return init_nonrobot_car(game.images["cpu"], car_state,
                             game.cpu_speed_ratio_max * game.height,
                             game.cpu_speed_ratio_min * game.height,
                             control=[(0, 0)], **kwargs)


def load_images(img_dir=osp.join(FILEDIR, "img"),
                agent_img="robot_car.png", cpu_img="car.png"):
    # Loads all images required by the game, and returns a dictionary of them

    def load_image(img_file_name):
        file_name = osp.join(img_dir, img_file_name)
        # converting all images before use speeds up blitting
        img = pygame.image.load(file_name).convert_alpha()
        return img

    return {"cpu": load_image(cpu_img),
            "agent": load_image(agent_img)}


def add_car_top(dt, robot_car, cars, img, speed_limit, speed_min, ydiff, rng, **kwargs):
    # If last-added car is more than 1 car length down the road, add new car
    # with 1% probability at each time step
    lane_centers = kwargs["lane_centers"]
    img_height = kwargs["screen_height"]
    car_height = kwargs["car_height"]
    heading = kwargs["default_heading"]
    speed = kwargs["default_speed_ratio_cpu"] * img_height
    prob_car = 0.8
    if "prob_car" in kwargs:
        prob_car = kwargs["prob_car"]
        if prob_car > 1:
            prob_car /= 100.  # turn from percentage into decimal

    if len(cars) > 0 and np.min([c.y - ydiff for c in cars]) < car_height:
        return None

    chance_appear = 1 - (1 - prob_car) ** dt  # Insert a new car with this probability
    if rng.random_sample() > chance_appear:
        return None

    lane_idx = rng.randint(len(lane_centers))
    car = init_nonrobot_car(img,
                            [lane_centers[lane_idx], -(car_height / 2) + ydiff, heading, speed],
                            speed_limit, speed_min, controls=[(0, 0)], **kwargs)
    return car


class Driving(PyGameWrapper, gym.Env):
    """
    Adapted from criticalstates repo (https://github.com/shhuang/criticalstates(
    under criticalstates/envs/driving_env, to work with PLE setup.

    Parameters
    ----------
    width: int
        Screen width

    height: int
        Screen height, recommended to be 1.5x width

    cpu_speed_ratio_max: float
        Maximum speed of other cars (useful for curriculum learning)

    cpu_speed_ratio_min: float
        Minimum speed of other cars

    players_speed_ratio: float
        Speed of player (useful for curriculum learning)

    """

    def __init__(self, width=640, height=960, cpu_speed_ratio_max=0.1,
                 cpu_speed_ratio_min=0.05, players_speed_ratio=0.2,
                 continuous_actions=True,
                 theta=np.array([-1., 0., -10., -1., 1., -0.01]),
                 # theta=np.array([-1.,0.,-10.,-1.,1.,-0.1]),\
                 # theta=np.array([-1.,1.,-10.,-1.,1.]),\
                 MAX_SCORE=1,
                 state_ft_fn=get_state_ft, reward_ft_fn=get_reward_ft,
                 add_car_fn=add_car_top, COLLISION_PENALTY=-100., n_noops=230,
                 default_dt=30, prob_car=50, time_reward=True, time_limit=500, **kwargs):

        assert continuous_actions, "Driving simulator can only handle continuous actions"
        self.continuous_actions = continuous_actions
        actions = {}
        PyGameWrapper.__init__(self, width, height, actions=actions)

        self.n_noops = n_noops
        self.default_dt = default_dt
        self.MAX_SCORE = MAX_SCORE  # Maximum number of crashes allowed

        self.cpu_speed_ratio_max = cpu_speed_ratio_max
        self.cpu_speed_ratio_min = cpu_speed_ratio_min
        self.players_speed_ratio = players_speed_ratio
        self.get_game_state_fn = state_ft_fn
        self.get_reward_ft_fn = reward_ft_fn
        self.add_car_fn = add_car_fn
        self.theta = theta
        self.time_reward = time_reward
        self.time_limit = time_limit

        self.images = None  # Load in init(), after game screen is created
        self.backdrop = None  # Load in init()
        self.agent_img = "robot_car.png"
        if "agent_img" in kwargs:
            self.agent_img = kwargs["agent_img"]

        self.switch_prob = kwargs.get('switch_prob', 0.0)

        # Define environment visualization constants
        n_lanes = 3
        lane_width = percent_round_int(width, 0.125)
        car_width = percent_round_int(width, 0.0625)
        car_height = 2 * car_width
        border_width = percent_round_int(lane_width, 0.05)
        grass_width = percent_round_int(
            width - n_lanes * lane_width - (n_lanes + 1) * border_width, 0.5)
        lane_mark_length = percent_round_int(lane_width, 0.25)
        lane_mark_btwn_length = percent_round_int(lane_mark_length, 0.5)
        lane_center_first = grass_width + border_width + int(lane_width / 2)
        lane_centers = np.array([lane_center_first + i * (border_width + lane_width)
                                 for i in range(n_lanes)])
        road_bounds = np.array([grass_width + int(border_width / 2),
                                width - grass_width - int(border_width / 2)])

        self.constants = {}
        for k in ["n_lanes", "lane_width", "car_width", "car_height",
                  "border_width", "grass_width", "lane_mark_length",
                  "lane_mark_btwn_length", "lane_center_first",
                  "lane_centers", "road_bounds"]:
            self.constants[k] = locals()[k]
        self.constants["screen_width"] = width
        self.constants["screen_height"] = height
        self.constants["default_heading"] = math.pi / 2
        self.constants["default_speed_ratio_agent"] = self.players_speed_ratio
        self.constants["default_speed_ratio_cpu"] = self.cpu_speed_ratio_max
        self.constants["speed_max"] = 0.4 * height  # for normalizing features
        self.constants["heading_max"] = 2 / 3. * math.pi  # for normalizing features
        self.constants["heading_min"] = 1 / 3. * math.pi
        # The "- car_width" is because car can't be partially off the road
        # self.constants["drivable_width"] = road_bounds[1] - road_bounds[0] - car_width
        self.constants["drivable_width"] = road_bounds[1] - road_bounds[0]
        # The "+ car_height" is because car can be partially off-screen
        self.constants["drivable_height"] = height + car_height
        self.constants["prob_car"] = prob_car

        self.score_sum = 0.  # Total (non-discounted) reward
        self.n_crashes = 0  # Number of crashes

        # self.action_dim = 2  # [turning, acceleration], each between +1 and -1
        self.action_dim = 1  # [turning], between +1 and -1
        if 'action_dim' in kwargs:
            self.action_dim = kwargs['action_dim']
        self.full_action_dim = 2  # [turning, acceleration]
        self.noop = np.array([0.0, 0.0])
        self.action_to_take = np.array(self.noop)
        self.COLLISION_PENALTY = COLLISION_PENALTY
        self.time_steps = 0

    def _handle_player_events(self):
        self.dy = 0

        if __name__ == "__main__":
            # for debugging mode
            pygame.event.get()
            keys = pygame.key.get_pressed()
            self.action_to_take = np.array(self.noop)
            if keys[K_w]:
                self.action_to_take[0] = 1.
            elif keys[K_s]:
                self.action_to_take[0] = -1.
            elif keys[K_a]:
                self.action_to_take[1] = -1.
            elif keys[K_d]:
                self.action_to_take[1] = 1.
            if keys[pygame.QUIT]:
                pygame.quit()
                sys.exit()
            pygame.event.pump()
        else:
            # consume events from act
            assert len(self.action_to_take) == self.full_action_dim
            for i in range(len(self.action_to_take)):
                self.action_to_take[i] = min(1., max(-1., self.action_to_take[i]))

    def _setAction(self, action, last_action):
        # print("\taction:", action)
        self.action_to_take = np.array(self.noop)
        if action is not None:
            if len(action) == self.full_action_dim:
                self.action_to_take = np.array(action)
            else:
                self.action_to_take[0] = action[0]

    def getGameState(self):
        """
        Gets a non-visual state representation of the game, in the form of a
        dictionary mapping feature name to feature value.
        """
        return self.get_game_state_fn(self.agent_car, self.cpu_cars, **self.constants)

    def setGameState(self, state):
        # NOTE: This hasn't been thoroughly tested
        assert self.images is not None and self.backdrop is not None
        self.score_sum = 0.0  # reset cumulative reward
        self.n_crashes = 0

        robot_state, cpu_states = get_car_states_from_ft(state, **self.constants)
        self.agent_car = robot_car_from_state(self, robot_state)
        self.cars_group = pygame.sprite.Group()
        self.cars_group.add(self.agent_car)
        self.cpu_cars = []
        for cpu_state in cpu_states:
            car = nonrobot_car_from_state(self, cpu_state)
            # Ignore car if it's out of bound
            if out_of_bound(car, self.ydiff, **self.constants):
                continue
            # Snap car to center of lane
            lane_centers = self.constants["lane_centers"]
            lane_idx = np.argmin(np.abs(lane_centers - car.state[0]))
            car.x = lane_centers[lane_idx]

            self.cpu_cars.append(car)
            self.cars_group.add(car)

    def getScore(self):
        return self.score_sum

    @property
    def ydiff(self):
        return self.agent_car.ydiff

    def crashed(self):
        is_out_of_bound = out_of_bound(self.agent_car, self.ydiff, **self.constants)
        is_collision = collision_exists(self.agent_car, self.cpu_cars, **self.constants)
        # car must face forward
        # heading_out_of_range = self.agent_car.heading < self.constants["heading_min"] \
        #                       or self.agent_car.heading > self.constants["heading_max"]
        # if is_out_of_bound:
        #    print("WENT OUT OF BOUNDS", self.ydiff)
        # if is_collision:
        #    print("COLLIDED", self.ydiff)
        # if heading_out_of_range:
        #    print("HEADING OUT OF RANGE")
        return is_out_of_bound or is_collision  # or heading_out_of_range

    def game_over(self):
        # Game is over if the player crashes
        return self.crashed() or self.time_steps >= self.time_limit
        # return (self.n_crashes >= self.MAX_SCORE)

    def init(self):
        if self.images is None:
            self.images = load_images(agent_img=self.agent_img)

        if self.backdrop is None:
            self.backdrop = Backdrop(**self.constants)

        self.score_sum = 0.0
        robot_init_state = [self.width / 2., self.height - 200,
                            self.constants["default_heading"],
                            self.constants["default_speed_ratio_agent"] * self.height]
        self.agent_car = init_robot_car(self.images["agent"], robot_init_state,
                                        self.players_speed_ratio * self.height,
                                        self.cpu_speed_ratio_max * self.height,
                                        **self.constants)
        self.cpu_cars = []
        self.cars_group = pygame.sprite.Group()
        self.cars_group.add(self.agent_car)

        self.action_to_take = np.array(self.noop)
        for _ in range(self.n_noops):  # To get to more interesting position in game
            self.step(self.default_dt, add_reward=False)
        self.time_steps = 0

    def reset(self):
        self.init()

    def step(self, dt, add_reward=True):
        dt /= 1000.0
        self.screen.fill((0, 0, 0))

        self._handle_player_events()

        self.agent_car.update(control=self.action_to_take, dt=dt)

        new_car = self.add_car_fn(dt, self.agent_car, self.cpu_cars,
                                  self.images["cpu"],
                                  self.cpu_speed_ratio_max * self.height,
                                  self.cpu_speed_ratio_min * self.height,
                                  self.ydiff, self.rng, **self.constants)
        if new_car is not None:
            self.cpu_cars.append(new_car)
            self.cars_group.add(new_car)

        for car in self.cpu_cars:
            should_switch = not car.switch_duration > 0 and random.random() < self.switch_prob
            if should_switch:
                car.start_switch_lane(**self.constants)
            car.update(ydiff=self.ydiff, dt=dt)
            if out_of_bound(car, self.ydiff, **self.constants):
                self.cpu_cars.remove(car)
                self.cars_group.remove(car)

        if self.crashed():
            self.score_sum += self.COLLISION_PENALTY
            self.n_crashes += 1
            # print("# crashes:", self.n_crashes)
            if not self.n_crashes >= self.MAX_SCORE:
                self.reset()
        elif add_reward:
            if self.time_reward:
                self.score_sum += 1
            else:
                reward_ft = self.get_reward_ft_fn(self.agent_car, self.cpu_cars,
                                                  self.action_to_take,
                                                  **self.constants)
                self.score_sum += np.dot(self.theta, reward_ft)

        self.backdrop.update(self.ydiff)
        self.backdrop.draw_background(self.screen)
        self.cars_group.draw(self.screen)
        self.time_steps += 1

    def getSamplerKeys(self):
        """
        Returns tuples of keys for sampling states, based on state features
        """
        state = self.getGameState()
        n_cars = get_n_cpu_cars_from_ft(state)
        lane_fts = sorted([x for x in state.keys() if x.startswith("in_lane")])
        speed_fts = sorted([x for x in state.keys() if x.startswith("in_speed")])
        return [tuple(["agent_x", "agent_y"] + lane_fts),
                ("agent_h",),
                tuple(["agent_v"] + speed_fts)] + \
               [("cpu" + str(i + 1) + "_x", "cpu" + str(i + 1) + "_y", "cpu" + str(i + 1) + "_h",
                 "cpu" + str(i + 1) + "_v", "cpu" + str(i + 1) + "_dummy") for i in range(n_cars)]


if __name__ == "__main__":
    pygame.init()
    game = Driving(width=640, height=960)
    game.screen = pygame.display.set_mode(game.getScreenDims(), 0, 32)
    game.clock = pygame.time.Clock()
    game.rng = np.random.RandomState(24)
    game.init()

    while True:
        dt = game.clock.tick_busy_loop(60)
        game.step(dt)
        pygame.display.update()
