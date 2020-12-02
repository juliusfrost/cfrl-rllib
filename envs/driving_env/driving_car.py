import math
import random

import numpy as np
import pygame


def standardize_heading(h):
    stand_h = h % (2 * math.pi)
    if stand_h < 0:
        stand_h += 2 * math.pi
    return stand_h


def set_alphas(color_tuple, transparency):
    color = list(color_tuple)
    color[-1] = int(color[-1] * transparency)
    return tuple(color)


class Car(pygame.sprite.Sprite):
    def __init__(self, img, initial_state, width, height, speed_limit, speed_min,
                 controls=None, alpha=0, stationary=False, ydiff=0, **kwargs):
        # alpha - friction coefficient
        # TODO: Try non-zero alpha

        pygame.sprite.Sprite.__init__(self)

        self.img = img
        self.alpha = alpha
        self.width = int(width)
        self.height = int(height)
        self.speed_limit = speed_limit
        self.speed_min = speed_min
        self.heading = standardize_heading(initial_state[2])
        self.heading_max = kwargs["heading_max"]
        self.heading_min = kwargs["heading_min"]
        lane_width = kwargs['lane_width']
        # Rotate (CCW) car by heading
        self.orig_img = pygame.transform.scale(img, [self.height, self.width])
        self.TRANSPARENCY = 1.
        if "transparency" in kwargs:
            self.TRANSPARENCY = kwargs["transparency"]
            for row in range(self.orig_img.get_height()):
                for col in range(self.orig_img.get_width()):
                    self.orig_img.set_at((col, row),
                                         set_alphas(self.orig_img.get_at((col, row)), self.TRANSPARENCY))

        self.image = pygame.transform.rotate(self.orig_img, math.degrees(self.heading))
        self.x = initial_state[0]  # x-coordinate of *center* of car
        self.y = initial_state[1]  # y-coordinate of *center* of car
        self.speed = initial_state[3]  # Pixels per frame
        self.stationary = stationary  # y-coordinate never changes (scene moves around this car)
        self.ydiff = ydiff

        self.steering_resistance = kwargs.get('steering_resistance', 100.)

        self.controls = controls
        self.control_idx = 0

        self.rect = self.image.get_rect()
        self.rect.center = (self.x, self.y - ydiff)
        self.switching = len(initial_state) > 4
        self.switch_duration = kwargs.get('switch_duration', 50)

        if self.switching:
            self.switching_direction = initial_state[-2]
            self.switch_duration_remaining = initial_state[-1]
        else:
            self.switching_direction = 0
            self.switch_duration_remaining = 0

        self.switch_step = (lane_width + 4) * self.switching_direction / self.switch_duration

        self.dummy = False

    @property
    def game_pos(self):
        x = int(round(self.x - self.width / 2))
        y = int(round(self.y - self.height / 2))
        return x, y

    @property
    def state(self):
        return np.array([self.x, self.y, self.heading, self.speed])

    def dynamics_f(self, s, u, dt):
        """
        The dynamics function
        :param s: state vector of size 4 at time t-1
            x_tm1: x coordinate at time t-1
            y_tm1: y coordinate at time t-1
            h_tm1: heading (steering angle) at time t-1
            v_tm1: speed at time t-1
        :param u: action vector of size 2
            dh_t: heading (steering angle) delta
            a_t: acceleration
        :param dt: delta time
        :return: state at time t
        """
        x_tm1, y_tm1, h_tm1, v_tm1 = s
        dh_t, a_t = u
        # u = (u[0]/100., u[1])  # otherwise turning is too sharp
        steering_delta = dh_t / self.steering_resistance
        # u = (u_0 / self.steering_resistance, u_1)  # otherwise turning is too sharp
        x = x_tm1 + v_tm1 * np.cos(2 * math.pi - h_tm1) * dt
        y = y_tm1 + v_tm1 * np.sin(2 * math.pi - h_tm1) * dt
        h = h_tm1 + v_tm1 * steering_delta * dt
        # if self.stationary:
        #    print("control:", u)
        #    print("heading changed by:", s[3]*u[0]*dt)
        assert not hasattr(h, '__len__')
        h = standardize_heading(h)
        # TODO: possibly remove clipping of heading
        h = max(self.heading_min, min(self.heading_max, h))  # car must face forward
        v = v_tm1 + a_t - self.alpha * v_tm1 * dt  # velocity plus acceleration minus friction
        # velocity cannot be below self.speed_min
        if v < self.speed_min:
            v = self.speed_min
        # For non-agent, speed cannot be above self.speed_limit
        if not self.stationary and v > self.speed_limit:
            v = self.speed_limit

        return np.array([x, y, h, v])

    def get_future_states(self, horizon, dt, controls=None):
        states = []
        curr_state = [self.x, self.y, self.heading, self.speed]
        c_idx = 0
        if controls is None:
            c_idx = self.control_idx
            controls = self.controls

        for i in range(horizon):
            curr_state = self.dynamics_f(curr_state, self.controls[c_idx], dt)
            states.append(curr_state)
            c_idx = (c_idx + 1) % len(controls)
        return states

    def update_car_state(self, control, dt, eps=1e-6):
        last_state = np.copy(self.state)
        self.x, self.y, self.heading, self.speed = self.dynamics_f(self.state, control, dt)
        if self.stationary:
            self.ydiff += self.y - last_state[1]
        if abs(self.heading - last_state[2]) > eps:
            self.image = pygame.transform.rotate(self.orig_img, math.degrees(self.heading))

    def update_car_state_directly(self, next_state, eps=1e-6):
        last_state = np.copy(self.state)
        self.x, self.y, self.heading, self.speed = next_state
        if self.stationary:
            self.ydiff += self.y - last_state[1]
        if abs(self.heading - last_state[2]) > eps:
            self.image = pygame.transform.rotate(self.orig_img, math.degrees(self.heading))

    def update(self, ydiff=None, control=None, next_state=None, dt=1):
        if dt != 1:
            assert next_state is None
        if self.switch_duration_remaining > 0:
            self.x += self.switch_step
            self.switch_duration_remaining -= 1
            if self.switch_duration_remaining == 0:
                # self.image = pygame.transform.rotate(self.image, math.degrees(self.switching_direction))
                self.switching_direction = 0
                self.state[0] = self.x
        if control is None and next_state is None:
            self.update_car_state(self.controls[self.control_idx], dt)
            self.control_idx = (self.control_idx + 1) % len(self.controls)
            # If there are steps remaining to take, 
        elif control is not None:
            self.update_car_state(control, dt)
        elif next_state is not None:
            self.update_car_state_directly(next_state)

        if not self.stationary:
            assert ydiff is not None
            self.rect.center = self.x, self.y - ydiff
        else:
            self.rect.center = self.x, self.y - self.ydiff

    def start_switch_lane(self, rng, **kwargs):
        self.switch_duration_remaining = self.switch_duration
        lane_centers = kwargs['lane_centers']
        lane_width = kwargs['lane_width']
        lane = np.argmin(list(map(lambda lane_coord: (lane_coord - self.x) ** 2, lane_centers)))
        if lane == 1:
            self.switching_direction = random.randint(0, 1) * 2 - 1
        elif lane == 0:
            self.switching_direction = 1
        else:
            self.switching_direction = -1
        # self.heading = self.switching_direction * 0.1
        self.switch_step = (lane_width + 4) * self.switching_direction / self.switch_duration
        self.switch_duration_remaining -= 1
        # self.x += self.switch_step


class Backdrop():
    def __init__(self, **kwargs):
        # kwargs should contain game constants (e.g., lane width)
        for k in kwargs:
            setattr(self, k, kwargs[k])

        self.WHITE = (255, 255, 255)
        self.BLACK = (47, 45, 49)
        self.RED = (255, 0, 0)
        self.GREEN = (105, 155, 103)
        self.BLUE = (0, 0, 255)

        self.init_surface()
        self.init_lane_surface()
        self.update(0)

    def init_surface(self):
        surface = pygame.Surface((self.screen_width, self.screen_height))
        surface.convert()  # Speeds up blitting
        surface.fill(self.WHITE)  # Clears the screen
        road_endx = self.grass_width + self.n_lanes * (self.border_width + self.lane_width)
        # Road boundaries
        surface.fill(self.WHITE, rect=[self.grass_width, 0,
                                       self.border_width, self.screen_height])
        surface.fill(self.WHITE, rect=[road_endx, 0,
                                       self.border_width, self.screen_height])
        # Road surface
        surface.fill(self.BLACK, rect=[self.grass_width + self.border_width, 0,
                                       self.n_lanes * road_endx, self.screen_height])
        # Grass
        surface.fill(self.GREEN, rect=[0, 0, self.grass_width, self.screen_height])
        surface.fill(self.GREEN, rect=[road_endx + self.border_width, 0,
                                       self.grass_width, self.screen_height])
        self.surface = surface

    def init_lane_surface(self):
        surface = pygame.Surface((self.border_width, self.screen_height))
        surface.convert()  # Speeds up blitting
        self.lane_surface = surface

    def update_lane_surface(self, offset_y):
        offset_y = int(offset_y)
        offset_y = offset_y % (self.lane_mark_length + self.lane_mark_btwn_length)

        self.lane_surface.fill(self.BLACK)
        # draw first lane boundary
        if offset_y < self.lane_mark_length:
            self.lane_surface.fill(self.WHITE, rect=[0, 0, self.border_width,
                                                     self.lane_mark_length - offset_y])
        offset = (self.lane_mark_length + self.lane_mark_btwn_length) - offset_y
        for i in range(offset, self.screen_height,
                       self.lane_mark_length + self.lane_mark_btwn_length):
            self.lane_surface.fill(self.WHITE, rect=[0, i, self.border_width,
                                                     self.lane_mark_length])

    def update(self, ydiff):
        self.update_lane_surface(ydiff)

        for i in range(1, self.n_lanes):
            offset = self.grass_width + i * (self.border_width + self.lane_width)
            self.surface.blit(self.lane_surface, (offset, 0))

    def draw_background(self, screen):
        screen.blit(self.surface, (0, 0))
