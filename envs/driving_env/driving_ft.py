#!/usr/bin/env python
import math
import numpy as np
from envs.driving_env.driving_car import Car

class DummyCar:
    def __init__(self, position_x, position_y, heading=None, speed=None, **kwargs):
        self.x = position_x
        self.y = position_y
        if heading is not None:
            self.heading = heading
        else:
            self.heading = kwargs["default_heading"]
        if speed is not None:
            self.speed = speed
        else:
            self.speed = kwargs["default_speed_ratio_cpu"] * kwargs["screen_height"]
        self.dummy = True


def get_n_cpu_cars_from_ft(ft_dict):
    n_cars = max([int(x.split("_")[0][3:]) for x in list(ft_dict.keys())
                  if x.startswith("cpu")])
    return n_cars


def get_car_states_from_ft(ft_dict, **kwargs):
    # ignore dummy cpu cars (which were added to make features more consistent
    # and easier for policy to learn from)
    drivable_width = kwargs["drivable_width"]
    drivable_height = kwargs["drivable_height"]
    speed_max = kwargs["speed_max"]
    heading_max = kwargs["heading_max"]
    road_bounds = kwargs["road_bounds"]

    robot_state = [ft_dict["agent_" + k] for k in ["x", "y", "h", "v"]]
    robot_state[0] = robot_state[0] * drivable_width + road_bounds[0]
    robot_state[1] *= drivable_height
    robot_state[2] *= heading_max
    robot_state[3] *= speed_max
    cpu_states = []
    n_cars = get_n_cpu_cars_from_ft(ft_dict)
    for i in range(n_cars):
        prefix = "cpu" + str(i + 1) + "_"
        if ft_dict[prefix + "dummy"]:
            continue
        cpu_state = [ft_dict[prefix + k] for k in
                     ["x", "y", "h", "v", "switching_direction", "switch_duration_remaining"]]
        # next two lines are because saved x,y are offsets from robot's x,y
        # and normalized to be between 0 and 1
        cpu_state[0] = (cpu_state[0] - 0.5) * (2 * drivable_width) + robot_state[0]
        cpu_state[1] = (cpu_state[1] - 0.5) * (2 * drivable_height) + robot_state[1]
        cpu_state[2] *= heading_max
        cpu_state[3] *= speed_max
        cpu_states.append(cpu_state)

    return robot_state, cpu_states


def get_cars_near_robot(robot_car, other_cars, **kwargs):
    # Returns info for closest six cars to robot:
    #     closest in front of robot, in same lane
    #     closest behind robot, in same lane
    #     two closest to robot, in lane directly to left (ordered by y)
    #     two closest to robot, in lane directly to right (ordered by y)
    #
    # If there are less than two cars in each of these three possible lanes,
    # set y to be the maximum possible distance (based on dimensions of game
    # window) for car to be just barely off-screen
    lane_centers = kwargs["lane_centers"]
    car_height = kwargs["car_height"]
    screen_height = kwargs["screen_height"]
    ydiff = robot_car.ydiff

    robot_lane_idx = np.argmin(np.abs(lane_centers - robot_car.x))
    others_lane_idx = [np.argmin(np.abs(lane_centers - car.x)) for car in other_cars]
    cars_in_front = []  # cars in same lane, in front of robot
    cars_behind = []  # cars in same lane, behind robot
    cars_in_left = []  # cars in lane directly to left of robot
    cars_in_right = []  # cars in lane directly to right of robot

    for i, lane_idx in enumerate(others_lane_idx):
        if lane_idx == robot_lane_idx and other_cars[i].y < robot_car.y:
            cars_in_front.append(other_cars[i])  # because y=0 at top of screen
        elif lane_idx == robot_lane_idx and other_cars[i].y > robot_car.y:
            cars_behind.append(other_cars[i])
        elif lane_idx == robot_lane_idx - 1:
            cars_in_left.append(other_cars[i])
        elif lane_idx == robot_lane_idx + 1:
            cars_in_right.append(other_cars[i])

    # Rank other cars by their distance to robot car (along y axis); keep closest two
    cars_in_front = sorted(cars_in_front, key=lambda c: np.abs(c.y - robot_car.y))[:1]
    cars_behind = sorted(cars_behind, key=lambda c: np.abs(c.y - robot_car.y))[:1]
    cars_in_left = sorted(cars_in_left, key=lambda c: np.abs(c.y - robot_car.y))[:2]
    cars_in_right = sorted(cars_in_right, key=lambda c: np.abs(c.y - robot_car.y))[:2]

    dist_btwn_lanes = lane_centers[1] - lane_centers[0]
    if robot_car.y - ydiff > screen_height / 2.:
        further_away_y = -(car_height / 2)
    else:
        further_away_y = screen_height + car_height / 2

    if len(cars_in_front) == 0:
        cars_in_front.append(DummyCar(lane_centers[robot_lane_idx], \
                                      -(car_height / 2) + ydiff, **kwargs))
    if len(cars_behind) == 0:
        cars_behind.append(DummyCar(lane_centers[robot_lane_idx], \
                                    screen_height + car_height / 2 + ydiff, **kwargs))
    while len(cars_in_left) < 2:
        cars_in_left.append(DummyCar(lane_centers[robot_lane_idx] - dist_btwn_lanes, \
                                     further_away_y + ydiff, **kwargs))
    while len(cars_in_right) < 2:
        cars_in_right.append(DummyCar(lane_centers[robot_lane_idx] + dist_btwn_lanes, \
                                      further_away_y + ydiff, **kwargs))

    return cars_in_front + cars_behind + cars_in_left + cars_in_right


def check_tailgating(lead_car : Car, follow_car : Car, **kwargs):
    lane_width = kwargs["lane_width"]
    car_height = kwargs["car_height"]
    same_lane = np.abs(lead_car.x - follow_car.x) < lane_width / 2
    y_diff = follow_car.y - lead_car.y
    y_close_behind = 0 < y_diff < car_height * 3
    return same_lane and y_close_behind


def check_robot_tailgating_cpu(robot_car, cpu_cars, **kwargs):
    for cpu_car in cpu_cars:
        adjacent = check_tailgating(cpu_car, robot_car, **kwargs)
        if adjacent:
            return True
    return False


def check_adjacency(car_1 : Car, car_2 : Car, **kwargs):
    lane_width = kwargs["lane_width"]
    car_height = kwargs["car_height"]
    x_close = np.abs(car_1.x - car_2.x) < lane_width
    y_close = np.abs(car_1.y - car_2.y) < car_height * .75
    return x_close and y_close


def check_robot_near_cpu(robot_car, cpu_cars, **kwargs):
    for cpu_car in cpu_cars:
        adjacent = check_adjacency(robot_car, cpu_car, **kwargs)
        if adjacent:
            return True
    return False


def get_lane(robot_car, **kwargs):
    lane_centers = kwargs["lane_centers"]
    robot_lane_idx = np.argmin(np.abs(lane_centers - robot_car.x))
    return robot_lane_idx

def dist_to_car_in_front(robot_car, cpu_cars, **kwargs):
    min_dist_in_front = 1000
    for cpu_car in cpu_cars:
        if get_lane(robot_car, **kwargs) == get_lane(cpu_car, **kwargs) and robot_car.y > cpu_car.y:
            min_dist_in_front = min(min_dist_in_front, robot_car.y - cpu_car.y)
    return min_dist_in_front / 1000


def get_game_state_ft(robot_car, other_cars, **kwargs):
    # Required for setting game state, in game.setGameState
    # Consists of the following:
    #     state of robot car: x, y, heading, velocity
    #     state of each cpu car: x, y, heading, velocity
    #                            (x and y are offsets from robot car's)
    #
    # Note that ydiff is first subtracted from all y positions (because in
    # setGameState, ydiff is assumed to be zero)
    #
    ydiff = robot_car.ydiff
    drivable_width = kwargs["drivable_width"]
    drivable_height = kwargs["drivable_height"]
    speed_max = kwargs["speed_max"]
    heading_max = kwargs["heading_max"]
    road_bounds = kwargs["road_bounds"]

    ft_dict = {}
    ft_dict["agent_x"] = (robot_car.x - road_bounds[0]) / drivable_width
    ft_dict["agent_y"] = (robot_car.y - ydiff) / drivable_height
    ft_dict["agent_h"] = robot_car.heading / heading_max
    ft_dict["agent_v"] = robot_car.speed / speed_max

    cpu_cars = get_cars_near_robot(robot_car, other_cars, **kwargs)
    for i, car in enumerate(cpu_cars):
        # ft_dict["cpu"+str(i+1)+"_x"] = (car.x - road_bounds[0]) / drivable_width
        # ft_dict["cpu"+str(i+1)+"_y"] = (car.y - ydiff) / drivable_height
        ft_dict["cpu" + str(i + 1) + "_x"] = (car.x - robot_car.x) / (2 * drivable_width) + 0.5
        ft_dict["cpu" + str(i + 1) + "_y"] = (car.y - robot_car.y) / (2 * drivable_height) + 0.5
        ft_dict["cpu" + str(i + 1) + "_h"] = car.heading / heading_max
        ft_dict["cpu" + str(i + 1) + "_v"] = car.speed / speed_max
        if hasattr(car, "dummy") and car.dummy:
            # ft_dict["cpu"+str(i+1)+"_x"] = 0
            # ft_dict["cpu"+str(i+1)+"_y"] = 0
            # ft_dict["cpu"+str(i+1)+"_h"] = 0
            # ft_dict["cpu"+str(i+1)+"_v"] = 0
            ft_dict["cpu" + str(i + 1) + "_dummy"] = 1
        else:
            ft_dict["cpu" + str(i + 1) + "_dummy"] = 0

    return ft_dict


def get_game_state_to_save_ft(robot_car, other_cars, **kwargs):
    # Required for setting game state, in game.setGameStateSave
    # Is different in that it saves all cpu cars, not dummy cars...
    # Consists of the following:
    #     state of robot car: x, y, heading, velocity
    #     state of each cpu car: x, y, heading, velocity
    #                            (x and y are offsets from robot car's)
    #
    # Note that ydiff is first subtracted from all y positions (because in
    # setGameState, ydiff is assumed to be zero)
    #
    ydiff = robot_car.ydiff
    drivable_width = kwargs["drivable_width"]
    drivable_height = kwargs["drivable_height"]
    speed_max = kwargs["speed_max"]
    heading_max = kwargs["heading_max"]
    road_bounds = kwargs["road_bounds"]

    ft_dict = {}
    ft_dict["agent_x"] = (robot_car.x - road_bounds[0]) / drivable_width
    ft_dict["agent_y"] = (robot_car.y - ydiff) / drivable_height
    ft_dict["agent_h"] = robot_car.heading / heading_max
    ft_dict["agent_v"] = robot_car.speed / speed_max

    cpu_cars = other_cars
    for i, car in enumerate(cpu_cars):
        # ft_dict["cpu"+str(i+1)+"_x"] = (car.x - road_bounds[0]) / drivable_width
        # ft_dict["cpu"+str(i+1)+"_y"] = (car.y - ydiff) / drivable_height
        ft_dict["cpu" + str(i + 1) + "_x"] = (car.x - robot_car.x) / (2 * drivable_width) + 0.5
        ft_dict["cpu" + str(i + 1) + "_y"] = (car.y - robot_car.y) / (2 * drivable_height) + 0.5
        ft_dict["cpu" + str(i + 1) + "_h"] = car.heading / heading_max
        ft_dict["cpu" + str(i + 1) + "_v"] = car.speed / speed_max
        ft_dict["cpu" + str(i + 1) + "_switching_direction"] = car.switching_direction
        ft_dict["cpu" + str(i + 1) + "_switch_duration_remaining"] = car.switch_duration_remaining
        if hasattr(car, "dummy") and car.dummy:
            # ft_dict["cpu"+str(i+1)+"_x"] = 0
            # ft_dict["cpu"+str(i+1)+"_y"] = 0
            # ft_dict["cpu"+str(i+1)+"_h"] = 0
            # ft_dict["cpu"+str(i+1)+"_v"] = 0
            ft_dict["cpu" + str(i + 1) + "_dummy"] = 1
        else:
            ft_dict["cpu" + str(i + 1) + "_dummy"] = 0

    return ft_dict


def get_state_ft(robot_car, other_cars, return_dist=True, **kwargs):
    lane_centers = kwargs["lane_centers"]
    car_height = kwargs["car_height"]
    s = robot_car.state
    s_other = [car.state for car in other_cars]
    ft_dict = {}
    n_ft = 0
    game_state_ft = get_game_state_ft(robot_car, other_cars, **kwargs)
    ft_dict.update(game_state_ft)
    n_ft += len(game_state_ft)

    lane_ft = lane_indicator(s, lane_centers, return_dict=True)
    speed_ft = speed_wrt_limit(s, robot_car.speed_limit, return_dict=True)
    ft_dict.update(lane_ft)
    ft_dict.update(speed_ft)
    n_ft += len(lane_ft)
    n_ft += len(speed_ft)

    assert len(ft_dict) == n_ft

    return ft_dict


def get_state_save_ft(robot_car, other_cars, return_dist=True, **kwargs):
    lane_centers = kwargs["lane_centers"]
    car_height = kwargs["car_height"]
    s = robot_car.state
    s_other = [car.state for car in other_cars]
    ft_dict = {}
    n_ft = 0
    game_state_ft = get_game_state_to_save_ft(robot_car, other_cars, **kwargs)
    ft_dict.update(game_state_ft)
    n_ft += len(game_state_ft)

    lane_ft = lane_indicator(s, lane_centers, return_dict=True)
    speed_ft = speed_wrt_limit(s, robot_car.speed_limit, return_dict=True)
    ft_dict.update(lane_ft)
    ft_dict.update(speed_ft)
    n_ft += len(lane_ft)
    n_ft += len(speed_ft)

    assert len(ft_dict) == n_ft

    return ft_dict


def get_reward_ft(robot_car, other_cars, action, speed_multiplier, **kwargs):
    # Assumes action[0] is turning
    action_turn = action[0]

    # Max distance to lane center / road boundary
    lane_width = kwargs["lane_width"]
    border_width = kwargs["border_width"]
    car_width = kwargs["car_width"]
    car_height = kwargs["car_height"]
    lane_centers = kwargs["lane_centers"]
    max_dist = (lane_width - car_width) / 2.

    s = robot_car.state
    s_other = [car.state for car in other_cars]
    ft_lanes = lane_mindist(s, lane_centers, max_dist=max_dist)
    ft_speed = 1 - speed_limit_ft(s, robot_car.speed_limit * speed_multiplier)
    ft_carnear = overlap_centered(s, s_other, lane_width + border_width,
                                  car_height * 3, return_dist=True, **kwargs)
    ft_turn = turn_ft(s, kwargs["default_heading"], kwargs["heading_max"])
    ft_forward = forward_progress(s, robot_car.speed_limit)
    ft_sharpturn = np.square(action_turn)
    # ft = [ft_lanes, ft_speed, ft_carnear, ft_turn, ft_forward]
    ft_adjacent = int(check_robot_near_cpu(robot_car, other_cars, **kwargs))
    ft_tailgating = int(check_robot_tailgating_cpu(robot_car, other_cars, **kwargs))
    lane = get_lane(robot_car, **kwargs)
    dist_in_front = dist_to_car_in_front(robot_car, other_cars, **kwargs)
    ft = [ft_lanes, ft_speed, ft_carnear, ft_turn, ft_forward, ft_sharpturn, ft_adjacent, ft_tailgating,
          lane == 0, lane == 1, lane == 2, dist_in_front]

    assert np.all(np.array(ft) <= 1.0) and np.all(np.array(ft) >= 0.0)
    return np.array(ft)


def turn_ft(s, heading, max_heading):
    max_val = np.square(max_heading - heading)
    return np.minimum(np.square(s[2] - heading), max_val) / max_val


def out_of_bound(car, ydiff, **kwargs):
    road_bounds = kwargs["road_bounds"]
    screen_height = kwargs["screen_height"]
    car_width = kwargs["car_width"]
    car_height = kwargs["car_height"]
    x = car.x
    y = car.y - ydiff
    return x < road_bounds[0] + car_width / 2 or x > road_bounds[1] - car_width / 2 \
           or y > screen_height + car_height / 2


def collision_exists(robot_car, other_cars, **kwargs):
    car_width = kwargs["car_width"]
    car_height = kwargs["car_height"]
    s = robot_car.state
    s_other = [car.state for car in other_cars]
    collide = overlap_centered(s, s_other, car_width, car_height, rotate_overlap=True, **kwargs)
    return collide


def lane_mindist(s, centers, reverse=False, min_dist=0, max_dist=1):
    # centers: N-dimensional vector
    # feature is normalized to be between 0 and 1
    ft = np.maximum(np.min(np.r_[np.abs(centers - s[0]), max_dist]), min_dist)
    ft = (ft - min_dist) / float(max_dist - min_dist)
    if reverse:
        ft = 1 - ft
    return ft


def lane_indicator(s, lane_centers, return_dict=False):
    lane_idx = np.argmin(np.abs(lane_centers - s[0]))
    ft = [0] * len(lane_centers)
    ft[lane_idx] = 1
    if return_dict:
        ft_dict = {}
        for i in range(len(ft)):
            ft_dict["in_lane_" + str(i + 1)] = ft[i]
        return ft_dict
    return ft


def speed_limit_ft(s, speed, max_val=16.0):
    # return np.minimum(np.square(np.maximum(s[3]-speed, 0)),max_val)
    return np.minimum(np.square(s[3] - speed), max_val) / max_val


def speed_wrt_limit(s, speed, return_dict=False):
    intervals = [0, 0.25 * speed, 0.5 * speed, 0.75 * speed, 1.0 * speed, 1.25 * speed, 1.5 * speed, float("inf")]
    ft = [0] * (len(intervals) - 1)
    for i in range(len(intervals) - 1):
        if s[3] >= intervals[i] and s[3] < intervals[i + 1]:
            ft[i] = 1
    assert np.sum(ft) == 1
    if return_dict:
        ft_dict = {}
        for i in range(len(ft)):
            ft_dict["in_speed_interval_" + str(i + 1)] = ft[i]
        return ft_dict
    return ft


def get_othercar_ft(directions, s, s_other, return_dist, dist, shift=False, \
                    return_dict=True, **kwargs):
    othercar_ft = []
    absence_ft = []
    for direction in directions:
        ft = overlap_direction(s, s_other, direction, dist, return_dist=return_dist, shift=shift, **kwargs)
        othercar_ft.append(ft)
        absence_ft.append(1 * (ft == 0))
    if return_dict:
        ft_dict = {}
        for i in range(len(othercar_ft)):
            key = "overlap_" + directions[i] + "_within_" + str(dist)
            if shift:
                key += "_shift"
            ft_dict[key] = othercar_ft[i]
            ft_dict["no_" + key] = absence_ft[i]
        return ft_dict

    return absence_ft + othercar_ft


def rotate_around(pt, center, theta):
    # Translate point to origin
    temp_x = pt[0] - center[0]
    temp_y = pt[1] - center[1]

    # Apply rotation
    theta *= -1  # Because y axis is flipped (decreases as you go up) in pygame
    # (0,0) is in upper left hand corner
    rot_x = temp_x * math.cos(theta) - temp_y * math.sin(theta)
    rot_y = temp_x * math.sin(theta) + temp_y * math.cos(theta)

    # Translate back
    x = rot_x + center[0]
    y = rot_y + center[1]
    return np.array([x, y])


def get_vertices(c, dim):
    # Returns vertices of rectangle, given the center, half-width, and half-height
    w, h = dim
    return np.array([[c[0] - w, c[1] - h],
                     [c[0] + w, c[1] - h],
                     [c[0] + w, c[1] + h],
                     [c[0] - w, c[1] + h]])


def get_car_vertices(s, car_width, car_height, rotate=False):
    vertices = get_vertices((s[0], s[1]), (car_width / 2, car_height / 2))
    rot_vertices = rotate
    if rotate:
        c = (s[0], s[1])
        h = s[2] - math.pi / 2
        vertices = [rotate_around(v, c, h) for v in vertices]
    return vertices


def overlap(s, s_other, overlap_v, return_dist=False, shift=False, **kwargs):
    # overlap_v: vertices of overlap region
    car_width = kwargs["car_width"]
    car_height = kwargs["car_height"]

    robot_v = get_car_vertices(s, car_width, car_height)
    max_dist = np.maximum(
        np.min([x[1] for x in robot_v]) - np.min([x[1] for x in overlap_v]),
        np.max([x[1] for x in overlap_v]) - np.max([x[1] for x in robot_v]))
    if return_dist:
        assert max_dist > 0
    mindist_to_other = max_dist
    for s2 in s_other:
        if shift:
            robot_v = get_car_vertices((s2[0], s[1]), car_width, car_height)
        car_v = get_car_vertices(s2, car_width, car_height, rotate=True)

        if intersects(overlap_v, car_v):
            # print("overlap_v:", overlap_v)
            # print("car_v:", car_v)
            if not return_dist:
                return 1
            else:
                mindist_to_other = np.minimum(distance_btwn(robot_v, car_v), mindist_to_other)
    if not return_dist:
        return 0
    return np.maximum(0, 1 - mindist_to_other / float(max_dist))


def overlap_centered(s, s_other, buf_w, buf_h, return_dist=False, rotate_overlap=False, **kwargs):
    # test overlap region that's centered around robot car
    # buf_w: width of overlap region
    # buf_h: height of overlap region
    overlap_v = get_vertices((s[0], s[1]), (buf_w / 2, buf_h / 2))
    if rotate_overlap:
        c = (s[0], s[1])
        h = s[2] - math.pi / 2
        orig_overlap_v = np.array(overlap_v)
        overlap_v = [rotate_around(v, c, h) for v in overlap_v]
    is_overlap = overlap(s, s_other, overlap_v, return_dist=return_dist, **kwargs)
    return is_overlap


def overlap_direction(s, s_other, direction, dist, return_dist=False, \
                      shift=False, **kwargs):
    car_height = kwargs["car_height"]
    lane_width = kwargs["lane_width"]
    border_width = kwargs["border_width"]

    # direction - direction of overlap
    if "front" in direction:
        # dist *= 3
        car_topy = s[1] - car_height / 2
        y = (s[1] + car_topy - dist) / 2  # y-coord of center of regions in front of car
        buf = [(lane_width + border_width) / 2, (dist + car_height / 2) / 2]
    elif "back" in direction:
        car_boty = s[1] + car_height / 2
        y = (s[1] + car_boty + dist) / 2  # y-coord of center of regions behind car
        buf = [(lane_width + border_width) / 2, (dist + car_height / 2) / 2]
    elif direction == "right" or direction == "left":
        car_topy = s[1] - car_height / 2
        car_boty = s[1] + car_height / 2
        y = (car_boty + dist + car_topy - dist * 3) / 2  # y-coord of center of regions to side of car
        buf = [(lane_width + border_width) / 2, (dist * 4 + car_height) / 2]
    else:
        raise NotImplementedError

    if "left" in direction:
        overlap_v = get_vertices((s[0] - lane_width - border_width, y), buf)
    elif "right" in direction:
        overlap_v = get_vertices((s[0] + lane_width + border_width, y), buf)
    elif direction == "front" or direction == "back":
        overlap_v = get_vertices((s[0], y), buf)
    else:
        raise NotImplementedError

    return overlap(s, s_other, overlap_v, return_dist=return_dist, shift=shift, **kwargs)


def separating_axis_exists(rect1, rect2):
    for i in range(len(rect1)):
        j = (i + 1) % len(rect1)
        edge = rect1[i] - rect1[j]
        normal = np.zeros(2)
        normal[0] = -1 * edge[1]
        normal[1] = edge[0]
        r1_sides = set()
        r2_sides = set()
        for r1_vert in rect1:
            r1_sides.add(np.sign(np.dot(normal, r1_vert - rect1[i])))
        for r2_vert in rect2:
            r2_sides.add(np.sign(np.dot(normal, r2_vert - rect1[i])))
        for sides in [r1_sides, r2_sides]:
            if 0 in sides:
                sides.remove(0)
        assert len(r1_sides) == 1
        if len(r2_sides) == 2:
            continue
        # All vertices in rectangle 2 are on one side
        if len(r1_sides.intersection(r2_sides)) == 0:
            return True
    return False


def intersects(rect1, rect2):
    # Returns 1 if the two rectangles intersect, 0 otherwise
    # rect - list of vertices of rectangle (in CW or CCW order)
    # Source; https://stackoverflow.com/questions/115426/algorithm-to-detect-intersection-of-two-rectangles
    return not separating_axis_exists(rect1, rect2) and not separating_axis_exists(rect2, rect1)


def distance_btwn(rect1, rect2):
    if intersects(rect1, rect2):
        return 0
    mindist = float("inf")
    for i in range(len(rect1)):
        j = (i + 1) % len(rect1)
        for v2 in rect2:
            d = pnt2line(v2, rect1[i], rect1[j])
            mindist = np.minimum(d, mindist)
        for v1 in rect1:
            d = pnt2line(v1, rect2[i], rect2[j])
            mindist = np.minimum(d, mindist)
    return mindist


def pnt2line(pnt, start, end):
    # Source: http://www.fundza.com/vectors/point2line/index.html
    # Given a line with coordinates 'start' and 'end' and the
    # coordinates of a point 'pnt' the proc returns the shortest
    # distance from pnt to the line and the coordinates of the
    # nearest point on the line.
    #
    # 1  Convert the line segment to a vector ('line_vec').
    # 2  Create a vector connecting start to pnt ('pnt_vec').
    # 3  Find the length of the line vector ('line_len').
    # 4  Convert line_vec to a unit vector ('line_unitvec').
    # 5  Scale pnt_vec by line_len ('pnt_vec_scaled').
    # 6  Get the dot product of line_unitvec and pnt_vec_scaled ('t').
    # 7  Ensure t is in the range 0 to 1.
    # 8  Use t to get the nearest location on the line to the end
    #    of vector pnt_vec_scaled ('nearest').
    # 9  Calculate the distance from nearest to pnt_vec_scaled.
    # 10 Translate nearest back to the start/end line.
    line_vec = start - end
    pnt_vec = start - pnt
    line_len = np.linalg.norm(line_vec)
    line_unitvec = line_vec / line_len
    pnt_vec_scaled = pnt_vec / line_len
    t = np.dot(line_unitvec, pnt_vec_scaled)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    nearest = line_vec * t
    dist = np.linalg.norm(nearest - pnt_vec)
    nearest = start + nearest
    return dist


def forward_progress(s, speed, max_val=1.0):
    return np.minimum(s[3] / float(speed), max_val)
