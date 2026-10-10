"""Nonblocking transitions around the existing exploration state machine."""

import math
import time

import numpy as np


FEATURE_SCALING = {'information_gain_cells': 100., 'path_cost_metres': 10.}


def map_stamp(message):
    """Read acquisition time, not callback arrival time."""
    return message.header.stamp.sec * 10**9 + message.header.stamp.nanosec


def copy_state(state):
    """Detach candidate features from clusters whose IDs will be reused."""
    return {key: np.asarray(value, dtype=np.float32).copy()
            for key, value in state.items()}


def explored_cells(previous, current, previous_origin=(0., 0.),
                   current_origin=(0., 0.), resolution=1., require_overlap=False):
    """
    Count cells becoming known where their previous physical footprint was unknown.

    SLAM can enlarge and shift an occupancy grid. Cells outside the old grid
    have no observed previous value and are conservatively excluded. For fractional
    shifts, every old cell overlapping a new cell must have been unknown; this
    avoids rewarding a shifted raster boundary between known and unknown space.
    """
    shift = (np.asarray(previous_origin) - current_origin) / resolution
    if not np.allclose(shift, np.round(shift), atol=1e-4, rtol=0):
        old_x = np.arange(current.shape[1]) - shift[0]
        old_y = np.arange(current.shape[0]) - shift[1]
        x0, x1 = np.floor(old_x + 1e-9).astype(int), np.ceil(old_x + 1. - 1e-9).astype(int) - 1
        y0, y1 = np.floor(old_y + 1e-9).astype(int), np.ceil(old_y + 1. - 1e-9).astype(int) - 1
        valid_x = np.flatnonzero((x0 >= 0) & (x1 < previous.shape[1]))
        valid_y = np.flatnonzero((y0 >= 0) & (y1 < previous.shape[0]))
        if not len(valid_x) or not len(valid_y):
            if require_overlap:
                raise ValueError('Map grids do not overlap; gain and stagnation skipped')
            return 0
        known = current[np.ix_(valid_y, valid_x)] != -1
        for ys in (y0, y1):
            for xs in (x0, x1):
                known &= previous[np.ix_(ys[valid_y], xs[valid_x])] == -1
        return int(np.count_nonzero(known))
    dx, dy = np.round(shift).astype(int)
    old_y, old_x = max(0, -dy), max(0, -dx)
    new_y, new_x = max(0, dy), max(0, dx)
    height = min(previous.shape[0] - old_y, current.shape[0] - new_y)
    width = min(previous.shape[1] - old_x, current.shape[1] - new_x)
    if height <= 0 or width <= 0:
        if require_overlap:
            raise ValueError('Map grids do not overlap; gain and stagnation skipped')
        return 0
    old = previous[old_y:old_y + height, old_x:old_x + width]
    new = current[new_y:new_y + height, new_x:new_x + width]
    return int(np.count_nonzero((old == -1) & (new != -1)))


class Env:
    """Keep one pending action; main_node owns all navigation and callbacks."""

    def __init__(self, node, gain_weight=.7, distance_weight=.2,
                 stagnation_weight=.1, collision_weight=0.,
                 stagnation_threshold=.05, collision_enabled=False,
                 map_max_age=5., map_settle_time=.2, map_wait_timeout=30.):
        """Validate weighted rewards and initialize one pending navigation transition."""
        weights = (gain_weight, distance_weight, stagnation_weight, collision_weight)
        if (not all(math.isfinite(w) and w >= 0 for w in weights)
                or not math.isclose(sum(weights), 1., rel_tol=0., abs_tol=1e-6)):
            raise ValueError('Reward weights must be finite, nonnegative and sum to 1')
        if (not math.isfinite(stagnation_threshold) or stagnation_threshold < 0
                or not all(math.isfinite(v) and v > 0
                           for v in (map_max_age, map_wait_timeout))
                or not math.isfinite(map_settle_time) or map_settle_time < 0):
            raise ValueError('Invalid stagnation threshold or map timing settings')
        self.node = node
        self.gain_weight = gain_weight
        self.distance_weight = distance_weight
        self.collision_weight = collision_weight
        self.stagnation_weight = stagnation_weight
        self.stagnation_threshold = stagnation_threshold
        self.collision_enabled = collision_enabled
        self.map_max_age = map_max_age
        self.map_settle_time = map_settle_time
        self.map_wait_timeout = map_wait_timeout
        self.last_map_stamp = -1
        self.post_map = None
        self.wait_started = None
        self.previous_map = None
        self.current_state = None
        self.current_action = None
        self.episode_done = False
        self.awaiting_next_state = False
        self.navigation_success = False
        self.collision_count = 0
        self.distance_cost = 0.
        self.last_info = {}

    def _geometry(self, message):
        info = message.info
        return ((info.origin.position.x, info.origin.position.y),
                info.resolution,
                (info.origin.orientation.x, info.origin.orientation.y,
                 info.origin.orientation.z, info.origin.orientation.w),
                message.header.frame_id)

    def observe_map(self, message):
        """Reject duplicate/out-of-order acquisitions, including delayed deliveries."""
        stamp = map_stamp(message)
        if stamp <= self.last_map_stamp:
            return False
        self.last_map_stamp = stamp
        return True

    def capture_post_map(self, message):
        """Freeze one outcome only after completion and the SLAM settling interval."""
        if not self.awaiting_next_state:
            return False
        if self.post_map is not None:
            return True
        cutoff = max(self.previous_stamp, self.navigation_finished_stamp)
        stamp = map_stamp(message)
        age = (self.node.get_clock().now().nanoseconds - stamp) / 10**9
        if (stamp <= cutoff + int(self.map_settle_time * 10**9)
                or not -.5 <= age <= self.map_max_age):
            return False
        self.post_map = self._grid(message)
        self.post_geometry = self._geometry(message)
        self.wait_started = None
        return True

    @staticmethod
    def _grid(message):
        """Copy occupancy cells after validating their physical cell size."""
        if not math.isfinite(message.info.resolution) or message.info.resolution <= 0:
            raise ValueError('Occupancy map resolution must be finite and positive')
        return np.asarray(message.data, dtype=np.int8).reshape(
            message.info.height, message.info.width).copy()

    def reset(self):
        """Start an episode, not an individual frontier-selection cycle."""
        if self.current_action is not None:
            raise RuntimeError('Cannot reset an unfinished navigation transition')
        self.current_state = None
        self.current_action = None
        self.episode_done = False
        self.awaiting_next_state = False
        self.collision_count = 0
        self.last_info = {}
        return self.get_state()

    def get_state(self):
        """Scale by fixed cells/metres constants, preserving absolute path costs."""
        clusters = list(self.node.frontier.values())
        if not clusters:
            return {}
        gain = np.asarray([c.information_gain for c in clusters], dtype=np.float32)
        cost = np.asarray([c.path_cost for c in clusters], dtype=np.float32)
        if not np.all(np.isfinite(gain)) or not np.all(np.isfinite(cost)):
            raise ValueError('RL candidates must have finite, estimated features')
        gain /= FEATURE_SCALING['information_gain_cells']
        cost /= FEATURE_SCALING['path_cost_metres']
        return {c.id: np.array([gain[i], cost[i]], dtype=np.float32)
                for i, c in enumerate(clusters)}

    def step(self, action, state=None):
        """Snapshot a decision and return its cluster for main_node to navigate."""
        if self.episode_done or self.current_action is not None:
            raise RuntimeError('Episode ended or an action is already pending')
        state = self.get_state() if state is None else state
        if action not in state or action not in self.node.frontier:
            raise ValueError(f'Frontier {action} is not available at this decision')
        message = getattr(self.node, 'latest_map', None)
        if message is None:
            raise RuntimeError('No fresh pre-action map is available')
        stamp = map_stamp(message)
        age = (self.node.get_clock().now().nanoseconds - stamp) / 10**9
        if not -.5 <= age <= self.map_max_age:
            raise RuntimeError('Pre-action map is stale or has an incompatible clock')
        previous_map = self._grid(message)
        self.current_state = copy_state(state)
        self.current_action = action
        self.previous_map = previous_map
        self.previous_geometry = self._geometry(message)
        self.previous_stamp = stamp
        self.post_map = None
        self.distance_cost = float(self.node.frontier[action].path_cost)
        self.collision_count = 0
        self.awaiting_next_state = False
        return self.node.frontier[action]

    def navigation_finished(self, success):
        """Record completion; wait for fresh map detection and path estimation."""
        if self.current_action is not None:
            self.navigation_success = bool(success)
            self.awaiting_next_state = True
            self.navigation_finished_stamp = self.node.get_clock().now().nanoseconds
            self.wait_started = time.monotonic()

    def record_collision(self):
        """Accept an actual contact event from an optional external sensor hook."""
        if (self.collision_enabled and self.current_action is not None
                and not self.awaiting_next_state):
            self.collision_count += 1

    def calculate_reward(self):
        """Reward new known area in square metres; penalize planned path metres."""
        if self.post_map is None:
            raise RuntimeError('A fresh post-navigation map has not been captured')
        origin, resolution, orientation, frame = self.post_geometry
        old_origin, old_resolution, old_orientation, old_frame = self.previous_geometry
        gain = 0
        comparable = False
        if (resolution > 0 and np.isclose(resolution, old_resolution, rtol=0, atol=1e-9)
                and orientation == old_orientation == (0., 0., 0., 1.)
                and frame == old_frame and np.all(np.isfinite(origin + old_origin))):
            try:
                gain = explored_cells(self.previous_map, self.post_map,
                                      old_origin, origin, resolution, require_overlap=True)
                comparable = True
            except ValueError as error:
                self.node.get_logger().warn(str(error))
        else:
            self.node.get_logger().warn('Map geometry changed; gain skipped this step')
        area = gain * resolution ** 2
        stagnation = int(comparable and area < self.stagnation_threshold)
        self.last_info = {
            'explored_cells': gain, 'explored_area': area,
            'distance_cost': self.distance_cost,
            'collisions': self.collision_count,
            'stagnation_penalty': stagnation, 'map_comparable': comparable,
            'map_alignment': ('aligned' if np.allclose(
                (np.asarray(old_origin) - origin) / resolution,
                np.round((np.asarray(old_origin) - origin) / resolution),
                rtol=0, atol=1e-4) else 'conservative_overlap') if comparable else 'unavailable',
            'navigation_success': self.navigation_success,
            'known_fraction': float(np.mean(self.post_map != -1)),
        }
        return (self.gain_weight * area - self.distance_weight * self.distance_cost
                - self.collision_weight * self.collision_count
                - self.stagnation_weight * stagnation)

    def complete_transition(self, next_state, done=False):
        """Finalize only after navigation and estimation of the next candidate set."""
        if self.current_action is None:
            self.episode_done = bool(done or not next_state)
            return None
        if not self.awaiting_next_state:
            raise RuntimeError('Navigation has not completed')
        done = bool(done or not next_state)
        transition = (copy_state(self.current_state), self.current_action,
                      self.calculate_reward(), copy_state(next_state), done)
        self.current_state = None
        self.current_action = None
        self.awaiting_next_state = False
        self.episode_done = done
        return transition
