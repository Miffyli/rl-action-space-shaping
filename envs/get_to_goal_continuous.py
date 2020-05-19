# Simple environment where player needs to reach the goal
# on a 2D plane
#

from math import sin, cos
import random

import numpy as np
import gym
import gym.spaces
from PIL import Image, ImageDraw

# Length of the arena per axis,
# with one corner being origo/zeros
LENGTH_PER_AXIS = 1

# Maximum distance of player from
# goal before it is considered "reached"
PLAYER_SIZE = 0.05
# Precomputed square of the player size,
# just for a tiny speedup
PLAYER_SIZE_SQR = PLAYER_SIZE ** 2

# How fast player moves per step
PLAYER_SPEED = 0.05
# How fast player turn per step, in
# radians
PLAYER_TURN_SPEED = 2 * np.pi * 0.05

# Constants for rendering
RENDER_RESOLUTION = 200
PLAYER_COLOR = "blue"
PLAYER_HEADING_COLOR = "green"
RENDER_GOAL_SIZE = 5
GOAL_COLOR = "red"


def rad_to_unit_vec(angle):
    """
    Turn angle (in rads) into a 2D unit vector

    Arguments:
        angle (float): Angle in rads

    Returns:
        vec (ndarray of (2,)): The unit vector
    """
    return np.array([cos(angle), sin(angle)])


def create_uniform_directions(num_directions):
    """
    Create equally spaced direction unit-vectors on
    a 2D plane. First direction is always [0, 1].

    Arguments:
        num_directions (int): How many direction vectors
            should be returned

    Returns:
        directions (ndarray of [num_directions, 2]):
            Directions as unit-vectors
    """
    rads_per_spacing = 2 * np.pi / num_directions

    directions = []
    current_rad = 0.0
    for i in range(num_directions):
        directions.append(rad_to_unit_vec(current_rad))
        current_rad += rads_per_spacing

    # Round vectors to smooth out the machine-precision
    # errors
    return np.round(np.array(directions), 8)


class GetToGoalContinuous(gym.Env):
    """
    Simple environment where
    player has to reach a goal.

    Observations are simply x/y/... distances
    to the goal. Actions vary (for study-purposes),
    but can be e.g. tank-controls or simple
    "move on diagonal".

    Arena is a box restricted to size [-1, 1] for convenient
    numbers
    """

    def __init__(
        self,
        timeout,
        action_space_type="discrete",
        goal_reward=1.0,
        timeout_reward=0.0,
        num_directions=4,
        num_bogus_actions=0,
        allow_backward=True,
        allow_strafe=False,
    ):
        """
        Arguments:
            timeout (int): Number of steps before game times out,
                           returns done=True and reward=timeout_reward.
            action_space_type (str): String specifying which type of action
                                space we have.
            goal_reward (float): Reward obtained upon reaching goal.
            timeout_reward (float): Reward obtained upon time-out.
            num_directions (int): Discrete and multidiscrete
                                  action-space will have this many
                                  directions to choose from, split
                                  equally to all directions.
            num_bogus_actions (int): Number of additional actions
                                     added to the action-space that
                                     do not do anything.
            allow_backward (bool): Allow agent to move backward
                                   in tank-like action spaces.
            allow_strafe (bool): Allow agent to move sideways in
                                 tank-like action spaces.
        """
        super().__init__()

        # TODO current limitation
        if not allow_backward and allow_strafe:
            raise ValueError("Disabling backward but allowing strafing is not supported.")

        self.goal_reward = goal_reward

        self.timestep = 0
        self.timeout = timeout
        self.timeout_reward = timeout_reward

        # Coordinates of the player and goal.
        # In order [y, x], and [0, 0] is top-left corner.
        self.player_position = np.zeros((2,), dtype=np.float32)
        self.goal_position = np.zeros((2,), dtype=np.float32)

        self.num_bogus_actions = num_bogus_actions
        self.allow_backward = allow_backward
        self.allow_strafe = allow_strafe

        # Create direction vectors for discrete and multidiscrete
        # actions (only for "discrete" and "multidiscrete" spaces)
        self.num_directions = num_directions
        self.directions = None
        if action_space_type == "discrete" or action_space_type == "multidiscrete":
            self.directions = create_uniform_directions(num_directions)

        # Current player heading, used in the "tank-like" controls where
        # player has a heading. Heading is in rads.
        self.player_heading = 0

        # Observation is direction-to-goal vector
        self.observation_space = gym.spaces.Box(
            -LENGTH_PER_AXIS, LENGTH_PER_AXIS,
            shape=(4,), dtype=np.float32
        )

        # Select which function will handle the movement action,
        # and define the action.
        self.action_space_type = action_space_type
        self.action_handler = None
        if action_space_type == "discrete":
            self.action_handler = self._execute_discrete_action
            self.action_space = gym.spaces.Discrete(self.num_directions + num_bogus_actions)
        elif action_space_type == "multidiscrete":
            self.action_handler = self._execute_multidiscrete_action
            # {0,1} decision for all actions
            self.action_space = gym.spaces.MultiDiscrete([2] * (self.num_directions + num_bogus_actions))
        elif action_space_type == "continuous":
            # Agent selects [0, 2*PI] as the next direction to move to
            # map it to [-1, 1] actions for "easier-to-learn"
            self.action_handler = self._execute_continuous_action
            self.action_space = gym.spaces.Box(-1, 1, shape=(1 + num_bogus_actions,), dtype=np.float32)
        elif action_space_type == "tank-discrete":
            # Tank-discrete action-space, where player can choose to
            # {TURN_LEFT, TURN_RIGHT, FORWARD, BACKWARD, MOVE_LEFT, MOVE_RIGHT}
            # Restrict action space by limiting number of actions provided
            num_actions = 3 + int(allow_backward) + (2 * int(allow_strafe))
            self.action_handler = self._execute_tank_discrete_action
            self.action_space = gym.spaces.Discrete(num_actions + num_bogus_actions)
        elif action_space_type == "tank-multidiscrete":
            # Tank-multidiscrete action-space, where player can choose to
            # {TURN_LEFT, TURN_RIGHT, FORWARD, BACKWARD, MOVE_LEFT, MOVE_RIGHT}
            # Restrict action space by limiting number of actions provided
            num_actions = 3 + int(allow_backward) + (2 * int(allow_strafe))
            self.action_handler = self._execute_tank_multidiscrete_action
            self.action_space = gym.spaces.MultiDiscrete([2] * (num_actions + num_bogus_actions))

    def _test_if_reached_goal(self):
        """
        Returns True if player has reached the goal,
        otherwise returns False
        """
        distance = np.sum((self.player_position - self.goal_position) ** 2)
        if distance <= PLAYER_SIZE_SQR:
            return True
        return False

    def _build_observation(self):
        """
        Build observation for the agent, i.e. the vector
        towards goal + player's current heading as
        cos(angle) + sin(angle).
        """
        return_observation = np.zeros((4,), dtype=np.float32)
        return_observation[:2] = self.goal_position - self.player_position
        return_observation[2] = cos(self.player_heading)
        return_observation[3] = sin(self.player_heading)
        return return_observation

    def _execute_tank_discrete_action(self, action):
        """
        Action is one of {0,1,2,3,4,5}, mapped to
        {TURN_LEFT, TURN_RIGHT, FORWARD, BACKWARD, MOVE_LEFT, MOVE_RIGHT}
        """
        if action == 0:
            self.player_heading += PLAYER_TURN_SPEED
        elif action == 1:
            self.player_heading -= PLAYER_TURN_SPEED
        elif action == 2:
            direction_vec = rad_to_unit_vec(self.player_heading)
            self.player_position += direction_vec * PLAYER_SPEED
        elif action == 3:
            direction_vec = rad_to_unit_vec(self.player_heading)
            self.player_position -= direction_vec * PLAYER_SPEED
        elif action == 4:
            direction_vec = rad_to_unit_vec(self.player_heading + (np.pi / 2))
            self.player_position += direction_vec * PLAYER_SPEED
        elif action == 5:
            direction_vec = rad_to_unit_vec(self.player_heading - (np.pi / 2))
            self.player_position += direction_vec * PLAYER_SPEED

    def _execute_tank_multidiscrete_action(self, action):
        """
        Action has false/true buttons for
        {TURN_LEFT, TURN_RIGHT, FORWARD, BACKWARD, MOVE_LEFT, MOVE_RIGHT}
        """
        if action[0] == 1:
            self.player_heading += PLAYER_TURN_SPEED
        if action[1] == 1:
            self.player_heading -= PLAYER_TURN_SPEED
        # Accumulate movement
        movement_vector = np.zeros((2,), dtype=np.float32)
        if action[2] == 1:
            direction_vec = rad_to_unit_vec(self.player_heading)
            movement_vector += direction_vec
        if self.allow_backward and action[3] == 1:
            direction_vec = rad_to_unit_vec(self.player_heading)
            movement_vector -= direction_vec
        if self.allow_strafe and action[4] == 1:
            direction_vec = rad_to_unit_vec(self.player_heading + (np.pi / 2))
            movement_vector += direction_vec
        if self.allow_strafe and action[5] == 1:
            direction_vec = rad_to_unit_vec(self.player_heading - (np.pi / 2))
            movement_vector += direction_vec
        # Clip speed
        movement_norm = np.sqrt(np.sum(movement_vector ** 2))
        if movement_norm > PLAYER_SPEED:
            movement_vector = (movement_vector / movement_norm) * PLAYER_SPEED
        self.player_position += movement_vector

    def _execute_continuous_action(self, action):
        """
        Move player according to continuous action in [-1, 1],
        mapping it to [0, 2*pi] and moving to that direction
        """
        action_in_rads = ((action + 1) / 2) * 2 * np.pi
        direction_vector = rad_to_unit_vec(action_in_rads)
        self.player_position += direction_vector * PLAYER_SPEED

    def _execute_multidiscrete_action(self, action):
        """
        Agent can press all buttons it wants, and
        movement is done accordingly.
        """
        # Sanity check: If none of the directions was chosen,
        # do not move.
        if sum(action) == 0:
            return

        # Sum all directions we want to go to,
        # and limit length
        selected_directions = list(map(bool, action))
        sum_direction = np.sum(self.directions[selected_directions], axis=0)
        direction_norm = np.sqrt(np.sum(sum_direction ** 2))

        if direction_norm < 1e-6:
            # If norm of the vector is too small, stay still.
            # There is some tiny machine-precision errors in the vectors,
            # which can blow up if we divide by too small of a number
            return

        unit_direction = sum_direction / direction_norm
        direction = unit_direction * PLAYER_SPEED

        self.player_position += direction

    def _execute_discrete_action(self, action):
        """
        Move player according to discrete actions, where
        action is one of {0,1,2,3}.
        """
        self.player_position += self.directions[action] * PLAYER_SPEED

    def _execute_action(self, action):
        """
        Move player according to chosen action, and
        check for boundaries
        """
        # Remove bogus actions
        if self.num_bogus_actions > 0:
            if self.action_space_type == "discrete":
                # Last num_bogus_actions are idles
                if action >= self.num_directions:
                    return
            elif self.action_space_type == "multidiscrete":
                # Remove last num_bogus_actions
                action = action[:self.num_directions]
            elif self.action_space_type == "tank-discrete":
                if action >= 4:
                    return
            elif self.action_space_type == "tank-multidiscrete":
                action = action[:4]
            else:
                # Continuous
                action = action[:1]

        # Move player according to action
        self.action_handler(action)

        # Clip boundaries
        np.clip(
            self.player_position,
            0,
            LENGTH_PER_AXIS,
            out=self.player_position
        )

    def step(self, action):
        self._execute_action(action)
        self.timestep += 1

        done = False
        reward = 0
        if self._test_if_reached_goal():
            done = True
            reward = self.goal_reward
        elif self.timestep >= self.timeout:
            done = True
            reward = self.timeout_reward

        obs = self._build_observation()

        return obs, reward, done, {}

    def reset(self):
        # Pick new goal
        self.goal_position = np.random.random(size=(2,)) * LENGTH_PER_AXIS
        # Pick new player positions until valid
        self.player_position = np.random.random(size=(2,)) * LENGTH_PER_AXIS
        while self._test_if_reached_goal():
            self.player_position = np.random.random(size=(2,)) * LENGTH_PER_AXIS
        # Pick new direction for player
        self.player_heading = random.random() * 2 * np.pi

        self.timestep = 0

        obs = self._build_observation()
        return obs

    def close(self):
        # Nothing to clean up, really
        pass

    def render(self, mode="pillow"):
        # Create a new Pillow image, where
        # the current situation is rendered
        assert mode == "pillow", "Only 'pillow' is supported for rendering."

        im = Image.new("RGB", [RENDER_RESOLUTION, RENDER_RESOLUTION], "black")
        draw = ImageDraw.Draw(im)

        coordinates_to_pixels = RENDER_RESOLUTION / LENGTH_PER_AXIS

        # Draw player
        player_radius = int((PLAYER_SIZE / LENGTH_PER_AXIS) * coordinates_to_pixels)
        player_location = (self.player_position / LENGTH_PER_AXIS) * coordinates_to_pixels
        draw.ellipse(
            (
                player_location[0] - player_radius,
                player_location[1] - player_radius,
                player_location[0] + player_radius,
                player_location[1] + player_radius,
            ),
            fill=PLAYER_COLOR
        )

        # Draw direction-line for player
        player_heading = rad_to_unit_vec(self.player_heading)
        player_heading = ((player_heading * PLAYER_SIZE) / LENGTH_PER_AXIS) * coordinates_to_pixels

        draw.line(
            (
                player_location[0],
                player_location[1],
                player_location[0] + player_heading[0],
                player_location[1] + player_heading[1],
            ),
            fill="green",
            width=2
        )

        # Draw goal
        goal_location = (self.goal_position / LENGTH_PER_AXIS) * coordinates_to_pixels
        draw.line(
            (
                goal_location[0],
                goal_location[1] - RENDER_GOAL_SIZE,
                goal_location[0],
                goal_location[1] + RENDER_GOAL_SIZE,
            ),
            fill=GOAL_COLOR
        )
        draw.line(
            (
                goal_location[0] - RENDER_GOAL_SIZE,
                goal_location[1],
                goal_location[0] + RENDER_GOAL_SIZE,
                goal_location[1],
            ),
            fill=GOAL_COLOR
        )

        return im

    def seed(self, seed):
        np.random.seed(seed)


def test_rendering():
    """
    Test rendering of a game of GetToGoal
    """
    TIMEOUT = 100
    env = GetToGoalContinuous(
        TIMEOUT,
        action_space_type="tank-multidiscrete",
        allow_strafe=True
    )
    imgs = []

    obs = env.reset()
    for i in range(TIMEOUT):
        imgs.append(env.render("pillow"))
        obs, reward, done, info = env.step(
            env.action_space.sample()
        )
        if done:
            env.reset()

    imgs[0].save(
        "gettogoal_render_test.gif",
        save_all=True,
        append_images=imgs[1:],
        duration=33,
        loop=0
    )


if __name__ == "__main__":
    test_rendering()
