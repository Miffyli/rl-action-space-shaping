#  atari_environment.py
#  Atari environments for action-space shaping tests,
#  where we try minimal action space (default), full action
#  space and multi-discrete action spaces.
#

import gym
from gym import spaces

from stable_baselines.common.atari_wrappers import NoopResetEnv, MaxAndSkipEnv, wrap_deepmind
from stable_baselines.bench.monitor import Monitor

# Original meaning-mapping
# for full atari action space.
# From:
#   https://github.com/openai/gym/blob/master/gym/envs/atari/atari_env.py
ACTION_MEANING = {
    0: "NOOP",
    1: "FIRE",
    2: "UP",
    3: "RIGHT",
    4: "LEFT",
    5: "DOWN",
    6: "UPRIGHT",
    7: "UPLEFT",
    8: "DOWNRIGHT",
    9: "DOWNLEFT",
    10: "UPFIRE",
    11: "RIGHTFIRE",
    12: "LEFTFIRE",
    13: "DOWNFIRE",
    14: "UPRIGHTFIRE",
    15: "UPLEFTFIRE",
    16: "DOWNRIGHTFIRE",
    17: "DOWNLEFTFIRE",
}

# A conversion-table for
# multidiscrete (joystick + fire-button) actions
# into a discrete action, as above.
# First dimension is for fire button (n=2),
# and second is for joystick (n=9).
# To be used like,
#   MULTIDISCRETE_TO_DISCRETE[fire_action][joystick_action]
MULTIDISCRETE_TO_DISCRETE = [
    [       # No FIRE
        0,  # NOOP
        2,  # UP
        3,  # RIGHT
        4,  # LEFT
        5,  # DOWN
        6,  # UPRIGHT
        7,  # UPLEFT
        8,  # DOWNRIGHT
        9   # DOWNLEFT
    ], [     # FIRE
        1,   # FIRE
        10,  # UPFIRE
        11,  # RIGHTFIRE
        12,  # LEFTFIRE
        13,  # DOWNFIRE
        14,  # UPRIGHTFIRE
        15,  # UPLEFTFIRE
        16,  # DOWNRIGHTFIRE
        17   # DOWNLEFTFIRE
    ]
]


class AtariActionsToMultiDiscrete(gym.Wrapper):
    """
    Turns full Atari action-spaces into multidiscrete, one
    discrete for the joystick and another for the fire-button.
    """

    def __init__(self, env):
        super().__init__(env)
        self.env = env

        if not (isinstance(env.action_space, spaces.Discrete) and env.action_space.n == 18):
            raise ValueError("Environment must have an action-space of Discrete n=18")

        self.action_space = spaces.MultiDiscrete([9, 2])

    def step(self, action):
        # Turn multidiscrete back into original action
        action = MULTIDISCRETE_TO_DISCRETE[action[1]][action[0]]
        return self.env.step(action)


def make_atari(env_id, monitor_path, full_action_space=False, multidiscrete=False):
    """
    Create a wrapped atari Environment, with all the bells and whistles
    added by Deepmind.

    Nicked from stable-baselines atari_wrappers.py.

    :param env_id: (str) the environment ID
    :param full_action_space: (bool) Whether to have full action space
    :param multidiscrete: (bool) Use multi-discrete action-space
    :return: (Gym Environment) the wrapped atari environment
    """
    assert 'NoFrameskip' in env_id
    if not full_action_space and multidiscrete:
        raise ValueError("Must enable full action-space for multidiscrete actions")
    env = gym.make(env_id, full_action_space=full_action_space)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    # Add monitor here, to avoid mess with
    # reward clipping and episodic lives
    env = Monitor(env, monitor_path)
    env = wrap_deepmind(env, frame_stack=True)
    # Add multi-discrete wrapper in the end,
    # just in case one of the above does not work
    # with multi-discrete actions
    if multidiscrete:
        env = AtariActionsToMultiDiscrete(env)
    return env
