import os

from gym.envs.registration import register
from .get_to_goal_continuous import GetToGoalContinuous
from .vizdoom_environment import DoomEnvironment

ObstacleTowerEnv = None
try:
    from .obstacle_tower_env import ObstacleTowerEnv
except Exception:
    print("[Warning] Could not import ObstacleTower env (is it installed?)")


# --------------------------------------------------
# GetToGoal
# --------------------------------------------------

TIMEOUT = 100

# Default environments for GetToGoal
register(
    id='GetToGoal-Discrete-v0',
    entry_point="envs.get_to_goal_continuous:GetToGoalContinuous",
    kwargs={
        'timeout': TIMEOUT,
        'action_space_type': 'discrete',
    }
)

register(
    id='GetToGoal-MultiDiscrete-v0',
    entry_point="envs.get_to_goal_continuous:GetToGoalContinuous",
    kwargs={
        'timeout': TIMEOUT,
        'action_space_type': 'multidiscrete',
    }
)

register(
    id='GetToGoal-TankDiscrete-v0',
    entry_point="envs.get_to_goal_continuous:GetToGoalContinuous",
    kwargs={
        'timeout': TIMEOUT,
        'action_space_type': 'tank-discrete',
    }
)

register(
    id='GetToGoal-TankDiscrete-NoBackward-v0',
    entry_point="envs.get_to_goal_continuous:GetToGoalContinuous",
    kwargs={
        'timeout': TIMEOUT,
        'action_space_type': 'tank-discrete',
        'allow_backward': False
    }
)

register(
    id='GetToGoal-TankDiscrete-Strafe-v0',
    entry_point="envs.get_to_goal_continuous:GetToGoalContinuous",
    kwargs={
        'timeout': TIMEOUT,
        'action_space_type': 'tank-discrete',
        'allow_strafe': True
    }
)

register(
    id='GetToGoal-TankMultiDiscrete-v0',
    entry_point="envs.get_to_goal_continuous:GetToGoalContinuous",
    kwargs={
        'timeout': TIMEOUT,
        'action_space_type': 'tank-multidiscrete',
    }
)

register(
    id='GetToGoal-TankMultiDiscrete-NoBackward-v0',
    entry_point="envs.get_to_goal_continuous:GetToGoalContinuous",
    kwargs={
        'timeout': TIMEOUT,
        'action_space_type': 'tank-multidiscrete',
        'allow_backward': False
    }
)

register(
    id='GetToGoal-TankMultiDiscrete-Strafe-v0',
    entry_point="envs.get_to_goal_continuous:GetToGoalContinuous",
    kwargs={
        'timeout': TIMEOUT,
        'action_space_type': 'tank-multidiscrete',
        'allow_strafe': True
    }
)

register(
    id='GetToGoal-Continuous-v0',
    entry_point="envs.get_to_goal_continuous:GetToGoalContinuous",
    kwargs={
        'timeout': TIMEOUT,
        'action_space_type': 'continuous',
    }
)

# Environments for sweeping over different number of actions
# Super-duper elegant :'D

MIN_ACTIONS = 2
MAX_ACTIONS = 100
ACTION_STEP = 1

for num_directions in range(MIN_ACTIONS, MAX_ACTIONS + 1, ACTION_STEP):
    register(
        id='GetToGoal-Discrete%d-v0' % num_directions,
        entry_point="envs.get_to_goal_continuous:GetToGoalContinuous",
        kwargs={
            'timeout': TIMEOUT,
            'action_space_type': 'discrete',
            'num_directions': num_directions,
        }
    )

    register(
        id='GetToGoal-MultiDiscrete%d-v0' % num_directions,
        entry_point="envs.get_to_goal_continuous:GetToGoalContinuous",
        kwargs={
            'timeout': TIMEOUT,
            'action_space_type': 'multidiscrete',
            'num_directions': num_directions,
        }
    )

# Environments for sweeping over different number of bogus-actions

MIN_BOGUS = 0
MAX_BOGUS = 50

for num_bogus_actions in range(MIN_BOGUS, MAX_BOGUS + 1):
    register(
        id='GetToGoal-Discrete-Bogus%d-v0' % num_bogus_actions,
        entry_point="envs.get_to_goal_continuous:GetToGoalContinuous",
        kwargs={
            'timeout': TIMEOUT,
            'action_space_type': 'discrete',
            'num_bogus_actions': num_bogus_actions
        }
    )

    register(
        id='GetToGoal-MultiDiscrete-Bogus%d-v0' % num_bogus_actions,
        entry_point="envs.get_to_goal_continuous:GetToGoalContinuous",
        kwargs={
            'timeout': TIMEOUT,
            'action_space_type': 'multidiscrete',
            'num_bogus_actions': num_bogus_actions
        }
    )

    register(
        id='GetToGoal-TankDiscrete-Bogus%d-v0' % num_bogus_actions,
        entry_point="envs.get_to_goal_continuous:GetToGoalContinuous",
        kwargs={
            'timeout': TIMEOUT,
            'action_space_type': 'tank-discrete',
            'num_bogus_actions': num_bogus_actions
        }
    )

    register(
        id='GetToGoal-TankMultiDiscrete-Bogus%d-v0' % num_bogus_actions,
        entry_point="envs.get_to_goal_continuous:GetToGoalContinuous",
        kwargs={
            'timeout': TIMEOUT,
            'action_space_type': 'tank-multidiscrete',
            'num_bogus_actions': num_bogus_actions
        }
    )

    register(
        id='GetToGoal-Continuous-Bogus%d-v0' % num_bogus_actions,
        entry_point="envs.get_to_goal_continuous:GetToGoalContinuous",
        kwargs={
            'timeout': TIMEOUT,
            'action_space_type': 'continuous',
            'num_bogus_actions': num_bogus_actions
        }
    )


# --------------------------------------------------
# ViZDoom
# --------------------------------------------------

# A list of button sets available for navigation tasks,
# along with a "pretty name"
VIZDOOM_NAV_BUTTON_SETS = (
    ("minimal", "Minimal"),
    ("bare-minimum", "BareMinimum"),
    ("backward", "Backward"),
    ("strafe", "Strafe"),
    ("all", "All"),
)

# Different navigation envs (which only require buttons pointed above).
# In order "Pretty name", "path to scenario", "only screen buffer"
VIZDOOM_NAV_ENVS = (
    ("GetToGoal", os.path.join(os.path.dirname(__file__), "../doom_scenarios/get_to_goal.cfg"), True),
    ("HGS", os.path.join(os.path.dirname(__file__), "../doom_scenarios/health_gathering_supreme.cfg"), False),
)

for env_name, env_config, env_only_screen in VIZDOOM_NAV_ENVS:
    for vz_button_set, button_set_name in VIZDOOM_NAV_BUTTON_SETS:
        # Discrete action space, Maximum of one button down
        register(
            id='ViZDoom-%s-Discrete1-%s-v0' % (env_name, button_set_name),
            entry_point="envs.vizdoom_environment:DoomEnvironment",
            kwargs={
                'config': env_config,
                'allowed_buttons': vz_button_set,
                'action_space_type': 'discrete',
                'discrete_max_buttons_down': 1,
                'only_screen_buffer': env_only_screen
            }
        )

        register(
            id='ViZDoom-%s-Discrete2-%s-v0' % (env_name, button_set_name),
            entry_point="envs.vizdoom_environment:DoomEnvironment",
            kwargs={
                'config': env_config,
                'allowed_buttons': vz_button_set,
                'action_space_type': 'discrete',
                'discrete_max_buttons_down': 2,
                'only_screen_buffer': env_only_screen
            }
        )

        register(
            id='ViZDoom-%s-Discrete3-%s-v0' % (env_name, button_set_name),
            entry_point="envs.vizdoom_environment:DoomEnvironment",
            kwargs={
                'config': env_config,
                'allowed_buttons': vz_button_set,
                'action_space_type': 'discrete',
                'discrete_max_buttons_down': 3,
                'only_screen_buffer': env_only_screen
            }
        )

        register(
            id='ViZDoom-%s-DiscreteAll-%s-v0' % (env_name, button_set_name),
            entry_point="envs.vizdoom_environment:DoomEnvironment",
            kwargs={
                'config': env_config,
                'allowed_buttons': vz_button_set,
                'action_space_type': 'discrete',
                'discrete_max_buttons_down': 999,
                'only_screen_buffer': env_only_screen
            }
        )

        # And then ye-olde multi-discrete
        register(
            id='ViZDoom-%s-MultiDiscrete-%s-v0' % (env_name, button_set_name),
            entry_point="envs.vizdoom_environment:DoomEnvironment",
            kwargs={
                'config': env_config,
                'allowed_buttons': vz_button_set,
                'action_space_type': 'multidiscrete',
                'only_screen_buffer': env_only_screen
            }
        )

        # And multi-discrete with mouse.
        if button_set_name is not "BareMinimum":
            register(
                id='ViZDoom-%s-Mouse-%s-v0' % (env_name, button_set_name),
                entry_point="envs.vizdoom_environment:DoomEnvironment",
                kwargs={
                    'config': env_config,
                    'allowed_buttons': "mouse-" + vz_button_set,
                    'action_space_type': 'mouse',
                    'only_screen_buffer': env_only_screen
                }
            )


# Repeat same, but for environments that require
# shooting
VIZDOOM_ATTACK_BUTTON_SETS = (
    ("minimal-attack", "Minimal"),
    ("bare-minimum-attack", "BareMinimum"),
    ("backward-attack", "Backward"),
    ("strafe-attack", "Strafe"),
    ("all", "All"),
)

# Different shooting envs
VIZDOOM_ATTACK_ENVS = (
    ("Deathmatch", os.path.join(os.path.dirname(__file__), "../doom_scenarios/deathmatch.cfg"), False),
)

for env_name, env_config, env_only_screen in VIZDOOM_ATTACK_ENVS:
    for vz_button_set, button_set_name in VIZDOOM_ATTACK_BUTTON_SETS:
        # Discrete action space, Maximum of one button down
        register(
            id='ViZDoom-%s-Discrete1-%s-v0' % (env_name, button_set_name),
            entry_point="envs.vizdoom_environment:DoomEnvironment",
            kwargs={
                'config': env_config,
                'allowed_buttons': vz_button_set,
                'action_space_type': 'discrete',
                'discrete_max_buttons_down': 1,
                'only_screen_buffer': env_only_screen
            }
        )

        register(
            id='ViZDoom-%s-Discrete2-%s-v0' % (env_name, button_set_name),
            entry_point="envs.vizdoom_environment:DoomEnvironment",
            kwargs={
                'config': env_config,
                'allowed_buttons': vz_button_set,
                'action_space_type': 'discrete',
                'discrete_max_buttons_down': 2,
                'only_screen_buffer': env_only_screen
            }
        )

        register(
            id='ViZDoom-%s-Discrete3-%s-v0' % (env_name, button_set_name),
            entry_point="envs.vizdoom_environment:DoomEnvironment",
            kwargs={
                'config': env_config,
                'allowed_buttons': vz_button_set,
                'action_space_type': 'discrete',
                'discrete_max_buttons_down': 3,
                'only_screen_buffer': env_only_screen
            }
        )

        register(
            id='ViZDoom-%s-DiscreteAll-%s-v0' % (env_name, button_set_name),
            entry_point="envs.vizdoom_environment:DoomEnvironment",
            kwargs={
                'config': env_config,
                'allowed_buttons': vz_button_set,
                'action_space_type': 'discrete',
                'discrete_max_buttons_down': 999,
                'only_screen_buffer': env_only_screen
            }
        )

        # And then ye-olde multi-discrete
        register(
            id='ViZDoom-%s-MultiDiscrete-%s-v0' % (env_name, button_set_name),
            entry_point="envs.vizdoom_environment:DoomEnvironment",
            kwargs={
                'config': env_config,
                'allowed_buttons': vz_button_set,
                'action_space_type': 'multidiscrete',
                'only_screen_buffer': env_only_screen
            }
        )

        # And multi-discrete with mouse.
        if button_set_name is not "BareMinimum":
            register(
                id='ViZDoom-%s-Mouse-%s-v0' % (env_name, button_set_name),
                entry_point="envs.vizdoom_environment:DoomEnvironment",
                kwargs={
                    'config': env_config,
                    'allowed_buttons': "mouse-" + vz_button_set,
                    'action_space_type': 'mouse',
                    'only_screen_buffer': env_only_screen
                }
            )

#
# Obstacle Tower Env
#
if ObstacleTowerEnv is not None:
    OT_BUTTON_SETS = [
        "full",
        "minimal",
        "backward",
        "strafe",
        "always-forward"
    ]
    OT_BUTTON_SET_NAMES = [
        "Full",
        "Minimal",
        "Backward",
        "Strafe",
        "AlwaysForward"
    ]

    for button_set, button_set_name in zip(OT_BUTTON_SETS, OT_BUTTON_SET_NAMES):
        register(
            id='ObstacleTower-{}-Discrete-v0'.format(button_set_name),
            entry_point="envs.obstacle_tower_env:ObstacleTowerEnv",
            kwargs={
                'environment_filename': "envs/ObstacleTower/obstacletower",
                'retro': True,
                'realtime_mode': False,
                'button_set': button_set,
                'multidiscrete': False,
                'config': {"total-floors": 6, "dense-reward": 1}
            }
        )

        register(
            id='ObstacleTower-{}-MultiDiscrete-v0'.format(button_set_name),
            entry_point="envs.obstacle_tower_env:ObstacleTowerEnv",
            kwargs={
                'environment_filename': "envs/ObstacleTower/obstacletower",
                'retro': True,
                'realtime_mode': False,
                'button_set': button_set,
                'multidiscrete': True,
                'config': {"total-floors": 6, "dense-reward": 1}
            }
        )
