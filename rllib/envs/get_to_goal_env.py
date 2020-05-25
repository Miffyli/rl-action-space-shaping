import logging
import os

import gym
from stable_baselines.bench import Monitor
from envs.get_to_goal_continuous import GetToGoalContinuous

logger = logging.getLogger(__name__)


class GetToGoalEnv(gym.Env):

    def __init__(self, env_config) -> None:
        super().__init__()
        self.env = gym.make(env_config['env_name'])

        if 'output' in env_config:
            monitor_file = os.path.join(
                env_config['output'], f"env_{env_config.worker_index:d}_{env_config.vector_index:d}")

            self.env = Monitor(self.env, monitor_file)

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def close(self):
        self.env.close()

    def render(self, mode='human'):
        self.env.render(mode)
