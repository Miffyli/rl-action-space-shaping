# Run experiments with rllib
import os
from argparse import ArgumentParser

import gym
import ray
import yaml
from ray.rllib.models import ModelCatalog
from ray.tune import run_experiments
from ray.tune.logger import Logger, TBXLogger

from rllib.envs.get_to_goal_env import GetToGoalEnv
from rllib.models.doom_nature_model import DoomNatureModel

parser = ArgumentParser("Run rllib action-space experiments.")
parser.add_argument("--config_file", type=str, default='./rllib/configs/vizdoom_ppo.yaml',
                    help="Path to config yaml-file.")


class CSVLogger(Logger):
    def _init(self):
        self._config = None
        self.out_files_dict = {}

    def on_result(self, result):
        experiment_tag = result.get('experiment_tag', 'no_experiment_tag')
        experiment_id = result.get('experiment_id', 'no_experiment_id')
        out_file = self.out_files_dict.get(experiment_tag)
        if out_file is None:
            config = result.get("config")
            os.makedirs(config['env_config']['logging_path'], exist_ok=True)
            out_file_path = os.path.join(
                config['env_config']['logging_path'], f"{config['env_config']['env_name']}-{experiment_id}")
            out_file = open(out_file_path, "a")
            self.out_files_dict[experiment_tag] = out_file
        out_file.write(f"{result['timesteps_total']}\t{result['episode_reward_mean']}\n")
        out_file.flush()

    def flush(self):
        for out_file in self.out_files_dict.values():
            out_file.flush()

    def close(self):
        for out_file in self.out_files_dict.values():
            out_file.close()


def make_env(id):
    # noinspection PyUnresolvedReferences
    import envs
    return gym.make(id)


def run_experiment(args):
    with open(args.config_file) as f:
        experiments = yaml.safe_load(f)

    ray.tune.register_env("doom", lambda env_config: make_env(env_config['env_name']))
    ray.tune.register_env("get_to_goal_env", lambda config: GetToGoalEnv(config))

    ModelCatalog.register_custom_model("doom_nature_model", DoomNatureModel)

    for exp in experiments.values():
        exp['loggers'] = [CSVLogger, TBXLogger]

    ray.init()
    ray.tune.run_experiments(experiments)


if __name__ == "__main__":
    args = parser.parse_args()
    run_experiment(args)
