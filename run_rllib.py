# Run experiments with rllib
from argparse import ArgumentParser

import gym
import ray
import yaml
from ray.rllib.models import ModelCatalog
from ray.tune import run_experiments

from rllib.envs.get_to_goal_env import GetToGoalEnv
from rllib.models.doom_nature_model import DoomNatureModel

parser = ArgumentParser("Run rllib action-space experiments.")
parser.add_argument("--config_file", type=str, default='./rllib/configs/vizdoom_ppo.yaml',
                    help="Path to config yaml-file.")


def make_env(id):
    import envs
    return gym.make(id)


def run_experiment(args):
    with open(args.config_file) as f:
        experiments = yaml.safe_load(f)

    ray.tune.register_env("doom", lambda env_config: make_env(env_config['env_name']))
    ray.tune.register_env("get_to_goal_env", lambda config: GetToGoalEnv(config))

    ModelCatalog.register_custom_model("doom_nature_model", DoomNatureModel)

    ray.init()
    ray.tune.run_experiments(experiments)


if __name__ == "__main__":
    args = parser.parse_args()
    run_experiment(args)
