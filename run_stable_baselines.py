# Run experiments with stable-baselines agents
import os
from argparse import ArgumentParser
import time
import random

import numpy as np
import gym
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.bench import Monitor
from stable_baselines.common.policies import MlpPolicy, CnnPolicy, MlpLstmPolicy, CnnLstmPolicy
from stable_baselines import PPO2, A2C
from stable_baselines.common.atari_wrappers import ClipRewardEnv, FrameStack

from envs.get_to_goal_continuous import GetToGoalContinuous
from envs.vizdoom_environment import AppendFeaturesToImageWrapper
from envs.atari_environment import make_atari
from stable_baselines_utils import create_augmented_nature_cnn

try:
    from envs.obstacle_tower_env import OTEpisodicFloors
except Exception:
    print("[Warning] Could not import ObstacleTower environment (is it installed?)")

parser = ArgumentParser("Run stable-baselines action-space experiments.")
parser.add_argument("--output", type=str, required=True, help="Directory where to put results.")
parser.add_argument("--env", required=True, help="Environment to play.")
parser.add_argument("--timesteps", type=int, required=True, help="How many timesteps to run.")
parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate for PPO2.")
parser.add_argument("--ent-coef", type=float, default=0.01, help="Weight of entropy loss.")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
parser.add_argument("--n-steps", type=int, default=256, help="n_steps for PPO2/A2C.")
parser.add_argument("--cliprange", type=float, default=0.2, help="cliprange for PPO2.")
parser.add_argument("--cliprange-vf", type=float, default=None, help="cliprange-vf for PPO2.")
parser.add_argument("--num-envs", type=int, default=8, help="Number of parallel environments.")
parser.add_argument("--net-arch", type=int, nargs="*", default=[64, 64], help="Network layers for MLPPolicy.")
parser.add_argument("--subprocenv", action="store_true", help="Use Subprocvecenv rather than dummy vecenv.")
parser.add_argument("--cnn", action="store_true", help="Use CNN policy rather than MlpPolicy.")
parser.add_argument("--num-direct-features", type=int, default=None, 
                    help="Number of features in second observation, for tuple observations")

# Arguments specific to Atari
parser.add_argument("--atari-full-actions", action="store_true", help="Use full actions of an Atari game.")
parser.add_argument("--atari-multidiscrete", action="store_true", help="Use multidiscrete action-space.")
parser.add_argument("--atari-ppo", action="store_true", help="Use PPO parameters from rl-baselines-zoo for Atari.")


def create_env(args, idx):
    """
    Create and return an environment according to args (parsed arguments).
    idx specifies idx of this environment among parallel environments.
    """
    env = None
    monitor_file = os.path.join(args.output, ("env_%d" % idx))

    if "NoFrameskip" in args.env:
        # Use "NoFrameskip" to detect Atari envs.
        # Monitor is added in make_atari
        env = make_atari(
            args.env,
            monitor_file,
            full_action_space=args.atari_full_actions,
            multidiscrete=args.atari_multidiscrete
        )
    else:
        if "ObstacleTower" in args.env:
            # Sleep for a moment in case we use subprocvecenv,
            # so that we interleave booting of environments
            time.sleep(random.random() * 20)
            # If ObstacleTower env, give index as worker_id
            # so we can launch multiple envs
            env = gym.make(args.env, worker_id=idx)
        else:
            env = gym.make(args.env)

        # Check observation space. We expect it to be single Box
        # (poor stable-baselines can't hold more spaces).
        # If a Tuple, assume it is of shape (image_obs, feature_obs),
        # and try using wrapper to put feature_obs on
        # image obs
        if isinstance(env.observation_space, gym.spaces.Tuple):
            if args.num_direct_features is None:
                raise ValueError(
                    "Encountered Tuple obs-space, but no number of direct features provided. " +
                    "Space of the second observation: %s" % str(env.observation_space[1])
                )
            env = AppendFeaturesToImageWrapper(env)

        env = Monitor(env, monitor_file)

        # If obstacle-tower, add episodic-floors here after
        # monitor to get correct monitoring
        if "ObstacleTower" in args.env:
            env = OTEpisodicFloors(env)

    return env


def run_experiment(args):
    vecEnv = []
    for i in range(args.num_envs):
        # Bit of trickery here to avoid referencing
        # to the same "i"
        vecEnv.append((
            lambda idx: lambda: create_env(args, idx))(i)
        )

    if args.subprocenv:
        vecEnv = SubprocVecEnv(vecEnv)
    else:
        vecEnv = DummyVecEnv(vecEnv)

    policy = None
    policy_kwargs = {}
    if args.cnn:
        policy = CnnPolicy
    else:
        policy = MlpPolicy
        policy_kwargs = {"net_arch": args.net_arch}

    if args.num_direct_features is not None:
        # Create new network to handle the
        # "augmented" observation space, where
        # image observation's last channel will
        # contain "direct features".
        policy_kwargs["cnn_extractor"] = create_augmented_nature_cnn(
            args.num_direct_features
        )

    agent = None
    if args.atari_ppo:
        # Hardcoded parameters from rl-baselines-zoo from here:
        #  https://github.com/araffin/rl-baselines-zoo/blob/master/hyperparams/ppo2.yml
        agent = PPO2(
            policy,
            vecEnv,
            ent_coef=0.01,
            n_steps=128,
            gamma=0.99,
            lam=0.95,
            verbose=1,
            nminibatches=4,
            noptepochs=4,
            policy_kwargs=policy_kwargs,
            # Progression is "other way around",
            # 1.0 being at the beginning of training.
            cliprange=lambda progression: 0.1 * progression,
            learning_rate=lambda progression: 2.5 * 1e-4 * progression,
            vf_coef=0.5,
            cliprange_vf=-1
        )
    else:
        agent = PPO2(
            policy,
            vecEnv,
            ent_coef=args.ent_coef,
            n_steps=args.n_steps,
            gamma=args.gamma,
            verbose=1,
            policy_kwargs=policy_kwargs,
            cliprange=args.cliprange,
            cliprange_vf=args.cliprange_vf,
            learning_rate=args.lr
        )

    agent.learn(total_timesteps=args.timesteps)

    vecEnv.close()

    # Dirty hack for OT:
    # For some reason a process stays alive
    # after training is done, and the script
    # never quits properly.
    # Hack: Write a file to tell we are done,
    #       and outside script will kill us.
    #       This WILL NOT work with multiple
    #       running instances and whatnot!
    if "ObstacleTower" in args.env:
        with open("_done_training", "w") as f:
            f.write("Just something dummy")


if __name__ == "__main__":
    args = parser.parse_args()
    run_experiment(args)
