# Bunch of misc utils who have nowhere else to go, poor fellahs...
import os
import json
from glob import glob

import numpy as np


def load_and_interleave_monitor_csvs(paths_to_csvs):
    """
    Load bunch of stable-baselines Monitor .csv files,
    interleave them by timestamps and return a Dictionary
    with keys "time", "rewards" and "length", each
    mapping to a ndarray of corresponding values at different
    episodes.

    Using VecEnvs with Monitors creates multiple Monitor files
    for one training run, so we need this interleaving to get
    one "learning curve".

    Arguments:
        paths_to_csvs (List of str): Paths to .csv files to load

    Returns:
        loaded_values (Dict): Dict with keys "time", "rewards" and
                              "length"
    """
    csv_datas = []
    smallest_timestamp = np.inf
    for path_to_csv in paths_to_csvs:
        # Load data as np array
        csv_data = None
        try:
            csv_data = np.genfromtxt(
                path_to_csv,
                names=True,
                skip_header=1,
                delimiter=","
            )
        except Exception as e:
            print("Could not load file %s" % path_to_csv)
            print(e)
        # Load the header as JSON, so we get the starting time.
        # Bit of waste to read the whole file, but oh well...
        # Also remove first character "#"
        header_json = open(path_to_csv).readlines()[0][1:]
        start_timestamp = json.loads(header_json)["t_start"]

        # Add this timestamp to timestamps in monitor
        csv_data["t"] += start_timestamp

        # Aaand track the smallest start_timestamp, which
        # we will use as time zero
        if start_timestamp < smallest_timestamp:
            smallest_timestamp = start_timestamp

        csv_datas.append(csv_data)

    # Interleave by concatenating everything and
    # then sorting. Super effecient :D

    all_data = []

    for csv_data in csv_datas:
        all_data.extend(
            zip(*[csv_data[key].tolist() for key in ["r", "l", "t"]])
        )
    # Sort by "t" (time)
    all_data = sorted(all_data, key=lambda x: x[2])
    # Unzip, convert back to ndarrays and return as dict
    rewards, lengths, timesteps = zip(*all_data)

    return_dict = {
        "rewards": np.array(rewards),
        "lengths": np.array(lengths),
        "timesteps": np.array(timesteps),
    }

    # Subtract the smallest timestep to "start from zero",
    # like monitor files do
    return_dict["timesteps"] -= smallest_timestamp

    return return_dict


def load_experiment_monitor(experiment_path):
    """
    Load monitor data from one experiment directory.

    Arguments:
        experiment_path (str): Path to the experiment directory

    Returns:
        loaded_values (Dict): Dict with keys "time", "rewards" and
                              "length"
    """
    csvs = glob(os.path.join(experiment_path, "*.csv"))
    return load_and_interleave_monitor_csvs(csvs)


def load_experiment(experiment_path):
    """
    Load a single learning-curve result, either a folder
    with bunch of csv files (stable-baselines Monitor outputs),
    or tsv-like files with each line being "[num_steps] [average_reward]

    Arguments:
        experiment_path (str): Path to the experiment to be loaded.
                               If a directory, assume it contains
                               bunch of stable-baselines monitor files.
                               If a file, assume it is a tsv file with
                               structure "[num_steps] [average_reward]"

    Returns:
        loaded_values (Dict): Dict with keys "steps" and "rewards",
                              "steps" being an array of number of steps
                              trained, and "rewards" corresponding
                              average episodic rewards
    """

    if os.path.isdir(experiment_path):
        # Bunch of stable-baselines Monitor CSVs
        data = load_experiment_monitor(experiment_path)
        # Turn episode lengths into numbers of steps
        steps = np.cumsum(data["lengths"])
        return_dict = {
            "steps": steps,
            "rewards": data["rewards"]
        }
        return return_dict
    else:
        # Assume a tsv file
        data = np.loadtxt(experiment_path)
        # Bit of preprocessing: In some cases,
        # steps has multiple instances, in which
        # case take an average.
        steps = []
        rewards = []
        raw_steps = data[:, 0]
        raw_rewards = data[:, 1]
        unique_steps = np.sort(np.unique(raw_steps))
        for unique_step in unique_steps:
            steps.append(unique_step)
            rewards.append(raw_rewards[raw_steps == unique_step].mean())

        return_dict = {
            "steps": np.array(steps),
            "rewards": np.array(rewards)
        }
        return return_dict


def find_solved_point(experiment_data, window_size=20, solved_threshold=1):
    """
    Find number of steps when agent has "solved" the environment,
    i.e. reached >= solved_threshold reward for window_size times

    Arguments:
        experiment_data: Loaded Monitor data (see `load_experiment`)
        window_size (int): How many successive games have to have
                           reward equal or above to solved_threshold
                           before environment is considered solved.
        solved_threshold (float): Reward threshold for considering
                                  environment "solved"

    Returns:
        agent_steps: Number of steps it took to solve the environment,
                     or None if environment was not solved
    """
    rewards_above_threshold = experiment_data["rewards"] >= solved_threshold
    successive_solves = np.convolve(np.ones((window_size,)), rewards_above_threshold, "same")
    solve_indexes = np.where(successive_solves >= window_size)[0]
    if len(solve_indexes) == 0:
        # Was not solved at any point
        return None
    else:
        # Return the number of steps where environment was solved
        solved_idx = solve_indexes[0]
        return experiment_data["steps"][solved_idx]
