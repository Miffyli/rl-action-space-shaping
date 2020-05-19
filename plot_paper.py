# A _very_ throw-away script for
# plotting the figures of the paper.

import argparse
import os
from glob import glob

import numpy as np
from matplotlib import pyplot

from utils import load_experiment, find_solved_point

# This code assumes very specific structure for the input directory.
# Also ad-hoc-ity ensues! This is meant to be throw-awayish code,
# so there is a lot duplication going on!

# Structure of the input directory:
#   stable_baselines*/
#       Experiments done with stable_baselines (a lot of directories)
#   rllib-vizdoom/
#       - ViZDoom-Deathmatch-Mouse-Minimal-v0_0
#       - ViZDoom-Deathmatch-Mouse-Minimal-v0_1
#       - ...
#       ViZDoom experiments done with Rllib
#   rllib-sc2/
#       - BuildMarines/
#       - CollectMineralsAndGas/
#       - CollectMineralShards/
#       - DefeatRoaches/
#       SC2 experiments with Rllib

parser = argparse.ArgumentParser("Plot all paper plots")
parser.add_argument("input", type=str, help="Directory where all experiments reside (see comments of this file for structure).")
parser.add_argument("output", type=str, help="Directory where to put all the figures.")
parser.add_argument("--smooth", type=int, default=50, help="Rolling average smoothing per curve (default: 50).")
parser.add_argument("--interp-points", type=int, default=100, help="Number of points used for interpolation (default: 100).")

# For plotting GetToGoal plot
parser.add_argument("--reward-threshold", type=float, default=1, help="What reward is considered 'solved'")
parser.add_argument("--window-size", type=int, default=20, help="How many successive 'solved' games are needed before we call it solved")


def rolling_average(x, window_length):
    """
    Do rolling average on vector x with window
    length window_length.

    Values in the beginning of the array are smoothed
    over only the valid number of samples
    """
    new_x = np.convolve(np.ones((window_length,)) / window_length, x, mode="valid")
    # Add missing points to the beginning
    num_missing = len(x) - len(new_x)
    new_x = np.concatenate(
        (np.cumsum(x[:num_missing]) / (np.arange(num_missing) + 1), new_x)
    )
    return new_x


def interpolate_and_average(xs, ys, interp_points=None):
    """
    Average bunch of repetitions (xs, ys)
    into one curve. This is done by linearly interpolating
    y values to same basis (same xs). Maximum x of returned
    curve is smallest x of repetitions.

    If interp_points is None, use maximum number of points
    in xs as number of points to interpolate. If int, use this
    many interpolation points.

    Returns [new_x, mean_y, std_y]
    """
    # Get the xs of shortest curve
    max_min_x = max(x.min() for x in xs)
    min_max_x = min(x.max() for x in xs)
    if interp_points is None:
        # Interop points according to curve with "max resolution"
        interp_points = max(x.shape[0] for x in xs)

    new_x = np.linspace(max_min_x, min_max_x, interp_points)
    new_ys = []

    for old_x, old_y in zip(xs, ys):
        new_ys.append(np.interp(new_x, old_x, old_y))

    # Average out
    # atleast_2d for case when we only have one reptition
    new_ys = np.atleast_2d(np.array(new_ys))
    new_y = np.mean(new_ys, axis=0)
    std_y = np.std(new_ys, axis=0)

    return new_x, new_y, std_y


def plot_vizdoom(args):
    # Y-axis for grid
    envs = ["ViZDoom-GetToGoal-{space}-{buttons}-v0", "ViZDoom-HGS-{space}-{buttons}-v0", "ViZDoom-Deathmatch-{space}-{buttons}-v0"]
    env_names = ["GetToGoal", "HGS", "Deathmatch"]
    # X-axis for grid
    button_sets = ["BareMinimum", "Minimal", "Backward", "Strafe"]
    button_set_names = ["Bare Minimum", "Minimal", "Backward", "Strafe"]
    action_spaces = ["Discrete1", "Discrete2", "DiscreteAll", "MultiDiscrete"]
    action_space_names = ["Discrete (n=1)", "Discrete (n=2)", "Discrete (All)", "Multi-Discrete", "Mouse"]
    # For each env and button set, load all spaces.
    # This will be of shape (num_envs, num_button_sets, num_action_spaces),
    # each of which contains (steps, mean_reward, std_reward)
    curves_grid = []
    for env in envs:
        curves_env = []
        for button_set in button_sets:
            curves = []
            for action_space in action_spaces:
                # Find all repetitions for this experiment
                glob_str = os.path.join(
                    args.input,
                    "stable_baselines_" +
                    env.format(space=action_space, buttons=button_set) +
                    "*"
                )
                repetitions = glob(glob_str)
                xs = []
                ys = []
                for repetition in repetitions:
                    data = load_experiment(repetition)
                    # Cut first 10 steps, so that first
                    # point has sensible reward
                    xs.append(data["steps"][10:])
                    ys.append(rolling_average(data["rewards"], args.smooth)[10:])
                x, y, std = interpolate_and_average(xs, ys, interp_points=args.interp_points)
                curves.append((x, y, std))
            curves_env.append(curves)
        curves_grid.append(curves_env)

    # Now add the mouse experiments. Add them to corresponding button sets
    for env_idx in range(len(envs)):
        env_curves = curves_grid[env_idx]
        for button_set_idx in range(len(button_sets)):
            button_set = button_sets[button_set_idx]
            if button_set == "BareMinimum":
                # Not included in the experiments
                continue
            glob_str = os.path.join(
                args.input,
                "rllib-vizdoom",
                envs[env_idx].format(space="Mouse", buttons=button_set) +
                "*"
            )
            repetitions = glob(glob_str)
            xs = []
            ys = []
            for repetition in repetitions:
                data = load_experiment(repetition)
                # Cut first 10 steps, so that first
                # point has sensible reward
                xs.append(data["steps"][10:])
                ys.append(rolling_average(data["rewards"], args.smooth)[10:])
            x, y, std = interpolate_and_average(xs, ys, interp_points=args.interp_points)
            env_curves[button_set_idx].append((x, y, std))

    fig, axs = pyplot.subplots(
        nrows=3,
        ncols=4,
        sharey="row",
        sharex="row",
        figsize=[2 * 6.4, 2 * 4.8]
    )

    for y in range(len(envs)):
        for x in range(len(button_sets)):
            ax = axs[y, x]
            ax.grid(alpha=0.2)
            ax.ticklabel_format(axis="x", scilimits=(0, 0))
            # Plot lines
            for curve in curves_grid[y][x]:
                x_points, y_points, std = curve
                ax.plot(x_points, y_points)
                ax.fill_between(
                    x_points,
                    y_points + std,
                    y_points - std,
                    alpha=0.2
                )
            # Include legend of experiments only to first plot
            if x == 0 and y == 0:
                # Bit of a hack: BareMinimum does not have a Mouse experiment,
                # so create a ghost line, add legends and remove line
                ghost_line, = ax.plot(x_points, y_points)
                ax.legend(action_space_names)
                ghost_line.remove()
            # If first column, include environment name
            if x == 0:
                ax.set_ylabel(env_names[y] + "\nepisodic reward", fontsize="large")
            # If last row, include x-axis
            if y == (len(envs) - 1):
                ax.set_xlabel("Agent steps")
            # If first row, include button set name
            if y == 0:
                ax.set_title(button_set_names[x])

    fig.tight_layout()
    fig.savefig(os.path.join(args.output, "vizdoom.pdf"), bbox_inches="tight", pad_inches=0.0)


def plot_atari(args):
    # Dictionary mapping human-like game name to
    # all experiment names, that should be plotted in that
    # figure.
    # NOTE: The ordering matters! Colors are assigned in this order!
    LEGEND = ["Multi-Discrete", "Full", "Minimal"]
    envs = {
        "SpaceInvaders": (
            "SpaceInvadersNoFrameskip-v4-MultiDiscrete",
            "SpaceInvadersNoFrameskip-v4-Full",
            "SpaceInvadersNoFrameskip-v4-Minimal",
        ),
        "MsPacman": (
            "MsPacmanNoFrameskip-v4-MultiDiscrete",
            "MsPacmanNoFrameskip-v4-Full",
            "MsPacmanNoFrameskip-v4-Minimal",
        ),
        "Breakout": (
            "BreakoutNoFrameskip-v4-MultiDiscrete",
            "BreakoutNoFrameskip-v4-Full",
            "BreakoutNoFrameskip-v4-Minimal",
        ),
        "Gravitar": (
            "GravitarNoFrameskip-v4-MultiDiscrete",
            "GravitarNoFrameskip-v4-Full",
        ),
        "Enduro": (
            "EnduroNoFrameskip-v4-MultiDiscrete",
            "EnduroNoFrameskip-v4-Full",
            "EnduroNoFrameskip-v4-Minimal",
        ),
        "Q*bert": (
            "QbertNoFrameskip-v4-MultiDiscrete",
            "QbertNoFrameskip-v4-Full",
            "QbertNoFrameskip-v4-Minimal",
        ),
    }

    env_names = []
    env_curves = []
    for env_name, env_experiments in envs.items():
        env_names.append(env_name)
        curves = []
        for env_experiment in env_experiments:
            # Find all repetitions for this experiment
            glob_str = os.path.join(
                args.input,
                "stable_baselines_" +
                env_experiment +
                "*"
            )
            repetitions = glob(glob_str)
            xs = []
            ys = []
            for repetition in repetitions:
                data = load_experiment(repetition)
                xs.append(data["steps"])
                ys.append(rolling_average(data["rewards"], args.smooth))
            x, y, std = interpolate_and_average(xs, ys, interp_points=args.interp_points)
            curves.append((x, y, std))
        env_curves.append(curves)

    fig, axs = pyplot.subplots(nrows=2, ncols=3, sharex="all", figsize=[1.5 * 6.4, 1.5 * 4.8])

    for y in range(2):
        for x in range(3):
            idx = y * 3 + x
            ax = axs[y, x]
            curves = env_curves[idx]
            title = env_names[idx]
            ax.set_title(title, fontsize="x-large")

            ax.ticklabel_format(axis="x", scilimits=(0, 0))
            ax.tick_params(axis='both', which='both', labelsize="large")
            # Plot lines
            for curve in curves:
                x_points, y_points, std = curve
                ax.plot(x_points, y_points)
                ax.fill_between(
                    x_points,
                    y_points + std,
                    y_points - std,
                    alpha=0.2
                )
            # Only include legend in first plot
            if x == 0 and y == 0:
                ax.legend(["Multi-Discrete", "Full", "Minimal"], prop={"size": "large"})
            # If first column, include y-label
            if x == 0:
                ax.set_ylabel("Episodic reward", fontsize="x-large")
            # If last row, include x-axis
            if y == 1:
                ax.set_xlabel("Agent steps", fontsize="x-large")

    fig.tight_layout()
    fig.savefig(os.path.join(args.output, "atari.pdf"), bbox_inches="tight", pad_inches=0.0)


def plot_sc2(args):
    # Dictionary mapping human-like game name to
    # all experiment names, that should be plotted in that
    # figure.
    # NOTE: The ordering matters! Colors are assigned in this order!
    LEGEND = [
        "Full",
        "Masked",
        "Minimal",
        "Masked + AR",
        "Minimal + AR",
    ]
    envs = {
        "CollectMineralShards": (
            "IMPALA + NOMASK",
            "IMPALA",
            "IMPALA + MIN",
            "IMPALA + AR",
            "IMPALA + AR + MIN",
        ),
        "DefeatRoaches": (
            "IMPALA + NOMASK",
            "IMPALA",
            "IMPALA + MIN",
            "IMPALA + AR",
            "IMPALA + AR + MIN",
        ),
        "CollectMineralsAndGas": (
            "IMPALA + NOMASK",
            "IMPALA",
            "IMPALA + MIN",
            "IMPALA + AR",
            "IMPALA + AR + MIN",
        ),
        "BuildMarines": (
            "IMPALA + NOMASK",
            "IMPALA",
            "IMPALA + MIN",
            "IMPALA + AR",
            "IMPALA + AR + MIN",
        ),
    }

    env_names = []
    env_curves = []
    for env_name, env_experiments in envs.items():
        env_names.append(env_name)
        curves = []
        for env_experiment in env_experiments:
            # Find all repetitions for this experiment
            glob_str = os.path.join(
                args.input,
                "rllib-sc2",
                env_name,
                env_experiment + "_*"
            )
            repetitions = glob(glob_str)
            xs = []
            ys = []
            for repetition in repetitions:
                data = load_experiment(repetition)
                xs.append(data["steps"])
                ys.append(rolling_average(data["rewards"], args.smooth))
            x, y, std = interpolate_and_average(xs, ys, interp_points=args.interp_points)
            curves.append((x, y, std))
        env_curves.append(curves)

    fig, axs = pyplot.subplots(nrows=2, ncols=2, sharex="all", figsize=[1.5 * 6.4, 1.5 * 4.8])

    for y in range(2):
        for x in range(2):
            idx = y * 2 + x
            ax = axs[y, x]
            curves = env_curves[idx]
            title = env_names[idx]
            ax.set_title(title, fontsize="x-large")

            ax.ticklabel_format(axis="x", scilimits=(0, 0))
            ax.tick_params(axis='both', which='both', labelsize="large")
            # Plot lines
            for curve in curves:
                x_points, y_points, std = curve
                ax.plot(x_points, y_points)
                ax.fill_between(
                    x_points,
                    y_points + std,
                    y_points - std,
                    alpha=0.2
                )
            # Only include legend in first plot
            if x == 0 and y == 0:
                ax.legend(LEGEND, prop={"size": "large"})
            # If first column, include y-label
            if x == 0:
                ax.set_ylabel("Episodic reward", fontsize="x-large")
            # If last row, include x-axis
            if y == 1:
                ax.set_xlabel("Agent steps", fontsize="x-large")

    fig.tight_layout()
    fig.savefig(os.path.join(args.output, "sc2.pdf"), bbox_inches="tight", pad_inches=0.0)


def plot_ot(args):
    # Dictionary mapping human-like game name to
    # all experiment names, that should be plotted in that
    # figure.
    # NOTE: The ordering matters! Colors are assigned in this order!
    LEGEND = [
        "Always Forward",
        "Minimal",
        "Minimal + Backward",
        "Minimal + Strafe",
        "Full"
    ]
    envs = {
        "Discrete": (
            "ObstacleTower-AlwaysForward-Discrete-v0",
            "ObstacleTower-Minimal-Discrete-v0",
            "ObstacleTower-Backward-Discrete-v0",
            "ObstacleTower-Strafe-Discrete-v0",
            "ObstacleTower-Full-Discrete-v0",
        ),
        "Multi-Discrete": (
            "ObstacleTower-AlwaysForward-MultiDiscrete-v0",
            "ObstacleTower-Minimal-MultiDiscrete-v0",
            "ObstacleTower-Backward-MultiDiscrete-v0",
            "ObstacleTower-Strafe-MultiDiscrete-v0",
            "ObstacleTower-Full-MultiDiscrete-v0",
        ),
    }

    env_names = []
    env_curves = []
    for env_name, env_experiments in envs.items():
        env_names.append(env_name)
        curves = []
        for env_experiment in env_experiments:
            # Find all repetitions for this experiment
            glob_str = os.path.join(
                args.input,
                "stable_baselines_" +
                env_experiment +
                "*"
            )
            repetitions = glob(glob_str)
            xs = []
            ys = []
            for repetition in repetitions:
                data = load_experiment(repetition)
                xs.append(data["steps"])
                ys.append(rolling_average(data["rewards"], args.smooth))
            x, y, std = interpolate_and_average(xs, ys, interp_points=args.interp_points)
            curves.append((x, y, std))
        env_curves.append(curves)

    fig, axs = pyplot.subplots(
        nrows=1,
        ncols=2,
        sharex="all",
        sharey="row",
        figsize=[1.3 * 6.4, 1.3 * 4.8],
        squeeze=False
    )

    for y in range(1):
        for x in range(2):
            idx = y * 2 + x
            ax = axs[y, x]
            ax.grid(alpha=0.2)
            curves = env_curves[idx]
            title = env_names[idx]
            ax.set_title(title, fontsize="x-large")

            ax.ticklabel_format(axis="x", scilimits=(0, 0))
            ax.tick_params(axis='both', which='both', labelsize="large")
            # Plot lines
            for curve in curves:
                x_points, y_points, std = curve
                ax.plot(x_points, y_points)
                ax.fill_between(
                    x_points,
                    y_points + std,
                    y_points - std,
                    alpha=0.2
                )
            # Only include legend in first plot
            if x == 0 and y == 0:
                ax.legend(LEGEND, prop={"size": "large"})
            # If first column, include y-label
            if x == 0:
                ax.set_ylabel("Episodic reward", fontsize="x-large")
            # If last row, include x-axis
            if y == 0:
                ax.set_xlabel("Agent steps", fontsize="x-large")

    fig.tight_layout()
    fig.savefig(os.path.join(args.output, "ot.pdf"), bbox_inches="tight", pad_inches=0.0)


def plot_gettogoal_steps_to_solve(args):
    # Two plots: bogus-actions and extra-actions,
    # in the gettogoal environment, with Discrete
    # and MultiDiscrete action spaces
    MIN_BOGUS = 1
    MAX_BOGUS = 20
    MIN_EXTRA = 3
    MAX_EXTRA = 50

    LEGEND = ["Discrete", "Multi-Discrete"]
    envs = {
        "Extra actions": (
            "GetToGoal-Discrete{actions}-v0",
            "GetToGoal-MultiDiscrete{actions}-v0",
        ),
        "Bogus actions": (
            "GetToGoal-Discrete-Bogus{actions}-v0",
            "GetToGoal-MultiDiscrete-Bogus{actions}-v0",
        ),
    }

    env_names = []
    env_curves = []
    for env_name, env_experiments in envs.items():
        env_names.append(env_name)
        curves = []
        for env_experiment in env_experiments:
            action_range = (
                range(MIN_BOGUS, MAX_BOGUS + 1) if "Bogus" in env_name
                else range(MIN_EXTRA, MAX_EXTRA + 1)
            )
            xs = []
            ys = []
            stds = []
            for num_actions in action_range:
                glob_str = os.path.join(
                    args.input,
                    "stable_baselines_" +
                    env_experiment.format(actions=num_actions) +
                    "*"
                )
                repetitions = glob(glob_str)
                solved_at_points = []
                for repetition in repetitions:
                    data = load_experiment(repetition)
                    solved_at_steps = find_solved_point(
                        data,
                        args.window_size,
                        args.reward_threshold
                    )
                    solved_at_points.append(solved_at_steps)

                # If there were Nones, do not plot this line
                # (and do not progress further)
                if None in solved_at_points:
                    break

                x = num_actions
                y = np.mean(solved_at_points)
                std = np.std(solved_at_points)

                xs.append(x)
                ys.append(y)
                stds.append(std)
            curves.append((
                np.array(xs),
                np.array(ys),
                np.array(stds)
            ))
        env_curves.append(curves)

    fig, axs = pyplot.subplots(
        nrows=1,
        ncols=2,
        sharey="row",
        figsize=[1.0 * 6.4, 0.7 * 4.8],
        squeeze=False
    )

    for y in range(1):
        for x in range(2):
            idx = y * 2 + x
            ax = axs[y, x]
            ax.grid(alpha=0.2)
            curves = env_curves[idx]
            title = env_names[idx]
            ax.set_title(title, fontsize="large")

            ax.ticklabel_format(axis="y", scilimits=(0, 0))
            ax.tick_params(axis='both', which='both', labelsize="medium")
            # Plot lines
            for curve in curves:
                x_points, y_points, std = curve
                ax.plot(x_points, y_points)
                ax.fill_between(
                    x_points,
                    y_points + std,
                    y_points - std,
                    alpha=0.2
                )
            # Only include legend in first plot
            if x == 0 and y == 0:
                ax.legend(LEGEND, prop={"size": "large"})
            # If first column, include y-label
            if x == 0:
                ax.set_ylabel("Steps to solve", fontsize="large")
            # If last row, include x-axis
            if y == 0:
                ax.set_xlabel("Number of actions", fontsize="large")

    fig.tight_layout()
    fig.savefig(os.path.join(args.output, "gettogoal_bogus_and_extra.pdf"), bbox_inches="tight", pad_inches=0.0)


def plot_gettogoal(args):
    # Dictionary mapping human-like game name to
    # all experiment names, that should be plotted in that
    # figure.
    # NOTE: The ordering matters! Colors are assigned in this order!
    LEGENDS = [
        [  # Main plot
            "Discrete",
            "Multi-Discrete",
            "Tank, Discrete",
            "Tank, Multi-Discrete",
            "Continuous"
        ],
        [
            "− Backward",
            "− Backward",
            "Tank, Discrete",
            "Tank, Multi-Discrete",
            "+ Strafe",
            "+ Strafe",
        ]
    ]

    envs = {
        "Main_plot": (
            "GetToGoal-Discrete-v0",
            "GetToGoal-MultiDiscrete-v0",
            "GetToGoal-TankDiscrete-v0",
            "GetToGoal-TankMultiDiscrete-v0",
            "GetToGoal-Continuous-v0",
        ),
        "Tank_plots": (
            "GetToGoal-TankDiscrete-NoBackward-v0",
            "GetToGoal-TankMultiDiscrete-NoBackward-v0",
            "GetToGoal-TankDiscrete-v0",
            "GetToGoal-TankMultiDiscrete-v0",
            "GetToGoal-TankDiscrete-Strafe-v0",
            "GetToGoal-TankMultiDiscrete-Strafe-v0",
        ),
    }

    env_names = []
    env_curves = []
    for env_name, env_experiments in envs.items():
        curves = []
        env_names.append(env_name)
        for i, env_experiment in enumerate(env_experiments):
            # Find all repetitions for this experiment
            glob_str = os.path.join(
                args.input,
                "stable_baselines_" +
                env_experiment +
                "*"
            )
            repetitions = glob(glob_str)
            xs = []
            ys = []
            for repetition in repetitions:
                data = load_experiment(repetition)
                xs.append(data["steps"])
                ys.append(rolling_average(data["rewards"], args.smooth))
            x, y, std = interpolate_and_average(xs, ys, interp_points=args.interp_points)
            # Hardcoded styling. Good luck figuring out the logic here
            style = "-"
            color = "C{}".format(i)
            if env_name == "Tank_plots":
                style = "--"
                if "TankDiscrete" in env_experiment:
                    color = "C2"
                else:
                    color = "C3"
            if "Strafe" in env_experiment:
                style = ":"
            elif "NoBackward" in env_experiment:
                style = "--"
            else:
                style = "-"
            curves.append((x, y, std, style, color))
        env_curves.append(curves)

    fig, axs = pyplot.subplots(
        nrows=1,
        ncols=2,
        sharex="all",
        sharey="row",
        figsize=[1.1 * 6.4, 0.8 * 4.8],
        squeeze=False
    )

    for y in range(1):
        for x in range(2):
            idx = y * 2 + x
            ax = axs[y, x]
            ax.grid(alpha=0.2)
            curves = env_curves[idx]
            name = env_names[idx]

            ax.ticklabel_format(axis="x", scilimits=(0, 0))
            ax.tick_params(axis='both', which='both', labelsize="large")
            # Plot lines
            lines = []
            for curve in curves:
                x_points, y_points, std, style, color = curve
                line, = ax.plot(x_points, y_points, style, color=color)
                if style == "--" or style == ":":
                    line, = ax.plot(x_points, y_points, style, color="k")
                    line.remove()
                lines.append(line)
                ax.fill_between(
                    x_points,
                    y_points + std,
                    y_points - std,
                    alpha=0.2,
                    color=color,
                    edgecolor="none",
                    linewidth=0.0
                )
            if name == "Tank_plots":
                # Bit of hardcoding legends for this one.
                # Only draw one dashed line and one dotted line
                legends = LEGENDS[idx]
                # Jesus fucking christ this is horrifying
                del lines[1]
                del lines[-1]
                del legends[1]
                del legends[-1]
                lines.insert(2, lines.pop(0))
                legends.insert(2, legends.pop(0))
                ax.legend(lines, legends, prop={"size": "medium"})
            else:
                ax.legend(lines, LEGENDS[idx], prop={"size": "medium"})
            # If first column, include y-label
            if x == 0:
                ax.set_ylabel("Episodic reward", fontsize="x-large")
            # If last row, include x-axis
            if y == 0:
                ax.set_xlabel("Agent steps", fontsize="x-large")

    fig.tight_layout()
    fig.savefig(os.path.join(args.output, "gettogoal.pdf"), bbox_inches="tight", pad_inches=0.0)


def main(args):
    plot_vizdoom(args)
    plot_atari(args)
    plot_sc2(args)
    plot_ot(args)
    plot_gettogoal_steps_to_solve(args)
    plot_gettogoal(args)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
