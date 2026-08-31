"""Plot and compare the average performance of different model versions.

The script scans the ``logs`` directory for one sub-folder per version (e.g.
``baseline``). Each version folder is expected to contain one or more CSV files
named ``run_seed_*.csv`` with columns::

    training_episode,win_rate,mean_episode_length,epsilon

For every version the seeded runs are combined into a single "average" curve by
taking the pointwise average of ``win_rate`` and ``mean_episode_length`` over
the training episodes that are present in *all* runs of that version.

A single figure with two stacked subplots is produced, one per metric:

* win rate (0-100), drawn with solid lines
* mean episode length (auto-scaled), drawn with dashed lines

Each version gets its own colour. The X axis spans training episodes from 0 up
to the highest episode found across all versions.

Usage::

    python3 plot_results.py [max_episodes] [filter]

* ``max_episodes`` (optional): the upper limit for the X axis, always given in
  ``{number}k`` form (e.g. ``30k`` means 30,000 episodes). Defaults to the
  highest episode found across all versions.
* ``filter`` (optional): a suffix filter applied to version names. A value
  prefixed with ``no_`` (e.g. ``no_bad``) excludes every version whose name
  ends with the remaining letters (``bad``); any other value keeps only the
  versions whose names end with it.
"""

from __future__ import annotations

import argparse
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd

# Directory layout: logs/<version>/run_seed_*.csv
LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")

# A distinct colour per version. Extend this list if more versions are added.
# (following the colours of the rainbow)
VERSION_COLORS = {
    "1-baseline": "#b41f1f", # red
    "2-faster-epsilon-decay": "#ff7f0e", # orange
    "3-bigger-replay-memory_bad": "#f0e000", # yellow
    "4-slower-sync-rate": "#00ff00", # green
    "5-higher-step-penalty": "#00ffff", # cyan
    "6-decreasing-lr": "#0000ff", # blue
    "7-linear-epsilon-decay_bad": "#ff00ff", # magenta
    # "8-exponential-epsilon-decay": "#400040", # purple

}


def _discover_versions(logs_dir: str) -> list[str]:
    """Return the names of all version folders found under ``logs_dir``."""
    versions = []
    for entry in sorted(os.listdir(logs_dir)):
        full_path = os.path.join(logs_dir, entry)
        if os.path.isdir(full_path):
            versions.append(entry)
    return versions


def _load_runs(version_dir: str) -> list[pd.DataFrame]:
    """Load every ``run_seed_*.csv`` file in ``version_dir`` as a DataFrame."""
    runs = []
    for csv_path in sorted(glob.glob(os.path.join(version_dir, "run_seed_*.csv"))):
        runs.append(pd.read_csv(csv_path))
    return runs


def _average_runs(runs: list[pd.DataFrame]) -> pd.DataFrame:
    """Pointwise-average the seeded runs over their common training episodes.

    Only training episodes present in *every* run are kept, so the average is
    well defined at each point. Returns a DataFrame with columns
    ``training_episode``, ``win_rate`` and ``mean_episode_length``.
    """
    if not runs:
        raise ValueError("No seeded runs found for this version.")

    # Intersection of training episodes across all runs.
    common_episodes = set(runs[0]["training_episode"])
    for run in runs[1:]:
        common_episodes &= set(run["training_episode"])

    if not common_episodes:
        raise ValueError("The seeded runs share no common training episodes.")

    common_episodes = sorted(common_episodes)

    win_rates = []
    mean_lengths = []
    for episode in common_episodes:
        win_rates.append(
            sum(run.loc[run["training_episode"] == episode, "win_rate"].iloc[0]
                for run in runs)
            / len(runs)
        )
        mean_lengths.append(
            sum(run.loc[run["training_episode"] == episode, "mean_episode_length"].iloc[0]
                for run in runs)
            / len(runs)
        )

    return pd.DataFrame(
        {
            "training_episode": common_episodes,
            "win_rate": win_rates,
            "mean_episode_length": mean_lengths,
        }
    )


def _parse_max_episodes(value: str | None) -> int | None:
    """Parse a ``{number}k`` string into an episode count (or ``None``).

    ``"30k"`` becomes ``30000``. A ``None`` value is returned unchanged.
    """
    if value is None:
        return None
    if not value.endswith("k"):
        raise ValueError(
            f"max_episodes must be given in '{{number}}k' form, got {value!r}."
        )
    try:
        return int(value[:-1]) * 1000
    except ValueError:
        raise ValueError(
            f"max_episodes must be given in '{{number}}k' form, got {value!r}."
        ) from None


def _filter_versions(versions: list[str], filter_arg: str | None) -> list[str]:
    """Filter version names by a suffix argument.

    A value prefixed with ``no_`` (e.g. ``no_bad``) excludes every version whose
    name ends with the remaining letters (``bad``). Any other value keeps only
    the versions whose names end with it. ``None`` keeps everything.
    """
    if filter_arg is None:
        return versions
    if filter_arg.startswith("no_"):
        suffix = filter_arg[3:]
        return [v for v in versions if not v.endswith(suffix)]
    return [v for v in versions if v.endswith(filter_arg)]


def plot_results(
    logs_dir: str = LOGS_DIR,
    max_episodes: str | None = None,
    filter_arg: str | None = None,
) -> None:
    """Build the comparison plots for all versions found in ``logs_dir``.

    ``max_episodes`` caps the X axis and is given in ``{number}k`` form (e.g.
    ``"30k"``). ``filter_arg`` optionally filters version names by suffix (see
    module docstring).
    """
    versions = _discover_versions(logs_dir)
    versions = _filter_versions(versions, filter_arg)
    if not versions:
        raise FileNotFoundError(f"No version folders found under {logs_dir!r}.")

    # A single figure with two stacked subplots, one per metric.
    fig, (ax_win, ax_len) = plt.subplots(
        nrows=2, ncols=1, figsize=(10, 10), sharex=True
    )

    highest_episode = 0

    for version in versions:
        runs = _load_runs(os.path.join(logs_dir, version))
        averaged = _average_runs(runs)

        color = VERSION_COLORS.get(version)
        if color is None:
            # Fall back to the next colour in the default matplotlib cycle.
            color = f"C{len(VERSION_COLORS)}"

        episodes = averaged["training_episode"]
        highest_episode = max(highest_episode, int(episodes.max()))

        # Win rate (0-100), solid line.
        ax_win.plot(
            episodes,
            averaged["win_rate"] * 100.0,
            color=color,
            linestyle="-",
            label=version,
        )

        # Mean episode length, dashed line.
        ax_len.plot(
            episodes,
            averaged["mean_episode_length"],
            color=color,
            linestyle="--",
            label=version,
        )

    # Cap the X axis at the requested maximum, or the highest episode found.
    x_max = _parse_max_episodes(max_episodes) or highest_episode

    # Win rate subplot.
    ax_win.set_ylabel("Win rate (%)")
    ax_win.set_ylim(0, 100)
    ax_win.set_xlim(1000, x_max)
    ax_win.legend(loc="lower left")

    # Mean episode length subplot, auto-scaled between lowest/highest found.
    ax_len.set_xlabel("Training episodes")
    ax_len.set_ylabel("Episode length")
    ax_len.set_xlim(1000, x_max)
    ax_len.legend(loc="lower left")

    fig.tight_layout()
    fig.savefig(os.path.join(logs_dir, "results.png"), dpi=150)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot and compare the average performance of model versions."
    )
    parser.add_argument(
        "max_episodes",
        nargs="?",
        default=None,
        help="Upper limit for the X axis, always in '{number}k' form (e.g. '30k').",
    )
    parser.add_argument(
        "filter_arg",
        nargs="?",
        default=None,
        help="Suffix filter for version names; prefix with 'no_' to exclude "
        "(e.g. 'no_bad' drops versions ending in 'bad').",
    )
    args = parser.parse_args()
    plot_results(max_episodes=args.max_episodes, filter_arg=args.filter_arg)