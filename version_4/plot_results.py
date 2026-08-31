"""Plot and compare the average performance of different model versions.

The script scans the ``logs`` directory for one sub-folder per version (e.g.
``baseline``). Each version folder is expected to contain one or more CSV files
named ``run_seed_*.csv`` with columns::

    training_episode,win_rate,mean_episode_length,epsilon

For every version the seeded runs are combined into a single "average" curve by
taking the pointwise average of ``win_rate`` and ``mean_episode_length`` over
the training episodes that are present in *all* runs of that version.

A single figure is produced with two Y axes:

* left axis  -> win rate (0-100), drawn with solid lines
* right axis -> mean episode length (auto-scaled), drawn with dashed lines

Each version gets its own colour. The X axis spans training episodes from 0 up
to the highest episode found across all versions.
"""

from __future__ import annotations

import glob
import os

import matplotlib.pyplot as plt
import pandas as pd

# Directory layout: logs/<version>/run_seed_*.csv
LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")

# A distinct colour per version. Extend this list if more versions are added.
VERSION_COLORS = {
    "baseline": "#1f77b4",
    # "dqn": "#ff7f0e",
    # "double_dqn": "#2ca02c",
    # ...
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


def plot_results(logs_dir: str = LOGS_DIR) -> None:
    """Build the comparison plot for all versions found in ``logs_dir``."""
    versions = _discover_versions(logs_dir)
    if not versions:
        raise FileNotFoundError(f"No version folders found under {logs_dir!r}.")

    fig, ax_win = plt.subplots(figsize=(10, 6))
    ax_len = ax_win.twinx()  # second Y axis for mean episode length

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

        # Win rate (0-100) on the left axis, solid line.
        ax_win.plot(
            episodes,
            averaged["win_rate"] * 100.0,
            color=color,
            linestyle="-",
            label=f"{version} (win rate)",
        )

        # Mean episode length on the right axis, dashed line.
        ax_len.plot(
            episodes,
            averaged["mean_episode_length"],
            color=color,
            linestyle="--",
            label=f"{version} (mean episode length)",
        )

    # Left axis: win rate.
    ax_win.set_xlabel("Training episodes")
    ax_win.set_ylabel("Win rate (%)")
    ax_win.set_ylim(0, 100)
    ax_win.set_xlim(0, highest_episode)

    # Right axis: mean episode length, auto-scaled between lowest/highest found.
    ax_len.set_ylabel("Mean episode length")

    # Single legend combining both axes.
    lines_win, labels_win = ax_win.get_legend_handles_labels()
    lines_len, labels_len = ax_len.get_legend_handles_labels()
    ax_win.legend(lines_win + lines_len, labels_win + labels_len, loc="best")

    fig.tight_layout()
    fig.savefig(os.path.join(logs_dir, "results_comparison.png"), dpi=150)
    plt.show()


if __name__ == "__main__":
    plot_results()