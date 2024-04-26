"""
Util functions for the plot.
"""

from typing import Dict
from pathlib import Path


def get_styles() -> Dict[str, Path]:
    """
    Get the matplotlib available styles from the polt_styles directory
    and return a dict of names and paths.

    Returns:
        Dict[str, Path]: dict, key being the name of style and the value being the path.
    """
    parent_directory = Path(__file__).parent
    plot_styles_directory = (parent_directory / "plot_styles").glob("**/*.mplstyle")
    files = [x for x in plot_styles_directory if x.is_file()]

    res = {}
    for file in files:
        res[file.stem] = file

    return res
