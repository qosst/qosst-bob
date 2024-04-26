# qosst-bob - Bob module of the Quantum Open Software for Secure Transmissions.
# Copyright (C) 2021-2024 Yoann Piétri

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Util functions for qosst-bob.
"""
from typing import Union, Tuple, Optional

import numpy as np
import matplotlib
from matplotlib.axes import Axes
import matplotlib.pyplot as plt


# pylint: disable=too-many-arguments
def heatmap(
    data_x: np.ndarray,
    data_y: np.ndarray,
    x_label="X",
    y_label="Y",
    title="Heatmap",
    cmap: Union[matplotlib.colors.Colormap, str] = "rainbow",
    axes: Optional[Axes] = None,
    clear: bool = True,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Draw a heatmap from data_x and data_y.

    Args:
        data_x (np.ndarray): the data to be put in the x-axis of the heatmap.
        data_y (np.ndarray): the data to be put in the y-axis of the heatmap.
        x_label (str, optional): the label of the x-axis. Defaults to "X".
        y_label (str, optional): the label of the y-axis. Defaults to "Y".
        title (str, optional): the title of the figure. Defaults to "Heatmap".
        cmap (Union[matplotlib.colors.Colormap, str], optional): the colormap to use. Defaults to rainbow.
        axes (matplotlib.axes.Axes, optional): use those axes to make the plot. Defaults to None.
        clear (bool): if True and if an axe was given, clear this axe.

    Returns:
        Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: the figure and the axe.
    """
    fig: Optional[matplotlib.figure.Figure]
    if not axes:
        fig = plt.figure()
        axes = fig.add_subplot()
    else:
        fig = axes.get_figure()
        if clear:
            if axes.images:
                # mypy is not happy with the next line but it works
                # Ignore the error for now, and maybe find a better solution
                # one day.
                axes.images[0].colobar.remove()  # type: ignore
            axes.clear()
    assert fig is not None
    current_heatmap, xedges, yedges = np.histogram2d(
        data_x,
        data_y,
        bins=50,
    )
    extent = (float(xedges[0]), float(xedges[-1]), float(yedges[0]), float(yedges[-1]))
    axes_image = axes.imshow(
        current_heatmap.T, extent=extent, origin="lower", cmap=cmap
    )
    fig.colorbar(axes_image)
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    axes.set_title(title)
    axes.grid()
    return fig, axes


def heatmap_complex(
    data: np.ndarray,
    x_label="X",
    y_label="Y",
    title="Heatmap",
    cmap: Union[matplotlib.colors.Colormap, str] = "rainbow",
    axes: Optional[Axes] = None,
    clear: bool = True,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Draw a heatmap from data, using data.real as the values for the x-axis and data.imag for the y-axis.

    Args:
        data (np.ndarray): the complex data to be put in the x-axis (real part) and y-axis (imag part) of the heatmap.
        x_label (str, optional): the label of the x-axis. Defaults to "X".
        y_label (str, optional): the label of the y-axis. Defaults to "Y".
        title (str, optional): the title of the figure. Defaults to "Heatmap".
        cmap (Union[matplotlib.colors.Colormap, str], optional): the colormap to use. Defaults to rainbow.
        axes (matplotlib.axes.Axes, optional): use those axes to make the plot. Defaults to None.
        clear (bool): if True and if an axe was given, clear this axe.

    Returns:
        Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: the figure and the axe.
    """
    return heatmap(
        data.real,
        data.imag,
        x_label=x_label,
        y_label=y_label,
        title=title,
        cmap=cmap,
        axes=axes,
        clear=clear,
    )
