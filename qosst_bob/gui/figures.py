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
Code to plots figures for the GUI.

The way it actually is done in the GUI:

This module provides code for every plot through a plot function
that is a Callable[[Bob, Axes], None].

This modules also provide a class for the figure with the init and plot
methods. The list of figure is then initialized, in particular giving its
name and the function that should be called to actually do the plot.

This list will be imported in the layout to create as many tabs and autoplot
checkboxes required.

This list will also be imported in the gui to detect the different events.
"""
from typing import Callable, Optional
from os import PathLike
import gc

import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import FreeSimpleGUI as sg

from qosst_bob.bob import Bob
from qosst_bob.utils import heatmap_complex


def plot_temporal(bob: Optional[Bob], axes: Axes) -> None:
    """
    Plot the acquired data as a function of time.

    Args:
        bob (Bob): Bob object.
        axes (Axes): the axes where to plot the data.
    """
    if bob is not None:
        assert bob.config is not None and bob.config.bob is not None
        axes.clear()

        if bob.signal_data is not None:
            data = bob.signal_data[0]
            times = np.arange(len(data)) / bob.config.bob.adc.rate
            axes.plot(times, data)

        if bob.end_electronic_shot_noise:
            axes.axvline(x=times[bob.end_electronic_shot_noise], color="red")
        if bob.begin_data and bob.end_electronic_shot_noise:
            axes.axvline(
                x=times[bob.begin_data + bob.end_electronic_shot_noise],
                color="black",
            )
        if bob.end_data and bob.end_electronic_shot_noise:
            axes.axvline(
                x=times[bob.end_data + bob.end_electronic_shot_noise],
                color="black",
            )
    axes.set_xlabel("Time [s]")
    axes.set_ylabel("Output [mV]")
    axes.set_title("Output vs. time")
    axes.grid(True)


def plot_frequential(bob: Optional[Bob], axes: Axes) -> None:
    """
    Plot the Power Spectral Density of the received data
    and, if available, of the shot noise and electronic noise.

    Args:
        bob (Bob): Bob object.
        axes (Axes): the axes where to plot the data.
    """
    if bob is not None:
        assert bob.config is not None and bob.config.bob is not None

        axes.clear()
        if bob.signal_data is not None:
            axes.psd(
                bob.signal_data[0],
                NFFT=2048,
                Fs=bob.config.bob.adc.rate,
                label="Signal",
            )

        if bob.electronic_noise is not None:
            axes.psd(
                bob.electronic_noise.data[0],
                NFFT=2048,
                Fs=bob.config.bob.adc.rate,
                label="Electronic noise",
            )
        if bob.electronic_shot_noise is not None:
            axes.psd(
                bob.electronic_shot_noise.data[0],
                NFFT=2048,
                Fs=bob.config.bob.adc.rate,
                label="Electronic and shot noise",
            )
        if bob.config.bob.dsp.exclusion_zone_pilots is not None:
            for begin_zone, end_zone in bob.config.bob.dsp.exclusion_zone_pilots:
                axes.axvspan(begin_zone, end_zone, alpha=0.3, color="black")
        axes.legend(fancybox=True, shadow=True)
    axes.set_xlabel("Frequency [Hz]")
    axes.set_ylabel("Power Spectral Density [dBm/Hz]")
    axes.set_title("PSD vs. frequency")
    axes.grid(True)


def plot_fft(bob: Optional[Bob], axes: Axes) -> None:
    """
    Plot the FFT of the acquired data.

    Args:
        bob (Bob): Bob object.
        axes (Axes): the axes where to plot the data.
    """
    if bob is not None:
        assert bob.config is not None and bob.config.bob is not None

        axes.clear()
        if bob.signal_data is not None:
            data = bob.signal_data[0]
            data_fft = fftpack.fft(data)
            data_fftfreq = fftpack.fftfreq(len(data), 1 / bob.config.bob.adc.rate)
            axes.plot(data_fftfreq, data_fft)
    axes.set_xlabel("Frequency [Hz]")
    axes.set_ylabel("FFT")
    axes.set_title("FFT")
    axes.grid(True)


def plot_tone(bob: Optional[Bob], axes: Axes) -> None:
    """
    Plot the recovered tone.

    Args:
        bob (Bob): Bob object.
        axes (Axes): the axes where to plot the data.
    """
    if bob is not None and bob.received_tone is not None:
        heatmap_complex(bob.received_tone, "I", "Q", "Tone", axes=axes, clear=True)
    else:
        axes.set_xlabel("I")
        axes.set_ylabel("Q")
        axes.set_title("Tone")
        axes.grid(True)


def plot_quantum_data(bob: Optional[Bob], axes: Axes) -> None:
    """
    Plot the uncorrected quantum data.

    Args:
        bob (Bob): Bob object.
        axes (Axes): the axes where to plot the data.
    """
    if bob is not None and bob.quantum_data_phase_noisy is not None:
        heatmap_complex(
            bob.quantum_data_phase_noisy,
            "I",
            "Q",
            "Quantum data",
            axes=axes,
            clear=True,
        )
    else:
        axes.set_xlabel("I")
        axes.set_ylabel("Q")
        axes.set_title("Quantum data")
        axes.grid(True)


def plot_recovered(bob: Optional[Bob], axes: Axes) -> None:
    """
    Plot the corrected quantum data.

    Args:
        bob (Bob): Bob object.
        axes (Axes): the axes where to plot the data.
    """
    if bob is not None and bob.quantum_symbols is not None:
        heatmap_complex(
            bob.quantum_symbols,
            "I",
            "Q",
            "Recovered symbols",
            axes=axes,
            clear=True,
        )
    else:
        axes.set_xlabel("I")
        axes.set_ylabel("Q")
        axes.set_title("Recovered symbols")
        axes.grid(True)


def draw_figure(canvas: sg.Canvas, figure: Figure) -> FigureCanvasTkAgg:
    """
    Creates and returns canvas to draw the figure on the GUI.

    Args:
        canvas (sg.Canvas): the canvas of the GUI.
        figure (Figure): the matplolib figure.

    Returns:
        FigureCanvasTkAgg: the tk canvas of the figure.
    """
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return figure_canvas_agg


# pylint: disable=too-many-instance-attributes
class QOSSTBobGUIFigure:
    """
    A class representing a GUI figure.
    """

    name: str  #: The name of the figure.
    key: str  #: The key of the figure.
    plot_key: str  #: The key of the plot button.
    autoplot_key: str  #: The key of the autoplot checkbox.
    save_key: str  #: The key of the save button.
    figure: Optional[Figure]  #: The matoplotlib figures.
    axes: Optional[Axes]  #: The matplotlib axes.
    canvas: Optional[FigureCanvasTkAgg]  #: The canvas to display in the GUI.
    func: Callable[[Optional[Bob], Axes], None]  #: The function to plot the content.
    default_autoplot: bool  #: The default value of the autoplot checkbox.

    def __init__(
        self,
        name: str,
        func: Callable[[Optional[Bob], Axes], None],
        default_autoplot: bool = False,
    ) -> None:
        """
        Args:
            name (str): name of the figure.
            func (Callable[[Bob, Axes], None]): function to call to plot the figure.
            default_autoplot (bool, optional): default value of the autoplot checkbox.. Defaults to False.
        """
        self.name = name
        self.key = f"-FIG-{self.name.upper()}-"
        self.plot_key = f"-PLOT-FIG-{self.name.upper()}-"
        self.save_key = f"-SAVE-FIG-{self.name.upper()}-"
        self.autoplot_key = f"-AUTOPLOT-FIG-{self.name.upper()}"
        self.figure = None
        self.axes = None
        self.canvas = None
        self.func = func
        self.default_autoplot = default_autoplot

    def init_figure(self, window: sg.Window):
        """
        Initialize the figure, the axes and make a dummy plot.

        Args:
            window (sg.Window): GUI window.
        """
        self.figure = plt.figure()
        self.axes = self.figure.add_subplot()
        self.func(None, self.axes)
        self.canvas = draw_figure(window[self.key].TKCanvas, self.figure)

    def plot(self, bob: Bob):
        """
        Actualise the plot

        Args:
            bob (Bob): Bob object.
        """
        if self.figure is None or self.canvas is None:
            raise TypeError("Figure was not initialized.")
        self.figure.clear()
        gc.collect()  # Call the garbage collector to free the memory of the cleared figure.
        self.axes = self.figure.add_subplot()
        self.func(bob, self.axes)
        self.canvas.draw()

    def save(self, path: PathLike):
        """
        Save figure to path.

        Args:
            path (PathLike): path to save the figure.
        """
        if self.figure is None:
            raise TypeError("Figure was not initialized.")
        self.figure.savefig(path)


all_figures = [
    QOSSTBobGUIFigure("temporal", plot_temporal, default_autoplot=True),
    QOSSTBobGUIFigure("frequential", plot_frequential, default_autoplot=True),
    QOSSTBobGUIFigure("fft", plot_fft),
    QOSSTBobGUIFigure("tone", plot_tone, default_autoplot=True),
    QOSSTBobGUIFigure("uncorrected", plot_quantum_data),
    QOSSTBobGUIFigure("recovered", plot_recovered),
]  #: List of all figures of the GUI.
