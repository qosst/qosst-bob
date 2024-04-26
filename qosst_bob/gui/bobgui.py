# qosst-bob - Bob module of the Quantum Open Software for Secure Transmissions.
# Copyright (C) 2021-2024 Yoann Piétri
# Copyright (C) 2021-2024 Matteo Schiavon

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
A graphical user interface for Bob.
"""
import logging
import argparse
from typing import Union, List, Tuple, Optional

import numpy as np
import PySimpleGUI as sg
import matplotlib
import matplotlib.pyplot as plt

from qosst_core import RELEASE_NAME
from qosst_core.utils import export_np
from qosst_core.logging import verbose_to_log_level
from qosst_core.infos import get_script_infos
from qosst_core.commands import THANKS_STRING

from qosst_bob import __version__
from qosst_bob.bob import Bob
from qosst_bob.gui.layout import layout
from qosst_bob.gui.figures import all_figures
from qosst_bob.gui.layout_content import (
    QOSSTGUIActions,
    QOSSTGUIContent,
    QOSSTGUIText,
    QOSSTGUIInput,
    THEME,
    SEQUENCE,
    LICENSE_TEXT,
)
from qosst_bob.gui.plot_utils import get_styles


# Configure logging
# By default, we will have three handles
# * One console handler
# * One stdout handler that will actually be redirected to the gui
# * One file handler
# pylint: disable=no-member
class GUIHandler(logging.StreamHandler):
    """
    A log handler to print the log in the console on the GUI.
    """

    window: sg.Window

    def __init__(self, window: sg.Window):
        """
        Initialize the associated StreamHandler.
        """
        self.window = window
        logging.StreamHandler.__init__(self)

    def emit(self, record: logging.LogRecord):
        """
        Print the log.

        Args:
            record (logging.LogRecord): the log to print.
        """
        self.window[QOSSTGUIText.LOGGER].print(self.format(record))


root_logger = logging.getLogger("")

logger = logging.getLogger(__name__)

figures_plot_keys = [figure.plot_key for figure in all_figures]
figures_save_keys = [figure.save_key for figure in all_figures]


def autoplot(bob: Bob, values: dict):
    """
    Iterate through all figures and actualise plot
    if autoplot is enabled for this figure.

    Args:
        bob (Bob): Bob object.
        values (dict): current values of the GUI.
    """
    for figure in all_figures:
        if values[figure.autoplot_key]:
            figure.plot(bob)


def change_enable_status(
    window: sg.Window,
    content: Union[QOSSTGUIContent, List[QOSSTGUIContent]],
    disabled=False,
):
    """
    Change the enable status of content.

    Args:
        window (sg.Window): the window of the GUI.
        content (Union[QOSSTGUIContent, List[QOSSTGUIContent]]): a GUI object or a list of GUI object to set as either enabled or disabled.
        disabled (bool, optional): if True, the content is disabled. If False, the content is enabled. Defaults to False.
    """
    if isinstance(content, QOSSTGUIContent):
        window[content].update(disabled=disabled)
    else:
        for item in content:
            window[item].update(disabled=disabled)


def block_focus(window: sg.Window):
    """
    Block focus on every button of the window. This is used for popups.

    Args:
        window (sg.Window): pysimplegui window.
    """
    for key in window.key_dict:  # Remove dash box of all Buttons
        element = window[key]
        if isinstance(element, sg.Button):
            element.block_focus()


def popup_save_electronic_noise(location: str) -> Tuple[bool, str, str]:
    """Open a popup for the saving of the electronic noise.

    This popup will have two fields, that can be added to the data container of electronic noise:
    the name of the detector and a comment.

    Args:
        location (str): location where the data is going to be saved.

    Returns:
        Tuple[bool, str, str]: return True if the operation was not cancelled, and if not cancelled, the name of the detector and a comment.
    """
    popup_layout = [
        [sg.Text(f"Saving electronic noise at location {location}")],
        [sg.Text("Optional information: ")],
        [sg.Text("Detector"), sg.Input(key="-POPUP-ELECTRONIC-NOISE-DETECTOR-")],
        [sg.Text("Comment"), sg.Input(key="-POPUP-ELECTRONIC-NOISE-COMMENT-")],
        [sg.Button("Save"), sg.Button("Cancel")],
    ]
    window = sg.Window(
        "Save electronic noise",
        popup_layout,
        use_default_focus=False,
        finalize=True,
        modal=True,
    )
    block_focus(window)
    while True:
        event, values = window.read()
        if event == "Save":
            window.close()
            return (
                True,
                values["-POPUP-ELECTRONIC-NOISE-DETECTOR-"],
                values["-POPUP-ELECTRONIC-NOISE-COMMENT-"],
            )

        # Cancel
        window.close()
        return False, "", ""


def popup_save_electronic_shot_noise(
    location: str,
) -> Tuple[bool, str, Optional[float], str]:
    """Open a popup for the saving of the electronic and shot noise.

    This popup will have three fields, that can be added to the data container of electronic noise:
    the name of the detector, the power of the local oscillator and a comment.

    Args:
        location (str): location where the data is going to be saved.

    Returns:
        Tuple[bool, str, Optional[float], str]: return True if the operation was not cancelled, and if not cancelled, the name of the detector, the power and a comment.
    """
    popup_layout = [
        [sg.Text(f"Saving electronic and shot noise at location {location}")],
        [sg.Text("Optional information: ")],
        [sg.Text("Detector"), sg.Input(key="-POPUP-ELECTRONIC-SHOT-NOISE-DETECTOR-")],
        [sg.Text("Power"), sg.Input(key="-POPUP-ELECTRONIC-SHOT-NOISE-POWER-")],
        [sg.Text("Comment"), sg.Input(key="-POPUP-ELECTRONIC-SHOT-NOISE-COMMENT-")],
        [sg.Button("Save"), sg.Button("Cancel")],
    ]
    window = sg.Window(
        "Save electronic and shot noise",
        popup_layout,
        use_default_focus=False,
        finalize=True,
        modal=True,
    )
    block_focus(window)
    while True:
        event, values = window.read()
        if event == "Save":
            window.close()
            power = None
            if values["-POPUP-ELECTRONIC-SHOT-NOISE-POWER-"]:
                try:
                    power = float(values["-POPUP-ELECTRONIC-SHOT-NOISE-POWER-"])
                except ValueError:
                    pass
            return (
                True,
                values["-POPUP-ELECTRONIC-SHOT-NOISE-DETECTOR-"],
                power,
                values["-POPUP-ELECTRONIC-SHOT-NOISE-COMMENT-"],
            )

        # Cancel
        window.close()
        return False, "", None, ""


def _create_parser() -> argparse.ArgumentParser:
    """
    Create the parser for the command line tool.

    Returns:
        argparse.ArgumentParser: the created parser.
    """
    parser = argparse.ArgumentParser(prog="qosst-bob-gui")

    parser.add_argument("--version", action="version", version=__version__)

    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Level of verbosity. If none, nothing is printed to the console. -v will print warnings and errors, -vv will add info and -vvv will print all debug logs.",
    )
    return parser


# pylint: disable=too-many-locals, too-many-branches, too-many-statements, no-member
def main():
    """
    Main entrypoint for the GUI.
    """
    # Create the window
    bob = None

    parser = _create_parser()
    args = parser.parse_args()

    gui_console_level = verbose_to_log_level(args.verbose)

    root_logger.setLevel(gui_console_level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(gui_console_level)
    console_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)

    window = sg.Window(
        "QOSST Bob",
        layout,
        resizable=True,
        finalize=True,
        element_justification="c",
        return_keyboard_events=True,
        use_default_focus=False,
    )

    window.bind("<KeyPress-a>", "a:97")
    window.bind("<KeyPress-b>", "b:98")

    gui_handler = GUIHandler(window=window)
    gui_handler.setLevel(gui_console_level)
    gui_handler.setFormatter(formatter)

    root_logger.addHandler(gui_handler)

    sg.theme(THEME)

    for figure in all_figures:
        figure.init_figure(window)

    window[QOSSTGUIText.LOGGER].expand(True, True)

    window[QOSSTGUIText.LOGGER].print(get_script_infos(motd=False))

    hardware_enabled = False
    initialization_done = False
    sequence_pointer = 0

    # Create an event loop
    # pylint: disable=too-many-nested-blocks
    while True:
        event, values = window.read()
        if event in (QOSSTGUIActions.EXIT, sg.WIN_CLOSED):
            break
        if event == QOSSTGUIActions.READ_CONFIGURATION:
            # change_enable_status()
            window[QOSSTGUIInput.CONFIGURATION_PATH].update(disabled=True)
            window[QOSSTGUIActions.BROWSE].update(disabled=True)
            if bob is None:
                config_path = values[QOSSTGUIInput.CONFIGURATION_PATH]
                bob = Bob(config_path, enable_laser=False)

                # Set file logs
                if bob.config.logs.logging:
                    root_logger.setLevel(min(bob.config.logs.level, gui_console_level))
                    file_handler = logging.FileHandler(bob.config.logs.path)
                    file_handler.setLevel(bob.config.logs.level)
                    file_handler.setFormatter(formatter)
                    root_logger.addHandler(file_handler)
            else:
                bob.load_configuration()

            # Set configuration in the QI tab
            window[QOSSTGUIText.CONFIGURATION_QI_NUM_SYMBOLS].update(
                bob.config.frame.quantum.num_symbols
            )
            window[QOSSTGUIText.CONFIGURATION_QI_SHIFT].update(
                bob.config.frame.quantum.frequency_shift * 1e-6
            )
            window[QOSSTGUIText.CONFIGURATION_QI_SYMBOL_RATE].update(
                bob.config.frame.quantum.symbol_rate * 1e-6
            )
            window[QOSSTGUIText.CONFIGURATION_QI_ROLL_OFF].update(
                bob.config.frame.quantum.roll_off
            )
            window[QOSSTGUIText.CONFIGURATION_QI_MODULATION].update(
                bob.config.frame.quantum.modulation_cls.__name__
            )
            window[QOSSTGUIText.CONFIGURATION_QI_MODULATION_SIZE].update(
                bob.config.frame.quantum.modulation_size
            )

            # Set configuration in ZC tab
            window[QOSSTGUIText.CONFIGURATION_ZC_ROOT].update(
                bob.config.frame.zadoff_chu.root
            )
            window[QOSSTGUIText.CONFIGURATION_ZC_LENGTH].update(
                bob.config.frame.zadoff_chu.length
            )
            window[QOSSTGUIText.CONFIGURATION_ZC_RATE].update(
                bob.config.frame.zadoff_chu.rate * 1e-6
            )

            # Set configuration in the pilots tab
            window[QOSSTGUIText.CONFIGURATION_PILOTS_FREQUENCIES].update(
                ", ".join([str(x * 1e-6) for x in bob.config.frame.pilots.frequencies])
            )

            # Set configuration in the DSP tab
            window[QOSSTGUIText.CONFIGURATION_DSP_TONE_CUTOFF].update(
                bob.config.bob.dsp.tone_filtering_cutoff * 1e-6
            )
            window[QOSSTGUIText.CONFIGURATION_DSP_SUBFRAMES_SIZE].update(
                bob.config.bob.dsp.subframes_size
            )
            window[QOSSTGUIText.CONFIGURATION_DSP_ABORT_CLOCK_RECOVERY].update(
                bob.config.bob.dsp.abort_clock_recovery
            )
            window[QOSSTGUIText.CONFIGURATION_DSP_ALICE_DAC_RATE].update(
                bob.config.bob.dsp.alice_dac_rate * 1e-6
            )
            window[QOSSTGUIText.CONFIGURATION_DSP_EXCLUSION_ZONE].update(
                bob.config.bob.dsp.exclusion_zone_pilots
            )
            window[QOSSTGUIText.CONFIGURATION_DSP_PHASE_FILTERING].update(
                bob.config.bob.dsp.pilot_phase_filtering_size
            )

            # Set export directory
            window[QOSSTGUIText.EXPORT_DIRECTORY].update(
                bob.config.bob.export_directory
            )

            # Enable buttons
            change_enable_status(
                window,
                [
                    QOSSTGUIActions.INITIALIZE_HARDWARE,
                    QOSSTGUIActions.CONNECT,
                    QOSSTGUIActions.LOAD_ELEC_NOISE,
                    QOSSTGUIActions.LOAD_SHOT_NOISE,
                    QOSSTGUIActions.SAVE_ELEC_NOISE,
                    QOSSTGUIActions.SAVE_SHOT_NOISE,
                ],
                disabled=False,
            )
        elif event == QOSSTGUIActions.INITIALIZE_HARDWARE:
            if bob:
                bob.open_hardware()
                change_enable_status(
                    window, QOSSTGUIActions.INITIALIZE_HARDWARE, disabled=True
                )
                change_enable_status(
                    window,
                    [
                        QOSSTGUIActions.ACQUISITION_ELEC_NOISE,
                        QOSSTGUIActions.ACQUISITION_SHOT_NOISE,
                    ],
                    disabled=False,
                )
                if initialization_done:
                    change_enable_status(
                        window,
                        [
                            QOSSTGUIActions.QIE,
                            QOSSTGUIActions.DSP,
                            QOSSTGUIActions.PARAMETERS_ESTIMATION,
                            QOSSTGUIActions.ERROR_CORRECTION,
                            QOSSTGUIActions.PRIVACY_AMPLITICATION,
                        ],
                        disabled=False,
                    )
                hardware_enabled = True
            else:
                sg.popup_ok("Please read configuration first.")
        elif event == QOSSTGUIActions.CONNECT:
            if bob:
                res = bob.connect()
                if res:
                    window[QOSSTGUIText.CONNECTION_STATUS].update(
                        f"Status: Connected to {bob.config.bob.network.server_address}"
                    )
                    change_enable_status(window, QOSSTGUIActions.CONNECT, disabled=True)
                    change_enable_status(
                        window, QOSSTGUIActions.IDENTIFICATION, disabled=False
                    )
                else:
                    sg.popup_error(
                        f"Couldn't connect to host {bob.config.bob.network.server_address}"
                    )
            else:
                sg.popup_ok("Please read configuration first.")
        elif event == QOSSTGUIActions.IDENTIFICATION:
            if bob and bob.is_connected:
                res = bob.identification()
                if res:
                    window[QOSSTGUIText.IDENTIFICATION_STATUS].update(
                        "Identification status : Done"
                    )
                    change_enable_status(
                        window, QOSSTGUIActions.INITIALIZATION, disabled=False
                    )
                else:
                    window[QOSSTGUIText.IDENTIFICATION_STATUS].update(
                        "Identification status : Done"
                    )
            else:
                sg.popup_ok("Please read configuration and connect Bob first.")
        elif event == QOSSTGUIActions.INITIALIZATION:
            if bob and bob.is_connected:
                res = bob.initialization()

                if res:
                    window[QOSSTGUIText.FRAME_UUID].update(
                        f"Frame UUID : {bob.frame_uuid}"
                    )
                    initialization_done = True
                    if hardware_enabled:
                        change_enable_status(
                            window,
                            [
                                QOSSTGUIActions.QIE,
                                QOSSTGUIActions.DSP,
                                QOSSTGUIActions.PARAMETERS_ESTIMATION,
                                QOSSTGUIActions.ERROR_CORRECTION,
                                QOSSTGUIActions.PRIVACY_AMPLITICATION,
                            ],
                            disabled=False,
                        )
                else:
                    window[QOSSTGUIText.FRAME_UUID].update("Initialization failed")
            else:
                sg.popup_ok("Please read configuration and connect Bob first.")
        elif event == QOSSTGUIActions.QIE:
            if bob:
                res = bob.quantum_information_exchange()
                if res:
                    window[QOSSTGUIText.QIE_STATUS].update("QIE status: Done")
                    if values[QOSSTGUIInput.AUTO_EXPORT_DATA]:
                        filename = export_np(
                            bob.signal_data,
                            bob.config.bob.export_directory,
                            data_name="signal",
                        )
                        if filename:
                            window[QOSSTGUIText.LAST_EXPORT].update(filename)
                    autoplot(bob, values)
                else:
                    window[QOSSTGUIText.QIE_STATUS].update("QIE status: Failed")
            else:
                sg.popup_ok("Please read configuration first.")
        elif event == QOSSTGUIActions.DSP:
            if bob:
                res = bob.dsp()
                if res:
                    window[QOSSTGUIText.DSP_STATUS].update("DSP status: Done")
                else:
                    window[QOSSTGUIText.DSP_STATUS].update("DSP status: failed")
                autoplot(bob, values)

            else:
                sg.popup_ok("Please read configuration first.")
        elif event == QOSSTGUIActions.PARAMETERS_ESTIMATION:
            if bob:
                res = bob.parameters_estimation()
                if res:
                    window[QOSSTGUIText.PARAMETERS_ESTIMATION_STATUS].update(
                        "PE status: Done"
                    )
                    window[QOSSTGUIText.PE_ETA].update(bob.config.bob.eta)
                    window[QOSSTGUIText.PE_SHOT].update(
                        np.var(bob.electronic_shot_symbols)
                        - np.var(bob.electronic_symbols)
                    )
                    window[QOSSTGUIText.PE_VEL].update(bob.vel)
                    window[QOSSTGUIText.PE_ETA_T].update(bob.transmittance)
                    window[QOSSTGUIText.PE_T].update(
                        bob.transmittance / bob.config.bob.eta
                    )
                    window[QOSSTGUIText.PE_EXCESS_NOISE_BOB].update(
                        bob.excess_noise_bob
                    )
                    window[QOSSTGUIText.PE_EXCESS_NOISE_ALICE].update(
                        bob.excess_noise_bob / bob.transmittance
                    )
                    window[QOSSTGUIText.PE_SKR].update(bob.skr * 1e-3)
                    window[QOSSTGUIText.PE_PHOTON_NUMBER].update(bob.photon_number)
                    window[QOSSTGUIText.PE_DISTANCE].update(
                        -10 * np.log10(bob.transmittance / bob.config.bob.eta) / 0.2
                    )
                else:
                    window[QOSSTGUIText.PARAMETERS_ESTIMATION_STATUS].update(
                        "PE status: Failed"
                    )
            else:
                sg.popup_ok("Please read configuration first.")
        elif event == QOSSTGUIActions.ERROR_CORRECTION:
            if bob:
                sg.popup_error("Error correction is not yet implemented.")
            else:
                sg.popup_ok("Please read configuration first.")
        elif event == QOSSTGUIActions.PRIVACY_AMPLITICATION:
            if bob:
                sg.popup_error("Privacy amplification is not yet implemented.")
            else:
                sg.popup_ok("Please read configuration first.")
        elif event == QOSSTGUIActions.ACQUISITION_ELEC_NOISE:
            if bob:
                bob.get_electronic_noise_data()
                window[QOSSTGUIText.ELEC_DATA_STATUS].update("Status : Acquired")
                if values[QOSSTGUIInput.AUTO_EXPORT_DATA]:
                    filename, _ = export_np(
                        bob.electronic_noise,
                        bob.config.bob.export_directory,
                        "elec_data",
                    )
                    window[QOSSTGUIText.LAST_EXPORT].update(filename)
            else:
                sg.popup_ok("Please read configuration first.")
        elif event == QOSSTGUIActions.LOAD_ELEC_NOISE:
            if bob:
                bob.load_electronic_noise_data()
                window[QOSSTGUIText.ELEC_DATA_STATUS].update("Status : Loaded")
            else:
                sg.popup_ok("Please read configuration first.")
        elif event == QOSSTGUIActions.SAVE_ELEC_NOISE:
            if bob:
                save, detector, comment = popup_save_electronic_noise(
                    str(bob.config.bob.electronic_noise.path)
                )
                if save:
                    bob.save_electronic_noise_data(detector=detector, comment=comment)
            else:
                sg.popup_ok("Please read configuration first.")
        elif event == QOSSTGUIActions.ACQUISITION_SHOT_NOISE:
            if bob:
                bob.get_electronic_shot_noise_data()
                window[QOSSTGUIText.SHOT_DATA_STATUS].update("Status : Acquired")
                if values[QOSSTGUIInput.AUTO_EXPORT_DATA]:
                    filename, _ = export_np(
                        bob.electronic_shot_noise,
                        bob.config.bob.export_directory,
                        "elec_shot_data",
                    )
                    window[QOSSTGUIText.LAST_EXPORT].update(filename)
            else:
                sg.popup_ok("Please read configuration first.")
        elif event == QOSSTGUIActions.LOAD_SHOT_NOISE:
            if bob:
                bob.load_electronic_shot_noise_data()
                window[QOSSTGUIText.SHOT_DATA_STATUS].update("Status : Loaded")
            else:
                sg.popup_ok("Please read configuration first.")
        elif event == QOSSTGUIActions.SAVE_SHOT_NOISE:
            if bob:
                save, detector, power, comment = popup_save_electronic_shot_noise(
                    bob.config.bob.electronic_shot_noise.path
                )
                if save:
                    bob.save_electronic_shot_noise_data(detector, power, comment)
            else:
                sg.popup_ok("Please read configuration first.")
        elif event in figures_plot_keys:
            for figure in all_figures:
                if event == figure.plot_key:
                    figure.plot(bob)
        elif event in figures_save_keys:
            for figure in all_figures:
                if event == figure.save_key:
                    path = sg.popup_get_file(
                        message="Select location to save figure",
                        save_as=True,
                        file_types=plt.gcf().canvas.get_supported_filetypes().items(),
                        default_extension=".png",
                    )
                    if path:
                        figure.save(path)
                        logger.info("Figure %s was saved to %s", figure.name, str(path))
        elif event == QOSSTGUIActions.EXPORT_ELEC_NOISE:
            filename, _ = export_np(
                bob.electronic_noise.data, bob.config.bob.export_directory, "elec_data"
            )
            window[QOSSTGUIText.LAST_EXPORT].update(filename)
        elif event == QOSSTGUIActions.EXPORT_SHOT_NOISE:
            filename, _ = export_np(
                bob.electronic_shot_noise.data,
                bob.config.bob.export_directory,
                "elec_shot_data",
            )
            window[QOSSTGUIText.LAST_EXPORT].update(filename)
        elif event == QOSSTGUIActions.EXPORT_SIGNAL:
            filename, _ = export_np(
                bob.signal_data, bob.config.bob.export_directory, "signal"
            )
            window[QOSSTGUIText.LAST_EXPORT].update(filename)
        elif event == QOSSTGUIActions.ABOUT:
            sg.popup_ok(LICENSE_TEXT, title="About")
        elif event == QOSSTGUIInput.SELECT_PLOT_STYLE:
            new_style = values[QOSSTGUIInput.SELECT_PLOT_STYLE]
            logger.info("Resetting to default matplotlib style")
            matplotlib.rcParams.update(matplotlib.rcParamsDefault)
            if new_style != "default":
                logger.info("Using new style for plots: %s", new_style)
                styles = get_styles()
                plt.style.use(styles[new_style])
            autoplot(bob, values)
        elif ":" in event:
            # Keyboard events are returned in the format
            # key:number
            # I am not sure what number it is since it doesn't seem
            # to be the ordinal.
            try:
                key_ord = int(event.split(":")[1])
            except ValueError:
                continue
            if key_ord == SEQUENCE[sequence_pointer]:
                sequence_pointer += 1
            else:
                sequence_pointer = 0

            if sequence_pointer == len(SEQUENCE):
                sg.popup_ok(
                    THANKS_STRING.format(release_name=RELEASE_NAME), title="Thanks"
                )
                sequence_pointer = 0
    if bob:
        bob.close()

    window.close()


if __name__ == "__main__":
    main()
