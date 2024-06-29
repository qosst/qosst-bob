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
Layout for Bob gui.
"""
import FreeSimpleGUI as sg

from qosst_bob import __version__
from qosst_bob.gui.figures import all_figures
from qosst_bob.gui.layout_content import (
    QOSSTGUIActions,
    QOSSTGUIText,
    QOSSTGUIInput,
    THEME,
    DEFAULT_CONFIG_LOCATION,
    LOGO_LOCATION,
)
from qosst_bob.gui.plot_utils import get_styles

plot_styles = get_styles()

sg.theme(THEME)

first_column = [
    [sg.Text("Configuration file")],
    [
        sg.Input(str(DEFAULT_CONFIG_LOCATION), key=QOSSTGUIInput.CONFIGURATION_PATH),
        sg.FileBrowse(key=QOSSTGUIActions.BROWSE),
    ],
    [
        sg.Button("Read configuration", key=QOSSTGUIActions.READ_CONFIGURATION),
        sg.Button(
            "Init hardware", key=QOSSTGUIActions.INITIALIZE_HARDWARE, disabled=True
        ),
    ],
    [
        sg.Text(
            "GUI is not enabling laser. It should be configured manually.",
            font="Any 9 italic",
        )
    ],
    [sg.HorizontalSeparator()],
    [sg.Text("Socket connection")],
    [
        sg.Text(
            "Status: Not connected", size=(40, 1), key=QOSSTGUIText.CONNECTION_STATUS
        )
    ],
    [sg.Text("Address for Alice : ", key=QOSSTGUIText.ALICE_ADDRESS)],
    [sg.Button("Connect", key=QOSSTGUIActions.CONNECT, disabled=True)],
    [sg.HorizontalSeparator()],
    [sg.Text("Identification")],
    [
        sg.Text(
            "Identification status : Not done", key=QOSSTGUIText.IDENTIFICATION_STATUS
        )
    ],
    [sg.Button("Identification", key=QOSSTGUIActions.IDENTIFICATION, disabled=True)],
    [sg.HorizontalSeparator()],
    [sg.Text("Initialization (start new frame)")],
    [sg.Text("Frame UUID : ", key=QOSSTGUIText.FRAME_UUID)],
    [sg.Button("Initialization", key=QOSSTGUIActions.INITIALIZATION, disabled=True)],
    [sg.HorizontalSeparator()],
    [sg.Text("Quantum Information Exchange")],
    [sg.Text("QIE status : Not done", key=QOSSTGUIText.QIE_STATUS)],
    [sg.Button("QIE", key=QOSSTGUIActions.QIE, disabled=True)],
    [sg.HorizontalSeparator()],
    [sg.Text("Digital Signal Processing")],
    [sg.Text("DSP status : Not done", key=QOSSTGUIText.DSP_STATUS)],
    [sg.Button("DSP", key=QOSSTGUIActions.DSP, disabled=True)],
    [sg.HorizontalSeparator()],
    [sg.Text("Parameters Estimation")],
    [sg.Text("PE status : Not done", key=QOSSTGUIText.PARAMETERS_ESTIMATION_STATUS)],
    [
        sg.Button(
            "Parameters Estimation",
            key=QOSSTGUIActions.PARAMETERS_ESTIMATION,
            disabled=True,
        )
    ],
    [sg.HorizontalSeparator()],
    [sg.Text("Error Correction")],
    [sg.Text("EC status : Not done", key=QOSSTGUIText.ERROR_CORRECTION_STATUS)],
    [
        sg.Button(
            "Error Correction", key=QOSSTGUIActions.ERROR_CORRECTION, disabled=True
        )
    ],
    [sg.HorizontalSeparator()],
    [sg.Text("Privacy Amplification")],
    [sg.Text("PA status : Not done", key=QOSSTGUIText.PRIVACY_AMPLIFICATION_STATUS)],
    [
        sg.Button(
            "Privacy Amplification",
            key=QOSSTGUIActions.PRIVACY_AMPLITICATION,
            disabled=True,
        )
    ],
    [sg.HorizontalSeparator()],
    [
        sg.Button("Exit", key=QOSSTGUIActions.EXIT),
        sg.Button("About", key=QOSSTGUIActions.ABOUT),
    ],
]

tab_noises = [
    [
        sg.Button(
            "Acquire electronic noise samples",
            key=QOSSTGUIActions.ACQUISITION_ELEC_NOISE,
            disabled=True,
        ),
        sg.Button(
            "Load electronic noise samples",
            key=QOSSTGUIActions.LOAD_ELEC_NOISE,
            disabled=True,
        ),
        sg.Text("Status : Empty", key=QOSSTGUIText.ELEC_DATA_STATUS),
    ],
    [
        sg.Button(
            "Acquire shot noise and electronic noise samples",
            key=QOSSTGUIActions.ACQUISITION_SHOT_NOISE,
            disabled=True,
        ),
        sg.Button(
            "Load shot noise and electronic noise samples",
            key=QOSSTGUIActions.LOAD_SHOT_NOISE,
            disabled=True,
        ),
        sg.Text("Status : Empty", key=QOSSTGUIText.SHOT_DATA_STATUS),
    ],
    [
        sg.Button(
            "Save electronic noise", key=QOSSTGUIActions.SAVE_ELEC_NOISE, disabled=True
        ),
        sg.Button(
            "Save electronic and shot noise",
            key=QOSSTGUIActions.SAVE_SHOT_NOISE,
            disabled=True,
        ),
    ],
]

tab_exports = [
    [
        sg.Checkbox(
            "Auto export data ?", default=False, key=QOSSTGUIInput.AUTO_EXPORT_DATA
        ),
    ],
    [
        sg.Button("Export electronic noise", key=QOSSTGUIActions.EXPORT_ELEC_NOISE),
        sg.Button(
            "Export electronic and shot noise", key=QOSSTGUIActions.EXPORT_SHOT_NOISE
        ),
        sg.Button("Export signal", key=QOSSTGUIActions.EXPORT_SIGNAL),
    ],
    [sg.Text("Export directory : "), sg.Text("", key=QOSSTGUIText.EXPORT_DIRECTORY)],
    [sg.Text("Last export : "), sg.Text("", key=QOSSTGUIText.LAST_EXPORT)],
]

parameters_estimation_column_1 = [
    [
        sg.Text("ETA"),
        sg.Text("", key=QOSSTGUIText.PE_ETA),
    ],
    [
        sg.Text("<n>"),
        sg.Text("", key=QOSSTGUIText.PE_PHOTON_NUMBER),
        sg.Text("photon/symbol"),
    ],
    [
        sg.Text("Total transmittance"),
        sg.Text("", key=QOSSTGUIText.PE_ETA_T),
    ],
    [
        sg.Text("Transmittance"),
        sg.Text("", key=QOSSTGUIText.PE_T),
    ],
    [
        sg.Text("Equivalent distance @0.2dB/km"),
        sg.Text("", key=QOSSTGUIText.PE_DISTANCE),
        sg.Text("km"),
    ],
]

parameters_estimation_column_2 = [
    [
        sg.Text("Shot noise"),
        sg.Text("", key=QOSSTGUIText.PE_SHOT),
        sg.Text("a.u."),
    ],
    [
        sg.Text("Vel"),
        sg.Text("", key=QOSSTGUIText.PE_VEL),
        sg.Text("SNU"),
    ],
    [
        sg.Text("Excess noise Bob"),
        sg.Text("", key=QOSSTGUIText.PE_EXCESS_NOISE_BOB),
        sg.Text("SNU"),
    ],
    [
        sg.Text("Excess noise Alice"),
        sg.Text("", key=QOSSTGUIText.PE_EXCESS_NOISE_ALICE),
        sg.Text("SNU"),
    ],
    [
        sg.Text("SKR"),
        sg.Text("", key=QOSSTGUIText.PE_SKR),
        sg.Text("kbit/s"),
    ],
]

figures_tab_group_layout = []

autoplot_figures = [
    sg.Text("Autoplot : "),
]

for figure in all_figures:
    tab = [
        [sg.Canvas(key=figure.key)],
        [
            sg.Button(f"Plot {figure.name}", key=figure.plot_key),
            sg.Button(f"Export {figure.name}", key=figure.save_key),
        ],
    ]
    figures_tab_group_layout.append(
        sg.Tab(figure.name.capitalize(), tab),
    )

    autoplot_figures.append(
        sg.Checkbox(
            figure.name.capitalize(),
            default=figure.default_autoplot,
            key=figure.autoplot_key,
        )
    )

second_column = [
    [
        sg.TabGroup(
            [
                [
                    sg.Tab("Noises", tab_noises),
                    sg.Tab("Exports", tab_exports),
                ]
            ]
        )
    ],
    [sg.HorizontalSeparator()],
    autoplot_figures,
    [
        sg.Text("Plot style: "),
        sg.Combo(
            ["default"] + list(plot_styles),
            default_value="default",
            readonly=True,
            expand_x=True,
            key=QOSSTGUIInput.SELECT_PLOT_STYLE,
            enable_events=True,
        ),
    ],
    [
        sg.TabGroup(
            [figures_tab_group_layout],
        )
    ],
    [sg.HorizontalSeparator()],
    [sg.Text("Parameters estimation")],
    [
        sg.Column(parameters_estimation_column_1),
        sg.Push(),
        sg.VerticalSeparator(),
        sg.Column(parameters_estimation_column_2),
    ],
]

tab_configuration_qi = [
    [
        sg.Column(
            [
                [
                    sg.Text("Num. Symbols"),
                    sg.Text("", key=QOSSTGUIText.CONFIGURATION_QI_NUM_SYMBOLS),
                ],
                [
                    sg.Text("Shift"),
                    sg.Text("", key=QOSSTGUIText.CONFIGURATION_QI_SHIFT),
                    sg.Text("MHz"),
                ],
                [
                    sg.Text("Symbol Rate"),
                    sg.Text("", key=QOSSTGUIText.CONFIGURATION_QI_SYMBOL_RATE),
                    sg.Text("MBaud"),
                ],
            ]
        ),
        sg.Column(
            [
                [
                    sg.Text("Roll Off"),
                    sg.Text("", key=QOSSTGUIText.CONFIGURATION_QI_ROLL_OFF),
                ],
                [
                    sg.Text("Modulation"),
                    sg.Text("", key=QOSSTGUIText.CONFIGURATION_QI_MODULATION),
                ],
                [
                    sg.Text("Modulation Size"),
                    sg.Text("", key=QOSSTGUIText.CONFIGURATION_QI_MODULATION_SIZE),
                ],
            ]
        ),
    ]
]

tab_configuration_zc = [
    [sg.Text("Root"), sg.Text("", key=QOSSTGUIText.CONFIGURATION_ZC_ROOT)],
    [sg.Text("Length"), sg.Text("", key=QOSSTGUIText.CONFIGURATION_ZC_LENGTH)],
    [
        sg.Text("Rate"),
        sg.Text("", key=QOSSTGUIText.CONFIGURATION_ZC_RATE),
        sg.Text("MSamples/s"),
    ],
]

tab_configuration_pilots = [
    [
        sg.Text("Frequencies"),
        sg.Text("", key=QOSSTGUIText.CONFIGURATION_PILOTS_FREQUENCIES),
    ]
]

tab_configuration_dsp = [
    [
        sg.Column(
            [
                [
                    sg.Text("Tone Cutoff"),
                    sg.Text("", key=QOSSTGUIText.CONFIGURATION_DSP_TONE_CUTOFF),
                    sg.Text("MHz"),
                ],
                [
                    sg.Text("Subframes Size"),
                    sg.Text("", key=QOSSTGUIText.CONFIGURATION_DSP_SUBFRAMES_SIZE),
                ],
                [
                    sg.Text("Abort Clock Recovery"),
                    sg.Text(
                        "", key=QOSSTGUIText.CONFIGURATION_DSP_ABORT_CLOCK_RECOVERY
                    ),
                ],
            ]
        ),
        sg.Column(
            [
                [
                    sg.Text("Alice DAC Rate"),
                    sg.Text("", key=QOSSTGUIText.CONFIGURATION_DSP_ALICE_DAC_RATE),
                    sg.Text("MSamples/s"),
                ],
                [
                    sg.Text("Exclusion Zone"),
                    sg.Text("", key=QOSSTGUIText.CONFIGURATION_DSP_EXCLUSION_ZONE),
                ],
                [
                    sg.Text("Phase Filtering"),
                    sg.Text("", key=QOSSTGUIText.CONFIGURATION_DSP_PHASE_FILTERING),
                ],
            ]
        ),
    ]
]

third_column = [
    [sg.Push(), sg.Image(source=str(LOGO_LOCATION), subsample=6), sg.Push()],
    [sg.Text("Logs")],
    [
        sg.Multiline(
            size=(80, 35),
            autoscroll=True,
            text_color="white",
            background_color="black",
            key=QOSSTGUIText.LOGGER,
        ),
    ],
    [sg.HorizontalSeparator()],
    [
        sg.Text("Configuration"),
    ],
    [
        sg.TabGroup(
            [
                [
                    sg.Tab("QI", tab_configuration_qi),
                    sg.Tab("ZC", tab_configuration_zc),
                    sg.Tab("Pilots", tab_configuration_pilots),
                    sg.Tab("DSP", tab_configuration_dsp),
                    sg.Tab("Other", []),
                ]
            ]
        )
    ],
]

layout_tab_controls = [
    [
        sg.Column(first_column),
        sg.VSeparator(),
        sg.Column(second_column),
        sg.VSeparator(),
        sg.Column(third_column),
    ]
]

layout = [
    [
        layout_tab_controls,
    ],
    [sg.Push(), sg.Text(f"qosst-bob version {__version__}")],
]
