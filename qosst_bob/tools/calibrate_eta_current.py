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
Script to calibrate eta for detectors using the photocurrents.
"""
import logging
import argparse
import time
import datetime
from dataclasses import dataclass, field, fields

import numpy as np

from qosst_hal.amperemeter import GenericAmpereMeter
from qosst_hal.powermeter import GenericPowerMeter
from qosst_hal.voa import GenericVOA
from qosst_core.data import BaseQOSSTData
from qosst_core.utils import configuration_menu, get_object_by_import_path

logger = logging.getLogger(__name__)

DEFAULT_AMPEREMETER1_DEVICE: str = "qosst_hal.amperemeter.FakeAmpereMeter"
DEFAULT_AMPEREMETER2_DEVICE: str = "qosst_hal.amperemeter.FakeAmpereMeter"
DEFAULT_AMPEREMETER1_LOCATION: str = ""
DEFAULT_AMPEREMETER2_LOCATION: str = ""
DEFAULT_AMPEREMETER1_TIMEOUT: int = 1000
DEFAULT_AMPEREMETER2_TIMEOUT: int = 1000
DEFAULT_AMPEREMETER1_RANGE: float = 1e-3
DEFAULT_AMPEREMETER2_RANGE: float = 1e-3

DEFAULT_POWERMETER_DEVICE: str = "qosst_hal.powermeter.FakePowerMeter"
DEFAULT_POWERMETER_LOCATION: str = ""

DEFAULT_VOA_CLASS: str = "qosst_hal.voa.FakeVOA"
DEFAULT_VOA_LOCATION: str = ""
DEFAULT_VOA_START_VALUE: float = 0.0
DEFAULT_VOA_END_VALUE: float = 5.0
DEFAULT_VOA_STEP_VALUE: float = 0.05

DEFAULT_BEAM_SPLITTER_CONVERSION_FACTOR_PM_TO_BOB: float = 1


class CalibrateEtaCurrentData(BaseQOSSTData):
    """
    Data container for the calibration of eta.
    """

    power_values: np.ndarray
    current1_values: np.ndarray
    current2_values: np.ndarray
    date: datetime.datetime

    def __init__(
        self,
        power_values: np.ndarray,
        current1_values: np.ndarray,
        current2_values: np.ndarray,
    ) -> None:
        """
        Args:
            power_values (np.ndarray): array of optical powers.
            current1_values (np.ndarray): array of currents on photodiode 1.
            current2_values (np.ndarray): array of currents on photodiode 2.
        """
        self.power_values = power_values
        self.current1_values = current1_values
        self.current2_values = current2_values
        self.date = datetime.datetime.now()


# pylint: disable=too-many-instance-attributes
@dataclass
class Configuration:
    """
    Configuration object for the eta current script.
    """

    amperemeter1_device: str = field(default=DEFAULT_AMPEREMETER1_DEVICE)
    amperemeter2_device: str = field(default=DEFAULT_AMPEREMETER2_DEVICE)
    amperemeter1_location: str = field(default=DEFAULT_AMPEREMETER1_LOCATION)
    amperemeter2_location: str = field(default=DEFAULT_AMPEREMETER2_LOCATION)
    amperemeter1_timeout: int = field(default=DEFAULT_AMPEREMETER1_TIMEOUT)
    amperemeter2_timeout: int = field(default=DEFAULT_AMPEREMETER2_TIMEOUT)
    amperemeter1_range: float = field(default=DEFAULT_AMPEREMETER1_RANGE)
    amperemeter2_range: float = field(default=DEFAULT_AMPEREMETER2_RANGE)

    powermeter_device: str = field(default=DEFAULT_POWERMETER_DEVICE)
    powermeter_location: str = field(default=DEFAULT_POWERMETER_LOCATION)

    voa_class: str = field(default=DEFAULT_VOA_CLASS)
    voa_location: str = field(default=DEFAULT_VOA_LOCATION)
    voa_start_value: float = field(default=DEFAULT_VOA_START_VALUE)
    voa_end_value: float = field(default=DEFAULT_VOA_END_VALUE)
    voa_step_value: float = field(default=DEFAULT_VOA_STEP_VALUE)

    beam_splitter_conversion_factor_pm_to_bob: float = field(
        default=DEFAULT_BEAM_SPLITTER_CONVERSION_FACTOR_PM_TO_BOB
    )

    def __str__(self) -> str:
        res = ""
        for class_field in fields(self.__class__):
            res += f"{class_field.name}: {getattr(self, class_field.name)}\n"
        return res


# pylint: disable=too-many-locals, too-many-statements
def calibration_eta_current(args: argparse.Namespace):
    """
    Measure eta using photodiode currents.

    Args:
        args (argparse.Namespace): the arguments given to the command line script.
    """
    config = Configuration()

    print("###########################################################################")
    print("## Welcome to the calibration of Bob's detector efficiency using current ##")
    print("###########################################################################")

    print(
        "A laser followed by a voa followed by a known beam splitter should be connected to Bob's input (output port 1 of beam splitter) and the power meter (output port 2 of beam splitter).\n"
    )
    print("The current of the photodetectors should be measured to the amperemeters.\n")

    print("The following configuration is going to be used: \n")
    print(config)
    cont = input("Use this configuration ? [Y/n] ")

    if cont == "n":
        configuration_menu(config)

    voa_values = np.arange(
        config.voa_start_value, config.voa_end_value, config.voa_step_value
    )

    amperemeter1_class = get_object_by_import_path(config.amperemeter1_device)
    amperemeter2_class = get_object_by_import_path(config.amperemeter2_device)
    powermeter_class = get_object_by_import_path(config.powermeter_device)
    voa_class = get_object_by_import_path(config.voa_class)

    amperemeter1: GenericAmpereMeter = amperemeter1_class(
        config.amperemeter1_location,
        timeout=config.amperemeter1_timeout,
        current_range=config.amperemeter1_range,
    )
    amperemeter2: GenericAmpereMeter = amperemeter2_class(
        config.amperemeter2_location,
        timeout=config.amperemeter2_timeout,
        current_range=config.amperemeter2_range,
    )
    powermeter: GenericPowerMeter = powermeter_class(
        config.powermeter_location, timeout=100
    )
    voa: GenericVOA = voa_class(f"{config.voa_location}")

    power_values = np.zeros(shape=len(voa_values))
    current1_values = np.zeros(shape=len(voa_values))
    current2_values = np.zeros(shape=len(voa_values))

    amperemeter1.open()
    amperemeter2.open()
    powermeter.open()
    voa.open()

    for i, voa_value in enumerate(voa_values):
        logger.info("Starting %i/%i", i + 1, len(voa_values))

        logger.info("Setting voa to %f", voa_value)
        voa.set_value(voa_value)
        time.sleep(1)
        power_values[i] = powermeter.read()
        current1_values[i] = amperemeter1.get_current()
        current2_values[i] = amperemeter2.get_current()

        logger.info("Power was estimated at %f mW", power_values[i] * 1e3)
        logger.info("Current 1 was estimated at %f A", current1_values[i])
        logger.info("Current 2 was estimated at %f A", current2_values[i])
        time.sleep(1)

    amperemeter1.close()
    amperemeter2.close()
    powermeter.close()
    voa.close()

    total_photocurrent = current1_values + current2_values
    responsivity, _ = np.polyfit(
        power_values * config.beam_splitter_conversion_factor_pm_to_bob,
        total_photocurrent,
        1,
    )
    eta = responsivity / 1.25  # Only works at 1550nm

    logger.info(
        "Responsivity was estimated at %.20f A/W (eta @ 1550nm : %.20f)",
        responsivity,
        eta,
    )

    if args.save:
        to_save = CalibrateEtaCurrentData(
            power_values=power_values,
            current1_values=current2_values,
            current2_values=current2_values,
        )
        filename = "calibration-eta-current.qosst"
        to_save.save(filename)
        logger.info("Data was saved at %s.", filename)
