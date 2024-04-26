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
Script to calibrate eta for detectors with monitoring output in voltage.
"""
import logging
import argparse
import time
import datetime
from dataclasses import dataclass, field, fields

import numpy as np

from qosst_hal.voltmeter import GenericVoltMeter
from qosst_hal.powermeter import GenericPowerMeter
from qosst_hal.voa import GenericVOA
from qosst_core.data import BaseQOSSTData
from qosst_core.utils import configuration_menu, get_object_by_import_path

logger = logging.getLogger(__name__)

DEFAULT_VOLTMETER1_DEVICE: str = "qosst_hal.voltmeter.FakeVoltMeter"
DEFAULT_VOLTMETER2_DEVICE: str = "qosst_hal.voltmeter.FakeVoltMeter"
DEFAULT_VOLTMETER1_LOCATION: str = ""
DEFAULT_VOLTMETER2_LOCATION: str = ""
DEFAULT_VOLTMETER1_TIMEOUT: int = 1000
DEFAULT_VOLTMETER2_TIMEOUT: int = 1000

DEFAULT_POWERMETER_DEVICE: str = "qosst_hal.powermeter.FakePowerMeter"
DEFAULT_POWERMETER_LOCATION: str = ""

DEFAULT_VOA_CLASS: str = "qosst_hal.voa.FakeVOA"
DEFAULT_VOA_LOCATION: str = ""
DEFAULT_VOA_START_VALUE: float = 0.0
DEFAULT_VOA_END_VALUE: float = 5.0
DEFAULT_VOA_STEP_VALUE: float = 0.05

DEFAULT_BEAM_SPLITTER_CONVERSION_FACTOR_PM_TO_BOB: float = 1


class CalibrateEtaVoltageData(BaseQOSSTData):
    """
    Data container for the calibration of eta.
    """

    power_values: np.ndarray
    voltage1_values: np.ndarray
    voltage2_values: np.ndarray
    date: datetime.datetime

    def __init__(
        self,
        power_values: np.ndarray,
        voltage1_values: np.ndarray,
        voltage2_values: np.ndarray,
    ) -> None:
        """
        Args:
            power_values (np.ndarray): array of optical powers.
            voltage1_values (np.ndarray): array of voltages on monitoring output 1.
            voltage2_values (np.ndarray): array of voltages on monitoring output 2.
        """
        self.power_values = power_values
        self.voltage1_values = voltage1_values
        self.voltage2_values = voltage2_values
        self.date = datetime.datetime.now()


# pylint: disable=too-many-instance-attributes
@dataclass
class Configuration:
    """
    Configuration object for the eta voltage script.
    """

    gain: float = field()
    voltmeter1_device: str = field(default=DEFAULT_VOLTMETER1_DEVICE)
    voltmeter2_device: str = field(default=DEFAULT_VOLTMETER2_DEVICE)
    voltmeter1_location: str = field(default=DEFAULT_VOLTMETER1_LOCATION)
    voltmeter2_location: str = field(default=DEFAULT_VOLTMETER2_LOCATION)
    voltmeter1_timeout: int = field(default=DEFAULT_VOLTMETER1_TIMEOUT)
    voltmeter2_timeout: int = field(default=DEFAULT_VOLTMETER2_TIMEOUT)

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
def calibration_eta_voltage(args: argparse.Namespace):
    """
    Measure eta using monitoring voltage.

    Args:
        args (argparse.Namespace): the arguments given to the command line script.
    """
    config = Configuration(gain=args.gain)

    print("###########################################################################")
    print("## Welcome to the calibration of Bob's detector efficiency using voltage ##")
    print("###########################################################################")

    print(
        "A laser followed by a voa followed by a known beam splitter should be connected to Bob's input (output port 1 of beam splitter) and the power meter (output port 2 of beam splitter).\n"
    )
    print(
        "The voltage monitoring output of the detector should be connected to the volt meters.\n"
    )

    print("The following configuration is going to be used: \n")
    print(config)
    cont = input("Use this configuration ? [Y/n] ")

    if cont == "n":
        configuration_menu(config)

    voa_values = np.arange(
        config.voa_start_value, config.voa_end_value, config.voa_step_value
    )

    voltmeter1_class = get_object_by_import_path(config.voltmeter1_device)
    voltmeter2_class = get_object_by_import_path(config.voltmeter2_device)
    powermeter_class = get_object_by_import_path(config.powermeter_device)
    voa_class = get_object_by_import_path(config.voa_class)

    voltmeter1: GenericVoltMeter = voltmeter1_class(
        config.voltmeter1_location, timeout=config.voltmeter1_timeout
    )
    voltmeter2: GenericVoltMeter = voltmeter2_class(
        config.voltmeter2_location, timeout=config.voltmeter2_timeout
    )
    powermeter: GenericPowerMeter = powermeter_class(
        config.powermeter_location, timeout=100
    )
    voa: GenericVOA = voa_class(f"{config.voa_location}")

    power_values = np.zeros(shape=len(voa_values))
    voltage1_values = np.zeros(shape=len(voa_values))
    voltage2_values = np.zeros(shape=len(voa_values))

    voltmeter1.open()
    voltmeter2.open()
    powermeter.open()
    voa.open()

    for i, voa_value in enumerate(voa_values):
        logger.info("Starting %i/%i", i + 1, len(voa_values))

        logger.info("Setting voa to %f", voa_value)
        voa.set_value(voa_value)
        time.sleep(1)
        power_values[i] = powermeter.read()
        voltage1_values[i] = voltmeter1.get_voltage()
        voltage2_values[i] = voltmeter2.get_voltage()

        logger.info("Power was estimated at %f mW", power_values[i] * 1e3)
        logger.info("Voltage 1 was estimated at %f V", voltage1_values[i])
        logger.info("Voltage 2 was estimated at %f V", voltage2_values[i])
        time.sleep(1)

    voltmeter1.close()
    voltmeter2.close()
    powermeter.close()
    voa.close()

    total_photocurrent = (voltage1_values + voltage2_values) / config.gain
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
        to_save = CalibrateEtaVoltageData(
            power_values=power_values,
            voltage1_values=voltage1_values,
            voltage2_values=voltage2_values,
        )
        filename = "calibration-eta-voltage.qosst"
        to_save.save(filename)
        logger.info("Data was saved at %s.", filename)
