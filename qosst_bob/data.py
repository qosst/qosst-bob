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
Module for QOSST data specific to Bob.
"""
from typing import List, Dict, Optional
import datetime

import numpy as np

from qosst_core.data import BaseQOSSTData
from qosst_core.configuration import Configuration


class ElectronicNoise(BaseQOSSTData):
    """
    QOSST data class to hold electronic noise data.
    """

    data: List[np.ndarray]  #: The actual data that was acquired.
    detector: Optional[
        str
    ]  #: Optional detector that was used for this electronic noise.
    comment: Optional[str]  #: Optional comment.
    date: datetime.datetime  #: Datetime of the electronic noise acquisition.

    def __init__(
        self,
        data: List[np.ndarray],
        detector: Optional[str] = None,
        comment: Optional[str] = None,
    ) -> None:
        """
        Args:
            data (List[np.ndarray]): list of data, each element is a ndarray corresponding to a channel.
            detector (Optional[str], optional): name of the detector. Defaults to None.
            comment (Optional[str], optional): comment on the acquisition. Defaults to None.
        """
        self.data = data
        self.detector = detector
        self.comment = comment
        self.date = datetime.datetime.now()

    def __str__(self) -> str:
        res = (
            f"Electronic noise of detector {self.detector} ({len(self.data)} channels)."
        )
        if self.comment:
            res += f" Comment: {self.comment}"
        return res


class ElectronicShotNoise(BaseQOSSTData):
    """
    QOSST data class to hold electronic and shot noise data.
    """

    data: List[np.ndarray]  #: The actual data that was acquired.
    detector: Optional[
        str
    ]  #: Optional detector that was used for this electronic noise.
    power: Optional[float]  #: Optional power.
    comment: Optional[str]  #: Optional comment.
    date: datetime.datetime  #: Datetime of the electronic nois acquisition.

    def __init__(
        self,
        data: List[np.ndarray],
        detector: Optional[str] = None,
        power: Optional[float] = None,
        comment: Optional[str] = None,
    ) -> None:
        """
        Args:
            data (List[np.ndarray]): list of data, each element is a ndarray corresponding to a channel.
            detector (Optional[str], optional): name of the detector. Defaults to None.
            comment (Optional[str], optional): comment on the acquisition. Defaults to None.
        """
        self.data = data
        self.detector = detector
        self.power = power
        self.comment = comment
        self.date = datetime.datetime.now()

    def __str__(self) -> str:
        res = f"Electronic and shot noise of detector {self.detector} and power {self.power} ({len(self.data)} channels)."
        if self.comment:
            res += f" Comment: {self.comment}"
        return res


# pylint: disable=too-many-instance-attributes
class ExcessNoiseResults(BaseQOSSTData):
    """
    Data class for the results of a qosst-bob-excess-noise measurement.
    """

    configuration: Configuration  #: Configuration that was used for the experiment.
    date: datetime.datetime  #: Datetime of the experiment.
    num_rep: int  #: Number of repetitions for this experiment.
    excess_noise_bob: np.ndarray  #: Array of excess noise results.
    transmittance: np.ndarray  #: Array of transmittance results.
    photon_number: np.ndarray  #: Array of photon numbers.
    datetimes: np.ndarray  #: List of datetimes for each point of the experiment.
    electronic_noise: np.ndarray  #: Array of electronic noise (in SNU).
    shot_noise: np.ndarray  #: Array of shot noise (in SNU).
    source_script: str  #: Script that was used for this experiment.
    command_line: str  #: Command line that was used for this experiment.

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        configuration: Configuration,
        num_rep: int,
        excess_noise_bob: np.ndarray,
        transmittance: np.ndarray,
        photon_number: np.ndarray,
        datetimes: np.ndarray,
        electronic_noise: np.ndarray,
        shot_noise: np.ndarray,
        source_script: str,
        command_line: str,
    ) -> None:
        """
        Args:
            configuration (Configuration): configuration that was used for the experiment.
            num_rep (int): number of repetitions in the experiment.
            excess_noise_bob (np.ndarray): array of estimated excess noise bob results (in SNU).
            transmittance (np.ndarray): array of estimated transmittance.
            photon_number (np.ndarray): array of photon numbers.
            datetimes (np.ndarray): list of datetime for each point.
            electronic_noise (np.ndarray): array of estimated electronic noise (in SNU).
            shot_noise (np.ndarray): array of estimated shot noise (in SNU).
            source_script (str): source script that was used for the experiment.
            command_line (str): command line that was used for the experiment.
        """
        self.configuration = configuration
        self.num_rep = num_rep
        self.excess_noise_bob = excess_noise_bob
        self.transmittance = transmittance
        self.photon_number = photon_number
        self.datetimes = datetimes
        self.electronic_noise = electronic_noise
        self.shot_noise = shot_noise
        self.source_script = source_script
        self.command_line = command_line
        self.date = datetime.datetime.now()


class TransmittanceResults(ExcessNoiseResults):
    """
    Data class for the results of a qosst-bob-transmittance measurement.
    """

    attenuation_values: (
        np.ndarray
    )  #: Array of attenuations for this particular experiment.

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        configuration: Configuration,
        num_rep: int,
        excess_noise_bob: np.ndarray,
        transmittance: np.ndarray,
        photon_number: np.ndarray,
        datetimes: np.ndarray,
        electronic_noise: np.ndarray,
        shot_noise: np.ndarray,
        source_script: str,
        command_line: str,
        attenuation_values: np.ndarray,
    ) -> None:
        """
        Args:
            configuration (Configuration): configuration that was used for the experiment.
            num_rep (int): number of repetitions in the experiment.
            excess_noise_bob (np.ndarray): array of estimated excess noise bob results (in SNU).
            transmittance (np.ndarray): array of estimated transmittance.
            photon_number (np.ndarray): array of photon numbers.
            datetimes (np.ndarray): list of datetime for each point.
            electronic_noise (np.ndarray): array of estimated electronic noise (in SNU).
            shot_noise (np.ndarray): array of estimated shot noise (in SNU).
            source_script (str): source script that was used for the experiment.
            command_line (str): command line that was used for the experiment.
            attenuation_values (np.ndarray): array of attenuation values for the tranmisttance experiment.
        """
        self.attenuation_values = attenuation_values
        super().__init__(
            configuration,
            num_rep,
            excess_noise_bob,
            transmittance,
            photon_number,
            datetimes,
            electronic_noise,
            shot_noise,
            source_script,
            command_line,
        )


class OptimizationResults(ExcessNoiseResults):
    """
    Data class for the results of a qosst-bob-optimize measurement.
    """

    parameters: Dict  #: Dict of updated parameters.

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        configuration: Configuration,
        num_rep: int,
        excess_noise_bob: np.ndarray,
        transmittance: np.ndarray,
        photon_number: np.ndarray,
        datetimes: np.ndarray,
        electronic_noise: np.ndarray,
        shot_noise: np.ndarray,
        source_script: str,
        command_line: str,
        parameters: Dict,
    ) -> None:
        """
        Args:
            configuration (Configuration): configuration that was used for the experiment.
            num_rep (int): number of repetitions in the experiment.
            excess_noise_bob (np.ndarray): array of estimated excess noise bob results (in SNU).
            transmittance (np.ndarray): array of estimated transmittance.
            photon_number (np.ndarray): array of photon numbers.
            datetimes (np.ndarray): list of datetime for each point.
            electronic_noise (np.ndarray): array of estimated electronic noise (in SNU).
            shot_noise (np.ndarray): array of estimated shot noise (in SNU).
            source_script (str): source script that was used for the experiment.
            command_line (str): command line that was used for the experiment.
            parameters (Dict): dict of updated parameters.
        """
        self.parameters = parameters
        super().__init__(
            configuration,
            num_rep,
            excess_noise_bob,
            transmittance,
            photon_number,
            datetimes,
            electronic_noise,
            shot_noise,
            source_script,
            command_line,
        )
