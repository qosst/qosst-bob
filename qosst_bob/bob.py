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
Client code for QOSST Bob.
"""
import logging
import time
from typing import Any, Optional
import uuid
import traceback
from copy import deepcopy

import numpy as np

from qosst_core.configuration import Configuration

from qosst_core.configuration.exceptions import InvalidConfiguration
from qosst_core.control_protocol import QOSST_VERSION
from qosst_core.control_protocol.codes import QOSSTCodes
from qosst_core.control_protocol.sockets import QOSSTClient
from qosst_core.notifications import QOSSTNotifier
from qosst_core.participant import Participant
from qosst_hal.switch import GenericSwitch
from qosst_hal.laser import GenericLaser
from qosst_hal.adc import GenericADC
from qosst_hal.polarisation_controller import (
    GenericPolarisationController,
    PolarisationControllerChannel,
)
from qosst_hal.powermeter import GenericPowerMeter

from qosst_bob.dsp import dsp_bob, special_dsp
from qosst_bob.dsp.dsp import find_global_angle
from qosst_bob.data import ElectronicNoise, ElectronicShotNoise

logger = logging.getLogger(__name__)


# pylint: disable=too-many-instance-attributes, too-many-public-methods
class Bob:
    """
    Client class that will interact with the sockets and the hardware.
    """

    config_path: str  #: Location of the configuration path.
    frame_uuid: Optional[uuid.UUID]  #: UUID of the current frame.

    is_connected: bool  #: True if the client is connected to the server.

    electronic_noise: Optional[ElectronicNoise]  #: Electronic noise object.
    electronic_shot_noise: Optional[
        ElectronicShotNoise
    ]  #: Electronic shot noise object.
    adc_data: Optional[list[np.ndarray]]  #: List of arrays containing current ADC
    signal_data: Optional[
        list[np.ndarray]
    ]  #: List of arrays containing the signal data.
    begin_data: Optional[int]  #: Indice of beginning of data
    end_data: Optional[int]  #: Indice of end of data
    end_electronic_shot_noise: (
        int  #: End of the electronic shot noise data in case of automatic calibration.
    )

    quantum_symbols: Optional[np.ndarray]  #: Array of corrected symbols.
    electronic_symbols: Optional[
        np.ndarray
    ]  #: Array of electronic noise symbols after DSP.
    electronic_shot_symbols: Optional[
        np.ndarray
    ]  #: Array of electronic+shot noise symbols after DSP.
    quantum_data_phase_noisy: Optional[np.ndarray]  #: Array of symbols with phase noise
    received_tone: Optional[np.ndarray]  #: Received stone

    indices: Optional[np.ndarray]  #: Indices to ask for to Alice
    alice_symbols: Optional[np.ndarray]  #: Symbols of Alice

    transmittance: float  #: Total transmittance estimated.
    excess_noise_bob: float  #: Excess noise estimated at Bob.
    vel: float  #: Normalised electronic noise.
    skr: float  #: Secret key rate.
    photon_number: float  #: Mean photon number at Alice's side.

    adc: Optional[GenericADC]  #: ADC device for Bob.
    switch: Optional[GenericSwitch]  #: Switch device for Bob.
    laser: Optional[GenericLaser]  #: Laser device for Bob.
    polarisation_controller: Optional[
        GenericPolarisationController
    ]  #: For polarisation recovery, polarisation controller for Bob.
    powermeter: Optional[
        GenericPowerMeter
    ]  #: For polarisation recovery, powermeter for Bob.

    notifier: QOSSTNotifier  #: Notifier for Bob.

    config: Optional[Configuration]  #: Configuration object.

    enable_laser: (
        bool  #: Enable the laser if True. Meant to be False when using the GUI.
    )

    def __init__(self, config_path: str, enable_laser: bool = True):
        """
        Args:
            config_path (str): configuration path.
            enable_laser (bool, optional): if True, the laser will be enabled by the client. If False, no interaction is made with the laser. Useful for the GUI. Defaults to True.
        """
        self.config_path = config_path
        self.is_connected = False
        self.config = None

        self.electronic_noise = None
        self.electronic_shot_noise = None

        self.electronic_symbols = None
        self.electronic_shot_symbols = None

        self.end_electronic_shot_noise = 0

        self.indices = None
        self.alice_symbols = None

        self.transmittance = 1
        self.excess_noise_bob = 0
        self.vel = 0
        self.skr = 0

        self.adc_data = None
        self.signal_data = None
        self.begin_data = None
        self.end_data = None

        self.received_tone = None
        self.quantum_data_phase_noisy = None
        self.quantum_symbols = None

        self.enable_laser = enable_laser

        self.laser = None
        self.switch = None
        self.adc = None

        logger.info("Initializing Bob...")

        self.frame_uuid = None

        self.load_configuration()
        self._init_socket()
        self._init_notifier()

    def load_configuration(self) -> None:
        """
        Load or reload the configuration.
        """
        if self.config is not None:
            logger.info("Reloading configuration at location %s", self.config_path)
        else:
            logger.info("Loading configuration at location %s", self.config_path)
        try:
            self.config = Configuration(self.config_path)
        except InvalidConfiguration as exc:
            logger.fatal(
                "The configuration cannot be read (%s). Priting the full traceback and closing the server.",
                str(exc),
            )
            print(traceback.format_exc())
            self.close()

        assert self.config is not None
        if self.config.bob is None:
            raise InvalidConfiguration(
                "bob section is not present and configuration is therefore not valid for Bob client."
            )
        if self.config.frame is None:
            raise InvalidConfiguration(
                "frame section is not present and configuration is therefore not valid for Bob client."
            )

        if self.config.clock is None:
            raise InvalidConfiguration(
                "clock section is not present and configuration is therefore not valid for Bob client."
            )

        if self.config.local_oscillator is None:
            raise InvalidConfiguration(
                "local_oscillator section is not present and configuration is therefore not valid for Bob client."
            )
        # Check schema
        logger.info("Checking the detection schema")
        detection_schema = self.config.bob.schema
        logger.info("Schema is %s", str(detection_schema))
        detection_schema.check()
        logger.info("Detection schema accepted.")

    def open_hardware(self) -> None:
        """
        Open the ADC, the switch and the laser (if enable laser is True).
        """
        assert self.config is not None and self.config.bob is not None
        logger.info("Opening hardware")
        self._init_data_adc()
        logger.info(
            "Opening switch (%s) at location %s",
            str(self.config.bob.switch.device),
            self.config.bob.switch.location,
        )
        self.switch = self.config.bob.switch.device(
            self.config.bob.switch.location, timeout=self.config.bob.switch.timeout
        )
        self.switch.open()
        self.switch.set_state(self.config.bob.switch.signal_state)

        if self.enable_laser:
            logger.info(
                "Opening laser (%s) at location %s",
                str(self.config.bob.laser.device),
                self.config.bob.laser.location,
            )
            self.laser = self.config.bob.laser.device(self.config.bob.laser.location)
            self.laser.open()

            logger.info(
                "Setting parameters to the laser: %s",
                str(self.config.bob.laser.parameters),
            )
            self.laser.set_parameters(**self.config.bob.laser.parameters)

            logger.info("Enabling the laser.")
            self.laser.enable()
            logger.info("Laser enabled.")
        else:
            self.laser = None

        if self.config.bob.polarisation_recovery.use:
            logging.info("Opening hardware for polarisation recovery")
            logging.info(
                "Opening polarisation controller %s at location %s",
                str(
                    self.config.bob.polarisation_recovery.polarisation_controller.device
                ),
                self.config.bob.polarisation_recovery.polarisation_controller.location,
            )
            self.polarisation_controller = self.config.bob.polarisation_recovery.polarisation_controller.device(
                self.config.bob.polarisation_recovery.polarisation_controller.location
            )

            logging.info(
                "Opening powermeter %s at location %s",
                str(self.config.bob.polarisation_recovery.powermeter.device),
                self.config.bob.polarisation_recovery.powermeter.location,
            )
            self.powermeter = self.config.bob.polarisation_recovery.powermeter.device(
                self.config.bob.polarisation_recovery.powermeter.location,
                timeout=self.config.bob.polarisation_recovery.powermeter.timeout,
            )

            self.polarisation_controller.open()
            self.powermeter.open()

            self.polarisation_controller.home()
        else:
            self.polarisation_controller = None
            self.powermeter = None

    def close_hardware(self) -> None:
        """
        Close the ADC, the switch and the laser (if it was open).
        """
        logger.info("Closing hardware.")
        if self.laser is not None:
            logger.info("Closing laser.")
            self.laser.disable()
            self.laser.close()
        if self.adc:
            logger.info("Closing ADC.")
            self.adc.close()
        if self.switch:
            logger.info("Closing switch.")
            self.switch.close()
        if self.polarisation_controller:
            logger.info("Closing polarisation controller")
            self.polarisation_controller.close()
        if self.powermeter:
            logger.info("Closing powermeter")
            self.powermeter.close()

    def _init_socket(self):
        """
        Init the QOSST socket.
        """
        self.socket = QOSSTClient(
            self.config.bob.network.server_address, self.config.bob.network.server_port
        )
        self.socket.open()

    def _init_data_adc(self):
        """
        Configure the acquisition of the ADC.
        """
        acquisition_time = self.config.bob.adc.acquisition_time
        if not acquisition_time:  # acquisition_time = 0
            logger.info("Automatically computing the acquisition time.")
            num_samples = (
                self.config.bob.dsp.alice_dac_rate
                / self.config.frame.quantum.symbol_rate
            ) * self.config.frame.quantum.num_symbols
            acquisition_time = (
                self.config.bob.adc.overhead_time
                + (num_samples + self.config.frame.zadoff_chu.length)
                / self.config.bob.dsp.alice_dac_rate
            )
        if self.config.clock.sharing:
            if self.config.clock.master == Participant.ALICE:
                self.adc = self.config.bob.adc.device(
                    self.config.bob.adc.location,
                    self.config.bob.adc.channels,
                    clk_in=True,
                    gpioa_out=True,
                )
            else:
                self.adc = self.config.bob.adc.device(
                    self.config.bob.adc.location,
                    self.config.bob.adc.channels,
                    clk_out=True,
                    gpioa_out=True,
                )
        else:
            self.adc = self.config.bob.adc.device(
                self.config.bob.adc.location,
                self.config.bob.adc.channels,
                gpioa_out=True,
            )
        self.adc.set_acquisition_parameters(
            acquisition_time=self.config.bob.adc.acquisition_time,
            target_rate=self.config.bob.adc.rate,
            **self.config.bob.adc.extra_acquisition_parameters
        )

    def _get_adc_data(self):
        """
        Get the data from the ADC.
        """
        self.adc_data = self.adc.get_data()

    def _init_notifier(self):
        """
        Initialize the notifier.
        """
        if self.config.notifications is not None and self.config.notifications.notify:
            logger.info("Initializing notifier")
            self.notifier = self.config.notifications.notifier(
                self.config.notifications.args
            )
        else:
            self.notifier = None

    def connect(self) -> bool:
        """
        Connect the client to the server.

        Returns:
            bool: True if the operation succeded. False otherwise.
        """
        try:
            self.socket.connect()
            self.is_connected = True
            return True
        except ConnectionRefusedError:
            return False

    def close(self) -> None:
        """
        Close the socket and Bob.
        """
        self.socket.close()
        self.close_hardware()

    def identification(self) -> bool:
        """
        Identify the client to the server and get the server identification.

        Returns:
            bool: True if the identification was successful. False otherwise.
        """
        assert self.config is not None
        code, _ = self.socket.request(
            QOSSTCodes.IDENTIFICATION_REQUEST,
            {
                "serial_number": self.config.serial_number,
                "qosst_version": QOSST_VERSION,
            },
        )

        return code == QOSSTCodes.IDENTIFICATION_RESPONSE

    def initialization(self) -> bool:
        """
        Start a new frame and initializalize to the server.

        Returns:
            bool: True if the initialization was successful. False otherwise.
        """
        self.frame_uuid = uuid.uuid4()
        code, _ = self.socket.request(
            QOSSTCodes.INITIALIZATION_REQUEST, {"frame_uuid": str(self.frame_uuid)}
        )

        if code == QOSSTCodes.INITIALIZATION_ACCEPTED:
            return True
        return False

    def quantum_information_exchange(self) -> bool:
        """
        Start the Quantum Information Exchange (QIE).

        Returns:
            bool: True if the QIE was successful. False otherwise.
        """
        assert self.config is not None and self.config.bob is not None
        assert self.switch is not None

        if self.config.bob.polarisation_recovery.use:
            logging.info("Performing automatic polarisation recovery.")
            code, _ = self.socket.request(QOSSTCodes.REQUEST_POLARISATION_RECOVERY)

            if code != QOSSTCodes.POLARISATION_RECOVERY_ACK:
                return False

            self._optimal_polarisation_finding()
            code, _ = self.socket.request(QOSSTCodes.END_POLARISATION_RECOVERY)

            if code != QOSSTCodes.POLARISATION_RECOVERY_ENDED:
                return False

        code, _ = self.socket.request(QOSSTCodes.QIE_REQUEST)

        if code != QOSSTCodes.QIE_READY:
            return False

        if self.config.bob.automatic_shot_noise_calibration:
            self.get_electronic_shot_noise_data()

        if self.config.bob.switch.switching_time:
            # Set the switch in calibration state
            logger.info(
                "Performating automating switching with switching time %f",
                self.config.bob.switch.switching_time,
            )
            self.switch.set_state(self.config.bob.switch.calibration_state)

        self._start_acquisition()

        if self.config.bob.switch.switching_time:
            # and wait the time, then go back to signal state
            # and trigger
            time.sleep(self.config.bob.switch.switching_time)
            self.switch.set_state(self.config.bob.switch.signal_state)

        code, _ = self.socket.request(QOSSTCodes.QIE_TRIGGER)

        if code == QOSSTCodes.QIE_EMISSION_STARTED:
            self._wait_for_timer()

            code, _ = self.socket.request(QOSSTCodes.QIE_ACQUISITION_ENDED)
            self._get_adc_data()

            self._stop_acquisition()
            self.signal_data = deepcopy(self.adc_data)

            assert self.signal_data is not None
            if self.config.bob.switch.switching_time:
                self.end_electronic_shot_noise = int(
                    self.config.bob.switch.switching_time * self.config.bob.adc.rate
                )
                self.electronic_shot_noise = ElectronicShotNoise(
                    [
                        channel_data[: self.end_electronic_shot_noise]
                        for channel_data in self.signal_data
                    ]
                )
            return code == QOSSTCodes.QIE_ENDED

        self._stop_acquisition()

        return False

    def dsp(self) -> bool:
        """
        Apply the Digital Signal Processing (DSP) algorithm.

        Returns:
            bool: True if the DSP was successful, False otherwise.
        """
        self._do_dsp()
        return True

    def parameters_estimation(self) -> bool:
        """
        Run the parameters estimation algorithm.

        Returns:
            bool: True if the parameters estimation wen well, False otherwise.
        """
        assert (
            self.config is not None
            and self.config.bob is not None
            and self.config.frame is not None
        )
        assert self.quantum_symbols is not None
        assert self.alice_symbols is not None
        assert self.electronic_symbols is not None
        assert self.electronic_shot_symbols is not None
        logger.info("Start of parameters estimation")
        logger.info("Requesting photon number to Alice")
        self.photon_number = self.get_alice_photon_number()

        logger.info("Measuring transmittance and excess noise")
        (
            self.transmittance,
            self.excess_noise_bob,
            self.vel,
        ) = self.config.bob.parameters_estimation.estimator.estimate(
            self.alice_symbols,
            self.quantum_symbols[self.indices],
            self.photon_number,
            self.electronic_symbols,
            self.electronic_shot_symbols,
        )

        logger.info("Total transmittance was estimated at %f", self.transmittance)
        logger.info(
            "Transmittance was estimated at %f",
            self.transmittance / self.config.bob.eta,
        )
        logger.info("Excess noise Bob was estimated at %f", self.excess_noise_bob)
        logger.info(
            "Excess noise Alice was estimated at %f",
            self.excess_noise_bob / self.transmittance,
        )
        logger.info("Electronic noise was estimated at %f", self.vel)

        logger.info("Computing secret key rate")
        try:
            self.skr = (
                self.config.frame.quantum.symbol_rate
                * self.config.bob.parameters_estimation.skr_calculator.skr(
                    Va=2 * self.photon_number,
                    T=self.transmittance / self.config.bob.eta,
                    xi=self.excess_noise_bob / self.transmittance,
                    eta=self.config.bob.eta,
                    Vel=self.vel,
                    beta=0.95,
                )
            )
        except ValueError:
            self.skr = -1

        logger.info("Secret key rate was computed at %f bit/s", self.skr)

        logger.info("Sending parameters estimation results to Alice.")
        code, _ = self.socket.request(
            QOSSTCodes.PE_FINISHED,
            {
                "n_photon": self.photon_number,
                "transmittance": self.transmittance / self.config.bob.eta,
                "excess_noise": self.excess_noise_bob / self.transmittance,
                "eta": self.config.bob.eta,
                "key_rate": self.skr,
                "electronic_noise": self.vel,
            },
        )

        if code == QOSSTCodes.PE_APPROVED:
            return True
        return False

    def error_correction(self) -> bool:
        """
        Apply error correction on the data.

        Raises:
            NotImplementedError: This function is not yet implemented.

        Returns:
            bool: True if the operation was successful, False otherwise.
        """
        raise NotImplementedError("Error correction has not yet been implemented.")

    def privacy_amplification(self) -> bool:
        """
        Apply privacy amplification on the data.

        Raises:
            NotImplementedError: This function is not yet implemented.

        Returns:
            bool: True if the operation was successful, False otherwise.
        """
        raise NotImplementedError("Privacy amplification has not yet been implemented.")

    def _start_acquisition(self):
        """
        Start the acquisition.
        """
        self.adc.arm_acquisition()
        self.adc.trigger()

    def _stop_acquisition(self):
        """
        Stop the acquisition.
        """
        self.adc.stop_acquisition()

    def _do_dsp(self):
        """
        Actually apply the DSP to the data.

        Also request the symbols to Alice, in order to correct the global phase.
        """
        logger.info("DSP start")

        logger.info("Applying DSP on quantum data")
        data = self.signal_data[0]

        if self.config.bob.switch.switching_time:
            self.end_electronic_shot_noise = int(
                self.config.bob.switch.switching_time * self.config.bob.adc.rate
            )
            data = data[self.end_electronic_shot_noise :]

        self.quantum_symbols, params, dsp_debug = dsp_bob(data, self.config)

        # Correct global phase of each frame of quantum symbols
        logger.info("Correcting global frame on each subframe")
        self.indices = []
        self.alice_symbols = []
        last_indice = -1
        for i, frame in enumerate(self.quantum_symbols):
            logger.info(
                "Finding global angle at frame %i/%i",
                i + 1,
                len(self.quantum_symbols),
            )

            # Generate indices
            indices = np.arange(last_indice + 1, last_indice + 1 + len(frame))
            np.random.shuffle(indices)
            indices = indices[
                : int(len(frame) * self.config.bob.parameters_estimation.ratio)
            ]

            # Request symbols to Alice
            logger.info("Requesting %i symbols to Alice", len(indices))

            _, data = self.socket.request(
                QOSSTCodes.PE_SYMBOLS_REQUEST, {"indices": indices.tolist()}
            )

            alice_symbols = np.array(data["symbols_real"]) + 1j * np.array(
                data["symbols_imag"]
            )

            # Find global angle

            angle, cov = find_global_angle(
                frame[indices - (last_indice + 1)], alice_symbols
            )

            logger.info("Global angle found : %.2f with cov %.2f", angle, cov)

            self.quantum_symbols[i] = np.exp(1j * angle) * frame

            # Add indices and symbols, update last_indice
            self.indices.append(indices)
            self.alice_symbols.append(alice_symbols)
            last_indice = last_indice + len(frame)

        self.quantum_symbols = np.concatenate(self.quantum_symbols)
        self.indices = np.concatenate(self.indices)
        self.alice_symbols = np.concatenate(self.alice_symbols)

        self.quantum_data_phase_noisy = np.concatenate(dsp_debug.uncorrected_data)
        self.received_tone = np.concatenate(dsp_debug.tones)
        self.begin_data = dsp_debug.begin_data
        self.end_data = dsp_debug.end_data

        logger.info(
            "Time between end of shot noise and signal : %f ms",
            self.begin_data / self.config.bob.adc.rate * 1e3,
        )

        logger.info("Applying DSP on elec and elec+shot noise data")

        params.elec_noise_estimation_ratio = self.config.bob.dsp.elec_noise_estimation_ratio
        params.elec_shot_noise_estimation_ratio = self.config.bob.dsp.elec_shot_noise_estimation_ratio
        self.electronic_symbols, self.electronic_shot_symbols = special_dsp(
            self.electronic_noise.data, self.electronic_shot_noise.data, params
        )

        logger.info("DSP end")

    def _wait_for_timer(self):
        """
        Timer for the acquisition.
        """
        logger.info("Starting timer of 1 second")
        time.sleep(1)
        logger.info("Timer ended")

    def get_electronic_noise_data(self):
        """
        Acquire electronic+shot noise data using the configured adc.

        It will make an acquisition, and compute the noise density.
        """
        self._start_acquisition()
        self._get_adc_data()
        assert self.adc_data is not None
        self.electronic_noise = ElectronicNoise(deepcopy(self.adc_data))
        self._stop_acquisition()

    def load_electronic_noise_data(self):
        """
        Load electronic noise from a numpy file.
        """
        self.electronic_noise = ElectronicNoise.load(
            self.config.bob.electronic_noise.path
        )

    def save_electronic_noise_data(self, detector: str = "", comment: str = "") -> None:
        """
        Save the electronic noise data as numpy file.
        """
        assert self.config is not None and self.config.bob is not None
        if self.electronic_noise:
            self.electronic_noise.detector = detector
            self.electronic_noise.comment = comment
            self.electronic_noise.dump(self.config.bob.electronic_noise.path)
            logger.info(
                "Electronic noise was saved at location %s (detector: %s, comment: %s)",
                str(self.config.bob.electronic_noise.path),
                detector,
                comment,
            )

    def get_electronic_shot_noise_data(self) -> None:
        """
        Acquire electronic+shot noise data using the configured adc.

        It will switch the state of the optical switch to the calibration state,
        make an acquisition, switch back, and finally compute the noise density.
        """
        assert self.config is not None and self.config.bob is not None
        assert self.switch is not None
        if self.config.bob.switch.switching_time:
            logger.warning(
                "Getting electronic and shot noise while automatic calibration is enabled (switching time is not 0)."
            )
        logger.info("Starting calibration of shot noise.")
        self.switch.set_state(self.config.bob.switch.calibration_state)
        self._start_acquisition()
        self._get_adc_data()
        assert self.adc_data is not None
        self.electronic_shot_noise = ElectronicShotNoise(deepcopy(self.adc_data))
        self._stop_acquisition()
        self.switch.set_state(self.config.bob.switch.signal_state)
        logger.info("Calibration of shot noise finished.")

    def load_electronic_shot_noise_data(self) -> None:
        """
        Load electronic+shot noise from a numpy file.
        """
        assert self.config is not None and self.config.bob is not None
        if self.config.bob.switch.switching_time:
            logger.warning(
                "Loading electronic and shot noise while automatic calibration is enabled (switching time is not 0)."
            )
        self.electronic_shot_noise = ElectronicShotNoise.load(
            self.config.bob.electronic_shot_noise.path
        )

    def save_electronic_shot_noise_data(
        self, detector: str = "", power: Optional[float] = None, comment: str = ""
    ) -> None:
        """
        Save the electronic+shot noise data as numpy file.
        """
        assert self.config is not None and self.config.bob is not None
        if self.electronic_shot_noise:
            self.electronic_shot_noise.detector = detector
            self.electronic_shot_noise.power = power
            self.electronic_shot_noise.comment = comment
            self.electronic_shot_noise.dump(self.config.bob.electronic_shot_noise.path)
            logger.info(
                "Electronic and shot noise was saved at location %s (detector: %s, power: %s, comment: %s)",
                str(self.config.bob.electronic_shot_noise.path),
                detector,
                str(power),
                comment,
            )

    def get_alice_photon_number(self) -> float:
        """Request variance to Alice.

        Raises:
            Exception: if the received message is not the expected message.

        Returns:
            float: the variance of Alice's symbols.
        """
        logger.info("Request Va to Alice")
        code, data = self.socket.request(QOSSTCodes.PE_NPHOTON_REQUEST)

        if code != QOSSTCodes.PE_NPHOTON_RESPONSE:
            logging.error("Code %i was received", code)
            return 0

        return data["n_photon"]

    def request_parameter_change(self, parameter: str, new_value: Any) -> None:
        """Request a parameter change to the server.

        Args:
            parameter (str): full module name of the parameter.
            new_value (Any): the requested value for the parameter.
        """
        logger.info(
            "Requesting to change parameter %s to new value %s",
            parameter,
            str(new_value),
        )

        code, data = self.socket.request(
            QOSSTCodes.CHANGE_PARAMETER_REQUEST,
            {"parameter": parameter, "value": new_value},
        )

        if code == QOSSTCodes.PARAMETER_CHANGED:
            old_value = data["old_value"]
            new_value = data["new_value"]
            parameter = data["parameter"]

            logger.info(
                "Parameter %s was successfully changed from %s to %s",
                parameter,
                str(old_value),
                str(new_value),
            )
        else:
            logger.warning("The value of the parameter %s was not changed", parameter)

    def _optimal_polarisation_finding(self):
        """
        The goal of this function is to minimize the power on the powermeter, that corresponds
        to the vertical polarisation.

        This is done by minimizing for the three paddles.
        """
        assert self.powermeter
        assert self.polarisation_controller
        logger.info("Starting optimal position finding for polarisaton.")
        for channel in (
            PolarisationControllerChannel.QWP_1,
            PolarisationControllerChannel.HWP,
            PolarisationControllerChannel.QWP_2,
        ):
            positions = []
            powers = []
            position = self.config.bob.polarisation_recovery.start_course
            while position < self.config.bob.polarisation_recovery.end_course:
                self.polarisation_controller.move_to(position, channel)
                positions.append(self.polarisation_controller.get_position(channel))
                powers.append(self.powermeter.read())
                position += self.config.bob.polarisation_recovery.step
                time.sleep(self.config.bob.polarisation_recovery.wait_time)
            optimal_index = np.argmin(powers)
            optimal_position = positions[optimal_index]
            logger.info(
                "Optimal position for channel %s found at position %f with power %f",
                str(channel),
                optimal_position,
                powers[optimal_index],
            )
            self.polarisation_controller.move_to(optimal_position, channel)

        logger.info("Optimal position found for polarisation")
