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
Code to use the dsp offline.
"""

import os
import logging
import argparse
import datetime
from pathlib import Path
from typing import Tuple
import warnings

import numpy as np

from qosst_core.infos import get_script_infos
from qosst_core.logging import create_loggers
from qosst_core.configuration.config import Configuration

from qosst_bob import __version__
from qosst_bob.dsp.dsp import dsp_bob, special_dsp, find_global_angle

logger = logging.getLogger(__name__)


# pylint: disable=too-many-locals
def offline_dsp(
    config: Configuration,
    data: np.ndarray,
    electronic_noise_data: np.ndarray,
    electronic_shot_noise_data: np.ndarray,
    all_alice_symbols: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform offline DSP given the configuration object,
    the data, the electronic noise data, the electronic
    and shot noise date and (all) Alice's symbols

    Args:
        config (Configuration): configuration object (for Bob).
        data (np.ndarray): acquisition of the signal.
        electronic_noise_data (np.ndarray): data of the electronic noise.
        electronic_shot_noise_data (np.ndarray): data of the electronic and shot noise.
        all_alice_symbols (np.ndarray): all Alice's symbols.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: recovered symbols, indices used for global angle correction, equivalent electronic symbols, equivalent electronic and shot symbols.
    """
    warnings.warn("Offline DSP is experimental.")
    logger.info("Starting offline DSP")

    if config.bob.switch.switching_time:
        end_electronic_shot_noise = int(
            config.bob.switch.switching_time * config.bob.adc.rate
        )
        data = data[end_electronic_shot_noise:]

    quantum_symbols, params, dsp_debug = dsp_bob(data, config)

    # Correct global phase of each frame of quantum symbols
    all_indices = []
    last_indice = -1
    for i, frame in enumerate(quantum_symbols):
        logger.info(
            "Finding global angle at frame %i/%i",
            i + 1,
            len(quantum_symbols),
        )

        # Generate indices
        indices = np.arange(last_indice + 1, last_indice + 1 + len(frame))
        np.random.shuffle(indices)
        indices = indices[: int(len(frame) * config.bob.parameters_estimation.ratio)]

        alice_symbols = all_alice_symbols[indices]

        # Find global angle
        angle, cov = find_global_angle(
            frame[indices - (last_indice + 1)], alice_symbols
        )

        logger.info("Global angle found : %.2f with cov %.2f", angle, cov)

        quantum_symbols[i] = np.exp(1j * angle) * frame

        # Add indices and symbols, update last_indice
        all_indices.append(indices)
        last_indice = last_indice + len(frame)

    quantum_symbols = np.concatenate(quantum_symbols)
    all_indices = np.concatenate(all_indices)

    begin_data = dsp_debug.begin_data

    logger.info(
        "Time between end of shot noise and signal : %f ms",
        begin_data / config.bob.adc.rate * 1e3,
    )

    logger.info("Applying DSP on elec and elec+shot noise data")

    params.elec_noise_estimation_ratio = config.bob.dsp.elec_noise_estimation_ratio
    params.elec_shot_noise_estimation_ratio = config.bob.dsp.elec_shot_noise_estimation_ratio

    electronic_symbols, electronic_shot_symbols = special_dsp(
        electronic_noise_data, electronic_shot_noise_data, params
    )

    return quantum_symbols, all_indices, electronic_symbols, electronic_shot_symbols


def _create_parser() -> argparse.ArgumentParser:
    """
    Create the parser for qosst-bob-offline-dsp.

    Returns:
        argparse.ArgumentParser: parser for the qosst-bob-offline-dsp.
    """
    default_config_location = Path(os.getcwd()) / "config.toml"
    parser = argparse.ArgumentParser(prog="qosst-bob-offline-dsp")

    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Level of verbosity. If none, nothing is printed to the console. -v will print warnings and errors, -vv will add info and -vvv will print all debug logs.",
    )
    parser.add_argument(
        "-f",
        "--file",
        default=default_config_location,
        help=f"Path of the configuration file. Default : {default_config_location}.",
    )
    parser.add_argument(
        "--no-save", dest="save", action="store_false", help="Don't save the data."
    )
    parser.add_argument("data", type=str, help="Path to the data.")
    parser.add_argument(
        "elec_data", type=str, help="Path to the electronic noise data."
    )
    parser.add_argument(
        "elec_shot_data", type=str, help="Path to the electronic and shot noise data."
    )
    parser.add_argument("alice_symbols", type=str, help="Path to Alice's symbols.")
    parser.add_argument(
        "alice_photon_number",
        type=float,
        help="Mean number of photons per symbol at Alice side.",
    )
    return parser


def main():
    """
    Main script for the offline DSP.
    """
    print(get_script_infos())

    parser = _create_parser()
    args = parser.parse_args()

    create_loggers(args.verbose, args.file)

    # Load configuration
    configuration = Configuration(args.file)

    # Load data and symbols
    electronic_noise_data = np.load(args.elec_data, allow_pickle=True)
    if args.elec_data.endswith('.qosst'):
        electronic_noise_data = electronic_noise_data.data[0]
    electronic_shot_noise_data = np.load(args.elec_shot_data, allow_pickle=True)
    if args.elec_shot_data.endswith('.qosst'):
        electronic_shot_noise_data = electronic_shot_noise_data.data[0]
    data = np.load(args.data)
    if len(data.shape) == 2:
        data = data[0]
    alice_symbols = np.load(args.alice_symbols)
    # Offline DSP
    quantum_symbols, indices, electronic_symbols, electronic_shot_symbols = offline_dsp(
        configuration,
        data,
        electronic_noise_data,
        electronic_shot_noise_data,
        alice_symbols,
    )

    # Parameters estimation
    transmittance, excess_noise, electronic_noise = (
        configuration.bob.parameters_estimation.estimator.estimate(
            alice_symbols[indices],
            quantum_symbols[indices],
            args.alice_photon_number,
            electronic_symbols,
            electronic_shot_symbols,
        )
    )

    skr = (
        configuration.frame.quantum.symbol_rate
        * configuration.bob.parameters_estimation.skr_calculator.skr(
            Va=2 * args.alice_photon_number,
            T=transmittance / configuration.bob.eta,
            xi=excess_noise / transmittance,
            eta=configuration.bob.eta,
            Vel=electronic_noise,
            beta=0.95,
        )
    )

    logger.info("Transmittance: %f", transmittance)
    logger.info("Excess noise (Bob): %f", excess_noise)
    logger.info("Electronic noise: %f", electronic_noise)

    logger.info("Secret key rate: %f MBit/s", skr * 1e-6)

    # Save results

    if args.save:
        np.save(
            f"offline_quantum_symbols_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.npy",
            quantum_symbols,
        )
        np.save(
            f"offline_elec_symbols_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.npy",
            electronic_symbols,
        )
        np.save(
            f"offline_elec_shot_symbols_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.npy",
            electronic_shot_symbols,
        )


if __name__ == "__main__":
    main()
