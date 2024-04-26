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
Script to measure excess noise and repeating exchanges of frames.
"""

import argparse
import logging
from pathlib import Path
import os
import sys
import datetime

import numpy as np
import matplotlib.pyplot as plt

from qosst_core.logging import create_loggers
from qosst_core.infos import get_script_infos
from qosst_core.participant import Participant

from qosst_bob import __version__
from qosst_bob.bob import Bob
from qosst_bob.data import ExcessNoiseResults

logger = logging.getLogger(__name__)

MAX_ERRORS = 5


def _create_parser() -> argparse.ArgumentParser:
    """
    Create the parser for qosst-bob-excess-noise.

    Returns:
        argparse.ArgumentParser: parser for the qosst-bob-excess-noise.
    """
    default_config_location = Path(os.getcwd()) / "config.toml"
    parser = argparse.ArgumentParser(prog="qosst-bob-excess-noise")

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
    parser.add_argument(
        "--plot", dest="plot", action="store_true", help="Plot the data."
    )
    parser.add_argument(
        "num_rep", type=int, help="Number of repetitions of the experiment."
    )
    return parser


# pylint: disable=too-many-locals,disable=too-many-statements
def main():
    """
    Entry point of the excess noise script.
    """
    print(get_script_infos())

    parser = _create_parser()
    args = parser.parse_args()

    create_loggers(args.verbose, args.file)

    bob = Bob(args.file)

    all_excess_noise_values = np.zeros(shape=args.num_rep, dtype=float)
    all_transmittance_values = np.zeros(shape=args.num_rep, dtype=float)
    all_photon_number_values = np.zeros(shape=args.num_rep, dtype=float)
    all_electronic_noise_values = np.zeros(shape=args.num_rep, dtype=float)
    all_shot_noise_values = np.zeros(shape=args.num_rep, dtype=float)
    datetimes = np.zeros(shape=args.num_rep, dtype=datetime.datetime)

    # Init hardware
    bob.open_hardware()

    # Load electronic noise
    logger.info("Loading electronic noise")
    bob.load_electronic_noise_data()

    # Connect to Alice
    bob.connect()

    # Identification
    bob.identification()

    # Initialization
    bob.initialization()

    if bob.notifier:
        bob.notifier.send_notification("Experiment on excess noise started.")

    voa_channel = None
    if (
        bob.config.channel.voa is not None
        and bob.config.channel.voa.use
        and bob.config.channel.voa.applier == Participant.BOB
    ):
        voa_channel = bob.config.channel.voa.device(
            bob.config.channel.voa.location, **bob.config.channel.voa.extra_args
        )
        voa_channel.open()
        voa_channel.set_value(bob.config.channel.voa.value)

    j = 0
    error = 0
    while j < args.num_rep:
        try:
            logger.info("Starting round %i/%i", j + 1, args.num_rep)

            if not bob.config.bob.switch.switching_time:
                logger.info("Manual shot noise acquisition")
                bob.get_electronic_shot_noise_data()

            current_datetime = datetime.datetime.now()

            logger.info("Quantum data acquisition")
            bob.quantum_information_exchange()

            # DSP
            bob.dsp()

            quantum_symbols = bob.quantum_symbols
            indices = bob.indices
            symbols_alice = bob.alice_symbols

            photon_number = bob.get_alice_photon_number()

            (
                transmittance,
                excess_noise,
                electronic_noise,
            ) = bob.config.bob.parameters_estimation.estimator.estimate(
                symbols_alice,
                quantum_symbols[indices],
                photon_number,
                bob.electronic_symbols,
                bob.electronic_shot_symbols,
            )
            all_excess_noise_values[j] = excess_noise
            all_transmittance_values[j] = transmittance
            all_photon_number_values[j] = photon_number
            all_electronic_noise_values[j] = electronic_noise
            all_shot_noise_values[j] = np.var(bob.electronic_shot_symbols) - np.var(
                bob.electronic_symbols
            )
            datetimes[j] = current_datetime
            j += 1
            error = 0
        except ValueError as exc:
            if error < MAX_ERRORS:
                error += 1
                logger.error(
                    "There was an exception in this round (see next log for the exception). This round will be done again (%i attempts remaning)",
                    MAX_ERRORS - error,
                )
                logger.error(exc)
            else:
                logger.critical(
                    "The same round failed for %i times in a row. The script is now aborting",
                    MAX_ERRORS,
                )
                bob.close()
                if voa_channel is not None:
                    voa_channel.close()
                return

    if bob.notifier:
        bob.notifier.send_notification("Experiment on excess noise ended.")

    if args.save:
        to_save = ExcessNoiseResults(
            configuration=bob.config,
            num_rep=args.num_rep,
            excess_noise_bob=all_excess_noise_values,
            transmittance=all_transmittance_values,
            photon_number=all_photon_number_values,
            datetimes=datetimes,
            electronic_noise=all_electronic_noise_values,
            shot_noise=all_shot_noise_values,
            source_script="qosst-bob-excess-noise",
            command_line=" ".join(sys.argv),
        )
        filename = (
            f"results_excessnoise_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.npy"
        )
        to_save.save(filename)
        logger.info("Results have saved to %s", filename)

    if voa_channel is not None:
        voa_channel.close()
    bob.close()

    if args.plot:
        plt.figure()
        plt.plot(all_transmittance_values, "o")
        plt.xlabel("Round")
        plt.ylabel("Transmittance")
        plt.grid()

        plt.figure()
        plt.plot(all_excess_noise_values, "o")
        plt.xlabel("Round")
        plt.ylabel("$\\xi_B$")
        plt.grid()
        plt.show()


if __name__ == "__main__":
    main()
