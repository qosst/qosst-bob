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
Script to optimize the excess noise over a DSP parameter.
"""
import sys
import argparse
import logging
import datetime
from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt

from qosst_bob import __version__
from qosst_bob.bob import Bob
from qosst_bob.data import OptimizationResults
from qosst_bob.optimization import Updater

logger = logging.getLogger(__name__)

MAX_ERRORS = 5


# pylint: disable=too-many-locals, too-many-statements too-many-branches
def optimize(args: argparse.Namespace, config_path: str):
    """
    Launch Bob and start optimizing given a specific updater.

    Save the results and plot depending on the arguments.

    Args:
        args (argparse.Namespace): arguments int the command line.
        config (Configuration): configuration object.
    """
    logger.info(
        "Starting optimization script.",
    )

    bob = Bob(config_path)

    updater: Updater = args.updater(args, bob, bob.config)

    number_of_rounds = updater.number_of_rounds()

    # Initialize the result arrays

    all_excess_noise_values = np.zeros(
        shape=(number_of_rounds, args.num_rep), dtype=float
    )

    all_transmittance_values = np.zeros(
        shape=(number_of_rounds, args.num_rep), dtype=float
    )

    all_photon_number_values = np.zeros(
        shape=(number_of_rounds, args.num_rep), dtype=float
    )

    all_electronic_noise_values = np.zeros(
        shape=(number_of_rounds, args.num_rep), dtype=float
    )

    all_shot_noise_values = np.zeros(
        shape=(number_of_rounds, args.num_rep), dtype=float
    )
    datetimes = np.zeros(
        shape=(number_of_rounds, args.num_rep), dtype=datetime.datetime
    )

    parameters: Dict[Any, Any] = {}

    logger.info("Load electronic noise")
    bob.load_electronic_noise_data()

    # Init hardware
    bob.open_hardware()

    # Connect to Alice
    bob.connect()

    # Identification
    bob.identification()

    # Initialization
    bob.initialization()

    assert bob.config is not None and bob.config.bob is not None

    if bob.notifier:
        bob.notifier.send_notification("Experiment on optimization started.")

    for i in range(number_of_rounds):
        logger.info("Starting round %i/%i", i + 1, number_of_rounds)

        # Call the updater to update the value
        logger.info("Calling the updater")
        values = updater.update()

        for key, value in values.items():
            if key in parameters:
                parameters[key].append(value)
            else:
                parameters[key] = [value]

        excess_noise_current = np.zeros(shape=args.num_rep)
        transmittance_current = np.zeros(shape=args.num_rep)
        photon_number_current = np.zeros(shape=args.num_rep)
        electronic_noise_current = np.zeros(shape=args.num_rep)
        shot_noise_current = np.zeros(shape=args.num_rep)
        datetimes_current = np.zeros(shape=args.num_rep, dtype=datetime.datetime)

        error = 0
        j = 0
        while j < args.num_rep:
            try:
                logger.info("Starting repetition %i/%i", j + 1, args.num_rep)

                if not bob.config.bob.switch.switching_time:
                    logger.info("Manual shot noise acquisition")
                    bob.get_electronic_shot_noise_data()

                current_datetime = datetime.datetime.now()

                logger.info("Quantum data acquisition")
                bob.quantum_information_exchange()

                logger.info("Do DSP")
                bob.dsp()

                alice_photon_number = bob.get_alice_photon_number()
                assert bob.alice_symbols is not None
                assert bob.quantum_symbols is not None
                assert bob.electronic_symbols is not None
                assert bob.electronic_shot_symbols is not None
                (
                    transmittance,
                    excess_noise_bob,
                    electronic_noise,
                ) = bob.config.bob.parameters_estimation.estimator.estimate(
                    bob.alice_symbols,
                    bob.quantum_symbols[bob.indices],
                    alice_photon_number,
                    bob.electronic_symbols,
                    bob.electronic_shot_symbols,
                )

                excess_noise_current[j] = excess_noise_bob
                transmittance_current[j] = transmittance
                photon_number_current[j] = alice_photon_number
                electronic_noise_current[j] = electronic_noise
                shot_noise_current[j] = np.var(bob.electronic_shot_symbols) - np.var(
                    bob.electronic_symbols
                )
                datetimes_current[j] = current_datetime
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
                    return

        all_excess_noise_values[i] = excess_noise_current
        all_transmittance_values[i] = transmittance_current
        all_photon_number_values[i] = photon_number_current
        all_electronic_noise_values[i] = electronic_noise_current
        all_shot_noise_values[i] = shot_noise_current
        datetimes[i] = datetimes_current

    if bob.notifier:
        bob.notifier.send_notification("Experiment on optimization started.")

    if args.save:
        to_save = OptimizationResults(
            configuration=bob.config,
            num_rep=args.num_rep,
            excess_noise_bob=all_excess_noise_values,
            transmittance=all_transmittance_values,
            photon_number=all_photon_number_values,
            datetimes=datetimes,
            electronic_noise=all_electronic_noise_values,
            shot_noise=all_shot_noise_values,
            source_script="qosst-bob-optimize",
            command_line=" ".join(sys.argv),
            parameters=parameters,
        )
        filename = f"results_parameter_optimization_{updater.name()}_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.qosst"
        to_save.save(filename)
        logger.info("Results have saved to %s", filename)

    logger.info("Closing Bob.")
    bob.close()

    if args.plot:
        excess_noise_values = np.mean(all_excess_noise_values, axis=-1)
        transmittance_values = np.mean(all_transmittance_values, axis=-1)
        photon_number_values = np.mean(all_photon_number_values, axis=-1)
        plt.figure()
        plt.title("$\\xi_B$ vs round")
        plt.plot(range(len(excess_noise_values)), excess_noise_values, "--o")
        plt.grid()
        plt.xlabel("Round")
        plt.ylabel("$\\xi_B$ [SNU]")

        plt.figure()
        plt.title("$\\eta T$ vs round")
        plt.plot(range(len(transmittance_values)), transmittance_values, "--o")
        plt.grid()
        plt.xlabel("Round")
        plt.ylabel("$\\eta T$ [SNU]")

        plt.figure()
        plt.title("$\\xi$ vs round")
        plt.plot(
            range(len(excess_noise_values)),
            excess_noise_values / transmittance_values,
            "--o",
        )
        plt.grid()
        plt.xlabel("Round")
        plt.ylabel("$\\xi$ [SNU]")

        plt.figure()
        plt.title("$\\langle n \\rangle$ vs round")
        plt.plot(
            range(len(photon_number_values)),
            photon_number_values,
            "--o",
        )
        plt.grid()
        plt.xlabel("Round")
        plt.ylabel("$\\langle n \\rangle$ [Photons/symbol]")

        plt.show()
