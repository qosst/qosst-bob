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
Experiment to measure the transmittance while varying attenuation on the channel.
"""
import os
import sys
from pathlib import Path
import logging
import argparse
import datetime

import numpy as np
import matplotlib.pyplot as plt

from qosst_core.logging import create_loggers
from qosst_core.infos import get_script_infos
from qosst_core.participant import Participant
from qosst_core.configuration.exceptions import InvalidConfiguration

from qosst_bob import __version__
from qosst_bob.bob import Bob
from qosst_bob.data import TransmittanceResults

logger = logging.getLogger(__name__)

MAX_ERRORS = 5


def _create_parser() -> argparse.ArgumentParser:
    """
    Create parser for the transmittance experiment.

    Returns:
        argparse.ArgumentParser: parser for the transmittance experiment.
    """
    default_config_location = Path(os.getcwd()) / "config.toml"

    parser = argparse.ArgumentParser(prog="qosst-transmittance")

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
        "-n",
        "--num_rep",
        type=int,
        default=1,
        help="Number of repetitions of the experiment. Default : 1",
    )
    parser.add_argument("start_voa", type=float, help="Start value of the VOA in V.")
    parser.add_argument(
        "end_voa", type=float, help="End value of the VOA in V (not included)."
    )
    parser.add_argument("step_voa", type=float, help="Step value of the VOA in V.")
    return parser


# pylint: disable=too-many-locals, too-many-statements, too-many-branches
def main():
    """
    Main entrypoint of the script.
    """
    print(get_script_infos())

    parser = _create_parser()
    args = parser.parse_args()

    create_loggers(args.verbose, args.file)

    attenuations = np.arange(args.start_voa, args.end_voa, args.step_voa)

    total_rounds = args.num_rep * len(attenuations)

    transmittances = np.zeros(shape=(len(attenuations), args.num_rep), dtype=float)
    excess_noises_bob = np.zeros(shape=(len(attenuations), args.num_rep), dtype=float)
    electronic_noise_values = np.zeros(
        shape=(len(attenuations), args.num_rep), dtype=float
    )
    photon_number_values = np.zeros(
        shape=(len(attenuations), args.num_rep), dtype=float
    )
    shot_noise_values = np.zeros(shape=(len(attenuations), args.num_rep), dtype=float)
    datetimes = np.zeros(
        shape=(len(attenuations), args.num_rep), dtype=datetime.datetime
    )

    bob = Bob(args.file)
    bob.open_hardware()
    logger.info("Connecting Bob.")
    bob.connect()

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

    if voa_channel is None:
        bob.close()
        raise InvalidConfiguration(
            "Please configuration the channel.voa section to use a VOA as channel and have the applier begin Bob for this experiment."
        )

    logger.info("Loading electronic noise data.")
    bob.load_electronic_noise_data()

    logger.info("Identification.")
    bob.identification()

    logger.info("Initialization")
    bob.initialization()

    if bob.notifier:
        bob.notifier.send_notification("Experiment on transmittance started.")

    for i, att in enumerate(attenuations):
        logger.info("Starting attenuation %i/%i", i + 1, len(attenuations))
        logger.info("Setting VOA attenuation voltage to %f", att)
        voa_channel.set_value(att)

        j = 0
        error = 0
        while j < args.num_rep:
            try:
                logger.info(
                    "Starting round %i/%i", i * args.num_rep + j + 1, total_rounds
                )

                if not bob.config.bob.switch.switching_time:
                    logger.info("Manual shot noise calibration")
                    bob.get_electronic_shot_noise_data()

                current_datetime = datetime.datetime.now()

                logger.info("Starting QIE")
                bob.quantum_information_exchange()

                logger.info("DSP")
                bob.dsp()

                alice_photon_number = bob.get_alice_photon_number()

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

                logger.info("T = %f, ξ = %f", transmittance, excess_noise_bob)

                transmittances[i][j] = transmittance
                excess_noises_bob[i][j] = excess_noise_bob
                electronic_noise_values[i][j] = electronic_noise
                photon_number_values[i][j] = alice_photon_number
                shot_noise_values[i][j] = np.var(bob.electronic_shot_symbols) - np.var(
                    bob.electronic_symbols
                )
                datetimes[i][j] = current_datetime
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
        bob.notifier.send_notification("Experiment on transmittance ended.")
    if args.save:
        to_save = TransmittanceResults(
            configuration=bob.config,
            num_rep=args.num_rep,
            excess_noise_bob=excess_noises_bob,
            transmittance=transmittances,
            photon_number=photon_number_values,
            datetimes=datetimes,
            electronic_noise=electronic_noise_values,
            shot_noise=shot_noise_values,
            source_script="qosst-bob-transmittance",
            command_line=" ".join(sys.argv),
            attenuation_values=attenuations,
        )
        filename = (
            f"results_transmittance_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.npy"
        )
        to_save.save(filename)
        logger.info("Results have saved to %s", filename)

    if voa_channel is not None:
        voa_channel.close()
    bob.close()

    if args.plot:
        plt.figure(100)
        plt.plot(attenuations, np.mean(transmittances, axis=-1), "--o")
        plt.xlabel("Attenuation voltage [V]")
        plt.ylabel("Transmittance")
        plt.title("Transmittance vs. attenuation voltage")
        plt.grid()
        plt.figure(101)
        plt.plot(
            attenuations,
            np.mean(transmittances, axis=-1)
            / np.mean(transmittances, axis=-1)[0]
            * 100,
            "--o",
        )
        plt.xlabel("Attenuation voltage [V]")
        plt.ylabel("Transmittance (ratio) [%]")
        plt.title("Transmittance (ratio) vs. attenuation voltage")
        plt.grid()
        plt.figure(102)
        plt.plot(attenuations, np.mean(excess_noises_bob, axis=-1), "--o")
        plt.xlabel("Attenuation voltage [V]")
        plt.ylabel("Excess noise at Bob's side [SNU]")
        plt.title("Excess noise vs. attenuation voltage")
        plt.grid()
        plt.show()


if __name__ == "__main__":
    main()
