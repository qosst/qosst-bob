# qosst-bob - Bob module of the Quantum Open Software for Secure Transmissions.
# Copyright (C) 2021-2025 Yoann Piétri

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
Script to continuously run.
"""

import argparse
import logging
from pathlib import Path
import os

from qosst_core.logging import create_loggers
from qosst_core.infos import get_script_infos
from qosst_core.participant import Participant

from qosst_bob import __version__
from qosst_bob.bob import Bob

logger = logging.getLogger(__name__)


def _create_parser() -> argparse.ArgumentParser:
    """
    Create the parser for qosst-bob-continuous.

    Returns:
        argparse.ArgumentParser: parser for the qosst-bob-continuous.
    """
    default_config_location = Path(os.getcwd()) / "config.toml"
    parser = argparse.ArgumentParser(prog="qosst-bob-continuous")

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

    # Init hardware
    bob.open_hardware()

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

    # Load electronic noise
    logger.info("Loading electronic noise")
    bob.load_electronic_noise_data()

    # Connect to Alice
    bob.connect()

    # Identification
    bob.identification()

    try:
        while True:
            logger.info("Starting new frame")
            # Initialization
            bob.initialization()

            if not bob.config.bob.switch.switching_time:
                logger.info("Manual shot noise acquisition")
                bob.get_electronic_shot_noise_data()

            # Quantum Information Exchange
            bob.quantum_information_exchange()

            # DSP
            bob.dsp()

            # Parameter estimation
            bob.parameters_estimation()

            # Error correction
            bob.error_correction()

            # Privacy amplification
            bob.privacy_amplification()

            # End frame, push key
            bob.end_frame()
    except KeyboardInterrupt:
        print("Interruption received. Closing Bob.")
    finally:
        bob.close()


if __name__ == "__main__":
    main()
