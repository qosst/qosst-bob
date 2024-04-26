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
Commands for Bob tools submodule.
"""
import logging
import argparse

from qosst_core.logging import create_loggers

from qosst_bob import __version__
from qosst_bob.tools.calibrate_eta_voltage import calibration_eta_voltage
from qosst_bob.tools.calibrate_eta_current import calibration_eta_current

logger = logging.getLogger(__name__)


def _create_parser() -> argparse.ArgumentParser:
    """
    Create the parser for bob tools.

    Commands:
        eta_voltage
        gain

    Returns:
        argparse.ArgumentParser: the main parser.
    """
    parser = argparse.ArgumentParser(prog="qosst-bob-tools")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Level of verbosity. If none, nothing is printed to the console. -v will print warnings and errors, -vv will add info and -vvv will print all debug logs.",
    )

    subparsers = parser.add_subparsers()

    eta_voltage_parser = subparsers.add_parser(
        "eta-voltage", help="Compute eta using voltage"
    )
    eta_voltage_parser.add_argument(
        "gain", type=float, help="Gain of the TIA of the monitoring outputs."
    )
    eta_voltage_parser.add_argument(
        "--no-save",
        dest="save",
        action="store_false",
        help="Don't save the results.",
    )
    eta_voltage_parser.set_defaults(func=calibration_eta_voltage)

    eta_current_parser = subparsers.add_parser(
        "eta-current", help="Compute eta using current."
    )
    eta_current_parser.add_argument(
        "--no-save", dest="save", action="store_false", help="Don't save the results."
    )
    eta_current_parser.set_defaults(func=calibration_eta_current)
    return parser


def main():
    """
    Main entrypoint of the command.
    """
    parser = _create_parser()
    args = parser.parse_args()

    create_loggers(args.verbose, None)

    if hasattr(args, "func"):
        args.func(args)
    else:
        print("No command specified. Run with -h|--help to see the possible commands.")


if __name__ == "__main__":
    main()
