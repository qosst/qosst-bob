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
Entrypoint for the optimization script
"""
import os
from pathlib import Path
import argparse

from qosst_core.infos import get_script_infos
from qosst_core.logging import create_loggers

from qosst_bob import __version__
from qosst_bob.optimization.optimize import optimize
from qosst_bob.optimization.updaters.xi_versus_va import XiVsVaUpdater
from qosst_bob.optimization.updaters.roll_off import RollOffUpdater
from qosst_bob.optimization.updaters.pilots_amplitude import PilotsAmplitudeUpdater
from qosst_bob.optimization.updaters.conversion_factor import ConversionFactorUpdater
from qosst_bob.optimization.updaters.baud_rate import BaudRateUpdater
from qosst_bob.optimization.updaters.subframe_size import SubframeSizeUpdater
from qosst_bob.optimization.updaters.frequency_cutoff_tone import (
    FrequencyCutoffToneUpdater,
)
from qosst_bob.optimization.updaters.frequency_shift import FrequencyShiftUpdater
from qosst_bob.optimization.updaters.average_tone_size import AverageToneSizeUpdater
from qosst_bob.optimization.updaters.pilot_difference import PilotDifferenceUpdater

DEFAULT_CHANNEL_VOA = 0


# pylint: disable=too-many-statements
def _create_parser() -> argparse.ArgumentParser:
    """
    Create the parser for the optimization module command.

    Subcommands:
        * xi-vs-va
        * roll-off
        * pilots-amplitude
        * conversion-factor
        * baud-rate
        * subframe-size
        * frequency-cutoff-tone
        * frequency-shift
        * pilot-difference-tone

    Returns:
        argparse.ArgumentParser: parser for the optimization module command.
    """
    default_config_location = Path(os.getcwd()) / "config.toml"

    parser = argparse.ArgumentParser(prog="qosst-bob-optimize")

    parser.add_argument("--version", action="version", version=__version__)
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
        "--num-rep",
        type=int,
        help="Number of repetitions of the experiment.",
        default=10,
    )
    parser.add_argument(
        "--voa-channel",
        type=float,
        help="Attenuation in V to apply to the channel VOA",
        default=DEFAULT_CHANNEL_VOA,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Level of verbosity. If none, nothing is printed to the console. -v will print warnings and errors, -vv will add info and -vvv will print all debug logs.",
    )

    subparsers = parser.add_subparsers()
    xivsva_parser = subparsers.add_parser(
        "xi-vs-va", help="Compute the excess noise while varying the variance."
    )
    xivsva_parser.set_defaults(updater=XiVsVaUpdater)
    xivsva_parser.add_argument(
        "begin_variance", type=float, help="Value for the first variance"
    )
    xivsva_parser.add_argument(
        "end_variance", type=float, help="Value for the last variance (excluded)"
    )
    xivsva_parser.add_argument(
        "step_variance", type=float, help="Value for the step of the variance"
    )

    roll_off_parser = subparsers.add_parser(
        "roll-off", help="Compute the excess noise while varying the roll-off."
    )
    roll_off_parser.set_defaults(updater=RollOffUpdater)
    roll_off_parser.add_argument(
        "begin_roll_off", type=float, help="Value for the first roll-off"
    )
    roll_off_parser.add_argument(
        "end_roll_off", type=float, help="Value for the last roll-off (excluded)"
    )
    roll_off_parser.add_argument(
        "step_roll_off", type=float, help="Value for the step of the roll-off"
    )

    pilots_amplitude_parser = subparsers.add_parser(
        "pilots-amplitude",
        help="Compute the excess noise while varying the amplitude of the pilots.",
    )
    pilots_amplitude_parser.set_defaults(updater=PilotsAmplitudeUpdater)
    pilots_amplitude_parser.add_argument(
        "begin_amplitude", type=float, help="Value for the first amplitude"
    )
    pilots_amplitude_parser.add_argument(
        "end_amplitude", type=float, help="Value for the last amplitude (excluded)"
    )
    pilots_amplitude_parser.add_argument(
        "step_amplitude", type=float, help="Value for the step of the amplitude"
    )

    conversion_factor_parser = subparsers.add_parser(
        "conversion-factor",
        help="Compute the excess noise while varying the conversion factor of Alice.",
    )
    conversion_factor_parser.set_defaults(updater=ConversionFactorUpdater)
    conversion_factor_parser.add_argument(
        "initial_value", type=float, help="Initial value of the conversion factor."
    )
    conversion_factor_parser.add_argument(
        "begin_error", type=float, help="Value for the first error (in %)"
    )
    conversion_factor_parser.add_argument(
        "end_error", type=float, help="Value for the last error (in %, excluded)"
    )
    conversion_factor_parser.add_argument(
        "step_error", type=float, help="Value for the step of the error (in %)"
    )

    baud_rate_parser = subparsers.add_parser(
        "baud-rate",
        help="Compute the excess noise while varying the baud rate of Alice. WARNING: Make sure to have enough frequency space before doing so.",
    )
    baud_rate_parser.set_defaults(updater=BaudRateUpdater)
    baud_rate_parser.add_argument(
        "baud_rates", type=int, nargs="+", help="The list of baud rates to try."
    )

    subframe_size_parser = subparsers.add_parser(
        "subframe-size",
        help="Compute the excess noise while varing the size of the subframe of the DSP at Bob side.",
    )
    subframe_size_parser.set_defaults(updater=SubframeSizeUpdater)
    subframe_size_parser.add_argument(
        "sizes", type=int, nargs="+", help="The list of subframe sizes to try."
    )

    frequency_cutoff_tone_parsor = subparsers.add_parser(
        "frequency-cutoff-tone",
        help="Compute the excess noise while varying the cutoff for the filtering of the tone on Bob side.",
    )
    frequency_cutoff_tone_parsor.set_defaults(updater=FrequencyCutoffToneUpdater)
    frequency_cutoff_tone_parsor.add_argument(
        "begin_cutoff", type=float, help="Value for the first cutoff"
    )
    frequency_cutoff_tone_parsor.add_argument(
        "end_cutoff", type=float, help="Value for the last cutoff (excluded)"
    )
    frequency_cutoff_tone_parsor.add_argument(
        "step_cutoff", type=float, help="Value for the step of the cutoff"
    )

    frequency_shit_parser = subparsers.add_parser(
        "frequency-shift",
        help="Compute the excess noise while varying the frequency shift of Alice. WARNING: Make sure to have enough frequency space before doing so.",
    )
    frequency_shit_parser.set_defaults(updater=FrequencyShiftUpdater)
    frequency_shit_parser.add_argument(
        "frequency_shifts",
        type=float,
        nargs="+",
        help="The list of frequency shifts to try.",
    )

    average_tone_size_parser = subparsers.add_parser(
        "average-tone-size",
        help="Compute the excess noise while varying the size for the averaging of the pilot at phase correction.",
    )
    average_tone_size_parser.set_defaults(updater=AverageToneSizeUpdater)
    average_tone_size_parser.add_argument(
        "sizes",
        type=int,
        nargs="+",
        help="The list of sizes to try.",
    )

    pilot_difference_parser = subparsers.add_parser(
        "pilot-difference-tone",
        help="Compute the excess noise while varying the difference between the two pilots. The first pilot will be left unchanged.",
    )
    pilot_difference_parser.set_defaults(updater=PilotDifferenceUpdater)
    pilot_difference_parser.add_argument(
        "begin_difference", type=float, help="Value for the first difference"
    )
    pilot_difference_parser.add_argument(
        "end_difference", type=float, help="Value for the last difference (excluded)"
    )
    pilot_difference_parser.add_argument(
        "step_difference", type=float, help="Value for the step of the difference"
    )
    return parser


def main():
    """
    Main function of the script. Entrypoint of the script.
    """
    parser = _create_parser()

    args = parser.parse_args()

    print(get_script_infos())

    # Configure loggers

    create_loggers(args.verbose, args.file)

    optimize(args, args.file)


if __name__ == "__main__":
    main()
