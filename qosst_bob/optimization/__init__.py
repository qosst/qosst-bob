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
Optimization submodules of Bob.

Also contains the Updater abstract class.
"""
import abc
import argparse
from typing import Dict

from qosst_core.configuration import Configuration

from qosst_bob.bob import Bob


class Updater(abc.ABC):
    """
    An abstract class for updaters for optimization.

    Args:
        args (argparse.Namespace): arguments passed to the command line.
        bob (Bob): Bob object to request changes to Alice.
        config (Configuration): configuration object to change the parameters on Bob side.
    """

    args: argparse.Namespace  #: The arguments of the command line.
    bob: Bob  #: The class of Bob, to request parameter changes to Alice
    config: Configuration  #: The configuration, to change the parameter on Bob side.
    round: int  #: A counter to keep memory of the turn

    def __init__(
        self, args: argparse.Namespace, bob: Bob, config: Configuration
    ) -> None:
        """
        Args:
            args (argparse.Namespace): arguments passed to the command line.
            bob (Bob): Bob object to request changes to Alice.
            config (Configuration): configuration object to change the parameters on Bob side.
        """
        self.args = args
        self.bob = bob
        self.config = config
        self.round = 0

        self._init_parameters()

    @abc.abstractmethod
    def _init_parameters(self):
        """
        This function is called at the end of init and should initialize the parameters arrays.
        """

    @abc.abstractmethod
    def number_of_rounds(self) -> int:
        """
        Return a number of rounds the script should do.

        Returns:
            int: number of rounds of the experiment.
        """

    @abc.abstractmethod
    def update(self) -> Dict:
        """
        This function should
            * update the parameter(s) on Bob side
            * request the parameter(s) to be changed to Alice
            * return a dict with the name of parameters as key and the new value as value

        Returns:
            Dict: dict with the name of parameters as key and the new value as value
        """

    @abc.abstractmethod
    def name(self) -> str:
        """
        Name of the updater. To be used in the name of the saved file.

        Returns:
            str: name of the updater.
        """
