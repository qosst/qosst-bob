# qosst-bob - Bob module of the Quantum Open Software for Secure Transmissions.
# Copyright (C) 2021-2026 Yoann Piétri
# Copyright (C) 2021-2024 Valentina Marulanda Acosta
# Copyright (C) 2021-2024 Matteo Schiavon
# Copyright (C) 2021-2026 Thomas Liege


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
Main module for the DSP algorithm.

Warning: the DSP _dsp_bob_shared_clock_shared_lo, _dsp_bob_shared_clock_unshared_lo and _dsp_bob_unshared_clock_shared_lo
are adapted versions of old DSP and might not work. They are untested.
"""

from .general import GeneralDSP
from .general_direct_tracking import GeneralDirectTrackingDSP
from .shared_clock_shared_lo import SharedClockSharedLODSP
from .shared_clock_unshared_lo import SharedClockUnsharedLODSP
from .unshared_clock_shared_lo import UnsharedClockSharedLODSP
