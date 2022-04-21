# Copyright 2020 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""TBR Matched Markets: Class representing a TBR Design Candidate.
"""
from typing import Optional, Set, Text, Union

import dataclasses

from matched_markets.methodology import tbrmmdiagnostics
from matched_markets.methodology import tbrmmscore

GeoID = Text
GeoIndex = int
GeoSet = Set[Union[GeoID, GeoIndex]]
TBRMMDiagnostics = tbrmmdiagnostics.TBRMMDiagnostics
TBRMMScore = tbrmmscore.TBRMMScore


@dataclasses.dataclass
class TBRMMDesign:
  """Class representing a TBR design candidate.

  TBRMMDesign objects can be sorted by their score.

  Attributes:
    score: A score indicating the goodness of the design, used for sorting. The
      higher the better. Can be numeric, tuple or other object that defines
      comparable.
    diag: The associated TBRMMDiagnostics object.
    treatment_geos: List of treatment geos (IDs or integers).
    control_geos: List of control geos (IDs or integers).
  """

  score: TBRMMScore
  treatment_geos: GeoSet
  control_geos: GeoSet
  diag: Optional[TBRMMDiagnostics] = None

  def __post_init__(self):
    if not self.treatment_geos:
      raise ValueError('No Treatment geos')
    if not self.control_geos:
      raise ValueError('No Control geos')
    overlapping_geos = self.treatment_geos & self.control_geos
    if overlapping_geos:
      overlapping_geos = sorted(list(overlapping_geos))
      raise ValueError(
          'Control and Treatment geos overlap: \'' +
          '\', \''.join(overlapping_geos) + '\'')

  def __lt__(self, other: 'TBRMMDesign'):
    return self.score < other.score
