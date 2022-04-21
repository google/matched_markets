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
"""TBR Matched Markets: Class representing a TBR Design Score.
"""
import collections
from typing import Union
import dataclasses

from matched_markets.methodology import tbrmmdiagnostics

Number = Union[int, float]
TBRMMDiagnostics = tbrmmdiagnostics.TBRMMDiagnostics
Scoring = collections.namedtuple('Scoring', [
    'corr_test', 'aa_test', 'bb_test', 'dw_test', 'corr', 'inv_required_impact'
])


@dataclasses.dataclass
class TBRMMScore:
  """Class representing a TBR design score.

  TBRMMScore defines a score which can be used to sort TBRMMDesign. The scoring
  function is a NamedTuple with properties:

  - corr_test, aa_test, bb_test, dw_test which are True if the corresponding
    diagnostics tests pass. These are prioritize when sorting TBRMMDesign, i.e.
    a design which does not pass one of these tests will be worse than a design
    which pass all of them.

  - corr is the correlation between the current design treatment and
    control group.

  - inv_required_impact is the inverse of the minimum lift which we need to
    generate for a significant result. So, 1/required_impact can be thought of
    as the inverse of the minimum detectable iROAS for a budget of 1$.


  The idea of using the logical result of the tests is due to the fact that it
  could happen that none of the designs searched pass at least 1 of these tests
  (it does not have to be the same tests for all designs). To be constructive,
  we still want to output the best design that can be found, but we will
  highlight/flag the risk of running an experiment which fails one of the tests.
  Knowing which test failed can be useful as well to understand what is "wrong".

  In future, the binary outcome for the tests can be replaced by a p-value or
  similar continuous metrics. For example. the aa test result already provide
  the probability of a false positive result (if the result is positive).

  Attributes:
    diag: a TBRMMDiagnostics object.
  """

  diag: TBRMMDiagnostics  # design diagnostics
  _score = None  # score of the corresponding design

  def __post_init__(self):
    if self.diag.x is None:
      raise ValueError('No Control time series was specified')
    if self.diag.corr is None:
      corr = self.diag.corr
    if self.diag.required_impact is None:
      impact = self.diag.required_impact

  def __lt__(self, other: 'TBRMMScore'):
    return self.score < other.score

  @property
  def score(self):
    """Score of the design.

    Returns:
      Score of the design defined by the treatment and control groups as in diag
      and according to the design parameters.
    """
    if self._score is None:
      self._score = Scoring(
          int(self.diag.corr_test), int(self.diag.aatest.test_ok),
          int(self.diag.bbtest.test_ok), int(self.diag.dwtest.test_ok),
          round(self.diag.corr, 2), 1 / self.diag.required_impact)

    return self._score

  @score.setter
  def score(self, value: Scoring):
    self._score = value
