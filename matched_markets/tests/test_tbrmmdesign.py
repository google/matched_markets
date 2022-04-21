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
"""Test TBR Matched Markets.
"""
from matched_markets.methodology import tbrmmdesign
from matched_markets.methodology import tbrmmdesignparameters
from matched_markets.methodology import tbrmmdiagnostics
from matched_markets.methodology import tbrmmscore
import unittest

TBRMMDiagnostics = tbrmmdiagnostics.TBRMMDiagnostics
TBRMMDesignParameters = tbrmmdesignparameters.TBRMMDesignParameters
TBRMMDesign = tbrmmdesign.TBRMMDesign
TBRMMScore = tbrmmscore.TBRMMScore


class TBRMMDesignTest(unittest.TestCase):
  """Test class TBRMMDesign."""

  def setUp(self):
    super().setUp()
    par = TBRMMDesignParameters(n_test=14, iroas=3.0)
    y = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    self.diag = TBRMMDiagnostics(y, par)
    self.geos1 = {'G1', 'G2', 'G3'}
    self.geos2 = {'G4', 'G5'}
    self.geos3 = {'G6'}

  def testScoreAndGeosAreMandatory(self):
    """The score attribute is mandatory."""
    with self.assertRaisesRegex(
        TypeError,
        r'missing 3 required positional arguments: \'score\', '
        r'\'treatment_geos\', and \'control_geos\''):
      TBRMMDesign()

  def testScoreAttribute(self):
    """The score attribute is passed to the object."""
    score = 0.123
    self.assertIs(TBRMMDesign(score, self.geos1, self.geos2).score, score)

  def testTreatmentGeosAttribute(self):
    """The treatment_geos attribute is passed to the object."""
    self.assertIs(
        TBRMMDesign(0.123, self.geos1, self.geos2).treatment_geos,
        self.geos1)

  def testNumericTreatmentGeosAreAllowed(self):
    """Numeric geo ids are allowed."""
    self.assertSetEqual(
        TBRMMDesign(0.123, {0, 1}, {2, 3}).treatment_geos, {0, 1})
    self.assertSetEqual(TBRMMDesign(0.123, {0, 1}, {2, 3}).control_geos, {2, 3})

  def testTreatmentGeosSetAttribute(self):
    """The treatment_geos attribute can be set."""
    dg = TBRMMDesign(0.123, self.geos1, self.geos2)
    dg.treatment_geos = self.geos3
    self.assertIs(dg.treatment_geos, self.geos3)

  def testControlGeosAttribute(self):
    """The control_geos attribute is passed to the object."""
    self.assertIs(
        TBRMMDesign(0.123, self.geos1, self.geos2).control_geos,
        self.geos2)

  def testControlGeosSetAttribute(self):
    """The control_geos attribute can be set."""
    dg = TBRMMDesign(0.123, self.geos1, self.geos2)
    dg.control_geos = self.geos3
    self.assertIs(dg.control_geos, self.geos3)

  def testMissingControlGeos(self):
    """Control geos must not be an empty set."""
    with self.assertRaisesRegex(ValueError, 'No Control geos'):
      TBRMMDesign(0.123, self.geos1, set())

  def testMissingTreatmentGeos(self):
    """Treatment geos must not be an empty set."""
    with self.assertRaisesRegex(ValueError, 'No Treatment geos'):
      TBRMMDesign(0.123, set(), self.geos2)

  def testDiagAttribute(self):
    """The diag attribute is passed to the object."""
    self.assertIs(TBRMMDesign(0.123, self.geos1, self.geos2, self.diag).diag,
                  self.diag)

  def testOverlappingGeos(self):
    """Control and Treatment sets must not overlap."""
    with self.assertRaisesRegex(
        ValueError,
        r'Control and Treatment geos overlap: \'G1\', \'G2\', \'G3\''):
      TBRMMDesign(0.123, self.geos1, self.geos1)

  def testCompareScoreTuples(self):
    """Objects with smaller scores are 'less' than those with higher ones."""
    self.diag.x = (1.0, 2.0, 3.0, 4.0, 5.0, 6.5)
    score1 = TBRMMScore(self.diag)
    score2 = TBRMMScore(self.diag)
    score1._score = (1, 1, 1, 1, 0.8, 1.2)
    score2._score = (1, 1, 1, 1, 0.9, 1.2)
    self.assertLess(TBRMMDesign(score1, self.geos1, self.geos2),
                    TBRMMDesign(score2, self.geos1, self.geos2))

  def testSort(self):
    """Sorting by score is possible."""
    self.diag.x = (1.0, 2.0, 3.0, 4.0, 5.0, 6.5)
    score1 = TBRMMScore(self.diag)
    score2 = TBRMMScore(self.diag)
    score3 = TBRMMScore(self.diag)
    score1._score = (1, 1, 1, 1, 0.9, 1.2)
    score2._score = (1, 1, 1, 1, 0.8, 1.2)
    score3._score = (1, 0, 1, 1, 0.9, 1.2)
    dg1 = TBRMMDesign(1, self.geos1, self.geos2)
    dg2 = TBRMMDesign(2, self.geos1, self.geos2)
    dg3 = TBRMMDesign(3, self.geos1, self.geos2)
    x = [dg3, dg2, dg1]
    x.sort()
    self.assertEqual(x, [dg1, dg2, dg3])


if __name__ == '__main__':
  unittest.main()
