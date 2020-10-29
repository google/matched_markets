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
"""Test TBRMMScore.
"""
from matched_markets.methodology import tbrmatchedmarkets
from matched_markets.methodology import tbrmmdata
from matched_markets.methodology import tbrmmdesignparameters
from matched_markets.methodology import tbrmmdiagnostics
from matched_markets.methodology import tbrmmscore
import numpy as np

import unittest

TBRMatchedMarkets = tbrmatchedmarkets.TBRMatchedMarkets
TBRMMDiagnostics = tbrmmdiagnostics.TBRMMDiagnostics
TBRMMDesignParameters = tbrmmdesignparameters.TBRMMDesignParameters
TBRMMData = tbrmmdata.TBRMMData
TBRMMScore = tbrmmscore.TBRMMScore


class ExhaustiveSearch(unittest.TestCase):
  """Test class TBRMMScore."""

  def setUp(self):
    super().setUp()
    self.par = TBRMMDesignParameters(n_test=7, iroas=3.0, min_corr=0.8,
                                     sig_level=0.9, power_level=0.5,
                                     flevel=0.9)
    # Corr(x, y) = 0.86357459.
    y = (255.2, 165.8, 186.0, 160.9, 218.1, 165.7, 212.9, 207.7, 224.4, 205.0,
         247.2, 145.1, 191.1, 173.6)
    x = (132.5, 87.8, 89.4, 78.5, 117.3, 54.0, 134.9, 84.8, 106.4, 95.0, 129.2,
         58.8, 93.6, 92.36)
    self.x = np.array(x)
    self.y = np.array(y)
    self.corr = np.corrcoef(x, y)[0, 1]
    self.diag = TBRMMDiagnostics(self.y, self.par)

  def testCorrectInitialization(self):
    """Check the correct initialization of the class."""
    self.diag.x = self.x
    score = TBRMMScore(self.diag)
    self.assertEqual(score.diag.corr, self.corr)
    self.assertTrue(all(score.diag.y == self.y))
    self.assertTrue(all(score.diag.x == self.x))
    self.assertIsNone(score._score)

  def testScorePropertySetter(self):
    """The score property setter works."""
    self.diag.x = self.x
    score = TBRMMScore(self.diag)
    score.score = (1, 1, 1, 1, 0.5, 2.0)  # Change value.
    self.assertTupleEqual(score._score, (1, 1, 1, 1, 0.5, 2.0))

  def testNoControlGroup(self):
    """An error is raised if the control group is not specified."""
    with self.assertRaisesRegex(
        ValueError,
        r'No Control time series was specified'):
      TBRMMScore(self.diag)

  def testCorrectScore(self):
    """The score is as expected for a valid design."""
    self.diag.x = self.x
    score = TBRMMScore(self.diag)
    self.assertTupleEqual(
        score.score,
        (1, 1, 1, 1, round(self.corr, 2), 1 / self.diag.required_impact))

  def testScoreWhenTestFails(self):
    """The score is as expected for a design which fails a test."""
    # use as control a time series which has low correlation (-0.1227604)
    self.diag.x = [10 + 1 * i for i in range(len(self.y))]
    score = TBRMMScore(self.diag)
    self.assertTupleEqual(
        score.score,
        (0, 1, 1, 1, round(-0.12, 2), 1 / self.diag.required_impact))

  def testCorrectSortingOfMultipleDesigns(self):
    """Check the <= function for the class."""
    self.diag.x = self.x
    # this design pass all tests and has high correlation
    score_optimal = TBRMMScore(self.diag)

    # The pair below implies a failure for the D-W test only.
    x = np.array([132.5, 87.8, 89.4, 78.5, 117.3, 54.0, 134.9, 84.8, 106.4,
                  95.0, 129.2, 58.8, 93.6, 92.3, 122.7, 78.0, 96.6, 82.4,
                  100.8, 111.7, 78.0])
    y = np.array([487.9, 393.6, 388.8, 375.0, 420.9, 305.5, 451.1, 364.2, 423.4,
                  376.2, 450.5, 303.9, 370.3, 371.2, 445.1, 333.7, 397.9,
                  398.0, 416.4, 419.6, 338.2])
    diag_suboptimal = TBRMMDiagnostics(y, self.par)
    diag_suboptimal.x = x
    score_suboptimal = TBRMMScore(diag_suboptimal)

    # this design fails the correlation test
    diag_worst = TBRMMDiagnostics(self.y, self.par)
    diag_worst.x = [10 + 1 * i for i in range(len(self.y))]
    score_worst = TBRMMScore(diag_worst)

    # the scores should be (from best to worst): score_optimal, score_suboptimal
    # and score_worst
    self.assertGreater(score_optimal, score_suboptimal)
    self.assertGreater(score_optimal, score_worst)
    self.assertGreater(score_suboptimal, score_worst)

  def testNegativeCorrelationInScore(self):
    """Negative correlations are handled as expected."""
    self.diag.x = self.x
    # score_positive has a correlation of 0.86
    score_positive = TBRMMScore(self.diag)
    score_negative = TBRMMScore(self.diag)
    # change the correlation value to -0.99
    score_negative.score = score_negative.score._replace(corr=-0.99)
    self.assertLess(score_negative, score_positive)

if __name__ == '__main__':
  unittest.main()
