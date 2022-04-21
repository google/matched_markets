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
"""Tests for common_classes."""

from matched_markets.methodology import common_classes
import pandas as pd

import unittest

TimeSeries = common_classes.TimeSeries
TimeWindow = common_classes.TimeWindow
EstimatedTimeSeriesWithConfidenceInterval = common_classes.EstimatedTimeSeriesWithConfidenceInterval


class CommonClassesTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self._t1 = '2019-01-01'
    self._t2 = '2020-01-01'
    self.df = pd.DataFrame({'date': ['2020-10-10', '2020-10-11'],
                            'lower': [0.0, 0.0],
                            'upper': [2.1, 3.2],
                            'estimate': [1.24, 2.5]})

  def testTimeWindowCorrectInitialization(self):
    result = TimeWindow(self._t1, self._t2)
    self.assertEqual(pd.Timestamp(self._t1), result.first_day)
    self.assertEqual(pd.Timestamp(self._t2), result.last_day)

  def testTimeWindowInitializationError(self):
    # first day is chronologically after the last day
    with self.assertRaises(ValueError):
      TimeWindow('2020-02-01', self._t2)

  def testTimeSeriesCorrectInitialization(self):
    # the dataframe has the correct columns
    time_series = EstimatedTimeSeriesWithConfidenceInterval(self.df)
    self.assertTrue(time_series.equals(self.df))

  def testTimeSeriesMissingColumn(self):
    # the dataframe is missing a column
    with self.assertRaisesRegex(
        KeyError, r'The time series must contain the columns ' +
        '"date", "estimate", "lower", "upper".'):
      EstimatedTimeSeriesWithConfidenceInterval(self.df.drop(columns='upper'))

  def testTimeSeriesIncorrectLowerBound(self):
    # the dataframe has an incorrect lower bound
    self.df['lower'] = [0.0, 3.3]
    with self.assertRaisesRegex(
        ValueError, r'lower bound is not smaller than point estimate.'):
      EstimatedTimeSeriesWithConfidenceInterval(self.df)

  def testTimeSeriesIncorrectUpperBound(self):
    # the dataframe has an incorrect upper bound
    self.df['upper'] = [1.24, 2.3]
    with self.assertRaisesRegex(
        ValueError, r'upper bound is not larger than point estimate.'):
      EstimatedTimeSeriesWithConfidenceInterval(self.df)

  def testTimeSeriesClassCorrectInitialization(self):
    # the time series has the correct attributes
    data = EstimatedTimeSeriesWithConfidenceInterval(self.df)
    time_series = TimeSeries(data, data, data)
    self.assertTrue(time_series.counterfactual.equals(self.df))
    self.assertTrue(time_series.pointwise_difference.equals(self.df))
    self.assertTrue(time_series.cumulative_effect.equals(self.df))

if __name__ == '__main__':
  unittest.main()
