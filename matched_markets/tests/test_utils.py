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
"""Test utilities.
"""

from matched_markets.methodology import common_classes
from matched_markets.methodology import utils
import altair as alt
import numpy as np
import pandas as pd

import unittest

TimeWindow = common_classes.TimeWindow


class UtilsTest(unittest.TestCase):

  def testRandomizeStrata(self):
    """Check that randomize_strata() works."""

    # Mappings are possible even when number of items is 1.
    self.assertEqual(utils.randomize_strata(1, [1]), [1])
    self.assertLess(set(utils.randomize_strata(1, [1, 2])), {1, 2})

    # Mappings are possible even when number of items <= number of groups.
    self.assertEqual(utils.randomize_strata(2, [1]), [1, 1])
    self.assertEqual(utils.randomize_strata(3, [1]), [1, 1, 1])

    # Check that the mapping contains the expected group ids.
    self.assertCountEqual(utils.randomize_strata(2, [1, 2]), [1, 2])
    self.assertCountEqual(utils.randomize_strata(4, [1, 2]), [1, 2] * 2)
    self.assertCountEqual(utils.randomize_strata(30, [1, 2, 3]), [1, 2, 3] * 10)

    # Mappings are possible also when the number of items is not a multiple of
    # groups.
    groups = utils.randomize_strata(4, [1, 2, 3])
    self.assertTrue(len(groups) == 4)  # pylint: disable=g-generic-assert
    self.assertEqual(set(groups), set([1, 2, 3]))

    # String-valued group ids are possible.
    self.assertCountEqual(utils.randomize_strata(30, ['a', 'b', 'c']),
                          ['a', 'b', 'c'] * 10)

  def testBrownianBridgeBounds(self):
    """Check that brownian_bridge_bounds() are calculated correctly."""

    with self.assertRaisesRegex(ValueError, 'n must be >= 1'):
      utils.brownian_bridge_bounds(0, 1)

    with self.assertRaisesRegex(ValueError, 'sd_bound_multiplier must be > 0'):
      utils.brownian_bridge_bounds(1, 0)

    with self.assertRaisesRegex(ValueError, 'sd_bound_multiplier must be > 0'):
      utils.brownian_bridge_bounds(1, -1)

    # Unit standard deviation.
    self.assertEqual(utils.brownian_bridge_bounds(1, 1), [0.0])
    self.assertEqual(utils.brownian_bridge_bounds(2, 1), [np.sqrt(0.5), 0.0])

    expected_one = utils.brownian_bridge_bounds(3, 1)
    self.assertAlmostEqual(expected_one[0], np.sqrt(2.0 / 3.0))
    self.assertAlmostEqual(expected_one[1], np.sqrt(2.0 / 3.0))
    self.assertAlmostEqual(expected_one[2], 0)

    # S.d. not equal to 1.
    self.assertEqual(utils.brownian_bridge_bounds(2, 2), [np.sqrt(2.0), 0.0])
    expected_two = utils.brownian_bridge_bounds(3, np.sqrt(3))
    self.assertAlmostEqual(expected_two[0], np.sqrt(2))
    self.assertAlmostEqual(expected_two[1], np.sqrt(2))
    self.assertAlmostEqual(expected_two[2], 0)

  def testCredibleIntervalWholeNumbers(self):
    simulations = np.arange(1, 101)
    level = 0.9

    expected = np.array([5.0, 50.0, 95.0])
    obtained = utils.credible_interval(simulations, level)

    np.testing.assert_array_almost_equal(expected, obtained)

  def testCredibleIntervalInterpolation(self):
    simulations = np.arange(1, 101)
    level = 0.88

    expected = np.array([6.0, 50.0, 94.0])
    obtained = utils.credible_interval(simulations, level)

    np.testing.assert_array_almost_equal(expected, obtained)

  def testCredibleIntervalRaisesOnLargeLevel(self):
    simulations = np.arange(1, 101)
    level = 0.999

    with self.assertRaises(ValueError):
      utils.credible_interval(simulations, level)

  def testFindDaysToExclude(self):
    day_week_exclude = [
        '2020/10/10', '2020/11/10-2020/12/10', '2020/08/10']
    days_to_remove = utils.find_days_to_exclude(day_week_exclude)
    expected_days = [
        TimeWindow(pd.Timestamp('2020-10-10'), pd.Timestamp('2020-10-10')),
        TimeWindow(pd.Timestamp('2020-11-10'), pd.Timestamp('2020-12-10')),
        TimeWindow(pd.Timestamp('2020-08-10'), pd.Timestamp('2020-08-10')),
    ]
    for x in range(len(expected_days)):
      self.assertEqual(days_to_remove[x].first_day, expected_days[x].first_day)
      self.assertEqual(days_to_remove[x].last_day, expected_days[x].last_day)

  def testWrongDateFormat(self):
    incorrect_day = ['2020/13/13', '2020/03/03']
    with self.assertRaises(ValueError):
      utils.find_days_to_exclude(incorrect_day)

    incorrect_time_window = ['2020/10/13 - 2020/13/11', '2020/03/03']
    with self.assertRaises(ValueError):
      utils.find_days_to_exclude(incorrect_time_window)

    incorrect_format = ['2020/10/13 - 2020/13/11 . 2020/10/10']
    with self.assertRaises(ValueError):
      utils.find_days_to_exclude(incorrect_format)

  def testExpandTimeWindows(self):
    day_week_exclude = [
        '2020/10/10', '2020/11/10-2020/12/10', '2020/08/10']
    days_to_remove = utils.find_days_to_exclude(day_week_exclude)
    periods = utils.expand_time_windows(days_to_remove)
    expected = [
        pd.Timestamp('2020-10-10', freq='D'),
        pd.Timestamp('2020-08-10', freq='D'),
    ]
    expected += pd.date_range(
        start='2020-11-10', end='2020-12-10', freq='D').to_list()
    self.assertEqual(len(periods), len(expected))
    for x in periods:
      self.assertIn(x, expected)

  def testHumanReadableFormat(self):
    numbers = [123, 10765, 13987482, 8927462746, 1020000000000]
    numb_formatted = [
        utils.human_readable_number(num) for num in numbers
    ]
    self.assertEqual(numb_formatted, ['123', '10.8K', '14M', '8.93B', '1.02tn'])

  def testDefaultGeoAssignment(self):
    geo_level_time_series = pd.DataFrame({
        'geo': [1, 2, 3, 4],
        'response': [1.1, 2.2, 3.3, 4.4]
    })
    geo_eligibility = pd.DataFrame({
        'geo': [1, 3],
        'control': [1, 0],
        'treatment': [0, 1],
        'exclude': [0, 0]
    })

    updated_eligibility = utils.default_geo_assignment(geo_level_time_series,
                                                       geo_eligibility)
    self.assertTrue(
        updated_eligibility.equals(
            pd.DataFrame({
                'geo': [1, 2, 3, 4],
                'control': [1, 1, 0, 1],
                'treatment': [0, 1, 1, 1],
                'exclude': [0, 1, 0, 1]
            })))

  def testPlotIroasOverTime(self):
    iroas_df = pd.DataFrame({
        'date': [
            '2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05'
        ],
        'lower': [0, 0.5, 1, 1.5, 2],
        'mean': [1, 1.5, 2, 2.5, 3],
        'upper': [2, 2.5, 3, 3.5, 4]
    })
    experiment_dates = pd.DataFrame({
        'date': ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04'],
        'color': [
            'Pretest period', 'Pretest period', 'Experiment period',
            'Experiment period'
        ]
    })
    cooldown_date = pd.DataFrame({
        'date': ['2020-01-05'],
        'color': ['End of cooldown period']
    })
    iroas_chart = utils.plot_iroas_over_time(iroas_df, experiment_dates,
                                             cooldown_date)
    self.assertIsInstance(iroas_chart, alt.LayerChart)

  def testFindFrequency(self):
    dates = list(pd.date_range(start='2020-01-01', end='2020-02-01', freq='D'))
    geos = [1, 2, 3, 4]
    df = pd.DataFrame({
        'date': dates * len(geos),
        'geo': sorted(geos * len(dates))
    })

    frequency = utils.infer_frequency(df, 'date', 'geo')
    self.assertEqual(frequency, 'D')

    weeks = list(pd.date_range(start='2020-01-01', end='2020-02-01', freq='W'))
    df = pd.DataFrame({
        'date': weeks * len(geos),
        'geo': sorted(geos * len(weeks))
    })

    frequency = utils.infer_frequency(df, 'date', 'geo')
    self.assertEqual(frequency, 'W')

  def testDifferentFrequencies(self):
    dates = list(pd.date_range(start='2020-01-01', end='2020-02-01', freq='D'))
    weeks = list(pd.date_range(start='2020-01-01', end='2020-02-01', freq='W'))
    geos = [1] * len(dates) + [2] * len(weeks)
    df = pd.DataFrame({
        'date': dates + weeks,
        'geo': geos
    })

    with self.assertRaises(ValueError) as cm:
      utils.infer_frequency(df, 'date', 'geo')
    self.assertEqual(
        str(cm.exception),
        'The provided time series seem to have irregular frequencies.')

  def testFindFrequencyDataNotSorted(self):
    dates = list(pd.date_range(start='2020-01-01', end='2020-02-01', freq='D'))
    geos = [1, 2, 3, 4]
    df = pd.DataFrame({
        'date': dates * len(geos),
        'geo': sorted(geos * len(dates))
    })
    # permute the order of the rows, so that the dataset is not sorted by date
    df = df.sample(frac=1, replace=False)
    frequency = utils.infer_frequency(df, 'date', 'geo')
    self.assertEqual(frequency, 'D')

  def testInsufficientData(self):
    dates = list(pd.date_range(start='2020-01-01', end='2020-01-01', freq='D'))
    geos = [1, 2]
    df = pd.DataFrame({
        'date': dates * len(geos),
        'geo': sorted(geos * len(dates))
    })
    with self.assertRaises(ValueError) as cm:
      utils.infer_frequency(df, 'date', 'geo')
    self.assertEqual(
        str(cm.exception),
        'At least one series with more than one observation must be provided.')

  def testUnknownFrequency(self):
    dates = list(pd.to_datetime(['2020-10-10', '2020-10-13', '2020-10-16']))
    geos = [1, 2]
    df = pd.DataFrame({
        'date': dates * len(geos),
        'geo': sorted(geos * len(dates))
    })

    with self.assertRaises(ValueError) as cm:
      utils.infer_frequency(df, 'date', 'geo')
    self.assertEqual(str(cm.exception),
                     'Frequency could not be identified. Got 3 days.')

if __name__ == '__main__':
  unittest.main()
