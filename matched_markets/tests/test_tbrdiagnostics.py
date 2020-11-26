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
"""Test TBR diagnostics.
"""

import os

from absl import flags
from matched_markets.examples import geoxdiag_data
from matched_markets.methodology import tbrdiagnostics
import pandas as pd
from scipy import stats

import unittest

CSV_PATH = 'matched_markets/csv/'



class TBRDiagnosticsTest(unittest.TestCase):

  def setUp(self):
    """This method will be run before each of the test methods in the class."""
    super().setUp()
    # Load the geoxdiag dataset.
    csv_dir = os.path.join("", CSV_PATH)
    data = geoxdiag_data.read_data(csv_dir)
    self.data = data
    # Names of the columns in the data frame.
    self.response = 'response'
    self.group_col = 'group'
    # Control and treatment IDs in the data frame.
    self.ctl_id = 1
    self.trt_id = 2
    # Alternative column name which does not exist in the data.
    self.alt_response = 'response2'

  def testAttributes(self):
    """Checks if the attributes are available."""

    tbrdiag = tbrdiagnostics.TBRDiagnostics()

    # Other attributes.
    self.assertIsNone(tbrdiag._data)
    self.assertIsNone(tbrdiag._analysis_data)
    self.assertIsNone(tbrdiag._tests_passed)

    diag = tbrdiag._diagnostics
    self.assertIsInstance(diag, dict)
    self.assertIsNone(diag['corr_test'])
    self.assertIsNone(diag['enough_data'])
    self.assertIsNone(diag['pretest_start'])
    self.assertIsNone(diag['outlier_dates'])

    # Accessors.
    self.assertEqual(tbrdiag._diagnostics, tbrdiag.get_test_results())
    self.assertIsNone(tbrdiag.tests_passed())
    self.assertIsNone(tbrdiag.get_analysis_data())
    self.assertIsNone(tbrdiag.get_data())

  def testSemanticsAvailable(self):
    """Checks if semantics for the data are properly stored and available."""

    tbrdiag = tbrdiagnostics.TBRDiagnostics()

    # Set column names, group and period ids to non-default values.
    data2 = self.data.copy()
    new_column_names = [name + '2' for name in data2.columns.values]
    data2.columns = new_column_names
    data2.loc[:, 'group2'] = data2.loc[:, 'group2'] * 100
    data2.loc[:, 'period2'] = data2.loc[:, 'period2'] + 10
    tbrdiag.fit(data2,
                target='response2',
                key_date='date2',
                key_geo='geo2',
                key_response='response2',
                key_group='group2',
                key_period='period2',
                group_control=100,
                group_treatment=200,
                period_pre=10,
                period_test=11,
                period_cooldown=12)

    self.assertEqual(tbrdiag._df_names.date, 'date2')
    self.assertEqual(tbrdiag._df_names.geo, 'geo2')
    self.assertEqual(tbrdiag._df_names.response, 'response2')
    self.assertEqual(tbrdiag._df_names.group, 'group2')
    self.assertEqual(tbrdiag._df_names.period, 'period2')
    self.assertEqual(tbrdiag._groups.control, 100)
    self.assertEqual(tbrdiag._groups.treatment, 200)
    self.assertEqual(tbrdiag._periods.pre, 10)
    self.assertEqual(tbrdiag._periods.test, 11)
    self.assertEqual(tbrdiag._periods.cooldown, 12)

  def testAnalysisData(self):
    """Checks if the analysis data set is properly created."""
    tbrdiag = tbrdiagnostics.TBRDiagnostics()

    target = self.response

    tbrdiag.fit(self.data, target=target)
    orig_data = self.data
    data = tbrdiag.get_analysis_data()

    self.assertEqual(data.columns.values.tolist(), ['period', 'x', 'y'])
    self.assertEqual(list(data.index.names), [tbrdiag._df_names.date])

    group_col = self.group_col
    ctl_id = self.ctl_id
    trt_id = self.trt_id

    x_sum = orig_data.loc[orig_data[group_col] == ctl_id, target].sum()
    y_sum = orig_data.loc[orig_data[group_col] == trt_id, target].sum()
    self.assertAlmostEqual(data.x.sum(), x_sum)
    self.assertAlmostEqual(data.y.sum(), y_sum)

    dates_in_original_df = orig_data.date.unique()
    dates_in_new_df = data.index.get_level_values('date').values
    self.assertTrue(all(date in dates_in_new_df
                        for date in dates_in_original_df))

    # Raise an error if there is no control group.
    mutated_data = orig_data.copy()
    mutated_data.loc[mutated_data[group_col] == ctl_id, group_col] = -1
    with self.assertRaisesRegex(
        ValueError, 'Both control and treatment group'
        ' ids must be present'):
      tbrdiag.fit(mutated_data)

    # Raise an error if there is no treatment group.
    mutated_data = orig_data.copy()
    mutated_data.loc[mutated_data[group_col] == trt_id, group_col] = -1
    with self.assertRaisesRegex(
        ValueError, 'Both control and treatment group'
        ' ids must be present'):
      tbrdiag.fit(mutated_data)

  def testObservedCorrelation(self):
    """Tests that the observed correlation works."""

    tbrdiag = tbrdiagnostics.TBRDiagnostics()
    tbrdiag.fit(self.data)
    self.assertAlmostEqual(tbrdiag.obs_cor(), 0.99680134)

  def testCorrelationTestMinThreshold(self):
    """Tests the correlation test threshold computation."""

    tbrdiag = tbrdiagnostics.TBRDiagnostics()
    threshold12 = tbrdiag._min_correlation_threshold(n=12,
                                                     min_cor=0.5,
                                                     credible_level=0.95)
    self.assertAlmostEqual(threshold12, 0.799631922)

    # With n>=4, no error is given.
    threshold4 = tbrdiag._min_correlation_threshold(n=4,
                                                    min_cor=0.5,
                                                    credible_level=0.95)
    self.assertAlmostEqual(threshold4, 0.975461634)

    # With n<=3, error is raised.
    with self.assertRaisesRegex(ValueError, 'Number of observations must be'
                                ' at least 4'):
      tbrdiag._min_correlation_threshold(n=3, min_cor=0.5, credible_level=0.95)

  def testCorrelationTest(self):
    """Tests that the correlation test works."""

    tbrdiag = tbrdiagnostics.TBRDiagnostics()
    target = self.response
    tbrdiag.fit(self.data, target=target)
    self.assertTrue(tbrdiag._correlation_test)
    self.assertTrue(tbrdiag._diagnostics['corr_test'])

    # Modify the data to have a negative correlation and check that the test
    # fails.
    data = self.data.copy()
    x = data.loc[data[self.group_col] == self.ctl_id, target]
    data.loc[data[self.group_col] == self.ctl_id, target] = -x
    tbrdiag.fit(data, target=target)
    self.assertFalse(tbrdiag._diagnostics['corr_test'])

  def testThereAreNoOutliers(self):
    """Tests that the geoxdiag data set has no outliers by default."""

    tbrdiag = tbrdiagnostics.TBRDiagnostics()
    tbrdiag.fit(self.data, target=self.response)
    self.assertEqual(tbrdiag._diagnostics['outlier_dates'], [])

  def testThereAreOutliers(self):
    """Tests that an outlier date is detected."""

    tbrdiag = tbrdiagnostics.TBRDiagnostics()
    perturbed_data = self.data.copy()
    bad_date = pd.Timestamp('2018-01-30')
    select_geo_and_date = ((perturbed_data['date'] == bad_date) &
                           (perturbed_data['geo'] == 19))
    perturbed_data.loc[select_geo_and_date, 'response'] = 1000
    tbrdiag.fit(perturbed_data)
    self.assertEqual(tbrdiag._diagnostics['outlier_dates'], [bad_date])

  def testNoisyGeos(self):
    """Tests that noisy geos are detected."""

    # The default data set has no noisy geos.
    tbrdiag = tbrdiagnostics.TBRDiagnostics()
    target = self.response
    tbrdiag.fit(self.data, target=target)
    self.assertEqual(tbrdiag._diagnostics['noisy_geos'], [])

    # Remove a single day for a single geo, this geo should NOT be detected.
    data = self.data.copy()
    data = data[~((data['geo'] == 1) & (data['date'] == '2018-01-02'))]
    tbrdiag = tbrdiagnostics.TBRDiagnostics()
    target = self.response
    tbrdiag.fit(data, target=target)
    self.assertEqual(tbrdiag._diagnostics['noisy_geos'], [])

    # Replace Geo #1 with noise, expect it to be detected.
    data = self.data.copy()
    n_days = sum(data.geo == 1)
    data.loc[data.geo == 1, target] = stats.norm.rvs(loc=10,
                                                     scale=1, size=n_days)
    tbrdiag = tbrdiagnostics.TBRDiagnostics()
    tbrdiag.fit(data, target=target)
    self.assertCountEqual(tbrdiag._diagnostics['noisy_geos'], [1])
    # The data have been modified to exclude geo 1.
    self.assertFalse(tbrdiag.get_data().geo.isin([1]).any())

    # Replace Geo #2 with zeros, expect it to be detected as well.
    data.loc[data.geo == 2, target] = 0
    tbrdiag = tbrdiagnostics.TBRDiagnostics()
    tbrdiag.fit(data, target=target)
    self.assertCountEqual(tbrdiag._diagnostics['noisy_geos'], [1, 2])
    # The data have been modified to exclude geos 1 and 2.
    self.assertFalse(tbrdiag.get_data().geo.isin([1, 2]).any())

    # If the data set has 3 geos or fewer, None is returned.
    data.loc[data.geo >= 3, 'geo'] = 3
    tbrdiag = tbrdiagnostics.TBRDiagnostics()
    tbrdiag.fit(data, target=target)
    self.assertIsNone(tbrdiag.get_test_results()['noisy_geos'])

  def testTargetSpecification(self):
    """Tests that 'target' specification works."""

    # Fit the data with the default target (=default response name).
    tbrdiag0 = tbrdiagnostics.TBRDiagnostics()
    tbrdiag0.fit(self.data)
    data0 = tbrdiag0.get_analysis_data()

    # Drop the response column and set up another with a different name.
    mod_data = self.data.copy().drop(columns=self.response)
    alt_target = self.alt_response
    mod_data.loc[:, alt_target] = self.data.loc[:, self.response]

    # First fit with the target column specified.
    tbrdiag1 = tbrdiagnostics.TBRDiagnostics()
    tbrdiag1.fit(mod_data, target=alt_target)
    data1 = tbrdiag1.get_analysis_data()

    # Fit with the target column.
    tbrdiag2 = tbrdiagnostics.TBRDiagnostics()
    tbrdiag2.fit(mod_data, key_response=alt_target)
    data2 = tbrdiag2.get_analysis_data()

    # All of the data should be equivalent.
    self.assertTrue(data0.equals(data1))
    self.assertTrue(data0.equals(data2))

  def testFit(self):
    """Tests that the accessor methods are available."""

    tbrdiag = tbrdiagnostics.TBRDiagnostics()
    tbrdiag.fit(self.data)

    # The data obtained by get_data() must be a copy.
    self.assertIsNot(self.data, tbrdiag.get_data())
    self.assertTrue(self.data.equals(tbrdiag.get_data()))


if __name__ == '__main__':
  unittest.main()
