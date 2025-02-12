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
"""Tests for the robust TBR model."""

import os
from absl import flags
from matched_markets.examples import salesandcost
from matched_markets.methodology import robust
from matched_markets.methodology import semantics
import pandas as pd
import unittest
from absl.testing import parameterized




class RobustTBRTest(parameterized.TestCase, unittest.TestCase):
  """Tests for the RobustTBR class."""

  def setUp(self):
    """This method will be run before each of the test methods in the class."""

    super(RobustTBRTest, self).setUp()

    # Load the salesandcost dataset.
    csv_path = 'matched_markets/csv/'
    csv_dir = os.path.join("", csv_path)
    self.data = salesandcost.example_data_formatted(
        csv_dir, calibration_duration=15
    )

    # Data frame names for the salesandcost example.
    self.key_response = 'sales'
    self.key_cost = 'cost'
    self.key_group = 'geo.group'
    self.key_period = 'period'
    self.key_geo = 'geo'
    self.key_date = 'date'

    # Semantics for groups and periods.
    self.groups = semantics.GroupSemantics()
    self.periods = semantics.PeriodSemantics()

  def test_semantics_available(self):
    """Check if semantics for the data are available."""

    # Fully set up a TBR object.
    tbr_model = robust.RobustTBR()
    target = self.key_response
    tbr_model.fit(
        self.data,
        target,
        key_response=self.key_response,
        key_cost=self.key_cost,
        key_group=self.key_group,
        key_period=self.key_period,
        key_date=self.key_date,
    )

    # Check one member of each of the col names, group and period semantics.
    self.assertEqual(tbr_model.df_names.group, self.key_group)
    self.assertEqual(tbr_model.groups.treatment, self.groups.treatment)
    self.assertEqual(tbr_model.periods.cooldown, self.periods.cooldown)
    self.assertEqual(tbr_model.periods.calibration, self.periods.calibration)

  def test_analysis_data_generated(self):
    """Checks whether the salesandcost example data is available."""

    # Fully set up a TBR object.
    tbr_model = robust.RobustTBR()
    target = self.key_response
    tbr_model.fit(
        self.data,
        target,
        key_response=self.key_response,
        key_cost=self.key_cost,
        key_group=self.key_group,
        key_period=self.key_period,
        key_date=self.key_date,
    )

    constructed_cols = set(tbr_model.analysis_data.keys())
    correct_cols = {target, self.key_period}

    self.assertCountEqual(constructed_cols, correct_cols)

  def test_period_index_fails_with_empty_periods(self):
    """Tests making a period index for an empty iterable raises a ValueError."""

    # Fully set up a TBR object.
    tbr_model = robust.RobustTBR()
    target = self.key_response
    tbr_model.fit(
        self.data,
        target,
        key_response=self.key_response,
        key_cost=self.key_cost,
        key_group=self.key_group,
        key_period=self.key_period,
        key_date=self.key_date,
    )
    # pylint: disable=protected-access
    with self.assertRaises(ValueError):
      tbr_model._make_period_index([])
    # pylint: enable=protected-access

  def test_period_index_works_for_zero(self):
    """Tests making a period index for an empty iterable raises a ValueError."""

    # Fully set up a TBR object.
    tbr_model = robust.RobustTBR()
    target = self.key_response
    tbr_model.fit(
        self.data,
        target,
        key_response=self.key_response,
        key_cost=self.key_cost,
        key_group=self.key_group,
        key_period=self.key_period,
        key_date=self.key_date,
    )

    num_in_period = sum(tbr_model.analysis_data[self.key_period] == 0)
    # pylint: disable=protected-access
    index_count = sum(tbr_model._make_period_index(0))
    # pylint: enable=protected-access
    self.assertEqual(index_count, num_in_period)

  def test_response_model_correct(self):
    """Tests whether model for response has correct coefficients."""

    # Fully set up a TBR object.
    tbr_model = robust.RobustTBR()
    target = self.key_response
    tbr_model.fit(
        self.data,
        target,
        key_response=self.key_response,
        key_cost=self.key_cost,
        key_group=self.key_group,
        key_period=self.key_period,
        key_date=self.key_date,
    )

    # Extract slope coefficient from python model.
    response_coef_py = tbr_model.pre_period_model.params[1]

    # Slope coefficient under the R package.
    response_coef_r = 0.9997001

    self.assertAlmostEqual(response_coef_py, response_coef_r)

  def test_causal_effect_bias_correction_fails_with_no_calibration_data(self):
    """Tests whether model for response fails with no calibration data."""
    tbr_model = robust.RobustTBR()
    target = self.key_response
    calibration_period = semantics.PeriodSemantics().calibration
    tbr_model.fit(
        self.data.loc[self.data[self.key_period] != calibration_period],
        target,
        key_response=self.key_response,
        key_cost=self.key_cost,
        key_group=self.key_group,
        key_period=self.key_period,
        key_date=self.key_date,
    )
    with self.assertRaises(ValueError):
      tbr_model.causal_effect(self.periods.test, enable_aa_bias_correction=True)

  @parameterized.named_parameters(
      dict(
          testcase_name='with_aa_bias_correction',
          enable_aa_bias_correction=True,
      ),
      dict(
          testcase_name='without_aa_bias_correction',
          enable_aa_bias_correction=False,
      ),
  )
  def test_causal_effect_works(self, enable_aa_bias_correction):
    """Tests whether the `causal_effect` method works."""
    # Fully set up a TBR object.
    tbr_model = robust.RobustTBR()
    target = self.key_response
    tbr_model.fit(
        self.data,
        target,
        key_response=self.key_response,
        key_cost=self.key_cost,
        key_group=self.key_group,
        key_period=self.key_period,
        key_date=self.key_date,
    )
    causal_effect = tbr_model.causal_effect(
        self.periods.test, enable_aa_bias_correction=enable_aa_bias_correction
    )
    self.assertIsInstance(causal_effect, pd.Series)
    self.assertLen(causal_effect, 28)
    self.assertIsInstance(causal_effect.iloc[0], float)
    # Total causal effect is expected to be positive.
    self.assertGreater(causal_effect.sum(), 0)

  @parameterized.named_parameters(
      dict(
          testcase_name='use_cooldown',
          use_cooldown=True,
      ),
      dict(
          testcase_name='donot_use_cooldown',
          use_cooldown=False,
      ),
  )
  def test_cumulative_causal_effect_works(self, use_cooldown):
    """Tests whether the `cumulative_causal_effect` method works."""
    # Fully set up a TBR object.
    tbr_model = robust.RobustTBR(use_cooldown=use_cooldown)
    target = self.key_response
    tbr_model.fit(
        self.data,
        target,
        key_response=self.key_response,
        key_cost=self.key_cost,
        key_group=self.key_group,
        key_period=self.key_period,
        key_date=self.key_date,
    )
    cumulative_causal_effect = tbr_model.cumulative_causal_effect()
    self.assertIsInstance(cumulative_causal_effect, float)
    # Total causal effect is expected to be positive.
    self.assertGreater(cumulative_causal_effect, 0)

  def test_cumulative_causal_effect_interval_works(self):
    """Tests whether the `cumulative_causal_effect_interval` method works."""
    # Fully set up a TBR object.
    tbr_model = robust.RobustTBR(use_cooldown=False)
    target = self.key_response
    tbr_model.fit(
        self.data,
        target,
        key_response=self.key_response,
        key_cost=self.key_cost,
        key_group=self.key_group,
        key_period=self.key_period,
        key_date=self.key_date,
    )
    tbr_analysis_data = tbr_model.analysis_data.copy()
    cumulative_causal_effect = tbr_model.cumulative_causal_effect()
    ci_lower, ci_upper = tbr_model.cumulative_causal_effect_interval(level=0.8)
    self.assertGreater(ci_lower, 0)
    self.assertGreater(ci_upper, 0)
    self.assertGreater(ci_upper, ci_lower)
    # Cumulative causal effect is expected to be within the confidence interval.
    self.assertGreater(cumulative_causal_effect, ci_lower)
    self.assertLess(cumulative_causal_effect, ci_upper)
    pd.testing.assert_frame_equal(tbr_model.analysis_data, tbr_analysis_data)

  def test_cumulative_causal_effect_interval_fails_with_invalid_tails(self):
    """Tests whether the `cumulative_causal_effect_interval` method fails."""
    # Fully set up a TBR object.
    tbr_model = robust.RobustTBR(use_cooldown=False)
    target = self.key_response
    tbr_model.fit(
        self.data,
        target,
        key_response=self.key_response,
        key_cost=self.key_cost,
        key_group=self.key_group,
        key_period=self.key_period,
        key_date=self.key_date,
    )
    with self.assertRaises(NotImplementedError):
      tbr_model.cumulative_causal_effect_interval(tails=1)

  def test_cumulative_causal_effect_interval_fails_with_small_initial_width(
      self,
  ):
    """Tests whether the `cumulative_causal_effect_interval` method fails."""
    # Fully set up a TBR object.
    tbr_model = robust.RobustTBR(use_cooldown=False)
    target = self.key_response
    tbr_model.fit(
        self.data,
        target,
        key_response=self.key_response,
        key_cost=self.key_cost,
        key_group=self.key_group,
        key_period=self.key_period,
        key_date=self.key_date,
    )
    with self.assertRaises(ValueError):
      tbr_model.cumulative_causal_effect_interval(initial_width_multiplier=0.01)

  def test_permute_analysis_data_works(self):
    """Tests whether the `_permute_analysis_data` method works."""
    # Fully set up a TBR object.
    tbr_model = robust.RobustTBR(use_cooldown=False)
    target = self.key_response
    tbr_model.fit(
        self.data,
        target,
        key_response=self.key_response,
        key_cost=self.key_cost,
        key_group=self.key_group,
        key_period=self.key_period,
        key_date=self.key_date,
    )
    # pylint: disable=protected-access
    indices = tbr_model._get_indices_by_group(
        (self.periods.test, self.periods.calibration)
    )
    self.assertIsInstance(indices, dict)
    self.assertLen(indices, 2)
    self.assertCountEqual(
        indices.keys(), [self.groups.treatment, self.groups.control]
    )
    # 28 days in the test period and 15 days in the calibration period.
    self.assertLen(indices[self.groups.treatment], 43)
    self.assertLen(indices[self.groups.control], 43)
    permuted_data = tbr_model._permute_analysis_data(
        tbr_model.analysis_data, indices
    )
    # pylint: enable=protected-access
    permuted_data_list = list(permuted_data)
    # In total 42 permutations are generated.
    self.assertLen(permuted_data_list, 42)
    self.assertIsInstance(permuted_data_list[0], pd.DataFrame)
    # Index and columns are the same as the original analysis data. Only the
    # target values are modified.
    self.assertCountEqual(
        permuted_data_list[0].columns, tbr_model.analysis_data.columns
    )
    self.assertCountEqual(
        permuted_data_list[0].index, tbr_model.analysis_data.index
    )
    self.assertCountEqual(
        permuted_data_list[0][self.key_period],
        tbr_model.analysis_data[self.key_period],
    )

  def test_permute_analysis_data_raises_warning_with_insufficient_data(self):
    """Tests whether the `_permute_analysis_data` method fails."""
    # Fully set up a TBR object.
    tbr_model = robust.RobustTBR(use_cooldown=False)
    target = self.key_response
    tbr_model.fit(
        # Query the data between 2015-02-04 and 2015-02-11 to make the number of
        # data points less than 10.
        self.data.query("date > '2015-02-03' and date < '2015-02-11'"),
        target,
        key_response=self.key_response,
        key_cost=self.key_cost,
        key_group=self.key_group,
        key_period=self.key_period,
        key_date=self.key_date,
    )
    # pylint: disable=protected-access
    indices = tbr_model._get_indices_by_group(
        (self.periods.cooldown, self.periods.calibration)
    )
    with self.assertWarns(Warning):
      permuted_data = tbr_model._permute_analysis_data(
          tbr_model.analysis_data, indices
      )
      next(permuted_data)

  @parameterized.named_parameters(
      dict(
          testcase_name='use_cooldown',
          use_cooldown=True,
      ),
      dict(
          testcase_name='donot_use_cooldown',
          use_cooldown=False,
      ),
  )
  def test_summary_works(self, use_cooldown):
    """Tests whether the `summary` method works."""
    # Fully set up a TBR object.
    tbr_model = robust.RobustTBR(use_cooldown=use_cooldown)
    target = self.key_response
    tbr_model.fit(
        self.data,
        target,
        key_response=self.key_response,
        key_cost=self.key_cost,
        key_group=self.key_group,
        key_period=self.key_period,
        key_date=self.key_date,
    )
    report = tbr_model.summary()
    self.assertIsInstance(report, pd.DataFrame)
    self.assertCountEqual(
        report.columns,
        [
            'estimate',
            'lower',
            'upper',
            'level',
        ],
    )
    self.assertLen(report, 1)
    self.assertGreater(report.upper.iloc[0], report.estimate.iloc[0])
    self.assertGreater(report.estimate.iloc[0], report.lower.iloc[0])
    self.assertAlmostEqual(report.level.iloc[0], 0.8)

  def test_summary_fails_with_invalid_level(self):
    """Tests whether the `summary` method fails with invalid level."""
    # Fully set up a TBR object.
    tbr_model = robust.RobustTBR(use_cooldown=False)
    target = self.key_response
    tbr_model.fit(
        self.data,
        target,
        key_response=self.key_response,
        key_cost=self.key_cost,
        key_group=self.key_group,
        key_period=self.key_period,
        key_date=self.key_date,
    )
    with self.assertRaises(ValueError):
      tbr_model.summary(level=-0.2)

  def test_summary_fails_with_invalid_report_type(self):
    """Tests whether the `summary` method fails with invalid report type."""
    # Fully set up a TBR object.
    tbr_model = robust.RobustTBR(use_cooldown=False)
    target = self.key_response
    tbr_model.fit(
        self.data,
        target,
        key_response=self.key_response,
        key_cost=self.key_cost,
        key_group=self.key_group,
        key_period=self.key_period,
        key_date=self.key_date,
    )
    with self.assertRaises(NotImplementedError):
      tbr_model.summary(report='all')

  def test_summary_fails_without_fitting_tbr_model(self):
    """Tests whether the `summary` method fails without fitting the model."""
    # Fully set up a TBR object.
    tbr_model = robust.RobustTBR(use_cooldown=False)
    with self.assertRaises(RuntimeError):
      tbr_model.summary()

if __name__ == '__main__':
  unittest.main()
