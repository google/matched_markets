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
"""Tests for the robust TBR reporting."""

import os

from absl import flags
from matched_markets.examples import salesandcost
from matched_markets.methodology import robust_iroas
from matched_markets.methodology import semantics
import pandas as pd

import unittest
from absl.testing import parameterized





class RobustTBRiROASTest(parameterized.TestCase, unittest.TestCase):
  """Class for testing tbr_iroas."""

  def setUp(self):
    """This method will be run before each of the test methods in the class."""

    super(RobustTBRiROASTest, self).setUp()

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

  def test_summary_method_works_under_fixed_scenario(self):
    """Tests whether the summary method works under fixed cost scenario."""
    # Fully set up a TBR object.
    iroas_model = robust_iroas.RobustTBRiROAS(use_cooldown=False)
    iroas_model.fit(
        self.data,
        key_response=self.key_response,
        key_cost=self.key_cost,
        key_group=self.key_group,
        key_period=self.key_period,
        key_date=self.key_date,
    )
    test_summary = iroas_model.summary()
    self.assertIsInstance(test_summary, pd.DataFrame)
    self.assertCountEqual(
        test_summary.columns,
        [
            'estimate',
            'lower',
            'upper',
            'level',
            'incremental_response',
            'incremental_cost',
            'scenario',
        ],
    )
    self.assertLen(test_summary, 1)
    self.assertGreater(
        test_summary.upper.iloc[0], test_summary.estimate.iloc[0]
    )
    self.assertGreater(
        test_summary.estimate.iloc[0], test_summary.lower.iloc[0]
    )
    # The lower end of iROAS is expected to be > 2.
    self.assertGreater(test_summary.lower.iloc[0], 2)
    # The upper end of iROAS is expected to be < 4.
    self.assertLess(test_summary.upper.iloc[0], 4)
    self.assertGreater(test_summary.incremental_cost.iloc[0], 0)
    self.assertGreater(test_summary.incremental_response.iloc[0], 0)
    self.assertAlmostEqual(test_summary.level.iloc[0], 0.8)
    self.assertEqual(test_summary.scenario.iloc[0], 'fixed')

  @parameterized.named_parameters(
      dict(
          testcase_name='with_equalized_tbr_slope',
          equalize_tbr_slope=True,
      ),
      dict(
          testcase_name='without_equalized_tbr_slope',
          equalize_tbr_slope=False,
      ),
  )
  def test_summary_method_works_under_variable_scenario(
      self, equalize_tbr_slope
  ):
    """Tests whether the summary method works under variable cost scenario."""
    # Fully set up a TBR object.
    iroas_model = robust_iroas.RobustTBRiROAS(use_cooldown=False)
    variable_cost_data = self.data.copy()
    group_semantics = semantics.GroupSemantics()
    period_semantics = semantics.PeriodSemantics()
    # Ensure the cost is non-zero for the pre-test (including calibration)
    # period.
    variable_cost_data[self.key_cost] = variable_cost_data[self.key_cost] + 50
    # Swap the control and treatment groups. After that, we expect incremental
    # response to be negative.
    variable_cost_data[self.key_group] = variable_cost_data[
        self.key_group
    ].map({
        group_semantics.control: group_semantics.treatment,
        group_semantics.treatment: group_semantics.control,
    })
    # For the treatment group, set the cost to zero during the test period due
    # to ads ablation.
    variable_cost_data.loc[
        (variable_cost_data[self.key_group] == group_semantics.treatment)
        & (variable_cost_data[self.key_period] == period_semantics.test),
        self.key_cost,
    ] = 0
    iroas_model.fit(
        variable_cost_data,
        key_response=self.key_response,
        key_cost=self.key_cost,
        key_group=self.key_group,
        key_period=self.key_period,
        key_date=self.key_date,
    )
    test_summary = iroas_model.summary(equalize_tbr_slope=equalize_tbr_slope)
    self.assertIsInstance(test_summary, pd.DataFrame)
    self.assertCountEqual(
        test_summary.columns,
        [
            'estimate',
            'lower',
            'upper',
            'level',
            'incremental_response',
            'incremental_cost',
            'scenario',
        ],
    )
    self.assertLen(test_summary, 1)
    self.assertGreater(
        test_summary.upper.iloc[0], test_summary.estimate.iloc[0]
    )
    self.assertGreater(
        test_summary.estimate.iloc[0], test_summary.lower.iloc[0]
    )
    self.assertLess(test_summary.incremental_cost.iloc[0], 0)
    self.assertLess(test_summary.incremental_response.iloc[0], 0)
    self.assertAlmostEqual(test_summary.level.iloc[0], 0.8)
    self.assertEqual(test_summary.scenario.iloc[0], 'variable')


if __name__ == '__main__':
  unittest.main()
