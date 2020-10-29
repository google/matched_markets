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
"""Tests for //ads/amt/geoexperiments/methodology/tbr.py."""

import os
from absl import flags
from matched_markets.examples import salesandcost
from matched_markets.methodology import semantics
from matched_markets.methodology import tbr
import unittest




class TBRTest(unittest.TestCase):

  def setUp(self):
    """This method will be run before each of the test methods in the class."""

    super(TBRTest, self).setUp()

    # Load the salesandcost dataset.
    csv_path = 'matched_markets/csv/'
    csv_dir = os.path.join("", csv_path)
    self.data = salesandcost.example_data_formatted(csv_dir)

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

  def testSemanticsAvailable(self):
    """Check if semantics for the data are available."""

    # Fully set up a TBR object.
    tbr_model = tbr.TBR()
    target = self.key_response
    tbr_model.fit(self.data,
                  target,
                  key_response=self.key_response,
                  key_cost=self.key_cost,
                  key_group=self.key_group,
                  key_period=self.key_period,
                  key_date=self.key_date)

    # Check one member of each of the col names, group and period semantics.
    self.assertEqual(tbr_model.df_names.group, self.key_group)
    self.assertEqual(tbr_model.groups.treatment, self.groups.treatment)
    self.assertEqual(tbr_model.periods.cooldown, self.periods.cooldown)

  def testAnalysisDataGenerated(self):
    """Checks whether the salesandcost example data is available."""

    # Fully set up a TBR object.
    tbr_model = tbr.TBR()
    target = self.key_response
    tbr_model.fit(self.data,
                  target,
                  key_response=self.key_response,
                  key_cost=self.key_cost,
                  key_group=self.key_group,
                  key_period=self.key_period,
                  key_date=self.key_date)

    constructed_cols = set(tbr_model.analysis_data.keys())
    correct_cols = {target, self.key_period}

    self.assertCountEqual(constructed_cols, correct_cols)

  def testPeriodIndexFailsWithEmptyPeriods(self):
    """Tests making a period index for an empty iterable raises a ValueError."""

    # Fully set up a TBR object.
    tbr_model = tbr.TBR()
    target = self.key_response
    tbr_model.fit(self.data,
                  target,
                  key_response=self.key_response,
                  key_cost=self.key_cost,
                  key_group=self.key_group,
                  key_period=self.key_period,
                  key_date=self.key_date)

    with self.assertRaises(ValueError):
      tbr_model._make_period_index([])

  def testPeriodIndexWorksForZero(self):
    """Tests making a period index for an empty iterable raises a ValueError."""

    # Fully set up a TBR object.
    tbr_model = tbr.TBR()
    target = self.key_response
    tbr_model.fit(self.data,
                  target,
                  key_response=self.key_response,
                  key_cost=self.key_cost,
                  key_group=self.key_group,
                  key_period=self.key_period,
                  key_date=self.key_date)

    num_in_period = sum(tbr_model.analysis_data[self.key_period] == 0)
    index_count = sum(tbr_model._make_period_index(0))
    self.assertEqual(index_count, num_in_period)

  def testResponseModelCorrect(self):
    """Tests whether model for response has correct coefficients."""

    # Fully set up a TBR object.
    tbr_model = tbr.TBR()
    target = self.key_response
    tbr_model.fit(self.data,
                  target,
                  key_response=self.key_response,
                  key_cost=self.key_cost,
                  key_group=self.key_group,
                  key_period=self.key_period,
                  key_date=self.key_date)

    # Extract slope coefficient from python model.
    response_coef_py = tbr_model.pre_period_model.params[1]

    # Slope coefficient under the R package.
    response_coef_r = 0.9997001

    self.assertAlmostEqual(response_coef_py, response_coef_r)

  def testCausalCumulativePeriods(self):
    """Tests whether model for response has correct coefficients."""

    # Fully set up a TBR object.
    tbr_model = tbr.TBR()
    target = self.key_response

    # Engineer some 'causal' costs in the cooldown period.
    data = self.data.copy()
    cool_index = data[self.key_period] == 2
    treat_index = data[self.key_group] == 2
    data.loc[(cool_index & treat_index), target] += 100.0

    tbr_model.fit(data, target,
                  key_response=self.key_response,
                  key_cost=self.key_cost,
                  key_group=self.key_group,
                  key_period=self.key_period,
                  key_date=self.key_date)
    dist_test = tbr_model.causal_cumulative_distribution(periods=(1))
    dist_cool = tbr_model.causal_cumulative_distribution(periods=(1, 2))

    val_test = dist_test.mean()[-1]
    val_cool = dist_cool.mean()[-1]
    self.assertLessEqual(val_test, val_cool)


if __name__ == '__main__':
  unittest.main()
