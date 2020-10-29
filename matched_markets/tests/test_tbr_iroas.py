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
"""Tests for //ads/amt/geoexperiments/methodology/tbr_iroas.py."""

import copy
import os

from absl import flags
from matched_markets.examples import salesandcost
from matched_markets.methodology import semantics
from matched_markets.methodology import tbr_iroas
from matched_markets.methodology import utils
import numpy as np

import unittest




class TBRiROASTest(unittest.TestCase):

  def setUp(self):
    """This method will be run before each of the test methods in the class."""

    super(TBRiROASTest, self).setUp()

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

  def testFixedCostIROASSummary(self):

    # Fully set up a TBR object.
    iroas_model = tbr_iroas.TBRiROAS(use_cooldown=False)
    iroas_model.fit(self.data,
                    key_response=self.key_response,
                    key_cost=self.key_cost,
                    key_group=self.key_group,
                    key_period=self.key_period,
                    key_date=self.key_date)

    # Arguments for the type of tests to conduct.
    level = 0.9
    posterior_threshold = 0.0
    tails = 1

    # Summary values from R.
    r_estimate = 2.946742
    r_precision = 0.120548
    r_lower = 2.826194
    r_incr_resp = 147337.122
    r_incr_cost = 50000
    r_probability = 1.0

    # Summary values from python.
    iroas = iroas_model.summary(
        level=level, posterior_threshold=posterior_threshold, tails=tails)
    py_estimate = iroas['estimate'].iloc[0]
    py_precision = iroas['precision'].iloc[0]
    py_lower = iroas['lower'].iloc[0]
    py_incr_resp = iroas['incremental_response'].iloc[0]
    py_incr_cost = iroas['incremental_cost'].iloc[0]
    py_probability = iroas['probability'].iloc[0]

    # Must do it like this as the R value is given with lower number of dps.
    order_estimate = utils.float_order(r_estimate - py_estimate)
    order_precision = utils.float_order(r_precision - py_precision)
    order_lower = utils.float_order(r_lower - py_lower)
    order_iresp = utils.float_order(r_incr_resp - py_incr_resp)
    order_icost = utils.float_order(r_incr_cost - py_incr_cost)
    order_probability = utils.float_order(r_probability - py_probability)

    # Conduct the tests.
    self.assertLess(order_estimate, -5)
    self.assertLess(order_precision, -5)
    self.assertLess(order_lower, -5)
    self.assertLess(order_iresp, -2)  # incremental_response is a larger number.
    self.assertLess(order_icost, -5)
    self.assertLess(order_probability, -5)

  def testVariableCostIROASSummary(self, seed=1234):

    # Make behaviour deterministic.
    np.random.seed(seed=seed)

    # Fully set up a TBR object.
    iroas_model = tbr_iroas.TBRiROAS(use_cooldown=False)
    data = copy.copy(self.data)
    data.cost += 0.00001*np.random.normal(size=data.shape[0])

    iroas_model.fit(data,
                    key_response=self.key_response,
                    key_cost=self.key_cost,
                    key_group=self.key_group,
                    key_period=self.key_period,
                    key_date=self.key_date)

    # Arguments for the type of tests to conduct.
    level = 0.9
    posterior_threshold = 0.0
    tails = 1

    # Summary values from R, treated as constants.
    # pylint: disable=invalid-name
    R_ESTIMATE = 2.946742
    R_PRECISION = 0.120548
    R_LOWER = 2.826194
    R_INCR_RESP = 147337.122
    R_INCR_COST = 50000
    R_PROBABILITY = 1.0
    # pylint: enable=invalid-name

    # Summary values from python.
    iroas = iroas_model.summary(
        level=level, posterior_threshold=posterior_threshold, tails=tails)
    py_estimate = iroas['estimate'].iloc[0]
    py_precision = iroas['precision'].iloc[0]
    py_lower = iroas['lower'].iloc[0]
    py_incr_resp = iroas['incremental_response'].iloc[0]
    py_incr_cost = iroas['incremental_cost'].iloc[0]
    py_probability = iroas['probability'].iloc[0]

    # Must do it like this as the R value is given with lower number of dps.
    order_estimate = utils.float_order(R_ESTIMATE - py_estimate)
    order_precision = utils.float_order(R_PRECISION - py_precision)
    order_lower = utils.float_order(R_LOWER - py_lower)
    order_iresp = utils.float_order(R_INCR_RESP - py_incr_resp)
    order_icost = utils.float_order(R_INCR_COST - py_incr_cost)
    order_probability = utils.float_order(R_PROBABILITY - py_probability)

    # Conduct the tests. Easier threshold as we added some noise.
    self.assertLess(order_estimate, -2)
    self.assertLess(order_precision, -2)
    self.assertLess(order_lower, -2)
    self.assertLess(order_iresp, -2)  # incremental_response is a larger number.
    self.assertLess(order_icost, -2)
    self.assertLess(order_probability, -2)

  def testIROASSummaryWithCooldown(self):

    # Fully set up a TBR object.
    iroas_model = tbr_iroas.TBRiROAS(use_cooldown=True)
    iroas_model.fit(self.data,
                    key_response=self.key_response,
                    key_cost=self.key_cost,
                    key_group=self.key_group,
                    key_period=self.key_period,
                    key_date=self.key_date)

    # Arguments for the type of tests to conduct.
    level = 0.9
    posterior_threshold = 0.0
    tails = 1

    # Summary values from R, treated as constants.
    r_estimate = 2.946742

    # Summary values from python.
    iroas = iroas_model.summary(
        level=level, posterior_threshold=posterior_threshold, tails=tails)
    py_estimate = iroas['estimate'].iloc[0]

    # Must do it like this as the R value is given with lower number of dps.
    order_estimate = utils.float_order(r_estimate - py_estimate)

    # Conduct the tests. Easier threshold as we added some noise.
    self.assertLessEqual(order_estimate, -2)


if __name__ == '__main__':
  unittest.main()
