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
  """Class for testing tbr_iroas."""

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
    # Fully set up a TBR object.
    self.iroas_model = tbr_iroas.TBRiROAS(use_cooldown=False)
    self.iroas_model.fit(
        self.data,
        key_response=self.key_response,
        key_cost=self.key_cost,
        key_group=self.key_group,
        key_period=self.key_period,
        key_date=self.key_date)
    self.treatment_response = self.data.loc[(self.data[self.key_group] == 2) & (
        self.data[self.key_period] == 1), self.key_response].sum()

  def testFixedCostIROASSummary(self):
    """Checks the TBR results for an holdback experiment."""
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
    r_incr_response_lower = r_lower * r_incr_cost
    r_lift = 0.173998
    r_lift_lower = 0.165686

    # Summary values from python.
    iroas = self.iroas_model.summary(
        level=level, posterior_threshold=posterior_threshold, tails=tails)
    py_estimate = iroas['estimate'].iloc[0]
    py_precision = iroas['precision'].iloc[0]
    py_lower = iroas['lower'].iloc[0]
    py_upper = iroas['upper'].iloc[0]
    py_incr_resp = iroas['incremental_response'].iloc[0]
    py_incr_cost = iroas['incremental_cost'].iloc[0]
    py_probability = iroas['probability'].iloc[0]
    py_incr_resp_lower = iroas['incremental_response_lower'].iloc[0]
    py_incr_resp_upper = iroas['incremental_response_upper'].iloc[0]
    py_lift = iroas['relative_lift'].iloc[0]
    py_lift_lower = iroas['relative_lift_lower'].iloc[0]
    py_lift_upper = iroas['relative_lift_upper'].iloc[0]

    # Must do it like this as the R value is given with lower number of dps.
    order_estimate = utils.float_order(r_estimate - py_estimate)
    order_precision = utils.float_order(r_precision - py_precision)
    order_lower = utils.float_order(r_lower - py_lower)
    order_iresp = utils.float_order(r_incr_resp - py_incr_resp)
    order_icost = utils.float_order(r_incr_cost - py_incr_cost)
    order_probability = utils.float_order(r_probability - py_probability)
    order_iresp_lower = utils.float_order(r_incr_response_lower -
                                          py_incr_resp_lower)
    order_lift = utils.float_order(r_lift - py_lift)
    order_lift_lower = utils.float_order(r_lift_lower - py_lift_lower)
    # Conduct the tests.
    self.assertLess(order_estimate, -5)
    self.assertLess(order_precision, -5)
    self.assertLess(order_lower, -5)
    self.assertEqual(py_upper, np.inf)
    self.assertLess(order_iresp, -2)  # incremental_response is a larger number.
    self.assertLess(order_icost, -5)
    self.assertLess(order_probability, -5)
    self.assertLessEqual(order_iresp_lower, -2)
    self.assertEqual(py_incr_resp_upper, np.inf)
    self.assertLessEqual(order_lift, -4)
    self.assertLessEqual(order_lift_lower, -4)
    self.assertEqual(py_lift_upper, np.inf)

  def testVariableCostIROASSummary(self):
    """Checks the TBR results for an go-dark/heavy-up experiment."""

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
    r_estimate = 2.946742
    r_precision = 0.120548
    r_lower = 2.826194
    r_incr_resp = 147337.122
    r_incr_cost = 50000
    r_probability = 1.0
    r_incr_resp_lower = r_lower * r_incr_cost
    r_lift = 0.173998
    r_lift_lower = 0.165686

    # Summary values from python. Specify random_state to make results
    # deterministic.
    iroas = iroas_model.summary(
        level=level,
        posterior_threshold=posterior_threshold,
        tails=tails,
        random_state=np.random.RandomState(1234))
    py_estimate = iroas['estimate'].iloc[0]
    py_precision = iroas['precision'].iloc[0]
    py_lower = iroas['lower'].iloc[0]
    py_upper = iroas['upper'].iloc[0]
    py_incr_resp = iroas['incremental_response'].iloc[0]
    py_incr_cost = iroas['incremental_cost'].iloc[0]
    py_probability = iroas['probability'].iloc[0]
    py_incr_resp_lower = iroas['incremental_response_lower'].iloc[0]
    py_incr_resp_upper = iroas['incremental_response_upper'].iloc[0]
    py_lift = iroas['relative_lift'].iloc[0]
    py_lift_lower = iroas['relative_lift_lower'].iloc[0]
    py_lift_upper = iroas['relative_lift_upper'].iloc[0]

    # Must do it like this as the R value is given with lower number of dps.
    order_estimate = utils.float_order(r_estimate - py_estimate)
    order_precision = utils.float_order(r_precision - py_precision)
    order_lower = utils.float_order(r_lower - py_lower)
    self.assertEqual(py_upper, np.inf)
    order_iresp = utils.float_order(r_incr_resp - py_incr_resp)
    order_icost = utils.float_order(r_incr_cost - py_incr_cost)
    order_probability = utils.float_order(r_probability - py_probability)
    order_iresp_lower = utils.float_order(r_incr_resp_lower -
                                          py_incr_resp_lower)
    order_lift = utils.float_order(r_lift - py_lift)
    order_lift_lower = utils.float_order(r_lift_lower - py_lift_lower)
    # Conduct the tests. Easier threshold as we added some noise.
    self.assertLess(order_estimate, -2)
    self.assertLess(order_precision, -2)
    self.assertLess(order_lower, -2)
    self.assertLess(order_iresp, -2)  # incremental_response is a larger number.
    self.assertLess(order_icost, -2)
    self.assertLess(order_probability, -2)
    self.assertLessEqual(order_iresp_lower, -2)
    self.assertEqual(py_incr_resp_upper, np.inf)
    self.assertLessEqual(order_lift, -4)
    self.assertLessEqual(order_lift_lower, -4)
    self.assertEqual(py_lift_upper, np.inf)

  def testVariableCostIROASSummaryTwoTails(self):
    """Checks the TBR results when reporting two sided CI."""

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
    tails = 2

    # Summary values from R, treated as constants.
    r_estimate = 2.947012
    r_precision = 0.1557932
    r_lower = 2.79135
    r_upper = 3.102936
    r_incr_resp = 147337.122
    r_incr_cost = 50000
    r_probability = 1.0
    r_incr_resp_lower = r_lower * r_incr_cost
    r_incr_resp_upper = r_upper * r_incr_cost
    r_lift = 0.173998
    r_lift_lower = 0.163273
    r_lift_upper = 0.184885

    # Summary values from python. Specify random_state to make results
    # deterministic.
    iroas = iroas_model.summary(
        level=level,
        posterior_threshold=posterior_threshold,
        tails=tails,
        random_state=np.random.RandomState(1234))
    py_estimate = iroas['estimate'].iloc[0]
    py_precision = iroas['precision'].iloc[0]
    py_lower = iroas['lower'].iloc[0]
    py_upper = iroas['upper'].iloc[0]
    py_incr_resp = iroas['incremental_response'].iloc[0]
    py_incr_cost = iroas['incremental_cost'].iloc[0]
    py_probability = iroas['probability'].iloc[0]
    py_incr_resp_lower = iroas['incremental_response_lower'].iloc[0]
    py_incr_resp_upper = iroas['incremental_response_upper'].iloc[0]
    py_lift = iroas['relative_lift'].iloc[0]
    py_lift_lower = iroas['relative_lift_lower'].iloc[0]
    py_lift_upper = iroas['relative_lift_upper'].iloc[0]

    print(r_lift_lower)
    print(py_lift_lower)
    # Must do it like this as the R value is given with lower number of dps.
    order_estimate = utils.float_order(r_estimate - py_estimate)
    order_precision = utils.float_order(r_precision - py_precision)
    order_lower = utils.float_order(r_lower - py_lower)
    order_upper = utils.float_order(r_upper - py_upper)
    order_iresp = utils.float_order(r_incr_resp - py_incr_resp)
    order_icost = utils.float_order(r_incr_cost - py_incr_cost)
    order_probability = utils.float_order(r_probability - py_probability)
    # Use relative error for incremental response due to different RNG in R and
    # Python
    order_iresp_lower = utils.float_order(
        (r_incr_resp_lower - py_incr_resp_lower) * 100 / r_incr_resp_lower)
    order_iresp_upper = utils.float_order(
        (r_incr_resp_upper - py_incr_resp_upper) * 100 / r_incr_resp_upper)
    order_lift = utils.float_order(r_lift - py_lift)
    order_lift_lower = utils.float_order(r_lift_lower - py_lift_lower)
    order_lift_upper = utils.float_order(r_lift_upper - py_lift_upper)

    # Conduct the tests. Easier threshold as we added some noise.
    self.assertLess(order_estimate, -2)
    self.assertLess(order_precision, -2)
    self.assertLess(order_lower, -2)
    self.assertLess(order_upper, -2)
    self.assertLess(order_iresp, -2)  # incremental_response is a larger number.
    self.assertLess(order_icost, -2)
    self.assertLess(order_probability, -2)
    self.assertLessEqual(order_iresp_lower, -2)
    self.assertLessEqual(order_iresp_upper, -2)
    self.assertLessEqual(order_lift, -4)
    self.assertLessEqual(order_lift_lower, -4)
    self.assertLessEqual(order_lift_upper, -4)

  def testIROASSummaryWithCooldown(self):
    """Checks the TBR results when including the cooldown period in the analysis."""
    # Arguments for the type of tests to conduct.
    level = 0.9
    posterior_threshold = 0.0
    tails = 1

    # Summary values from R, treated as constants.
    r_estimate = 2.946742

    # Summary values from python.
    iroas = self.iroas_model.summary(
        level=level, posterior_threshold=posterior_threshold, tails=tails)
    py_estimate = iroas['estimate'].iloc[0]

    # Must do it like this as the R value is given with lower number of dps.
    order_estimate = utils.float_order(r_estimate - py_estimate)

    # Conduct the tests. Easier threshold as we added some noise.
    self.assertLessEqual(order_estimate, -2)

  def testPointwiseDifferenceTailError(self):
    """Checks that the correct number of tails are specified."""
    with self.assertRaisesRegex(ValueError, r'tails must be 1 or 2.'):
      self.iroas_model.estimate_pointwise_and_cumulative_effect(
          metric='tbr_response', level=0.9, tails=3)

  def testPointwiseDifferenceMetricError(self):
    """Checks that the correct metric is specified."""
    iroas_model = tbr_iroas.TBRiROAS(use_cooldown=True)
    iroas_model.fit(
        self.data,
        key_response=self.key_response,
        key_cost=self.key_cost,
        key_group=self.key_group,
        key_period=self.key_period,
        key_date=self.key_date)
    with self.assertRaisesRegex(
        ValueError, r'The metric must be one of ' +
        'tbr_response or tbr_cost, ' + 'got unknown'):
      iroas_model.estimate_pointwise_and_cumulative_effect(
          metric='unknown', level=0.9, tails=1)

  def testPointwiseDifferenceFitNotCalled(self):
    """Checks that the method fit has been called before."""
    iroas_model = tbr_iroas.TBRiROAS(use_cooldown=False)
    with self.assertRaisesRegex(ValueError,
                                r'The method "fit\(\)" has not been called.'):
      iroas_model.estimate_pointwise_and_cumulative_effect(
          metric='tbr_response', level=0.9, tails=1)

  def testPointwiseDifferenceFitCalledWithoutCooldown(self):
    """Checks that the method fit has been called with use_cooldown=True."""
    iroas_model = tbr_iroas.TBRiROAS(use_cooldown=False)
    iroas_model.fit(
        self.data,
        key_response=self.key_response,
        key_cost=self.key_cost,
        key_group=self.key_group,
        key_period=self.key_period,
        key_date=self.key_date)
    with self.assertRaisesRegex(ValueError,
                                r'The method "fit\(\)" must have been called ' +
                                'with use_cooldown=True.'):
      iroas_model.estimate_pointwise_and_cumulative_effect(
          metric='tbr_response', level=0.9, tails=1)

  def testPointwiseDifferenceFixedCost(self):
    """Checks that the pointwise effect is correct when cost is fixed."""
    iroas_model = tbr_iroas.TBRiROAS(use_cooldown=True)
    iroas_model.fit(
        self.data,
        key_response=self.key_response,
        key_cost=self.key_cost,
        key_group=self.key_group,
        key_period=self.key_period,
        key_date=self.key_date)
    time_series = (
        iroas_model.estimate_pointwise_and_cumulative_effect(
            metric='tbr_response', level=0.8, tails=2))
    metric_data = iroas_model.tbr_response.analysis_data.copy().reset_index()

    counterfactual_df = time_series.counterfactual
    pointwise_response_df = time_series.pointwise_difference
    pretest_start_date = metric_data.loc[
        metric_data['period'] == iroas_model.periods.pre, 'date'].min()
    test_start_date = metric_data.loc[
        metric_data['period'] == iroas_model.periods.test, 'date'].min()

    # checks for response
    tmp_data = self.data.groupby(['date', 'geo.group', 'period'],
                                 as_index=False).sum()
    observed_value = tmp_data.loc[tmp_data['geo.group'] == 2, 'sales'].values
    expected_difference = observed_value - counterfactual_df['estimate'].values
    self.assertAlmostEqual(counterfactual_df['estimate'].values[0],
                           29255.57829994382)
    self.assertAlmostEqual(counterfactual_df['lower'].values[0],
                           29255.57829994382)
    self.assertAlmostEqual(counterfactual_df['upper'].values[0],
                           29255.57829994382)
    test_counterfactual = counterfactual_df[counterfactual_df['date'] ==
                                            test_start_date]
    self.assertAlmostEqual(test_counterfactual['lower'].values[0],
                           34302.50979221002)
    self.assertAlmostEqual(test_counterfactual['upper'].values[0],
                           35918.87409473002)
    self.assertAlmostEqual(test_counterfactual['estimate'].values[0],
                           35110.69194347361)
    self.assertTrue(
        np.allclose(pointwise_response_df['estimate'].values,
                    expected_difference))
    pretest_data = pointwise_response_df[pointwise_response_df['date'] ==
                                         pretest_start_date]
    test_data = pointwise_response_df[pointwise_response_df['date'] ==
                                      test_start_date]
    cumulative_data = time_series.cumulative_effect[
        time_series.cumulative_effect['date'] == test_start_date]
    self.assertTrue(
        np.allclose(pretest_data['lower'],
                    pretest_data['estimate']))
    self.assertTrue(
        np.allclose(pretest_data['upper'],
                    pretest_data['estimate']))
    self.assertAlmostEqual(test_data['lower'].values[0], 4750.63590527)
    self.assertAlmostEqual(test_data['upper'].values[0], 6367.00020779)
    self.assertAlmostEqual(cumulative_data['lower'].values[0], 4750.63590527)
    self.assertAlmostEqual(cumulative_data['upper'].values[0], 6367.00020779)
    self.assertAlmostEqual(cumulative_data['estimate'].values[0],
                           5558.818056526405)
    self.assertAlmostEqual(
        time_series.cumulative_effect['estimate'].values[-1],
        sum(expected_difference))

    # checks for cost
    time_series_cost = (
        iroas_model.estimate_pointwise_and_cumulative_effect(
            metric='tbr_cost', level=0.8, tails=2))
    metric_data = iroas_model.tbr_cost.analysis_data.copy().reset_index()
    counterfactual_df = time_series_cost.counterfactual
    pointwise_cost_df = time_series_cost.pointwise_difference
    cumulative_cost_df = time_series_cost.cumulative_effect

    observed_value = tmp_data.loc[tmp_data['geo.group'] == 2, 'cost'].values
    self.assertTrue(np.allclose(counterfactual_df['estimate'].values, 0))
    self.assertTrue(np.allclose(counterfactual_df['lower'].values, 0))
    self.assertTrue(np.allclose(counterfactual_df['upper'].values, 0))
    self.assertTrue(
        np.allclose(pointwise_cost_df['estimate'].values,
                    observed_value))
    cumulative_data = cumulative_cost_df[cumulative_cost_df['date'] ==
                                         test_start_date]
    self.assertTrue(
        np.allclose(pointwise_cost_df['lower'].values, observed_value))
    self.assertTrue(
        np.allclose(pointwise_cost_df['upper'].values, observed_value))
    self.assertAlmostEqual(
        cumulative_cost_df['estimate'].values[-1],
        sum(observed_value))
    self.assertAlmostEqual(cumulative_cost_df['lower'].values[-1],
                           sum(observed_value))
    self.assertAlmostEqual(cumulative_cost_df['upper'].values[-1],
                           sum(observed_value))

  def testPointwiseDifferenceVariableCost(self, seed=1234):
    """Checks that the pointwise effect is correct when cost is variable."""
    # Make behaviour deterministic.
    np.random.seed(seed=seed)
    # Fully set up a TBR object.
    iroas_model = tbr_iroas.TBRiROAS(use_cooldown=True)
    data = copy.copy(self.data)
    data.cost += 0.1*np.random.normal(size=data.shape[0])
    iroas_model.fit(data,
                    key_response=self.key_response,
                    key_cost=self.key_cost,
                    key_group=self.key_group,
                    key_period=self.key_period,
                    key_date=self.key_date)
    time_series = (
        iroas_model.estimate_pointwise_and_cumulative_effect(
            metric='tbr_response', level=0.8, tails=2))
    metric_data = iroas_model.tbr_response.analysis_data.copy().reset_index()

    counterfactual_df = time_series.counterfactual
    pointwise_response_df = time_series.pointwise_difference

    pretest_start_date = metric_data.loc[
        metric_data['period'] == iroas_model.periods.pre, 'date'].min()
    test_start_date = metric_data.loc[
        metric_data['period'] == iroas_model.periods.test, 'date'].min()

    # checks for response
    tmp_data = data.groupby(['date', 'geo.group', 'period'],
                            as_index=False).sum()
    observed_value = tmp_data.loc[tmp_data['geo.group'] == 2, 'sales'].values
    expected_difference = observed_value - counterfactual_df['estimate'].values
    self.assertAlmostEqual(counterfactual_df['estimate'].values[0],
                           29255.57829994382)
    self.assertAlmostEqual(counterfactual_df['lower'].values[0],
                           29255.57829994382)
    self.assertAlmostEqual(counterfactual_df['upper'].values[0],
                           29255.57829994382)
    test_counterfactual = counterfactual_df[counterfactual_df['date'] ==
                                            test_start_date]
    self.assertAlmostEqual(test_counterfactual['lower'].values[0],
                           34302.50979221002)
    self.assertAlmostEqual(test_counterfactual['upper'].values[0],
                           35918.87409473002)
    self.assertAlmostEqual(test_counterfactual['estimate'].values[0],
                           35110.69194347361)
    self.assertTrue(
        np.allclose(pointwise_response_df['estimate'].values,
                    expected_difference))
    pretest_data = pointwise_response_df[pointwise_response_df['date'] ==
                                         pretest_start_date]
    test_data = pointwise_response_df[pointwise_response_df['date'] ==
                                      test_start_date]
    cumulative_data = time_series.cumulative_effect[
        time_series.cumulative_effect['date'] == test_start_date]
    self.assertTrue(
        np.allclose(pretest_data['lower'],
                    pretest_data['estimate']))
    self.assertTrue(
        np.allclose(pretest_data['upper'],
                    pretest_data['estimate']))
    self.assertAlmostEqual(test_data['lower'].values[0], 4750.63590527)
    self.assertAlmostEqual(test_data['upper'].values[0], 6367.00020779)
    self.assertAlmostEqual(cumulative_data['lower'].values[0], 4750.63590527)
    self.assertAlmostEqual(cumulative_data['upper'].values[0], 6367.00020779)
    self.assertAlmostEqual(cumulative_data['estimate'].values[0],
                           5558.818056526405)
    self.assertAlmostEqual(
        time_series.cumulative_effect['estimate'].values[-1],
        sum(expected_difference))

    # checks for cost
    time_series_cost = (
        iroas_model.estimate_pointwise_and_cumulative_effect(
            metric='tbr_cost', level=0.8, tails=2))
    metric_data = iroas_model.tbr_cost.analysis_data.copy().reset_index()
    counterfactual_df = time_series_cost.counterfactual
    pointwise_cost_df = time_series_cost.pointwise_difference
    cumulative_cost_df = time_series_cost.cumulative_effect

    observed_value = tmp_data.loc[tmp_data['geo.group'] == 2, 'cost'].values
    expected_difference = observed_value - counterfactual_df['estimate'].values
    self.assertAlmostEqual(counterfactual_df['estimate'].values[0],
                           0.08533087113252846)
    self.assertAlmostEqual(counterfactual_df['lower'].values[0],
                           0.08533087113252846)
    self.assertAlmostEqual(counterfactual_df['upper'].values[0],
                           0.08533087113252846)
    test_counterfactual = counterfactual_df[counterfactual_df['date'] ==
                                            test_start_date]
    self.assertAlmostEqual(test_counterfactual['lower'].values[0],
                           -0.779657713103461)
    self.assertAlmostEqual(test_counterfactual['upper'].values[0],
                           0.8595944785759002)
    self.assertAlmostEqual(test_counterfactual['estimate'].values[0],
                           0.0399683827363333)
    self.assertTrue(
        np.allclose(pointwise_cost_df['estimate'].values,
                    expected_difference))
    pretest_data = pointwise_cost_df[pointwise_cost_df['date'] ==
                                     pretest_start_date]
    test_data = pointwise_cost_df[pointwise_cost_df['date'] == test_start_date]
    cumulative_data = cumulative_cost_df[cumulative_cost_df['date'] ==
                                         test_start_date]
    self.assertTrue(
        np.allclose(pretest_data['lower'], pretest_data['estimate']))
    self.assertTrue(
        np.allclose(pretest_data['upper'], pretest_data['estimate']))
    self.assertAlmostEqual(test_data['lower'].values[0], 2046.944579665883)
    self.assertAlmostEqual(test_data['upper'].values[0], 2048.5838318575625)
    self.assertAlmostEqual(cumulative_data['lower'].values[0],
                           2046.944579665883)
    self.assertAlmostEqual(cumulative_data['upper'].values[0],
                           2048.5838318575625)
    self.assertAlmostEqual(cumulative_data['estimate'].values[0],
                           2047.7642057617227)
    self.assertAlmostEqual(cumulative_cost_df['estimate'].values[-1],
                           sum(expected_difference))


if __name__ == '__main__':
  unittest.main()
