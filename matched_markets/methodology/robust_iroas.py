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
"""Robust TBR geoexperiment methodology for iROAS."""

import copy
from matched_markets.methodology import base
from matched_markets.methodology import common_classes
from matched_markets.methodology import robust


class RobustTBRiROAS(base.BaseReporting):
  """Robust Time Based Regression geoexperiment methodology."""

  def initialize_regression_model(self):
    """Initializes the parameteric TBR regression model."""
    self.tbr_response = robust.RobustTBR(use_cooldown=self.use_cooldown)
    self.tbr_cost = robust.RobustTBR(use_cooldown=self.use_cooldown)

  def summary(self, level=0.8, tails=2, equalize_tbr_slope=True, **kwargs):
    """Estimates the control-treatment relationship in the post- period.

    Args:
      level: `float` in (0,1). Determines width of CIs.
      tails: `int`, specifies number of tails to use in tests. Currently only 2
        tails are supported.
      equalize_tbr_slope: `bool`. If True, the slope of the TBR model is
        equalized between the response and cost models.
      **kwargs: Additional arguments to pass to RobustTBR instances.

    Returns:
      `pd.DataFrame`, a summary of cumulative causal effect until the most
        recent available date, containing 7 columns:
      - estimate. The median estimate of iROAS.
      - lower. The value of the (1-level)/tails quantile.
      - upper. If tails=2, the 1 - 0.5 * (1 - level) quantile, otherwise inf.
      - level. Records the level parameter used to generate the report.
      - incremental_cost. The incremental cost over the test period.
      - incremental_response. The incremental response over the test period.
      - scenario. The scenario of the study cost, can be either 'fixed' or
        'variable'.

    Raises:
      NotImplementedError: if tails is not 2.
      RuntimeError: if the cumulative incremental cost is not positive under
        a fixed cost scenario or is not negative under a variable cost scenario.
    """
    if tails != 2:
      raise NotImplementedError(
          "Only two-tailed confidence interval is supported."
      )
    inital_cost_model = copy.deepcopy(self.tbr_cost.pre_period_model)
    if equalize_tbr_slope:
      self.tbr_cost.pre_period_model = self.tbr_response.pre_period_model
    # Summarizes the incremental response from the TBR response model.
    response_summary = self.tbr_response.summary(
        level=level, tails=tails, **kwargs
    )
    response_summary["incremental_response"] = response_summary["estimate"]
    # Estimates the cumulative incremental cost during the post-period.
    total_cost = self.tbr_cost.cumulative_causal_effect()
    response_summary["incremental_cost"] = total_cost
    # Updates with the iROAS estimate.
    response_summary["estimate"] = (
        response_summary["incremental_response"] / total_cost
    )

    if not self._is_fixed_cost_scenario():
      if total_cost >= 0:
        raise RuntimeError("The cumulative incremental cost is not negative.")
      response_summary["scenario"] = "variable"
      # Updates with the iROAS confidence interval.
      response_summary["lower"], response_summary["upper"] = (
          response_summary["upper"] / total_cost,  # total_cost is negative.
          response_summary["lower"] / total_cost,
      )
    else:
      if total_cost <= 0:
        raise RuntimeError("The cumulative incremental cost is not positive.")
      response_summary["scenario"] = "fixed"
      # Updates with the iROAS confidence interval.
      response_summary["lower"] = response_summary["lower"] / total_cost
      response_summary["upper"] = response_summary["upper"] / total_cost

    # Resets the cost model to its original state.
    self.tbr_cost.pre_period_model = inital_cost_model

    return response_summary

  def estimate_pointwise_and_cumulative_effect(
      self, metric: str, **kwargs
  ) -> common_classes.TimeSeries:
    """Estimates the pointwise and cumulative effect for the target metric."""
    raise NotImplementedError
