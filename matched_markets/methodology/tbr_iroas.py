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
"""Time Based Regression geoexperiment methodology for iROAS.
"""

from matched_markets.methodology import semantics
from matched_markets.methodology import utils
from matched_markets.methodology.tbr import TBR
import numpy as np
import pandas as pd


class TBRiROAS(object):
  """Time Based Regression geoexperiment methodology.

  This class estimates the incremental Return on Ad Spend (iROAS)
  of a geo experiment.
  For details see [Kerman 2017](https://ai.google/research/pubs/pub45950).
  """

  def __init__(self, use_cooldown=True):
    """Initializes a TBR iROAS analysis.

    Args:
      use_cooldown: bool. Whether to include the cooldown period in test.
    """
    self.df_names = None
    self.groups = None
    self.periods = None
    self.analysis_data = None
    self.target = None
    # Set up container for the response model, and potentially a cost model.
    self.tbr_cost = TBR(use_cooldown=use_cooldown)
    self.tbr_response = TBR(use_cooldown=use_cooldown)
    self.use_cooldown = use_cooldown

  def fit(self, data_frame, **kwargs):
    """Fit the TBRiROAS model to the supplied data frame.

    See optional kwargs for interpretation of the data frame.

    Args:
      data_frame: a pandas.DataFrame. Should contain the columns and indices
      corresponding to the **kwargs information below. Must be indexed by date.
      **kwargs: optional column/index names for the data and related semantics:
        key_geo='geo' - geo data frame index name.
        key_period='period' - experimental period column name.
        key_group='group' - group assignment column name.
        key_cost='cost' - cost column name.
        key_response='response' - response column name.
        key_date='date' - date index name.
        key_incr_cost='_incr_cost' - incremental cost column name.
        key_incr_response='_incr_response' - incremental response column name.
        group_control=1 - value representing the control group in the data.
        group_treat=2 - value representing the treatment group in the data.
        period_pre=0 - value representing the pre-test period in the data.
        period_test=1 - value representing the test period in the data.
        period_cool=2 - value representing the cooldown period in the data.
    """

    # Extract any column / index name information supplied by the user
    user_df_names = utils.kwarg_subdict('key_', **kwargs)
    self.df_names = semantics.DataFrameNameMapping(**user_df_names)

    # Extract any semantics for control / treatment supplied by user
    user_group_semantics = utils.kwarg_subdict('group_', **kwargs)
    self.groups = semantics.GroupSemantics(**user_group_semantics)

    # Extract any semantics for experimental period supplied by user
    user_period_semantics = utils.kwarg_subdict('period_', **kwargs)
    self.periods = semantics.PeriodSemantics(**user_period_semantics)

    # Fit seprate TBR models for response and cost
    self.tbr_response.fit(data_frame, self.df_names.response, **kwargs)
    self.tbr_cost.fit(data_frame, self.df_names.cost, **kwargs)

  def _is_fixed_cost_scenario(self):
    """Determines whether we are dealing with a fixed cost scenario.

    Returns:
      `bool`. True iff the sum of costs outside the treatment group in the test
        period is approx equal to zero.
    """

    # Access the relevant analysis data.
    adata = self.tbr_cost.analysis_data

    # Short names for important variables.
    key_cost = self.df_names.cost
    pre = self.periods.pre
    test = self.periods.test
    cntrl = self.groups.control

    # Get the cost data for the pre-period.
    pre_costs = adata.loc[adata[self.df_names.period] == pre][key_cost]
    # Get the cost data for the test-period for the control group.
    subset = adata.loc[adata[self.df_names.period] == test]
    test_costs_cntrl = subset.loc[cntrl][key_cost]
    # Sum of costs.
    tot_costs = sum(pre_costs) + sum(test_costs_cntrl)

    # Declare the costs to be zero if their sum is very small.
    tot_costs_order = utils.float_order(tot_costs)
    return tot_costs_order < -10

  def summary(self, level=0.9, posterior_threshold=0.0, tails=1, nsims=10000):
    """Estimates the control-treatment relationship in the pre- period.

    Args:
      level: `float` in (0,1). Determines width of CIs.
      posterior_threshold: `float`. Tests whether Delta is greater than
        posterior_threshold.
      tails: `int` in {1,2}. Specifies number of tails to use in tests.
      nsims: `int`. In the case of variable costs, this is the number of
        simulations to use.

    Returns:
      `pd.DataFrame`, a summary at level, with alpha=1-level, containing:
      - estimate. The median estimate of iROAS.
      - precision. Distance between the (1-level)/tails and 0.5 quantiles.
      - lower. The value of the (1-level)/tails quantile.
      - upper. If tails=2, the level/tails quantile, otherwise inf.
      - probability. The probability that Delta > posterior_threshold.
      - level. Records the level parameter used to generate the report.
      - posterior_threshold. Records the posterior_threshold parameter.
      - incremental_cost. The incremental cost over the test period.
      - incremental_response. The incremental response over the test period.

    Raises:
      ValueError: if tails is not 1 or 2.
    """

    if tails not in (1, 2):
      raise ValueError('tails must be 1 or 2.')

    # Define periods to credit to test.
    if self.use_cooldown:
      periods = (self.periods.test, self.periods.cooldown)
    else:
      periods = (self.periods.test,)

    # iROAS analysis assuming fixed costs.
    if self._is_fixed_cost_scenario():
      cost = np.sum(self.tbr_cost.causal_effect(periods))
      report = self.tbr_response.summary(rescale=1.0 / cost,
                                         tails=tails,
                                         level=level,
                                         threshold=posterior_threshold)
      report['incremental_cost'] = cost
      causal_effect = self.tbr_response.causal_effect(periods)
      report['incremental_response'] = np.sum(causal_effect)
      report['scenario'] = 'fixed'
      # Return the report, less the scale column.
      return report.drop('scale', axis=1)

    # iROAS analysis with variable costs modelled via TBR.
    else:
      alpha = (1 - level)/tails

      # Obtain the distributions of the two sets of causal effects
      delta_response = self.tbr_response.causal_cumulative_distribution(time=-1)
      # We know that causal costs only arose during the test period.
      delta_cost = self.tbr_cost.causal_cumulative_distribution(
          periods=(self.periods.test,),
          time=-1)

      # Simulate the iROAS
      sims_cost = delta_cost.rvs(nsims)
      sims_response = delta_response.rvs(nsims)
      sims_iroas = sims_response / sims_cost

      # This needs to be used twice.
      ci_lower = np.percentile(sims_iroas, 100 * alpha)

      # Construct the report.
      causal_effect = self.tbr_cost.causal_effect(periods)
      report = pd.DataFrame(index=[causal_effect.index[-1]])
      report.index.name = 'date'
      report['estimate'] = np.mean(sims_iroas)
      report['precision'] = report['estimate'] - ci_lower
      report['lower'] = ci_lower
      if tails == 1:
        report['upper'] = np.inf
      else:
        report['upper'] = np.percentile(sims_iroas, 100 * (1 - alpha))
      report['probability'] = np.mean(sims_iroas > posterior_threshold)
      report['level'] = level
      report['posterior_threshold'] = posterior_threshold
      report['incremental_cost'] = delta_cost.kwds['loc']
      report['incremental_response'] = delta_response.kwds['loc']
      report['scenario'] = 'variable'

      return report
