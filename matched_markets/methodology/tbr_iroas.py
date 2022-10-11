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

from matched_markets.methodology import common_classes
from matched_markets.methodology import semantics
from matched_markets.methodology import utils
from matched_markets.methodology.tbr import TBR
import numpy as np
import pandas as pd


class TBRiROAS():
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

  def summary(self,
              level=0.9,
              posterior_threshold=0.0,
              tails=1,
              nsims=10000,
              random_state=None):
    """Estimates the control-treatment relationship in the pre- period.

    Args:
      level: `float` in (0,1). Determines width of CIs.
      posterior_threshold: `float`. Tests whether Delta is greater than
        posterior_threshold.
      tails: `int` in {1,2}. Specifies number of tails to use in tests.
      nsims: `int`. In the case of variable costs, this is the number of
        simulations to use.
      random_state: the random_state for the RNG to fix simulation results.

    Returns:
      `pd.DataFrame`, a summary at level, with alpha=1-level, containing:
      - estimate. The median estimate of iROAS.
      - precision. Distance between the (1-level)/tails and 0.5 quantiles.
      - lower. The value of the (1-level)/tails quantile.
      - upper. If tails=2, the 1 - 0.5 * (1 - level) quantile, otherwise inf.
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

    tail_probability = (1 - level) / tails

    response_data = self.tbr_response.analysis_data.copy().reset_index()
    observed_treatment_response = response_data.loc[
        (response_data[self.df_names.group] == self.groups.treatment) &
        (response_data['period'].isin(periods)), self.df_names.response].sum()

    # Obtain the distributions of the causal effects on response
    delta_response = self.tbr_response.causal_cumulative_distribution(time=-1)
    # Simulate the incremental response and relative lift
    sims_response = delta_response.rvs(nsims, random_state=random_state)
    sims_relative_lift = sims_response / (
        observed_treatment_response - sims_response)
    relative_lift = np.median(sims_relative_lift)
    relative_lift_lower = np.percentile(sims_relative_lift,
                                        100 * tail_probability)
    if tails == 1:
      relative_lift_upper = np.inf
    else:
      relative_lift_upper = np.percentile(sims_relative_lift,
                                          100 * (1.0 - tail_probability))

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
      report['incremental_response_lower'] = report['lower'] * cost
      report['incremental_response_upper'] = report['upper'] * cost
      report['relative_lift'] = relative_lift
      report['relative_lift_lower'] = relative_lift_lower
      report['relative_lift_upper'] = relative_lift_upper
      report['scenario'] = 'fixed'
      # Return the report, less the scale column.
      return report.drop('scale', axis=1)

    # iROAS analysis with variable costs modelled via TBR.
    else:

      # Obtain the distributions of the causal effects on cost
      # We know that causal costs only arose during the test period.
      delta_cost = self.tbr_cost.causal_cumulative_distribution(
          periods=(self.periods.test,),
          time=-1)

      # Simulate the iROAS
      sims_cost = delta_cost.rvs(nsims, random_state=random_state)
      sims_iroas = sims_response / sims_cost

      ci_lower = np.percentile(sims_iroas, 100 * tail_probability)

      # Construct the report.
      causal_effect = self.tbr_cost.causal_effect(periods)
      report = pd.DataFrame(index=[causal_effect.index[-1]])
      report.index.name = 'date'
      report['estimate'] = np.mean(sims_iroas)
      report['precision'] = report['estimate'] - ci_lower
      report['lower'] = ci_lower
      if tails == 1:
        report['upper'] = np.inf
        report['incremental_response_upper'] = np.inf
      else:
        report['upper'] = np.percentile(sims_iroas,
                                        100 * (1 - tail_probability))
        report['incremental_response_upper'] = delta_response.ppf(
            1 - tail_probability)
      report['probability'] = np.mean(sims_iroas > posterior_threshold)
      report['level'] = level
      report['posterior_threshold'] = posterior_threshold
      report['incremental_cost'] = delta_cost.kwds['loc']
      report['incremental_response'] = delta_response.kwds['loc']
      report['incremental_response_lower'] = delta_response.ppf(
          tail_probability)
      report['relative_lift'] = relative_lift
      report['relative_lift_lower'] = relative_lift_lower
      report['relative_lift_upper'] = relative_lift_upper
      report['scenario'] = 'variable'

      return report

  def estimate_pointwise_and_cumulative_effect(
      self,
      metric: str,
      level: float = 0.9,
      tails: int = 1
  ) -> common_classes.TimeSeries:
    """Estimates the pointwise and cumulative effect.

    This function estimates the pointwise and cumulative effect on a metric
    with confidence intervals.

    If the cost is fixed as defined in _is_fixed_cost_scenario(), then the
    counterfactual cost would be 0, and the pointwise difference is equal to the
    observed cost.

    Example usage:
      iroas_model = tbr_iroas.TBRiROAS(use_cooldown=True)
      iroas_model.fit(data)
      iroas_model.estimate_pointwise_and_cumulative_effect(
        metric='tbr_response')

    Args:
      metric: variable for which we want to compute the pointwise and cumulative
        effect. It should be one of tbr_response or tbr_cost.
      level: `float` in (0,1). Determines width of CIs.
      tails: `int` in {1,2}. Specifies number of tails to use in tests.

    Returns:
      counterfactual_df: a time series with the counterfactual estimate of the
        metric and its pointwise confidence bounds.
        The dataframe has columns (date, estimate, lower, upper).
      pointwise_difference_df: a time series with the pointwise difference
        between observed and counterfactual metric, with confidence interval
        bounds. The dataframe has columns (date, estimate, lower, upper).
      cumulative_effect_df: a time series with the cumulative difference between
        observed and counterfactual metric, with confidence interval bounds.
        The dataframe has columns (date, estimate, lower, upper).
    Raises:
      ValueError:
        - if tails is not 1 or 2.
        - if the metric is not one of [tbr_response, tbr_cost].
        - if fit was not called previously.
        - if fit was called with use_cooldown=False.
    """
    if tails not in (1, 2):
      raise ValueError('tails must be 1 or 2.')

    if self.periods is None:
      raise ValueError('The method "fit()" has not been called.')

    if not self.use_cooldown:
      raise ValueError('The method "fit()" must have been called with ' +
                       'use_cooldown=True.')

    if metric == 'tbr_response':
      metric_df = self.tbr_response
      metric_col = self.df_names.response
    elif metric == 'tbr_cost':
      metric_df = self.tbr_cost
      metric_col = self.df_names.cost
    else:
      raise ValueError('The metric must be one of tbr_response or tbr_cost, ' +
                       f'got {metric}')

    periods = (self.periods.pre, self.periods.test, self.periods.cooldown)
    tail_probability = (1 - level) / tails

    metric_data = metric_df.analysis_data.copy().reset_index()

    dates = metric_data.loc[metric_data['period'].isin(periods),
                            'date'].unique()
    experiment_dates = metric_data.loc[
        metric_data['period'].isin([self.periods.test, self.periods.cooldown]),
        'date'].unique()

    if self._is_fixed_cost_scenario() and metric == 'tbr_cost':
      tmp_data = metric_data[metric_data[self.df_names.group] ==
                             self.groups.treatment].reset_index(drop=True)
      counterfactual_df = common_classes.EstimatedTimeSeriesWithConfidenceInterval(
          {
              'date': dates,
              'lower': 0,
              'upper': 0,
              'estimate': 0,
          })
      pointwise_difference_df = common_classes.EstimatedTimeSeriesWithConfidenceInterval(
          {
              'date': dates,
              'lower': tmp_data['cost'],
              'upper': tmp_data['cost'],
              'estimate': tmp_data['cost'],
          })

      cumulative_effect = np.cumsum(
          tmp_data.loc[tmp_data['date'].isin(experiment_dates), 'cost'])
      cumulative_effect_df = common_classes.EstimatedTimeSeriesWithConfidenceInterval(
          {
              'date': experiment_dates,
              'lower': cumulative_effect,
              'upper': cumulative_effect,
              'estimate': cumulative_effect,
          })

    else:
      test_start_date = experiment_dates.min()
      cooldown_end_date = experiment_dates.max()

      delta_metric = metric_df.causal_cumulative_distribution()
      pointwise_difference = metric_df.causal_effect(
          periods).reset_index().rename(columns={0: 'metric'})
      lower = np.diff(delta_metric.ppf(tail_probability), prepend=0)
      lower = np.concatenate((pointwise_difference.loc[
          pointwise_difference['date'] < test_start_date,
          'metric'].values, lower))
      upper = np.diff(delta_metric.ppf(1 - tail_probability), prepend=0)
      upper = np.concatenate((pointwise_difference.loc[
          pointwise_difference['date'] < test_start_date,
          'metric'].values, upper))
      # Get the test- period data in the form needed for regression.
      treat_vec = metric_data.loc[
          metric_data[self.df_names.group] == self.groups.treatment,
          metric_col].reset_index(drop=True)

      counterfactual_df = common_classes.EstimatedTimeSeriesWithConfidenceInterval(
          {
              'date': dates,
              'lower': treat_vec - upper,
              'upper': treat_vec - lower,
              'estimate': treat_vec - pointwise_difference['metric']
          })
      pointwise_difference_df = common_classes.EstimatedTimeSeriesWithConfidenceInterval(
          {
              'date': dates,
              'lower': lower,
              'upper': upper,
              'estimate': pointwise_difference['metric']
          })

      cumulative_effect = np.cumsum(
          metric_df.causal_effect(periods)).reset_index().rename(
              columns={0: 'metric'})
      cumulative_effect = cumulative_effect.loc[
          cumulative_effect['date'].between(test_start_date, cooldown_end_date),
          'metric']
      cumulative_effect_df = common_classes.EstimatedTimeSeriesWithConfidenceInterval(
          {
              'date': experiment_dates,
              'lower': delta_metric.ppf(tail_probability),
              'upper': delta_metric.ppf(1 - tail_probability),
              'estimate': cumulative_effect
          })

    return common_classes.TimeSeries(counterfactual_df, pointwise_difference_df,
                                     cumulative_effect_df)
