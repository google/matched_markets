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
"""Base classes for time-based regression methodology."""

import abc
import collections
import functools
from typing import Generic, TypeVar

from matched_markets.methodology import common_classes
from matched_markets.methodology import semantics
from matched_markets.methodology import utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.regression import linear_model


class BaseRegression(metaclass=abc.ABCMeta):
  """Base class for Time-Based Regression (TBR) models."""

  def __init__(self, use_cooldown=True):
    """Initializes a Time-Based Regression analysis.

    Args:
      use_cooldown: bool. Whether cooldown period should be utilised.
    """
    self.df_names: semantics.DataFrameNameMapping = None
    self.groups: semantics.GroupSemantics = None
    self.periods: semantics.PeriodSemantics = None
    self.analysis_data: pd.DataFrame = None
    self.target = None
    # Set up container for the response model, and potentially a cost model.
    self.pre_period_model: linear_model.RegressionResultsWrapper = None
    self.use_cooldown = use_cooldown

  def fit(self, data_frame, target, **kwargs):
    """Fit the TBR model to the supplied data frame.

    See optional kwargs for interpretation of the data frame.

    Args:
      data_frame: a pandas.DataFrame. Should contain the columns and indices
      corresponding to the **kwargs information below. Only one of response
      or cost need be present, corresponding to the supplied `target`. Must be
      indexed by date.
      target: `str`. The name of the column to be analysed.
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
        group_treatment=2 - value representing the treatment group in the data.
        period_pre=0 - value representing the pre-test period in the data.
        period_test=1 - value representing the test period in the data.
        period_cool=2 - value representing the cooldown period in the data.
        period_calibration=3 - value representing the calibration period in the
          data.
    """

    # Set the target of the analysis.
    self.target = target

    # Extract any column / index name information supplied by the user.
    user_df_names = utils.kwarg_subdict('key_', **kwargs)
    self.df_names = semantics.DataFrameNameMapping(**user_df_names)

    # Extract any semantics for control / treatment supplied by user.
    user_group_semantics = utils.kwarg_subdict('group_', **kwargs)
    self.groups = semantics.GroupSemantics(**user_group_semantics)

    # Extract any semantics for experimental period supplied by user.
    user_period_semantics = utils.kwarg_subdict('period_', **kwargs)
    self.periods = semantics.PeriodSemantics(**user_period_semantics)

    # Set up the analysis data.
    self._construct_analysis_data(data_frame)
    # Fit pre-period models for response and for cost.
    self.pre_period_model = self._fit_pre_period_model()

  def _construct_analysis_data(self, data):
    """Stores group-wise time series by aggregating over control/treat geos."""
    preserve = [self.df_names.group, self.df_names.date]
    agg_style = {
        self.target: 'sum',
        self.df_names.period: 'max'  # preserve the period info of the ts.
    }
    self.analysis_data = data.groupby(preserve).agg(agg_style)

  @abc.abstractmethod
  def _fit_pre_period_model(self):
    """Estimates the control-treatment relationship in the pre-period."""
    raise NotImplementedError

  def predict(self, cntrl_mat):
    """Counterfactual prediction for treatment group series in the test period.

    Args:
        cntrl_mat: a T by 2 `np.matrix`, representing a constant concatenated
        to the control group time series, with T the test period length.

    Returns:
        A vector representing the expected treatment group time series.
    """
    return self.pre_period_model.predict(cntrl_mat)

  def _make_period_index(self, periods):
    """Returns an index for analysis_data rows in the desired time periods.

    Args:
      periods: int or non-empty iterable of int. The labels of the periods to
        consider.

    Returns: a pandas.Series of bools indicating whether each time point lies in
      the supplied periods.

    Raises:
      ValueError: if an empty periods argument is passed.
    """

    # Ensure we can iterate through periods.
    if not isinstance(periods, collections.abc.Iterable):
      period_itr = (periods,)
    else:
      if periods:
        period_itr = periods
      else:
        raise ValueError('Periods must not be an empty iterable.')

    # Construct a list of bool valued pandas.Series indicating for each period
    # whether each time point is in that period.
    subset = self.analysis_data[self.df_names.period]
    indices = [subset == i for i in period_itr]
    return functools.reduce(np.logical_or, indices)

  @abc.abstractmethod
  def causal_effect(self, periods, **kwargs):
    """Estimates the causal effect of treatment, implemented in each TBR model.
    """
    raise NotImplementedError

  def _uncorrected_causal_effect(self, periods):
    """Returns the incremental effect without bias correction.

    The incremental effect is the difference between actual and counterfactual
    prediction.

    Args:
      periods: int or iterable of int. The labels of the periods to consider.

    Returns:
       A vector representing the estimated causal effect of the treatment on the
       target variable.
    """

    period_index = self._make_period_index(periods)

    # Get the test- period data in the form needed for regression.
    treat_vec = self._response_vector(period_index)
    cntrl_mat = self._design_matrix(period_index)

    # Calculate the causal effect of the campaign.
    treat_counter = self.predict(cntrl_mat)
    return treat_vec - treat_counter

  def _response_vector(self, period_index):
    """Return the treatment group's time-series for the specified period."""
    adata = self.analysis_data
    return adata[period_index].loc[self.groups.treatment][self.target]

  def _design_matrix(self, period_index):
    """Return the design matrix for `periods`."""

    # Short variable names
    adata = self.analysis_data
    cntrl = self.groups.control
    target = self.target

    # Construct the design matrix.
    cntrl_vec = adata[period_index].loc[cntrl][target]
    cntrl_mat = cntrl_vec.to_frame()
    cntrl_mat.insert(0, 'const', 1)
    return cntrl_mat

  @abc.abstractmethod
  def summary(self, **kwargs):
    raise NotImplementedError

  def plot(self, target, experiment_dates=None, margin=0.05):
    """Plot the control and treatment time series for the target variable.

    Args:
      target: str. The name of the target variable.
      experiment_dates: iterable of str. Dates to mark with a vertical line.
      margin: float. Determines the space at the top and bottom of the y-axis.
    """
    # Labels of the group timeseries to be plotted.
    groups = [self.groups.treatment, self.groups.control]

    # Set the plotting limits.
    column = self.analysis_data[target]
    colmax = column.max()
    colmin = column.min()
    gap = margin*max(np.abs(colmax), margin*np.abs(colmin))
    ymax = colmax + gap
    ymin = colmin - gap

    # Plot the timeseries.
    for i in groups:
      plt.plot(self.analysis_data.loc[i][target], label='Group %s' % i)
    plt.legend()
    plt.ylim((ymin, ymax))

    # Place vertical lines on important dates.
    if experiment_dates:
      date_marks = pd.to_datetime(experiment_dates)
      for dt in date_marks:
        plt.vlines(dt, ymin, ymax, linestyles='dashed')


RegressionType = TypeVar('RegressionType', bound=BaseRegression)


class BaseReporting(Generic[RegressionType], metaclass=abc.ABCMeta):
  """Base class for reporting functions."""

  def __init__(self, use_cooldown=True):
    """Initializes a reporting instance for causal effect analysis.

    Args:
      use_cooldown: bool. Whether to include the cooldown period in test.
    """
    self.df_names: semantics.DataFrameNameMapping = None
    self.groups: semantics.GroupSemantics = None
    self.periods: semantics.PeriodSemantics = None
    self.analysis_data = None
    self.target = None
    # Set up container for the response model, and potentially a cost model.
    self.use_cooldown = use_cooldown
    self.tbr_response: type(RegressionType) = None
    self.tbr_cost: type(RegressionType) = None
    self.initialize_regression_model()

  @abc.abstractmethod
  def initialize_regression_model(self):
    """Initializes both regression models for response and cost."""
    raise NotImplementedError

  def fit(self, data_frame, **kwargs):
    """Fit TBR regression model to the supplied data frame.

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
        preriod_calibration=3 - value representing the calibration period in the
          data.
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

  @abc.abstractmethod
  def summary(self, **kwargs):
    """Estimates the control-treatment relationship in the post- period."""
    raise NotImplementedError

  @abc.abstractmethod
  def estimate_pointwise_and_cumulative_effect(
      self,
      metric: str, **kwargs
  ) -> common_classes.TimeSeries:
    """Estimates the pointwise and cumulative effect for the target metric."""
    raise NotImplementedError
