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
"""Time Based Regression geoexperiment methodology.
"""

from matched_markets.methodology import base
import numpy as np
import pandas as pd
import scipy as sp
import statsmodels.api as sm


class TBR(base.BaseRegression):
  """Time Based Regression geoexperiment methodology.

  This class models the relationship between control and treatment time series.

  For details see [Kerman 2017](https://ai.google/research/pubs/pub45950).
  """

  def _fit_pre_period_model(self):
    """Estimates the control-treatment relationship in the pre-period."""

    # Get the pre- period data in the form needed for regression.
    period_index = self.analysis_data[self.df_names.period] == self.periods.pre
    treat_vec = self._response_vector(period_index)
    cntrl_mat = self._design_matrix(period_index)

    # Fit an OLS model to the pre- period data.
    return sm.OLS(treat_vec.values, cntrl_mat.values).fit()

  def causal_effect(self, periods):
    """Estimates the causal effect of treatment on the target variable.

    Args:
      periods: int or iterable of int. The labels of the periods to consider.

    Returns:
       A vector representing the estimated causal effect of the treatment on the
       target variable.
    """
    return self._uncorrected_causal_effect(periods)

  def causal_cumulative_distribution(self,
                                     time=None,
                                     rescale=1.0,
                                     periods=None):
    """Return the distribution of the cumulative causal effect.

    Args:
      time: `int`. If specified, returns only the cumulative distribution at
        this time index.
      rescale: `float`. Additional scaling factor for the t-distribution.
      periods: optional tuple of `int` (default None). The periods over which to
      infer causal effects. If not supplied, the periods considered will include
      the test period and also the cooldown period if the model was constructed
      with use_cooldown=True.

    Returns:
      A t-distribution of type `scipy.stats._distn_infrastructure.rv_frozen`.
    """

    # Define periods to credit to test.
    if self.use_cooldown and periods is None:
      periods = (self.periods.test, self.periods.cooldown)
    elif periods is None:
      periods = (self.periods.test,)

    # Predict the causal effects of the experiment on response.
    causal_response = self.causal_effect(periods)
    # Counter of length test period.
    period_index = self._make_period_index(periods)
    cntrl_mat = self._design_matrix(period_index)
    len_test = cntrl_mat.shape[0]
    one_to_t = np.arange(1, len_test + 1)
    one_to_t.shape = (len_test, 1)

    # Scale contribution from parameters
    cntrl_cum_mat = np.array(np.array(cntrl_mat.cumsum()) / one_to_t)
    # Obtain the parameter covariance matrix.
    vsigma = np.array(self.pre_period_model.cov_params())
    # Each point in test-period has a different contribution.
    var_params = []
    for t in np.arange(len_test):
      # Sum of parameter variance terms from eqn 5 of Kerman 2017.
      var_t = (cntrl_cum_mat[t,] @ vsigma @ cntrl_cum_mat[t,].T)
      var_params.append(var_t)
    var_params = np.array(var_params).reshape(len_test, 1)
    # Scale the results by T\sigma^2
    var_from_params = var_params * one_to_t**2

    # Scale contribution from test observations.
    sigmasq = self.pre_period_model.scale
    var_from_observations = one_to_t * sigmasq

    # Set up the t-distribution.
    delta_mean = rescale * np.array(np.cumsum(causal_response)).flatten()
    delta_var = var_from_params + var_from_observations
    delta_scale = rescale * np.sqrt(delta_var).flatten()
    delta_df = self.pre_period_model.df_resid

    # Return a frozen t-distribution with the correct parameters.
    if time is None:
      return sp.stats.t(delta_df, loc=delta_mean, scale=delta_scale)
    else:
      return sp.stats.t(delta_df, loc=delta_mean[time], scale=delta_scale[time])

  def summary(self, level=0.9,
              threshold=0.0,
              tails=1,
              report='last',
              rescale=1.0):
    """Summarise the posterior of the cumulative causal effect, Delta.

    Args:
      level: `float` in (0,1). Determines width of CIs.
      threshold: `float`. Tests whether Delta is greater than threshold.
      tails: `int` in {1,2}. Specifies number of tails to use in tests.
      report: `str`, whether to report on 'all' or 'last' day in test period.
      rescale: `float`, an additional scaling factor for Delta.

    Returns:
      pd.DataFrame, a summary at level, with alpha=1-level, containing:
      - estimate, the median of Delta.
      - precision, distance between the (1-level)/tails and 0.5 quantiles.
      - lower, the value of the (1-level)/tails quantile.
      - upper, if tails=2, the level/tails quantile, otherwise inf.
      - scale, the scale parameter of Delta.
      - level, records the level parameter used to generate the report.
      - threshold, records the threshold parameter.
      - probability, the probability that Delta > threshold.

    Raises:
      ValueError: if tails is neither 1 nor 2.
      ValueError: if level is outside of the interval [0,1].
    """
    # Enforce constraints on the arguments.
    if tails not in (1, 2):
      raise ValueError('tails should be either 1 or 2.')

    if level < 0.0 or level > 1.0:
      raise ValueError('level should be between 0.0 and 1.0.')

    # Calculate the relevant points to evaluate.
    alpha = (1-level) / tails
    if tails == 1:
      pupper = 1.0
    elif tails == 2:
      pupper = 1.0 - alpha

    # Obtain the appropriate posterior distribution.
    delta = self.causal_cumulative_distribution(rescale=rescale)

    # Define periods to credit to test.
    if self.use_cooldown:
      periods = [self.periods.test, self.periods.cooldown]
    else:
      periods = [self.periods.test]

    # Facts about the date index.
    dates = self.causal_effect(periods).index
    ndates = len(dates)
    dates_ones = np.ones(ndates)

    # Data for the report.
    values = {
        'dates': dates,
        'estimate': delta.mean(),
        'precision': np.abs(delta.ppf(alpha) - delta.ppf(0.5)).reshape(ndates),
        'lower': delta.ppf(alpha).reshape(ndates),
        'upper': delta.ppf(pupper).reshape(ndates),
        'scale': delta.kwds['scale'].reshape(ndates),
        'level': level * dates_ones,
        'posterior_threshold': threshold * dates_ones,
        'probability': 1.0 - delta.cdf(threshold).reshape(ndates)
    }

    # Ordering for the report.
    ordering = ['estimate',
                'precision',
                'lower',
                'upper',
                'scale',
                'level',
                'probability',
                'posterior_threshold'
               ]

    # Construct the report, put it in the desired ordering.
    result = pd.DataFrame(values, index=dates)
    result = result[ordering]

    # Decide how much of the report to report.
    if report == 'all':
      lines = result.shape[0]
    elif report == 'last':
      lines = 1

    # Return the report for `lines` last days of the test period.
    return result.tail(lines)
