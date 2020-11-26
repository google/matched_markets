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
"""Diagnostics class for screening and validating data for TBR analysis.
"""

import math

from matched_markets.methodology import semantics
from matched_markets.methodology import utils
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import OLSInfluence
from statsmodels.stats.stattools import medcouple


class TBRDiagnostics(object):
  """TBR Diagnostics.

  Performs the pre-experiment and post-experiment diagnostics, removing
  geos or time points, and restricting historical data to a range that is
  validated for a TBR analysis. Can also reject the data altogether.

  Usage:
     tbrd = TBRDiagnostics()
     tbrd.fit(original_data_frame, target="sales")
     if not tbrd.tests_passed():
       raise Exception("Data is invalid")
     validated_dataset = tbrd.get_data()
  """
  _SKEWNESS_COEF = 3  # For the correlation bound.

  def __init__(self):
    self._df_names = None
    self._groups = None
    self._periods = None

    self._target = None
    self._data = None
    self._analysis_data = None

    self._diagnostics = {
        'corr_test': None,
        'noisy_geos': None,
        'enough_data': None,
        'pretest_start': None,
        'outlier_dates': None
    }
    self._tests_passed = None

  def tests_passed(self):
    """Returns the state of the diagnostic test.

    Returns:
      True, False, or None. If None, the tests have not been run. If True, the
      diagnostic tests passed. If False, the tests did not pass.
    """
    return self._tests_passed

  def get_test_results(self):
    """"Returns the results of the individual diagnostic tests.

    Returns:
      A dictionary with the results of the diagnostic test and the accompanying
      information, containing the following keys and values:
      corr_test: (bool) True if and only if the correlation test passed.
      enough_data: (bool) True if and only if a long enough stable data set
        was found.
      pretest_start: (datetime) Start of the stable pre-test period.
      outlier_dates: (list) Dates that are detected as outliers.
    """
    return self._diagnostics.copy()

  def get_data(self):
    """Returns the modified data frame in the original format.

    Returns:
      Modified data frame in the original format, with the disrupted geos,
      outlier time points removed and the pre-test period truncated.
    """
    if self._data is None:
      return None
    return self._data.copy(deep=True)

  def get_analysis_data(self):
    """Returns the modified data frame in a TBR-friendly aggregated format.

    Returns:
      Modified data frame in the aggregated TBR-friendly format, with the
      disrupted geos, outlier time points removed and the pre-test period
      truncated. Namely, the following columns:
      'date' (date) Date when the response was observed.
      'period' (int) Indicator indicating pre/post or cooldown period.
      'x' (response in control group) Aggregate response over geos.
      'y' (response in treatment group) Aggregate response over geos.
    """
    return self._analysis_data

  def _create_analysis_data(self):
    """Transforms the data frame to a TBR-friendly format.

    Sets the _analysis_data private attribute to a transformed
    pandas DataFrame  DataFrame with columns:
    'date' (date) Date when the response was observed.
    'period' (int) Indicator indicating pre/post or cooldown period.
    'x' (response in control group) Aggregate response over geos.
    'y' (response in treatment group) Aggregate response over geos.

    Raises:
      ValueError: Data frame does not contain data for both the control group
      and the treatment group.
    """

    columns = [self._df_names.date, self._df_names.period,
               self._df_names.group, self._target]
    data = self._data[columns].copy()
    # Map the numeric group ids to 'x' and 'y'.
    group_map = {self._groups.control: 'x', self._groups.treatment: 'y'}
    group = data[self._df_names.group]
    unique_groups = group.unique()
    if not all(i in unique_groups for i in group_map):
      raise ValueError('Both control and treatment group ids must be present'
                       ' in the data')
    new_group = group.map(group_map, na_action='ignore')
    data.loc[:, self._df_names.group] = new_group

    self._analysis_data = data.pivot_table(index=columns[0:2],
                                           columns=columns[2],
                                           values=columns[3], aggfunc=np.sum)
    # Drop 'period' from index, leave 'date'.
    self._analysis_data.reset_index(level=columns[1], inplace=True)

  def obs_cor(self):
    adata = self._analysis_data
    return np.corrcoef(adata.x, adata.y)[0, 1]

  def _min_correlation_threshold(self, n, min_cor, credible_level):
    """Computes the minimum threshold for the correlation test.

    For a given observed Pearson correlation obs_cor computed from n
    observations, this threshold satisfies,
      obs_cor > threshold
      <=>
      Pr(true correlation > min_cor | obs_cor) > credible_level.

    The underlying model assumes (due to Fisher transformation),
      atanh(obs_cor) ~ Normal(atanh(true_cor), sd=1 / sqrt(n - 3)),
    from where we obtain the posterior,
      atanh(true_cor)|obs_cor ~ N(atanh(obs_cor), sd=1 / sqrt(n - 3)).

    Args:
      n: (int) number of observations.
      min_cor: (float) minimum acceptable correlation.
      credible_level: (float, between 0 and 1) credible level; the true
        correlation should exceed min_cor with this probability.

    Raises:
      ValueError: number of observations must be at least 4.

    Returns:
      The minimum correlation threshold.
    """

    if n < 4:
      raise ValueError('Number of observations must be at least 4.')

    return np.tanh(np.arctanh(min_cor) + stats.norm.ppf(credible_level) /
                   np.sqrt(n - 3))

  def _correlation_test(self, min_cor, prefer_cor, credible_level):
    """Performs the minimum correlation test.

    Test that the observed correlation is sufficiently high.

    The test being true is equivalent to satisfying the dual
    Bayesian criterion,
      Pr(true correlation > min_cor | obs_cor) > credible_level.
      and
      Pr(true correlation > prefer_cor | obs_cor) > 0.5;

    The second criterion is equivalent to obs_cor > prefer_cor.

    Args:
      min_cor: (float) minimum acceptable correlation.
      prefer_cor: (float) preferred acceptable correlation. The observed
        correlation must exceed this number.
      credible_level: (float, between 0 and 1) credible level; the true
        correlation should exceed min_cor with this probability.

    Returns:
      True if and only if the test passes.
    """

    n = self._analysis_data.shape[0]
    min_threshold = self._min_correlation_threshold(n, min_cor, credible_level)
    return self.obs_cor() >= max(prefer_cor, min_threshold)

  def _detect_outliers(self, max_prob):
    """Detects outlier time points.

    Find the dates in the data set that are recommended to be removed as
    outliers.

    Args:
      max_prob: (float between 0 and 1) Maximum acceptable probability of
        having observed percentile of maximum studentized residual be greater
        than the reference distribution.

    Returns:
      A list of dates (in the data set) that were detected to be outliers.
    """
    excluded_dates = []
    while True:
      data_subset = self._analysis_data.drop(excluded_dates)
      if data_subset.shape[0] == 0:
        break
      reg_fit = smf.ols('y ~ x', data=data_subset).fit()
      absresid = abs(OLSInfluence(reg_fit).get_resid_studentized_external())
      pretest_len = data_subset.shape[0] - len(excluded_dates)
      beta_quantile = stats.beta.ppf(1 - max_prob, pretest_len, 1)
      threshold = stats.t.ppf((1 + beta_quantile) / 2, df=pretest_len - 3)
      max_resid = max(absresid)
      if max_resid < threshold:
        break
      exclude_date = list(data_subset.index[absresid == max_resid])
      excluded_dates.extend(exclude_date)

    return excluded_dates

  def _correlation_bound(self, values, iqr_coef):
    """Computes the modified Tukey bound.

    Args:
      values: A list of correlation values.
      iqr_coef: (float) Coefficient to apply to the interquartile range.

    Returns:
      The correlation bound (a scalar).
    """
    quartiles = np.percentile(values, (25, 75))
    iqr = quartiles[1] - quartiles[0]
    mc = medcouple(np.array(values), axis=None)
    return quartiles[0] - iqr_coef * iqr * math.exp(-self._SKEWNESS_COEF * mc)

  def _detect_noisy_geos(self, iqr_coef, max_threshold):
    """Detects geos that differ from the general time series pattern.

    Compares the leave-one-out time series correlation of each geo to the
    aggregate time series correlation. If the correlation is significantly
    different, flag as 'noisy'. Also geos that are constant in value across time
    are considered 'noisy'.

    Args:
      iqr_coef: float. Coefficient to apply to the interquartile range in the
        formula to determine the threshold.
      max_threshold: float. Maximum threshold.

    Returns:
      List of the ids of the geos that were detected as 'noisy'. An empty set if
      none were detected; if there are fewer than 4 geos, returns None.
    """

    # If there is a 'period' column, use only the pre-test period.
    if self._df_names.period in self._data.columns.values.tolist():
      data = self._data[self._data[self._df_names.period] == self._periods.pre]
    else:
      data = self._data

    data = data.pivot_table(index=self._df_names.geo,
                            columns=self._df_names.date,
                            values=self._target, aggfunc=np.sum, fill_value=0)
    agg_timeseries = data.sum(axis=0)
    geos = data.index
    n_geos = len(geos)
    if n_geos < 4:
      return None

    correlations = {}
    noisy_geos = []

    for i in range(n_geos):
      timeseries_i = data.iloc[i]
      corr = stats.pearsonr(timeseries_i, agg_timeseries - timeseries_i)[0]
      geo_id = geos[i]
      if np.isnan(corr):
        # Special case: sd of geo is zero => corr = NaN. Add to noisy geos.
        noisy_geos.append(geo_id)
      else:
        correlations[geo_id] = corr

    corr_bound = self._correlation_bound(list(correlations.values()),
                                         iqr_coef=iqr_coef)
    threshold = min(max_threshold, corr_bound)

    for geo_id in correlations:
      if correlations[geo_id] < threshold:
        noisy_geos.append(geo_id)

    return noisy_geos

  def fit(self, data_frame, target=None, **kwargs):
    """Runs the TBR diagnostics suite.

    This method executes the following diagnostics: (1) detect and remove the
    disrupted geos; (2) detect and remove the outlier time points (3)
    correlation test and (4) the structural stability (A/A) test removing part
    of the pre-test period. The results of these diagnostics are stored in the
    _test_results attribute. The resulting modified data frame is stored in the
    _data attribute and accessible via the get_data() method.

    Note. This method makes a copy of the original data_frame, and it doesn't
    modify the original.

    See optional kwargs for interpretation of the data frame.

    Args:
      data_frame: (pandas.DataFrame) Should contain the columns and indices
        corresponding to the **kwargs information below. Only one of response
        need be present, corresponding to the supplied `target`. Must be
        indexed by date.
      target: (str) name of the target metric (data frame column). If not
        specified, the column specified as key_response will be assumed.
      **kwargs: optional column/index names for the data and related semantics:
        key_geo (string) column name for geo (default: 'geo').
        key_period (string) column name for period (default: 'period').
        key_group (string) column name for group (default: 'group').
        key_response (string) response column name (default: 'response').
        key_date (string) date index name (default: 'date').
        group_control (int) control group id (default: 1).
        group_treat (int) treatment group id (default: 2).
        period_pre (int) pre-test period id (default: 0).
        period_test (int) test period id (default: 1).
        period_cool (int) cooldown period id (default: 2).
    """
    self._data = data_frame.copy()

    user_df_names = utils.kwarg_subdict('key_', **kwargs)
    self._df_names = semantics.DataFrameNameMapping(**user_df_names)

    user_group_semantics = utils.kwarg_subdict('group_', **kwargs)
    self._groups = semantics.GroupSemantics(**user_group_semantics)

    user_period_semantics = utils.kwarg_subdict('period_', **kwargs)
    self._periods = semantics.PeriodSemantics(**user_period_semantics)

    if target is None:
      target = self._df_names.response
    self._target = target

    remove_geos = self._detect_noisy_geos(iqr_coef=1.5, max_threshold=0.5)

    self._diagnostics['noisy_geos'] = remove_geos

    if remove_geos:
      exclude = self._data[self._df_names.geo].isin(remove_geos)
      self._data = self._data[~ exclude]

    self._create_analysis_data()

    remove_dates = self._detect_outliers(max_prob=0.1)
    self._diagnostics['outlier_dates'] = remove_dates

    if remove_dates:
      exclude_dates = self._data[self._df_names.date].isin(remove_dates)
      self._data = self._data[~ exclude_dates]
      self._create_analysis_data()

    self._diagnostics['corr_test'] = self._correlation_test(min_cor=0.5,
                                                            prefer_cor=0.8,
                                                            credible_level=0.95)
