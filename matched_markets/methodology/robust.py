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
"""Robust TBR geoexperiment methodology."""

from collections.abc import Iterable
import enum
import math
from typing import Callable, Dict, Generator, Tuple, Union
import warnings

from matched_markets.methodology import base
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression import linear_model


@enum.unique
class PermutationTestOutcome(enum.Enum):
  """The outcome of the two-sided permutation test.

  We use binary search and permutation test to determine the lower and upper
  ends of the confidence interval of the ground truth cumulative causal effect.
  In each iteration of binary search, we attempt a value as potential lower or
  upper end of the interval. We determine if this attempted value should be
  further adjusted to a larger or smaller value given the permutation test
  outcome, until the convergence criteria is met. We are mainly intersted in the
  following hyppothesis testing problem:
  H0: The attempted value in an iteration of binary search = actual ground truth
  cumulative effect, with two alternative hypotheses:
  H1a: The attempted value is less than actual cumulative effect.
  H1b: The attempted value is more than actual cumulative effect.
  """

  LESS = -1  # Rejects H0 in testing between H0 vs. H1a, and goes with H1a.
  MORE = 1  # Rejects H0 in testing between H0 vs. H1b, and goes with H1b.
  NEUTRAL = 0  # Cannot reject H0.


class RobustTBR(base.BaseRegression):
  """Robust TBR geoexperiment methodology.

  This class is an enhanced version of original TBR with bias-corrected effect
  estimator and permutation inference to ensure robustness.
  """

  def _fit_pre_period_model(self) -> linear_model.RegressionResultsWrapper:
    """Estimates the control-treatment relationship in pre&calibration period."""
    # Both the pre and calibration period data are needed for training the TBR
    # model.
    period_index = self.analysis_data[self.df_names.period].isin(
        [self.periods.pre, self.periods.calibration]
    )
    treat_vec = self._response_vector(period_index)
    cntrl_mat = self._design_matrix(period_index)
    self._post_only_periods = (
        (self.periods.test, self.periods.cooldown)
        if self.use_cooldown
        else (self.periods.test,)
    )

    # Fit an OLS model to the pre- period data.
    return sm.OLS(treat_vec.values, cntrl_mat.values).fit()

  def _check_calibration_period_in_analysis_data(self):
    """Checks if the calibration period is present in the analysis data.

    Raises:
      ValueError: If the calibration period is not present in the analysis data
        when aa bias correction is enabled.
    """
    analysis_data_periods = self.analysis_data[self.df_names.period].unique()
    if self.periods.calibration not in analysis_data_periods:
      raise ValueError(
          "Bias correction failed because calibration period is not present"
          " in the analysis data."
      )

  def causal_effect(
      self,
      periods: Union[int, Iterable[int]],
      enable_aa_bias_correction: bool = True,
  ) -> pd.Series:
    """Estimates the causal effect of treatment on the target variable.

    Args:
      periods: int or iterable of int. The labels of the periods to consider.
      enable_aa_bias_correction: bool. Whether to enable AA bias correction in
        the TBR causal effect estimation.

    Returns:
       A vector representing the estimated causal effect of the treatment on the
       target variable.
    """
    uncorrected_causal_effect = self._uncorrected_causal_effect(periods)
    if not enable_aa_bias_correction:
      return uncorrected_causal_effect
    else:
      self._check_calibration_period_in_analysis_data()
      calibration_period_causal_effect = self._uncorrected_causal_effect(
          self.periods.calibration
      )
      # Bias is estimated as the mean of uncorrected causal effects during the
      # calibration period running AA.
      return uncorrected_causal_effect - np.mean(
          calibration_period_causal_effect
      )

  def cumulative_causal_effect(self) -> float:
    """Returns the post-period cumulative causal effect on the target variable."""
    return self.causal_effect(periods=self._post_only_periods).sum()

  def _get_indices_by_group(
      self,
      periods: Union[int, Iterable[int]],
  ) -> Dict[int, pd.Index]:
    """Returns the selected indices of the data frame by group."""
    period_index = self._make_period_index(periods)
    indices = {}
    for group in (self.groups.treatment, self.groups.control):
      index = period_index[
          np.logical_and(
              period_index.index.get_level_values(self.df_names.group) == group,
              period_index,
          )
      ].index
      indices[group] = index
    return indices

  def _substract_cumulative_effect(
      self,
      analysis_data: pd.DataFrame,
      indices: Dict[int, pd.Index],
      cumulative_effect: float,
  ):
    """Substracts the cumulative effect from the treatment group.

    Equal amount of effect is subtracted from each post-period date in the
    treatment group.

    Args:
      analysis_data: pd.DataFrame. The data frame containing the analysis data.
      indices: Dict[int, pd.Index]. A dictionary containing the selected indices
        of the treatment and control groups.
      cumulative_effect: float. The cumulative effect to subtract from the
        treatment group.

    Returns:
      A data frame containing the processed analysis data that has the
      cumulative effect subtracted from the treatment group.
    """
    processed_data = analysis_data.copy()
    treat_index = indices[self.groups.treatment]
    processed_data.loc[treat_index, self.target] -= cumulative_effect / len(
        treat_index
    )
    return processed_data

  def _permute_analysis_data(
      self, analysis_data: pd.DataFrame, indices: Dict[int, pd.Index]
  ) -> Generator[pd.DataFrame, None, None]:
    """Permutes the analysis data that preserves the time series structure.

    In order to preserve the time series autocorrelation structure, we adopt a
    sliding window permutation similar to the method in
    https://www.tandfonline.com/doi/full/10.1080/01621459.2021.1920957.

    Args:
      analysis_data: pd.DataFrame. The data frame containing the analysis data.
      indices: Dict[int, pd.Index]. A dictionary containing the selected indices
        of the treatment and control groups.

    Yields:
      A data frame containing the permuted data, used for permutation inference.
    """
    treat_index = indices[self.groups.treatment]
    cntrl_index = indices[self.groups.control]
    assert len(treat_index) == len(cntrl_index)
    if len(treat_index) <= 10:
      warnings.warn(
          "The number of treatment group data points is too small for the"
          " permutation test."
      )
    for shift in range(1, len(treat_index)):
      permuted_data = analysis_data.copy()
      # Permute the treatment group and control group by shifting the target
      # variable.
      permuted_data.loc[treat_index, self.target] = np.roll(
          analysis_data.loc[treat_index, self.target].values,
          shift=shift,
      )
      permuted_data.loc[cntrl_index, self.target] = np.roll(
          analysis_data.loc[cntrl_index, self.target].values,
          shift=shift,
      )
      yield permuted_data

  def _setup_permutation_test(
      self, original_analysis_data: pd.DataFrame, p_lower: float, p_upper: float
  ) -> Tuple[float, Callable[[float], PermutationTestOutcome]]:
    """Sets up the permutation test for inferring the cumulative causal effect.

    Args:
      original_analysis_data: pd.DataFrame. The copy of the original analysis
        data without permutation and processing.
      p_lower: float. The corresponding percentile of the lower end in the
        confidence interval.
      p_upper: float. The corresponding percentile of the upper end in the
        confidence interval.

    Returns:
      A tuple containing the binary search width which determines the inital
      values of binary search and the function returning the permutation test
      outcome. This function is used to determine whether the attempted value of
      binary search is less than, more than or neutral to the actual cumulative
      causal effect.
    """
    post_only_indices = self._get_indices_by_group(
        periods=self._post_only_periods
    )
    post_and_calibration_indices = self._get_indices_by_group(
        periods=self._post_only_periods + (self.periods.calibration,)
    )
    # The initial width of the binary search for constructing the confidence
    # interval.
    initial_binary_search_width = 0
    # The number of permutations to run.
    nperm = 0
    for permuted_dataset in self._permute_analysis_data(
        original_analysis_data, post_and_calibration_indices
    ):
      nperm += 1
      self.analysis_data = permuted_dataset
      permuted_cumulative_effect = self.cumulative_causal_effect()
      initial_binary_search_width = max(
          initial_binary_search_width, abs(permuted_cumulative_effect)
      )
    kmin = math.floor(nperm * p_lower)
    kmax = math.ceil(nperm * p_upper) - 1

    def _permutation_test_fn(attempted_value: float) -> PermutationTestOutcome:
      """Returns the permutation test outcome.

      Binary search is used to find the boundaries of the confidence interval of
      the cumulative causal effect. This function is used to determine whether
      the attempted value of binary search is less than, more than or neutral to
      the actual cumulative causal effect.

      Args:
        attempted_value: float. The attempted value of the cumulative causal
          effect.

      Returns:
        The permutation test outcome, indicating whether the attempted value is
        less than, more than or neutral to the actual cumulative causal effect.
      """
      permuted_cumative_effect_stats = []
      # Substracts the cumulative effect from the treatment group.
      self.analysis_data = self._substract_cumulative_effect(
          original_analysis_data, post_only_indices, attempted_value
      )
      # Calculates the cumulative effect on the current data without
      # permutation.
      cumulative_effect = self.cumulative_causal_effect()
      for permuted_dataset in self._permute_analysis_data(
          self.analysis_data, post_and_calibration_indices
      ):
        self.analysis_data = permuted_dataset
        # Generates the cumulative effect distribution on the permuted data.
        permuted_cumative_effect_stats.append(self.cumulative_causal_effect())
      permuted_cumative_effect_stats.sort()
      if cumulative_effect > permuted_cumative_effect_stats[kmax]:
        # The cumulative_effect is significnatly higher than the permutation
        # distribution. The attempted value is significantly less than the
        # actual effect.
        return PermutationTestOutcome.LESS
      elif cumulative_effect < permuted_cumative_effect_stats[kmin]:
        # The cumulative_effect is significantly lower than the permutation
        # distribution. The attempted value is significantly more than the
        # actual effect.
        return PermutationTestOutcome.MORE
      else:
        # None of the above.
        return PermutationTestOutcome.NEUTRAL

    return initial_binary_search_width, _permutation_test_fn

  def cumulative_causal_effect_interval(
      self, level=0.8, tails=2, pct_precision=0.01, initial_width_multiplier=3.0
  ) -> Tuple[float, float]:
    """Returns the confidence interval of the cumulative causal effect.

    Args:
      level: float. The confidence level of the confidence interval.
      tails: int. The number of tails of the confidence interval. Currently only
        two-tailed confidence interval is supported.
      pct_precision: float. The percentage of the cumulative effect point
        estimate to be used as the convergence criteria for the binary search.
      initial_width_multiplier: float. Determines the initial lower and upper
        bounds of confidence interval before binary search.

    Raises:
      ValueError: If the initial width of the confidence interval before binary
        search is too narrow.
      NotImplementedError: If the number of tails is not 2. Currently only
        two-tailed confidence interval is supported.
      RuntimeError: If the binary search fails to find a valid confidence
        interval.

    Returns:
      A tuple containing the lower and upper end of the confidence interval.
    """
    alpha = (1 - level) / tails
    if tails != 2:
      raise NotImplementedError(
          "Only two-tailed confidence interval is supported."
      )
    else:
      p_upper = 1.0 - alpha
    # Saves the original analysis data.
    original_analysis_data = self.analysis_data.copy()
    cumulative_effect_estimate = self.cumulative_causal_effect()
    # Determines the convergence criteria for the binary search, which is
    # determined by the percentage of the cumulative effect point estimate.
    precision = abs(cumulative_effect_estimate * pct_precision)
    # Initializes the binary search.

    # Setups the permutation test.
    search_width, permutation_test_fn = self._setup_permutation_test(
        original_analysis_data, p_lower=alpha, p_upper=p_upper
    )
    # Initializes the lower and upper limits of binary search for the confidence
    # interval.
    ci_lower_min, ci_lower_max, ci_upper_min, ci_upper_max = (
        cumulative_effect_estimate - initial_width_multiplier * search_width,
        cumulative_effect_estimate,
        cumulative_effect_estimate,
        cumulative_effect_estimate + initial_width_multiplier * search_width,
    )
    if (
        permutation_test_fn(ci_lower_min) != PermutationTestOutcome.LESS
        or permutation_test_fn(ci_upper_max) != PermutationTestOutcome.MORE
    ):
      # Recovers the original analysis data.
      self.analysis_data = original_analysis_data
      raise ValueError(
          "The initial width of the confidence interval before binary search is"
          "  too narrow. Fails to start the permutation test. Try to increase"
          " the value of `initial_width_multiplier` instead."
      )

    # Finds the lower and upper end of confidence interval iteratively through
    # binary search.
    try:
      while (
          ci_lower_min + precision
          < ci_lower_max
          # Precision is not met, keep searching.
      ):
        ci_lower_mid = (ci_lower_min + ci_lower_max) / 2
        if permutation_test_fn(ci_lower_mid) == PermutationTestOutcome.LESS:
          # The middle value is significantly less than the actual effect.
          # Replaces the lower end of the confidence interval with the middle
          # value.
          ci_lower_min = ci_lower_mid
        else:
          ci_lower_max = ci_lower_mid
      while (
          ci_upper_max
          > ci_upper_min + precision
          # Precision is not met, keep searching.
      ):
        ci_upper_mid = (ci_upper_min + ci_upper_max) / 2
        if permutation_test_fn(ci_upper_mid) == PermutationTestOutcome.MORE:
          # The middle value is significantly more than the actual effect.
          # Replaces the upper end of the confidence interval with the middle
          # value.
          ci_upper_max = ci_upper_mid
        else:
          ci_upper_min = ci_upper_mid
    except Exception as exc:
      # Recovers the original analysis data.
      self.analysis_data = original_analysis_data
      raise RuntimeError(
          "Binary search fails to find a valid confidence interval."
      ) from exc
    # Recovers the original analysis data.
    self.analysis_data = original_analysis_data
    return ci_lower_min, ci_upper_max

  def summary(self, level=0.8, tails=2, report="last", **kwargs):
    """Returns a summary of cumulative causal effect with robust TBR approach.

    Args:
      level: `float` in [0,1]. Determines the confidence level of confidence
        interval.
      tails: `int` in {1,2}. Specifies number of tails to use in confidence
        interval. Currently only 2 tails are supported.
      report: `str`, whether to report on "all" or "last" day in post-period.
        Currently only "last" is supported.
      **kwargs: Additional arguments to pass to
        `cumulative_causal_effect_interval`.

    Returns:
      pd.DataFrame, a summary indexed by dates, containing columns:
      - estimate, the point estimate of the cumulative causal effect.
      - lower, the lower end of the confidence interval.
      - upper, the upper end of the confidence interval.
      - level, records the level parameter used to generate the report.

    Raises:
      ValueError: if level is outside of the interval [0,1].
      RuntimeError: if the TBR model has not been fitted yet.
      NotImplementedError: if the value of `tails` is not 2.
    """
    # Enforce constraints on the arguments.
    if level < 0.0 or level > 1.0:
      raise ValueError("level should be between 0.0 and 1.0.")
    if self.analysis_data is None:
      raise RuntimeError("The TBR model has not been fitted yet.")
    if report != "last":
      raise NotImplementedError(
          "Only report at the most recent available date is supported for"
          " Robust TBR."
      )
    else:
      # Finds point estimate and confidence interval of the cumulative causal
      # effect at the most recent available date.
      cumulative_effect_estimate = self.cumulative_causal_effect()
      dates = self.causal_effect(self._post_only_periods).index[-1]
      cumulative_effect_lower, cumulative_effect_upper = (
          self.cumulative_causal_effect_interval(
              level=level, tails=tails, **kwargs
          )
      )
      values = {
          "dates": dates,
          "estimate": cumulative_effect_estimate,
          "lower": cumulative_effect_lower,
          "upper": cumulative_effect_upper,
          "level": level,
      }
      report = pd.DataFrame([values])
      return report.set_index("dates")

