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
"""TBR matched markets v2 algorithm for designing go-dark experiments."""

import collections
import copy
import dataclasses
import timeit
from typing import Generator, Optional, Set

from matched_markets.methodology import heapdict
from matched_markets.methodology import tbrmatchedmarkets
from matched_markets.methodology import tbrmmdata
from matched_markets.methodology import tbrmmdesign
from matched_markets.methodology import tbrmmdesignparameters
from matched_markets.methodology import tbrmmdiagnostics
import numpy as np
from scipy import stats

ScoringV2 = collections.namedtuple(
    'ScoringV2',
    ['tests_ok', 'negative_extra_control_investment', 'control_share'],
)


class TBRMMDiagnosticsV2(tbrmmdiagnostics.TBRMMDiagnostics):
  """TBRMMDiagnostics for TBR matched markets v2."""

  _spend_y = None
  _spend_x = None
  _spend_pretestfit = None
  _extra_control_investment = None
  _control_share = None
  _optimal_budget = None

  @property
  def spend_y(self) -> Optional[np.ndarray]:
    return self._spend_y

  @property
  def spend_x(self) -> Optional[np.ndarray]:
    return self._spend_x

  @property
  def control_share(self) -> Optional[float]:
    return self._control_share

  @spend_y.setter
  def spend_y(self, value: tbrmmdiagnostics.Vector):
    if self._diagnostics_type == tbrmmdiagnostics.DiagnosticsType.TRAINING:
      spend_y = np.array(value[: -self._par.n_test])
    else:
      spend_y = np.array(value)
    if spend_y.ndim != 1:
      raise ValueError('spend_y must be a one-dimensional vector')
    if len(spend_y) != len(self._y):
      raise ValueError(
          'spend_y must have the same length as y (%d)' % len(self._y)
      )
    self._spend_y = spend_y

  @spend_x.setter
  def spend_x(self, value: tbrmmdiagnostics.Vector):
    assert self._spend_y is not None
    if value is None:
      spend_x = None
    else:
      if self._diagnostics_type == tbrmmdiagnostics.DiagnosticsType.TRAINING:
        spend_x = np.array(value[: -self._par.n_test])
      else:
        spend_x = np.array(value)
      if spend_x.ndim != 1:
        raise ValueError('spend_x must be a one-dimensional vector')
      if len(spend_x) != len(self._y):
        raise ValueError(
            'spend_x must have the same length as y (%d)' % len(self._y)
        )
    self._spend_x = spend_x

  @control_share.setter
  def control_share(self, value: float):
    self._control_share = value

  @property
  def spend_pretestfit(self) -> Optional[tbrmmdiagnostics.LinregResult]:
    """Simple linear regression ('OLS') fit of the pretest spend data."""
    n_test = self._par.n_test
    x, y = self._spend_x, self._spend_y
    if x is None:
      return None
    elif self._spend_pretestfit is None:
      if self._diagnostics_type == tbrmmdiagnostics.DiagnosticsType.EVAL:
        x, y = self._spend_x[-n_test:], self._spend_y[-n_test:]
      try:
        b, a, *_ = stats.linregress(x, y)
      except ValueError:
        b, a = np.nan, np.nan
      resid = y - a - b * x
      sigma = np.std(resid, ddof=2)
      self._spend_pretestfit = tbrmmdiagnostics.LinregResult(a, b, sigma, resid)

    return self._spend_pretestfit

  @property
  def extra_control_investment(self) -> Optional[float]:
    return self.estimate_extra_control_investment()

  def _get_joined_tbr_slope(
      self,
      response_y: np.ndarray,
      response_x: np.ndarray,
      spend_y: np.ndarray,
      spend_x: np.ndarray,
  ) -> float:
    """Returns the joined TBR slope using both response and spend data."""
    response_y_norm = (response_y - response_y.mean()) / response_y.std()
    spend_y_norm = (spend_y - spend_y.mean()) / spend_y.std()
    # Scale the response_x by the same scale used to scale response_y, so
    # that beta is the same as the unscaled version.
    response_x_norm = (response_x - response_x.mean()) / response_y.std()
    # Scale the spend_x by the same scale used to scale spend_y, so
    # that beta is the same as the unscaled version.
    spend_x_norm = (spend_x - spend_x.mean()) / spend_y.std()

    x = np.concatenate((response_x_norm, spend_x_norm))
    y = np.concatenate((response_y_norm, spend_y_norm))
    b, _, *_ = stats.linregress(x, y)
    return b

  def estimate_extra_control_investment(
      self, corr: Optional[float] = None, b: Optional[float] = None
  ) -> Optional[float]:
    # Minimizes for the extra control investment regardless the potential budget
    # shift from geo to geo if a BAU design can not be found.
    n_test = self._par.n_test
    response_x, response_y = self._x, self._y
    spend_x, spend_y = self._spend_x, self._spend_y

    if response_y is None:
      return None
    if response_x is None:
      return None
    if spend_x is None:
      return None
    if spend_y is None:
      return None

    if self._diagnostics_type == tbrmmdiagnostics.DiagnosticsType.EVAL:
      response_x, response_y = self._x[-n_test:], self._y[-n_test:]
      spend_x, spend_y = self._spend_x[-n_test:], self._spend_y[-n_test:]

    if b is None:
      b = self._get_joined_tbr_slope(response_y, response_x, spend_y, spend_x)

    # TODO(yupuchen): decide if we should use the last n_test days of spend or
    # the n*(daily spend of entire training period). Because required_budget
    # uses correlation and sd(y) from training period. Assumption is the last n
    # days training period is similar to the test period.
    last_n_trt_spend = spend_y[-n_test:].sum()
    if corr is None:
      corr = self.corr
    required_impact = self.estimate_required_impact(corr)
    required_buget = required_impact / self._par.iroas
    return (required_buget - last_n_trt_spend) / b

  def tbr_slope_diff(self) -> Optional[float]:
    """Symmetric percentage difference between slope parameters in spend and response models.

    Compute the symmetric percentage difference
    (https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error)
    between the slope parameters in spend and response models. The symmetric
    percentage difference is symmetric so it doesn't depend on the relative
    scale of slope parameter in spend and response models. The symmetric
    percentage difference has a lower bound of 0% and a upper bound of 200%.
    If the slope paratmeter is 0, return infinity.

    Returns:
      The symmetric percentage difference between slope parameters in spend and
      response models. None if either spend or response pretest tbr fit is not
      available.
    """
    spend_pretestfit = self.spend_pretestfit

    if self.pretestfit is None:
      return None
    elif spend_pretestfit is None:
      return None
    else:
      _, b_response, _, *_ = self.pretestfit
      _, b_spend, _, *_ = spend_pretestfit

    if b_response == 0 or b_spend == 0:
      return np.inf
    else:
      return 2 * abs(b_response - b_spend) / (abs(b_response) + abs(b_spend))

  @property
  def check_equal_tbr_slope(self) -> Optional[bool]:
    """Checks if the slope parameters are equal in spend and response models."""
    tbr_slope_diff = self.tbr_slope_diff()
    if tbr_slope_diff is None:
      return None
    else:
      return tbr_slope_diff <= self._par.slope_diff_tolerance

  @property
  def optimal_budget(self) -> Optional[float]:
    """The optimal budget for the design.

    Calculates the last n days of control spend and the required extra control
    investment, and uses them to estimate the optimal budget. If the required
    extra control investment is negative, the optimal budget is the last n days
    control spend which corresponds to a BAU in control design.

    Returns:
      The estimated optimal budget. None if last n days of control spend or the
      required extra control investment is not available.
    """
    if self._optimal_budget is None:
      extra_control_investment = self.extra_control_investment
      if extra_control_investment is None:
        return None
      if self.spend_x is None:
        return None
      last_n_ctrl_spend = self.spend_x[-self._par.n_test :].sum()
      self._optimal_budget = (
          max(0, extra_control_investment) + last_n_ctrl_spend
      )
    return self._optimal_budget

  @property
  def tests_ok(self) -> Optional[int]:
    """The value of the joint diagnostic test.

    Returns:
      Return 1 iff all individual diagnostic tests pass, 0 if any individual
      test fails, None if any test returns None. A value of None may be due to
      (1) 'x' is undefined causing corr_test to return None; or (2)
      check_equal_tbr_slope is None.
    """
    if self._tests_ok is None:
      if self.corr_test is None:
        return None
      elif self.check_equal_tbr_slope is None:
        return None
      else:
        self._tests_ok = int(self.corr_test and self.check_equal_tbr_slope)
    return self._tests_ok

  def __repr__(self):
    return 'TBRMMDiagnosticsV2(tests_ok={}, corr={})'.format(
        self.tests_ok, self.corr
    )


@dataclasses.dataclass
class TBRMMScoreV2:
  """Score of a TBR matched markets v2 design."""

  diag: TBRMMDiagnosticsV2  # design diagnostics
  _score = None  # score of the corresponding design

  def __lt__(self, other: 'TBRMMScoreV2'):
    return self.score < other.score

  @property
  def score(self):
    assert self.diag.tests_ok is not None
    assert self.diag.extra_control_investment is not None
    assert self.diag.control_share is not None
    if self._score is None:
      self._score = ScoringV2(
          self.diag.tests_ok,
          min(0, self.diag.extra_control_investment * -1),
          self.diag.control_share,
      )
    return self._score

  @score.setter
  def score(self, value: ScoringV2):
    self._score = value


class TBRMatchedMarketsV2(tbrmatchedmarkets.TBRMMDesignBase):
  """TBR matched markets v2 design."""

  _MIN_GEO_SHARE = 0.001  # Lower bound for geo share.

  def __init__(
      self,
      response_data: tbrmmdata.TBRMMData,
      spend_data: tbrmmdata.TBRMMData,
      parameters: tbrmmdesignparameters.TBRMMDesignParameters,
      use_holdout: bool = True,
      timeout: Optional[int] = None,
  ):
    self.spend_data = spend_data
    super().__init__(response_data, parameters, use_holdout, timeout)
    assert self.parameters.control_share_range is not None

  def _post_init_setup(self):
    """Customizes the setup for TBR matched markets v2."""
    # Makes sure the spend and response data are under the same geo indexing.
    geos_included = self.geos_within_constraints
    # Order geos in the order of implied budget size ('expensive' first).
    geos_in_order = list(self.geo_req_impact.index)
    geo_index = [geo for geo in geos_in_order if geo in geos_included]
    self.data.geo_index = geo_index
    self.spend_data.geo_index = geo_index
    # Initializes a distance matrix among all geos based on response time
    # series.
    self._initialize_geo_response_dist()
    self._n_geos = len(self._response_dist)

  @property
  def geos_over_budget(self) -> Set[tbrmatchedmarkets.GeoID]:
    """Identify geos which do not satisfy the max ad spend budget condition."""
    return set()

  @property
  def geos_too_large(self) -> Set[tbrmatchedmarkets.GeoID]:
    """Identify geos which do not satisfy the maximum geo share condition."""
    return set()

  @property
  def geos_too_small(self) -> Set[tbrmatchedmarkets.GeoID]:
    """Identify geos that represent < 0.1% of the market share in response or cost."""
    geo_share = self.data.geo_share
    geo_spend_share = self.spend_data.geo_share
    response_too_small = geo_share.index[geo_share < self._MIN_GEO_SHARE]
    cost_too_small = geo_spend_share.index[
        geo_spend_share < self._MIN_GEO_SHARE
    ]
    geos = set(response_too_small) | set(cost_too_small)
    return geos

  def _initialize_geo_response_dist(self):
    """Initializes the correlation-based distance metrics for market matching."""
    eiligible_geos = self.data.geo_index
    self._response_dist = 1.0 - self.data.df.T[eiligible_geos].corr().values

  def _get_design_score(
      self,
      treatment_group: Set[tbrmatchedmarkets.GeoIndex],
      control_group: Set[tbrmatchedmarkets.GeoIndex],
      diagnostics_type: tbrmmdiagnostics.DiagnosticsType,
      return_score_upper_bound: bool = False,
  ) -> TBRMMScoreV2:
    """Returns the score of a design."""
    diag = TBRMMDiagnosticsV2(
        y=self.data.aggregate_time_series(treatment_group),
        par=self.parameters,
        diagnostics_type=diagnostics_type,
    )
    diag.x = self.data.aggregate_time_series(control_group)
    diag.spend_y = self.spend_data.aggregate_time_series(treatment_group)
    diag.spend_x = self.spend_data.aggregate_time_series(control_group)
    diag.control_share = self.data.aggregate_geo_share(control_group)
    tbrmm_score = TBRMMScoreV2(diag)
    if return_score_upper_bound:
      rho_max = self.parameters.rho_max
      b_max = (
          1 - self.parameters.control_share_range[0]
      ) / self.parameters.control_share_range[0]
      tbrmm_score.score = ScoringV2(
          tests_ok=1,
          negative_extra_control_investment=min(
              0, diag.estimate_extra_control_investment(rho_max, b_max) * -1
          ),
          control_share=diag.control_share,
      )
    return tbrmm_score

  def _treatment_group_generator(
      self, n: int
  ) -> Generator[Set[tbrmatchedmarkets.GeoIndex], None, None]:
    """Generates all possible treatment groups of given size.

    Args:
      n: Size of the treatment group.

    Raises:
      ValueError if n is not positive or larger than half of the geos.

    Yields:
      Sets of geo indices, of length n each.
    """
    if n <= 0:
      raise ValueError('Treatment group size n must be positive')
    if n > int(0.5 * self._n_geos):
      raise ValueError(
          'Treatment group size n cannot be greater than half of the all geos'
      )

    def estimate_required_impact(y):
      return TBRMMDiagnosticsV2(y, self.parameters).estimate_required_impact(
          self.parameters.rho_max
      )

    unique_group = set()
    unsorted_treatment_groups = []
    for geo in self.data.geo_assignments.t:
      geo_distance = self._response_dist[geo]
      geo_to_include = geo_distance.argsort()[: n * 2]
      trt_group = frozenset(geo_to_include[::2])
      if trt_group not in unique_group:
        unique_group.add(trt_group)
        trt_geo_set = set(trt_group)
        y = self.data.aggregate_time_series(trt_geo_set)
        impact_y = (
            self.spend_data.aggregate_time_series(trt_geo_set)
            * self.parameters.iroas
        )
        impact_diff = estimate_required_impact(y) - sum(
            impact_y[-self.parameters.n_test :]
        )
        unsorted_treatment_groups.append((trt_geo_set, impact_diff))
    # The treatment groups are sorted by the difference between the
    # required budget and the actual spend, in ascending order.
    unsorted_treatment_groups.sort(key=lambda x: x[1])
    for trt_geo_set, _ in unsorted_treatment_groups:
      yield trt_geo_set

  def _excluded_geo_generator(
      self,
      treatment_group: Set[tbrmatchedmarkets.GeoIndex],
      control_group: Set[tbrmatchedmarkets.GeoIndex],
  ) -> Generator[tbrmatchedmarkets.GeoIndex, None, None]:
    """Yields geo to exclude from the control group sorted from the best to the worst, given the treatment group.

    Args:
      treatment_group: the given treatment group.
      control_group: the control group where the excluded geo is selected from.

    Yields:
      Geo to exclude from the control group sorted from the best to the worst.
    """

    control_group_geo_index = [self.data.geo_index[g] for g in control_group]
    ctl_group_ts = self.data.df.T[control_group_geo_index]
    ctl_group_ts['trt'] = self.data.aggregate_time_series(treatment_group)
    proxy_metric = ctl_group_ts.cov()['trt'][:-1].sort_values(ascending=True)
    for geo in proxy_metric.index:
      yield self.data.geo_index.index(geo)

  def greedy_search(self):
    """Searches the Matched Markets V2 design for a TBR experiment.

    Uses a greedy hill climbing algorithm to provide recommended 'matched
    markets' experimental group assignments that appear to lead to valid and
    effective TBR models relative to the pretest period.

    Returns:
      the set of feasible designs found given the design parameters,
      with their corresponding treatment/control groups and score.
    """
    self._post_init_setup()
    self.tests_failed_counter = collections.defaultdict(int)

    start_time = timeit.default_timer()
    search_not_timeout = True
    self._search_results = heapdict.HeapDict(size=self.parameters.n_designs)
    n_min, n_max = 1, int(0.5 * self._n_geos)
    if self.parameters.treatment_geos_range is not None:
      n_min = max(n_min, self.parameters.treatment_geos_range[0])
      n_max = min(n_max, self.parameters.treatment_geos_range[1])
    for n in range(n_min, n_max):
      if not search_not_timeout:
        break
      for treatment_group in self._treatment_group_generator(n):
        if (
            self._timeout is not None
            and timeit.default_timer() - start_time > self._timeout
        ):
          search_not_timeout = False
        if not search_not_timeout:
          break
        treatment_share = self.data.aggregate_geo_share(treatment_group)
        if (
            self.parameters.treatment_share_range is not None
            and treatment_share < self.parameters.treatment_share_range[0]
        ):
          # Skip if the treatment group is too small
          continue
        if (
            self.parameters.treatment_share_range is not None
            and treatment_share > self.parameters.treatment_share_range[1]
        ):
          # Skip if the treatment group is too large
          continue

        best_control_group = set(range(self._n_geos)).difference(
            treatment_group
        )
        control_share = self.data.aggregate_geo_share(best_control_group)
        current_score_upper_bound = self._get_design_score(
            treatment_group,
            best_control_group,
            self._diagnostics_type,
            return_score_upper_bound=True,
        )
        design_scores_in_heap = self._search_results.get_result()
        if (
            design_scores_in_heap
            and len(design_scores_in_heap[0]) == self.parameters.n_designs
            and current_score_upper_bound < design_scores_in_heap[0][-1].score
        ):
          # Skip if the upper bound of the score given this treatment group is
          # still lower than the lowest score we have kept in the heap, which
          # means no design with this treatment group will be among the best
          # designs.
          continue
        current_score = self._get_design_score(
            treatment_group, best_control_group, self._diagnostics_type
        )

        for ex_geo in self._excluded_geo_generator(
            treatment_group, best_control_group
        ):
          if control_share < self.parameters.control_share_range[0]:
            break
          new_control_group = best_control_group.difference([ex_geo])
          if not self.design_within_constraints(
              treatment_group, new_control_group
          ):
            # Skip if the proposed design does not meet the constraints.
            continue
          else:
            # Compute the score based on new design.
            new_score = self._get_design_score(
                treatment_group, new_control_group, self._diagnostics_type
            )
          if new_score > current_score:
            current_score = new_score
            best_control_group = new_control_group
            control_share = self.data.aggregate_geo_share(best_control_group)
          elif new_score.score.tests_ok:
            # If tests ok but cannot further improve the design score, stop
            # moving more control geos to excluded group.
            break

        design = tbrmmdesign.TBRMMDesign(
            score=current_score,
            treatment_geos=treatment_group,
            control_geos=best_control_group,
            diag=copy.deepcopy(current_score.diag),
        )
        # Push the design to search results only if it is a valid design
        # where TBR slope parameters are equal in spend and response models,
        # and the correlation between treatment and control groups is
        # sufficiently high.
        if current_score.score.tests_ok:
          self._search_results.push(0, design)
        else:
          # Update the counter of tests failed and determine the reason of
          # the failure.
          self.tests_failed_counter['total'] += 1
          if not current_score.diag.check_equal_tbr_slope:
            self.tests_failed_counter['unequal_tbr_slope'] += 1
          if not current_score.diag.corr_test:
            self.tests_failed_counter['low_corr'] += 1

    return self.search_results()

  def eval_scores(self) -> Optional[TBRMMScoreV2]:
    """The scores of the selected design based on the evaluation data."""

    if not self._use_holdout:
      return None

    result = self._search_results.get_result()
    if result:
      d = result[0][0]
      design_eval_score = self._get_design_score(
          d.treatment_geos,
          d.control_geos,
          tbrmmdiagnostics.DiagnosticsType.EVAL,
      )
      return design_eval_score
