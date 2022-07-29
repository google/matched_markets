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
"""TBR Matched Markets preanalysis.
"""
import copy
import itertools
from typing import Generator, List, Set, Text, TypeVar

from matched_markets.methodology import geoeligibility
from matched_markets.methodology import heapdict
from matched_markets.methodology import tbrmmdata
from matched_markets.methodology import tbrmmdesign
from matched_markets.methodology import tbrmmdesignparameters
from matched_markets.methodology import tbrmmdiagnostics
from matched_markets.methodology import tbrmmscore
import numpy as np
import pandas as pd
from scipy import special as scipy_special

TBRMMDesignParameters = tbrmmdesignparameters.TBRMMDesignParameters
TBRMMDiagnostics = tbrmmdiagnostics.TBRMMDiagnostics
TBRMMData = tbrmmdata.TBRMMData
TBRMMDesign = tbrmmdesign.TBRMMDesign
TBRMMScore = tbrmmscore.TBRMMScore
GeoID = Text
GeoIndex = int
DictKey = TypeVar('DictKey', str, int, float)


class TBRMatchedMarkets:
  """TBR Matched Market preanalysis.

  Attributes:
    data: The TBRMMData object.
    parameters: The TBRMMDesignParameters object.
    geo_req_impact: Required minimum incremental impact for each individual geo.
  """

  data: TBRMMData
  parameters: TBRMMDesignParameters
  geo_req_impact: pd.Series

  def __init__(self, data: TBRMMData, parameters: TBRMMDesignParameters):
    """Initialize a TBRMatchedMarkets object.

    Args:
      data: A TBRMMData object.
      parameters: a TBRMMDesignParameters object.
    """

    def estimate_required_impact(y):
      return TBRMMDiagnostics(y,
                              parameters).estimate_required_impact(
                                  parameters.rho_max)
    # Consider only the most recent n_pretest_max time points
    data.df = data.df.iloc[:, -parameters.n_pretest_max:]
    # Calculate the required impact estimates for each geo.
    geo_req_impact = data.df.apply(estimate_required_impact, axis=1)

    self.geo_req_impact = geo_req_impact
    self.data = data
    self.parameters = parameters

  @property
  def geos_over_budget(self) -> Set[GeoID]:
    """Identify geos which do not satisfy the max ad spend budget condition."""
    if self.parameters.budget_range is not None:
      max_impact = self.parameters.budget_range[1] * self.parameters.iroas
      geo_impact = self.geo_req_impact
      geos = set(geo_impact.index[geo_impact > max_impact])
    else:
      geos = set()
    return geos

  @property
  def geos_too_large(self) -> Set[GeoID]:
    """Identify geos which do not satisfy the maximum geo share condition."""
    if self.parameters.treatment_share_range is not None:
      max_trt_share = self.parameters.treatment_share_range[1]
      geo_share = self.data.geo_share
      geos = set(geo_share.index[geo_share > max_trt_share])
    else:
      geos = set()
    return geos

  @property
  def geos_must_include(self) -> Set[GeoID]:
    """Set of geos that must be included in each design."""
    geo_assignments = self.data.geo_eligibility.get_eligible_assignments()
    return geo_assignments.all - geo_assignments.x

  @property
  def geos_within_constraints(self) -> Set[GeoID]:
    """Set of geos that are within the geo-specific constraints.

    Returns:
      Geos that are assignable to control or treatment but not over budget nor
      too large, plus those that must be assigned to the treatment or control
      group (even if over budget or too large). If the maximum number is
      specified, the geos with the highest impact on budget are chosen.
    """
    geos_exceed_size = self.geos_too_large | self.geos_over_budget
    geos = (self.data.assignable - geos_exceed_size) | self.geos_must_include
    n_geos_max = self.parameters.n_geos_max
    if n_geos_max is not None and len(geos) > n_geos_max:
      geos_with_max_impact = list(
          self.geo_req_impact.sort_values(ascending=False).index)
      geos_in_order = list(geo for geo in geos_with_max_impact if geo in geos)
      geos = set(geos_in_order[:n_geos_max])
    return geos

  @property
  def geo_assignments(self) -> geoeligibility.GeoAssignments:
    """Return the possible geo assignments."""

    geos_included = self.geos_within_constraints

    # Order geos in the order of implied budget size ('expensive' first).
    geos_in_order = list(self.geo_req_impact.index)
    geo_index = [geo for geo in geos_in_order if geo in geos_included]

    self.data.geo_index = geo_index
    return self.data.geo_assignments

  def treatment_group_size_range(self) -> range:
    """Range from smallest to largest possible treatment group size."""
    n_treatment_min = max(1, len(self.geo_assignments.t_fixed))
    n_treatment_max = len(self.geo_assignments.t)
    if not self.geo_assignments.cx | self.geo_assignments.c_fixed:
      # No geos left outside the group 't', so reserve at least one geo for the
      # control group.
      n_treatment_max -= 1

    treatment_geos_range = self.parameters.treatment_geos_range
    if treatment_geos_range is None:
      n_geos_from, n_geos_to = (n_treatment_min, n_treatment_max)
    else:
      n_geos_from = max(treatment_geos_range[0], n_treatment_min)
      n_geos_to = min(treatment_geos_range[1], n_treatment_max)

    return range(n_geos_from, n_geos_to + 1)

  def _control_group_size_generator(
      self,
      n_treatment_geos: int) -> Generator[int, None, None]:
    """Acceptable control group sizes, given treatment group size.

    Args:
      n_treatment_geos: Number of treatment geos.

    Yields:
      Control group sizes that agree with the range and ratio constraints.
    """
    n_control_min = max(1, len(self.geo_assignments.c_fixed))
    n_control_max = len(self.geo_assignments.c)

    control_geos_range = self.parameters.control_geos_range
    if control_geos_range is None:
      n_geos_from, n_geos_to = (n_control_min, n_control_max)
    else:
      n_geos_from = max(control_geos_range[0], n_control_min)
      n_geos_to = min(control_geos_range[1], n_control_max)

    if self.parameters.geo_ratio_tolerance is None:
      yield from range(n_geos_from, n_geos_to + 1)
    else:
      geo_tol_max = 1.0 + self.parameters.geo_ratio_tolerance
      geo_tol_min = 1.0 / geo_tol_max
      for n_control_geos in range(n_geos_from, n_geos_to + 1):
        geo_ratio = n_control_geos / n_treatment_geos
        if geo_ratio >= geo_tol_min and geo_ratio <= geo_tol_max:
          yield n_control_geos

  def treatment_group_generator(
      self,
      n: int) -> Generator[Set[GeoIndex], None, None]:
    """Generates all possible treatment groups of given size.

    The indices will generated in the order from smallest to largest, e.g.,
    choosing n=2 geos out of {0, 1, 2, 3} will yield the sequence {0, 1}, {0,
    2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}. The indices refer to geos from largest
    to smallest, i.e., geo 0 is the largest, 4 is the smallest. The fixed geos
    will be added to the set.

    Args:
      n: Size of the treatment group.

    Raises:
      ValueError if n is not positive.

    Yields:
      Sets of geo indices, of length n each. If there are not enough geos
      available, does not yield anything.
    """
    if n <= 0:
      raise ValueError('Treatment group size n must be positive')

    fixed_treatment_geos = self.geo_assignments.t_fixed
    varying_treatment_geos = self.geo_assignments.t - fixed_treatment_geos
    n_remaining = n - len(fixed_treatment_geos)
    if n_remaining == 0 and fixed_treatment_geos:
      yield fixed_treatment_geos  # pytype: disable=bad-return-type
    elif n_remaining > 0:
      it = itertools.combinations(varying_treatment_geos, n_remaining)
      for treatment_geos_combination in it:
        treatment_group = fixed_treatment_geos | set(treatment_geos_combination)
        yield treatment_group  # pytype: disable=bad-return-type

  def control_group_generator(
      self,
      treatment_group: Set[GeoIndex]) -> Generator[Set[GeoIndex], None, None]:
    """Iterates over control geo combinations, given a treatment group.

    Args:
      treatment_group: Set of treatment geos. The sequence of control groups is
      constructed from the remaining geos.

    Yields:
      Sets of geo indices representing control groups. If there are not enough
      geos available, does not yield anything.
    """
    if not treatment_group:
      raise ValueError('Treatment group must not be empty')

    # The treatment group must be a subset of the available treatment geos.
    invalid_geo_indices = treatment_group - self.geo_assignments.t
    if invalid_geo_indices:
      geos = [str(geo_index) for geo_index in sorted(invalid_geo_indices)]
      raise ValueError(
          'Invalid treatment geo indices: ' + ', '.join(geos))

    # The fixed control geos are those that belong to either 'c_fixed' or 'ct'.
    # The geos in the group 'ct' must be assigned to control or treatment, but
    # not excluded.
    ct_geos_not_in_treatment = self.geo_assignments.ct - treatment_group
    fixed_control_geos = self.geo_assignments.c_fixed | ct_geos_not_in_treatment
    possible_control_geos = self.geo_assignments.c - treatment_group

    # The 'varying control geos' can be in the groups 'cx' or 'ctx' only.
    varying_control_geos = possible_control_geos - fixed_control_geos

    n_treatment_geos = len(treatment_group)

    for n_control_geos in self._control_group_size_generator(n_treatment_geos):
      n_remaining = n_control_geos - len(fixed_control_geos)
      if n_remaining == 0 and fixed_control_geos:
        yield fixed_control_geos  # pytype: disable=bad-return-type
      elif n_remaining > 0:
        # If n_remaining > len(varying_control_geos), the generator will not
        # return anything.
        it = itertools.combinations(varying_control_geos, n_remaining)
        for control_geos in it:
          control_group = fixed_control_geos | set(control_geos)
          yield control_group  # pytype: disable=bad-return-type

  def count_max_designs(self) -> int:
    """Count (fast) how many designs there are at most.

    Only the group sizes and their ratio is used as a constraint.

    Returns:
      Maximum number of designs under the constraint of the geo eligibility
      matrix, and the geo group sizes and allowed ratios.
    """
    n_t_fixed = len(self.geo_assignments.t_fixed)
    n_c_fixed = len(self.geo_assignments.c_fixed)
    n_cx = len(self.geo_assignments.cx)
    n_tx = len(self.geo_assignments.tx)
    n_ct = len(self.geo_assignments.ct)
    n_ctx = len(self.geo_assignments.ctx)
    trt_sizes = set(self.treatment_group_size_range())
    # Pre-compute the control sizes to avoid repetition within the loop.
    control_group_sizes = {}
    for n_trt in trt_sizes:
      group_sizes = set(self._control_group_size_generator(n_trt))
      control_group_sizes[n_trt] = group_sizes
    n_designs = 0
    # Split the space into the subspaces cx, tx, ct, ctx.
    comb = scipy_special.comb
    for i_ct in range(1 + n_ct):
      n1 = comb(n_ct, i_ct, exact=True)
      for i_tx in range(1 + n_tx):
        n2 = comb(n_tx, i_tx, exact=True)
        for i_ctx in range(1 + n_ctx):
          n_trt = n_t_fixed + i_tx + i_ctx + i_ct
          if n_trt in trt_sizes:
            ctl_sizes = control_group_sizes[n_trt]
            n3 = comb(n_ctx, i_ctx, exact=True)
            for i_cx in range(1 + n_cx):
              n4 = comb(n_cx, i_cx, exact=True)
              for i_cctx in range(1 + n_ctx - i_ctx):
                n_ctl = n_c_fixed + i_cx + i_cctx + (n_ct - i_ct)
                if n_ctl in ctl_sizes:
                  n5 = comb(n_ctx - i_ctx, i_cctx, exact=True)
                  n_designs += n1 * n2 * n3 * n4 * n5
    return n_designs

  def exhaustive_search(self) -> List[TBRMMDesign]:
    """Search the design space for acceptable designs, within the constraints.

    Returns:
      the set of feasible designs found given the design parameters,
        with their corresponding treatment/control groups and score.
    """
    treatment_share_range = self.parameters.treatment_share_range
    budget_range = self.parameters.budget_range

    # Do not store patterns when we have the last treatment pattern size.
    skip_this_trt_group_size = list(self.treatment_group_size_range()).pop()
    skip_treatment_geo_patterns = []

    results = heapdict.HeapDict(size=self.parameters.n_designs)

    def skip_if_subset(geos: Set[GeoIndex]) -> bool:
      """Check if one of the stored geo patterns is a subset of the geos.

      Args:
        geos: Set of geo indices.

      Returns:
        bool: True if one of the stored groups is a subset of the geos.
      """
      for p in skip_treatment_geo_patterns:
        if set(p).issubset(geos):
          return True
      return False

    volume_tol = self.parameters.volume_ratio_tolerance
    if volume_tol is not None:
      tol_min = 1.0 / (1.0 + volume_tol)
      tol_max = 1.0 + volume_tol

    treatment_group_sizes = self.treatment_group_size_range()
    for treatment_group_size in treatment_group_sizes:

      # Treatment groups are saved for the purpose of the inclusion check.
      save_treatment_groups = (treatment_group_size != skip_this_trt_group_size)

      treatment_groups = self.treatment_group_generator(treatment_group_size)
      for treatment_group in treatment_groups:
        treatment_share = self.data.aggregate_geo_share(treatment_group)
        if treatment_share_range is not None:
          # Skip this treatment group if the group implies too low or high share
          # of response volume.
          if (treatment_share > treatment_share_range[1] or
              treatment_share < treatment_share_range[0]):
            continue
        elif skip_if_subset(treatment_group):
          # If the group is a superset of a group that we already know has too
          # high a share or budget, then skip this group too.
          continue
        y = self.data.aggregate_time_series(treatment_group)
        diag = TBRMMDiagnostics(y, self.parameters)
        req_impact = diag.estimate_required_impact(self.parameters.rho_max)
        req_budget = req_impact / self.parameters.iroas
        if budget_range is not None:
          # If the budget is too high, skip this treatment group.
          if req_budget > budget_range[1]:
            if save_treatment_groups:
              # We skip all treatment groups that are a superset of a treatment
              # group that has too high an estimated budget.
              skip_treatment_geo_patterns.append(treatment_group)
              continue
            # If the budget is too low, skip this treatment group.
          elif req_budget < budget_range[0]:
            continue
        control_groups = self.control_group_generator(treatment_group)
        for control_group in control_groups:
          if volume_tol is not None:
            control_share = self.data.aggregate_geo_share(control_group)
            xy_share = control_share / treatment_share
            if xy_share > tol_max or xy_share < tol_min:
              continue
          diag.x = self.data.aggregate_time_series(control_group)
          corr = diag.corr  # pylint: disable=unused-variable
          req_impact = diag.required_impact
          req_budget = req_impact / self.parameters.iroas
          if (budget_range is not None and (self._constraint_not_satisfied(
              req_budget, budget_range[0], budget_range[1]))):
            continue

          # deepcopy is needed otherwise diag.corr gets overwritten, and so
          # it will not be equal to diag.score.score.corr for some reason
          design_score = TBRMMScore(copy.deepcopy(diag))
          score = design_score.score
          if budget_range is not None:
            # If the budget was specified then we use the inverse of the
            # minimum detectable iROAS for the max. budget as the last value
            # in the scoring, instead of using the same for a budget of 1$
            iroas = req_impact / budget_range[1]
            design_score.score = score._replace(inv_required_impact=1 / iroas)

          # deepcopy is needed otherwise diag.corr gets overwritten, and so
          # it will not be equal to diag.score.score.corr for some reason
          design = TBRMMDesign(
              design_score, treatment_group, control_group,
              copy.deepcopy(diag))
          results.push(0, design)

    self._search_results = results
    return self.search_results()

  def search_results(self):
    """Outputs the results of the exhaustive search in a friendly format.

    Returns:
      results: the set of feasible designs found given the design parameters,
        with their corresponding treatment/control groups and score.

    """
    result = self._search_results.get_result()
    output_result = []
    if result:
      design = result[0]
      # map from geo indices to geo IDs.
      for d in design:
        treatment_geos = {self.data.geo_index[x] for x in d.treatment_geos}
        control_geos = {self.data.geo_index[x] for x in d.control_geos}
        d.treatment_geos = treatment_geos
        d.control_geos = control_geos
        output_result.append(d)

    return output_result

  @staticmethod
  def _constraint_not_satisfied(parameter_value: float,
                                constraint_lower: float,
                                constraint_upper: float) -> bool:
    """Checks if the parameter value is in the interval [constraint_lower, constraint_upper]."""
    return (parameter_value < constraint_lower) | (
        parameter_value > constraint_upper)

  def design_within_constraints(self, treatment_geos: Set[GeoIndex],
                                control_geos: Set[GeoIndex]):
    """Checks if a set of control/treatment geos passes all constraints.

    Given a set of control and treatment geos we verify that some of their
    metrics are within the bounds specified in TBRMMDesignParameters.

    Args:
      treatment_geos: Set of geo indices referring to the geos in treatment.
      control_geos: Set of geo indices referring to the geos in control.

    Returns:
      False if any specified constraint is not satisfied.
    """
    if self.parameters.volume_ratio_tolerance is not None:
      volume_ratio = (
          self.data.aggregate_geo_share(control_geos)/
          self.data.aggregate_geo_share(treatment_geos))
      if self._constraint_not_satisfied(
          volume_ratio, 1 / (1 + self.parameters.volume_ratio_tolerance),
          1 + self.parameters.volume_ratio_tolerance):
        return False

    if self.parameters.geo_ratio_tolerance is not None:
      geo_ratio = len(control_geos) / len(treatment_geos)
      if self._constraint_not_satisfied(
          geo_ratio, 1 / (1 + self.parameters.geo_ratio_tolerance),
          1 + self.parameters.geo_ratio_tolerance):
        return False

    if self.parameters.treatment_share_range is not None:
      treatment_response_share = self.data.aggregate_geo_share(
          treatment_geos) / self.data.aggregate_geo_share(
              self.geo_assignments.all)
      if self._constraint_not_satisfied(
          treatment_response_share, self.parameters.treatment_share_range[0],
          self.parameters.treatment_share_range[1]):
        return False

    if self.parameters.treatment_geos_range is not None:
      num_treatment_geos = len(treatment_geos)
      if self._constraint_not_satisfied(
          num_treatment_geos, self.parameters.treatment_geos_range[0],
          self.parameters.treatment_geos_range[1]):
        return False

    if self.parameters.control_geos_range is not None:
      num_control_geos = len(control_geos)
      if self._constraint_not_satisfied(num_control_geos,
                                        self.parameters.control_geos_range[0],
                                        self.parameters.control_geos_range[1]):
        return False

    return True

  def greedy_search(self):
    """Searches the Matched Markets for a TBR experiment.

    Uses a greedy hill climbing algorithm to provide recommended 'matched
    markets' experimental group assignments that appear to lead to valid and
    effective TBR models relative to the pretest period.  This is accomplished
    by using a greedy hill climbing alogirhtm that alternates between two
    routines:
    1) Looks for the best set of control geos given the current
       set of treatment geos.
    2) Adds one new geo to the set of treatment geos given
       the current control group.

    See Au (2018) for more details.

    Returns:
      the set of feasible designs found given the design parameters,
        with their corresponding treatment/control groups and score.
    """
    budget_range = self.parameters.budget_range
    results = heapdict.HeapDict(size=self.parameters.n_designs)

    if self.parameters.treatment_geos_range is None:
      n_treatment = len(self.geo_assignments.t)
      max_treatment_size = n_treatment
      n_remaining = len(self.geo_assignments.all) - n_treatment
      if n_remaining == 0:
        max_treatment_size = n_treatment - 1
      self.parameters.treatment_geos_range = (1, max_treatment_size)
    else:
      max_treatment_size = self.parameters.treatment_geos_range[1]

    if self.parameters.control_geos_range is None:
      n_control = len(self.geo_assignments.c)
      max_control_size = n_control
      n_remaining = len(self.geo_assignments.all) - n_control
      if n_remaining == 0:
        max_control_size = n_control - 1
      self.parameters.control_geos_range = (1, max_control_size)

    kappa_0 = len(self.geo_assignments.t_fixed)
    group_star_trt = {kappa_0: self.geo_assignments.t_fixed}
    tmp_diag = TBRMMDiagnostics(np.random.normal(range(100)), self.parameters)
    tmp_diag.x = list(range(len(tmp_diag.y)))
    tmp_score = TBRMMScore(tmp_diag)
    tmp_score.score = tmp_score.score._replace(
        corr_test=0,
        aa_test=0,
        bb_test=0,
        dw_test=0,
        corr=0,
        inv_required_impact=0)
    score_star = {kappa_0: tmp_score}
    group_ctl = self.geo_assignments.c
    if kappa_0 == 0:
      group_star_ctl = {kappa_0: group_ctl}
      needs_matching = False
    else:
      group_star_ctl = {}
      needs_matching = True

    k = kappa_0
    while (k < max_treatment_size) | (needs_matching):
      # Find the best control group given the current treatment group
      if needs_matching:
        r_control = self.geo_assignments.c - (group_ctl | group_star_trt[k])
        r_unassigned = (group_ctl & self.geo_assignments.x) - group_star_trt[k]

        reassignable_geos = r_control | r_unassigned
        treatment_time_series = self.data.aggregate_time_series(
            group_star_trt[k])
        current_design = TBRMMDiagnostics(treatment_time_series,
                                          self.parameters)
        current_design.x = self.data.aggregate_time_series(group_ctl)
        current_score = TBRMMScore(current_design)

        group_ctl_tmp = group_ctl
        for geo in reassignable_geos:
          neighboring_control_group = group_ctl.symmetric_difference([geo])
          # we skip checking constraints for designs with less than the minimum
          # number of treatment geos, or above the maximum number of control
          # geos. Otherwise, we will never be able to augment the size of
          # treatment (to reach a size which would pass the checks) or decrease
          # the size of control
          if (k >= self.parameters.treatment_geos_range[0]) and (
              len(neighboring_control_group) <=
              self.parameters.control_geos_range[1]):
            if (not neighboring_control_group) or (
                not self.design_within_constraints(group_star_trt[k],
                                                   neighboring_control_group)):  # pytype: disable=wrong-arg-types
              continue

          neighbor_design = tbrmmdiagnostics.TBRMMDiagnostics(
              treatment_time_series, self.parameters)
          neighbor_design.x = self.data.aggregate_time_series(
              neighboring_control_group)
          req_impact = neighbor_design.required_impact
          req_budget = req_impact / self.parameters.iroas
          if (budget_range is not None) and (self._constraint_not_satisfied(
              req_budget, budget_range[0], budget_range[1])):
            continue

          score = TBRMMScore(neighbor_design)
          if score > current_score:
            group_ctl_tmp = neighboring_control_group
            current_score = score

        if current_score > TBRMMScore(current_design):
          group_ctl = group_ctl_tmp
        else:
          group_star_ctl[k] = group_ctl_tmp
          score_star[k] = current_score
          needs_matching = False
      # add one geo to treatment given the current control group
      elif k < max_treatment_size:
        r_treatment = self.geo_assignments.t - group_star_trt[k]

        current_score = copy.deepcopy(tmp_score)
        group_trt = group_star_trt[k]
        for geo in r_treatment:
          augmented_treatment_group = group_star_trt[k].union([geo])
          updated_control_group = group_star_ctl[k] - set([geo])
          # see comment on lines 566-567 for the same if statement
          if (k >= self.parameters.treatment_geos_range[0]) and (
              len(updated_control_group) <=
              self.parameters.control_geos_range[1]):
            if (not updated_control_group) or (
                not self.design_within_constraints(augmented_treatment_group,
                                                   updated_control_group)):
              continue
          treatment_time_series = self.data.aggregate_time_series(
              augmented_treatment_group)
          neighbor_design = TBRMMDiagnostics(
              treatment_time_series, self.parameters)
          neighbor_design.x = self.data.aggregate_time_series(
              updated_control_group)
          req_impact = neighbor_design.required_impact
          req_budget = req_impact / self.parameters.iroas
          if (budget_range is not None) and (self._constraint_not_satisfied(
              req_budget, budget_range[0], budget_range[1])):
            continue
          score = TBRMMScore(neighbor_design)
          if score > current_score:
            group_ctl = updated_control_group
            group_trt = augmented_treatment_group
            current_score = score

        group_star_trt[k+1] = group_trt
        k = k + 1
        needs_matching = True

    # if some geos are fixed to treatment, we did not check that the design
    # with treatment group = {all geos fixed in treatment} and control group =
    # {all geos that can be assigned to control} pass the diagnostic tests
    if kappa_0 > 0:
      diagnostic = TBRMMDiagnostics(
          self.data.aggregate_time_series(group_star_trt[kappa_0]),
          self.parameters)
      diagnostic.x = self.data.aggregate_time_series(group_star_ctl[kappa_0])
      req_impact = diagnostic.required_impact
      req_budget = req_impact / self.parameters.iroas
      if (not group_star_ctl[kappa_0]) or (not self.design_within_constraints(
          group_star_trt[kappa_0], group_star_ctl[kappa_0])):
        if (budget_range is not None) and (self._constraint_not_satisfied(
            req_budget, budget_range[0], budget_range[1])):
          group_star_trt.pop(kappa_0, None)
          group_star_ctl.pop(kappa_0, None)
          score_star.pop(kappa_0, None)

    group_star_trt.pop(0, None)
    group_star_ctl.pop(0, None)
    score_star.pop(0, None)
    for k in group_star_trt:
      if self.design_within_constraints(group_star_trt[k], group_star_ctl[k]):
        design_diag = TBRMMDiagnostics(
            self.data.aggregate_time_series(group_star_trt[k]), self.parameters)
        design_diag.x = self.data.aggregate_time_series(group_star_ctl[k])
        design_score = TBRMMScore(design_diag)
        design = TBRMMDesign(
            design_score, group_star_trt[k], group_star_ctl[k],
            copy.deepcopy(design_diag))
        results.push(0, design)

    self._search_results = results
    return self.search_results()
