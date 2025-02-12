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
"""TBR Matched Markets: Design Parameters.
"""
import operator
import typing

from typing import Optional, Tuple

import dataclasses

OptionalFloat = Optional[float]
OptionalInt = Optional[int]
OptionalRange = Optional[Tuple[float, float]]


@dataclasses.dataclass
class TBRMMDesignParameters:
  """User-supplied design parameters and constraints for the desired designs.

  Attributes:
    n_test: An integer >= 1. Number of time points in the future geo
      experiment. Required.
    iroas: A float >= 0.0. Assumed true target iROAS, with the given
      power_level (see the attribute below). Required.
    volume_ratio_tolerance: A float > 0. (Optional) constraint on the ratio of
      the average control group and treatment group volumes.  The ratio is
      constrained within (1/(1 + tol), 1 + tol).
    geo_ratio_tolerance: A float > 0. (Optional) constraint on the ratio of the
      number of geos in the control group and the number of geos in the
      treatment group. The ratio is constrained within (1/(1 + tol), 1 + tol).
    treatment_share_range: A tuple of two floats both strictly between (0, 1).
      (Optional) constraint on the range of the maximum acceptable treatment
      group share of response volume (share of response across geos and across
      the whole time series).
    budget_range: A tuple of two floats >= 0.0. (Optional) constraint on the
      minimum and maximum acceptable *change* in the ad spend budget. This
      change is an increase in a heavy-up experiment, but in a go-dark
      experiment the budget change can also mean reducing the current ad spend
      budget.  In either case budget_range is specified as positive numbers. The
      estimated budget change is equal to the estimated impact divided by the
      (assumed) iROAS causing that impact. Designs implying a budget out of
      range are dismissed.
    treatment_geos_range: A tuple of two integers >= 1. (Optional) range of
      treatment geos to include in the design. If not specified, the whole
      available range will be considered in the search for designs.
    control_geos_range: A tuple of two integers >= 1. (Optional) range of
      control geos to include in the design. If not specified, the whole
      available range will be considered in the search for designs.
    n_geos_max: An integer >= 2. (Optional) maximum number of geos to include in
      the search. If not specified, all available geos (that satisfy other
      constraints) will be included in the search.
    n_pretest_max: An integer >= 3. Maximum number of pretest timepoints to
      include in the time series for the purpose of estimating minimum
      detectable response, correlation and other diagnostics. Default 90.
    n_designs: An integer >= 1. Maximum number of designs to store during the
      search. Default 1.
    rho_max: A float >= 0.9 and < 1.0. Maximum assumed treatment-control
      correlation to use for estimating the Minimum Detectable Response for a
      given treatment group. Default 0.995. The closer to 1, the less likely it
      is that larger geos will be excluded from the search.
    sig_level: A float > 0.0 and < 1.0. Significance level of the one-sided
      interval.
    power_level: A float > 0.0 and < 1.0. Required statistical power. Default
      0.8.
    min_corr: A float >= 0.8 and < 1.0. Minimum acceptable Pearson correlation
      between the treatment and control time series. A correlation less than
      min_corr will automatically disqualify a design. Default 0.8.
    flevel: A float >= 0.9 and < 1.0. Inverse quantile of the f distribution
      parameter 'phi' used in the TBR preanalysis formula.
  """

  _MIN_CORR = 0.8  # Lower bound for min_corr.
  _N_TEST_MIN = 1  # Lower bound for n_test.
  _N_PRETEST_MIN = 3  # Lower bound for n_pretest_min.
  _MIN_IROAS = 0.0  # Minimum acceptable value for iroas.

  _test_functions = {'>': operator.gt,
                     '<': operator.lt,
                     '<=': operator.le,
                     '>=': operator.ge}

  _inverse_op = {'<': '>', '<=': '>=', '>': '<', '>=': '<='}

  n_test: int
  iroas: float
  volume_ratio_tolerance: OptionalFloat = None
  geo_ratio_tolerance: OptionalFloat = None
  treatment_share_range: OptionalRange = None
  budget_range: OptionalRange = None
  treatment_geos_range: OptionalRange = None
  control_geos_range: OptionalRange = None
  n_geos_max: OptionalInt = None
  n_pretest_max: int = 90
  n_designs: int = 1
  sig_level: float = 0.9
  power_level: float = 0.8
  min_corr: float = 0.8
  rho_max: float = 0.995
  flevel: float = 0.9

  def __post_init__(self):
    """Validate the values in the object, output informative error messages.
    """

    self._test_value_vs_threshold('n_test', '>=', self._N_TEST_MIN)
    self._test_value_vs_threshold('iroas', '>=', self._MIN_IROAS)

    self._test_value_vs_threshold('volume_ratio_tolerance', '>', 0.0)
    self._test_value_vs_threshold('geo_ratio_tolerance', '>', 0.0)

    self._test_range(0.0, '<', ('treatment_share_range', '<'), '<', 1.0)
    self._test_range(0.0, '<=', ('budget_range', '<'), '<', float('inf'))
    self._test_range(1, '<=', ('treatment_geos_range', '<='), '<', float('inf'))
    self._test_range(1, '<=', ('control_geos_range', '<='), '<', float('inf'))

    self._test_value_vs_threshold('n_geos_max', '>=', 2)
    self._test_value_vs_threshold('n_pretest_max', '>=', self._N_PRETEST_MIN)
    self._test_value_vs_threshold('n_designs', '>=', 1)

    self._test_value_within_bounds(0.9, '<=', 'rho_max', '<', 1.0)
    self._test_value_within_bounds(0.0, '<', 'sig_level', '<', 1.0)
    self._test_value_within_bounds(0.0, '<', 'power_level', '<', 1.0)
    self._test_value_within_bounds(self._MIN_CORR, '<=', 'min_corr', '<', 1.0)
    self._test_value_within_bounds(0.9, '<=', 'flevel', '<', 1.0)

  def __eq__(self, other: 'TBRMMDesignParameters'):
    """Checks if two instances of TMDesignParameters are equal."""
    if not isinstance(other, TBRMMDesignParameters):
      raise NotImplementedError('Cannot compare instance of '
                                'TBRMMDesignParameters '
                                f'with instance of {type(other)}')
    if self is other:
      return True

    return dataclasses.asdict(self) == dataclasses.asdict(other)

  def _is_optional(self, attr):
    """Check if a given attribute is optional or not.

    Args:
      attr: Name of the attribute.

    Returns:
      True if and only if the type hint of the attribute is optional.
    """

    type_hint = typing.get_type_hints(self)[attr]
    return type_hint in {OptionalRange, OptionalFloat, OptionalInt}

  def _test_value_vs_threshold(self, attr, op, bound):
    """Test that the value of the attribute satisfies the threshold.

    Args:
      attr: Name of the attribute.
      op: String. Operator, one of '<', '>', '<=', '>='.
      bound: Float. Threshold to test the attribute value against.

    Raises:
      ValueError: if value is out of the accepted range, non-numeric, bad type,
        or if it is not integer-valued when the bound is.
    """

    value = getattr(self, attr)

    specified = value is not None
    if specified:
      # 'None' is taken care by the test below.
      value_ok = isinstance(value, int) or isinstance(value, float)
      if not value_ok:
        raise ValueError('{} must be numeric'.format(attr))
    elif self._is_optional(attr):
      return None

    testf = self._test_functions[op]
    test_ok = specified and testf(value, bound)
    if not test_ok:
      raise ValueError('{} must be {} {}'.format(attr, op, bound))
    if isinstance(bound, int) and int(value) != value:
      raise ValueError('{} must be an integer'.format(attr))

  def _test_value_within_bounds(self, lower, op1, attr, op2, upper):
    """Test that the value of the attribute is within the given bounds.

    Args:
      lower: Float. Lower bound.
      op1: String. Operator to apply between lower and attribute value.
      attr: Name of the attribute.
      op2: String. Operator to apply between the value and the upper bound.
      upper: Float. Upper bound.

    Raises:
      ValueError: if value is out of the accepted range, non-numeric, bad type,
        or if it is not integer-valued when the bound is.
    """

    value = getattr(self, attr)

    specified = value is not None
    if specified:
      # 'None' is taken care by the test below.
      value_ok = isinstance(value, int) or isinstance(value, float)
      if not value_ok:
        raise ValueError('{} must be numeric'.format(attr))
    elif self._is_optional(attr):
      return None

    testf1 = self._test_functions[op1]
    testf2 = self._test_functions[op2]
    test_ok = specified and testf1(lower, value) and testf2(value, upper)
    if not test_ok:
      inv_op1 = self._inverse_op[op1]
      template = '{} must be {} {} and {} {}'
      raise ValueError(template.format(attr, inv_op1, lower, op2, upper))
    if isinstance(lower, int) and int(value) != value:
      raise ValueError('{} must be an integer'.format(attr))

  def _test_range(self, lower, op1, attr_op, op2, upper):
    """Test that the value (range) is within the bounds.

    Args:
      lower: Float. Lower bound.
      op1: String. Operator to apply between lower bound and the lower value
        of the attribute which must be a tuple with two numbers.
      attr_op: A tuple with the attribute name and an operator. The operator
        is applied to the test between the two numbers of the attribute value.
      op2: String. Operator to apply between the upper value of the attribute
        and the upper bound.
      upper: Float. Upper bound. Can be float('inf').

    Raises:
      ValueError: if the attribute values are out of the accepted range, the
        attribute tuple order is incorrect, non-numeric, bad type, or if it is
        not integer-valued in the case the lower bound is.
    """

    attr, op3 = attr_op
    value = getattr(self, attr)
    optional = self._is_optional(attr)
    specified = value is not None
    if optional and not specified:
      return None
    value_ok = (isinstance(value, tuple) and
                len(value) == 2 and
                all(isinstance(x, int) or isinstance(x, float) for x in value))
    if value_ok:
      testf1 = self._test_functions[op1]
      testf2 = self._test_functions[op2]
      lower_range, upper_range = value
      test_ok = testf1(lower, lower_range) and testf2(upper_range, upper)
      if test_ok:
        testf3 = self._test_functions[op3]
        range_ok = testf3(lower_range, upper_range)
        if not range_ok:
          template = 'Lower bound of {} must be {} upper bound'
          raise ValueError(template.format(attr, op3))
        elif (isinstance(lower, int) and
              (int(lower_range) != lower_range or
               int(upper_range) != upper_range)):
          raise ValueError('{} must be integers'.format(attr))
      else:
        inv_op1 = self. _inverse_op[op1]
        if upper is float('inf'):
          template = '{0} must be {1} {2}'
        else:
          template = '{} must be {} {} and {} {}'
          raise ValueError(template.format(attr, inv_op1, lower, op2, upper))
    else:  # Not value_ok.
      raise ValueError('{} requires a range of two numbers'.format(attr))
