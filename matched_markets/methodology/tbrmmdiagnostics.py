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
import collections
import functools
from typing import List, Optional

from matched_markets.methodology import tbrmmdesignparameters
import numpy as np
from scipy import stats

TBRMMDesignParameters = tbrmmdesignparameters.TBRMMDesignParameters
Vector = List[float]
LinregResult = collections.namedtuple('LinregResult',
                                      ['a', 'b', 'sigma', 'resid'])
AATestResult = collections.namedtuple('AATestResult',
                                      ['test_ok', 'bounds', 'prob'])
BBTestResult = collections.namedtuple('BBTestResult',
                                      ['test_ok', 'abscumresid', 'bounds'])
DWTestResult = collections.namedtuple('DWTestResult', ['test_ok', 'dwstat'])
TBRFit = collections.namedtuple('TBRFit',
                                ['estimate', 'cihw', 'sigma', 'scale'])


class TBRMMDiagnostics:
  """Evaluate characteristics of a given TBR Matched Markets design.

  Purpose:

  1. Calculate the required incremental response to obtain a statistically
     significant result at a given power level. Can estimate the quantity given
     only the treament group time series and an assumed correlation between the
     control and treatment time series.

  2. Perform the following tests:

    a. Correlation test. The Pearson correlation between the control and
       treatment time series must be at least a given minimum limit. Rationale:
       a low correlation implies that there is no possible causal link between
       the two groups.

    b. Structural break test. Check whether the given time series has signs of
       structural breaks.

    c. Autocorrelation test. Passes if the Durbin-Watson statistic is within the
       conventionally acceptable bounds.

  Example of usage:

     # Create an instance, passing the treatment time series and a
     # TBRMMDesignParameters object.

     diag = TBRMMDiagnostics(y, par)

     # Next, estimate the impact given only an optimistic guess of 'rho'
     # (correlation).

     impact_estimate = diag.estimate_required_impact(rho)

     # At this point, we may test if the required impact is within given
     # constraints. If it is not, no need to test for different control geo
     # assignments.

     # If the impact is within constraints, generate a control group and produce
     # its time series 'x'. Store it:

     diag.x = x

     # Then obtain 'corr' (correlation between x and y) and 'required_impact'
     # which are available as properties.

     corr = diag.corr
     impact = diag.required_impact

     # If impact is within constraints, next run other diagnostic tests:

     if diag.tests_ok:
       ...

     # The design is acceptable if and only if tests_ok is True.

     # If a new 'x' is stored into the object, the corr, required_impact and
     # other properties are reset.


  Attributes:
    x: Control time series.
    y: Treatment time series.
    corr: Correlation between x and y.
    required_impact: Estimated required impact given (x, y).
    pretestfit: Linear regression fit, with residuals.
    bbtest: Brownian Bridge test result.
    tests_ok: True if and only if all tests pass. None if tests have not been
      run.
  """
  # Minimum required number of timepoints in x, y.
  _min_timepoints = 3
  # Brownian Bridge bound multiplier for the TBR stability check.
  _bb_bound = 3.0
  # Durbin-Watson range of acceptable values.
  _dw_range = (1.5, 2.5)
  # A/A test threshold probability level.
  _aa_threshold_prob = 0.2

  _x = None  # Control time series.
  _y = None  # Treatment time series.
  _corr = None  # Control-Treatment correlation.
  _required_impact = None  # Estimated required impact given x, y.
  _pretestfit = None  # Linear regression fit.
  _aatest = None  # A/A Test result.
  _bbtest = None  # Brownian Bridge test.
  _dwtest = None  # Durbin-Watson test result.
  _par = None  # TBRMMDesignParameters object.
  _tests_ok = None  # True iff all tests pass.
  _y_mean = None  # Mean of y.
  _x_mean = None  # Mean of x.

  def __init__(self, y: Vector, par: TBRMMDesignParameters):
    """Initializes the object.

    Args:
      y: Vector of floats. Treatment group time series.
      par: A TBRDesignParameters object.
    """
    self.y = y
    self._par = par

  @property
  def y(self):
    return self._y

  @y.setter
  def y(self, value: Vector):
    y = np.array(value)
    if y.ndim != 1:
      raise ValueError('y must be a one-dimensional vector')
    if len(y) < self._min_timepoints:
      raise ValueError('y must have length >= %d' % self._min_timepoints)
    self._y = y
    self._y_mean = y.mean()
    self.x = None

  @property
  def x(self):
    return self._x

  @x.setter
  def x(self, value: Vector):
    if value is None:
      x = None
      self._x_mean = None
    else:
      x = np.array(value)
      if x.ndim != 1:
        raise ValueError('x must be a one-dimensional vector')
      if len(x) != len(self._y):
        raise ValueError('x must have the same length as y (%d)' % len(self._y))
      self._x_mean = x.mean()
    self._x = x
    self._corr = None
    self._required_impact = None
    self._pretestfit = None
    self._aatest = None
    self._bbtest = None
    self._dwtest = None

  @property
  def corr(self):
    """Correlation between x and y.

    Returns:
      Correlation between x and y, None if x is not available.
    """
    if self._x is None:
      return None

    if self._corr is None:
      self._corr = np.corrcoef(self._x, self.y)[0, 1]
    return self._corr

  # The cache ensures values are calculated only once to save time.
  @functools.lru_cache()
  def _brownian_bridge_bounds(self, n: int) -> Vector:
    """Calculate the Brownian Bridge bounds for a given pretest period length.

    Used for the Brownian Bridge test for checking for surprising values in
    standardized residuals from regression analysis.

    Args:
      n: Length of vector.

    Returns:
      The one-dimensional vector of length n - 1.
    """
    n_range = np.arange(1, n)  # 1 , ..., n - 1.
    bound = TBRMMDiagnostics._bb_bound
    bb = bound * np.sqrt(n_range * (1.0 - n_range / float(n)))
    return bb

  @functools.lru_cache()
  def _impact_estimate(
      self,
      n_test: int,
      n: int,
      flevel: float,
      sig_level: float,
      power_level: float) -> float:
    """Estimate the required incremental impact, without the sigma term.

    Args:
      n_test: Number of test period time points.
      n: Number of pretest period time points.
      flevel: Inverse quantile of the F(1, n - 1) distribution.
      sig_level: Significance level (one-sided).
      power_level: Required statistical power.

    Returns:
      The value of the impact formula without the residual s.d. (sigma) term.
    """
    phi = stats.f(dfn=1, dfd=n - 1).ppf(flevel)
    tq_sig = stats.t.ppf(sig_level, df=n - 2)
    tq_pow = stats.t.ppf(power_level, df=n - 2)
    sq = np.sqrt(phi * (n + 1) / (n * n_test * (n - 1)) + 1 / n + 1 / n_test)
    term = (tq_sig + tq_pow) * n_test * sq
    return term

  def estimate_required_impact(self, corr) -> float:
    """Estimate the required incremental impact in a TBR analysis.

    Args:
      corr: Pearson correlation coefficient between the control and treatment
        time series.

    Returns:
      An estimate of the required impact.

    Raises:
      ValueError if correlation is not between -1 and 1. Correlation of -1 or 1
      are not accepted as this cannot happen in real life geo experiment data.
    """
    if corr <= -1 or corr >= 1:
      raise ValueError('corr must be between -1 and 1')

    par = self._par
    n = len(self.y)
    term = self._impact_estimate(par.n_test,
                                 n,
                                 par.flevel,
                                 par.sig_level,
                                 par.power_level)
    sigma = np.std(self.y, ddof=2) * np.sqrt(1 - corr ** 2)
    impact = term * sigma
    return impact

  @property
  def required_impact(self) -> Optional[float]:
    """The required incremental impact.

    Calculates the Pearson correlation between the control and treatment time
    series and uses that to estimate the actual required incremental
    impact. Uses the previously saved treatment time series and stores the
    control time series into the instance for later calculations.

    Returns:
      The estimated required incremental impact. None if correlation is not
      available.
    """
    if self._required_impact is None:
      corr = self.corr
      if corr is None:
        return None
      self._required_impact = self.estimate_required_impact(self.corr)

    return self._required_impact

  @property
  def pretestfit(self) -> Optional[LinregResult]:
    """Simple linear regression ('OLS') fit of the pretest data.

    Model: y = a + bx + error, where error ~ N(0, sigma^2), 'OLS'.

    Returns:
      None if 'x' is not set. Otherwise a namedtuple object with (a, b, sigma,
      residuals), where a, b are the regression parameter estimates, sigma the
      residual standard deviation, and the non-standardized residuals.
    """
    x = self._x
    if x is None:
      return None
    elif self._pretestfit is None:
      y = self._y
      try:
        b, a, *_ = stats.linregress(x, y)
      except ValueError:
        b, a = np.nan, np.nan
      resid = y - a - b * x
      sigma = np.std(resid, ddof=2)
      self._pretestfit = LinregResult(a, b, sigma, resid)

    return self._pretestfit

  @property
  def bbtest(self) -> Optional[BBTestResult]:
    """Brownian bridge test of residuals.

    Checks if the cumulative residuals of a (successful) regression fit exceed
    the Brownian Bridge (BB) at any point; if residuals exceed the BB or if the
    regression fit was not possible, the test fails.

    Returns:
      None if 'x' is not set. Otherwise a namedtuple object with (test_ok,
      abscumresid, bounds).
    """

    if self._x is None:
      return None

    if self._bbtest is None:
      a, _, sigma, resid = self.pretestfit  # pytype: disable=attribute-error  # strict-namedtuple-checks
      # Failure to fit the regression implies failure of the test.
      if np.isnan(a):
        self._bbtest = BBTestResult(False, None, None)
      else:
        n = len(resid)
        std_resid = resid / sigma
        # Drop the last item of the cumulative sum array (theoretically zero).
        abs_cum_std_resid = abs(np.cumsum(std_resid)[:-1])
        bb_bounds = self._brownian_bridge_bounds(n)
        test_ok = not any(abs_cum_std_resid > bb_bounds)
        self._bbtest = BBTestResult(test_ok, abs_cum_std_resid, bb_bounds)

    return self._bbtest

  def tbrfit(self, xt: float, yt: float) -> Optional[TBRFit]:
    """Compute the TBR point estimate and credible interval half-width.

    Point estimate: n_test * (dy - b * dx)
    Credible interval half-width: tq * n_test * sigma *
      sqrt((1 + dx^2/var(x, ddof=0))/n + 1/n_test)

    Here, x = 'control', y = 'treatment', where
      dx = test period mean of x - pretest period mean of y
      dy = test period mean of y - pretest period mean of y
      b = coefficient of 'x' in the TBR regression equation y = a + bx + e.
      n_test = number of test period time points.
      tq = a quantile of the t-distribution with n-2 degrees of freedom
      sigma = residual standard deviation.

    Args:
      xt: Test period mean of the Control group time series.
      yt: Test period mean of the Treatment group time series.

    Returns:
      None if the attribute 'pretestfit' is None. Otherwise a namedtuple object
      'TBRFit' with (point, cihw), corresponding to the point estimate and the
      credible interval half-width.
    """
    pretestfit = self.pretestfit
    if pretestfit is None:
      return None

    par = self._par
    n_test = par.n_test
    _, b, sigma, _ = pretestfit
    n = len(self._x)
    xn = self._x_mean
    yn = self._y_mean
    dx = xt - xn
    dy = yt - yn
    estimate = n_test * (dy - b * dx)

    dv = dx ** 2 / np.var(self._x, ddof=0)
    tq_sig = stats.t.ppf(par.sig_level, df=n - 2)
    scale = n_test * sigma * np.sqrt((1 + dv) / n + 1 / n_test)
    cihw = tq_sig * scale

    return TBRFit(estimate, cihw, sigma, scale)

  @property
  def dwtest(self) -> Optional[DWTestResult]:
    """Durbin-Watson test for autocorrelation.

    Returns:
      A DWTestResult namedtuple, with test_ok (True/False), and dwstat (D-W
      statistic, float). None if 'pretestfit' is None.
    """
    if self._dwtest is None:
      pretestfit = self.pretestfit
      if pretestfit is None:
        return None
      resid = pretestfit.resid
      d = resid[1:] - resid[:-1]
      dwstat = np.sum(d ** 2) / np.sum(resid ** 2)
      dw_min, dw_max = self._dw_range
      test_ok = dw_min < dwstat < dw_max
      self._dwtest = DWTestResult(test_ok, dwstat)
    return self._dwtest

  @property
  def aatest(self) -> Optional[AATestResult]:
    """An A/A test, testing for a high probability of a false positive result.


    Returns:
      An AATestResult namedtuple, with test_ok (True/False), bounds (lower and
      upper credible interval, tuple), prob (probability of false positive
      result). None if 'x' is not set.
    """
    if self._aatest is not None:
      return self._aatest
    x = self._x
    if x is None:
      return None

    y = self._y
    n_test = self._par.n_test
    n_pretest = len(y) - n_test
    if n_pretest < self._min_timepoints:
      return AATestResult(None, None, None)
    diag = TBRMMDiagnostics(y[:-n_test], self._par)
    diag.x = x[:-n_test]
    yt = y[-n_test:].mean()
    xt = x[-n_test:].mean()
    estimate, cihw, sigma, _ = diag.tbrfit(xt, yt)  # pytype: disable=attribute-error  # strict-namedtuple-checks
    bounds = (estimate - cihw, estimate + cihw)
    lower, upper = bounds
    if lower * upper < 0:
      # Interval contains zero, test passes.
      ok = True
      prob_sig_result = None  # No need to compute.
    else:
      # Interval outside zero. Check if this is a significant event.
      # Conservative estimate of the true mean: interval bound closest to zero.
      true_mean = min(abs(lower), abs(upper))
      tq_sig = cihw / sigma
      posterior_scale = sigma * np.sqrt(1 / n_pretest + 1 / n_test)
      tq1 = tq_sig - true_mean / posterior_scale
      tq2 = -tq_sig - true_mean / posterior_scale
      prob_sig_result = (1 - stats.t.cdf(tq1, df=n_pretest - 2) +
                         stats.t.cdf(tq2, df=n_pretest - 2))
      ok = prob_sig_result <= self._aa_threshold_prob

    self._aatest = AATestResult(ok, bounds, prob_sig_result)
    return self._aatest

  @property
  def corr_test(self) -> Optional[bool]:
    """True iff the correlation test passes."""
    corr = self.corr
    if corr is None:
      return None
    return corr >= self._par.min_corr

  @property
  def tests_ok(self) -> Optional[bool]:
    """The value of the joint diagnostic test.

    Returns:
      True iff all individual diagnostic tests pass, False if any individual
      test fails, None if any test returns None. A value of None may be due to
      (1) 'x' is undefined causing corr_test to return None; or (2)
      aatest.test_ok is None.
    """
    if self._tests_ok is None:
      self._tests_ok = (self.corr_test and
                        self.bbtest.test_ok and
                        self.dwtest.test_ok and
                        self.aatest.test_ok)
    return self._tests_ok

  def __repr__(self):
    return 'TBRMMDiagnostics(tests_ok={}, corr={})'.format(self._tests_ok,
                                                           self._corr)
