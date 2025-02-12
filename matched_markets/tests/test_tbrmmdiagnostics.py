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
"""Test TBR Matched Markets Diagnostics.
"""
from matched_markets.methodology import tbrmmdesignparameters
from matched_markets.methodology import tbrmmdiagnostics
import numpy as np

import unittest

TBRMMDesignParameters = tbrmmdesignparameters.TBRMMDesignParameters
TBRMMDiagnostics = tbrmmdiagnostics.TBRMMDiagnostics


class TBRMMDiagnosticsTest(unittest.TestCase):

  def setUp(self):
    """Set up a valid Geo Eligibility data frame."""
    super().setUp()
    self.par = TBRMMDesignParameters(n_test=14, iroas=3.0, min_corr=0.8,
                                     sig_level=0.9, power_level=0.5,
                                     flevel=0.9)
    # Corr(x, y) = 0.900.
    y = (-1.4, -0.8, -0.8, -0.7, -0.6, -0.5, -0.5, -0.5, -0.2,
         -0.2, -0.0, 0.3, 0.4, 0.5, 0.6, 0.6, 1.0, 1.3, 1.6, 1.9, 2.2)
    x = (-0.8, -1.1, -1.2, -0.9, -1.0, -0.3, -0.4, -1.2, 0.5, 0.0,
         0.1, -0.5, 0.4, 0.6, 1.1, -0.1, 1.3, 1.4, 1.8, 1.4, 1.6)
    self.x = np.array(x)
    self.y = np.array(y)
    self.x_error_msg = 'x must be a one-dimensional vector'
    self.y_error_msg = 'y must be a one-dimensional vector'
    self.twodim_array = np.array([list(range(10)), list(range(10))])
    self.corr = np.corrcoef(x, y)[0, 1]
    self.xy_short = (1, 2)  # Minimum length = 3.

  def testInit(self):
    """The object must be properly initialized."""
    obj = TBRMMDiagnostics(self.y, self.par)
    self.assertTrue(all(obj.y == self.y))
    self.assertIsNone(obj.x)
    self.assertIsNone(obj.corr)
    self.assertIsNone(obj.required_impact)

  def testYPropertySetter(self):
    """The y property setter works."""
    obj = TBRMMDiagnostics(self.y, self.par)
    obj.y = self.x  # Change value.
    self.assertTrue(all(obj.y == self.x))

  def testYPropertyDimension(self):
    """The y property requires a (1-dimensional) vector."""
    with self.assertRaisesRegex(ValueError, self.y_error_msg):
      TBRMMDiagnostics(self.twodim_array, self.par)

    obj = TBRMMDiagnostics(self.y, self.par)
    with self.assertRaisesRegex(ValueError, self.y_error_msg):
      obj.y = self.twodim_array

  def testYPropertyLength(self):
    """The y property must satisfy a minimum length requirement."""
    with self.assertRaisesRegex(ValueError, 'y must have length >= 3'):
      TBRMMDiagnostics(self.xy_short, self.par)

  def testYPropertyNoneValue(self):
    """The y property disallows None."""
    with self.assertRaisesRegex(ValueError, self.y_error_msg):
      TBRMMDiagnostics(None, self.par)

    obj = TBRMMDiagnostics(self.y, self.par)
    with self.assertRaisesRegex(ValueError, self.y_error_msg):
      obj.y = None

  def testXProperty(self):
    """The x property setter/getter works."""
    obj = TBRMMDiagnostics(self.y, self.par)
    obj.x = self.x
    self.assertTrue(all(obj.x == self.x))

  def testXPropertyLength(self):
    """The x property must have length = length of y."""
    with self.assertRaisesRegex(
        ValueError,
        r'x must have the same length as y \(21\)'):
      obj = TBRMMDiagnostics(self.y, self.par)
      obj.x = self.xy_short

  def testXPropertyBadValue(self):
    """The x property requires a (1-dimensional) vector."""
    obj = TBRMMDiagnostics(self.y, self.par)
    with self.assertRaisesRegex(ValueError, self.x_error_msg):
      obj.x = self.twodim_array

  def testXPropertyNoneOk(self):
    """The x property can be set to None."""
    obj = TBRMMDiagnostics(self.y, self.par)
    obj.x = self.x
    self.assertIsNotNone(obj.x)
    obj.x = None
    self.assertIsNone(obj.x)

  def testCorrProperty(self):
    """The corr property returns the correlation if x is set."""
    obj = TBRMMDiagnostics(self.y, self.par)
    obj.x = self.x
    self.assertAlmostEqual(obj.corr, 0.9, places=3)

  def testCorrPropertyResetsToNone(self):
    """The corr property returns None if x is reset to None."""
    obj = TBRMMDiagnostics(self.y, self.par)
    obj.x = self.x
    self.assertIsNotNone(obj.corr)
    obj.x = None
    self.assertIsNone(obj.corr)

  def testEstimateImpactCorrOutOfBound(self):
    """Bad values of corr are caught."""
    obj = TBRMMDiagnostics(self.y, self.par)
    for bad_corr in (-1, 1):
      with self.assertRaisesRegex(ValueError, 'corr must be between -1 and 1'):
        obj.estimate_required_impact(bad_corr)


class ImpactEstimateTest(TBRMMDiagnosticsTest):

  # tuples = (n_test, n, flevel, sig_level, power_level)
  # for the estimate without the sigma term.
  parameters = {1: (14, 21, 0.90, 0.90, 0.80),
                2: (28, 21, 0.90, 0.90, 0.80),
                3: (14, 14, 0.90, 0.90, 0.80),
                4: (14, 21, 0.99, 0.90, 0.80),
                5: (14, 21, 0.90, 0.95, 0.80),
                6: (14, 21, 0.90, 0.90, 0.90)}

  correct_values = {1: 11.055, 2: 18.272, 3: 12.533,
                    4: 11.841, 5: 13.083, 6: 13.413}

  def setUp(self):
    """Set up a valid Geo Eligibility data frame."""
    super().setUp()
    self.obj = TBRMMDiagnostics(self.y, self.par)

  def testRequiredImpactGivenCorr(self):
    """The impact estimates are correctly calculated.

    The formula can be decomposed into 2 components:

      impact = sigma(corr, y) * f(n_test, n, flevel, sig_level, power_level)

    where

      sigma(corr, y) = sqrt(y, ddof=2) *  sqrt(1 - corr ** 2)

    We'll test the function f() against known values, and check further that the
    impact function varies as expected based on different values of y and corr.
    """
    for corr in (0.5, 0.9):
      for key, params in self.parameters.items():
        n_test, n, flevel, sig_level, power_level = params
        y = self.y[:n]
        par = TBRMMDesignParameters(n_test=n_test, iroas=3.0, flevel=flevel,
                                    sig_level=sig_level,
                                    power_level=power_level)
        diag = TBRMMDiagnostics(y, par)
        sigma = np.std(y, ddof=2) * np.sqrt(1 - corr ** 2)
        correct_estimate = self.correct_values[key] * sigma
        self.assertAlmostEqual(diag.estimate_required_impact(corr),
                               correct_estimate,
                               places=3)

  def testRequiredImpactProperty(self):
    """Required impact must be correctly calculated, given 'x'."""
    diag = self.obj
    diag.x = self.x
    self.assertEqual(
        diag.required_impact,
        diag.estimate_required_impact(diag.corr))

  def testRequiredImpactPropertyResets(self):
    """Required impact resets if x is reset."""
    diag = self.obj
    diag.x = self.x
    diag.x = None
    self.assertIsNone(diag.required_impact)


class PretestFitTest(TBRMMDiagnosticsTest):

  def setUp(self):
    super().setUp()
    obj = TBRMMDiagnostics(self.y, self.par)
    obj.x = self.x
    self.obj = obj

    # The expected linear regression estimates.
    #
    # The linear regression y = a + bx + epsilon yields:
    # b = corr(x, y) * sqrt(var(y) / var(x))
    # a = mean(y) - b * mean(x)
    # sigma = sd(y, ddof=2) * sqrt(1 - rho ** 2)

    self.sigma = np.std(obj.y, ddof=2) * np.sqrt(1 - self.corr ** 2)
    self.b = 0.900 * np.std(obj.y) / np.std(obj.x)
    self.a = np.mean(obj.y) - self.b * np.mean(obj.x)
    self.resid = obj.y - self.a - self.b * obj.x

  def testNoneIfXIsNone(self):
    """The pretestfit attribute must return None if x is None."""
    obj = self.obj
    obj.x = None
    self.assertIsNone(self.obj.pretestfit)

  def testResultIsNamedTuple(self):
    """The pretestfit result is a named tuple."""
    result = self.obj.pretestfit
    a, b, sigma, resid = result
    self.assertIsInstance(result, tuple)
    self.assertIs(result.a, a)
    self.assertIs(result.b, b)
    self.assertIs(result.sigma, sigma)
    self.assertIs(result.resid, resid)

  def testAB(self):
    """The pretestfit values (a, b) must be correctly estimated."""
    a, b, sigma, resid = self.obj.pretestfit
    self.assertAlmostEqual(a, self.a, places=3)
    self.assertAlmostEqual(b, self.b, places=3)
    self.assertAlmostEqual(sigma, self.sigma, places=3)
    self.assertTrue(len(resid) == len(self.x))  # pylint: disable=g-generic-assert
    # The function assert_allclose returns None iff the numbers match.
    self.assertIsNone(np.testing.assert_allclose(resid, self.resid, atol=1e-3))


class BrownianBridgeBTest(TBRMMDiagnosticsTest):

  def setUp(self):
    super().setUp()
    obj = TBRMMDiagnostics(self.y, self.par)
    obj.x = self.x
    self.obj = obj

  def testResultIsANamedTuple(self):
    """The bbtest result is a named tuple."""
    result = self.obj.bbtest
    test_ok, abscumresid, bounds = result
    self.assertIsInstance(result, tuple)
    self.assertIs(result.test_ok, test_ok)
    self.assertIs(result.abscumresid, abscumresid)
    self.assertIs(result.bounds, bounds)

  def testGoodResult(self):
    """The default result is True."""
    result = self.obj.bbtest
    self.assertTrue(result.test_ok)

  def testBadResult(self):
    """An outlier causes the test go False."""
    obj = TBRMMDiagnostics(self.y, self.par)
    x = list(self.x)
    x[10] = 10.0  # Make an outlier.
    obj.x = x
    result = obj.bbtest
    self.assertFalse(result.test_ok)


class TBRFitTest(TBRMMDiagnosticsTest):
  """Method tbrfit."""

  def setUp(self):
    super().setUp()
    obj = TBRMMDiagnostics(self.y, self.par)
    self.obj = obj

  def testMandatoryArguments(self):
    """The two arguments xt, yt are mandatatory."""
    with self.assertRaisesRegex(
        TypeError,
        r'missing 2 required positional arguments: \'xt\' and \'yt\''):
      self.obj.tbrfit()

  def testNoX(self):
    """The value is none if 'x' is not set."""
    self.assertIsNone(self.obj.tbrfit(0, 0))

  def testValue(self):
    """The value is a named tuple of length 2."""
    obj = self.obj
    obj.x = self.x
    result = obj.tbrfit(0, 0)
    self.assertTrue(len(result) == 4)  # pylint: disable=g-generic-assert
    self.assertEqual(result.estimate, result[0])
    self.assertEqual(result.cihw, result[1])

  def testCalculation(self):
    """The Credible interval half-width and the estimate are correct."""
    # Structure: ((n_test, sig_level, xt, yt), true_cihw, true_estimate).
    actual_results = {1: ((14, 0.9, 0, 0), 2.8047, -1.225),
                      2: ((28, 0.9, 0, 0), 4.7001, -2.45),
                      3: ((14, 0.8, 0, 0), 1.8187, -1.225),
                      4: ((14, 0.9, 10, 0), 18.0534, -123.725),
                      5: ((14, 0.9, 0, 10), 2.8047, 138.775)}

    for value in actual_results.values():
      args, cihw, estimate = value
      n_test, sig_level, xt, yt = args
      par = TBRMMDesignParameters(n_test=n_test, iroas=1.0,
                                  sig_level=sig_level)
      obj = TBRMMDiagnostics(self.y, par)
      obj.x = self.x
      self.assertAlmostEqual(obj.tbrfit(xt, yt).cihw, cihw, places=3)
      self.assertAlmostEqual(obj.tbrfit(xt, yt).estimate, estimate, places=3)


class DWTestTest(TBRMMDiagnosticsTest):
  """Property dwtest."""

  def setUp(self):
    super().setUp()
    self.x2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

  def testNoX(self):
    """The value is None if 'x' is not set."""
    diag = TBRMMDiagnostics(self.y, self.par)
    self.assertIsNone(diag.dwtest)

  def testDWStatistic(self):
    """The result is a DWTestResult diagect."""
    diag = TBRMMDiagnostics(self.y, self.par)
    diag.x = self.x
    self.assertIsInstance(diag.dwtest, tbrmmdiagnostics.DWTestResult)

  def testDWTestFailsBelow(self):
    """The D-W test fails for dwstat < 1.5."""
    y = [-1, 0, 1, 2, 3, 12.8, 3, 2, 1, 0]
    diag = TBRMMDiagnostics(y, self.par)
    diag.x = self.x2
    self.assertAlmostEqual(diag.dwtest.dwstat, 1.4996, places=4)
    self.assertFalse(diag.dwtest.test_ok)

  def testDWTestSucceedsLowerBound(self):
    """The D-W test succeeds for dwstat >= 1.5 and <= 2.5."""
    # A data set with autocorrelation, dw statistic 0.43.
    y = [-1, 0, 1, 2, 3, 12.9, 3, 2, 1, 0]
    diag = TBRMMDiagnostics(y, self.par)
    diag.x = self.x2
    self.assertAlmostEqual(diag.dwtest.dwstat, 1.5058, places=4)
    self.assertTrue(diag.dwtest.test_ok)

  def testDWTestSucceedsUpperBound(self):
    """The D-W test succeeds for dwstat >= 1.5 and <= 2.5."""
    y = [-1, 7.3, 1, 2, 3, 4, 3, 2, 1, 0]
    diag = TBRMMDiagnostics(y, self.par)
    diag.x = self.x2
    self.assertAlmostEqual(diag.dwtest.dwstat, 2.4974, places=4)
    self.assertTrue(diag.dwtest.test_ok)

  def testDWTestFailsAbove(self):
    """The D-W test fails for > 2.5."""
    y = [-1, 7.4, 1, 2, 3, 4, 3, 2, 1, 0]
    diag = TBRMMDiagnostics(y, self.par)
    diag.x = self.x2
    self.assertAlmostEqual(diag.dwtest.dwstat, 2.5119, places=4)
    self.assertFalse(diag.dwtest.test_ok)


class AATestTest(TBRMMDiagnosticsTest):
  """Property aatest."""

  def setUp(self):
    super().setUp()
    # Correlated time series without effect.
    self.x2 = np.array([132.5, 87.8, 89.4, 78.5, 117.3, 54.0, 134.9, 84.8,
                        106.4, 95.0, 129.2, 58.8, 93.6, 92.3, 122.7, 78.0, 96.6,
                        82.4, 100.8, 111.7, 78.0])
    self.y2 = np.array([487.9, 393.6, 388.8, 375.0, 420.9, 305.5, 451.1, 364.2,
                        423.4, 376.2, 450.5, 303.9, 370.3, 371.2, 445.1, 333.7,
                        397.9, 398.0, 416.4, 419.6, 338.2])
    self.n_test = 7
    self.par = TBRMMDesignParameters(n_test=self.n_test, iroas=1.0,
                                     sig_level=0.9)

  def testNoX(self):
    """The value is None if 'x' is not set."""
    diag = TBRMMDiagnostics(self.y, self.par)
    self.assertIsNone(diag.aatest)

  def testTooFewPretestDatapoints(self):
    """test_ok is None if the number of time points is < 2 + n_test."""
    n_pretest = 2
    n_test = 7
    y = list(range(n_pretest + n_test))
    x = y
    par = TBRMMDesignParameters(n_test=n_test, iroas=1.0)
    obj = TBRMMDiagnostics(y, par)
    obj.x = x
    self.assertIsNone(obj.aatest.test_ok)

  def testBounds(self):
    """The lower-upper bounds are correctly calculated.."""
    diag = TBRMMDiagnostics(self.y2, self.par)
    diag.x = self.x2
    self.assertAlmostEqual(diag.aatest.bounds[0], -45.6, places=1)
    self.assertAlmostEqual(diag.aatest.bounds[1], 89.0, places=1)

  def testOkAndProbIsNoneIfBoundsEncloseZero(self):
    """Test succeeds & prob is not calculated if bounds enclose zero."""
    diag = TBRMMDiagnostics(self.y2, self.par)
    diag.x = self.x2
    self.assertLess(diag.aatest.bounds[0], 0)
    self.assertGreater(diag.aatest.bounds[1], 0)
    self.assertTrue(diag.aatest.test_ok)
    self.assertIsNone(diag.aatest.prob)

  def testProbIsCalculated(self):
    """Prob is calculated if lower bound is above zero."""
    n_test = self.n_test
    y = self.y2
    y[-n_test:] = y[-n_test:] + 9.0  # Add an effect.
    diag = TBRMMDiagnostics(y, self.par)
    diag.x = self.x2
    self.assertGreater(diag.aatest.bounds[0], 0)
    self.assertAlmostEqual(diag.aatest.prob, 0.038, places=3)

  def testTestOkPositiveEffectLowProb(self):
    """Test succeeds if effect is > 0 and prob <= 0.2."""
    n_test = self.n_test
    y = self.y2
    y[-n_test:] = y[-n_test:] + 10.08  # Add an effect.
    diag = TBRMMDiagnostics(y, self.par)
    diag.x = self.x2
    self.assertGreater(diag.aatest.bounds[0], 0)
    self.assertAlmostEqual(diag.aatest.prob, 0.199, places=3)
    self.assertTrue(diag.aatest.test_ok)

  def testTestFailsPositiveEffectHighProb(self):
    """Test succeeds if effect is > 0 and prob >= 0.2."""
    n_test = self.n_test
    y = self.y2
    y[-n_test:] = y[-n_test:] + 10.09  # Add an effect.
    diag = TBRMMDiagnostics(y, self.par)
    diag.x = self.x2
    self.assertGreater(diag.aatest.bounds[0], 0)
    self.assertAlmostEqual(diag.aatest.prob, 0.202, places=3)
    self.assertFalse(diag.aatest.test_ok)

  def testTestOkNegativeEffectLowProb(self):
    """Test succeeds if effect is < 0 and prob <= 0.2."""
    n_test = self.n_test
    y = self.y2
    y[-n_test:] = y[-n_test:] - 16.275  # Add a negative effect.
    diag = TBRMMDiagnostics(y, self.par)
    diag.x = self.x2
    self.assertLess(diag.aatest.bounds[1], 0)
    self.assertAlmostEqual(diag.aatest.prob, 0.199, places=3)
    self.assertTrue(diag.aatest.test_ok)

  def testTestNotOkNegativeEffectHighProb(self):
    """Test succeeds if effect is < 0 and prob >= 0.2."""
    n_test = self.n_test
    y = self.y2
    y[-n_test:] = y[-n_test:] - 16.29  # Add a negative effect.
    diag = TBRMMDiagnostics(y, self.par)
    diag.x = self.x2
    self.assertLess(diag.aatest.bounds[1], 0)
    self.assertAlmostEqual(diag.aatest.prob, 0.202, places=3)
    self.assertFalse(diag.aatest.test_ok)


class CorrTestTest(TBRMMDiagnosticsTest):
  """Property corr_test."""

  def testNoX(self):
    """The value is none if 'x' is not set."""
    diag = TBRMMDiagnostics(self.y, self.par)
    self.assertIsNone(diag.corr_test)

  def testDefaultPass(self):
    """The default correlation threshold is 0.80, higher correlations pass."""
    diag = TBRMMDiagnostics(self.y, self.par)
    diag.x = (-0.8, -1.1, -1.2, -0.9, -1.0, -0.3, -0.4, -1.2, 0.5, 0.35,
              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    self.assertAlmostEqual(diag.corr, 0.8002, places=4)
    self.assertTrue(diag.corr_test)

  def testDefaultFail(self):
    """The default correlation threshold is 0.80, lower correlations fail."""
    diag = TBRMMDiagnostics(self.y, self.par)
    diag.x = (-0.8, -1.1, -1.2, -0.9, -1.0, -0.3, -0.4, -1.2, 0.5, 0.36,
              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    self.assertAlmostEqual(diag.corr, 0.7999, places=4)
    self.assertFalse(diag.corr_test)

  def testChangeThreshold(self):
    """Changing correlation threshold changes the behavior of the test."""
    # Change the correlation threshold to 0.9. Lower correlations fail the test.
    par = TBRMMDesignParameters(n_test=14, iroas=3.0, min_corr=0.9)
    diag = TBRMMDiagnostics(self.y, par)
    diag.x = (-0.8, -1.1, -1.2, -0.9, -1.0, -0.3, -0.4, -1.2, 0.5, 0.0,
              0.1, -0.5, 0.4, 0.6, 1.1, -0.1, 1.3, 1.4, 1.8, 1.4, 1.59)
    self.assertAlmostEqual(diag.corr, 0.8997, places=4)
    self.assertFalse(diag.corr_test)


class TestsOkTest(TBRMMDiagnosticsTest):
  """Property 'tests_ok'."""

  def setUp(self):
    super().setUp()
    # Correlated time series without effect. All tests pass.
    self.y0 = np.array([255.2, 165.8, 186.0, 160.9, 218.1, 165.7, 212.9, 207.7,
                        224.4, 205.0, 247.2, 145.1, 191.1, 173.6])
    self.x0 = np.array([132.5, 87.8, 89.4, 78.5, 117.3, 54.0, 134.9, 84.8,
                        106.4, 95.0, 129.2, 58.8, 93.6, 92.3])

    self.n_test = 7
    self.par = TBRMMDesignParameters(n_test=self.n_test, iroas=1.0)

  def testNoX(self):
    """The value is none if 'x' is not set."""
    diag = TBRMMDiagnostics(self.y, self.par)
    self.assertIsNone(diag.tests_ok)

  def testAllOkTestsOk(self):
    """If all tests pass, the result is True."""
    diag = TBRMMDiagnostics(self.y0, self.par)
    diag.x = self.x0
    self.assertTrue(diag.corr_test)
    self.assertTrue(diag.bbtest.test_ok)
    self.assertTrue(diag.dwtest.test_ok)
    self.assertTrue(diag.aatest.test_ok)
    self.assertTrue(diag.tests_ok)

  def testCorrelationTestFailsTestsFail(self):
    """If the correlation test fails, the result is False."""
    # The pair (self.x, self.y) implies a failure for the A/A test only.
    par = TBRMMDesignParameters(n_test=self.n_test, iroas=1.0, min_corr=0.90)
    diag = TBRMMDiagnostics(self.y0, par)
    diag.x = self.x0
    self.assertFalse(diag.corr_test)  # Fails as corr. = 0.86 < 0.90.
    self.assertTrue(diag.bbtest.test_ok)
    self.assertTrue(diag.dwtest.test_ok)
    self.assertTrue(diag.aatest.test_ok)
    self.assertFalse(diag.tests_ok)

  def testDWTestFailsTestsFail(self):
    """If the D-W test fails, the result is False."""
    # The pair (self.x2, self.y2) implies a failure for the D-W test only.
    x = np.array([132.5, 87.8, 89.4, 78.5, 117.3, 54.0, 134.9, 84.8, 106.4,
                  95.0, 129.2, 58.8, 93.6, 92.3, 122.7, 78.0, 96.6, 82.4,
                  100.8, 111.7, 78.0])
    y = np.array([487.9, 393.6, 388.8, 375.0, 420.9, 305.5, 451.1, 364.2, 423.4,
                  376.2, 450.5, 303.9, 370.3, 371.2, 445.1, 333.7, 397.9,
                  398.0, 416.4, 419.6, 338.2])
    diag = TBRMMDiagnostics(y, self.par)
    diag.x = x
    self.assertTrue(diag.corr_test)
    self.assertTrue(diag.bbtest.test_ok)
    self.assertFalse(diag.dwtest.test_ok)
    self.assertTrue(diag.aatest.test_ok)
    self.assertFalse(diag.tests_ok)

  def testBBTestFailsImpliesFail(self):
    """If only the Brownian Bridge test fails, the result is False."""
    x = np.array([181., 69., 74., 46., 143., -15., 187., 62., 116., 88., 173.,
                  -3., 84., 81., 157., 45.])
    y = x.copy()
    y[0] = y[0] - 100.0
    y[1] = y[1] + 20.0
    par = TBRMMDesignParameters(n_test=self.n_test, iroas=1.0)
    diag = TBRMMDiagnostics(y, par)
    diag.x = x
    self.assertTrue(diag.corr_test)
    self.assertFalse(diag.bbtest.test_ok)
    self.assertTrue(diag.dwtest.test_ok)
    self.assertTrue(diag.aatest.test_ok)
    self.assertFalse(diag.tests_ok)

  def testAATestFailsTestsFail(self):
    """If the A/A Test fails, the result is False."""
    # The pair (self.x, self.y) implies a failure for the A/A test only.
    diag = TBRMMDiagnostics(self.y, self.par)
    diag.x = self.x
    self.assertTrue(diag.corr_test)
    self.assertTrue(diag.bbtest.test_ok)
    self.assertTrue(diag.dwtest.test_ok)
    self.assertFalse(diag.aatest.test_ok)
    self.assertFalse(diag.tests_ok)

  def testAATestNoneImpliesNone(self):
    """If the A/A Test returns None, the result is None."""
    # A/A Test returns None if there are not enough time points.
    x = np.array([181., 69., 74., 46., 143., -15., 187., 62.])
    y = x.copy()
    y[0] = y[0] + 1
    y[7] = y[7] + 1
    diag = TBRMMDiagnostics(y, self.par)
    diag.x = x
    self.assertTrue(diag.corr_test)
    self.assertTrue(diag.bbtest.test_ok)
    self.assertTrue(diag.dwtest.test_ok)
    self.assertIsNone(diag.aatest.test_ok)
    self.assertIsNone(diag.tests_ok)


if __name__ == '__main__':
  unittest.main()
