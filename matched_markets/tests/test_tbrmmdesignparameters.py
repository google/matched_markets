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
"""Test TBRMMDesignParameters.
"""
from matched_markets.methodology import tbrmmdesignparameters
import unittest


TBRMMDesignParameters = tbrmmdesignparameters.TBRMMDesignParameters


class TBRMMDesignParametersTest(unittest.TestCase):
  """Base class for all tests involving parameters.

  The tests for each parameter are located in its proper subclass, which
  also provides the name of the parameter and the default error message.

  This class provides the helper functions that each test inheriting from this
  class is supposed to call.
  """

  name: str
  default_error_message: str

  def setUp(self):
    """Set up defaults."""

    super().setUp()

    self.default_args = {'n_test': 14, 'iroas': 1.0}  # Mandatory parameters.
    self.par = TBRMMDesignParameters(n_test=14, iroas=1.0)

  def _testBadValue(self, value, message=None):  # pylint: disable=invalid-name
    """Raise an error if the value is out of the acceptable range."""

    if not message:
      # The name of the parameter and the default error message are class
      # attributes in the subclasses.
      message = self.default_error_message

    kwargs = self.default_args
    kwargs[self.name] = value
    with self.assertRaisesRegex(ValueError, message.format(self.name)):
      TBRMMDesignParameters(**kwargs)

  def _testValueOk(self, value):  # pylint: disable=invalid-name
    """No exception is raised if the Value is accepted."""

    kwargs = self.default_args
    kwargs[self.name] = value
    self.assertEqual(getattr(TBRMMDesignParameters(**kwargs), self.name), value)

  def _testValueNoneOk(self):  # pylint: disable=invalid-name
    """Value can be set to None."""

    kwargs = self.default_args
    kwargs[self.name] = None
    self.assertIsNone(getattr(TBRMMDesignParameters(**kwargs), self.name))

  def _testValueNonNumeric(self):  # pylint: disable=invalid-name
    """Non-numeric values raise an error."""

    non_numeric_items = ('1.0', (1,), (0.1, None))
    for value in non_numeric_items:
      self._testBadValue(value, '{} must be numeric')

  def _testRangeStructure(self):  # pylint: disable=invalid-name
    """Values that are not tuples of length 2 raise an error.

    Only applied to parameters that have 2-tuples (ranges) as values.
    """

    for value in (0.1, (0.1,), (0.1, 0.2, 0.3), '0.1'):
      self._testBadValue(value, '{} requires a range of two numbers')

  def _testRangeNonNumeric(self):  # pylint: disable=invalid-name
    """Non-numeric value (range) raises an error.

    Only applied to parameters that have 2-tuples (ranges) as values.
    """

    non_numeric_items = ((0.1, None), (None, 0.2), ('0.1', 0.2), (0.1, (1,)))

    for value in non_numeric_items:
      self._testBadValue(value, '{} requires a range of two numbers')


class DefaultsTest(TBRMMDesignParametersTest):

  def testNoArguments(self):
    """Verify that calling with no arguments is an error."""

    with self.assertRaisesRegex(
        TypeError, 'missing 2 required positional arguments: '
        "'n_test' and 'iroas'"):
      TBRMMDesignParameters()

  def testDefaults(self):
    """Verify that the default values are correct."""

    par = TBRMMDesignParameters(n_test=14, iroas=1.0)

    self.assertIsNone(par.volume_ratio_tolerance)
    self.assertIsNone(par.geo_ratio_tolerance)
    self.assertIsNone(par.treatment_share_range)
    self.assertIsNone(par.budget_range)
    self.assertIsNone(par.treatment_geos_range)
    self.assertIsNone(par.control_geos_range)
    self.assertIsNone(par.n_geos_max)
    self.assertEqual(par.n_designs, 1)
    self.assertEqual(par.sig_level, 0.9)
    self.assertEqual(par.power_level, 0.8)
    self.assertEqual(par.min_corr, 0.8)
    self.assertEqual(par.n_pretest_max, 90)
    self.assertEqual(par.rho_max, 0.995)
    self.assertEqual(par.flevel, 0.9)

  def testSameInstance(self):
    """Checks that an instance is equal to itself."""
    par2 = self.par
    self.assertEqual(self.par, par2)

  def testEquality(self):
    """Checks that two instances with same values are equal."""
    par2 = TBRMMDesignParameters(n_test=14, iroas=1.0)
    self.assertEqual(self.par, par2)

  def testNotEqual(self):
    """Checks that two instances with different values are not equal."""
    par2 = TBRMMDesignParameters(n_test=28, iroas=1.0)
    self.assertNotEqual(self.par, par2)

  def testNotImplementedEquality(self):
    """Raise error for equality of classes that cannot be compared."""
    with self.assertRaisesRegex(NotImplementedError,
                                r'Cannot compare instance of '
                                'TBRMMDesignParameters'
                                r' with instance of <class \'str\'>'):
      self.par.__eq__('string')


class NTestTest(TBRMMDesignParametersTest):

  name = 'n_test'
  default_error_message = '{} must be >= 1'

  def testValueNotInteger(self):
    self._testBadValue(7.1, '{} must be an integer')

  def testValueNonNumeric(self):
    self._testValueNonNumeric()

  def testValueTooLow(self):
    self._testBadValue(0)

  def testValueOk(self):
    self._testValueOk(7)

  def testValueNone(self):
    self._testBadValue(None)


class IroasTest(TBRMMDesignParametersTest):

  name = 'iroas'
  default_error_message = r'{} must be >= 0\.0'

  def testValueNonNumeric(self):
    self._testValueNonNumeric()

  def testValueTooLow(self):
    self._testBadValue(-0.99)

  def testValueOk(self):
    self._testValueOk(3.0)
    self._testValueOk(0.0)

  def testValueNone(self):
    self._testBadValue(None)


class VolumeRatioToleranceTest(TBRMMDesignParametersTest):

  name = 'volume_ratio_tolerance'
  default_error_message = '{} must be > 0.0'

  def testValueNonNumeric(self):
    self._testValueNonNumeric()

  def testValueNoneOk(self):
    self._testValueNoneOk()

  def testValueOk(self):
    self._testValueOk(0.99)

  def testValueTooLow(self):
    self._testBadValue(-0.01)
    self._testBadValue(0.0)


class GeoRatioToleranceTest(TBRMMDesignParametersTest):

  name = 'geo_ratio_tolerance'
  default_error_message = r'{} must be > 0\.0'

  def testValueNonNumeric(self):
    self._testValueNonNumeric()

  def testValueNoneOk(self):
    self._testValueNoneOk()

  def testValueOk(self):
    self._testValueOk(0.01)
    self._testValueOk(0.99)

  def testValueTooLow(self):
    self._testBadValue(-0.01)
    self._testBadValue(0.0)


class TreatmentShareRangeTest(TBRMMDesignParametersTest):

  name = 'treatment_share_range'
  default_error_message = r'{} must be > 0\.0 and < 1\.0'

  def testRangeStructure(self):
    self._testRangeStructure()

  def testRangeNonNumeric(self):
    self._testRangeNonNumeric()

  def testRangeInverted(self):
    self._testBadValue((0.3, 0.2), 'Lower bound of {} must be < upper bound')

  def testValueNoneOk(self):
    self._testValueNoneOk()

  def testValueOk(self):
    self._testValueOk((0.1, 0.3))

  def testValueTooLow(self):
    self._testBadValue((-1.0, -0.1))
    self._testBadValue((-1.0, 0.0))
    self._testBadValue((0.0, 1.0))

  def testValueTooHigh(self):
    self._testBadValue((1.0, 1.1))
    self._testBadValue((1.1, 1.2))


class BudgetRangeTest(TBRMMDesignParametersTest):

  name = 'budget_range'
  default_error_message = '{} must be >= 0'

  def testRangeStructure(self):
    self._testRangeStructure()

  def testRangeNonNumeric(self):
    self._testRangeNonNumeric()

  def testRangeInverted(self):
    self._testBadValue((2.0, 1.0), 'Lower bound of {} must be < upper bound')

  def testValueNoneOk(self):
    self._testValueNoneOk()

  def testValueOk(self):
    self._testValueOk((1.0, 1e6))

  def testValueTooLow(self):
    self._testBadValue((-1.0, -0.01))
    self._testBadValue((-0.01, 1.0))

  def testValueTooHigh(self):
    inf = float('inf')
    self._testBadValue((0.1, inf))
    self._testBadValue((inf, inf))


class TreatmentGeosRangeTest(TBRMMDesignParametersTest):

  name = 'treatment_geos_range'
  default_error_message = '{} must be >= 1'

  def testRangeStructure(self):
    self._testRangeStructure()

  def testValueNotInteger(self):
    self._testBadValue((1.1, 2), '{} must be integers')

  def testRangeNonNumeric(self):
    self._testRangeNonNumeric()

  def testRangeInverted(self):
    self._testBadValue((2, 1), 'Lower bound of {} must be <= upper bound')

  def testValueNoneOk(self):
    self._testValueNoneOk()

  def testValueOk(self):
    self._testValueOk((1, 10))

  def testValueTooLow(self):
    self._testBadValue((-1, 0))
    self._testBadValue((-1, 1))
    self._testBadValue((0, 1))

  def testValueTooHigh(self):
    inf = float('inf')
    self._testBadValue((1, inf))
    self._testBadValue((inf, inf))


class ControlGeosRangeTest(TBRMMDesignParametersTest):

  name = 'control_geos_range'
  default_error_message = '{} must be >= 1'

  def testRangeStructure(self):
    self._testRangeStructure()

  def testValueNotInteger(self):
    self._testBadValue((1.1, 2), '{} must be integers')

  def testRangeNonNumeric(self):
    self._testRangeNonNumeric()

  def testRangeInverted(self):
    self._testBadValue(
        (2, 1),
        'Lower bound of {} must be <= upper bound')

  def testValueNoneOk(self):
    self._testValueNoneOk()

  def testValueOk(self):
    self._testValueOk((1, 10))

  def testValueTooLow(self):
    self._testBadValue((-1, 0))
    self._testBadValue((-1, 1))
    self._testBadValue((0, 1))

  def testValueTooHigh(self):
    inf = float('inf')
    self._testBadValue((1, inf))
    self._testBadValue((inf, inf))


class NGeosMaxTest(TBRMMDesignParametersTest):

  name = 'n_geos_max'
  default_error_message = '{} must be >= 2'

  def testValueNonNumeric(self):
    self._testValueNonNumeric()

  def testValueNotInteger(self):
    self._testBadValue(2.1, '{} must be an integer')

  def testValueOk(self):
    self._testValueOk(2)

  def testValueNoneOk(self):
    self._testValueNoneOk()

  def testValueTooLow(self):
    self._testBadValue(-1)
    self._testBadValue(0)
    self._testBadValue(1)


class NPretestMaxTest(TBRMMDesignParametersTest):

  name = 'n_pretest_max'
  default_error_message = '{} must be >= 3'

  def testValueNonNumeric(self):
    self._testValueNonNumeric()

  def testValueNotInteger(self):
    self._testBadValue(14.1, '{} must be an integer')

  def testValueOk(self):
    self._testValueOk(14)

  def testValueNoneNotOk(self):
    self._testBadValue(None)

  def testValueTooLow(self):
    self._testBadValue(-1)
    self._testBadValue(0)
    self._testBadValue(2)


class NDesignsTest(TBRMMDesignParametersTest):

  name = 'n_designs'
  default_error_message = '{} must be >= 1'

  def testValueNonNumeric(self):
    self._testValueNonNumeric()

  def testValueNotInteger(self):
    self._testBadValue(1.1, '{} must be an integer')

  def testValueOk(self):
    self._testValueOk(1)

  def testValueNoneNotOk(self):
    self._testBadValue(None)

  def testValueTooLow(self):
    self._testBadValue(-1)
    self._testBadValue(0)


class SigLevelTest(TBRMMDesignParametersTest):

  name = 'sig_level'
  default_error_message = r'{} must be > 0\.0 and < 1\.0'

  def testValueNonNumeric(self):
    self._testValueNonNumeric()

  def testValueOk(self):
    self._testValueOk(0.01)
    self._testValueOk(0.99)

  def testValueNoneNotOk(self):
    self._testBadValue(None)

  def testValueTooLow(self):
    self._testBadValue(-1)
    self._testBadValue(0.0)


class PowerLevelTest(TBRMMDesignParametersTest):

  name = 'power_level'
  default_error_message = r'{} must be > 0\.0 and < 1\.0'

  def testValueNonNumeric(self):
    self._testValueNonNumeric()

  def testValueOk(self):
    self._testValueOk(0.01)
    self._testValueOk(0.99)

  def testValueNoneNotOk(self):
    self._testBadValue(None)

  def testValueTooLow(self):
    self._testBadValue(0.0)
    self._testBadValue(-0.1)


class MinCorrTest(TBRMMDesignParametersTest):

  name = 'min_corr'
  default_error_message = r'{} must be >= 0\.8 and < 1\.0'

  def testValueNonNumeric(self):
    self._testValueNonNumeric()

  def testValueOk(self):
    self._testValueOk(0.8)
    self._testValueOk(0.99)

  def testValueNoneNotOk(self):
    self._testBadValue(None)

  def testValueTooLow(self):
    self._testBadValue(0)
    self._testBadValue(0.79)


class RhoMaxTest(TBRMMDesignParametersTest):

  name = 'rho_max'
  default_error_message = r'{} must be >= 0\.9 and < 1\.0'

  def testValueNonNumeric(self):
    self._testValueNonNumeric()

  def testValueOk(self):
    self._testValueOk(0.9)
    self._testValueOk(0.99)

  def testValueNoneNotOk(self):
    self._testBadValue(None)

  def testValueTooLow(self):
    self._testBadValue(0)
    self._testBadValue(0.89)


class FlevelTest(TBRMMDesignParametersTest):

  name = 'flevel'
  default_error_message = r'{} must be >= 0\.9 and < 1\.0'

  def testValueNonNumeric(self):
    self._testValueNonNumeric()

  def testValueOk(self):
    self._testValueOk(0.9)
    self._testValueOk(0.99)

  def testValueNoneNotOk(self):
    self._testBadValue(None)

  def testValueTooLow(self):
    self._testBadValue(0)
    self._testBadValue(0.89)


if __name__ == '__main__':
  unittest.main()
