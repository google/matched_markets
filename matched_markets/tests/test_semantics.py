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
"""Test that default settings are available for the module.
"""

from absl import flags
from matched_markets.methodology import semantics

import unittest




class SemanticsTest(unittest.TestCase):

  def testDefaultsExist(self):
    """Check defaults are accessible."""

    # Summon the defaults.
    col_names = semantics.DataFrameNameMapping()
    groups = semantics.GroupSemantics()
    periods = semantics.PeriodSemantics()

    # Check default col names available.
    self.assertIsInstance(col_names.geo, str)
    self.assertIsInstance(col_names.group, str)
    self.assertIsInstance(col_names.period, str)
    self.assertIsInstance(col_names.response, str)
    self.assertIsInstance(col_names.cost, str)
    self.assertIsInstance(col_names.incr_response, str)
    self.assertIsInstance(col_names.incr_cost, str)

    #  Check default data semantics available.
    self.assertIsInstance(groups.control, int)
    self.assertIsInstance(groups.treatment, int)
    self.assertIsInstance(groups.unassigned, int)

    self.assertIsInstance(periods.pre, int)
    self.assertIsInstance(periods.test, int)
    self.assertIsInstance(periods.cooldown, int)
    self.assertIsInstance(periods.unassigned, int)

  def testRepeatedValuesRaiseError(self):
    """Trying to define semantics with repeated values should raise an error."""

    bad_argument = {'key_one': 'value_one', 'key_two': 'value_one'}
    with self.assertRaises(ValueError):
      semantics.BaseSemantics(bad_argument)


if __name__ == '__main__':
  unittest.main()
