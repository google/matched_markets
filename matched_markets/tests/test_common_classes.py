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

"""Tests for common_classes."""

from matched_markets.methodology import common_classes
import pandas as pd

import unittest

TimeWindow = common_classes.TimeWindow


class CommonClassesTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self._t1 = '2019-01-01'
    self._t2 = '2020-01-01'

  def testTimeWindowCorrectInitialization(self):
    result = TimeWindow(self._t1, self._t2)
    self.assertEqual(pd.Timestamp(self._t1), result.first_day)
    self.assertEqual(pd.Timestamp(self._t2), result.last_day)

  def testTimeWindowInitializationError(self):
    # first day is chronologically after the last day
    with self.assertRaises(ValueError):
      TimeWindow('2020-02-01', self._t2)

if __name__ == '__main__':
  unittest.main()
