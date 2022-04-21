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
"""Tests for the HeapDict class."""
from typing import Any

from matched_markets.methodology.heapdict import HeapDict
import unittest


class HeapDictTest(unittest.TestCase):
  """Checks the functionality of the HeapDict."""

  class PriorityItem:
    """A sortable arbitrary object with a priority number."""

    def __init__(self, priority: float, item: Any):
      self.thing = (priority, item)

    def __lt__(self, other):
      return self.thing[0] < other.thing[0]

  def testZeroSize(self):
    """Verifies that the priority queues can have zero size."""
    hd = HeapDict(size=0)
    hd.push('a', 1)
    hd.push('b', 1)
    self.assertEqual(hd.get_result(), {'a': [], 'b': []})

  def testOneSize(self):
    """Verifies that queues can have size 1, storing the largest item."""
    hd = HeapDict(size=1)
    hd.push('a', 2)
    hd.push('a', 1)
    hd.push('b', 3)
    hd.push('b', 4)
    self.assertEqual(hd.get_result(), {'a': [2], 'b': [4]})

  def testSorting(self):
    """Verifies that queues are sorted in descending order."""
    hd = HeapDict(size=2)
    hd.push('a', 1)
    hd.push('a', 2)
    hd.push('b', 3)
    hd.push('b', 2)
    self.assertEqual(hd.get_result(), {'a': [2, 1], 'b': [3, 2]})

  def testMaxSize(self):
    """Verifies that queues have a maximum size and are sorted."""
    hd = HeapDict(size=2)
    hd.push('a', 1)
    hd.push('a', 2)
    hd.push('a', 3)
    hd.push('b', 3)
    hd.push('b', 2)
    hd.push('b', 1)
    # The order is always descending.
    self.assertEqual(hd.get_result(), {'a': [3, 2], 'b': [3, 2]})

  def testIntegerKeys(self):
    """Verifies that keys can also be integers."""
    hd = HeapDict(size=1)
    hd.push(1, 2)
    self.assertEqual(hd.get_result(), {1: [2]})

  def testTuples(self):
    """Verifies that tuples can be used as items, and are sorted."""
    hd = HeapDict(size=3)
    hd.push(1, (1, 10))
    hd.push(1, (0, 10))
    hd.push(1, (1, 100))
    hd.push(1, (-1, 1000))
    self.assertEqual(hd.get_result(), {1: [(1, 100), (1, 10), (0, 10)]})

  def testRepeatedGetResult(self):
    """Verifies that the result() method can be repeated."""
    hd = HeapDict(size=2)
    hd.push(1, (1, 10))
    hd.push(1, (1, 20))
    self.assertEqual(hd.get_result(), {1: [(1, 20), (1, 10)]})
    self.assertEqual(hd.get_result(), {1: [(1, 20), (1, 10)]})

  def testArbitraryItems(self):
    """Verifies that arbitrary items that have the __lt__ method can be used."""
    hd = HeapDict(size=2)
    item1 = self.PriorityItem(1.0, [None, 'Arbitrary item'])
    item2 = self.PriorityItem(2.0, {'Another item'})
    item3 = self.PriorityItem(3.0, (1, 'Third item'))
    item4 = self.PriorityItem(4.0, 0)
    hd.push(1, item1)
    hd.push(1, item3)
    hd.push(1, item2)
    hd.push(1, item4)
    self.assertEqual(hd.get_result(), {1: [item4, item3]})


if __name__ == '__main__':
  unittest.main()
