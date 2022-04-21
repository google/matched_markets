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
"""TBR Matched Markets: utilities.
"""

import collections
import heapq

from typing import Any, Dict, List, TypeVar

DictKey = TypeVar('DictKey', str, int, float)


class HeapDict:
  """A dictionary of priority queues of a given limited size.

  Each dictionary key points to a separate queue that has a fixed maximum
  size. Upon pushing an item in a queue, the smallest item will be discarded if
  the maximum size is exceeded. Hence each queue stores the largest items that
  have been pushed in.

  Each item must be sortable; an item of arbitrary class can be used if it
  features a custom __lt__ method.

  Example:
    h = HeapDict(1)  # Keep only the largest item.
    h.push(10, 0.5)
    h.push(10, 1.0)
    h.push(20, 1.0)
    h.push(20, 2.0)
    h.get_result()  # Returns {10: [1.0], 20: [2.0]}.
  """

  def __init__(self, size: int):
    """Initialize a HeapDict.

    Args:
      size: Maximum size of each heap (priority queue).
    """
    self._size = size
    self._result = collections.defaultdict(list)

  def push(self, key: DictKey, item: Any):
    """Push an item into the queue associated with the key.

    Args:
      key: A dictionary key, string, integer, or float.
      item: Any object. The queue corresponding to the key will be sorted based
        on this object.
    """
    queue = self._result[key]
    if len(queue) < self._size:
      heapq.heappush(queue, item)
    else:
      # Push the new item, and remove the smallest item.
      heapq.heappushpop(queue, item)
    self._result[key] = queue

  def get_result(self) -> Dict[DictKey, List[Any]]:
    """Return a copy of the dictionary, each queue sorted in descending order.

    Returns:
      A dictionary with the sorted lists as values, largest values first.
    """
    result = {}
    for key, q in self._result.items():
      result[key] = heapq.nlargest(len(q), q)
    return result
