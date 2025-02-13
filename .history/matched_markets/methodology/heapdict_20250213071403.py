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
"""TBR Matched Markets: utilities."""

import heapq
import collections
from typing import Any, Dict, List, TypeVar

DictKey = TypeVar('DictKey', str, int, float)


class HeapDict:
    """A dictionary of priority queues with a fixed maximum size per queue.

    Each dictionary key maps to a separate priority queue with a fixed maximum
    size. When adding a new item, the smallest item is discarded if the maximum
    size is exceeded. This ensures each queue retains only the largest items.

    Items must be sortable. Custom classes must implement __lt__ for sorting.

    Example:
        h = HeapDict(1)  # Keep only the largest item per key.
        h.push(10, 0.5)
        h.push(10, 1.0)
        h.push(20, 1.0)
        h.push(20, 2.0)
        h.get_result()  # Returns {10: [1.0], 20: [2.0]}.
    """

    def __init__(self, size: int):
        """Initializes the HeapDict with a fixed queue size.

        Args:
            size: Maximum number of elements per priority queue.
        """
        self._size = size
        self._result: Dict[DictKey, List[Any]] = collections.defaultdict(list)

    def push(self, key: DictKey, item: Any) -> None:
        """Pushes an item into the priority queue of the given key.

        Args:
            key: The dictionary key (string, integer, or float).
            item: The item to be inserted into the priority queue.
        """
        queue = self._result[key]
        if len(queue) < self._size:
            heapq.heappush(queue, item)
        else:
            heapq.heappushpop(queue, item)

    def get_result(self) -> Dict[DictKey, List[Any]]:
        """Returns a dictionary with sorted queues in descending order.

        Returns:
            A dictionary where each key maps to a sorted list of values
            (largest values first).
        """
        return {key: heapq.nlargest(len(q), q) for key, q in self._result.items()}
