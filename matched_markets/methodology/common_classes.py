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

"""A few common classes to be used for the library."""

import dataclasses
import pandas as pd


@dataclasses.dataclass
class TimeWindow:
  """Defines a time window using first day and last day."""
  first_day: pd.Timestamp
  last_day: pd.Timestamp

  def __post_init__(self):
    if not isinstance(self.first_day, pd.Timestamp):
      self.first_day = pd.Timestamp(self.first_day)
    if not isinstance(self.last_day, pd.Timestamp):
      self.last_day = pd.Timestamp(self.last_day)

    if self.first_day > self.last_day:
      raise ValueError('TimeWindow(): first_day > last_day: {!r}, {!r}'.format(
          self.first_day, self.last_day))
