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
import enum

import numpy as np
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


class GeoAssignment(enum.IntEnum):
  """Defines the values for Treatment/Control assignment."""
  CONTROL = 2
  TREATMENT = 1
  EXCLUDED = -1


class ExperimentPeriod(enum.IntEnum):
  """Defines the values for Pre-Experiment, Experiment and Post-Experiment."""
  PRE_EXPERIMENT = 0
  EXPERIMENT = 1
  POST_EXPERIMENT = 2


class EstimatedTimeSeriesWithConfidenceInterval(pd.DataFrame):
  """Defines an estimate time series with pointwise confidence intervals."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    if not {'date', 'estimate', 'lower', 'upper'}.issubset(self.columns):
      raise KeyError('The time series must contain the columns "date", ' +
                     '"estimate", "lower", "upper".')
    if np.any(self['lower'] > self['estimate']):
      raise ValueError('lower bound is not smaller than point estimate.')
    if np.any(self['upper'] < self['estimate']):
      raise ValueError('upper bound is not larger than point estimate.')


@dataclasses.dataclass
class TimeSeries:
  counterfactual: EstimatedTimeSeriesWithConfidenceInterval
  pointwise_difference: EstimatedTimeSeriesWithConfidenceInterval
  cumulative_effect: EstimatedTimeSeriesWithConfidenceInterval
