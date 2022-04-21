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
"""TBR Matched Markets: TBRMMData class.
"""
from typing import List, Optional, Set, Text, Tuple, Union

from matched_markets.methodology import geoeligibility

import numpy as np
import pandas as pd

Array = np.ndarray
GeoID = Text
GeoIndex = int
GeoRef = Union[GeoID, GeoIndex]
OrderedGeos = Union[List[GeoRef], Tuple[GeoRef]]
GeoIndexSet = Set[GeoIndex]
Vector = List[float]
GeoEligibility = geoeligibility.GeoEligibility
GeoAssignments = geoeligibility.GeoAssignments


class TBRMMData:
  """Geo time series data for TBR Matched Markets.

  Transforms the geo time series data into a canonical format for easier
  manipulation; calculates average aggregate share of each geo; derives the list
  of geos that are in data and in the geo eligibility matrix that can be
  assigned to treatment or control ('assignable' geos).

  The assignable geos is the intersection of geos in the GeoEligibility object
  and those in the data, minus those that are specified to be excluded.

  If the GeoEligibility object is not specified at initialization, a default
  GeoEligibility object is created, with no restrictions in geo assignment.

  Ensures that the set of geos in the geo eligibility object is a subset of
  those in the data, and that the geos that must be included in the design must
  also be there in the data. However the geo eligibility object can refer to a
  subset of geos in the data.

  Attributes:
    df: Data in canonical format (geos in rows, dates in columns).
    geo_eligibility: The GeoEligibility object.
    geo_share: Aggregate share of each geo in terms of the response volume.
    geos_in_data: A complete set of geos that are found in the data set.
    assignable: Set of geo IDs that are assignable to control and/or treatment.
    geo_index: A user-defined subset of the assignable geo IDs.
    geo_assignments: Geo assignments of geos specified by 'geo_index'.
  """
  # Minimum correlation bound for identifying 'noisy geos'.
  _min_corr_bound = 0.5

  df: pd.DataFrame
  geo_eligibility: GeoEligibility
  geo_share: pd.Series = None
  geos_in_data: Set[GeoID] = None
  assignable: Set[GeoID] = None
  geo_assignments: GeoAssignments = None
  _geo_index: OrderedGeos = None  # Storage for 'geo_index'.
  _array: Optional[Array] = None  # Time series of the geos by geo_index.
  _array_geo_share: Optional[Array] = None  # Subset of geo_share[geo_index].

  def __init__(
      self,
      df: pd.DataFrame,
      response_column: str,
      geo_eligibility: Optional[GeoEligibility] = None):
    """Initialize and validate a TBRMMData object.

    1. Pivots the data frame 'df' such that geos are in the rows and dates in
       columns.
    2. Calculates mean market share for each geo.
    3. Creates a default GeoEligibility object if omitted.

    Args:
      df: (pandas.DataFrame) DataFrame with mandatory columns 'geo', 'date' and
        the column representing the response.
      response_column: String. Name of the response metric column.
      geo_eligibility: a GeoEligibility object, or if not specified, None (in
        case a default GeoEligibility object will be constructed).
    """
    df = df.copy()

    required_columns = {'date', 'geo', response_column}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
      raise ValueError('Missing column(s): ' + ', '.join(missing_columns))

    # Ensure that the geo column is a string.
    df.geo = df.geo.astype('str')

    # Transform into a canonical format with geos in rows, dates in columns,
    # geos (rows) sorted with those with the largest volume first so that
    # the largest geos are iterated first (those with the smallest row index).
    df = df.pivot_table(values=response_column, index='geo', columns='date',
                        fill_value=0)

    # Calculate the average 'market share' based on all data.
    geo_means = df.mean(axis=1).sort_values(ascending=False)
    geo_share = geo_means / sum(geo_means)
    geos_in_data = set(geo_means.index)

    # For convenience sort the geos (rows) in descending order.
    self.df = df.loc[list(geo_means.index)]
    self.geo_share = geo_share
    self.geos_in_data = geos_in_data

    if geo_eligibility is None:
      # Default object will have all geos with all possible assignment
      # possibilities.
      gelig_dict = {'geo': list(geo_means.index),
                    'control': 1,
                    'treatment': 1,
                    'exclude': 1}
      gelig_df = pd.DataFrame(gelig_dict)
      geo_eligibility = GeoEligibility(gelig_df)

    geo_assignments = geo_eligibility.get_eligible_assignments()

    # Ensure that the geo eligibility object only has geos that are
    # in the data.
    common_geos = geos_in_data & geo_assignments.all
    if common_geos != geo_assignments.all:
      # Ensure that geos that cannot be excluded are not missing.
      geos_cannot_be_excluded = geo_assignments.all - geo_assignments.x
      geos_missing = geos_cannot_be_excluded - geos_in_data
      if geos_missing:
        raise ValueError('Required geos {} were not found '
                         'in the data'.format(sorted(geos_missing)))
      df_elig = geo_eligibility.data.loc[common_geos]
      geo_eligibility = GeoEligibility(df_elig)
      geo_assignments = geo_eligibility.get_eligible_assignments()

    assignable = geo_assignments.all - geo_assignments.x_fixed
    self.assignable = assignable  # pytype: disable=annotation-type-mismatch
    self.geo_eligibility = geo_eligibility

  @property
  def geo_index(self) -> OrderedGeos:
    return self._geo_index

  @geo_index.setter
  def geo_index(self, geos: OrderedGeos):
    """Fix the set of geos that will be used.

    1. Creates a subset of the DataFrame attribute .df as a Numpy array whose
    rows represent the time series of each geo in the argument 'geos'.  This is
    a convenient and computationally fast form to produce aggregate time series
    of the data.

    2. Creates a GeoAssignment object whose values refer to the rows of the
    array (values are integers 0 .. number of geos minus 1). Using row numbers
    for the geo IDs will be faster than using indices.

    Args:
      geos: Geo IDs for the subset of the geos that will be included in the
        Matched Markets analysis.
    """
    missing_geos = set(geos) - self.assignable
    if missing_geos:
      missing_geos = sorted(list(missing_geos))
      raise ValueError('Unassignable geo(s): ' + ', '.join(missing_geos))

    self.geo_assignments = self.geo_eligibility.get_eligible_assignments(
        geos,
        indices=True)

    self._geo_index = geos
    self._array = self.df.loc[geos].to_numpy()
    self._array_geo_share = np.array(self.geo_share[geos])

  def aggregate_time_series(self, geo_indices: GeoIndexSet) -> Vector:
    """Return the aggregate the time series over a set of chosen geos.

    Args:
      geo_indices: Set of geo indices referring to the geos in self.geo_index (0
        .. number of geos in geo_index - 1).

    Returns:
       A time series representing the sum of the geos indicated by
      'geo_indices'.
    """
    return self._array[list(geo_indices)].sum(axis=0)

  def aggregate_geo_share(self, geo_indices: GeoIndexSet) -> float:
    """Share of the given geos' response as percentage of the total.

    Args:
      geo_indices: Set of geo indices referring to the geos in self.geo_index (0
        .. number of geos in geo_index - 1).

    Returns:
       Aggregate share of geos indicated by 'geo_indices'.
    """
    return self._array_geo_share[list(geo_indices)].sum()

  @property
  def leave_one_out_correlations(self) -> pd.Series:
    """Correlations between each geo and the aggregate of the rest of the geos.

    Returns:
      A pd.Series of correlations indexed by 'geo'.
    """
    aggregate_ts = self.df.sum(axis=0)

    def corr_leave_one_out(x):
      return np.corrcoef(x, aggregate_ts - x)[0, 1]

    return self.df.apply(corr_leave_one_out, axis=1)

  @property
  def noisy_geos(self) -> Set[GeoID]:
    """Returns geos that have a low or negative correlation with the rest."""
    correlations = self.leave_one_out_correlations
    return set(correlations[correlations <= self._min_corr_bound].index)
