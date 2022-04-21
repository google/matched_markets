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
"""TBR Matched Markets preanalysis.
"""
from typing import List, Optional, Set, Text, Union
import dataclasses

GeoRef = Union[Text, int]


@dataclasses.dataclass
class GeoAssignments:
  """Representation of all possible geo assignments.

  All attributes are sets of references to geos, can be geo IDs (strings) or
  integers.

  Attributes:
    all: All geos.
    c: All geos that can be assigned to Control.
    t: All geos that can be assigned to Treatment.
    x: All geos that can be excluded.
    c_fixed: geos that must be assigned only to Control.
    t_fixed: geos that must be assigned only to Treatment.
    x_fixed: geos that must be excluded.
    cx: geos that can be in control or excluded (but not in treatment).
    tx: geos that can be in treatment or excluded (but not in control).
    ctx: geos that can be in either group or excluded.
    ct: geos that must be assigned to either control or treatment, but not
      excluded.
  """
  all: Set[GeoRef]
  c: Set[GeoRef]
  t: Set[GeoRef]
  x: Set[GeoRef]
  t_fixed: Set[GeoRef]
  c_fixed: Set[GeoRef]
  x_fixed: Set[GeoRef]
  ct: Set[GeoRef]
  cx: Set[GeoRef]
  ctx: Set[GeoRef]
  tx: Set[GeoRef]

  def __init__(self, c: Set[GeoRef], t: Set[GeoRef], x: Set[GeoRef]):
    self.c = c
    self.t = t
    self.x = x
    a = c | t | x
    self.all = a
    not_c = a - c
    not_t = a - t
    not_x = a - x
    self.c_fixed = c & not_t & not_x
    self.t_fixed = not_c & t & not_x
    self.x_fixed = not_c & not_t & x
    self.ct = c & t & not_x
    self.cx = c & not_t & x
    self.ctx = c & t & x
    self.tx = not_c & t & x


class GeoEligibility:
  """Validate a Geo Eligibility Matrix.

  A Geo Eligibility Matrix maps each geo to the possible mappings into treatment
  groups, or possible exclusion from the design. Used in the TBR Matched Markets
  preanalysis.
  """

  def __init__(self, df):
    """Initialize and validate a GeoEligibility object.

    Args:

      df: A DataFrame with columns 'geo', 'control' 'treatment', 'exclude'. Each
        row specifies to which groups each geo can be assigned to, by using
        codes 1 = possible and 0 = not possible. 'geo' can also be the index.

        control treatment exclude
           0        0        1     - geo must be excluded.
           0        1        0     - geo must be assigned to treatment.
           1        0        0     - geo must be assigned to control.
           1        1        1     - geo can be excluded, or included in either
                                     control or treatment.
           0        1        1     - geo can be assigned only to treatment, or
                                     excluded.
           1        0        1     - geo can be assigned only to control, or
                                     excluded.
           1        1        0     - geo must be included in either control or
                                     treatment but never excluded.
           0        0        0     - not allowed.

    Attributes:
      data: A copy of the dataframe, indexed by 'geo'.

    Raises:
      ValueError: if (a) the DataFrame does not have columns 'geo', 'control',
        'treatment' and 'exclude'; (b) any geo ids are duplicated; (c) if the
        values in columns 'control', 'treatment', and 'exclude' are something
        else than 0 and 1; (d) if any row in the columns 'control', 'treatment'
        and 'exclude' has all zeros in it.
    """

    df = df.copy().reset_index()

    if 'geo' not in df.columns:
      raise ValueError('There is no column or index \'geo\'')

    dups = df.columns.duplicated()
    if any(dups):
      raise ValueError('Duplicate column(s): ' + ', '.join(df.columns[dups]))

    # Ensure that the geo column is a string.
    df.geo = df.geo.astype('str')

    value_columns = ['control', 'treatment', 'exclude']
    if not set(value_columns).issubset(set(df.columns)):
      missing_columns = [x for x in value_columns if x not in df.columns]
      raise ValueError('Missing column(s): ' + ', '.join(missing_columns))

    all_column_names = ['geo'] + value_columns

    # Ensure the correct column order.
    df = df.loc[:, all_column_names]

    dup_geo_ids = set(df['geo'][df['geo'].duplicated()])
    if dup_geo_ids:
      raise ValueError('\'geo\' has duplicate values: ' +
                       ', '.join(str(id) for id in dup_geo_ids))

    if not all([set(df[col]) <= {0, 1} for col in value_columns]):
      raise ValueError('GeoEligibility objects must have only values '
                       '0, 1 in columns ' + ', '.join(value_columns))

    zero_row = df[value_columns].sum(axis=1) == 0
    if any(zero_row):
      geos = df['geo'][zero_row]
      raise ValueError('Three zeros found for geo(s) ' + ', '.join(geos))

    df.set_index('geo', inplace=True)
    self.data = df

  def __str__(self):
    return 'Geo eligibility matrix with %d geos' % self.data.shape[0]

  def get_eligible_assignments(self, geos: Optional[List[GeoRef]] = None,
                               indices: bool = False) -> GeoAssignments:
    """Get an object representing all possible geo assignment groups.

    Args:
      geos: A list of geo IDs to include. If None, all geos are included. The
        order is important if 'indices' are used.
      indices: Instead of generating sets of geo IDs, generate sets of the
        positional index numbers of the geo IDs in the list 'geos'. Raises an
        error if 'geos' is not specified.

    Returns:
      A GeoAssignments object.

    Raises:
      ValueError: if geos is not specified but indices is True.
    """

    df = self.data  # DataFrame indexed by the geo ID.

    if geos:
      df = df.loc[geos]
      if indices:
        df = df.reset_index()
    elif indices:
      raise ValueError('\'geos\' is not specified but indices=True')

    # Generate sets of geos (IDs or indices) indicating membership of the group.
    c = set(df.index[df['control'] == 1])
    t = set(df.index[df['treatment'] == 1])
    x = set(df.index[df['exclude'] == 1])

    return GeoAssignments(c, t, x)
