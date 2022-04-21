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
"""TBR Matched Markets: Test class TBRMMData.
"""
import itertools

from matched_markets.methodology import geoeligibility
from matched_markets.methodology import tbrmmdata
import pandas as pd

import unittest

GeoEligibility = geoeligibility.GeoEligibility
TBRMMData = tbrmmdata.TBRMMData


class TBRMMDataTest(unittest.TestCase):
  """Test class TBRMMData."""

  def setUp(self):
    """Set up defaults."""
    super().setUp()

    # Create a data frame with 5 geos and 21 days.
    n_geos = 5
    n_days = 21
    geos = {str(geo) for geo in range(n_geos)}
    dates = pd.date_range('2020-03-01', periods=n_days)

    df_data = [{'date': date, 'geo': geo} for geo, date in
               itertools.product(geos, dates)]
    df = pd.DataFrame(df_data)
    response_column = 'sales'

    # Create sales data.
    def day_geo_sales(geo, n_days):
      # Larger geos have different means and variances.
      return [100 * geo + 10 * geo * day + day for day in range(n_days)]

    df[response_column] = 0.0
    sums = {}
    sales = {}
    for geo in geos:
      sales_time_series = day_geo_sales(int(geo), n_days)
      df.loc[df.geo == geo, response_column] = sales_time_series
      sales[geo] = sales_time_series
      sums[geo] = sum(sales_time_series)

    self.df = df
    self.geos = geos
    self.n_geos = n_geos
    self.geos_in_order = ['4', '3', '2', '1', '0']
    self.sums = sums
    self.sales = sales
    self.response = response_column

    df_geo_elig = {'geo': list(geos),
                   'treatment': 1,
                   'control': 1,
                   'exclude': 1}
    self.geo_elig = GeoEligibility(pd.DataFrame(df_geo_elig))

  def testResponseColumnIsRequired(self):
    """Missing response_column raises an error."""
    with self.assertRaisesRegex(
        TypeError,
        r'missing 1 required positional argument: \'response_column\''):
      TBRMMData(self.df)

  def testMissingGeoColumn(self):
    """Missing geo column raises an error."""
    with self.assertRaisesRegex(ValueError, r'Missing column\(s\): geo'):
      TBRMMData(self.df[['date', self.response]], self.response)

  def testMissingDateColumn(self):
    """Missing date column raises an error."""
    with self.assertRaisesRegex(ValueError, r'Missing column\(s\): date'):
      TBRMMData(self.df[['geo', self.response]], self.response)

  def testMissingResponseColumn(self):
    """Missing response column raises an error."""
    with self.assertRaisesRegex(
        ValueError,
        r'Missing column\(s\): {}'.format(self.response)):
      TBRMMData(self.df[['geo', 'date']], self.response)

  def testDfAttributeIndex(self):
    """The .df attribute must have the geos in the index."""
    self.assertSetEqual(set(TBRMMData(self.df, self.response).df.index),
                        set(self.df.geo))

  def testDfAttributeDate(self):
    """The .df attribute must have dates in columns."""
    self.assertSetEqual(set(TBRMMData(self.df, self.response).df.columns),
                        set(self.df.date))

  def testDfAttributeGeoOrder(self):
    """The order of the geos of the .df attribute must be descending in size."""
    df = TBRMMData(self.df, self.response).df
    self.assertListEqual(list(df.index), self.geos_in_order)

  def testDefaultGeoEligibilityIsThere(self):
    """A default GeoEligibility object is created."""
    self.assertIsInstance(
        TBRMMData(self.df, self.response).geo_eligibility,
        GeoEligibility)

  def testDefaultGeoEligibilityIsAllOnes(self):
    """The default GeoEligibility object has all 1s in the columns."""
    g = TBRMMData(self.df, self.response).geo_eligibility.data
    self.assertSetEqual(set(g.index), self.geos)
    self.assertSetEqual(set(g.control), {1})
    self.assertSetEqual(set(g.treatment), {1})
    self.assertSetEqual(set(g.exclude), {1})

  def testGeoEligibility(self):
    """The same GeoEligibility object is stored in the object."""
    self.assertIs(TBRMMData(self.df,
                            self.response,
                            self.geo_elig).geo_eligibility,
                  self.geo_elig)

  def testGeoShare(self):
    """Share (proportion) of each geo is estimated."""
    geo_share = TBRMMData(self.df, self.response).geo_share
    share0 = self.sums['0'] / sum(value for key, value in self.sums.items())
    self.assertEqual(geo_share['0'], share0)
    self.assertTrue(len(geo_share) == self.n_geos)  # pylint: disable=g-generic-assert
    self.assertAlmostEqual(sum(geo_share), 1.0, places=5)

  def testAssignableGeosDefault(self):
    """The assignable geos is equal to all geos by default."""
    self.assertSetEqual(TBRMMData(self.df, self.response).assignable,
                        self.geos)

  def testAssignableGeos(self):
    """The assignable geos is equal to all geos minus those excluded."""
    df_geo_elig = self.geo_elig.data
    df_geo_elig.loc['1'] = [0, 0, 1]  # Exclude geo '1'.
    new_assignable_geos = self.geos - {'1'}
    new_geo_elig = GeoEligibility(df_geo_elig)
    self.assertSetEqual(
        TBRMMData(self.df, self.response, new_geo_elig).assignable,
        new_assignable_geos)

  def testGeosInData(self):
    """Geos-in-data property returns the set of geos found in the data set."""
    self.assertSetEqual(
        TBRMMData(self.df, self.response).geos_in_data,
        self.geos)

  def testGeoEligibilityIsMatchedToGeosInData(self):
    """The geo_eligibility object is matched to the subset of geos in data."""
    df_geo_elig = self.geo_elig.data  # Has data for geos '0', ..., '4'.
    geo_elig = GeoEligibility(df_geo_elig)
    # Change the data to a subset.
    geo_subset = {'1', '2', '3'}
    df_new = self.df.loc[self.df.geo.isin(geo_subset)]
    self.assertCountEqual(
        TBRMMData(df_new, self.response, geo_elig).geo_eligibility.data.index,
        geo_subset)
    self.assertCountEqual(
        TBRMMData(df_new, self.response, geo_elig).assignable,
        geo_subset)

  def testGeoEligibilityCanBeASubsetOfGeosInData(self):
    """The geo_eligibility object can refer to a subset of geos in the data.
    """
    geo_subset = {'1', '2', '3'}
    df_geo_elig = self.geo_elig.data.loc[geo_subset]
    geo_elig = GeoEligibility(df_geo_elig)
    self.assertCountEqual(
        TBRMMData(self.df, self.response, geo_elig).geo_eligibility.data.index,
        geo_subset)

  def testRequiredGeosMustBeInData(self):
    """The geos that must be included in the design must also be in the data."""
    df_geo_elig = self.geo_elig.data
    df_geo_elig.loc['1'] = [1, 1, 0]  # Geo '1' must be included.
    df_geo_elig.loc['2'] = [1, 0, 0]  # Geo '2' must be included.
    df_geo_elig.loc['3'] = [0, 1, 0]  # Geo '3' must be included.
    geo_elig = GeoEligibility(df_geo_elig)
    # Extract a subset of the data with geos '0' and '1' only.
    geo_subset = {'0', '1'}
    df_new = self.df.loc[self.df.geo.isin(geo_subset)]
    with self.assertRaisesRegex(
        ValueError,
        r'Required geos \[\'2\', \'3\'\] were not found in the data'):
      TBRMMData(df_new, self.response, geo_elig)

  def testGeosAreStrings(self):
    """Non-string valued geos will be translated to strings."""
    df = self.df
    df['geo'] = [int(geo) for geo in df['geo']]  # Geos are now integers.
    self.assertSetEqual(TBRMMData(self.df, self.response).geos_in_data,
                        self.geos)


class TBRMMDataGeoIndexTest(TBRMMDataTest):
  """Test the property geo_index and methods that use it."""

  def setUp(self):
    super().setUp()
    self.some_geo_index = ['4', '2', '0']

  def testGetterDefault(self):
    """Default value is None."""
    self.assertIsNone(TBRMMData(self.df, self.response).geo_index)

  def testSetAndGet(self):
    """Default value is None."""
    d = TBRMMData(self.df, self.response)
    d.geo_index = self.some_geo_index
    self.assertIs(d.geo_index, self.some_geo_index)

  def testNonexistingGeosAreCaught(self):
    """Geos that are not in the data are not accepted."""
    d = TBRMMData(self.df, self.response)
    with self.assertRaisesRegex(ValueError, r'Unassignable geo\(s\): 5'):
      d.geo_index = ['5', '4', '3']

  def testUnassignableGeosAreCaught(self):
    """Geos that are not assignable (although in the data) are not accepted."""
    df_geo_elig = self.geo_elig.data
    df_geo_elig.loc['1'] = [0, 0, 1]  # Exclude geo '1'.
    df_geo_elig.loc['3'] = [0, 0, 1]  # Exclude geo '3'.
    geo_elig = GeoEligibility(df_geo_elig)
    d = TBRMMData(self.df, self.response, geo_elig)
    with self.assertRaisesRegex(ValueError, r'Unassignable geo\(s\): 1, 3'):
      d.geo_index = self.geos  # Attempt to use all geos.

  def testGeoAssignmentsAreIndices(self):
    """Geo assignments are corresponding indices, not geo IDs."""
    d = TBRMMData(self.df, self.response)
    d.geo_index = self.geos
    a = d.geo_assignments
    self.assertSetEqual(a.all, {0, 1, 2, 3, 4})
    d.geo_index = ['4', '3']
    a = d.geo_assignments
    self.assertSetEqual(a.all, {0, 1})
    d.geo_index = ['0', '1']
    a = d.geo_assignments
    self.assertSetEqual(a.all, {0, 1})

  def testAggregateTimeseries(self):
    """Aggregate time series for the given geos are returned."""
    d = TBRMMData(self.df, self.response)
    d.geo_index = ['2']
    self.assertTrue(all(d.aggregate_time_series({0}) == self.sales['2']))
    d.geo_index = ['2', '3']
    sum_sales = [x + y for x, y in zip(self.sales['2'], self.sales['3'])]
    self.assertTrue(all(d.aggregate_time_series({0, 1}) == sum_sales))

  def testAggregateGeoShare(self):
    """Aggregate share for the given geos are returned."""
    d = TBRMMData(self.df, self.response)
    d.geo_index = ['2']
    self.assertEqual(d.aggregate_geo_share({0}), d.geo_share['2'])
    d.geo_index = ['2', '3']
    sum_share = d.geo_share['2'] + d.geo_share['3']
    self.assertEqual(d.aggregate_geo_share({0, 1}), sum_share)
    d.geo_index = self.geos
    self.assertAlmostEqual(d.aggregate_geo_share({0, 1, 2, 3, 4}),
                           1.0,
                           places=5)


class CorrelationsAndNoisyGeosTest(unittest.TestCase):
  """Test the property leave_one_out_correlations."""

  def setUp(self):
    # Create geo time series data with varying correlations.
    super().setUp()

    geos = ['X', 'Y', 'Z']
    n_days = 10
    dates = pd.date_range('2020-03-01', periods=n_days)

    df_data = [{'date': date, 'geo': geo} for geo, date in
               itertools.product(geos, dates)]
    df = pd.DataFrame(df_data)

    # Time series for the three geos.
    x = [-1.45, -0.77, 0.1, 0.59, 0.12, -0.06, -1.82, 0.52, -0.06, 2.19]
    y = [1.08, -0.18, -0.65, -0.81, 0.61, 0.0, -1.5, 0.21, -1.1, 2.04]
    z = [1.1, -0.17, -0.65, -0.83, 0.61, 0.0, -1.49, 0.21, -1.1, 2.03]

    self.response = 'sales'
    df[self.response] = 0.0
    df.loc[df.geo == 'X', self.response] = x
    df.loc[df.geo == 'Y', self.response] = y
    df.loc[df.geo == 'Z', self.response] = z
    self.df = df

  def testCorrelations(self):
    """The correlations are correctly calculated."""
    d = TBRMMData(self.df, self.response)
    self.assertAlmostEqual(d.leave_one_out_correlations['X'], 0.499, places=3)
    self.assertAlmostEqual(d.leave_one_out_correlations['Y'], 0.862, places=3)
    self.assertAlmostEqual(d.leave_one_out_correlations['Z'], 0.855, places=3)

  def testNoisyGeos(self):
    """Geo is noisy if correlation < 0.5."""
    d = TBRMMData(self.df, self.response)
    self.assertAlmostEqual(d.leave_one_out_correlations['X'], 0.499, places=3)
    self.assertCountEqual(d.noisy_geos, {'X'})

  def testNotNoisyGeos(self):
    """Geo is not noisy if correlation >= 0.5."""
    # Extract geos X and Y which have mutual correlation 0.503.
    df = self.df.loc[self.df.geo.isin({'X', 'Y'})]
    d = TBRMMData(df, self.response)
    self.assertAlmostEqual(d.leave_one_out_correlations['X'], 0.503, places=3)
    self.assertFalse(d.noisy_geos)

if __name__ == '__main__':
  unittest.main()
