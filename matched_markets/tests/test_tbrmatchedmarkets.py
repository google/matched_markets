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
"""Test TBR Matched Markets.
"""
import itertools
import os
import types

from absl import flags
from matched_markets.methodology import geoeligibility
from matched_markets.methodology import tbrmatchedmarkets
from matched_markets.methodology import tbrmmdata
from matched_markets.methodology import tbrmmdesignparameters
from matched_markets.methodology import tbrmmdiagnostics
import pandas as pd


import unittest

TBRMatchedMarkets = tbrmatchedmarkets.TBRMatchedMarkets
TBRMMDiagnostics = tbrmmdiagnostics.TBRMMDiagnostics
TBRMMDesignParameters = tbrmmdesignparameters.TBRMMDesignParameters
TBRMMData = tbrmmdata.TBRMMData




class TBRMatchedMarketsTest(unittest.TestCase):
  """Test class TBRMatchedMarkets."""

  def setUp(self):
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

    par = TBRMMDesignParameters(n_test=14, iroas=3.0)

    df[response_column] = 0.0
    sales = {}
    impact = {}
    for geo in geos:
      sales_time_series = day_geo_sales(int(geo), n_days)
      df.loc[df.geo == geo, response_column] = sales_time_series
      sales[geo] = sales_time_series
      dg = TBRMMDiagnostics(sales_time_series, par)
      impact[geo] = dg.estimate_required_impact(par.rho_max)

    self.impact = impact
    self.geos = geos
    self.par = par
    self.data = TBRMMData(df, response_column)
    self.response = response_column
    self.default_geo_eligibility_data = self.data.geo_eligibility.data
    self.df = df

  def testMandatoryArguments(self):
    """Both arguments are mandatory."""
    with self.assertRaisesRegex(
        TypeError,
        r'missing 2 required positional arguments: \'data\' '
        'and \'parameters\''):
      TBRMatchedMarkets()

  def testDataAttribute(self):
    """Data attribute must have the user-supplied data object."""
    self.assertEqual(TBRMatchedMarkets(self.data, self.par).data, self.data)

  def testParAttribute(self):
    """'Parameters' attribute must have the user-supplied parameters object."""
    self.assertEqual(TBRMatchedMarkets(self.data, self.par).parameters,
                     self.par)

  def testGeoReqImpactGeoSet(self):
    """Geo_req_impact must have an index equal to the geo IDs in the data."""
    mm = TBRMatchedMarkets(self.data, self.par)
    self.assertCountEqual(mm.geo_req_impact.index, self.geos)

  def testGeoReqImpactValue(self):
    """Geo_req_impact is the minimum required impact indexed by geo."""
    mm = TBRMatchedMarkets(self.data, self.par)
    # Test a couple of geos.
    self.assertEqual(mm.geo_req_impact['1'], self.impact['1'])
    self.assertEqual(mm.geo_req_impact['3'], self.impact['3'])

  def testNoBudgetSpecifiedNoGeoOverBudget(self):
    """If no budget range is specified, geos_over_budget returns set().
    """
    par = TBRMMDesignParameters(n_test=14, iroas=1.0)
    self.assertFalse(TBRMatchedMarkets(self.data, par).geos_over_budget)

  def testGeosOverBudget(self):
    """Geos exceeding the given maximum implied budget are identified."""
    iroas = 2.5
    # Maximum budget is the implied budget of geo '2', hence the geos over
    # budget will be '3' and '4'.
    budget_max = self.impact['2'] / iroas
    par = TBRMMDesignParameters(n_test=14,
                                iroas=iroas,
                                budget_range=(0.1, budget_max))
    mm = TBRMatchedMarkets(self.data, par)
    self.assertCountEqual(mm.geos_over_budget, {'3', '4'})

  def testNoTreatmentShareRangeSpecifiedNoGeoTooLarge(self):
    """If no treatment_share_range is specified, geos_too_large returns set().
    """
    par = TBRMMDesignParameters(n_test=14, iroas=1.0)
    self.assertSetEqual(TBRMatchedMarkets(self.data, par).geos_too_large,
                        set())

  def testGeosTooLarge(self):
    """Geos exceeding the given maximum share (fraction) are identified."""
    # Set the maximum to that of geo '3'.
    max_share = self.data.geo_share['3']
    share_range = (max_share / 2.0, max_share)
    par = TBRMMDesignParameters(n_test=14,
                                iroas=3.0,
                                treatment_share_range=share_range)
    mm = TBRMatchedMarkets(self.data, par)
    self.assertCountEqual(mm.geos_too_large, {'4'})

  def testGeosMustInclude(self):
    """Geos that must be included are identified."""
    df_geo_elig = self.default_geo_eligibility_data
    df_geo_elig.loc['1'] = [1, 1, 0]  # Cannot exclude geo 1.
    df_geo_elig.loc['2'] = [1, 0, 0]  # Cannot exclude geo 2.
    df_geo_elig.loc['3'] = [0, 1, 0]  # Cannot exclude geo 3.
    geo_elig = geoeligibility.GeoEligibility(df_geo_elig)
    data = TBRMMData(self.df, self.response, geo_elig)
    par = TBRMMDesignParameters(n_test=14, iroas=3.0)
    mm = TBRMatchedMarkets(data, par)
    self.assertCountEqual(mm.geos_must_include, {'1', '2', '3'})

  def testGeosMustIncludeDefault(self):
    """If no geo eligibility object is given, the set is empty."""
    # Use the default data object without a specific geo_eligibility object.
    mm = TBRMatchedMarkets(self.data, self.par)
    self.assertFalse(mm.geos_must_include)

  def testNPretestMaxDefault(self):
    """if n_pretest_max is not specified, all time points are used."""
    self.assertTrue(
        TBRMatchedMarkets(self.data, self.par).data.df.columns.equals(
            self.data.df.columns))

  def testNPretestMaxSpecified(self):
    """if n_pretest_max is specified, a subset of time points is used."""
    par = TBRMMDesignParameters(n_test=14, iroas=1.0, n_pretest_max=3)
    self.assertTrue(
        TBRMatchedMarkets(self.data, par).data.df.columns.equals(
            pd.to_datetime(['2020-03-19', '2020-03-20', '2020-03-21'])))

  def testMostRecentObservationsAreConsidered(self):
    """if n_pretest_max is specified, we select the n_pretest_max most recent observations."""
    # reverse the order of the time points
    data = TBRMMData(
        self.df.sort_values(by='date', ascending=False), self.response)
    par = TBRMMDesignParameters(n_test=14, iroas=1.0, n_pretest_max=3)
    self.assertTrue(
        TBRMatchedMarkets(data, par).data.df.columns.equals(
            pd.to_datetime(['2020-03-19', '2020-03-20', '2020-03-21'])))


class GeosWithinConstraintsTest(TBRMatchedMarketsTest):
  """Test property 'geos_within_constraints'."""

  def testDefault(self):
    """By default all geos are within constraints."""
    mm = TBRMatchedMarkets(self.data, self.par)
    self.assertCountEqual(mm.geos_within_constraints, self.geos)

  def testNGeosMaxSpecified(self):
    """Max. number of geos specified yields a subset of the max. impact geos."""
    test_base_dir = 'matched_markets/csv/'
    data_dir = os.path.join("", test_base_dir)
    example_data = pd.read_csv(
        open(os.path.join(data_dir, 'salesandcost.csv')),
        parse_dates=['date'])
    data = TBRMMData(example_data, 'sales')
    par = TBRMMDesignParameters(n_test=14, iroas=3.0, n_geos_max=6)
    mm = TBRMatchedMarkets(data, par)
    self.assertCountEqual(mm.geos_within_constraints,
                          {'1', '2', '3', '4', '5', '7'})

  def testTooLargeGeosAreExcluded(self):
    """Geos that are too large are excluded."""
    max_share = self.data.geo_share['3']  # Exclude '4'.
    share_range = (max_share / 2.0, max_share)
    par = TBRMMDesignParameters(n_test=14,
                                iroas=3.0,
                                treatment_share_range=share_range)
    mm = TBRMatchedMarkets(self.data, par)
    self.assertCountEqual(mm.geos_within_constraints, {'0', '1', '2', '3'})

  def testTooLargeGeosAreExcluded_with_NGeosMax(self):
    """Geos that are too large are excluded, and restricted to n_geos_max."""
    max_share = self.data.geo_share['3']  # Exclude '4'.
    share_range = (max_share / 2.0, max_share)
    par = TBRMMDesignParameters(n_test=14,
                                iroas=3.0,
                                n_geos_max=3,
                                treatment_share_range=share_range)
    mm = TBRMatchedMarkets(self.data, par)
    self.assertCountEqual(mm.geos_within_constraints, {'1', '2', '3'})

  def testNonEligibilityOverridesTooLargeGeo(self):
    """Geos that are too large are not excluded if not eligible."""
    max_share = self.data.geo_share['3']  # Geo '4' is 'too large'.
    share_range = (max_share / 2.0, max_share)
    df_geo_elig = self.default_geo_eligibility_data
    df_geo_elig.loc['4'] = [0, 1, 0]  # Cannot exclude geo 4.
    geo_elig = geoeligibility.GeoEligibility(df_geo_elig)
    data = TBRMMData(self.df, self.response, geo_elig)
    par = TBRMMDesignParameters(n_test=14,
                                iroas=3.0,
                                treatment_share_range=share_range)
    mm = TBRMatchedMarkets(data, par)
    self.assertCountEqual(mm.geos_within_constraints, self.geos)

  def testNonEligibilityOverridesTooLargeGeo_with_NGeosMax(self):
    """Too large geos excluded if not eligible & restricted <= n_geos_max."""
    max_share = self.data.geo_share['3']  # Geo '4' is 'too large'.
    share_range = (max_share / 2.0, max_share)
    df_geo_elig = self.default_geo_eligibility_data
    df_geo_elig.loc['4'] = [0, 1, 0]  # Cannot exclude geo 4.
    geo_elig = geoeligibility.GeoEligibility(df_geo_elig)
    data = TBRMMData(self.df, self.response, geo_elig)
    par = TBRMMDesignParameters(n_test=14,
                                iroas=3.0,
                                n_geos_max=3,
                                treatment_share_range=share_range)
    mm = TBRMatchedMarkets(data, par)
    self.assertCountEqual(mm.geos_within_constraints, {'4', '3', '2'})

  def testGeosOverBudgetAreExcluded(self):
    """Geos that must be included are identified."""
    iroas = 2.5
    budget_max = self.impact['2'] / iroas  # Exclude '3', '4'.
    par = TBRMMDesignParameters(n_test=14,
                                iroas=iroas,
                                budget_range=(0.1, budget_max))
    mm = TBRMatchedMarkets(self.data, par)
    self.assertCountEqual(mm.geos_within_constraints, {'0', '1', '2'})

  def testGeosOverBudgetAreExcluded_with_NGeosMax(self):
    """Geos that must be included are identified & <= n_geos_max."""
    iroas = 2.5
    budget_max = self.impact['2'] / iroas  # Exclude '3', '4'.
    par = TBRMMDesignParameters(n_test=14,
                                iroas=iroas,
                                n_geos_max=2,
                                budget_range=(0.1, budget_max))
    mm = TBRMatchedMarkets(self.data, par)
    self.assertCountEqual(mm.geos_within_constraints, {'1', '2'})

  def testNonEligibilityOverridesGeoOverBudget(self):
    """Geos that are too large are not excluded if not eligible."""
    df_geo_elig = self.default_geo_eligibility_data
    df_geo_elig.loc['4'] = [0, 1, 0]  # Cannot exclude geo 4.
    geo_elig = geoeligibility.GeoEligibility(df_geo_elig)
    data = TBRMMData(self.df, self.response, geo_elig)
    iroas = 2.5
    budget_max = self.impact['2'] / iroas  # Exclude '3', '4'.
    par = TBRMMDesignParameters(n_test=14,
                                iroas=iroas,
                                budget_range=(0.1, budget_max))
    mm = TBRMatchedMarkets(data, par)
    self.assertCountEqual(mm.geos_within_constraints, {'0', '1', '2', '4'})

  def testNonEligibilityOverridesGeoOverBudget_with_NGeosMax(self):
    """Too large geos not excluded if not eligible but <= n_geos_max."""
    df_geo_elig = self.default_geo_eligibility_data
    df_geo_elig.loc['4'] = [0, 1, 0]  # Cannot exclude geo 4.
    geo_elig = geoeligibility.GeoEligibility(df_geo_elig)
    data = TBRMMData(self.df, self.response, geo_elig)
    iroas = 2.5
    budget_max = self.impact['2'] / iroas  # Exclude '3', '4'.
    par = TBRMMDesignParameters(n_test=14,
                                iroas=iroas,
                                n_geos_max=3,
                                budget_range=(0.1, budget_max))
    mm = TBRMatchedMarkets(data, par)
    self.assertCountEqual(mm.geos_within_constraints, {'1', '2', '4'})


class GeoAssignmentsPropertyTest(TBRMatchedMarketsTest):
  """Test property 'geo_assignments'."""

  def setUp(self):
    super().setUp()
    self.all_indices = {0, 1, 2, 3, 4}

  def testDefault(self):
    """By default all indices are included in the group 'ctx'.

    Group 'ctx' = geos that can be assigned to Control, Treatment, or Excluded.
    """
    mm = TBRMatchedMarkets(self.data, self.par)
    self.assertCountEqual(mm.geo_assignments.ctx, self.all_indices)

  def testExcludedGeo(self):
    """Completely excluded geos (x_fixed) do not appear in geo_assignments.

    If a geo is excluded, geo indices will be renumbered.
    """
    df_geo_elig = self.default_geo_eligibility_data
    df_geo_elig.loc['2'] = [0, 0, 1]  # Group 'x_fixed'.
    geo_elig = geoeligibility.GeoEligibility(df_geo_elig)
    data = TBRMMData(self.df, self.response, geo_elig)
    mm = TBRMatchedMarkets(data, self.par)
    self.assertFalse(mm.geo_assignments.x_fixed)
    self.assertCountEqual(mm.geo_assignments.all, {0, 1, 2, 3})

  def testGeoIndexOrder(self):
    """Geos are indexed from the largest budget (index 0) to smallest (4)."""
    df_geo_elig = self.default_geo_eligibility_data
    df_geo_elig.loc['4'] = [1, 0, 0]  # Group 'c_fixed'. Largest geo -> index 0.
    df_geo_elig.loc['3'] = [0, 1, 0]  # Group 't_fixed'.
    df_geo_elig.loc['1'] = [1, 1, 0]  # Group 'ct'.
    df_geo_elig.loc['0'] = [1, 0, 1]  # Group 'cx'. Smallest geo -> index 4.
    geo_elig = geoeligibility.GeoEligibility(df_geo_elig)
    data = TBRMMData(self.df, self.response, geo_elig)
    mm = TBRMatchedMarkets(data, self.par)
    self.assertCountEqual(mm.geo_assignments.c_fixed, {0})
    self.assertCountEqual(mm.geo_assignments.t_fixed, {1})
    self.assertCountEqual(mm.geo_assignments.ct, {3})
    self.assertCountEqual(mm.geo_assignments.cx, {4})


class TreatmentGroupSizeRangeTest(TBRMatchedMarketsTest):
  """Test method 'treatment_group_size_range'."""

  def testDefault(self):
    """When there are no restrictions, group sizes vary from 1 to n_max - 1.

    The default case with n_max geos allows all geos to be assigned to
    Treatment. There must be at least one treatment geo and one control geo,
    hence the range must be from 1 to n_max - 1.
    """
    par = TBRMMDesignParameters(n_test=14, iroas=2.0)
    mm = TBRMatchedMarkets(self.data, par)
    # There are 5 geos, hence the range must be equal to [1, 2, 3, 4].
    self.assertEqual(mm.treatment_group_size_range(), range(1, 5))

  def testThereAreFixedTreatmentGeos(self):
    """The minimum number of geos must be at least len(t_fixed).

    obj.geo_assignments.t_fixed is the set of treatment geos that are always
    included in Treatment group. Hence the minimum must be adjusted accordingly.
    """
    par = TBRMMDesignParameters(n_test=14, iroas=2.0)
    df_geo_elig = self.data.geo_eligibility.data
    df_geo_elig.loc['1'] = [0, 1, 0]  # Geo '1' is always in Treatment.
    df_geo_elig.loc['2'] = [0, 1, 0]  # Geo '2' is always in Treatment.
    geo_elig = geoeligibility.GeoEligibility(df_geo_elig)
    data = TBRMMData(self.df, self.response, geo_elig)
    mm = TBRMatchedMarkets(data, par)
    # There are 5 geos, 2 fixed to Treatment, but none fixed to control, hence
    # the range must be equal to [2, 3, 4].
    self.assertEqual(mm.treatment_group_size_range(), range(2, 5))

  def testSomeGeosAreNeverInTreatment(self):
    """The max # of geos == len(t) if some geos are never in Treatment group.

    If there are geos that are never assigned to treatment, the maximum
    treatment group size does not have to be restricted.
    """
    par = TBRMMDesignParameters(n_test=14, iroas=2.0)
    df_geo_elig = self.data.geo_eligibility.data
    df_geo_elig.loc['1'] = [1, 0, 0]  # Geo '1' is never in Treatment.
    df_geo_elig.loc['2'] = [1, 0, 1]  # Geo '2' is never in Treatment.
    geo_elig = geoeligibility.GeoEligibility(df_geo_elig)
    data = TBRMMData(self.df, self.response, geo_elig)
    mm = TBRMatchedMarkets(data, par)
    # There are 5 geos, 2 never assigned to Treatment, hence up to 3 geos can be
    # assigned to treatment so the range must be equal to [1, 2, 3].
    self.assertEqual(mm.treatment_group_size_range(), range(1, 4))

  def testRangeIsSpecifiedButExceedsUpperBound(self):
    """The user-specified upper bound of the range is adjusted if too high."""
    par = TBRMMDesignParameters(n_test=14, iroas=2.0,
                                treatment_geos_range=(1, 5))
    mm = TBRMatchedMarkets(self.data, par)
    # There are 5 geos but all are eligible for Treatment so the upper bound
    # should be 4 as one must be reserved for Control. The upper bound is
    # therefore adjusted to 4.
    self.assertEqual(mm.treatment_group_size_range(), range(1, 5))

  def testRangeIsSpecifiedButLowerThanLowerBound(self):
    """The user-specified lower bound of the range is adjusted if too low."""
    par = TBRMMDesignParameters(n_test=14, iroas=2.0,
                                treatment_geos_range=(1, 4))
    df_geo_elig = self.data.geo_eligibility.data
    df_geo_elig.loc['1'] = [0, 1, 0]  # Geo '1' is always in Treatment.
    df_geo_elig.loc['2'] = [0, 1, 0]  # Geo '2' is always in Treatment.
    geo_elig = geoeligibility.GeoEligibility(df_geo_elig)
    data = TBRMMData(self.df, self.response, geo_elig)
    mm = TBRMatchedMarkets(data, par)
    # There are 5 geos, 2 always assigned to Treatment, hence the lower bound
    # must be 2. The upper bound is unchanged (4) as it is the maximum possible.
    self.assertEqual(mm.treatment_group_size_range(), range(2, 5))

  def testRangeIsWithinBounds(self):
    """The user-specified range is used if within the bounds."""
    par = TBRMMDesignParameters(n_test=14, iroas=2.0,
                                treatment_geos_range=(2, 3))
    mm = TBRMatchedMarkets(self.data, par)
    # The maximum range of [1, ..., 4] is adjusted to [2, 3].
    self.assertEqual(mm.treatment_group_size_range(), range(2, 4))


class TreatmentGroupGeneratorTest(TBRMatchedMarketsTest):
  """Test method 'treatment_group_generator'."""

  def setUp(self):
    super().setUp()
    self.mm = TBRMatchedMarkets(self.data, self.par)
    df_geo_elig = self.data.geo_eligibility.data.copy()
    # Assign geo '3' into Treatment group. In the order of size (in terms of
    # required budget), geo '3' will be index 1.
    df_geo_elig.loc['3'] = [0, 1, 0]
    geo_elig = geoeligibility.GeoEligibility(df_geo_elig)
    data = TBRMMData(self.df, self.response, geo_elig)
    self.mmfix = TBRMatchedMarkets(data, self.par)

  def testObjectType(self):
    """The object must be a generator."""
    self.assertIsInstance(
        self.mm.treatment_group_generator(0),
        types.GeneratorType)

  def testGroupSizeZero(self):
    """Group Size == 0 raises an error."""
    with self.assertRaisesRegex(
        ValueError, r'Treatment group size n must be positive'):
      list(self.mm.treatment_group_generator(0))

  def testGroupSizeNegative(self):
    """Group Size < 0 returns nothing."""
    with self.assertRaisesRegex(
        ValueError, r'Treatment group size n must be positive'):
      list(self.mm.treatment_group_generator(-1))

  def testGroupSizeTooLarge(self):
    """Group Size > number of items returns nothing."""
    # The test set has 5 geo indices: 0, ..., 4.
    self.assertEqual(list(self.mm.treatment_group_generator(6)), [])

  def testIndexOrderAndPatterns(self):
    """Correct patterns are generated, smallest indices first."""
    # The test set has 5 geo indices: 0, ..., 4.
    self.assertCountEqual(
        list(self.mm.treatment_group_generator(1)),
        [{0,}, {1,}, {2,}, {3,}, {4,}])
    self.assertCountEqual(
        list(self.mm.treatment_group_generator(2)),
        [{0, 1}, {0, 2}, {0, 3}, {0, 4}, {1, 2}, {1, 3}, {1, 4}, {2, 3},
         {2, 4}, {3, 4}])
    self.assertCountEqual(
        list(self.mm.treatment_group_generator(3)),
        [{0, 1, 2}, {0, 1, 3}, {0, 1, 4}, {0, 2, 3}, {0, 2, 4}, {0, 3, 4},
         {1, 2, 3}, {1, 2, 4}, {1, 3, 4}, {2, 3, 4}])
    self.assertCountEqual(
        list(self.mm.treatment_group_generator(4)),
        [{0, 1, 2, 3}, {0, 1, 2, 4}, {0, 1, 3, 4}, {0, 2, 3, 4}, {1, 2, 3, 4}])
    self.assertCountEqual(
        list(self.mm.treatment_group_generator(5)),
        [{0, 1, 2, 3, 4}])

  def testGroupSizeZeroFixedGeos(self):
    """Group Size == 0 raises an error if there are fixed treatment geos."""
    with self.assertRaisesRegex(
        ValueError, r'Treatment group size n must be positive'):
      list(self.mmfix.treatment_group_generator(0))

  def testGroupSizeNegativeFixedGeos(self):
    """Group Size < 0 raises an error if there are fixed treatment geos."""
    with self.assertRaisesRegex(
        ValueError, r'Treatment group size n must be positive'):
      list(self.mmfix.treatment_group_generator(-1))

  def testGroupSizeTooLargeWithFixedGeos(self):
    """Group Size > number of items returns nothing."""
    # The test set has 5 geo indices: 0, ..., 4.
    self.assertEqual(list(self.mmfix.treatment_group_generator(6)), [])

  def testIndexOrderAndPatternsWithFixedGeos(self):
    """Correct patterns generated, smallest indices first, with a fixed geo."""
    # The test set has 5 geo indices: 0, ..., 4. Geo '3' is fixed; it appears as
    # geo index 1 as it is the second largest one (largest one is index 0).
    self.assertCountEqual(
        list(self.mmfix.treatment_group_generator(1)),
        [{1,}])
    self.assertCountEqual(
        list(self.mmfix.treatment_group_generator(2)),
        [{0, 1}, {1, 2}, {1, 3}, {1, 4}])
    self.assertCountEqual(
        list(self.mmfix.treatment_group_generator(3)),
        [{0, 1, 2}, {0, 1, 3}, {0, 1, 4}, {1, 2, 3}, {1, 2, 4}, {1, 3, 4}])
    self.assertCountEqual(
        list(self.mmfix.treatment_group_generator(4)),
        [{0, 1, 2, 3}, {0, 1, 2, 4}, {0, 1, 3, 4}, {1, 2, 3, 4}])
    self.assertCountEqual(
        list(self.mmfix.treatment_group_generator(5)),
        [{0, 1, 2, 3, 4}])


class ControlGroupGeneratorTest(TBRMatchedMarketsTest):
  """Test method 'treatment_group_generator'."""

  def setUp(self):
    super().setUp()
    df_geo_elig = self.data.geo_eligibility.data.copy()
    # Object mm4: 4 geos.
    # Exclude geo, '1' from the set, use 4 geos for testing.
    df_geo_elig.loc['1'] = [0, 0, 1]
    # Note: the remaining 4 geos will be reindexed as 0, 1, 2, 3.
    geo_elig = geoeligibility.GeoEligibility(df_geo_elig)
    self.data4 = TBRMMData(self.df, self.response, geo_elig)
    self.mm4 = TBRMatchedMarkets(self.data4, self.par)
    df_geo_elig = self.data.geo_eligibility.data.copy()

    # Object mmfix: 3 geos + 1 fixed to control + 1 in group 'ct'.
    # Exclude geo, '1' from the set, use 4 geos for testing.
    df_geo_elig.loc['1'] = [1, 0, 0]  # Geo index 3, assigned to control.
    df_geo_elig.loc['2'] = [1, 1, 0]  # Geo index 2, Control or Treatment only.
    geo_elig = geoeligibility.GeoEligibility(df_geo_elig)
    self.datafix = TBRMMData(self.df, self.response, geo_elig)
    self.mmfix = TBRMatchedMarkets(self.datafix, self.par)

  def testEmptyTreatmentGroup(self):
    """Treatment group must not be empty."""
    with self.assertRaisesRegex(
        ValueError, r'Treatment group must not be empty'):
      list(self.mm4.control_group_generator(set()))

  def testInvalidTreatmentGeos(self):
    """Treatment group must be subset of the available treatment geos."""
    # The available indices are 0, 1, 2, 3.
    with self.assertRaisesRegex(
        ValueError, r'Invalid treatment geo indices: 4, 5'):
      list(self.mm4.control_group_generator({4, 5}))
    # Geo index 3 is fixed to Control, therefore cannot be in Treatment.
    with self.assertRaisesRegex(
        ValueError, r'Invalid treatment geo indices: 3'):
      list(self.mmfix.control_group_generator({2, 3}))

  def testPatternsNoConstraints(self):
    """Correct patterns are generated, no constraints."""
    self.assertCountEqual(
        list(self.mm4.control_group_generator({0,})),
        [{1,}, {2,}, {3,}, {1, 2}, {1, 3}, {2, 3}, {1, 2, 3}])
    self.assertCountEqual(
        list(self.mm4.control_group_generator({1,})),
        [{0,}, {2,}, {3,}, {0, 2}, {0, 3}, {2, 3}, {0, 2, 3}])
    self.assertCountEqual(
        list(self.mm4.control_group_generator({0, 1})),
        [{2,}, {3,}, {2, 3}])
    self.assertCountEqual(
        list(self.mm4.control_group_generator({2, 3})),
        [{0,}, {1,}, {0, 1}])
    self.assertCountEqual(
        list(self.mm4.control_group_generator({0, 1, 2})),
        [{3,}])

  def testPatternsControlGroupSizeConstrained(self):
    """Correct patterns are generated, control group size constrained."""
    par = TBRMMDesignParameters(n_test=14,
                                iroas=3.0,
                                control_geos_range=(1, 1))
    mm = TBRMatchedMarkets(self.data4, par)
    self.assertCountEqual(
        list(mm.control_group_generator({0,})),
        [{1,}, {2,}, {3,}])
    par = TBRMMDesignParameters(n_test=14,
                                iroas=3.0,
                                control_geos_range=(2, 3))
    mm = TBRMatchedMarkets(self.data4, par)
    self.assertCountEqual(
        list(mm.control_group_generator({0,})),
        [{1, 2}, {1, 3}, {2, 3}, {1, 2, 3}])

  def testPatternsGeoRatioConstrained100(self):
    """Correct patterns are generated if ratio is constrained to +/- 100%."""
    par = TBRMMDesignParameters(n_test=14,
                                iroas=3.0,
                                geo_ratio_tolerance=1.0)
    mm = TBRMatchedMarkets(self.data, par)  # Use data with 5 geos.
    # One treatment geo. Possible ratios (of control/treatment geos): 1/1, 2/1.
    self.assertCountEqual(
        list(mm.control_group_generator({0,})),
        [{1,}, {2,}, {3,}, {4,}, {1, 2}, {1, 3}, {1, 4}, {2, 3}, {2, 4},
         {3, 4}])
    # Two treatment geos. Possible ratios: 1/2, 3/2.
    self.assertCountEqual(
        list(mm.control_group_generator({0, 1})),
        [{2,}, {3,}, {4,}, {2, 3}, {2, 4}, {3, 4}, {2, 3, 4}])

  def testPatternsGeoRatioConstrained99(self):
    """Correct patterns are generated if ratio is constrained to +/- 99%."""
    par = TBRMMDesignParameters(n_test=14,
                                iroas=3.0,
                                geo_ratio_tolerance=0.99)
    mm = TBRMatchedMarkets(self.data, par)  # 5 geos.
    # One treatment geo. Possible ratios: 1/1.
    self.assertCountEqual(
        list(mm.control_group_generator({0,})),
        [{1,}, {2,}, {3,}, {4,}])
    # Two treatment geos. Possible ratios: 2/2, 3/2.
    # 1/2 is not allowed because 1/2 = 0.5 < 1 / (1 + 0.99).
    self.assertCountEqual(
        list(mm.control_group_generator({0, 1})),
        [{2, 3}, {2, 4}, {3, 4}, {2, 3, 4}])

  def testPatternsGeoRatioConstrained50(self):
    """Correct patterns are generated if ratio is constrained to +/- 50%."""
    par = TBRMMDesignParameters(n_test=14,
                                iroas=3.0,
                                geo_ratio_tolerance=0.5)
    mm = TBRMatchedMarkets(self.data, par)  # 5 geos.
    # One treatment geo. Possible ratios: 1/1.
    self.assertCountEqual(
        list(mm.control_group_generator({0,})),
        [{1,}, {2,}, {3,}, {4,}])
    # Two treatment geos. Possible ratios: 1/2, 2/2, 3/2.
    # 1/2 is not allowed because 1/2 = 0.5 < 1 / (1 + 0.5).
    self.assertCountEqual(
        list(mm.control_group_generator({0, 1})),
        [{2, 3}, {2, 4}, {3, 4}, {2, 3, 4}])

  def testPatternsFixedControlGeos(self):
    """Correct patterns are generated with fixed control geos."""
    # Here we have 2 fixed control geos. Index 3: c_fixed, index 2: ct (index 2
    # is not in treatment, hence it must be in control. No other constraints on
    # group size.
    self.assertCountEqual(
        list(self.mmfix.control_group_generator({0,})),
        [{2, 3}, {1, 2, 3}, {2, 3, 4}, {1, 2, 3, 4}])
    self.assertCountEqual(
        list(self.mmfix.control_group_generator({1, 4})),
        [{2, 3}, {0, 2, 3}])

  def testPatternsTreatmentGeoInGroupCT(self):
    """Correct patterns are generated if a treatment geo is in group 'ct'."""
    # Index 3: c_fixed, index 2: ct.
    self.assertCountEqual(
        list(self.mmfix.control_group_generator({2,})),
        [{3}, {0, 3}, {1, 3}, {3, 4}, {0, 1, 3}, {0, 3, 4}, {1, 3, 4},
         {0, 1, 3, 4}])


class CountMaxDesignsTest(TBRMatchedMarketsTest):
  """Test method 'count_max_designs'."""

  def setUp(self):
    super().setUp()
    self.df_geo_elig = self.data.geo_eligibility.data.copy()

  def testDefault(self):
    """Default geo eligibility and no group size restrictions except >= 1."""
    mm = TBRMatchedMarkets(self.data, self.par)
    # The default geo eligibility matrix has 5 geos, no restrictions ('ctx').
    # n = number of freely assignable geos (geos in 'ctx').
    # 3^n - 2^(n + 1) + 1 == 3^5 - 2^6 + 1 = 180.
    self.assertEqual(mm.count_max_designs(), 180)

  def testOneGeoExcluded_SizeUnbounded(self):
    """One geo excluded (x_fixed), no group size restrictions except >= 1."""
    df_geo_elig = self.df_geo_elig
    df_geo_elig.loc['0'] = [0, 0, 1]
    geo_elig = geoeligibility.GeoEligibility(df_geo_elig)
    data = TBRMMData(self.df, self.response, geo_elig)
    mm = TBRMatchedMarkets(data, self.par)
    # n = number of freely assignable geos (geos in 'ctx').
    # 3^n - 2^(n + 1) + 1 == 3^4 - 2^5 + 1 == 50.
    self.assertEqual(mm.count_max_designs(), 50)

  def testOneControlGeoFixed_SizeUnbounded(self):
    """One control geo fixed and no group size restrictions except >= 1."""
    # Default except one fixed control geo (c_fixed).
    df_geo_elig = self.df_geo_elig
    df_geo_elig.loc['0'] = [1, 0, 0]
    geo_elig = geoeligibility.GeoEligibility(df_geo_elig)
    data = TBRMMData(self.df, self.response, geo_elig)
    mm = TBRMatchedMarkets(data, self.par)
    # n = number of freely assignable geos (geos in 'ctx'). One fixed.
    # 3^n - 2^n == 3^4 - 2^4 == 65.
    self.assertEqual(mm.count_max_designs(), 65)

  def testOneTreatmentFixedDefault_SizeUnbounded(self):
    """One treatment geo fixed and no group size restrictions except >= 1."""
    df_geo_elig = self.df_geo_elig
    df_geo_elig.loc['0'] = [1, 0, 0]
    geo_elig = geoeligibility.GeoEligibility(df_geo_elig)
    data = TBRMMData(self.df, self.response, geo_elig)
    mm = TBRMatchedMarkets(data, self.par)
    # n = number of freely assignable geos (geos in 'ctx'). One fixed.
    # Total 3^4 - 2^4 == 65.
    self.assertEqual(mm.count_max_designs(), 65)

  def testOneGeoCT_SizeUnbounded(self):
    """One treatment geo in 'ct' and no group size restrictions except >= 1."""
    df_geo_elig = self.df_geo_elig
    df_geo_elig.loc['0'] = [1, 1, 0]
    geo_elig = geoeligibility.GeoEligibility(df_geo_elig)
    data = TBRMMData(self.df, self.response, geo_elig)
    mm = TBRMatchedMarkets(data, self.par)
    # n = number of freely assignable geos (geos in 'ctx'). One in 'ct'.
    # 2 * (3^n - 2^(n+1) + 2^n) == 2 * (3^4 - 2^5 + 2^4) = 2 * 65 = 130.
    self.assertEqual(mm.count_max_designs(), 130)

  def testOneGeoCX_SizeUnbounded(self):
    """One treatment geo in 'cx' and no group size restrictions except >= 1."""
    df_geo_elig = self.df_geo_elig
    df_geo_elig.loc['0'] = [1, 0, 1]
    geo_elig = geoeligibility.GeoEligibility(df_geo_elig)
    data = TBRMMData(self.df, self.response, geo_elig)
    mm = TBRMatchedMarkets(data, self.par)
    # N freely assignable geos and one geo in group 'cx'.
    # 2 * (3^n + 2^(n+1) + 1) + 2^n - 1.
    self.assertEqual(mm.count_max_designs(), 115)

  def testOneGeoTX_SizeUnbounded(self):
    """One treatment geo in 'tx' and no group size restrictions except >= 1."""
    df_geo_elig = self.df_geo_elig
    df_geo_elig.loc['0'] = [0, 1, 1]
    geo_elig = geoeligibility.GeoEligibility(df_geo_elig)
    data = TBRMMData(self.df, self.response, geo_elig)
    mm = TBRMatchedMarkets(data, self.par)
    # N freely assignable geos and one geo in group 'tx'.
    # 2 * (3^n + 2^(n+1) + 1) + 2^n - 1.
    self.assertEqual(mm.count_max_designs(), 115)

  def testDefault_SizeBounded(self):
    """Default geo eligibility, group size restriction."""
    # The default geo eligibility matrix has 5 geos, no restrictions ('ctx').
    par = TBRMMDesignParameters(n_test=14,
                                iroas=3.0,
                                control_geos_range=(2, 2),
                                treatment_geos_range=(2, 2))
    mm = TBRMatchedMarkets(self.data, par)
    # Total choose(5, 2) * choose(3, 2) == 30.
    self.assertEqual(mm.count_max_designs(), 30)

  def testDefault_MaxRatio(self):
    """Default geo eligibility, group size restriction."""
    # The default geo eligibility matrix has 5 geos, no restrictions ('ctx').
    par = TBRMMDesignParameters(n_test=14,
                                iroas=3.0,
                                geo_ratio_tolerance=0.01)
    # Only ratio 1.0 is allowed (1/1, 2/2).
    # Total choose(5, 1) * choose(4, 1) + choose(5, 2) * choose(3, 2) == 50.
    mm = TBRMatchedMarkets(self.data, par)
    self.assertEqual(mm.count_max_designs(), 50)

  def testAllGeoDifferentGroup_SizeUnbounded(self):
    """All geos in different groups, no group size restrictions except >= 1."""
    df_geo_elig = self.df_geo_elig
    df_geo_elig.loc['4'] = [1, 0, 0]  # 0 - 'c_fixed'.
    df_geo_elig.loc['3'] = [1, 0, 1]  # 1 - 'cx'.
    df_geo_elig.loc['2'] = [0, 1, 1]  # 2 - 'tx'.
    df_geo_elig.loc['1'] = [1, 1, 0]  # 3 - 'ct'.
    df_geo_elig.loc['0'] = [1, 1, 1]  # 4 - 'ctx'.
    # 20 eligible designs.
    # 0 1 2 3 4 | 0 1 2 3 4 | 0 1 2 3 4 | 0 1 2 3 4 |
    # c c t c c | c c x c t | c x t c c | c x x c t |
    # . . . c t | . . . t c | . . . c t | . . . t c |
    # . . . c x | . . . t t | . . . c x | . . . t t |
    # . . . t c | . . . t x | . . . t c | . . . t x |
    # . . . t t |           | . . . t t |           |
    # . . . t x |           | . . . t x |           |
    geo_elig = geoeligibility.GeoEligibility(df_geo_elig)
    data = TBRMMData(self.df, self.response, geo_elig)
    mm = TBRMatchedMarkets(data, self.par)
    self.assertEqual(mm.count_max_designs(), 20)

  def testAllGeoDifferentGroup_SizeBounded(self):
    """All geos in different groups, limit to size 2 only."""
    df_geo_elig = self.df_geo_elig
    df_geo_elig.loc['4'] = [1, 0, 0]  # 0 - 'c_fixed'.
    df_geo_elig.loc['3'] = [1, 0, 1]  # 1 - 'cx'.
    df_geo_elig.loc['2'] = [0, 1, 1]  # 2 - 'tx'.
    df_geo_elig.loc['1'] = [1, 1, 0]  # 3 - 'ct'.
    df_geo_elig.loc['0'] = [1, 1, 1]  # 4 - 'ctx'.
    # Group sizes can vary from 2 or 3, only 7 eligible designs.
    # 0 1 2 3 4 | 0 1 2 3 4 |
    # c c t c t | c c x t t |
    # c c t t c | c x t c t |
    # c c t t t | c x t t c |
    # c c t t x |           |
    geo_elig = geoeligibility.GeoEligibility(df_geo_elig)
    data = TBRMMData(self.df, self.response, geo_elig)
    par = TBRMMDesignParameters(n_test=14,
                                iroas=3.0,
                                control_geos_range=(2, 3),
                                treatment_geos_range=(2, 3))
    mm = TBRMatchedMarkets(data, par)
    self.assertEqual(mm.count_max_designs(), 7)

  def testAllGeoDifferentGroup_MaxRatioFixed(self):
    """All geos in different groups, limit to size 2 only."""
    df_geo_elig = self.df_geo_elig
    df_geo_elig.loc['4'] = [1, 0, 0]  # 0 - 'c_fixed'.
    df_geo_elig.loc['3'] = [1, 0, 1]  # 1 - 'cx'.
    df_geo_elig.loc['2'] = [0, 1, 1]  # 2 - 'tx'.
    df_geo_elig.loc['1'] = [1, 1, 0]  # 3 - 'ct'.
    df_geo_elig.loc['0'] = [1, 1, 1]  # 4 - 'ctx'.
    # Only control/treatment geo ratios 1/1, 2/3, 2/1, 1/2 allowed, 14 eligible
    # designs.
    # 0 1 2 3 4 | 0 1 2 3 4 | 0 1 2 3 4 | 0 1 2 3 4 |
    # c c t c t | c c x t t | c x t c t | c x x c t
    # . . . t c | . . . t x | . . . c x | . . . t c
    # . . . t t |           | . . . t c | . . . t t
    # . . . t x |           | . . . t x | . . . t x
    geo_elig = geoeligibility.GeoEligibility(df_geo_elig)
    data = TBRMMData(self.df, self.response, geo_elig)
    par = TBRMMDesignParameters(n_test=14,
                                iroas=3.0,
                                geo_ratio_tolerance=1.0)
    mm = TBRMatchedMarkets(data, par)
    self.assertEqual(mm.count_max_designs(), 14)


if __name__ == '__main__':
  unittest.main()
