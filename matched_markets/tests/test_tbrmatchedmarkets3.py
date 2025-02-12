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
"""Test TBR Matched Markets greedy search.
"""
import itertools

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


class DesignWithinConstraints(unittest.TestCase):
  """Test method 'design_within_constraints'."""

  def setUp(self):
    super().setUp()
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
    self.data.geo_index = ['0', '1', '2', '3', '4']

  def testWithinConstraint(self):
    """check that _constraint_not_satisfied returns False when it's ok."""
    mm = TBRMatchedMarkets(self.data, self.par)
    self.assertFalse(mm._constraint_not_satisfied(0.5, 0, 1))

  def testNotWithinConstraint(self):
    """check that _constraint_not_satisfied returns False."""
    mm = TBRMatchedMarkets(self.data, self.par)
    self.assertTrue(mm._constraint_not_satisfied(0.5, 0, 0.4))
    self.assertTrue(mm._constraint_not_satisfied(0.5, 0.6, 1))

  def testVolumeRatioBetweenBounds(self):
    """Volume ratio between the two groups is not within 1/2 and 2."""
    self.par.volume_ratio_tolerance = 1
    mm = TBRMatchedMarkets(self.data, self.par)
    treatment_group = set([0])
    control_group = set([4])
    self.assertFalse(
        mm.design_within_constraints(treatment_group, control_group))
    treatment_group = set([4])
    control_group = set([0])
    self.assertFalse(
        mm.design_within_constraints(treatment_group, control_group))

  def testGeoRatioBetweenBounds(self):
    """Number of geos ratio between the two groups is not within 1/2 and 2."""
    self.par.geo_ratio_tolerance = 1
    mm = TBRMatchedMarkets(self.data, self.par)
    treatment_group = set([0])
    control_group = set([2, 3, 4])
    self.assertFalse(
        mm.design_within_constraints(treatment_group, control_group))
    treatment_group = set([2, 3, 4])
    control_group = set([0])
    self.assertFalse(
        mm.design_within_constraints(treatment_group, control_group))

  def testTreatmentShareBetweenBounds(self):
    """Treatment response share between the two groups is not within (0.4,0.7)."""
    self.par.treatment_share_range = (0.4, 0.7)
    mm = TBRMatchedMarkets(self.data, self.par)
    treatment_group = set([0])
    control_group = set([2, 3, 4])
    self.assertFalse(
        mm.design_within_constraints(treatment_group, control_group))
    treatment_group = set([2, 3, 4])
    control_group = set([0])
    self.assertFalse(
        mm.design_within_constraints(treatment_group, control_group))

  def testNumberTreatmentGeosBetweenBounds(self):
    """Treatment group size is not within (2,3)."""
    self.par.treatment_geos_range = (2, 3)
    mm = TBRMatchedMarkets(self.data, self.par)
    treatment_group = set([0])
    control_group = set([2, 3, 4])
    self.assertFalse(
        mm.design_within_constraints(treatment_group, control_group))
    treatment_group = set([1, 2, 3, 4])
    self.assertFalse(
        mm.design_within_constraints(treatment_group, control_group))

  def testNumberControlGeosBetweenBounds(self):
    """Treatment group size is not within (2,3)."""
    self.par.control_geos_range = (2, 3)
    mm = TBRMatchedMarkets(self.data, self.par)
    treatment_group = set([0])
    control_group = set([1])
    self.assertFalse(
        mm.design_within_constraints(treatment_group, control_group))
    control_group = set([1, 2, 3, 4])
    self.assertFalse(
        mm.design_within_constraints(treatment_group, control_group))


class GreedySearch(unittest.TestCase):
  """Test method 'greedy_search'."""

  def setUp(self):
    super().setUp()
    response_geo1 = [
        255.2, 165.8, 186.0, 160.9, 218.1, 165.7, 212.9, 207.7, 224.4, 205.0,
        247.2, 145.1, 191.1, 173.6
    ]
    response_geo2 = [
        132.5, 87.8, 89.4, 78.5, 117.3, 54.0, 134.9, 84.8, 106.4, 95.0, 129.2,
        58.8, 93.6, 92.3
    ]
    response_geo3 = [10 + 1 * i for i in range(len(response_geo1))]
    dates = pd.date_range('2020-03-01', periods=len(response_geo1))
    df = pd.DataFrame({
        'date': list(dates) * 3,
        'geo': sorted([1, 2, 3] * len(dates)),
        'response': response_geo1 + response_geo2 + response_geo3
    })
    self.dataframe = df
    self.data = TBRMMData(df, 'response')
    self.par = TBRMMDesignParameters(
        n_test=7, iroas=3.0)
    self.default_geo_eligibility_data = self.data.geo_eligibility.data

  def testGreedySearch(self):
    """Search finds a design with 1 and 2 geos in treatment."""
    mm = TBRMatchedMarkets(self.data, self.par)
    designs = mm.greedy_search()
    treatment_groups = [{'1'}, {'1', '3'}]
    control_groups = [{'2'}, {'2'}]
    treatment_index = [[0], [0, 2]]
    control_index = [[1], [1]]
    for ind in range(len(designs)):
      diag = TBRMMDiagnostics(
          self.data.aggregate_time_series(set(treatment_index[ind])), self.par)
      diag.x = self.data.aggregate_time_series(set(control_index[ind]))
      corr = diag.corr
      required_impact = diag.estimate_required_impact(corr)
      self.assertSetEqual(designs[ind].treatment_geos, treatment_groups[ind])
      self.assertSetEqual(designs[ind].control_geos, control_groups[ind])
      self.assertTupleEqual(designs[ind].score.score,
                            (1, 1, 1, 1, round(corr, 2), 1 / required_impact))
      self.assertEqual(designs[ind].score.diag.corr, designs[ind].diag.corr)
      self.assertEqual(designs[ind].score.score.corr,
                       round(designs[ind].diag.corr, 2))

  def testGreedySearchScoreWithIroas(self):
    """Tests the optimal design is found based on correlation and iROAS."""
    self.par.budget_range = (0.01, 1000)
    mm = TBRMatchedMarkets(self.data, self.par)
    designs = mm.exhaustive_search()

    tmp_diag = TBRMMDiagnostics(
        self.data.aggregate_time_series(set([0])), self.par)
    tmp_diag.x = self.data.aggregate_time_series(set([1]))
    corr = tmp_diag.corr
    required_impact = tmp_diag.required_impact
    iroas = required_impact / self.par.budget_range[1]

    self.assertSetEqual(designs[0].treatment_geos, {'1'})
    self.assertSetEqual(designs[0].control_geos, {'2'})
    self.assertTupleEqual(designs[0].score.score,
                          (1, 1, 1, 1, round(corr, 2), 1 / iroas))

  def testGreedySearchFailsToFindDesignWithinConstraints(self):
    """Search does not find a valid design since correlation is too low."""
    self.par.min_corr = 0.9
    mm = TBRMatchedMarkets(self.data, self.par)

    designs = mm.greedy_search()
    treatment_groups = [{'1'}, {'1', '3'}]
    control_groups = [{'2'}, {'2'}]
    treatment_index = [[0], [0, 2]]
    control_index = [[1], [1]]
    for ind in range(len(designs)):
      diag = TBRMMDiagnostics(
          self.data.aggregate_time_series(set(treatment_index[ind])), self.par)
      diag.x = self.data.aggregate_time_series(set(control_index[ind]))
      corr = diag.corr
      required_impact = diag.estimate_required_impact(corr)
      self.assertSetEqual(designs[ind].treatment_geos, treatment_groups[ind])
      self.assertSetEqual(designs[ind].control_geos, control_groups[ind])
      self.assertTupleEqual(designs[ind].score.score,
                            (0, 1, 1, 1, round(corr, 2), 1 / required_impact))
      self.assertEqual(designs[ind].score.diag.corr, designs[ind].diag.corr)
      self.assertEqual(designs[ind].score.score.corr,
                       round(designs[ind].diag.corr, 2))

  def testGreedySearchFixedTreatmentGeo(self):
    """Search finds a design with geos fixed treatment group."""
    df_geo_elig = self.default_geo_eligibility_data
    df_geo_elig.loc['1'] = [0, 1, 0]  # geo 1 (index 0) is fixed to Treatment.
    geo_elig = geoeligibility.GeoEligibility(df_geo_elig)
    data = TBRMMData(self.dataframe, 'response', geo_elig)
    self.par.treatment_geos_range = (1, 1)
    mm = TBRMatchedMarkets(data, self.par)
    designs = mm.greedy_search()
    diag = TBRMMDiagnostics(data.aggregate_time_series(set([0])), self.par)
    diag.x = data.aggregate_time_series(set([1]))
    corr = diag.corr
    required_impact = diag.estimate_required_impact(corr)
    self.assertTrue(len(designs) == 1)  # pylint: disable=g-generic-assert
    self.assertSetEqual(designs[0].treatment_geos, {'1'})
    self.assertSetEqual(designs[0].control_geos, {'2'})
    self.assertTupleEqual(designs[0].score.score,
                          (1, 1, 1, 1, round(corr, 2), 1 / required_impact))
    self.assertEqual(designs[0].score.diag.corr, designs[0].diag.corr)
    self.assertEqual(designs[0].score.score.corr, round(designs[0].diag.corr,
                                                        2))

  def testGreedySearchFixedTreatmentGeoFail(self):
    """Search fails with fixed treatment group as all design do not pass dwtest."""
    df_geo_elig = self.default_geo_eligibility_data
    df_geo_elig.loc['2'] = [0, 1, 0]  # geo 2 (index 1) is fixed to Treatment.
    geo_elig = geoeligibility.GeoEligibility(df_geo_elig)
    data = TBRMMData(self.dataframe, 'response', geo_elig)
    self.par.treatment_geos_range = (1, 1)
    mm = TBRMatchedMarkets(data, self.par)
    designs = mm.greedy_search()
    diag = TBRMMDiagnostics(data.aggregate_time_series(set([1])), self.par)
    diag.x = data.aggregate_time_series(set([0]))
    corr = diag.corr
    required_impact = diag.estimate_required_impact(corr)
    self.assertTrue(len(designs) == 1)  # pylint: disable=g-generic-assert
    self.assertSetEqual(designs[0].treatment_geos, {'2'})
    self.assertSetEqual(designs[0].control_geos, {'1'})
    self.assertTupleEqual(designs[0].score.score,
                          (1, 1, 1, 0, round(corr, 2), 1 / required_impact))
    self.assertEqual(designs[0].score.diag.corr, designs[0].diag.corr)
    self.assertEqual(designs[0].score.score.corr, round(designs[0].diag.corr,
                                                        2))

  def testGreedySearchTreatmentSizeRange(self):
    """Search succeed with constrained treatment group size."""
    # limit the search to treatment groups of size 2
    self.par.treatment_geos_range = (2, 2)
    mm = TBRMatchedMarkets(self.data, self.par)
    designs = mm.greedy_search()
    diag = TBRMMDiagnostics(
        self.data.aggregate_time_series(set([0, 2])), self.par)
    diag.x = self.data.aggregate_time_series(set([1]))
    corr = diag.corr
    required_impact = diag.estimate_required_impact(corr)
    self.assertTrue(len(designs) == 1)  # pylint: disable=g-generic-assert
    self.assertSetEqual(designs[0].treatment_geos, {'1', '3'})
    self.assertSetEqual(designs[0].control_geos, {'2'})
    self.assertTupleEqual(designs[0].score.score,
                          (1, 1, 1, 1, round(corr, 2), 1 / required_impact))
    self.assertEqual(designs[0].score.diag.corr, designs[0].diag.corr)
    self.assertEqual(designs[0].score.score.corr, round(designs[0].diag.corr,
                                                        2))

  def testGreedySearchControlSizeRange(self):
    """Search succeed with constrained treatment group size."""
    # limit the search to control groups of size 2
    self.par.control_geos_range = (2, 2)
    mm = TBRMatchedMarkets(self.data, self.par)
    designs = mm.greedy_search()
    diag = TBRMMDiagnostics(
        self.data.aggregate_time_series(set([0])), self.par)
    diag.x = self.data.aggregate_time_series(set([1, 2]))
    corr = diag.corr
    required_impact = diag.estimate_required_impact(corr)
    self.assertTrue(len(designs) == 1)  # pylint: disable=g-generic-assert
    self.assertSetEqual(designs[0].treatment_geos, {'1'})
    self.assertSetEqual(designs[0].control_geos, {'2', '3'})
    self.assertTupleEqual(designs[0].score.score,
                          (1, 1, 1, 1, round(corr, 2), 1 / required_impact))
    self.assertEqual(designs[0].score.diag.corr, designs[0].diag.corr)
    self.assertEqual(designs[0].score.score.corr, round(designs[0].diag.corr,
                                                        2))

  def testGreedySearchTreatmentGeoShare(self):
    """Search succeed with constrained treatment group revenue share."""
    # limit the search to treatment groups with a revenue share >0.65
    # note that the largest single geos only has a share of 0.635031, so we
    # need at least two geos in treatment
    self.par.treatment_share_range = (0.65, 0.99)
    mm = TBRMatchedMarkets(self.data, self.par)
    designs = mm.greedy_search()
    diag = TBRMMDiagnostics(
        self.data.aggregate_time_series(set([0, 2])), self.par)
    diag.x = self.data.aggregate_time_series(set([1]))
    corr = diag.corr
    required_impact = diag.estimate_required_impact(corr)
    self.assertTrue(len(designs) == 1)  # pylint: disable=g-generic-assert
    self.assertSetEqual(designs[0].treatment_geos, {'1', '3'})
    self.assertSetEqual(designs[0].control_geos, {'2'})
    self.assertTupleEqual(designs[0].score.score,
                          (1, 1, 1, 1, round(corr, 2), 1 / required_impact))
    self.assertEqual(designs[0].score.diag.corr, designs[0].diag.corr)
    self.assertEqual(designs[0].score.score.corr, round(designs[0].diag.corr,
                                                        2))

  def testGreedySearchVolumeRatioTolerance(self):
    """Search succeed with constrained volume ratio tolerance."""
    # limit the search to groups with a volume ratio >0.5.
    # note that the best design (Treatment=geo1, Control=geo2) has a ratio of
    # 0.4909, so we need at least two geos in control
    self.par.volume_ratio_tolerance = 1
    mm = TBRMatchedMarkets(self.data, self.par)
    designs = mm.greedy_search()
    diag = TBRMMDiagnostics(self.data.aggregate_time_series(set([0])), self.par)
    diag.x = self.data.aggregate_time_series(set([1, 2]))
    corr = diag.corr
    required_impact = diag.estimate_required_impact(corr)
    self.assertTrue(len(designs) == 1)  # pylint: disable=g-generic-assert
    self.assertSetEqual(designs[0].treatment_geos, {'1'})
    self.assertSetEqual(designs[0].control_geos, {'2', '3'})
    self.assertTupleEqual(designs[0].score.score,
                          (1, 1, 1, 1, round(corr, 2), 1 / required_impact))
    self.assertEqual(designs[0].score.diag.corr, designs[0].diag.corr)
    self.assertEqual(designs[0].score.score.corr, round(designs[0].diag.corr,
                                                        2))

  def testGreedySearchGeoRatioTolerance(self):
    """Search succeed with constrained geo ratio tolerance."""
    # limit the search to groups with a geo ratio >0.52 and <1.9. The constraint
    # would raise an error (division by zero) since the search starts with
    # all geos in control and 0 geos in treatment
    self.par.geo_ratio_tolerance = 0.9
    mm = TBRMatchedMarkets(self.data, self.par)
    designs = mm.greedy_search()
    diag = TBRMMDiagnostics(self.data.aggregate_time_series(set([0])), self.par)
    diag.x = self.data.aggregate_time_series(set([1]))
    corr = diag.corr
    required_impact = diag.estimate_required_impact(corr)
    self.assertTrue(len(designs) == 1)  # pylint: disable=g-generic-assert
    self.assertSetEqual(designs[0].treatment_geos, {'1'})
    self.assertSetEqual(designs[0].control_geos, {'2'})
    self.assertTupleEqual(designs[0].score.score,
                          (1, 1, 1, 1, round(corr, 2), 1 / required_impact))
    self.assertEqual(designs[0].score.diag.corr, designs[0].diag.corr)
    self.assertEqual(designs[0].score.score.corr, round(designs[0].diag.corr,
                                                        2))


if __name__ == '__main__':
  unittest.main()
