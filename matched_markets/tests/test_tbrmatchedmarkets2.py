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


class ExhaustiveSearch(unittest.TestCase):
  """Test class TBRMatchedMarkets."""

  def setUp(self):
    super().setUp()
    # create a dataframe where geo 1 and geo 2 are highly correlated, while geo
    # 3 is almost independent of geo 1 and 2.
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
        'geo': sorted(['1', '2', '3'] * len(dates)),
        'response': response_geo1 + response_geo2 + response_geo3
    })

    par = TBRMMDesignParameters(n_test=7, iroas=3.0)

    self.iroas = 3
    self.dates = dates
    self.par = par
    self.data = TBRMMData(df, 'response')
    self.default_geo_eligibility_data = self.data.geo_eligibility.data
    self.df = df
    self.response_geo1 = response_geo1
    self.response_geo2 = response_geo2
    self.response_geo3 = response_geo3

  def testExhaustiveSearchFindsOptimalDesign(self):
    """Tests the optimal design is found based on correlation only."""
    mm = TBRMatchedMarkets(self.data, self.par)
    designs = mm.exhaustive_search()

    diag = TBRMMDiagnostics(self.data.aggregate_time_series(set([0])), self.par)
    diag.x = self.data.aggregate_time_series(set([1]))
    corr = diag.corr
    required_impact = diag.required_impact
    self.assertSetEqual(designs[0].treatment_geos, {'1'})
    self.assertSetEqual(designs[0].control_geos, {'2'})
    self.assertTupleEqual(designs[0].score.score,
                          (1, 1, 1, 1, round(corr, 2), 1 / required_impact))

  def testExhaustiveSearchMultipleDesigns(self):
    """Tests the best 3 designs are found based on correlation only."""
    self.par.n_designs = 3
    mm = TBRMatchedMarkets(self.data, self.par)
    designs = mm.exhaustive_search()
    treatment_groups = [{'1'}, {'1', '3'}, {'1'}]
    control_groups = [{'2'}, {'2'}, {'2', '3'}]
    treatment_index = [[0], [0, 2], [0]]
    control_index = [[1], [1], [1, 2]]
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

  def testExhaustiveSearchScoreWithIroas(self):
    """Tests the optimal design is found based on correlation and iROAS."""
    self.par.budget_range = (0.01, 1000)
    mm = TBRMatchedMarkets(self.data, self.par)
    designs = mm.exhaustive_search()

    tmp_diag = TBRMMDiagnostics(self.response_geo1, self.par)
    tmp_diag.x = self.response_geo2
    corr = tmp_diag.corr
    required_impact = tmp_diag.required_impact
    iroas = required_impact / self.par.budget_range[1]

    self.assertSetEqual(designs[0].treatment_geos, {'1'})
    self.assertSetEqual(designs[0].control_geos, {'2'})
    self.assertTupleEqual(designs[0].score.score,
                          (1, 1, 1, 1, round(corr, 2), 1 / iroas))

  def testExhaustiveSearchWithBudgetConstraint(self):
    """Tests search with maximum budget constraint."""
    self.par.n_designs = 8
    self.par.budget_range = (0.01, 1000)
    mm = TBRMatchedMarkets(self.data, self.par)
    designs = mm.exhaustive_search()
    # the optimal design without budget constraints is:
    # treatment = {1} and control = {2}
    optimal_design = designs[0]
    # the suboptimal design is the fourth best design, which has a lower
    # required impact/budget (but lower correlation and score)
    suboptimal_design = designs[3]
    max_budget = optimal_design.diag.required_impact / self.iroas
    # set max budget to be slightly lower than the required budget for the
    # optimal design
    self.par.budget_range = (0.01, 0.999 * max_budget)

    # the new optimal design is the one that was suboptimal before (the eight
    # best design, as all the other do not satisfy the budget constraint)
    mm = TBRMatchedMarkets(self.data, self.par)
    updated_designs = mm.exhaustive_search()
    self.assertSetEqual(updated_designs[0].treatment_geos,
                        suboptimal_design.treatment_geos)
    self.assertSetEqual(updated_designs[0].control_geos,
                        suboptimal_design.control_geos)
    self.assertAlmostEqual(updated_designs[0].diag.corr,
                           suboptimal_design.diag.corr)

  def testExhaustiveSearchGeoEligibility(self):
    """Tests search with geo eligibility constraints."""
    # without constraints, the optimal design would be treatment = {'1'} and
    # control = {'2'}, see testExhaustiveSearchFindsOptimalDesign
    df_geo_elig = self.default_geo_eligibility_data
    df_geo_elig.loc['1'] = [1, 1, 0]  # Cannot exclude geo 1.
    df_geo_elig.loc['2'] = [1, 0, 0]  # Cannot exclude geo 2.
    df_geo_elig.loc['3'] = [0, 1, 0]  # Cannot exclude geo 3.
    # given the fact that geo 1 and 2 are correlated, and geo 3 is approx.
    # independent of both, the optimal design will have treatment = {'1','3'}
    # and control = {'2'}. Since geos '1' and '2' should be in different groups
    # to achieve high correlation and geo '2' is fixed to control.
    geo_elig = geoeligibility.GeoEligibility(df_geo_elig)
    data = TBRMMData(self.df, 'response', geo_elig)
    mm = TBRMatchedMarkets(data, self.par)
    designs = mm.exhaustive_search()

    diag = TBRMMDiagnostics(
        data.aggregate_time_series(set([0, 2])), self.par)
    diag.x = data.aggregate_time_series(set([1]))
    corr = diag.corr
    required_impact = diag.required_impact
    self.assertSetEqual(designs[0].treatment_geos, {'1', '3'})
    self.assertSetEqual(designs[0].control_geos, {'2'})
    self.assertTupleEqual(
        designs[0].score.score,
        (1, 1, 1, 1, round(corr, 2), 1/required_impact))

  def testExhaustiveSearchNoFeasibleDesign(self):
    """Tests search with no feasible design."""
    # by removing geo 2, we are left with only geos 1 and 3 which are not
    # correlated. So, the exhaustive search will return the only possible
    # design, which is not feasible as it does not pass the correlation test
    df = self.df[self.df['geo'].isin(['1', '3'])]
    data = TBRMMData(df, 'response')
    mm = TBRMatchedMarkets(data, self.par)
    designs = mm.exhaustive_search()

    diag = TBRMMDiagnostics(data.aggregate_time_series(set([0])), self.par)
    diag.x = data.aggregate_time_series(set([1]))
    corr = diag.corr
    required_impact = diag.required_impact
    self.assertTupleEqual(
        designs[0].score.score,
        (0, 1, 1, 1, round(corr, 2), 1/required_impact))

if __name__ == '__main__':
  unittest.main()
