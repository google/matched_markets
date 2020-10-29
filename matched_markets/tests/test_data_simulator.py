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
"""Test the data generator example.
"""

from absl import flags
from matched_markets.examples.data_simulator import DataSimulator
import numpy as np

import unittest





class DataSimulatorTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    # Experimental design.
    self.n_control = 50
    self.n_treat = 50
    self.time_pre = 100
    self.time_test = 100

    # Linear params.
    self.hetresp = 1.0
    self.hetcost = 0.0
    self.beta = 0.0

    # Noise params.
    self.hetsked = 0.0
    self.sig_resp = 0.0
    self.sig_cost = 0.0

    # Column names.
    self.df_keys = {
        'key_response': 'sales',
        'key_cost': 'cost',
        'key_group': 'geo.group',
        'key_period': 'period',
        'key_geo': 'geo',
        'key_date': 'date'
        }

  def testSampleRows(self):
    """Checks if .sample() has the correct number of rows."""

    # Make simulator.
    simulator = DataSimulator(self.n_control,
                              self.n_treat,
                              self.time_pre,
                              self.time_test,
                              self.hetresp,
                              self.hetcost,
                              self.beta,
                              self.hetsked,
                              self.sig_resp,
                              self.sig_cost,
                              **self.df_keys)

    # Simulate data.
    fake_data = simulator.sample()

    # Derived constants.
    time_total = self.time_pre + self.time_test
    n_total = self.n_treat + self.n_control
    col_len = time_total * n_total

    self.assertEqual(len(fake_data.index), col_len)  # pylint: disable=g-generic-assert

  def testSampleColumns(self):
    """Check whether .sample() returns an appropriate `pd.DataFrame`."""

    # Make simulator.
    simulator = DataSimulator(self.n_control,
                              self.n_treat,
                              self.time_pre,
                              self.time_test,
                              self.hetresp,
                              self.hetcost,
                              self.beta,
                              self.hetsked,
                              self.sig_resp,
                              self.sig_cost,
                              **self.df_keys)

    # Simulate data.
    fake_data = simulator.sample()

    column_keys = ['sales', 'cost', 'geo.group', 'period', 'date', 'size']
    self.assertCountEqual(fake_data.columns, column_keys)

  def testSalesColumn(self):
    """Check whether .sample() returns an appropriate `pd.DataFrame`."""

    # Make simulator.
    simulator = DataSimulator(self.n_control,
                              self.n_treat,
                              self.time_pre,
                              self.time_test,
                              self.hetresp,
                              self.hetcost,
                              self.beta,
                              self.hetsked,
                              self.sig_resp,
                              self.sig_cost,
                              **self.df_keys)

    # Simulate data.
    fake_data = simulator.sample()
    total_sales = fake_data.sales.sum()

    # Derive the true total sales.
    time_total = self.time_pre + self.time_test
    sales_treat = self.n_treat * (self.n_treat + 1) / 2.0
    sales_control = self.n_control * (self.n_control + 1) / 2.0
    sales_true = (sales_treat + sales_control) * time_total

    self.assertAlmostEqual(sales_true, total_sales)

  def testFixingSeedResultsInSameData(self):
    """Checks simulators with the same random seed produce the same samples."""
    # Fix a seed for the random number generators.
    seed = 1234
    # Parameters ensuring the data contains noise.
    sig_resp = 1.0
    sig_cost = 1.0

    # Make simulator.
    simulator1 = DataSimulator(self.n_control,
                               self.n_treat,
                               self.time_pre,
                               self.time_test,
                               self.hetresp,
                               self.hetcost,
                               self.beta,
                               self.hetsked,
                               sig_resp,
                               sig_cost,
                               seed=seed,
                               **self.df_keys)
    # Simulate data, calculate a characteristic number.
    fake_data1 = simulator1.sample()
    sum1 = np.sum(fake_data1.values)

    # Make identical simulator.
    simulator2 = DataSimulator(self.n_control,
                               self.n_treat,
                               self.time_pre,
                               self.time_test,
                               self.hetresp,
                               self.hetcost,
                               self.beta,
                               self.hetsked,
                               sig_resp,
                               sig_cost,
                               seed=seed,
                               **self.df_keys)
    # Simulate (hopefully) identical data, calculate a characteristic number.
    fake_data2 = simulator2.sample()
    sum2 = np.sum(fake_data2.values)

    self.assertEqual(sum1, sum2)


if __name__ == '__main__':
  unittest.main()
