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
"""Test whether the sales and cost example data is available.
"""

import os

from absl import flags
from matched_markets.examples import salesandcost

import unittest




class SalesAndCostExampleDataTest(unittest.TestCase):

  def testDataAvailable(self):
    """Checks whether the salesandcost example data is available."""

    # Read the copy of the data included in the test environment.
    test_base_dir = ""
    data_dir = "matched_markets/csv/"
    csv_dir = os.path.join(test_base_dir, data_dir)

    # Read in the data, save its shape.
    snc_data_frame = salesandcost.example_data_formatted(csv_dir)
    snc_shape = snc_data_frame.shape

    # Confirm the dataframe has the requisite shape.
    # Number of rows should be 9225 if we are to later reproduce output from R.
    self.assertEqual(snc_shape, (9225, 5))


if __name__ == "__main__":
  unittest.main()
