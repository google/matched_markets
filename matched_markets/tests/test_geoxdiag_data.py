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
"""Test whether the geoxdiag data set is available.
"""

import os

from absl import flags
from matched_markets.examples import geoxdiag_data
import numpy as np

import unittest




class GeoxdiagDataTest(unittest.TestCase):

  def testDataAvailable(self):
    """Checks whether the geoxdiag example data is available."""

    # Read the copy of the data included in the test environment.
    test_base_dir = ""
    csv_dir = os.path.join(test_base_dir, geoxdiag_data.GEOX_CSV_DIR)

    # Read in the data.
    data_frame = geoxdiag_data.read_data(csv_dir)

    # Confirm the dataframe has the requisite shape.
    self.assertEqual(data_frame.shape, (1000, 5))

    # Check the total sum of the response column.
    self.assertEqual(np.round(np.sum(data_frame.response), 7), 104752.9754437)


if __name__ == "__main__":
  unittest.main()
