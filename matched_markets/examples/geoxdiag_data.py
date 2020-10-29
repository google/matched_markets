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
"""Functions to generate geoxdiag.DataFrame.

Helper functions to obtain and parse the 'geoxdiag' dataset
and create a pandas.DataFrame suitable for further processing.
"""

import os
import pandas as pd


GEOX_CSV_DIR = 'matched_markets/csv/'


def read_data(base_dir):
  """Returns example data 'geoxdiag_data'."""

  # Form paths to the example data.
  file_path = os.path.join(base_dir, 'geoxdiag_data.csv')

  # Read in the csv file.
  with open(file_path) as csvfile:
    geoxdiag = pd.read_csv(csvfile, parse_dates=['date'])

  geoxdiag.set_index('geo')
  return geoxdiag
