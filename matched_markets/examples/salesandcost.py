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
"""Functions to generate the salesandcost DataFrame.

Helper functions to obtain and parse the 'salesandcost' geo experiment
dataset and create a pandas.DataFrame suitable for further processing in the
pipeline.
"""

import datetime
import os
import numpy as np
import pandas as pd




def example_data(base_dir):
  """Returns example data: salesandcost, geoassignment and experiment dates."""

  # Form paths to the example data.
  snc_path = os.path.join(base_dir, 'salesandcost.csv')
  ga_path = os.path.join(base_dir, 'geoassignment.csv')

  # Read in the csv files for the experiment.
  with open(snc_path) as csvfile:
    snc = pd.read_csv(csvfile, parse_dates=['date'])
  with open(ga_path) as csvfile:
    ga = pd.read_csv(csvfile)

  # Define dates for the experiment.
  exdates = ['2015-01-05', '2015-02-16', '2015-03-15']
  return (snc, ga, exdates)


def _get_periods(dates, start_dates):
  period = -np.ones(dates.shape[0], dtype=np.int32)
  sd = [datetime.datetime.strptime(s, '%Y-%m-%d') for s in start_dates]
  sd[-1] += datetime.timedelta(days=1)
  for dt in sd:
    period += 1*(dates >= dt)
  return period


def format_example_data(data, geoassign, exdates):
  """Summons and formats a complete dataset."""
  better_data = data.set_index('geo').join(geoassign.set_index('geo'))
  better_data['period'] = _get_periods(better_data.date, exdates)
  return better_data


def example_data_formatted(srcdir):
  """Summon the data."""
  data, geoassign, exdates = example_data(srcdir)
  return format_example_data(data, geoassign, exdates)

