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
"""Generic utility functions."""

import random
import re
from typing import List

import altair as alt
from matched_markets.methodology import common_classes
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

TimeWindow = common_classes.TimeWindow


def kwarg_subdict(prefix, **kwargs):
  """Extract sub dict of `kwargs` prefixed by `prefix`, stripping `prefix`.

  E.g.
    kwarg_subdict('a', a_x=1, a_y=2, b_z=3)  # returns {x:1, y:2}

  Args:
    prefix: a string specifying the prefix to search for in the kwarg names.
    **kwargs: any number of named arguments.

  Returns:
    A subset of the supplied kwargs in the form of a dictionary.
  """

  # Define a regex to match the prefix.
  rgx = re.compile(r'%s(.*)' % prefix)

  # Extract the kwargs which match the regex.
  sub_kwargs = [k for k in kwargs.keys() if rgx.search(k)]

  # Return the matched kwargs, stripping off prefix.
  return {rgx.match(k).group(1): kwargs[k] for k in sub_kwargs}


def float_order(x):
  """Calculates the order of magnitude of x."""
  abs_x = np.abs(x)
  if abs_x > 0:
    return np.floor(np.log10(abs_x))
  else:
    return -np.inf


def randomize_strata(n_items, group_ids, seed=None):
  """Perform stratified randomization.

  Maps a number of items into group ids by dividing the items into strata of
  length equal to the number of groups, then assigning the group ids randomly
  within each strata. If the number of items do not divide the number of groups
  equally, the remaining items are assigned to groups randomly (with equal
  probability of assignment in each group).

  Example: randomize_strata(12, [1, 2, 3]) yields a list of 12 group ids, each
  1, 2, or 3. There are (3!)^4 = 6^4 = 1296 possible outcomes as there are 12 /
  3 = 4 strata, and within each strata there are 3! = 6 different ways to assign
  the items into groups. Therefore, the first 3 items of the list (that is,
  positions 0, 1, and 2) can be mapped only to the 3!=6 different mappings:
  [1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2] or [3, 2, 1].

  Args:
    n_items: (int) Number of items to assign to groups.
    group_ids: A list of group ids that are typically integers or strings, but
      can be of any type.
    seed: (int) Random seed, applied to a local instance of class random.Random;
      if not specified, the global random instance is used instead.

  Returns:
    A list of length n_items, consisting of the group ids whose positions in the
    list correspond to the items.
  """

  if seed is None:
    random_sampler = random.sample
  else:
    random_sampler = random.Random(seed).sample

  n_groups = len(group_ids)
  n_strata = n_items // n_groups
  groups = []
  for _ in range(n_strata):
    groups.extend(random_sampler(group_ids, n_groups))
  remaining = n_items - len(groups)
  if remaining > 0:
    groups.extend(random_sampler(group_ids, remaining))
  return groups


def brownian_bridge_bounds(n, sd_bound_multiplier):
  """Obtain bounds of cumulative residuals from Brownian Bridge process.

  The bounds for variance are proportional to t * (n - t) / n for residuals
  t = 1 .. n (for residual n, the bound is zero as the sum of residuals is
  always zero). This function returns the bounds for the standard deviation.

  Args:
    n: (int >= 1) Length of the time series of the cumulative residuals
     following a Brownian Bridge process.
   sd_bound_multiplier: (numeric > 0) Multiplier for bounds on cumulative
     standardized residuals.

  Returns:
    A list of length n, of the Brownian Bridge process bounds in absolute
    values (if n == 1, returns [0]).
  """
  if n < 1:
    raise ValueError('n must be >= 1')

  if sd_bound_multiplier <= 0:
    raise ValueError('sd_bound_multiplier must be > 0')

  n_range = np.arange(1, n + 1)  # 1, ..., n.
  bounds = sd_bound_multiplier * np.sqrt(n_range * (1.0 - n_range / float(n)))
  return bounds.tolist()


def credible_interval(simulations, level):
  """Construct the (1 - level, 0.5, level) central interval from simulations.

  Args:
    simulations: numeric arraylike. The simulations.
    level: float in (0, 1). The mass of the desired interval.

  Returns:
    An np.array representing the central credible interval at the given level.

  Raises:
    ValueError: if the requested level is too large (< 1/ len(sims)).
  """
  alpha = (1 - level)/2.0
  nvals = len(simulations)
  if alpha < 1.0/nvals:
    raise ValueError('Too few values to provide requested quantiles.')
  sims_sort = np.sort(np.copy(simulations))
  frac = nvals * np.array([alpha, 0.5, 1.0 - alpha]) - 1.0
  low = np.floor(frac).astype(np.int64)
  return sims_sort[low] + (frac - low)*(sims_sort[low + 1] - sims_sort[low])


def find_days_to_exclude(
    dates_to_exclude: List[str]) -> List[TimeWindow]:
  """Returns a list of time windows to exclude from a list of days and periods.

  Args:
    dates_to_exclude: a List of strings with format indicating a single day as
    '2020/01/01' (YYYY/MM/DD) or an entire time period as
    '2020/01/01 - 2020/02/01' (indicating start and end date of the time period)

  Returns:
    days_exclude: a List of TimeWindows obtained from the list in input.
  """
  days_exclude = []
  for x in dates_to_exclude:
    tmp = x.split('-')
    if len(tmp) == 1:
      try:
        days_exclude.append(
            TimeWindow(pd.Timestamp(tmp[0]), pd.Timestamp(tmp[0])))
      except ValueError:
        raise ValueError(f'Cannot convert the string {tmp[0]} to a valid date.')
    elif len(tmp) == 2:
      try:
        days_exclude.append(
            TimeWindow(pd.Timestamp(tmp[0]), pd.Timestamp(tmp[1])))
      except ValueError:
        raise ValueError(
            f'Cannot convert the strings in {tmp} to a valid date.')
    else:
      raise ValueError(f'The input {tmp} cannot be interpreted as a single' +
                       ' day or a time window')

  return days_exclude


def expand_time_windows(periods: List[TimeWindow]) -> List[pd.Timestamp]:
  """Return a list of days to exclude from a list of TimeWindows.

  Args:
    periods: List of time windows (first day, last day).

  Returns:
    days_exclude: a List of obtained by expanding the list in input.
  """
  days_exclude = []
  for window in periods:
    days_exclude += pd.date_range(
        window.first_day, window.last_day, freq='D').to_list()

  return list(set(days_exclude))


def human_readable_number(number: float) -> str:
  """Print a large number in a readable format.

  Return a readable format for a number, e.g. 123 milions becomes 123M.

  Args:
    number: a float to be printed in human readable format.

  Returns:
    readable_number: a string containing the formatted number.
  """
  number = float('{:.3g}'.format(number))
  magnitude = 0
  while abs(number) >= 1000 and magnitude < 4:
    magnitude += 1
    number /= 1000.0
  readable_number = '{}{}'.format('{:f}'.format(number).rstrip('0').rstrip('.'),
                                  ['', 'K', 'M', 'B', 'tn'][magnitude])
  return readable_number


def default_geo_assignment(geo_level_time_series: pd.DataFrame,
                           geo_eligibility: pd.DataFrame) -> pd.DataFrame:
  """Set the default assignment eligibility for missing geos.

  Geos missing in the geo assignment table but present in the geo level time
  series are considered unconstrained. So, they can be assigned to either
  treatment, control, or excluded.

  Args:
    geo_level_time_series: table containing the response time series at geo
      level.
    geo_eligibility: table containing the possible assignments for some of the
      geos.

  Returns:
    a table containing the possible assignments for all geos in
    geo_level_time_series.
  """

  if not is_numeric_dtype(geo_level_time_series['geo']):
    geo_level_time_series['geo'] = pd.to_numeric(geo_level_time_series['geo'])

  if not is_numeric_dtype(geo_eligibility['geo']):
    geo_eligibility['geo'] = pd.to_numeric(geo_eligibility['geo'])

  missing_geos = list(
      set(geo_level_time_series['geo']) - set(geo_eligibility['geo']))

  return geo_eligibility.append(
      pd.DataFrame({
          'geo': missing_geos,
          'control': 1,
          'treatment': 1,
          'exclude': 1
      })).sort_values(by='geo').reset_index(drop=True)


def plot_iroas_over_time(iroas_df: pd.DataFrame, experiment_dates: pd.DataFrame,
                         cooldown_date: pd.DataFrame):
  """Returns a chart of the iROAS estimate over time with confidence bands.

  This function provides a visualization of the evolution of the iROAS estimate
  over the duration of the experiment and cooldown, together with confidence
  bands.

  Args:
    iroas_df: a dataframe with columns: date, lower, mean, upper
    experiment_dates: dataframe with columns (date, color) which contains two
      dates for each period (start, end), and the column color is the label
      used in the chart to refer to the corresponding period, e.g. "Experiment
      period" or "Pretes period".
    cooldown_date: dataframe with column (date, color) with only one entry,
      where date indicates the last day in the cooldown period, and color is the
      label used in the plot legend, e.g. "End of cooldown period".

  Returns:
    iroas_chart: Chart containing the plot.
  """
  iroas_base = alt.Chart(iroas_df).mark_line().encode(
      x=alt.X('date:T', axis=alt.Axis(title='', format=('%b %e'))))

  iroas_selection = alt.selection_single(
      fields=['date'],
      nearest=True,
      on='mouseover',
      empty='none',
      clear='mouseout')

  iroas_lines = iroas_base.mark_line().encode(
      y=alt.Y('mean:Q', axis=alt.Axis(title=' ', format='.3')))

  iroas_points = iroas_lines.mark_point().transform_filter(iroas_selection)

  iroas_rule1 = iroas_base.mark_rule().encode(
      tooltip=['date:T', 'mean:Q', 'lower:Q', 'upper:Q'])

  iroas_rule = iroas_rule1.encode(
      opacity=alt.condition(iroas_selection, alt.value(0.3), alt.value(
          0))).add_selection(iroas_selection)

  iroas_ci_bands_rule = alt.Chart(iroas_df).mark_area(color='gray').encode(
      alt.X('date:T'), y='lower:Q', y2='upper:Q', opacity=alt.value(0.5))

  date_rule = alt.Chart(
      experiment_dates[experiment_dates['color'] ==
                       'Experiment period']).mark_rule(strokeWidth=2).encode(
                           x='date:T',
                           color=alt.Color(
                               'color',
                               scale=alt.Scale(
                                   domain=[
                                       'Experiment period',
                                       'End of cooldown period',
                                       'iROAS estimate'
                                   ],
                                   range=['black', 'black', '#1f77b4'])))
  cooldown_date_rule = alt.Chart(cooldown_date).mark_rule(
      strokeWidth=2, strokeDash=[5, 2], color='black').encode(
          x='date:T', color='color:N')
  # Compile chart
  iroas_chart = alt.layer(iroas_lines, iroas_rule, iroas_points, date_rule,
                          cooldown_date_rule, iroas_ci_bands_rule)

  return iroas_chart


def infer_frequency(data: pd.DataFrame, date_index: str,
                    series_index: str) -> str:
  """Infers frequency of data from pd.DataFrame with multiple indices.

  Infers frequency of data from pd.DataFrame with two indices, one for the slice
  name and one for the date-time.
  Example:
    df = pd.Dataframe{'date': [2020-10-10, 2020-10-11], 'geo': [1, 1],
      'response': [10, 20]}
    infer_frequency(df, 'date', 'geo')

  Args:
    data: a pd.DataFrame for which frequency needs to be inferred.
    date_index: string containing the name of the time index.
    series_index: string containing the name of the series index.

  Returns:
    A str, either 'D' or 'W' indicating the most likely frequency inferred
    from the data.

  Raises:
    ValueError: if it is not possible to infer frequency of sampling from the
      provided pd.DataFrame.
  """
  data = data.copy().set_index([series_index, date_index])
  data = data.sort_values(by=[date_index, series_index])
  # Infer most likely frequence for each series_index
  series_names = data.index.get_level_values(series_index).unique().tolist()
  series_frequencies = []
  for series in series_names:
    observed_times = data.iloc[data.index.get_level_values(series_index) ==
                               series].index.get_level_values(date_index)
    n_steps = len(observed_times)

    if n_steps > 1:
      time_diffs = (
          observed_times[1:n_steps] -
          observed_times[0:(n_steps - 1)]).astype('timedelta64[D]').array

      min_frequency = np.min(time_diffs)

      series_frequencies.append(min_frequency)

  if not series_frequencies:
    raise ValueError(
        'At least one series with more than one observation must be provided.')

  if series_frequencies.count(series_frequencies[0]) != len(series_frequencies):
    raise ValueError(
        'The provided time series seem to have irregular frequencies.')

  try:
    frequency = {
        1: 'D',
        7: 'W'
    }[series_frequencies[0]]
  except KeyError:
    raise ValueError('Frequency could not be identified. Got %d days.' %
                     series_frequencies[0])

  return frequency
