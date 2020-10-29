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
"""Data generator for the linear model."""

from matched_markets.methodology import semantics
from matched_markets.methodology import utils
import numpy as np
import pandas as pd
from scipy import stats


class DataSimulator(object):
  r"""Simulates geoexperiment datasets via a basic linear model.

  For the model:
  $y_{i,t} = \alpha_{sales} m_i + \beta \delta_{i,t} + m_i^\nu \epsilon_{i,t}$
  $x_{i,t} = \alpha_{cost} m_i  + \delta_{i,t} + m_i^\nu w_{i,t}$
  $\delta_{i,t} = m_i I(i \in treat) I(t \in test)$

  Where:
    * $y$ represents response values
    * $x$ represents input (cost) values.
    * $N_{ctrl}, N_{treat}$ represent the number of geos in the control and
      treatment groups respectively.
    * Geo means $m = [1, \ldots, N_{ctrl}, 1, \ldots, N_{treat}]$.
    * Heteroskedasticity parameter $\nu$, for example
      $\nu=0.5$ for $var(y_i) \propto m_i$
    * Causal cost in geo i at time t: $\delta_{i,t}$
    * $\epsilon_{i,t} \sim N(0, \sigma^2_{resp})$ normal error
      term for response.
    * $w_{i,t} \sim N(0, \sigma^2_{cost})$ normal error term for cost.

  Example:

  ```
  # Experimental design.
  n_control = 50
  n_treat = 50
  time_pre = 150
  time_test = 75

  # Linear params.
  hetresp = 1.0
  hetcost = 0.0
  beta = 0.0

  # Noise params.
  hetsked = 0.0
  sig_resp = 1.0
  sig_cost = 1.0

  # Column names.
  df_keys = {
             'key_response': 'sales',
             'key_cost': 'cost',
             'key_group': 'geo.group',
             'key_period': 'period',
             'key_geo': 'geo',
             'key_date': 'date'
             }

  # Make simulator.
  simulator = DataSimulator(n_control, n_treat,
                            time_pre, time_test,
                            hetresp, hetcost, beta,
                            hetsked, sig_resp, sig_cost,
                            **df_keys)

  # Simulate data.
  fake_data = simulator.sample()
  ```
  """

  def __init__(self,
               n_control, n_treat,
               time_pre, time_test,    # no cooldown as yet
               hetresp, hetcost, beta,
               hetsked, sig_resp, sig_cost,
               noise_treat_only=False,
               seed=None, **kwargs):
    """Creates a data simulator.

    Args:
      n_control: int. The number of control geos.
      n_treat: int. The number of treatment geos.
      time_pre: int. The number of pre-test period ticks.
      time_test: int. The number of test period ticks.
      hetresp: float. The degree of mean response variable heterogeneity.
      hetcost: float. The degree of mean cost variable heterogeneity.
      beta: float. The iROAS coefficient to be used.
      hetsked: float. The degree of heteroskedasticity in cost and response.
      sig_resp: float. The noise level in the response variable.
      sig_cost: float. The noise level in the cost variable.
      noise_treat_only: bool. Whether to add noise only in the treatment period.
      seed: int. Sets the seed of the random number generator.
      **kwargs: optional sematics for the produced data frame.
    """
    # Constants.
    self.n_control = n_control
    self.n_treat = n_treat
    self.time_pre = time_pre
    self.time_test = time_test
    self.time_total = time_pre + time_test

    # Model parameters.
    self.hetresp = hetresp
    self.hetcost = hetcost
    self.beta = beta
    self.hetsked = hetsked
    self.sig_resp = sig_resp
    self.sig_cost = sig_cost

    # Derived facts.
    self.n_total = self.n_treat + self.n_control
    self.col_len = self.n_total * self.time_total

    # Extract any column / index name information supplied by the user.
    user_df_names = utils.kwarg_subdict('key_', **kwargs)
    self._df_names = semantics.DataFrameNameMapping(**user_df_names)

    # Options
    self.noise_treat_only = noise_treat_only

    # Extract any semantics for control / treatment supplied by user.
    user_group_semantics = utils.kwarg_subdict('group_', **kwargs)
    self._groups = semantics.GroupSemantics(**user_group_semantics)

    # Extract any semantics for experimental period supplied by user.
    user_period_semantics = utils.kwarg_subdict('period_', **kwargs)
    self._periods = semantics.PeriodSemantics(**user_period_semantics)

    if seed is None:
      seed = np.random.randint(0, 2**32)
    self._rng = np.random.RandomState(seed=seed)

  def make_period_base(self):
    """Returns a vector indicating test period entries for one geo."""
    zeros_pre = np.zeros(self.time_pre)
    ones_test = np.ones(self.time_test)
    return np.hstack((zeros_pre, ones_test))

  def make_geo_sizes(self):
    """Returns a column of geo 'sizes' for constructing heterogeneity."""
    sizes_control = np.arange(1, self.n_control + 1)
    sizes_treat = np.arange(1, self.n_treat + 1)
    sizes = np.hstack((sizes_control, sizes_treat))
    return np.kron(sizes, np.ones(self.time_total))

  def make_geos(self):
    """Returns a column of geo labels."""
    geo_names = np.arange(1, self.n_total + 1)
    reps = np.ones(self.time_total)
    return np.kron(geo_names, reps)

  def make_periods(self):
    """Returns a column indicating experimental period of each entry."""
    period_base = self.make_period_base()
    return np.kron(np.ones(self.n_total), period_base)

  def make_groups(self):
    """Returns a vector of ones at treatment group entries, zero in control."""
    control = np.ones(self.n_control * self.time_total, dtype=int)
    treatment = 2*np.ones(self.n_treat * self.time_total, dtype=int)
    return np.hstack((control, treatment))

  def make_cost_causal(self):
    """Returns a column representing the cost caused by the experiment."""
    zeros_control = np.zeros(self.n_control)
    range_treat = np.arange(1, self.n_treat + 1)
    cost_base = np.hstack((zeros_control, range_treat))
    period_base = self.make_period_base()
    cost_causal = np.kron(cost_base, period_base)
    return cost_causal

  def make_test_mask(self):
    """Returns a column of ones in test period entries and zeros elsewhere."""
    return np.kron(np.ones(self.n_total), self.make_period_base())

  def make_noise(self, sig):
    """Returns a vector of additive noise with standard deviation sig."""
    sig_multiplier = sig * np.power(self.make_geo_sizes(), self.hetsked)
    white_noise = stats.norm.rvs(size=self.col_len, random_state=self._rng)
    noise = sig_multiplier * white_noise
    if self.noise_treat_only:
      noise *= self.make_test_mask()
    return noise

  def make_cost(self):
    """Returns a sales column for the dataset."""
    sizes = self.make_geo_sizes()
    cost_default = self.hetcost * sizes
    cost_causal = self.make_cost_causal()
    return cost_default + cost_causal + self.make_noise(self.sig_cost)

  def make_sales(self):
    """Returns a sales column for the dataset."""
    sizes = self.make_geo_sizes()
    means = self.hetresp * sizes
    incr_cost = self.make_cost_causal()
    return self.beta * incr_cost + means + self.make_noise(self.sig_resp)

  def make_dates(self):
    """Returns an integer column representing dates for the dataset."""
    return np.kron(np.ones(self.n_total), np.arange(self.time_total))

  def sample(self):
    """Draw a sample dataset from the model.

    Returns:
      A `pd.DataFrame`.
    """

    dates = self.make_dates()
    groups = self.make_groups()
    periods = self.make_periods()
    geos = self.make_geos()
    cost = self.make_cost()
    sales = self.make_sales()
    sizes = self.make_geo_sizes()

    data = {
        self._df_names.date: dates,
        self._df_names.group: groups,
        self._df_names.period: periods,
        self._df_names.geo: geos,
        self._df_names.response: sales,
        self._df_names.cost: cost,
        'size': sizes,
    }

    frame = pd.DataFrame(data, index=np.arange(self.col_len))
    frame = frame.set_index(self._df_names.geo, append=False)

    return frame
