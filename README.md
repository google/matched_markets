# A Time-Based Regression Matched Markets Approach for Designing Geo Experiments

Copyright (C) 2020 Google LLC. License: Apache 2.0

## Disclaimer

This is not an officially supported Google product. For research purposes only.

## Description

Randomized experiments (Vaver & Koehler, 2010) have been recognized as the
gold standard to measure the causal effect of advertising, but how to design
and analyze them properly is a non-trivial statistical problem.
Geo experiments have the advantage of being privacy safe. However, in geo experiments, it may
not always be possible to rely on randomization to create balanced groups due to
the geo heterogeneity, the small number of units, and/or constraints that the
advertisers want to impose, such as including certain geos in a specific
experimental group. Hence, Au (2018) introduced a greedy search algorithm to
find the best experimental groups among those that satisfy:
  * the constraints imposed by the advertiser (budget, geo assignments, ...)
  * the assumptions of Time Based Regression (Kerman, 2017)

This directory contains

  * Python package for geo experiment design using Time Based Regression and Matched Markets.

  * Python package for geo experiment post analysis using Time Based Regression.

  * Colab demos for design and post analysis, separately.

## Installation

## Usage

For Python programming, here is an example usage.

```python
import itertools
import numpy as np
import pandas as pd
from matched_markets.methodology.tbrmmdata import TBRMMData
from matched_markets.methodology.tbrmatchedmarkets import TBRMatchedMarkets
from matched_markets.methodology.tbrmmdiagnostics import TBRMMDiagnostics
from matched_markets.methodology.tbrmmdesignparameters import TBRMMDesignParameters

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
  return [
      100 * geo + 10 * geo * day + day + np.random.randint(10)
      for day in range(n_days)
  ]

df[response_column] = 0.0
for geo in geos:
  sales_time_series = day_geo_sales(int(geo), n_days)
  df.loc[df.geo == geo, response_column] = sales_time_series

parameters = TBRMMDesignParameters(n_test=14, iroas=3.0,
                                   budget_range=(0.1, 300000))
data = TBRMMData(df, response_column)

mm = TBRMatchedMarkets(data, parameters)
designs = mm.greedy_search()

```

Without programming, the best way to learn how to use the package is probably by
following one of the notebooks, and the recommended way of opening them is
Google Colaboratory.

* Design a Matched Markets geo experiment.
   - [GeoX Design with Matched Markets and TBR](https://colab.sandbox.google.com/github/google/matched_markets/blob/master/matched_markets/notebook/design_colab_for_tbrmm.ipynb).
* Post analysis of a geo experiment from a Matched Markets design
   - [GeoX Post Analysis with TBR](https://colab.sandbox.google.com/github/google/matched_markets/blob/master/matched_markets/notebook/post_analysis_colab_for_tbrmm.ipynb).

## Contribution

## References

Available at
[Matched Markets paper](https://research.google/pubs/pub48983/).
[TBR paper](https://research.google/pubs/pub45950/).
[GeoX paper](https://research.google/pubs/pub38355/)
