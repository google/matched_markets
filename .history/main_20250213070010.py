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
