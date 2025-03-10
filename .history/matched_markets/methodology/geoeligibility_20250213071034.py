from typing import List, Optional, Set, Text, Union
import dataclasses
import pandas as pd

GeoRef = Union[Text, int]

@dataclasses.dataclass
class GeoAssignments:
    """Representation of all possible geo assignments."""
    all: Set[GeoRef]
    c: Set[GeoRef]
    t: Set[GeoRef]
    x: Set[GeoRef]
    c_fixed: Set[GeoRef]
    t_fixed: Set[GeoRef]
    x_fixed: Set[GeoRef]
    ct: Set[GeoRef]
    cx: Set[GeoRef]
    ctx: Set[GeoRef]
    tx: Set[GeoRef]

    def __init__(self, c: Set[GeoRef], t: Set[GeoRef], x: Set[GeoRef]):
        self.c = c
        self.t = t
        self.x = x
        self.all = c | t | x
        
        self.c_fixed = c - (t | x)
        self.t_fixed = t - (c | x)
        self.x_fixed = x - (c | t)
        
        self.ct = (c & t) - x
        self.cx = (c & x) - t
        self.ctx = c & t & x
        self.tx = (t & x) - c

class GeoEligibility:
    """Validate a Geo Eligibility Matrix."""
    def __init__(self, df: pd.DataFrame):
        df = df.copy()
        df.reset_index(drop=True, inplace=True)

        required_columns = {'geo', 'control', 'treatment', 'exclude'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f'Missing required column(s): {", ".join(missing)}')

        if df.columns.duplicated().any():
            raise ValueError('Duplicate columns found in DataFrame.')

        df['geo'] = df['geo'].astype(str)
        
        if df.duplicated(subset=['geo']).any():
            dup_geo_ids = df['geo'][df.duplicated(subset=['geo'])].unique()
            raise ValueError(f'Duplicate geo values found: {", ".join(dup_geo_ids)}')
        
        if not all(df[col].isin([0, 1]).all() for col in ['control', 'treatment', 'exclude']):
            raise ValueError('Columns control, treatment, and exclude must contain only 0 or 1.')
        
        if (df[['control', 'treatment', 'exclude']].sum(axis=1) == 0).any():
            zero_rows = df['geo'][df[['control', 'treatment', 'exclude']].sum(axis=1) == 0]
            raise ValueError(f'Invalid rows with all zeros found for geos: {", ".join(zero_rows)}')
        
        df.set_index('geo', inplace=True)
        self.data = df
    
    def __str__(self):
        return f'Geo eligibility matrix with {self.data.shape[0]} geos'
    
    def get_eligible_assignments(self, geos: Optional[List[GeoRef]] = None, indices: bool = False) -> GeoAssignments:
        if indices and geos is None:
            raise ValueError('`geos` must be specified when `indices=True`.')
        
        df = self.data if geos is None else self.data.loc[geos]
        
        if indices:
            df = df.reset_index()
        
        c = set(df.index[df['control'] == 1])
        t = set(df.index[df['treatment'] == 1])
        x = set(df.index[df['exclude'] == 1])
        
        return GeoAssignments(c, t, x)
