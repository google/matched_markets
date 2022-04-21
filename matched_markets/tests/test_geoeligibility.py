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
"""Test the GeoEligibility class.
"""
from matched_markets.methodology import geoeligibility
import pandas as pd

import unittest


class GeoAssignmentsTest(unittest.TestCase):

  def testGeoAssignments(self):
    c = {'G0', 'G3', 'G5', 'G6'}
    t = {'G1', 'G3', 'G4', 'G6'}
    x = {'G2', 'G4', 'G5', 'G6'}
    ga = geoeligibility.GeoAssignments(c, t, x)
    self.assertEqual(ga.c, c)
    self.assertEqual(ga.t, t)
    self.assertEqual(ga.x, x)
    self.assertEqual(ga.ct, {'G3'})
    self.assertEqual(ga.tx, {'G4'})
    self.assertEqual(ga.cx, {'G5'})
    self.assertEqual(ga.ctx, {'G6'})


class GeoEligibilityTest(unittest.TestCase):

  def setUp(self):
    """Set up a valid Geo Eligibility data frame."""
    super(GeoEligibilityTest, self).setUp()

    geonames = ['G%d' % i for i in range(7)]
    df = pd.DataFrame({'geo': geonames, 'control': 0, 'treatment': 0,
                       'exclude': 0},
                      columns=['geo', 'control', 'treatment', 'exclude'])
    df = df.set_index('geo')
    # Add all 7 valid assignments (0, 0, 0 is invalid).
    df.loc['G0'] = [1, 0, 0]  # Control only.
    df.loc['G1'] = [0, 1, 0]  # Treatment only.
    df.loc['G2'] = [0, 0, 1]  # Excluded only.
    df.loc['G3'] = [1, 1, 0]  # Control or Treatment.
    df.loc['G4'] = [0, 1, 1]  # Treatment or Excluded.
    df.loc['G5'] = [1, 0, 1]  # Control or Excluded.
    df.loc['G6'] = [1, 1, 1]  # Control, Treatment, or Excluded.
    self.df = df
    # Verify that the above dataframe does not raise errors.
    self.obj = geoeligibility.GeoEligibility(df)

  def testGeoColumn(self):
    """Checks if the geo column is there (as an index or column)."""
    # An index or column 'geo' (case sensitive) must exist.
    df = self.df.copy()
    df.index.name = 'Geo'
    with self.assertRaisesRegex(
        ValueError, r'There is no column or index \'geo\''):
      geoeligibility.GeoEligibility(df)

    df.reset_index(inplace=True)
    with self.assertRaisesRegex(
        ValueError, r'There is no column or index \'geo\''):
      geoeligibility.GeoEligibility(df)

    # Column 'geo' is also possible. No error raised.
    df = self.df.copy().reset_index()
    geoeligibility.GeoEligibility(df)

  def testColumnNames(self):
    """Checks if the required columns are available."""
    # The required column names are case sensitive.
    df = self.df.copy()
    new_columns = list(df.columns)
    new_columns[0] = 'Control'
    df.columns = new_columns
    with self.assertRaisesRegex(ValueError, r'Missing column\(s\): control'):
      geoeligibility.GeoEligibility(df)

    # Required columns must exist.
    df = self.df.copy()
    del df['exclude']
    with self.assertRaisesRegex(ValueError, r'Missing column\(s\): exclude'):
      geoeligibility.GeoEligibility(df)

    # Other columns are allowed.
    df = self.df.copy()
    df['newcolumn'] = 1
    geoeligibility.GeoEligibility(df)

    # Duplicated columns are not allowed.
    df = self.df.copy()
    df['newcolumn'] = 1
    df.columns = ['control', 'treatment', 'exclude', 'control']
    with self.assertRaisesRegex(ValueError, r'Duplicate column\(s\): control'):
      geoeligibility.GeoEligibility(df)

  def testDuplicateGeos(self):
    """Checks if there are any duplicate geos in the geo column."""
    df = self.df.copy()
    geos = df.index.tolist()
    geos[1] = 'G0'
    df.index = geos
    df.index.name = 'geo'
    with self.assertRaisesRegex(
        ValueError, r'\'geo\' has duplicate values: G0'):
      geoeligibility.GeoEligibility(df)

  def testBadValues(self):
    """Checks if there are any illegal values in the value columns."""
    # Only zeros and ones are allowed.
    df = self.df.copy()
    df.loc['G1'] = [1, 0, -1]
    with self.assertRaisesRegex(
        ValueError, 'GeoEligibility objects must have only values '
        '0, 1 in columns control, treatment, exclude'):
      geoeligibility.GeoEligibility(df)

    # Three zeros is an illegal value.
    df.loc['G1'] = [0, 0, 0]
    with self.assertRaisesRegex(
        ValueError, r'Three zeros found for geo\(s\) G1'):
      geoeligibility.GeoEligibility(df)

  def testStr(self):
    """Check the string representation."""
    self.assertEqual(str(self.obj), 'Geo eligibility matrix with 7 geos')

  def testDataProperty(self):
    """Check the data property."""
    self.assertEqual(id(self.obj.data), id(self.obj.data))

  def testEligibleAssignmentsDefault(self):
    """Test the method get_eligible_assignments, default case."""
    ga = self.obj.get_eligible_assignments()

    # The sets contain geo IDs.
    self.assertEqual(ga.c, {'G0', 'G3', 'G5', 'G6'})
    self.assertEqual(ga.t, {'G1', 'G3', 'G4', 'G6'})
    self.assertEqual(ga.x, {'G2', 'G4', 'G5', 'G6'})
    self.assertEqual(ga.c_fixed, {'G0'})
    self.assertEqual(ga.t_fixed, {'G1'})
    self.assertEqual(ga.x_fixed, {'G2'})
    self.assertEqual(ga.ct, {'G3'})
    self.assertEqual(ga.tx, {'G4'})
    self.assertEqual(ga.cx, {'G5'})
    self.assertEqual(ga.ctx, {'G6'})
    self.assertEqual(ga.all, {'G0', 'G1', 'G2', 'G3', 'G4', 'G5', 'G6'})

  def testEligibleAssignmentsOnlyIndicesSpecified(self):
    """Test the method get_eligible_assignments, indices=True."""
    with self.assertRaisesRegex(
        ValueError, '\'geos\' is not specified but indices=True'):
      self.obj.get_eligible_assignments(indices=True)

    with self.assertRaisesRegex(
        ValueError, '\'geos\' is not specified but indices=True'):
      self.obj.get_eligible_assignments(geos=None, indices=True)

  def testEligibleAssignmentsSubset(self):
    """Test the method get_eligible_assignments, subset specified."""
    geos = ['G1', 'G2', 'G3', 'G4']
    ga = self.obj.get_eligible_assignments(geos=geos)

    # The sets contain geo IDs from the list 'geos'.
    self.assertEqual(ga.c, {'G3'})
    self.assertEqual(ga.t, {'G1', 'G3', 'G4'})
    self.assertEqual(ga.x, {'G2', 'G4'})
    self.assertEqual(ga.c_fixed, set())
    self.assertEqual(ga.t_fixed, {'G1'})
    self.assertEqual(ga.x_fixed, {'G2'})
    self.assertEqual(ga.ct, {'G3'})
    self.assertEqual(ga.tx, {'G4'})
    self.assertEqual(ga.cx, set())
    self.assertEqual(ga.ctx, set())
    self.assertEqual(ga.all, {'G1', 'G2', 'G3', 'G4'})

  def testEligibleAssignmentsSubsetIndices(self):
    """Test the method get_eligible_assignments, given subset + indices."""
    geos = ['G4', 'G3', 'G2', 'G1']
    ga = self.obj.get_eligible_assignments(geos=geos, indices=True)

    # The sets contain indices pointing to the geo IDs in 'geos'.
    self.assertEqual(ga.c, {1})
    self.assertEqual(ga.t, {0, 1, 3})
    self.assertEqual(ga.x, {0, 2})
    self.assertEqual(ga.c_fixed, set())
    self.assertEqual(ga.t_fixed, {3})
    self.assertEqual(ga.x_fixed, {2})
    self.assertEqual(ga.ct, {1})
    self.assertEqual(ga.tx, {0})
    self.assertEqual(ga.cx, set())
    self.assertEqual(ga.ctx, set())
    self.assertEqual(ga.all, {0, 1, 2, 3})


if __name__ == '__main__':
  unittest.main()
