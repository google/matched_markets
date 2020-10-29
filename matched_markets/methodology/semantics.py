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
"""Constants and defaults used in the package.
"""

from six.moves import zip


class BaseSemantics(object):
  """Infrastructure for defaults with executable string representations."""

  def __init__(self, context_dict):
    """Store context_dict contents as class attributes."""
    # Safety for when used as a super class.
    context_dict.pop("self", None)
    # Remember the names of the attributes we just created.
    self.__attr_names = list(context_dict.keys())
    self.check_unique(context_dict)

  def check_unique(self, context_dict):
    """Raises an error if the dictionary mapping is not injective."""
    present_context_vals = set(context_dict.values())
    if len(context_dict) != len(present_context_vals):
      raise ValueError("Values for semantic keys should be unique.")

  def __str__(self):
    """Executable representation of the class."""
    # Obtain the values of the attributes to represent
    attr_vals = [getattr(self, name) for name in self.__attr_names]
    # Pair the names with their values
    attr_zip = list(zip(self.__attr_names, attr_vals))
    # Turn each attribute-value pair into an appopriate string.
    attr_pairs = ["""%s='%s'""" % (n, v) for (n, v) in attr_zip]
    # Obtain a string representation of key value pairs.
    specification = ",".join(attr_pairs)
    # Paste into an executable representation.
    class_name = self.__class__.__name__
    return "%s(%s)" % (class_name, specification)

  def __repr__(self):
    return self.__str__()


class DataFrameNameMapping(BaseSemantics):
  """Specifies the columns needed for a geoexperiment analysis.

  Defaults are provided.
  """

  def __init__(self,
               cost="cost",
               date="date",
               geo="geo",
               group="group",
               incr_cost="_incr_cost",
               incr_response="_incr_response",
               period="period",
               response="response"):
    """Represents the names of key columns in a geo experiment data frame.

    Args:
      cost: str. Name of the column containing costs.
      date: str. Name of the column containing date.
      geo: str. Name of the column containing geo ids.
      group: str. Name of the column containing group membership labels.
      incr_cost: str. Name of the column containing incremental costs.
      incr_response: str. Name of the column containing incremental response.
      period: str. Name of the column containing experimental period labels.
      response: str. Name of the column containing the response variable.
    """
    self.cost = cost
    self.date = date
    self.geo = geo
    self.group = group
    self.incr_cost = incr_cost
    self.incr_response = incr_response
    self.period = period
    self.response = response
    # Set the relevant attributes to be output when printing an instance.
    BaseSemantics.__init__(self, locals())


class GroupSemantics(BaseSemantics):
  """Specifies the semantics for interpreting group data."""

  def __init__(self, control=1, treatment=2, unassigned=-1):
    """Represents group label semantics.

    Args:
      control: int. Label for control geos.
      treatment: int. Label for treatment geos.
      unassigned: int. Label for geos neither in treatment nor control group.
    """
    self.control = control
    self.treatment = treatment
    self.unassigned = unassigned
    # Set the relevant attributes to be output when printing an instance.
    BaseSemantics.__init__(self, locals())


class PeriodSemantics(BaseSemantics):
  """Specifies the semantics for interpreting period information."""

  def __init__(self, pre=0, test=1, cooldown=2, unassigned=-1):
    """Represents experimental period label semantics.

    Args:
      pre: int. Label for ticks in the pre-test phase.
      test: int. Label for ticks in the test phase.
      cooldown: int. Label for ticks in cooldown phase.
      unassigned: int. Label for ticks present but not assigned to a phase.
    """
    self.pre = pre
    self.test = test
    self.cooldown = cooldown
    self.unassigned = unassigned
    # Set the relevant attributes to be output when printing an instance.
    BaseSemantics.__init__(self, locals())
