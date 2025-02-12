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
"""Install script for setuptools."""

import setuptools

__version__ = '1.0.0'

PROJECT_NAME = 'matched_markets'

REQUIRED_PACKAGES = [
    'absl-py', 'altair', 'numpy>=1.8.0rc1', 'pandas', 'matplotlib', 'scipy',
    'seaborn', 'six', 'statsmodels'
]

setuptools.setup(
    name=PROJECT_NAME,
    version=__version__,
    description=('Matched Markets Python library and colab demos for the ' +
                 'design and post analysis of quasi-geoexperiments'),
    author='Matched Markets developers',
    license='Apache 2.0',
    packages=setuptools.find_packages(),
    install_requires=REQUIRED_PACKAGES,
    python_requires='>=3.6')
