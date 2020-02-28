# __init__.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

from .functions import FeatureBasedSelection
from .functions import MaxCoverageSelection

from .functions import FacilityLocationSelection
from .functions import SaturatedCoverageSelection
from .functions import SumRedundancySelection
from .functions import GraphCutSelection

from .functions import MixtureSelection

__version__ = '0.5.0'
