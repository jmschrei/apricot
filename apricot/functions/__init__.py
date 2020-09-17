# __init__.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

from .featureBased import FeatureBasedSelection
from .maxCoverage import MaxCoverageSelection

from .facilityLocation import FacilityLocationSelection
from .saturatedCoverage import SaturatedCoverageSelection
from .sumRedundancy import SumRedundancySelection
from .graphCut import GraphCutSelection

from .mixture import MixtureSelection

from .custom import CustomSelection
from .custom import CustomGraphSelection

from .base import BaseSelection
from .base import BaseGraphSelection