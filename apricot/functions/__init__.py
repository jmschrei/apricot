# __init__.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

from .featureBased import FeatureBasedSelection
from .maxCoverage import MaxCoverageSelection

from .facilityLocation import FacilityLocationSelection
from .saturatedCoverage import SaturatedCoverageSelection
from .sumRedundancy import SumRedundancySelection
from .graphCut import GraphCutSelection

from .mixture import MixtureSelection

from .base import BaseSelection
from .base import BaseGraphSelection
from .base import _calculate_pairwise_distances