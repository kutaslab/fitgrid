import os
import sys
from pathlib import Path
import pytest

# the following two lines are necessary to avoid segfaults when headless
import matplotlib

matplotlib.use('Agg')

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
)

import fitgrid

# LMER fits differ slightly depending on the build/env. Cap allowable
# variationo like so
#   np.allclose(actual, expected, atol=0, rtol=FIT_RTOL)
FIT_RTOL = 0.002


@pytest.fixture(scope="module")
def tpath(request):
    return Path(request.fspath).parent
