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
#   np.allclose(actual, expected, atol=FIT_ATOL, rtol=FIT_RTOL)
FIT_ATOL = 0.001   # 3rd decimal place comes into play for numbers < 1
FIT_RTOL = 0.001   # + two tenths of one percent comes into play for numbers >> 1

@pytest.fixture(scope="module")
def tpath(request):
    return Path(request.fspath).parent
