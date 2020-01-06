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

# LMER fits differ slightly depending r-lme4 and r-matrix version. Test
# with np.allclose(actual, expected, atol=FIT_ATOL, rtol=FIT_RTOL)
# atol comes into play for numbers < 1, rtol for numbers > 1

# throw warnings on these
FIT_ATOL = 1e-6
FIT_RTOL = 1e-6

# fail on these
FIT_ATOL_FAIL = 1e-2
FIT_RTOL_FAIL = 1e-2


@pytest.fixture(scope="module")
def tpath(request):
    return Path(request.fspath).parent
