import os
import sys
from pathlib import Path
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
)

import fitgrid


@pytest.fixture(scope="module")
def tpath(request):
    return Path(request.fspath).parent
