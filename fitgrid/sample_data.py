from pathlib import Path
import re

DATA_DIR = Path(__file__).parents[0] / "data"
DATA_URL = "https://doi.org/10.5281/zenodo.3968485/files"

# Zenodo epochs files for testing and docs
DATA_URL = r"https://zenodo.org/record/3968485/files/"
P3_100_FEATHER = "sub000p3.ms100.epochs.feather"
P5_100_FEATHER = "sub000p50.ms100.epochs.feather"
WR_100_FEATHER = "sub000wr.ms100.epochs.feather"
PM_100_FEATHER = "sub000pm.ms100.epochs.feather"

P3_1500_FEATHER = "sub000p3.ms1500.epochs.feather"
P5_1500_FEATHER = "sub000p50.ms1500.epochs.feather"
WR_1500_FEATHER = "sub000wr.ms1500.epochs.feather"
PM_1500_FEATHER = "sub000pm.ms1500.epochs.feather"


def _download(filename, url=DATA_URL):
    """download filename from repo url to fitgrid/data/ 


    Parameters
    ----------
    filename : str
       file to fetch

    url : str {DATA_URL}
       top-level URL to fetch from


    """

    import requests
    import shutil

    # shortcut if previously downloaded
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"downloading ./fitgrid/data/{filename} from {url} ... please wait")
    if not url[-1] == r"/":
        url += r"/"

    resp = requests.get(url + filename, stream=True)
    # resp.raw.decode_content = True
    if resp.status_code == 200:
        with open(DATA_DIR / filename, 'wb') as out_f:
            shutil.copyfileobj(resp.raw, out_f)
    else:
        raise ConnectionError(f"URL response code: {resp.status_code}")


def get_file(filename, url=DATA_URL):
    """checks file is present in fitgrid/data, downloads if not

    Defaults to `Zenodo eeg-workshops/mkpy_data_examples/data v.0.0.3 <https://zenodo.org/record/3968485>`_

    Parameters
    ----------
    filename : str
      filename in the repository, , e.g., `sub000p3.ms1500.epochs.feather`.
    url: str
      fully qualified URL

    """
    if (DATA_DIR / filename).exists():
        return
    else:
        _download(filename, url=url)
