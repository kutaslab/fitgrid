""".. _noaa_epochs:

NOAA tides and weather epochs
=============================

This example wrangles about 10 years of somewhat messy hourly NOAA
`meteorological observations 
<https://tidesandcurrents.noaa.gov/met.html?id=9410230&units=standard&timezone=GMT&action=data>`_
and `water levels
<https://tidesandcurrents.noaa.gov/waterlevels.html?id=9410230&units=standard&timezone=GMT&action=data>`_
into tidy `pandas.DataFrame` of time-stamped epochs ready to
load as `fitgrid.Epochs` for modeling.
 
1. Groom separate NOAA ocean water level and atmospheric observation
data files and merge into a single time-stamped `pandas.DataFrame`.

2. Add a column of event tags that mark the `high_tide` time-locking events of interest.

3. Snip the time-series apart into fixed length epochs and construct a
new column of time-stamps in each epoch with the `high_tide` event of
interest at time=0.

4. Export the epochs data frame to save for later use in `fitgrid`

Data Source: 

| NOAA CO-OPS-9419230  
| Station: La Jolla, CA 94102 (Scripps Pier)   
| August 1, 2010 - July 1, 2020  

The water levels are measured relative to mean sea level (MSL). For
further information about these data see `tide data options
<https://tidesandcurrents.noaa.gov/datum_options.html>`_, the
Computational Technniques for Tidal Datums Handbook [NOS-CO-OPS2]_
and the `NOAA Glossary
<https://tidesandcurrents.noaa.gov/glossary.html>`_.

"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import fitgrid as fg
from fitgrid import DATA_DIR

plt.style.use("seaborn-bright")
rc_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

np.random.seed(512)

# path to hourly water level and meteorology data
WDIR = DATA_DIR / "CO-OPS_9410230"

for pkg in [fg, np, pd]:
    print(pkg.__name__, pkg.__version__)


# %%
# Clean and merge tide and weather data
# -------------------------------------
#
# The `tides` variable is the hourly water level measurements. The
# original `date` and `time_(gmt)` columns are converted to a
# `pandas.Datetime` and serve as the index for merging the water level
# data with the meteorological observations.  Missing observations are coded `NaN`.

# read and tidy the NOAA water level .csv data files
tides = pd.concat(
    [pd.read_csv(tide_f, na_values='-') for tide_f in WDIR.glob('*msl_wl.csv')]
).drop(["Predicted (ft)", "Preliminary (ft)"], axis=1)

# sanitize the column names
tides.columns = [col.lower().replace(" ", "_") for col in tides.columns]

# make gmt date times usable
tides['date_time_gmt'] = pd.to_datetime(
    tides['date'] + ' ' + tides["time_(gmt)"]
)

# add local time at Scripps from the GMT
tides.insert(
    1,
    'hour_pst',
    (tides['date_time_gmt'] + pd.Timedelta(hours=-8)).dt.time.astype(str),
)

# drop unused columns
tides = tides.sort_values('date_time_gmt').drop(['date', 'time_(gmt)'], axis=1)

tides.rename(columns={"verified_(ft)": "water_level"}, inplace=True)

tides.set_index("date_time_gmt", inplace=True)
print(tides)


# %%
# `metobs` are hourly meteorological observations from the same NOAA station.
metobs = pd.concat(
    [pd.read_csv(tide_f, na_values='-') for tide_f in WDIR.glob('*met.csv')]
)
metobs.columns = [
    col.strip().lower().replace(" ", "_") for col in metobs.columns
]
metobs['date_time_gmt'] = pd.to_datetime(metobs.date_time)
metobs = metobs.drop(
    ["windspeed", "dir", "gusts", "relhum", "vis", "date_time"], axis=1
)[["date_time_gmt", "baro", "at"]]

metobs.set_index("date_time_gmt", inplace=True)
metobs.rename(columns={"baro": "mm_hg", "at": "air_temp"}, inplace=True)

print(metobs)


# %%
# The `data` pandas.DataFrame has the time-aligned tide and
# atmospheric observations, merged on their datetime stamp. Missing
# data `NaN` in either set of observations triggers exclusion of the
# entire row.

data = tides.join(metobs, on='date_time_gmt').dropna().reset_index()

# standardize the observations
for col in ["water_level", "mm_hg", "air_temp"]:
    data[col + "_z"] = (data[col] - data[col].mean()) / data[col].std()

# add a column of standard normal random values for comparison
data["std_noise_z"] = np.random.normal(loc=-0, scale=1.0, size=len(data))
print(data)

# %%
# set time=0 at high tide
# -----------------------
#
# The fixed length "epochs" are defined as intervals around the time=0
# "time-locking" event at `high_tide` defined by the local
# water-level maximum. Other time-lock events could be imagined:
# low-tide, high-to-low zero crossing etc.. Note that there are two
# high-tide events in each approximately 24 hour period.

# boolean vector True at water level local maxima , i.e., high tide
data['high_tide'] = (
    np.r_[
        False,
        data.water_level_z[1:].to_numpy() > data.water_level_z[:-1].to_numpy(),
    ]
    & np.r_[
        data.water_level_z[:-1].to_numpy() > data.water_level_z[1:].to_numpy(),
        False,
    ]
)

# these are just the high tide rows
print(data[data['high_tide'] == True])

# %%
# Define the epoch parameters: fixed-length duration, time-stamps, and epoch index.
#
# In this example, the epoch duration is 11 hours, beginning 3 hours before the high tides time lock event at time stamp = 0.
#
# 1. Fixed-length duration. This is the same for all epochs
#    in the data. In this example, the epoch is 11 hours, i.e., 11 measurements.#
#
# 2. Time stamps. This is an increasing sequence of integers, the same
#    length as the epoch. In this example the data are time-stamped
#    relative to high-tide at time=0, i.e., :math:`-3, -2, -1, 0, 1,
#    2, 3, 4, 5, 6, 7,`.
#
# 3. Assign each epoch an integer index that uniquely identifies
#    it. The indexes can be gappy but there can be no duplicates. In this
#    example the epoch index is a simple counter from 0 to the number of epochs - 1.
#
#
#

# 1. duration defined by the interval before and after the time lock event
pre, post = 3, 8

# 2. sequential time stamps
hrs = list(range(0 - pre, post))

# 3. epoch index is a counter for the high tide events.
n_obs = len(data)
ht_idxs = np.where(data.high_tide)[0]

# pre-compute epoch slice boundaries ... note these start-stop
# intervals may overlap in the original data
epoch_bounds = [
    (ht_idx - pre, ht_idx + post)
    for ht_idx in ht_idxs
    if ht_idx >= pre and ht_idx + post < n_obs
]

epochs = []
for start, stop in epoch_bounds:
    epoch = data.iloc[
        start:stop, :
    ].copy()  # slice the epoch interval from the original data
    epoch['epoch_id'] = len(epochs)  # construct
    epoch['time'] = hrs
    epochs.append(epoch)

epochs_df = pd.concat(epochs).set_index(['epoch_id', 'time'])
epochs_df.head()


# %%
# Visualize the epochs
dt_start, dt_stop = '2011-08-01', '2011-08-06 22:00:00'
aug_01_07_11 = data.query(
    "date_time_gmt >= @dt_start and date_time_gmt < @dt_stop"
)
print(aug_01_07_11)

f, ax = plt.subplots(figsize=(12, 8))
ax.set(ylim=(-3, 3))
ax.plot(
    aug_01_07_11.date_time_gmt,
    aug_01_07_11.water_level_z,
    lw=2,
    alpha=0.25,
    ls='-',
    marker='.',
    markersize=10,
    color='blue',
)

ax.plot(
    aug_01_07_11.date_time_gmt,
    aug_01_07_11.water_level_z,
    marker='.',
    markersize=10,
    lw=0,
)

ax.scatter(
    aug_01_07_11.date_time_gmt[aug_01_07_11.high_tide == True],
    aug_01_07_11.water_level_z[aug_01_07_11.high_tide == True],
    color='red',
    s=100,
    zorder=3,
    label="high tide",
)

for day, (_, ht) in enumerate(
    aug_01_07_11[aug_01_07_11.high_tide == True].iterrows()
):
    txt = ax.annotate(
        str(ht.hour_pst),
        (ht.date_time_gmt, ht.water_level_z),
        (ht.date_time_gmt, ht.water_level_z * 1.02),
        ha='left',
        va='bottom',
        fontsize=8,
        rotation=30,
    )
    if day in [3, 4, 5]:
        ax.axvline(ht.date_time_gmt, color='black', ls="--")
        ax.axvspan(
            ht.date_time_gmt - pd.Timedelta(pre, 'h'),
            ht.date_time_gmt + pd.Timedelta(post, 'h'),
            color='magenta',
            alpha=0.1,
        )
        if day == 5:
            ax.annotate(
                xy=(ht.date_time_gmt + pd.Timedelta(hours=1), 2.0),
                s=(
                    "Highlight indicates epoch bounds. Overlapping epochs\n"
                    "are legal but the observations are duplicated. This\n"
                    "will increase the epochs data size and may violate\n"
                    "modeling assumptions. This is not checked."
                ),
                size=12,
            )


ax.set_title(
    f"One week of standardized hourly water levels {dt_start} - {dt_stop} at La Jolla, CA "
)
ax.legend()
f.tight_layout()

# %%
# The tidied fixed-length, time-stamped epochs data may be saved for re-use as data tables.

# export time-stamped epochs for loading into fitgrid.Epochs
epochs_df.reset_index().to_feather(DATA_DIR / "CO-OPS_9410230.feather")
