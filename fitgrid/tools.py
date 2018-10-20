import numpy as np
from collections import defaultdict


def get_index_duplicates_table(df, level):
    """Return a string table of duplicate index values and their locations."""

    assert level in df.index.names

    level_values = df.index.get_level_values(level)
    dupe_mask = level_values.duplicated(keep=False)

    dupes = level_values[dupe_mask]
    dupe_indices = np.flatnonzero(dupe_mask)

    dupe_dict = defaultdict(list)
    for index, value in zip(dupe_indices, dupes):
        dupe_dict[value].append(index)

    # hardcoded padding
    msg = f'Duplicates in index level {level}:\n\n'
    msg += '{0:<20} {1}\n'.format(level, 'Locations')
    for value, locations in sorted(dupe_dict.items()):
        locations = (str(item) for item in locations)
        msg += '{0:<20} {1}\n'.format(value, ', '.join(locations))

    return msg


def get_first_group(groupby):

    first_group_name = list(groupby.groups)[0]
    first_group = groupby.get_group(first_group_name)
    return first_group
