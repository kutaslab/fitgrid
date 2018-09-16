import pandas as pd
import matplotlib.pyplot as plt


def stripchart(data, negative_up=True):

    with plt.rc_context({'font.size': 14}):
        if isinstance(data, pd.Series):
            data = data.unstack()
        _, n = data.shape
        fig, axes = plt.subplots(nrows=n, figsize=(16, n * 3), sharey=True)
        data.plot(subplots=True, ax=axes)

        for ax in axes:
            ax.set(xlabel='')

        plt.tight_layout()
        if negative_up:
            plt.gca().invert_yaxis()
