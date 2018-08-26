import matplotlib.pyplot as plt


def stripchart(data):

    plt.rcParams.update({'font.size': 15})

    _, n = data.shape
    fig, axes = plt.subplots(nrows=n, figsize=(16, n*3), sharey=True)
    data.plot(subplots=True, ax=axes)

    for ax in axes:
        ax.set(xlabel='')

    plt.tight_layout()
    plt.show()
