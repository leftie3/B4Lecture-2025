"""Solution to ex5 on variational inference of gaussian mixture models"""

import matplotlib.pyplot as plt
import numpy as np

from GMMVB import GMMVB


def plot_scatter(data: np.ndarray, title: str, filename: str) -> None:
    """Create and save scatterplot as image file.

    Params:
        data (ndarray): The input data to be plotted.
        title (str): The desired plot title.
        filename (str): Path to the desired output file.
    Returns: None
    """
    if len(data.shape) == 1 or data.shape[1] == 1:
        # 1-D data scatterplot (added violin plot for better visualisation)
        violin = plt.violinplot(
            data,
            orientation="horizontal",
            showextrema=True,
            showmeans=True,
        )["bodies"][0]
        violin.set_facecolor("#bf5cb2")
        plt.scatter(data, np.ones_like(data), c="#912583", zorder=2)

        plt.xlim(-3, 3)
        plt.title(title)
        plt.savefig(filename)
        plt.clf()
    else:
        # 2-D Data scatterplot
        plt.scatter(data[:, 0], data[:, 1], s=15, c="#912583", zorder=2)
        plt.title(title)
        plt.grid()
        plt.savefig(filename)
        plt.clf()


def main() -> None:
    """Execute main routine.

    Params: None
    Returns: None
    """
    # Data loading
    data1 = np.loadtxt("ex5/data1.csv", delimiter=",")[:, np.newaxis]
    data2 = np.loadtxt("ex5/data2.csv", delimiter=",")
    data3 = np.loadtxt("ex5/data3.csv", delimiter=",")

    # Scatterplots

    # Data 1 (added violin plot for better visualisation of 1-D data)
    plot_scatter(
        data1, title="Data 1 Scatterplot", filename="ex5/i_tzimas/data1_scatter.png"
    )
    # Data 2
    plot_scatter(
        data2, title="Data 2 Scatterplot", filename="ex5/i_tzimas/data2_scatter.png"
    )
    # Data 3
    plot_scatter(
        data3, title="Data 3 Scatterplot", filename="ex5/i_tzimas/data3_scatter.png"
    )

    # Variational Bayesian Inference
    vbgmm1 = GMMVB(K=2, filename="ex5/i_tzimas/data1_clusters.png")
    vbgmm1.execute(data1, iter_max=100, thr=0.001)

    vbgmm2 = GMMVB(K=3, filename="ex5/i_tzimas/data2_clusters.png")
    vbgmm2.execute(data2, iter_max=100, thr=0.001)

    vbgmm3 = GMMVB(K=3, filename="ex5/i_tzimas/data3_clusters.png")
    vbgmm3.execute(data3, iter_max=100, thr=0.001)


if __name__ == "__main__":
    main()
