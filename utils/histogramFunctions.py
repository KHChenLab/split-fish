"""
functions relating to histograms

Nigel 31 jul 19

License and readme found in https://github.com/khchenLab/split-fish
"""
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


def plotHistogramGenes(data: np.ndarray,
                       num_genes: int,
                       savepath: str,
                       gene_names: list = None,
                       dpi: int = 800,
                       ):
    """
    input data should be an array of pixel assignments to genes,
    with each value an integer corresponding to the gene index
    """
    sns.set_style("dark")
    gene_callout_fig, ax_hist = plt.subplots()

    sns.distplot(data,
                 bins=list(range(1, num_genes + 1)),
                 ax=ax_hist)
    ax_hist.set_xlim((0, num_genes - 1))
    ax_hist.set_title("Histogram of gene callouts by pixel")

    gene_callout_fig.savefig(savepath, dpi=dpi)
