"""
shared functions used by all figures
(1) Functions to save figures and tables
(2) Function to plot bulk FPKM correlation plots

Nigel Jan 2020

License and readme found in https://github.com/khchenLab/split-fish
"""

import os

import datetime

from typing import Union

import numpy as np
import pandas as pd

from matplotlib.figure import Figure
import matplotlib.font_manager as fm

import seaborn as sns

from processing.plotCountsFunctions import _calcCorrAndCountFromDF


def saveTable(dataframe: pd.DataFrame,
              start_str: str,
              save_path: str,
              save_format: str = ".tsv"
              ) -> None:
    """
    save a figure

    fig: matplotlib figure
    start_str: beginning of file name
    save_format: image format e.g. .png
    save_path: folder to save in
    """
    timestr = datetime.datetime.now().strftime("_%Y%m%d_%H%M")
    filename = start_str + timestr + save_format

    save_filepath = os.path.join(save_path, filename)
    dataframe.to_csv(save_filepath, "\t")

    print(f"\n   saved table...\n"
          f"   savepath: {save_filepath}\n")


def saveFigure(fig: Figure,
               start_str: str,
               save_format,
               save_path,
               ) -> None:
    """
    save a figure

    fig: matplotlib figure
    start_str: beginning of file name
    save_format: image format e.g. .png
    save_path: folder to save in
    """
    timestr = datetime.datetime.now().strftime("_%Y%m%d_%H%M")
    filename = start_str + timestr + save_format

    save_filepath = os.path.join(save_path, filename)
    fig.savefig(save_filepath)

    print(f"\n   saved images...\n"
          f"   savepath: {save_filepath}\n")


def plotScatter(ax: Figure.axes,
                df: pd.DataFrame,
                x_column: str,
                y_column: str,
                fontprops: fm.FontProperties,
                spot_size: float = 30,
                alpha: float = 0.6,
                is_inset: bool = False,
                background_alpha: float = 0.8,
                xlim_offset: Union[None, float] = None,
                ylim_offset: Union[None, float] = 1,
                ) -> None:
    """
    plot a scatterplot

    :param ax:
    :param df:
    :param x_column:
    :param y_column:
    :param fontprops:
    :return:
    """

    def _findLogLowerBound(array: np.ndarray,
                           ) -> float:
        """
        calculate the lower bound on a log plot for the given array of values
        """
        # remove 0s from array (log of 0 undefined), also reject negative values
        array = array[array > 0.]
        min_val = np.amin(array)

        return 10 ** (np.floor(np.log10(min_val)))

    if is_inset:
        # within another plot
        sns.set_style("dark")
        label_color = "white"
    else:
        # standalone plot
        sns.set_style("darkgrid")
        label_color = "black"

    # counts for each axis
    count_values = df[y_column].values
    fpkm_values = df[x_column].values

    scatterplot = ax.scatter(
        x=fpkm_values, y=count_values,
        s=spot_size, alpha=alpha,
        edgecolors="none",
    )

    ax.set_ylabel(
        "count", labelpad=0, font_properties=fontprops,
        color=label_color,
    )

    # ax_scatter.set_yscale("log")

    linthreshy = _findLogLowerBound(count_values)
    print(f"linthreshy (counts axis) : {linthreshy}\n")

    ax.set_yscale(
        "symlog", linthreshy=linthreshy, linscaley=0.2
    )

    ax.set_xlabel(
        "FPKM value", labelpad=0, font_properties=fontprops,
        color=label_color,
    )

    # find value of linthreshx that is closest to
    # the lowest nonzero FPKM value

    linthreshx = _findLogLowerBound(fpkm_values)
    print(f"linthreshx (FPKM axis): {linthreshx}\n")

    ax.set_xscale(
        "symlog", linthreshx=linthreshx, linscalex=0.2
    )

    # set default axes limits
    ax.set_xlim((linthreshx, None))
    ax.set_ylim((linthreshy, None))

    if xlim_offset is not None:
        nonzero_fpkm_counts = fpkm_values[fpkm_values>0]
        ax.set_xlim((np.nanmin(nonzero_fpkm_counts) - xlim_offset, None))

    if ylim_offset is not None:
        nonzero_counts = count_values[count_values > 0]
        ax.set_ylim((np.nanmin(nonzero_counts) - ylim_offset, None))

    if is_inset:
        ax.patch.set_alpha(background_alpha)
        for side in ['top',
                     'bottom', 'left',
                     'right']:
            ax.spines[side].set_visible(False)
        ax.tick_params(
            axis='both', which='major',
            labelsize=5,
            labelcolor=label_color,
            pad=0,
        )

    results_dict = _calcCorrAndCountFromDF(
        df, x_column=x_column, y_column=y_column
    )

    ax.text(
        0.05, 0.95, f"ρ = {results_dict['correlation']:0.2f}",
        fontproperties=fontprops,
        color="darkred",
        horizontalalignment='left',
        verticalalignment='top',
        transform=ax.transAxes
    )

    return scatterplot
