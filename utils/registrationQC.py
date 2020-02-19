"""
Functions for calculating and plotting
registration and shift-related quality-control metrics

1) showRegistrationByBit
   ---------------------
   show registration of images from each bit
   to a single reference frame

2) plotShiftsByFOV
   ---------------
   Plots shifts in x and y (absolute value)
   at each position on the FOV grid.
   also plots a histogram of all shifts (both x and y)

3) plotShiftsClustermap
   --------------------
   similar to (2). plots x and y shifts (separate plots)
   and clusters them by hyb and FOV

- Nigel 4 Jul 19
- added registration by bit function - nigel 15 Jul19

License and readme found in https://github.com/khchenLab/split-fish
"""

import os
import numpy as np
import pandas as pd

from typing import Tuple, Dict, Union, List

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from matplotlib.figure import Figure, Axes
import PyQt5.QtCore
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigCanvas
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigCanvas

from pprint import pprint as pp

import tkinter as tk
from tkinter import filedialog


def showRegistrationByBit(registered_images: np.ndarray,
                          reference_bit: int,
                          shifts: np.ndarray,
                          registration_error_dict: Dict[int, Dict[str, float]] = None,
                          fig_savepath: str = "",
                          filtered: bool = True,
                          fov_str: str = "",
                          dropped_bits: List[int] = (),
                          figure_grid: Tuple[int, int] = (3, 6),
                          pct_range: Tuple[float, float] = (10, 99.6),
                          fontsize: int = 14,
                          figsize: Tuple[float, float] = (18, 9),
                          dpi: int = 600,
                          ) -> None:
    """
    create a Figure showing registration of each image to the reference image
    reference image is in red and image is green (overlap yellow)
    reference bit image is plotted as grayscale
    accepts any Z by Y by X by num_bits ndarray containing registered image data

    saves the image directly to the figure savepath, then deletes the figure reference


    Parameters
    ----------
    registered_images: numpy array
        frames by Y by X by num_bits image array
    reference_bit: int
        bit used as reference for registration
    shifts: numpy array
        bits by 2 array of (y, x) shifts
    registration_error_dict: dict
        dictionary of registration error values
        keyed by bit
        each value is a dict of {"fine error":float, "pixel error": float}
    fig_savepath: str
        directory to save the figure in
    filtered: bool
        whether the image is filtered
    ______ Figure info _______
    fov_str: str
        a string representing the FOV from which images are displayed
    image_info: str
        additional info on image e.g. if it was max intensity projected etc.
    figure_grid: 2-tuple of integers
        the grid for displaying images. If there are less grid positions
        than images, will stop when figure is filled
    pct_range: 2-tuple of floats
        low and high end of intensity range by percentile for display
    ______ matplotlib figure options ______
    fontsize: int
        fontsize of the text indicating shifts and errors
    figsize: 2-tuple
        matplotlib figure size
    dpi: int
        dpi for saving figure
    """

    def _addText(ax: Axes,
                 text: str,
                 fontsize=fontsize,
                 ) -> None:
        """
        add text to the top right hand corner of an image
        """
        ax.text(0.02, 0.98, text,
                color='w',
                weight='bold',
                fontsize=fontsize,
                fontname='Arial',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes
                )

    print(f"\n-- Visualizing registration for images in FOV {fov_str}")
    # the number of bits is the length of the last dimension of the image array
    num_bits = registered_images.shape[-1]

    # set figure size, dpi and initiate a workable fig object
    fig_reg = Figure(figsize=figsize, dpi=dpi)
    canvas = FigCanvas(fig_reg)
    fig_reg.set_canvas(canvas)

    ax_reg = {}

    #
    # Set up RGB array and add reference image
    # ----------------------------------------
    #

    # temporary RGB array to compare the reference and offset images
    rgb_temp = np.zeros(
        (registered_images.shape[1], registered_images.shape[2], 3),
        dtype=np.float32,
    )

    #  get reference image
    ref_temp = registered_images[0, :, :, reference_bit]

    # get upper/lower intensity limits for reference image
    if filtered:
        ref_min = np.percentile(ref_temp, pct_range[0])
    else:
        ref_min = 0
    ref_max = np.percentile(ref_temp, pct_range[1])

    ref_norm = (ref_temp - ref_min) / (ref_max - ref_min)

    # add reference image into red channel of RGB array
    rgb_temp[:, :, 0] = ref_norm

    # maximum number of plots that can be shown
    # -----------------------------------------
    max_plots = min(num_bits, figure_grid[0] * figure_grid[1])

    for bit in range(max_plots):

        ax_reg[bit] = fig_reg.add_subplot(figure_grid[0],
                                          figure_grid[1],
                                          bit + 1)

        # if it is the reference bit, just show a grayscale image
        if bit == reference_bit:
            # show grayscale image of reference
            ax_reg[bit].imshow(np.clip(ref_norm, 0, 1),
                               cmap="gray")
            _addText(ax_reg[bit],
                     f"bit {bit:d}: reference bit")

        elif bit in dropped_bits:
            # show grayscale image of dropped bit
            ax_reg[bit].imshow(np.clip(ref_norm, 0, 1),
                               cmap="gray")
            _addText(ax_reg[bit],
                     f"bit {bit:d}: dropped")

        else:
            #  registered image
            reg_temp = registered_images[0, :, :, bit]
            if filtered:
                reg_min = np.percentile(reg_temp, pct_range[0])
            else:
                reg_min = 0
            reg_max = np.percentile(reg_temp, pct_range[1])

            rgb_temp[:, :, 1] = (reg_temp - reg_min) / (reg_max - reg_min)
            ax_reg[bit].imshow(np.clip(rgb_temp, 0, 1))

            _addText(ax_reg[bit],
                     f"bit {bit:d} to bit {reference_bit:d}:\n"
                     f"({shifts[bit, 0]:.2f}, {shifts[bit, 1]:.2f})\n"
                     + "\n".join(
                         [f"{param}: {registration_error_dict[bit][param]:.2e}"
                          for param in registration_error_dict[bit]
                          if registration_error_dict[bit][param] is not None])
                     )

        ax_reg[bit].axis('off')

    # Overall Figure Title
    # --------------------

    fig_reg.suptitle(f"FOV {fov_str} registration",
                     color="darkred",
                     fontsize=16,
                     fontname="Arial",
                     weight="bold"
                     )

    # Adjust figure spacing
    # ---------------------

    fig_reg.subplots_adjust(left=0.01, right=0.99,
                            bottom=0.02, top=0.95,
                            wspace=0.04, hspace=0.01)

    # save the images
    # ---------------

    if fig_savepath:
        fig_reg.savefig(
            os.path.join(fig_savepath,
                         f"FOV_{fov_str}_image_registration_plot.png")
        )
        print(f"   saved registered images for {fov_str} in\n"
              f"   <{fig_savepath}>\n")

        # close the canvas
        # ----------------
        canvas.close()
        fig_reg.clear()

    else:
        fig_reg.show()


def plotShiftsByFOV(shifts_dict: dict,  # dictionary of shifts keyed by FOVs
                    fov_grid: np.ndarray,
                    seperate_plots: bool = False,  # plot all in one figure (uses GridSpec) or seperately
                    save_filepath: str = "",
                    verbose: bool = True,
                    ):
    """
    plot the x and y shifts for each FOV and display on a grid
    also plot the histogram of all shifts
    """

    # find the number of bits from
    # the first dimension of one of the shifts-arrays in the dictionary
    num_bits = next(iter(shifts_dict.values())).shape[0]

    if verbose:
        print(f"shifts dictionary:\n{shifts_dict}\n"
              f"number of bits detected in shifts dictionary: {num_bits}")

    # create a num_fovs x num_bits x 2 array to store all shifts
    abs_shifts_combined = np.zeros((len(shifts_dict), num_bits, 2))
    for index, fov in enumerate(shifts_dict):
        abs_shifts_combined[index, :, :] = np.abs(shifts_dict[fov])
    maxshift_global = np.amax(abs_shifts_combined)

    bit_list = np.arange(num_bits)

    if seperate_plots:
        fig_shifts, ax_shifts = plt.subplots(fov_grid.shape[0], fov_grid.shape[1],
                                             sharex='col', sharey='row',
                                             figsize=(9, 12))
        for i in range(fov_grid.shape[0]):
            for j in range(fov_grid.shape[1]):
                if fov_grid[i, j] != "noimage":
                    shifts_perfov = np.abs(shifts_dict[fov_grid[i, j]])
                    # the x locations for the groups
                    width = 0.45  # the width of the bars
                    yshifts_bar = ax_shifts[i, j].bar(bit_list - width / 2, shifts_perfov[:, 0], width, )
                    xshifts_bar = ax_shifts[i, j].bar(bit_list + width / 2, shifts_perfov[:, 1], width, )
                    ax_shifts[i, j].set_ylim((0, maxshift_global))

        fig_shifts_hist, ax_shifts_hist = plt.subplots(figsize=(8, 8))
        sns.distplot(abs_shifts_combined.flat, ax=ax_shifts_hist)

    else:

        # gridspec inside gridspec
        fig_shifts = plt.figure(figsize=(14, 8))
        gs0 = gridspec.GridSpec(1, 2, figure=fig_shifts)
        gs00 = gridspec.GridSpecFromSubplotSpec(fov_grid.shape[0], fov_grid.shape[1], subplot_spec=gs0[0])
        # gs00 = gs0[0].subgridspec(fov_grid.shape[0], fov_grid.shape[1])

        ax_shifts = {}
        for i in range(fov_grid.shape[0]):
            for j in range(fov_grid.shape[1]):
                ax_temp = fig_shifts.add_subplot(gs00[i, j])
                if fov_grid[i, j] != "noimage":
                    shifts_perfov = np.abs(shifts_dict[fov_grid[i, j]])
                    # the x locations for the groups
                    width = 0.45  # the width of the bars
                    yshifts_bar = ax_temp.bar(bit_list - width / 2, shifts_perfov[:, 0], width, )
                    xshifts_bar = ax_temp.bar(bit_list + width / 2, shifts_perfov[:, 1], width, )
                    ax_temp.set_ylim((0, maxshift_global))

        ax_shifts_hist = fig_shifts.add_subplot(gs0[0, 1])
        sns.distplot(abs_shifts_combined.flat, ax=ax_shifts_hist)

    fig_shifts.tight_layout()

    if save_filepath:
        fig_shifts.savefig(os.path.join(save_filepath, "shifts_by_fov.png"), dpi=400)
        if seperate_plots:
            fig_shifts.savefig(os.path.join(save_filepath, "shifts_by_fov_hist.png"), dpi=400)


def _shiftsDictToDF(shifts_dict: dict,
                    verbose: bool = False,
                    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    convert a shifts dictionary into 2 separate dataframes (for Y and X shifts)
    with rows    = FOVs
    and  columns = bits
    :returns {"y_shifts": dataframe, x_shifts": dataframe}
    """
    # find the number of bits from
    # the first dimension of one of the shifts-arrays in the dictionary
    num_bits = next(iter(shifts_dict.values())).shape[0]

    yx_dataframes = {}

    yshifts_df = pd.DataFrame(np.nan,
                              columns=list(range(num_bits)),
                              index=list(shifts_dict.keys()))
    yshifts_df.index.name = "FOV"

    xshifts_df = pd.DataFrame(np.nan,
                              columns=list(range(num_bits)),
                              index=list(shifts_dict.keys()))
    xshifts_df.index.name = "FOV"

    for fov in shifts_dict:
        yshifts_df.loc[fov, :] = shifts_dict[fov][:, 0]
        xshifts_df.loc[fov, :] = shifts_dict[fov][:, 1]

    if verbose:
        print(f"__ y-shifts dataframe __ :\n{yshifts_df}\n{yshifts_df.dtypes}\n"
              f"__ x-shifts dataframe __ :\n{xshifts_df}\n{xshifts_df.dtypes}\n")

    return yshifts_df, xshifts_df


def plotShiftsClustermap(shifts_dict: dict,
                         save_filepath: str = "",
                         verbose: bool = False,
                         ):
    """
    plot clustermaps for the y-shifts and x-shifts
    :returns references to the facetgrids of the y clustermap and x clustermap
    """
    # convert shift dictionary to dataframes for y and x shifts
    yshifts_df, xshifts_df = _shiftsDictToDF(shifts_dict, verbose=verbose)

    # get largest postive and negative shifts in either x or y direction
    overall_max = max(yshifts_df.values.max(), xshifts_df.values.max())
    overall_min = min(yshifts_df.values.min(), xshifts_df.values.min())

    # y clustermap
    # ------------

    y_cluster = sns.clustermap(yshifts_df, center=0, cmap="vlag",
                               vmin=overall_min, vmax=overall_max)
    y_cluster.ax_heatmap.set_xlabel("bit number")
    plt.setp(y_cluster.ax_heatmap.get_yticklabels(), rotation=0)  # For y axis

    # x clustermap
    # ------------

    x_cluster = sns.clustermap(xshifts_df, center=0, cmap="vlag",
                               vmin=overall_min, vmax=overall_max)
    x_cluster.ax_heatmap.set_xlabel("bit number")
    plt.setp(x_cluster.ax_heatmap.get_yticklabels(), rotation=0)  # For y axis

    if save_filepath:
        y_cluster.savefig(os.path.join(save_filepath, "yshifts_clustermap.png"), dpi=500)
        x_cluster.savefig(os.path.join(save_filepath, "xshifts_clustermap.png"), dpi=500)

    return y_cluster, x_cluster


#
# ------------------------------------------------------------------------------------------
#                               Test with a set of 3 shifts
# ------------------------------------------------------------------------------------------
#

if __name__ == "__main__":
    shiftdict_eg = {"00": np.array([[1., 2.], [2., 1.], [6., 0.5]], dtype=np.float64),
                    "01": np.array([[3., 2.], [7., -1.], [-4.5, 0.5]], dtype=np.float64), }

    print(f"example dictionary:\n{pp(shiftdict_eg)}")

    yshifts_df, xshifts_df = _shiftsDictToDF(shiftdict_eg, verbose=True)
    plotShiftsClustermap(shiftdict_eg)

    plt.show()
