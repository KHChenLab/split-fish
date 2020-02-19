"""
Functions for diplaying results on grids

Nigel - updated 27 nov 19

License and readme found in https://github.com/khchenLab/split-fish
"""

import os

from typing import Dict, List, Tuple

import numpy as np

from matplotlib.figure import Figure
import PyQt5.QtCore
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigCanvas
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigCanvas
from mpl_toolkits.axes_grid1 import make_axes_locatable

import seaborn as sns


def plotHeatmaps(grid_dict: Dict[str,
                                 Tuple[np.ndarray, int, int, str, float, float]],
                 fig_savepath: str = "",
                 dpi: int = 500,
                 iteration: int = 0,
                 verbose: bool = False,
                 ) -> None:
    """
    plots a series of heatmaps

    Parameters
    ----------
    grid_dict: 
        dictionary of parameters sets, keyed by the type of grid (str) for plotting heatmaps
        Each element in the list should be a tuple of:
        1) FOV grid (ndarray)
        2) vmin (minimum of colormap)
        3) vmax (max of colourmap)
        4) colourmap
        5) colourmap center
        6) annotation size
    fig_savepath: str
        directory to save the figure
    dpi: int
        dpi for saving the figure
    iteration: int
        indicates which iteration of decoding you are using
        only affects the naming of the saved figure
    """
    if verbose:
        print(f"\n-- Plotting Grids for:\n{grid_dict}\n")

    # set figure size, dpi and initiate a workable fig object
    # -------------------------------------------------------

    num_grids = len(grid_dict)
    figsize = (num_grids * 8, 8)
    fig = Figure(figsize=figsize, dpi=dpi)
    canvas = FigCanvas(fig)
    fig.set_canvas(canvas)

    ax_grids = {}  # main image axes
    for grid_num, grid_name in enumerate(grid_dict):
        parameter_set = grid_dict[grid_name]

        grid_array = parameter_set[0]
        grid_maxvalue = np.amax(grid_array)
        grid_dims = grid_array.shape
        max_dim = max(grid_dims)

        if grid_maxvalue >= 100:
            # for large numbers, format with no decimal places
            fmt = "0.0f"
        elif grid_maxvalue <= 1:
            # for small numbers e.g. correlation, use 3 d.p.
            fmt = "0.3f"
        else:
            fmt = "0.1f"
        ax_grids[grid_num] = fig.add_subplot(1, num_grids, grid_num + 1)
        divider = make_axes_locatable(ax_grids[grid_num])
        cbar_ax = divider.append_axes("right",
                                      size="5%",
                                      pad=0.08)

        # set font size for grid values
        # -----------------------------

        # scale annotation font by grid size
        annotation_size = parameter_set[5] * 10 / max_dim

        sns.heatmap(
            parameter_set[0],  # the FOV grid
            vmin=parameter_set[1],  # vmin
            vmax=parameter_set[2],  # vmax
            cmap=parameter_set[3],  # colourmap
            center=parameter_set[4],  # value at which to center the colormap
            cbar_ax=cbar_ax,
            annot=True,
            annot_kws={"size": annotation_size},
            fmt=fmt,
            square=True,
            ax=ax_grids[grid_num],
        )

        ax_grids[grid_num].set_title(grid_name)

    fig.tight_layout()

    # save the figure
    # ---------------

    fig.savefig(
        os.path.join(fig_savepath,
                     f"results_grids_iter{iteration}.png")
    )
    print(f"   saved grids in <{fig_savepath}>\n")

    # close the canvas
    # ----------------

    canvas.close()
    fig.clear()
