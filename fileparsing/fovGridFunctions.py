"""
Functions for generating a grid of FOVs
using a dataframe of files-data (including stage-position info)
containing entries from a single FOV
Originally a part of filesClasses

nigel 21 nov 2019

License and readme found in https://github.com/khchenLab/split-fish
"""

import os
import re
import numpy as np
import pandas as pd
from pprint import pprint as pp
import time
import warnings

from typing import Tuple, Dict, Union, List

import matplotlib.pyplot as plt
import seaborn as sns


def _stageToImageCoords(stage_to_pixel_matrix: np.ndarray,
                        stage_coords: np.ndarray,
                        ) -> np.ndarray:
    """
    uses the stage-to-pixel tranform to convert
    stage X/Y coordinates, given as 1d ndarray [stage Y, stage X]
    to image coordinates
      ----> X
      |
      v
      Y
    returned as 2 element image coordinates (in pixels) ndarray [Y, X]

    Parameters
    ----------
    stage_pixel_matrix: numpy array
        transformation matrix from stage coordinates to pixel coordinates
        used to create a grid of FOVs correctly positioned in pixel space

    """
    if stage_coords.shape != (2,):
        raise ValueError(
            f"Invalid stage coords {stage_coords}: "
            f"should be a 1D ndarray with shape (2,)"
        )

    return np.squeeze(np.dot(stage_to_pixel_matrix,
                             np.array([stage_coords[0], stage_coords[1]])))


def _divideCanvas(image_coord_dict: dict,
                  min_separation: float = 200,
                  verbose: bool = True,
                  ) -> Tuple[list, list]:
    """
    divide the entire field into segments so we can assign each FOV to a grid point
    e.g. FOV25 | FOV24 | FOV23 ...
               ^       ^
    same in the vertical direction

    assume that most FOVs in a row/column should be
    separated by at least 200 pixels (min_separation parameter)
    but would be offset from one another by less than
    200 pixels along the orthogonal axis
    (usually the are almost perfectly lined up)

    returns
    -------
    tuple (list of y separators, list of x separators)
    """

    y_list, x_list = [], []

    for fov in image_coord_dict:
        y_list.append(image_coord_dict[fov][0])
        x_list.append(image_coord_dict[fov][1])

    y_sorted = np.sort(y_list)
    x_sorted = np.sort(x_list)

    y_diff_list = y_sorted[1:] - y_sorted[:-1]
    x_diff_list = x_sorted[1:] - x_sorted[:-1]

    y_separators, x_separators = [], []

    for index, (y_diff, x_diff) in enumerate(zip(y_diff_list, x_diff_list)):
        if y_diff > min_separation:
            y_separators.append((y_sorted[index + 1] + y_sorted[index]) / 2)
        if x_diff > min_separation:
            x_separators.append((x_sorted[index + 1] + x_sorted[index]) / 2)

    if verbose:
        print(f"Y separators: {y_separators}\n"
              f"X separators: {x_separators}\n")

    return y_separators, x_separators


#
# --------------------------------------------------------------------------------------------
#                             Main function for generating grid
# --------------------------------------------------------------------------------------------
#

def generateFovGrid(files_df: pd.DataFrame,
                    stage_to_pixel_matrix: np.ndarray,
                    fov_subset: List[str] = None,
                    hyb: int = None,
                    colour_type: str = None,
                    plot_grid: bool = True,
                    fig_savepath: str = None,
                    verbose: bool = True,
                    ) -> Tuple[np.ndarray,
                               Dict[str, np.ndarray],
                               Dict[str, np.ndarray]]:
    """
    Generates a grid (numpy string array) containing the FOVs at each given position on the grid
    if no image found for the grid position, entry for that grid point will be "noimage"
    (calls _stageToImageCoords, _divideCanvas and plotFovCoordinates)

    If the hyb round and type (colour) to use is not specified,
    will choose the first one found in the files dataframe

    Parameters
    ----------
    files_df:
        files dataframe 
        MUST contain only entries from a single ROI
    fov_list: list
        list of all fovs in the grid
        (usually corresponds to all FOVs in the folder)
    fov_subset: list
        subset of FOVs that you want to analyze/stitich
    hyb, colour_type: int, str
        the hyb round and colour channel or type
        to get stage coordinates from
    plot_grid: bool
        whether to plot out the grid coordinates
    fig_savepath: str
        directory to save the figure in

    returns:
     1) FOV grid
     2) image coordinates {FOV: [Y, X] 1D ndarray,...}
     3) stage coordinates {FOV: [stage Y, stage X] 1D ndarray,...}
    """

    # choose hyb and colour to use if not provided
    # --------------------------------------------

    first_index = files_df.index[0]

    if hyb is None:
        hyb = files_df.loc[first_index, "hyb"]
        print(f"hyb round to use for grid-coordinates not provided. "
              f"Randomly setting hyb to {hyb}.\n")

    if colour_type is None:
        colour_type = files_df.loc[first_index, "type"]
        print(f"colour-type to use for grid-coordinates not provided. "
              f"Randomly setting colour_type to {colour_type}.\n")

    all_fovs = files_df["fov"].unique()

    if fov_subset is None:
        fov_subset = all_fovs
    else:
        fov_subset = [fov for fov in fov_subset if fov in all_fovs]

    if verbose:
        print(f"List of all FOVs in folder:\n{all_fovs}\n\n"
              f"Using subset of FOVs:\n{fov_subset}")

    stage_coords = {}
    image_coords = {}

    for fov in all_fovs:

        # Check file dataframe for stage x and y coordinate info
        # ------------------------------------------------------

        search_str = f"FOV {fov}, colour_type={colour_type}, hyb = {hyb}"

        fov_mask = files_df["fov"] == fov
        colour_mask = files_df["type"] == colour_type
        hyb_mask = files_df["hyb"] == hyb

        selection_mask = fov_mask & colour_mask & hyb_mask
        num_entries_found = np.count_nonzero(selection_mask)

        if num_entries_found > 1:
            warnings.warn(f"more than one entry found for: {search_str}")

        ycoord, xcoord = files_df.loc[selection_mask, ["ypos", "xpos"]].values[0, :]

        if verbose:
            print("-" * 45 + f"\n{search_str}:\n"
                             f"coordinates: y = {ycoord}, x = {xcoord}\n")

        if ycoord == np.nan:
            raise ValueError(f"no data found for y stage coordinates for {search_str}")
        if xcoord == np.nan:
            raise ValueError(f"no data found for x stage coordinates for {search_str}")

        stage_coords[fov] = np.array((ycoord, xcoord))
        image_coords[fov] = _stageToImageCoords(stage_to_pixel_matrix, stage_coords[fov])

    # Find dimensions of the FOV grid, initialize a grid with "noimage" in all positions
    # ----------------------------------------------------------------------------------

    y_separators, x_separators = _divideCanvas(image_coords, verbose=True)

    max_grid_y = len(y_separators) + 1
    max_grid_x = len(x_separators) + 1

    fov_grid = np.array([["noimage"] * max_grid_x] * max_grid_y,
                        dtype=np.unicode_)

    # Populate the FOV grid with the FOVs in the subset list
    # ------------------------------------------------------

    for fov in fov_subset:
        img_coord = image_coords[fov]
        grid_y = int(np.digitize(img_coord[0], y_separators))
        grid_x = int(np.digitize(img_coord[1], x_separators))
        fov_grid[grid_y, grid_x] = fov

    if verbose:
        print(f"Maximum grid extents: Y = {max_grid_y}, X = {max_grid_x}\n"
              f"Initializing FOV grid:\n {fov_grid}\n"
              f"shape = {fov_grid.shape}\ndtype = {fov_grid.dtype}\n"
              f"\nFinal FOV grid:\n{fov_grid}")

    if plot_grid:

        sns.set_style("darkgrid")
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))

        plotFovCoordinates(stage_coords, ax[0],
                           title_str="Stage")

        plotFovCoordinates(image_coords, ax[1],
                           title_str="Image",
                           grid_lines=(y_separators, x_separators),
                           fov_grid=fov_grid, pad=50)

        if fig_savepath is not None:
            time_str = time.strftime('%Y%m%d_%H%M%S')
            gridplot_filepath = os.path.join(fig_savepath, f"gridplot_{time_str}.png")
            fig.savefig(gridplot_filepath, dpi=500)

        fig.tight_layout()

    return fov_grid, image_coords, stage_coords


#
# --------------------------------------------------------------------------------------------
#                                           Plotting
# --------------------------------------------------------------------------------------------
#


def plotFovCoordinates(coord_dict: Dict[str, np.ndarray],
                       ax,  # matplotlib axes to plot on
                       title_str: str = "",
                       fov_grid: np.ndarray = None,
                       grid_lines: Tuple[list, list] = None,
                       pad: int = 10,
                       ) -> None:
    """
    plots the coordinates (either stage or image) for each FOV,
    labelling each point with the respective FOV.
    """

    # Set font-sizes based on size of grid.
    # minimum font size is 4
    fontsize = max(int(80 // np.sqrt(len(coord_dict))), 4)

    for fov in coord_dict:

        # plot the point and annotate it with FOV name
        # --------------------------------------------

        ax.plot(coord_dict[fov][1], coord_dict[fov][0], "r.")
        ax.text(coord_dict[fov][1] + pad, coord_dict[fov][0] - pad, str(fov),
                fontsize=fontsize, fontweight="bold")

        if fov_grid is not None:

            # annotate the grid position (y grid position, x grid position)
            # -------------------------------------------------------------

            grid_position = np.argwhere(fov_grid == fov)
            # print(f"grid position for {fov}: {grid_position}, {grid_position.size}")

            if grid_position.size >= 2:
                ax.text(coord_dict[fov][1] + pad, coord_dict[fov][0] + pad,
                        f"({grid_position[0,0]},{grid_position[0,1]})",
                        fontsize=fontsize, verticalalignment='top')

    if grid_lines is not None:

        # add grid lines separating centers of FOVs
        # -----------------------------------------

        for line in grid_lines[0]:  # y coords, horizontal lines
            ax.axhline(y=line, alpha=0.7, linestyle="--", color="orangered")
        for line in grid_lines[1]:  # x coords, vertical lines
            ax.axvline(x=line, alpha=0.7, linestyle="--", color="orangered")

    ax.set_aspect('equal', 'box')
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.set_title(f"Plot of {title_str} coords")
