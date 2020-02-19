"""
Figure 3 and Fig 2c
-------------------

Plot of selected genes in whole imaging area
for a variety of organs
Plots the corresponding FPKM correlation plot which appears in fig 2c

set plot_images = False since stitched images (too big)
are not provided with repository

Nigel Jan 2020

License and readme found in https://github.com/khchenLab/split-fish
"""

import os
import h5py

from typing import Tuple, Dict, List

import numpy as np
import pandas as pd

from matplotlib.figure import Figure
import PyQt5.QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigCanvas

import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import matplotlib.gridspec as gridspec

import seaborn as sns

import tkinter as tk
from tkinter import filedialog

from splitFISH_manuscript_figures.figure_shared_functions import saveFigure, plotScatter


def makeFig3(data_path: str,
             subdir_params: Dict[
                 str,
                 Tuple[int, int, int, int, int, int, float, float,
                       List[Tuple[str, float, str]]
                 ],
             ],
             downsample: int = 2,
             fig_savepath: str = "",
             save_format: str = ".png",
             um_per_pixel: float = 0.12,
             pct_range: Tuple[float, float] = (45, 99.8),
             figsize: Tuple[float, float] = (9, 9),
             dpi: int = 500,
             label_text: bool = True,
             plot_image: bool = True,
             show_scatterplot: bool = False,
             verbose: bool = True,
             ) -> None:
    """
    Make a figure showing raw images from various tissues/cell types
    with selected genes plotted on top of them

    Parameters
    ----------
    data_path: str
        main data path with subdirectories containing
        images from different cell or tissue types
    subdir_params
        crop (upper,left, ydim,xdim) and
        intensity normalization (lower limit, upper limit)
        parameters for the pair of images in each subdirectory
    plot_image: bool
        show stitched image or leave as blank canvas
    """

    sns.set_style("white")

    fig = Figure(figsize=figsize, dpi=dpi)
    canvas = FigCanvas(fig)
    fig.set_canvas(canvas)

    folder_contents = os.listdir(data_path)

    if verbose:
        print(f"Files found in data folder:\n{folder_contents}\n")

    # specify dimensions of figure grid
    # ---------------------------------

    xgrid_length = 2
    ygrid_length = int(np.ceil(len(subdir_params) / xgrid_length))

    outer_grid = gridspec.GridSpec(
        ygrid_length, xgrid_length, figure=fig,
        hspace=0.05, wspace=0.05,
    )

    # Generate 2nd figure to show scatterplots seperately
    # ---------------------------------------------------

    if not show_scatterplot:
        fig2 = Figure(figsize=figsize, dpi=dpi)
        canvas2 = FigCanvas(fig2)
        fig2.set_canvas(canvas2)
        grid2 = gridspec.GridSpec(
            ygrid_length, xgrid_length, figure=fig2,
            hspace=0.2, wspace=0.2,
        )

    for dir_num, subdir in enumerate(subdir_params):

        subdir_path = os.path.join(data_path, subdir)

        assert os.path.isdir(subdir_path), f"subdirectory {subdir_path} could not be found."

        # Look for stitched mosaic and spots file
        # ---------------------------------------

        stitched_filepath = None
        spots_filepath = None
        counts_filepath = None

        for file in os.listdir(subdir_path):

            if file.endswith(".hdf5"):

                if file.startswith("stitched"):
                    stitched_filepath = os.path.join(subdir_path, file)
                elif file.startswith("coords_combined"):
                    spots_filepath = os.path.join(subdir_path, file)

            elif file.startswith("allFOVs_counts"):

                counts_filepath = os.path.join(subdir_path, file)

        if plot_image:
            assert stitched_filepath is not None, (
                f"Could not find stitched file in {subdir_path}!"
            )
        assert spots_filepath is not None, (
            f"Could not find spots file in {subdir_path}!"
        )
        if counts_filepath is None:
            print(f"Could not find counts file in {subdir_path}!")

        # Load and plot image
        # -------------------

        (corner_y, corner_x,
         y_length, x_length,
         vmin, vmax,
         text_ycorner, text_xcorner,
         xlim_offset, ylim_offset,
         genes_list,
         ) = subdir_params[subdir]

        end_corner_y = corner_y + y_length
        end_corner_x = corner_x + x_length

        ax = fig.add_subplot(
            outer_grid[dir_num // xgrid_length, dir_num % xgrid_length]
        )

        if plot_image:

            with h5py.File(stitched_filepath, 'r') as f:
                stitched_img = np.array(f["stitched"][::downsample, ::downsample])

            img_crop = stitched_img[corner_y:end_corner_y, corner_x:end_corner_x]

            print(
                f"Image crop for subdirectory {subdir}, "
                f"image {stitched_filepath}\n"
                f"has dimensions {img_crop.shape}."
            )

            if vmin is None:
                vmin = np.percentile(img_crop, pct_range[0])
            if vmax is None:
                vmax = np.percentile(img_crop, pct_range[1])

        else:
            img_crop = np.zeros((y_length, x_length))
            vmin = 0
            vmax = 1

        ax.imshow(
            img_crop, cmap="gray", vmin=vmin, vmax=vmax
        )
        ax.axis("off")

        # shared font properties for the gene labels
        label_fontprops = fm.FontProperties(
            size=10, family="Arial", weight="bold"
        )

        # label organ/tissue type on top left
        # -----------------------------------

        if label_text:
            ax.text(
                0.02, 0.98, subdir[0].upper() + subdir[1:],
                fontproperties=label_fontprops,
                color="white",
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes
            )

        with h5py.File(spots_filepath, 'r') as f:

            for gene, markersize, colour in genes_list:

                # Read and plot RNA spot positions
                # --------------------------------

                if f[gene].shape[0] == 0:
                    print(f"No RNA spots found for {gene}.\n")
                    continue

                y_coord = f[gene][:, 1]
                x_coord = f[gene][:, 2]

                y_coord /= downsample
                x_coord /= downsample

                # print(y_coord, type(y_coord))

                y_coord_mask = (y_coord > corner_y) & (y_coord < end_corner_y)
                x_coord_mask = (x_coord > corner_x) & (x_coord < end_corner_x)

                coord_mask = y_coord_mask & x_coord_mask

                ax.plot(
                    (x_coord[coord_mask] - corner_x),
                    (y_coord[coord_mask] - corner_y),
                    ".",
                    markersize=markersize,
                    color=colour,
                    alpha=.9,
                    fillstyle='full',
                    markeredgewidth=0.0,
                )

        # Add scalebar
        # ------------

        scalebar_fontprops = fm.FontProperties(
            size=7, family="Arial", weight="bold"
        )

        bar_pixel_length = 100 / um_per_pixel / downsample

        scalebar = AnchoredSizeBar(
            ax.transData,
            bar_pixel_length,
            r"100 μm", 8,  # lower center, lower left is 3
            pad=0.1,
            color='white',
            frameon=False,
            size_vertical=bar_pixel_length / 10,
            fontproperties=scalebar_fontprops
        )
        ax.add_artist(scalebar)

        # Add gene labels with colour matching spots
        # ------------------------------------------

        if label_text:

            text_offset = 0
            for gene, markersize, colour in genes_list:
                # label genes by colour on top right
                ax.text(
                    text_xcorner, text_ycorner - 0.04 * text_offset, gene,
                    fontproperties=label_fontprops,
                    color=colour,
                    horizontalalignment='left',
                    verticalalignment='top',
                    transform=ax.transAxes
                )
                text_offset += 1

        # Plot scatter plot
        # -----------------

        if counts_filepath is not None:

            df = pd.read_csv(counts_filepath, sep="\t")

            results_fontprops = fm.FontProperties(
                size=9, family="Arial", weight="bold"
            )

            sns.set_style("darkgrid")

            # Decide how to plot the scatterplot
            # ----------------------------------

            if show_scatterplot:

                ax_scatter = inset_axes(
                    ax, width=1.2, height=1, loc="lower left",
                    bbox_to_anchor=(0.08, 0.07),
                    bbox_transform=ax.transAxes,
                    borderpad=0,
                )
                spot_size = 10
                alpha = 1
                is_inset = True

            else:  # on a seperate figure

                ax_scatter = fig2.add_subplot(
                    grid2[dir_num // xgrid_length, dir_num % xgrid_length]
                )
                spot_size = 60
                alpha = 0.7
                is_inset = False

            # Plot scatter plot on given axes
            # -------------------------------

            plotScatter(
                ax_scatter, df,
                "FPKM_data", "spot_count",
                results_fontprops,
                spot_size=spot_size, alpha=alpha,
                is_inset=is_inset,
                background_alpha=1,
                xlim_offset=xlim_offset,
                ylim_offset=ylim_offset,
            )

        # Set limits so that spots do not go beyond image borders
        # -------------------------------------------------------

        print(f"ylim:{ax.get_ylim()}\n"
              f"xlim:{ax.get_xlim()}\n")

        ax.set_xlim(right=img_crop.shape[1] - 0.5)
        ax.set_ylim(bottom=img_crop.shape[0] - 0.5)

    # Adjust figure spacing
    # ---------------------

    fig.subplots_adjust(
        left=0.02, bottom=0.02,
        right=0.99, top=0.95,
        wspace=0.05, hspace=0.05
    )

    # Save the images
    # ---------------

    if fig_savepath is None:
        fig_savepath = data_path

    saveFigure(fig, "fig3", save_format, fig_savepath)
    if not show_scatterplot:
        saveFigure(fig2, "fig2c_scatterplots", save_format, fig_savepath)

    # close canvas
    # ------------

    canvas.close()
    fig.clear()


if __name__ == "__main__":
    # either specify data_path manually
    data_path = None
    # or get it from tkinter dialog box
    if data_path is None:
        root = tk.Tk()
        root.withdraw()
        data_path = filedialog.askdirectory(
            title="Please select folder containing the representative images"
        )
        root.destroy()

    size_big = 3
    size_medium = 2.5
    size_small = 1.5

    # size_kidney = 3
    # size_brain = 3

    # Choose which genes to plot
    # --------------------------
    # NOTE: the x and y coordinates and lengths
    # are with reference to the DOWNSAMPLED image

    subdir_params = {

        # "AML": (
        #     100, 100, 4000, 4000, 0, 80000, 0.96, 0.85,
        #     [("Utrn", size_big, "red"), ]
        # ),

        "brain": (
            900, 50, 5900, 5900, 0, 140000, 0.98, 0.82, 0.002, 5,
            [
                # ("Dock10", size_big, sns.xkcd_rgb["light purple"],),
                ("Map4", size_big, sns.xkcd_rgb["fire engine red"],),
                # Akap6
                # ("Atp1a2", size_big, sns.xkcd_rgb["medium green"]),
                ("Itpr1", size_big, sns.xkcd_rgb["vivid blue"]),
                # ("Grik3", size_big, sns.xkcd_rgb["medium green"]),
                ("Akap6", size_big, sns.xkcd_rgb["vivid green"]),
                ("Brip1", size_big, sns.xkcd_rgb["amber"]),

            ]
        ),

        "kidney": (
            50, 50, 9400, 9400, 0, 100000, 0.96, 0.85, None, 10,
            [
                # ("Arhgef12", size_kidney, sns.xkcd_rgb["light purple"],),
                # ("4932438A13Rik", size_kidney, sns.xkcd_rgb["vivid blue"],),
                ("Ppl", size_big, sns.xkcd_rgb["vivid blue"],),
                # ("Spon1", size_big, sns.xkcd_rgb["lightblue"],),
                ("Sptbn2", size_big, sns.xkcd_rgb["lightblue"],),
                # ("Utrn", size_big, sns.xkcd_rgb["vivid blue"],),
                # ("Acacb", size_big, sns.xkcd_rgb["vermillion"],),
                # ("Acacb", size_big, sns.xkcd_rgb["fire engine red"],),
                ("Irs1", size_big, sns.xkcd_rgb["fire engine red"],),
                ("Notch3", size_big, sns.xkcd_rgb["amber"],),
                ("Osbpl8", size_big, sns.xkcd_rgb["vivid green"],),
            ]
        ),

        "ovary": (
            450, 10, 6600, 6600, 0, 30000, 0.96, 0.85, None, 20,
            [
                ("Rnf213", size_big, sns.xkcd_rgb["light purple"],),
                # ("Foxo1", size_big, "yellow"),
                ("Plxnc1", size_big, "orange",),
                # ("Arhgef28", size_big, "orange"),
                # ("Dock9", size_big, "orange"),
                ("Myh11", size_big, sns.xkcd_rgb["vivid blue"],),
                ("Dsp", size_big, sns.xkcd_rgb["fire engine red"],),
                # ("Daam2", size_big, "purple"),
                ("Slc12a7", size_big, sns.xkcd_rgb["medium green"],),
            ]
        ),

        "liver": (
            100, 100, 6500, 6500, 0, 100000, 0.96, 0.85, 0.002, 2,
            [
                # ("Ppl", size_big, sns.xkcd_rgb["vivid blue"]),
                # ("Tnrc18", size_big, sns.xkcd_rgb["vivid blue"]),
                ("Ahnak", size_big, sns.xkcd_rgb["fire engine red"],),
                # ("Atp1a2", size_big, sns.xkcd_rgb["vivid blue"]),

                ("Flna", size_big, sns.xkcd_rgb["medium green"],),
                ("Atp1a2", size_big, sns.xkcd_rgb["amber"]),
                # ("Hspg2", size_big, sns.xkcd_rgb["medium green"]),
                ("Son", size_big, sns.xkcd_rgb["vivid blue"]),
                # ("Myh11", size_big, "orange",),
                # ("Synm", size_big, sns.xkcd_rgb["light purple"],),
                # ("Myom3", size_brain, "orange",),
            ]
        ),

    }

    # Plot figure
    # -----------

    # Note: when testing this script,
    # use plot_image = False
    # since the stitched files are too big
    # and we are unable to include them
    # it will show just the spots

    makeFig3(
        data_path,
        subdir_params,
        downsample=2,
        fig_savepath=data_path,
        save_format=".png",
        # save_format=".eps",
        label_text=True,
        # plot_image=True,
        plot_image=False,
        # show_scatterplot=True,
        show_scatterplot=False,
    )
