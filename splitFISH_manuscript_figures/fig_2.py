"""
Figure 2
--------
(1) decoding of AML 12 images
    (image not provided in repository. only spots will be plotted)

and

(2) plotting blanks comparison between Split and conventional
    for brain and liver

Figure 2c is plotted in the figure 3 script.

please see readme on how to set main directory if running this in Spyder.

nigel 15 jan 2020

License and readme found in https://github.com/khchenLab/split-fish
"""

import os
import h5py

from typing import Tuple, Dict, Union, List

import numpy as np
import pandas as pd

import scipy.stats as stats

from matplotlib.figure import Figure
import PyQt5.QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigCanvas

import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

import matplotlib.gridspec as gridspec
import matplotlib.cm as cm

import seaborn as sns

import tkinter as tk
from tkinter import filedialog

from splitFISH_manuscript_figures.figure_shared_functions import saveTable, saveFigure, plotScatter


def makeFig2(data_path: str,
             subdir_params: Dict[
                 str, Tuple[int, int, int, int, int, int, float, float,],
             ],
             fig_savepath: str = "",
             save_format: str = ".png",
             um_per_pixel: float = 0.12,
             figsize: Tuple[float, float] = (9, 9),
             dpi: int = 500,
             label_text: bool = True,
             plot_image: bool = True,
             verbose: bool = True,
             ) -> None:
    """
    plot figure 2(b)

    :param data_path:
        the main directory for figure 2
    :param subdir_params:
        crop and image normalization params for each subdirectory's dataset
    :param fig_savepath:
    :param save_format:
    :param um_per_pixel:
    :param label_text:
        whether to label the image with the subdirectory name
    :param plot_image:
        Whether to plot the image or only the spots
    """
    sns.set_style("darkgrid")

    fig = Figure(figsize=figsize, dpi=dpi)
    canvas = FigCanvas(fig)
    fig.set_canvas(canvas)

    folder_contents = os.listdir(data_path)

    if verbose:
        print(f"Files found in data folder:\n{folder_contents}\n")

    # Configure figure grid
    # --------------------_

    ygrid_length = 2
    xgrid_length = len(subdir_params)

    gs = gridspec.GridSpec(
        ygrid_length, xgrid_length, figure=fig,
        hspace=0.05, wspace=0.05,
    )

    for dir_num, subdir in enumerate(subdir_params):

        subdir_path = os.path.join(data_path, subdir)

        assert os.path.isdir(subdir_path), (
            f"Subdirectory {subdir_path} could not be found."
        )

        # Look for stitched mosaic and spots file
        # ---------------------------------------

        image_filepath = None
        spots_filepath = None
        counts_filepath = None

        for file in os.listdir(subdir_path):
            if file.endswith(".hdf5") and file.startswith("FOV_"):
                if "imagedata" in file:
                    image_filepath = os.path.join(subdir_path, file)
                elif "coord" in file:
                    spots_filepath = os.path.join(subdir_path, file)
            elif file.startswith("allFOVs_counts"):
                counts_filepath = os.path.join(subdir_path, file)

        if plot_image:
            assert image_filepath is not None, (
                f"Could not find stitched file in {subdir_path}!"
            )
        assert spots_filepath is not None, (
            f"Could not find spots file in {subdir_path}!"
        )
        if counts_filepath is None:
            print(f"Could not find counts file in {subdir_path}!")

        # Load and plot image
        # -------------------

        ax = fig.add_subplot(gs[0, dir_num])

        (corner_y, corner_x,
         y_length, x_length,
         vmin, vmax,
         text_ycorner, text_xcorner, markersize,) = subdir_params[subdir]

        end_corner_y = corner_y + y_length
        end_corner_x = corner_x + x_length

        if plot_image:

            with h5py.File(image_filepath, 'r') as f:
                img_array = np.array(f["raw"][0, ...])
                img = np.amax(img_array, axis=2)

            img_crop = img[corner_y:end_corner_y, corner_x:end_corner_x]

            print(
                f"Image crop for subdirectory {subdir}, image {image_filepath}\n"
                f"has dimensions {img_crop.shape}."
            )

        else:  # generate a blank image
            img_crop = np.zeros((y_length, x_length))
            vmin = 0
            vmax = 1

        ax.imshow(img_crop, cmap="gray", vmin=vmin, vmax=vmax)

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
                horizontalalignment="left",
                verticalalignment="top",
                transform=ax.transAxes
            )

        # Plot RNA spots
        # --------------

        with h5py.File(spots_filepath, "r") as f:

            gene_list = f.keys()
            num_genes = len(gene_list)

            # colors = sns.color_palette("bright", num_genes)
            colors = cm.rainbow(np.linspace(0, 1, num_genes))

            for gene, color in zip(gene_list, colors):

                if f[gene].shape[0] == 0:
                    print(f"No RNA spots found for {gene}.\n")
                    continue

                y_coord = f[gene][:, 1]
                x_coord = f[gene][:, 2]

                y_coord_mask = (y_coord > corner_y) & (y_coord < end_corner_y)
                x_coord_mask = (x_coord > corner_x) & (x_coord < end_corner_x)

                coord_mask = y_coord_mask & x_coord_mask

                ax.plot(
                    (x_coord[coord_mask] - corner_x),
                    (y_coord[coord_mask] - corner_y),
                    ".",
                    markersize=markersize,
                    # color=color,
                    mfc=color, mec='none',
                    alpha=.8,
                    # fillstyle='none',
                    markeredgewidth=0.5,
                )

                print(f"plotted {gene} with color: {color}")

        # Add scalebar
        # ------------

        scalebar_fontprops = fm.FontProperties(
            size=7, family="Arial", weight="bold"
        )

        pixel_length_um = 10

        bar_pixel_length = pixel_length_um / um_per_pixel

        scalebar = AnchoredSizeBar(
            ax.transData,
            bar_pixel_length,
            f"{pixel_length_um:d} μm", 3,  # lower center is 8, lower left is 3
            pad=0.1,
            color='white',
            frameon=False,
            size_vertical=bar_pixel_length / 10,
            fontproperties=scalebar_fontprops
        )
        ax.add_artist(scalebar)

        # Set limits so that spots do not go beyond image borders
        # -------------------------------------------------------

        print(f"ylim:{ax.get_ylim()}\n"
              f"xlim:{ax.get_xlim()}\n")

        ax.set_xlim(right=img_crop.shape[1] - 0.5)
        ax.set_ylim(bottom=img_crop.shape[0] - 0.5)

        if counts_filepath is not None:
            # plot correlation to bulk
            # ------------------------

            ax_scatter = fig.add_subplot(gs[1, dir_num])

            df = pd.read_csv(counts_filepath, sep="\t")

            results_fontprops = fm.FontProperties(
                size=12, family="Arial", weight="bold"
            )

            plotScatter(
                ax_scatter, df,
                "FPKM_data", "spot_count",
                results_fontprops,
                spot_size=30,
                alpha=0.6,
                is_inset=False,
            )

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

    saveFigure(fig, "fig2", save_format, fig_savepath)

    # close canvas
    # ------------

    canvas.close()
    fig.clear()


def makeFig2BoxPlot(data_path: str,
                    subdir_list: List[str],
                    counts_dict: Dict[str, Union[int, float]],
                    fig_savepath: str = "",
                    save_format: str = ".png",
                    figsize: Tuple[float, float] = (9, 9),
                    dpi: int = 500,
                    verbose: bool = True,
                    ) -> None:
    sns.set_style("whitegrid")

    fig = Figure(figsize=figsize, dpi=dpi)
    canvas = FigCanvas(fig)
    fig.set_canvas(canvas)

    folder_contents = os.listdir(data_path)

    if verbose:
        print(f"Files found in data folder:\n{folder_contents}\n")

    ax = fig.add_subplot(1, 1, 1)

    boxplot_data = []
    boxplot_labels = []

    for dir_num, subdir in enumerate(subdir_list):

        subdir_path = os.path.join(data_path, subdir)

        assert os.path.isdir(subdir_path), (
            f"Subdirectory {subdir_path} could not be found."
        )

        # Look for counts file
        # --------------------

        counts_filepath = None

        for file in os.listdir(subdir_path):

            if file.startswith("allFOVs_counts"):
                counts_filepath = os.path.join(subdir_path, file)
                counts_column = "spot_count"
                break

            elif file.startswith("summed_counts"):
                counts_filepath = os.path.join(subdir_path, file)
                counts_column = "regions"
                break

        assert counts_filepath is not None, (
            f"Could not find counts file in {subdir_path}!"
        )

        # Get counts for each blank
        # -------------------------

        df = pd.read_csv(counts_filepath, sep="\t")

        print(
            f"Counts dataframe from {counts_filepath} in {subdir} folder:\n{df}"
        )

        blank_mask = df["gene_names"].str.startswith(("blank", "Blank"))

        blank_counts = df[counts_column][blank_mask].values

        blank_counts_percell = blank_counts / counts_dict[subdir]

        print(
            f"Blank Counts : {blank_counts}\n type = {type(blank_counts)}\n"
            f"Blank Counts per-cell: {blank_counts_percell}\n"
        )

        boxplot_data.append(blank_counts_percell)
        boxplot_labels.append(subdir.split("_")[-1])

    from scipy.stats import ttest_ind
    split_data = []
    conv_data = []

    for dataset_num, data in enumerate(boxplot_data):
        label = subdir_list[dataset_num]
        if label.startswith("split"):
            split_data.append(data)
        elif label.startswith("conv"):
            conv_data.append(data)

    print(f"Split data: {split_data}\nConventional data: {conv_data}")

    ttest_matrix = np.zeros((len(conv_data), len(split_data)))

    for conv_num, conv_dataset in enumerate(conv_data):
        for split_num, split_dataset in enumerate(split_data):
            tval, pval = ttest_ind(conv_dataset, split_dataset, equal_var=False)
            ttest_matrix[conv_num, split_num] = pval

    print(f"\n T-test matrix:\n{ttest_matrix}\n\n")

    ax.boxplot(boxplot_data, notch=False, sym="")

    # plot individual points
    # ----------------------

    for dir_num, blanks in enumerate(boxplot_data):
        ax.plot(
            (dir_num + 1) * np.ones_like(blanks), blanks,
            ".", color="blue", alpha=0.7, markersize=3,
        )

    ax.set_xticklabels(boxplot_labels)
    ax.set_yscale("log")

    ax.xaxis.grid(which="major", linewidth=0)
    # ax.set_yscale(
    #     "symlog", linthreshy=0.1, linscaley=0.2
    # )
    boxplot_means = []
    boxplot_medians = []
    boxplot_sems = []

    for organ_num, organ in enumerate(boxplot_labels):
        blank_counts = boxplot_data[organ_num]

        mean = np.mean(blank_counts)
        boxplot_means.append(mean)

        median = np.median(blank_counts)
        boxplot_medians.append(median)

        sem = stats.sem(blank_counts)
        boxplot_sems.append(median)

        print(
            f"{organ} : mean = {mean}, median = {median}, sem = {sem}"
        )

    # Save the images
    # ---------------

    if fig_savepath is None:
        fig_savepath = data_path

    saveFigure(fig, "fig2_boxplot", save_format, fig_savepath)

    # close canvas
    # ------------

    canvas.close()
    fig.clear()


def makeCommonGeneCorrPlot(data_path: str,
                           subdir_pairs: List[Tuple[str, str]],
                           counts_dict: Dict[str, Union[int, float]],
                           fig_savepath: str = "",
                           save_format: str = ".png",
                           # figsize: Tuple[float, float] = (9, 9),
                           dpi: int = 500,
                           verbose: bool = True,
                           ) -> None:
    sns.set_style("whitegrid")

    num_plots = len(subdir_pairs)

    fig = Figure(figsize=(4 * num_plots, 4), dpi=dpi)
    canvas = FigCanvas(fig)
    fig.set_canvas(canvas)

    folder_contents = os.listdir(data_path)

    if verbose:
        print(f"Files found in data folder:\n{folder_contents}\n")

    for pair_num, subdir_pair in enumerate(subdir_pairs):

        df_pair = []

        sns.set_style("darkgrid")

        ax = fig.add_subplot(1, num_plots, pair_num + 1)

        for subdir in subdir_pair:

            subdir_path = os.path.join(data_path, subdir)

            assert os.path.isdir(subdir_path), (
                f"Subdirectory {subdir_path} could not be found."
            )

            # Look for counts file
            # --------------------

            counts_filepath = None

            for file in os.listdir(subdir_path):

                if file.startswith("allFOVs_counts"):
                    counts_filepath = os.path.join(subdir_path, file)
                    counts_column = "spot_count"
                    break

                elif file.startswith("summed_counts"):
                    counts_filepath = os.path.join(subdir_path, file)
                    counts_column = "regions"
                    break

            assert counts_filepath is not None, (
                f"Could not find counts file in {subdir_path}!"
            )

            # Get counts for each blank
            # -------------------------

            df = pd.read_csv(counts_filepath, sep="\t")

            df.set_index("gene_names", inplace=True)
            df[counts_column] /= counts_dict[subdir]
            df_pair.append(df[[counts_column, ]])

            print(
                f"Counts dataframe from {counts_filepath} in {subdir_path} folder:\n{df}"
            )

        shared_df = pd.merge(
            df_pair[0], df_pair[1],
            how='inner', left_index=True, right_index=True,
            suffixes=("_" + subdir_pair[0], "_" + subdir_pair[1]),
        )

        shared_df = shared_df[~shared_df.index.str.contains("blank", case=False, na=False)]

        print(f"Shared genes dataframe:\n{shared_df}")

        saveTable(shared_df, "_".join(subdir_pair), fig_savepath)

        # ax.plot(
        #     shared_df[shared_df.columns[0]], shared_df[shared_df.columns[1]],
        #     ".", color="blue", alpha=0.7, markersize=8,
        # )

        fontprops = fm.FontProperties(
            size=12, family="Arial", weight="bold"
        )

        plotScatter(
            ax, shared_df,
            shared_df.columns[0], shared_df.columns[1],
            fontprops,
            spot_size=10, alpha=0.6,
            is_inset=False,
        )
        xlim = ax.get_xlim()
        print(f"{subdir} xlim = {xlim}")
        ylim = ax.get_ylim()
        print(f"{subdir} xlim = {ylim}")
        min_limits = [max(xlim[0], ylim[0]), max(xlim[1], ylim[1])]

        ax.plot(
            min_limits, min_limits,
            # [0, max(xlim[1], ylim[1])], [0, max(xlim[1], ylim[1])],
            "b-", linewidth=3, alpha=0.6,
        )

        ax.set_ylabel(subdir_pair[1])
        ax.set_xlabel(subdir_pair[0])

    # Save the images
    # ---------------

    if fig_savepath is None:
        fig_savepath = data_path

    saveFigure(fig, "shared_genes_plot", save_format, fig_savepath)

    # close canvas
    # ------------

    canvas.close()
    fig.clear()


if __name__ == "__main__":

    counts_dict = {

        "conventional_AML": 1382,
        "conventional_brain": 2729,
        "conventional_liver": 2581,

        "split_AML": 789,
        # "split_AML": 853,
        "split_brain": 4043,
        "split_kidney": 26001,
        "split_liver": 7484,
        "split_ovary": 13405,

    }

    # get Data folder
    # ---------------

    # either specify data_path manually
    data_path = None
    # or get from dialog box
    if data_path is None:
        root = tk.Tk()
        root.withdraw()
        data_path = filedialog.askdirectory(
            title="Please select folder containing the representative images"
        )
        root.destroy()

    # Choose which genes to plot
    # --------------------------

    subdir_params = {

        "split_AML": (
            550, 1150, 400, 400, 0, 100000, 0.98, 0.82, 8,
        ),

    }

    # Plot figure
    # -----------

    makeFig2(
        data_path,
        subdir_params,
        fig_savepath=data_path,
        figsize=(4, 8),
        save_format=".png",
        # save_format=".eps",
        plot_image=False,
        label_text=True,
    )

    organ_list = [
        "conventional_brain",
        "split_brain",
        "conventional_liver",
        "split_liver",
    ]

    makeFig2BoxPlot(
        data_path,
        organ_list,
        counts_dict,
        figsize=(3, 3),
        fig_savepath=data_path,
        save_format=".png",
        # save_format=".eps",
    )

    makeCommonGeneCorrPlot(
        data_path,
        [("conventional_AML", "split_AML",), ],
        counts_dict,
        fig_savepath=data_path,
        save_format=".png",
        # save_format=".eps",
    )
