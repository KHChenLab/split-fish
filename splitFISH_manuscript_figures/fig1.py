"""
Figure 1
--------

comparison of representative images between
split probe and conventional
in: (1) AML
    (2) brain
add scale bar given pixel dimensions in um
plots histograms of each image

check readme on how to set main directory if running this with Spyder

Nigel Jan 2020

License and readme found in https://github.com/khchenLab/split-fish
"""

import os

from typing import Tuple, Dict

import numpy as np

from matplotlib.figure import Figure
import PyQt5.QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigCanvas
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.patches as patches

import matplotlib.gridspec as gridspec

import seaborn as sns

import tkinter as tk
from tkinter import filedialog

from utils.readClasses import readDoryImg
from splitFISH_manuscript_figures.figure_shared_functions import saveFigure


def makeComparisonFigure(data_path: str,
                         subdir_dict: Dict[str, Tuple[
                             Tuple[int, int, int, int, int, int],
                             Tuple[int, int, int, int, int, int],
                         ]],
                         fig_savepath: str = "",
                         save_format: str = ".png",
                         um_per_pixel: float = 0.12,
                         pct_range: Tuple[float, float] = (45, 99.8),
                         scale_histogram: bool = True,
                         figsize: Tuple[float, float] = (9, 12),
                         dpi: int = 500,
                         label_text: bool = True,
                         verbose: bool = True,
                         ) -> None:
    """
    Make a figure comparing representative raw images

    Parameters
    ----------
    data_path: str
        main data path with subdirectories containing
        images from different cell or tissue types
    subdir_dict
        crop (upper,left, ydim,xdim) and
        intensity normalization (lower limit, upper limit)
        parameters for the pair of images in each subdirectory
    """

    # set figure size, dpi and initiate fig object
    # --------------------------------------------

    sns.set_style("darkgrid")

    fig = Figure(figsize=figsize, dpi=dpi)
    canvas = FigCanvas(fig)
    fig.set_canvas(canvas)

    folder_contents = os.listdir(data_path)

    if verbose:
        print(f"Files found in data folder:\n{folder_contents}\n")

    subdirs = [subdir for subdir in list(subdir_dict.keys())
               if subdir in folder_contents]

    # specify dimensions of figure grid
    # ---------------------------------

    ygrid_length = len(subdirs)
    xgrid_length = 2

    outer_grid = gridspec.GridSpec(
        ygrid_length, xgrid_length, figure=fig,
        hspace=0.1, wspace=0.1,
    )

    # axes = []

    for dir_num, subdir in enumerate(subdirs):

        fullpath = os.path.join(data_path, subdir)

        # look for image files
        # --------------------

        split_filelist = []
        conv_filelist = []

        for file in os.listdir(fullpath):
            if file.endswith("split.dax"):
                split_filelist.append(file)
            elif file.endswith("conv.dax"):
                conv_filelist.append(file)

        assert len(split_filelist) == 1, (f"more than one split image file found."
                                          f"\nFiles:\n{split_filelist}")
        assert len(conv_filelist) == 1, (f"more than one conventional image file found."
                                         f"\nFiles:\n{conv_filelist}")

        files = [conv_filelist[0], split_filelist[0]]
        plot_txt = ["conventional", "split-probe", ]
        medians = []

        for img_num, img_file in enumerate(files):

            # ax = fig.add_subplot(ygrid_length, xgrid_length,
            #                      2 * dir_num + 1 + img_num)
            # axes.append(ax)

            gs = gridspec.GridSpecFromSubplotSpec(
                2, 1,
                # width_ratios=[16, 1],
                height_ratios=[4, 1],
                subplot_spec=outer_grid[dir_num, img_num],
                hspace=0.05, wspace=0.05,
            )
            img_ax = fig.add_subplot(gs[0, :])
            # cbar_ax = fig.add_subplot(gs[0, 1])
            hist_ax = fig.add_subplot(gs[1, :])

            # Read, crop and plot image
            # -------------------------

            img = readDoryImg(os.path.join(fullpath, img_file))

            (corner_y, corner_x,
             y_length, x_length,
             vmin, vmax) = subdir_dict[subdir][img_num]

            img_crop = img[corner_y:corner_y + y_length, corner_x:corner_x + x_length]

            print(f"Image crop for subdirectory {subdir}\n"
                  f"for image {img_file}\n"
                  f"has dimensions {img_crop.shape}.")

            if vmin is None:
                vmin = np.percentile(img_crop, pct_range[0])
            if vmax is None:
                vmax = np.percentile(img_crop, pct_range[1])

            img_plot = img_ax.imshow(
                img_crop, cmap='gray', vmin=vmin, vmax=vmax,
            )
            img_ax.axis("off")

            # inset image
            # -----------

            inset_img_ax = inset_axes(
                img_ax, width="25%", height="25%", loc=3,
            )
            inset_img_plot = inset_img_ax.imshow(
                img, cmap='gray', vmin=vmin, vmax=vmax,
            )
            inset_img_ax.set_xticks([], [])
            inset_img_ax.set_yticks([], [])

            for alpha, ec, fc, lw, fill in [
                (0.3, None, "darkred", 0, True),  # patch
                (1, "darkred", None, 0.5, False),  # border
            ]:
                roi_box = patches.Rectangle(
                    (corner_x, corner_y),
                    x_length, y_length,
                    alpha=alpha, ec=ec, fc=fc, lw=lw, fill=fill,
                    linestyle="--",
                    visible=True,
                )
                inset_img_ax.add_patch(roi_box)

            # add histogram
            # -------------

            # divider = make_axes_locatable(ax)
            # hist_ax = divider.append_axes(
            #     "bottom", size="20%", pad=2,
            # )

            # choose to use the whole image or just the crop
            # ----------------------------------------------

            img_for_hist = img
            # img_for_hist = img_crop

            hist_ax.hist(
                img_for_hist.ravel(),
                bins=800, range=(0, np.max(img_for_hist)),
                histtype="stepfilled",
                alpha=0.7, linewidth=0.2
            )

            if scale_histogram:
                # hist_ax.set_xlim(0, max(np.max(img_for_hist), vmax))
                hist_ax.set_xlim(0, 65535)

            hist_ax.set_yscale("symlog", linthreshy=1)
            # hist_ax.axvline(vmin, color='r', alpha=0.4, linewidth=2)
            hist_ax.axvline(vmax, color='r', alpha=0.4, linewidth=2)

            # add median line
            median_intensity = np.median(img_for_hist)
            assert median_intensity.ndim < 2
            hist_ax.axvline(median_intensity, color='purple', alpha=0.4, linewidth=1)

            medians.append(median_intensity)

            hist_ax.tick_params(
                labelsize=8, labelrotation=0, length=0, pad=1,
                # labelleft=False,
            )

            # axis labels
            hist_fontprops = fm.FontProperties(
                size=10, family="Arial", weight="bold"
            )
            hist_ax.set_ylabel(
                "log counts", labelpad=0, font_properties=hist_fontprops
            )
            hist_ax.set_xlabel(
                "intensity", font_properties=hist_fontprops
            )
            hist_ax.set_xticks(hist_ax.get_xticks()[::2])

            # add colorbar
            # ------------

            # divider2 = make_axes_locatable(ax)
            # cbar_ax = divider2.append_axes(
            #     "right", size="4%", pad=0.08,
            # )

            cbar_ax = inset_axes(img_ax, width="3%", height="20%", loc=4)

            tickvalue_list = [vmin, vmax]
            tickvalue_str_list = []

            for tickvalue in tickvalue_list:

                tickvalue_str = str(tickvalue)
                if tickvalue_str.endswith("000"):
                    tickvalue_str = tickvalue_str[:-3] + "k"
                tickvalue_str_list.append(tickvalue_str)

            fig.colorbar(
                img_plot, cax=cbar_ax, ticks=tickvalue_list,
            )
            cbar_ax.set_yticklabels(tickvalue_str_list)

            cbar_ax.yaxis.set_ticks_position('left')

            cbar_ax.tick_params(
                labelsize=12, labelrotation=0, length=0, pad=1, labelcolor="darkorange",
            )

            # add scalebar
            # ------------

            scalebar_fontprops = fm.FontProperties(
                size=12, family="Arial", weight="bold"
            )

            bar_pixel_length = 5 / um_per_pixel

            scalebar = AnchoredSizeBar(
                img_ax.transData,
                bar_pixel_length,
                r"10 μm", 8,  # lower center, lower left is 3
                pad=0.1,
                color='white',
                frameon=False,
                size_vertical=bar_pixel_length / 10,
                fontproperties=scalebar_fontprops
            )
            img_ax.add_artist(scalebar)

            # add label text
            # --------------

            if label_text:
                label_fontprops = fm.FontProperties(
                    size=20, family="Arial", weight="bold"
                )

                # label each image on top right corner
                label_text = subdir.replace("_", " ") + " " + plot_txt[img_num]
                img_ax.text(
                    0.98, 0.98, label_text,
                    fontproperties=label_fontprops,
                    color="darkorange",
                    horizontalalignment='right', verticalalignment='top',
                    transform=img_ax.transAxes
                )

        print(
            f"Medians: {medians}\n Median ratio for {subdir} = {medians[1]/medians[0]}\n"
        )

    # Adjust figure spacing
    # ---------------------

    fig.subplots_adjust(left=0.02, bottom=0.02,
                        right=0.99, top=0.95,
                        wspace=0.05, hspace=0.05)

    filename = f"fig1_comparison" + save_format

    # Save the images
    # ---------------

    if fig_savepath is None:
        fig_savepath = data_path

    saveFigure(fig, "fig1", save_format, fig_savepath)

    # close the canvas
    # ----------------

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

    tissuedir_dict = {
        # "AML": ((800, 1500, 400, 400, 0, 15000),
        #         (420, 800, 400, 400, 0, 15000)),
        # "mouse_brain": ((500, 500, 400, 400, 0, 55000),
        #                 (200, 400, 400, 400, 0, 55000)),
        # "AML": ((1000, 600, 400, 400, 0, 30000),
        #         (620, 400, 400, 400, 0, 30000)),
        "AML": ((1000, 600, 400, 400, 0, 30000),
                (550, 1150, 400, 400, 0, 30000)),
        "mouse_brain": ((50, 1000, 400, 400, 0, 30000),
                        (150, 450, 400, 400, 0, 30000)),
    }
    # 100, 650

    makeComparisonFigure(
        data_path, tissuedir_dict,
        fig_savepath=data_path,
        save_format=".png",
        # save_format=".eps",
        label_text=False,
    )
