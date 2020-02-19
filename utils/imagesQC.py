"""
functions for diplaying images and associated info e.g histograms & spectral information

nigel jul 2019

License and readme found in https://github.com/khchenLab/split-fish
"""

import os

from typing import Tuple

import numpy as np

from matplotlib.figure import Figure
import PyQt5.QtCore
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigCanvas
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigCanvas
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import seaborn as sns


def showImagesByBit(images: np.ndarray,
                    fig_savepath: str = "",
                    fov_str: str = "",
                    image_info: str = "",
                    figure_grid: Tuple[int, int] = (3, 8),
                    pct_range: Tuple[float, float] = (45, 99.8),
                    equalize_hyb_intensities: bool = False,
                    hyb_ref: int = 0,
                    scale_histogram: bool = False,
                    figure_subplot_fontsize: int = 25,
                    figsize: Tuple[float, float] = (18, 9.5),
                    dpi: int = 500,
                    ) -> None:
    """
    Create a Figure showing images (e.g. raw or filtered images) from each bit
    saves the image directly to the figure savepath, then deletes the figure reference

    Parameters
    ----------
    images: numpy array
        frames by Y by X by num_bits image array
    fig_savepath: str
        directory to save the figure in
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
    ______ Intensity normalization options ______
    equalize_hyb_intensities: bool
        whether to normalize to a single hyb
    hyb_ref: int
        reference hyb to normalize all other hybs against
    scale_histogram: bool
        set x limit of histogram by the upper percentile value
    corrected: bool
        whether the images have been field/distortion corrected
    ______ matplotlib figure options ______
    figure_subplot_fontsize: int
        fontsize of the text indicating each bit
    figsize: 2-tuple
        matplotlib figure size
    dpi: int
        dpi for saving figure
    """

    print(f"\n-- Visualizing images/image histograms for FOV {fov_str} --")

    # the number of bits is the length of the last dimension of the image array
    num_bits = images.shape[-1]

    # set figure size, dpi and initiate a workable fig object
    fig_rawimages = Figure(figsize=figsize, dpi=dpi)
    canvas = FigCanvas(fig_rawimages)
    fig_rawimages.set_canvas(canvas)

    ax_rawimages = {}  # main image axes
    ax_hist = {}  # inset axes for the histogram
    ax_psd = {}  # inset axes for 2D power spectrum

    # vectors for min and max intensities if we are equalizing hyb intensity
    percentile_lower = np.ones(num_bits + 1) * np.percentile(images[:, :, :, hyb_ref],
                                                             pct_range[0])
    percentile_upper = np.ones(num_bits + 1) * np.percentile(images[:, :, :, hyb_ref],
                                                             pct_range[1])

    # the maximum number of subplots that can fit into the figure
    max_subplots = figure_grid[0] * figure_grid[1]

    for bit in range(min(num_bits, max_subplots)):

        ax_rawimages[bit] = fig_rawimages.add_subplot(figure_grid[0],
                                                      figure_grid[1],
                                                      bit + 1)

        #
        # ================================  Main image  =====================================
        #

        frame_temp = images[:, :, :, bit]

        if not equalize_hyb_intensities:  # update with intensity range for each hyb
            percentile_lower[bit] = np.percentile(frame_temp, pct_range[0])
            percentile_upper[bit] = np.percentile(frame_temp, pct_range[1])

        ax_rawimages[bit].imshow(np.max(frame_temp, axis=0), cmap='gray',
                                 vmin=percentile_lower[bit],
                                 vmax=percentile_upper[bit])

        # label each image with the relevant bit
        ax_rawimages[bit].text(0.02, 0.98, f"bit {(bit + 1):d}",
                               fontsize=figure_subplot_fontsize, color='orangered', alpha=0.8,
                               weight='bold', fontname="Arial",
                               horizontalalignment='left', verticalalignment='top',
                               transform=ax_rawimages[bit].transAxes)
        ax_rawimages[bit].axis('off')

        #
        # ================================  Inset histogram  =====================================
        #

        ax_hist[bit] = inset_axes(ax_rawimages[bit], width="40%", height="25%", loc=4)
        # ax_hist[bit].hist(frame_temp.ravel(),
        #                    bins=512,
        #                    range=(0, np.amax(frame_temp)),
        #                    fc='b', ec='b')
        sns.distplot(frame_temp.ravel(), rug=False, kde=False,  # rug_kws={"color": "g"},
                     hist_kws={"histtype": "stepfilled", "linewidth": .3, "alpha": .7, },
                     ax=ax_hist[bit])
        if scale_histogram:
            ax_hist[bit].set_xlim(0, percentile_upper[bit] * 1.2)
        ax_hist[bit].set_yscale("symlog", linthreshy=1)
        ax_hist[bit].axvline(percentile_lower[bit],
                             color='r', alpha=0.6, linewidth=1)
        ax_hist[bit].axvline(percentile_upper[bit],
                             color='r', alpha=0.6, linewidth=1)
        ax_hist[bit].tick_params(axis="y",  # labelleft=False,
                                 colors="orangered", direction="in",
                                 length=2, labelsize=6)
        ax_hist[bit].tick_params(axis="x", colors="orangered",
                                 direction="in", length=2, pad=0.4,
                                 labelsize=6)
        ax_hist[bit].patch.set_alpha(0.4)

        #
        # =======================  Inset 2D power spectral density   ============================
        #

        ax_psd[bit] = inset_axes(ax_rawimages[bit], width="25%", height="25%", loc=3)

        ax_psd[bit].imshow(
            np.log(
                np.abs(
                    np.fft.fftshift(
                        np.fft.fft2(
                            np.squeeze(frame_temp, axis=0))
                    )
                )
            ),
            cmap="hot", alpha=0.7)
        ax_psd[bit].tick_params(axis="y", colors="orangered",
                                length=2, labelsize=5)
        ax_psd[bit].tick_params(axis="x", colors="orangered",
                                direction="in", length=2, pad=0.4,
                                labelsize=5)
        ax_psd[bit].grid(False)

    # print(f"intensity low: {percentile_lower}\nintensity high:{percentile_upper}\n")

    # Overall Figure Title
    # --------------------

    fig_rawimages.suptitle(f"Field of view {fov_str} ({image_info})",
                           color="darkred",
                           fontsize=18,
                           fontname="Arial",
                           weight="bold")

    # Adjust figure spacing
    # ---------------------

    fig_rawimages.subplots_adjust(left=0.02, bottom=0.02,
                                  right=0.99, top=0.95,
                                  wspace=0.04, hspace=0.01)

    # Save or show the images
    # -----------------------

    savename_image_info = image_info.replace(" ", "_")
    filename = f"FOV_{fov_str}_images_{savename_image_info}.png"

    if fig_savepath:
        fig_rawimages.savefig(os.path.join(fig_savepath, filename))
        print(f"   saved images and histograms for FOV {fov_str} ({image_info}) in\n"
              f"   <{fig_savepath}>\n")

        # close the canvas
        # ----------------
        canvas.close()
        fig_rawimages.clear()

    else:

        fig_rawimages.show()
