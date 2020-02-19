"""
Make or load a butterworth band-pass filter
for frequency space filtering

nigel - updated 3 dec 19

License and readme found in https://github.com/khchenLab/split-fish
"""

import os
import numpy as np

from typing import Union

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def butter2d(low_cut: Union[float, None] = None,
             high_cut: Union[float, None] = None,
             order: int = 1,
             filter_path: str = None,
             use_existing: bool = True,
             save: bool = True,
             image: np.ndarray = None,
             ydim: int = None,
             xdim: int = None,
             plot_filter: bool = False,
             plot_impulse_response: bool = False,
             verbose: bool = True,
             ) -> np.ndarray:
    """
    returns 2D butterworth filter centered around 0

    low_cut:
        lower frequency cut - acts as high pass filter
    high_cut:
        higher frequency cut - acts as low pass filter
    filter_path: str
        directory where filters are saved / will be saved
    use_existing: bool
        whether to use existing filter (if found)
    save: bool
        whether to save filter in data_path for future use
    order: int
        order of the butterworth filter
    image (2D ndarray), ydim, xdim:
        get dimension of image from the image provided, or
        directly specify y dimension (ydim) and x dimension (xdim)
    plot_filter:
        whether to show a plot of the filter
    plot_impulse_response:
        whether to plot the impulse response of the filter
    """

    # Create a descriptive name for the filter from provided parameters
    # -----------------------------------------------------------------

    filename = f"butter2d_order{order}"

    if low_cut is not None:
        filename += f"_lowcut_{low_cut}"

    if high_cut is not None:
        filename += f"_highcut_{high_cut}"

    filename += f"_{ydim}_{xdim}.npy"

    # Check if we already have the correct filter saved. 
    # If found, use it instead of computing from scratch

    if filter_path is not None:

        fullpath = os.path.join(filter_path, filename)

        if verbose:
            print("Filter filepath:", fullpath)

        if use_existing and os.path.isfile(fullpath):
            frequency_mask = np.load(fullpath)

            if verbose:
                print(f"\nExisting filter found: {filename}. Using existing...\n")

            return frequency_mask

    # Get dimensions of image
    # -----------------------

    if image is not None and len(image.shape) == 2:
        ydim, xdim = image.shape
    else:
        assert ydim is not None, f"y dimension not provided!"
        assert xdim is not None, f"x dimension not provided!"

    # Initialize arrays
    # -----------------

    high_cut_mask = np.ones((ydim, xdim), dtype=np.float64)
    low_cut_mask = np.ones_like(high_cut_mask)

    # find mid-point (must -0.5 to match pixel center position)
    y_mid, x_mid = ydim / 2 - 0.5, xdim / 2 - 0.5

    if verbose:
        print(f"Mid point of filter: y = {y_mid:.3f}, x = {x_mid:.3f}")

    grid = np.mgrid[0:ydim, 0:xdim]  # grid[0] is y-coordinate, grid[1] is x-coordinate

    distance_to_mid = np.sqrt((grid[0] - y_mid) ** 2 + (grid[1] - x_mid) ** 2)

    if high_cut is not None:
        # no need to worry about dividing by 0 because we are not dividing by the distance-to-mid
        high_cut_mask = 1 / np.sqrt(
            1 + (np.sqrt((grid[0] - y_mid) ** 2 + (grid[1] - x_mid) ** 2) / high_cut) ** (2 * order)
        )

    if low_cut is not None:
        # the following is done to prevent divide by 0 errors
        # right at the center where distance to mid is 0

        where_to_operate = distance_to_mid != 0
        zeros_array = np.zeros_like(low_cut_mask)

        omega_fraction = np.divide(low_cut,
                                   distance_to_mid,
                                   out=zeros_array,
                                   where=where_to_operate)

        # print(np.argwhere(np.logical_or(np.isnan(omega_fraction), np.isinf(omega_fraction))))

        low_cut_mask = np.divide(low_cut_mask,  # an array of ones
                                 np.sqrt(1 + omega_fraction ** (2 * order)),
                                 out=zeros_array,
                                 where=where_to_operate)

        # original equation:
        # low_cut_mask = 1 / np.sqrt(
        #     1 + (low_cut / np.sqrt((grid[0] - y_mid) ** 2 + (grid[1] - x_mid) ** 2)) ** (2 * order))

    frequency_mask = high_cut_mask * low_cut_mask

    # check for funny values
    # print(np.argwhere(np.logical_or(np.isnan(frequency_mask), np.isinf(frequency_mask))))
    # frequency_mask = np.nan_to_num(frequency_mask, copy=False)

    # save filter
    # -----------

    if save:
        if filter_path is None:
            print("Filter path not provided. Not saving filter")
        else:
            if not os.path.isdir(filter_path):
                os.mkdir(filter_path)
            fullpath = os.path.join(filter_path, filename)
            print(f"Saving file as: {fullpath}")
            np.save(fullpath, frequency_mask)

    #
    # Plot the filter and/or impulse response, with a colorbar
    # --------------------------------------------------------
    #

    def _plotTitle(start_str: str,
                   array: np.ndarray,
                   high_cut, low_cut,
                   ) -> str:

        title = start_str + f" of order {order} ({ydim} x {xdim}). "

        if high_cut is not None:
            title += f"High cut = {high_cut}. "
        if low_cut is not None:
            title += f"Low cut = {low_cut}. "

        min_value = np.min(array)
        max_value = np.max(array)
        minmax_str = f"\nMin value: {min_value:0.3f}, Max value: {max_value:0.3e}"

        print(start_str + " has ... " + minmax_str)

        title += minmax_str

        return title

    if plot_filter:
        figfilter, ax = plt.subplots(figsize=[8, 8])
        filterplot = ax.imshow(frequency_mask, cmap="hot")

        # colorbar
        # --------

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.1)
        figfilter.colorbar(filterplot, cax=cax)

        figfilter.suptitle(
            _plotTitle("2D butterworth filter",
                       frequency_mask,
                       high_cut, low_cut),
            fontname="arial", fontsize=14, fontweight="bold"
        )
        figfilter.tight_layout(pad=1.5)

    if plot_impulse_response:
        figir, axir = plt.subplots(figsize=[8, 8])
        size_in_pix = 5
        irplot = axir.imshow(
            np.fft.fftshift(np.fft.ifft2(frequency_mask)).real
            [int(y_mid) + 1 - size_in_pix:int(y_mid) + 2 + size_in_pix,
            int(x_mid) + 1 - size_in_pix:int(x_mid) + 2 + size_in_pix],
            cmap="hot")

        # colorbar
        # --------

        divider = make_axes_locatable(axir)
        cax = divider.append_axes("right", size="4%", pad=0.1)
        figir.colorbar(irplot, cax=cax)

        figir.suptitle(
            _plotTitle("Impulse Response of Freq filter",
                       frequency_mask,
                       high_cut, low_cut),
            fontname="arial", fontsize=14, fontweight="bold"
        )
        figir.tight_layout(pad=1.5)

    return frequency_mask


if __name__ == "__main__":

    #
    # Test the filter
    # ---------------
    #

    freq_filter = butter2d(
        400, 900,
        save=False,
        order=2,
        xdim=2048,
        ydim=2048,
        plot_filter=True,
        plot_impulse_response=True,
    )

    show_images = False
    # show_images = True

    if show_images:
        #
        # Show effect of filter on image (use .dax image to test)
        # -------------------------------------------------------
        #

        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        data_path = filedialog.askopenfilename(title="Please select file")
        root.destroy()

        # dir_path = os.path.dirname(os.path.realpath(__file__))

        # Read Image
        # ----------

        from utils.readClasses import DaxRead

        readfile = DaxRead(
            filename=data_path,
            x_pix=2048, y_pix=2048,
        )

        im = readfile.maxIntensityProjection().squeeze()
        print(f"Image shape is {im.shape}")

        # ____ FFT ____
        imfft = np.fft.fftshift(np.fft.fft2(im))
        # ____ Image (FFT and IFFT back) ____
        im_transback = np.fft.ifft2(np.fft.fft2(im))
        # ____ Filtered FFT ____
        imfft_after_filter = imfft * freq_filter
        # ____ Image (FFT, Filter and IFFT back) ____
        im_after_filter = np.fft.ifft2(np.fft.fftshift(imfft_after_filter))

        # ======================================================================
        #  Figures of image, freqency space image and filtered (image and freq)
        # ======================================================================

        # set up figure
        fig = plt.figure(figsize=[16, 9.2])
        num_rows, num_cols = 2, 3
        axes = fig.add_subplot(num_rows, num_cols, 1)

        # ____ Image ____
        implot = axes.imshow(
            im,
            vmax=np.percentile(im.flat, 99.8),
            vmin=np.percentile(im.flat, 40),
            cmap="gray"
        )

        # ____ FFT ____
        axes = fig.add_subplot(num_rows, num_cols, 2)
        fft_vmax = np.percentile(imfft.real.flat, 99)
        fft_vmin = np.percentile(imfft.real.flat, 40)
        fftplot = axes.imshow(imfft.real, vmax=fft_vmax, vmin=fft_vmin, cmap="hot")

        # ____ Image (FFT and IFFT back) ____
        axes = fig.add_subplot(num_rows, num_cols, 3)
        im_transback_plot = axes.imshow(
            im_transback.real,
            vmax=np.percentile(im_transback.real.flat, 99.8),
            vmin=np.percentile(im_transback.real.flat, 40),
            cmap="gray"
        )

        # ____ Filter ____
        axes = fig.add_subplot(num_rows, num_cols, 4)
        filter_plot = axes.imshow(
            np.fft.ifft2(freq_filter).real, cmap="hot",  # vmax=1, vmin=0,
        )

        # ____ Filtered FFT ____
        axes = fig.add_subplot(num_rows, num_cols, 5)
        fftafterfilter_plot = axes.imshow(
            imfft_after_filter.real,
            vmax=fft_vmax, vmin=fft_vmin, cmap="hot"
        )

        # ____ Image (FFT, Filter and IFFT back) ____
        axes = fig.add_subplot(num_rows, num_cols, 6)
        imafterfilter_plot = axes.imshow(
            im_after_filter.real,
            vmax=np.percentile(im_after_filter.real.flat, 99.8),
            vmin=np.percentile(im_after_filter.real.flat, 10),
            cmap="gray"
        )

        fig.subplots_adjust(
            wspace=0.02, hspace=0.1,
            top=0.98, bottom=0.04,
            left=0.05, right=0.95,
        )

    plt.show()
