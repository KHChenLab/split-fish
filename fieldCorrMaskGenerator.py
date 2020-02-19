"""
Classes involving field correction

getBackgroundIntensity:
-----------------------
get mean/median intensity from zero-light images

ApplyFieldCorrection:
---------------------
Apply field correction using existing field-correction mask
on an image or images

EstimateFieldCorrection:
------------------------
Estimate field correction from a sets of images (different colours)
in a folder containing image data

nigel - 27 jun 2019

License and readme found in https://github.com/khchenLab/split-fish
"""

import numpy as np
import os
import h5py
import datetime
import pprint as pp

import pandas as pd

from typing import Tuple, List

import skimage.io  # for loading multi-frame ome-tif files
from skimage.util import view_as_blocks
from scipy import optimize

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# for File Dialog Window
import tkinter as tk
from tkinter import filedialog

from fileparsing.filesClasses import getFileParser
from utils.readClasses import readDoryImg, readSpongebobImg


def getBackgroundIntensity(data_path: str,
                           image_type: str = ".tif",
                           verbose: bool = True,
                           ):
    """
    get the mean or median intensity
    from a series of image files in a folder.
    These images must be taken with
    no light incident on the sensor

    :return: (mean, median) intensity values
    """

    images = []

    for file in os.listdir(data_path):
        if os.path.splitext(file)[-1] == image_type:
            full_filepath = os.path.join(data_path, file)

            if image_type == ".tif":
                image = skimage.io.imread(full_filepath)
            elif image_type == ".dax":
                image = readDoryImg(full_filepath)
            else:
                raise TypeError(f"Image type {image_type} not recognised.")

            images.append(image)

    images = np.stack(images, axis=0)

    if verbose:
        print(
            "-" * 40 +
            f"\nNumber of images is {images.shape[0]}."
            f"Image intensities are in range "
            f"({images.min()}, {images.max()})"
            f"Image array has shape {images.shape}"
        )

    return images.mean(), np.median(images.flat)


#
#
# ============================================================================================
#
#


class ApplyFieldCorrection(object):

    def __init__(self,
                 hdf5_filepath: str,
                 zero_light: int = 325,
                 verbose: bool = False,
                 ) -> None:

        self.zero_light = zero_light
        self.verbose = verbose

        # Read masks from hdf5 file into a dictionary of image arrays
        # -----------------------------------------------------------

        self.correctionmask_dict = {}
        self.basetypes = []

        with h5py.File(hdf5_filepath, "r") as f:
            for colour in f:
                self.correctionmask_dict[colour] = np.array(f[colour])
                self.basetypes.append(colour)

        if verbose:
            print(f"...Read hdf file from {hdf5_filepath}:\n")
            pp.pprint(self.correctionmask_dict)

    def plotMasks(self,
                  sns_style: str = "white",
                  ) -> None:
        """
        Visualize all the correction masks
        """
        sns.set_style(sns_style)
        for basetype in self.basetypes:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(self.correctionmask_dict[basetype])
            ax.set_title(f"Correction mask for {basetype}")

    def applyFC(self,
                image: np.ndarray,  # a 2D image array
                basetype: str,  # basetype (e.g. Cy5) to use for correction
                ) -> np.ndarray:
        """
        apply field correction to an image (2d numpy array)
        """

        # Check dimensions of image
        # -------------------------

        if image.ndim != 2:
            raise ValueError(
                f"Image has {image.shape} dimensions. Should be 2.\n"
            )

        # elif all([img_dim == mask_dim
        #           for (img_dim, mask_dim)
        #           in zip(image.shape, self.correctionmask_dict[basetype].shape)]):
        elif image.shape != self.correctionmask_dict[basetype].shape:

            raise IndexError(
                f"Dimensions of mask {self.correctionmask_dict[basetype].shape}\n"
                f"and dimensions of image {image.shape} do not match!\n"
            )

        # Apply correction
        # ----------------

        if self.verbose:
            print(f"Applying mask for {basetype}\n")

        return (image - self.zero_light) / self.correctionmask_dict[basetype]


#
#
# ============================================================================================
#
#


class EstimateFieldCorrection(object):

    def __init__(self,
                 microscope_type: str = "Dory",
                 data_path: str = None,
                 basetypes: list = ("Cy5",),
                 basehyb: int = 0,
                 max_images: int = 40,
                 block_dims: tuple = (16, 16),
                 average_type: str = "median",
                 zero_light: int = 325,
                 ) -> None:
        """

        :param microscope_type:
            either "Dory" or "Spongebob", case insensitive
        :param data_path:
            path where all the image files are stored
        :param basetypes:
             the base type to use for stitching
        :param basehyb:
            the base hyb to use for stitching
        :param max_images:
            max number of images to use for each type
        :param block_dims:
            dowsampling block shape (y,x)
        :param average_type:
            type of averaging over images
        :param zero_light:
            zero-light value for the microscope
        """

        self.microscope_type = microscope_type.lower()
        self.data_path = data_path
        self.basetypes = basetypes
        self.basehyb = basehyb
        self.max_images = max_images
        self.block_dims = block_dims
        self.average_type = average_type
        self.zero_light = zero_light

        self.ydim, self.xdim = None, None  # for recording the dimensions of the images

        print("_" * 90,
              f"\nInitializing Field Correction...\n"
              f"- Microscope type:\t{self.microscope_type}\n"
              f"- basetypes:\t{self.basetypes}\n"
              f"- basehyb:\t{self.basehyb}\n")

        parser = getFileParser(
            self.data_path,
            self.microscope_type,
        )
        self.files_df = parser.files_df
        self.filenames_dict = self.dfToDict_byType(
            self.files_df, verbose=True,
        )

    def dfToDict_byType(self,
                        files_df: pd.DataFrame,
                        verbose: bool = False,
                        ) -> dict:
        """
        reads files dataframe and returns a dictionary with
            keys : basetypes
            values: list of image file names/paths matching that type
        """
        filenames = {}

        for basetype in self.basetypes:

            # filter dataframe for only rows with the correct type
            files_temp = files_df[
                (self.files_df["type"] == basetype) &
                (self.files_df["hyb"] == self.basehyb)
                ]

            # Get list of tuples: (filenames, tiff_frame)
            # for image files matching the basetype and basehyb
            filelist_temp = list(
                files_temp[
                    ["file_name", "tiff_frame"]
                ].itertuples(index=False, name=None)
            )

            if self.max_images is not None:
                if len(filelist_temp) > self.max_images:
                    print(f"\nNumber of images for "
                          f"type {basetype} is "
                          f"{len(filelist_temp)}. \n"
                          f"Truncating to maximum image count: "
                          f"{self.max_images}\n")
                    filelist_temp = filelist_temp[:self.max_images]

            if verbose:
                # print(f"Files dataframe for type {basetype}:\n{files_temp}\n")
                print(f"File list for {basetype}, "
                      f"{len(filelist_temp)} entries:\n")
                pp.pprint(filelist_temp, indent=2)

            filenames[basetype] = filelist_temp

        return filenames

    def _readImageFromFilename(self,
                               image_file,  # a tuple of (filepath, tiff_frame)
                               ):
        """
        adds filename/local filepath to main data-folder path
        then reads the image with the relevent image reader
        """
        image_filepath = os.path.join(self.data_path, image_file[0])

        # __ For Dory/Nemo | .dax file format images __
        if self.microscope_type in ["dory", "nemo"]:
            return readDoryImg(image_filepath)

        # __ For Spongebob | .tiff multiframe file format images __
        elif self.microscope_type == "spongebob":
            return readSpongebobImg(
                image_filepath, image_file[1],  # frame of the tiff
            )

    def _blockReduce(self,
                     image_array: np.ndarray,
                     block_dims: tuple,
                     average_type: str = "median",  # options: "mean","mean_quantile"
                     quantile: Tuple[float, float] = (0.25, 0.75),
                     verbose:bool = True,
                     ) -> np.ndarray:
        """
        return an array containing average across blocks of an image
        (Note: similar to the intended function of skimage block_reduce,
         which doesn't work properly with medians)
        Options:
         (1) Mean
         (2) Median
         (3) Mean of an inter-quantile range, with
             lower and upper quantiles specified by quantile option
        """
        if verbose:
            print(
                f"\nDividing Images into blocks ... \n"
                f"Block dimensions: {block_dims}\n"
                f"Image dimensions: {image_array.shape}\n"
            )

        for block_dim, image_dim in zip(block_dims, image_array.shape):
            assert image_dim % block_dim == 0, (
                f"Image dimension {image_dim} not divisible "
                f"by block dimension {block_dim}!\n"
            )

        view = view_as_blocks(image_array, block_dims)

        # collapse the last 4 dimensions into 1
        # so that the elements of each block are flat
        # The 4 dims to be collapsed are:
        # (1) images dimension (this is a singleton dimension
        #         because we set blockdim's 3rd dimension
        #         to be same as number of images)
        # (2) y dimension of the block
        # (3) x dimension of the block
        # (4) images dimension of the block
        flatten_view = view.reshape(
            view.shape[0], view.shape[1], -1
        )

        if average_type == "mean":
            return np.mean(flatten_view, axis=-1)

        elif average_type == "median":
            return np.median(flatten_view, axis=-1)

        elif average_type == "mean_quantile":
            sorted_view = np.sort(flatten_view, axis=-1)
            start_index = int(quantile[0] * sorted_view.shape[-1])
            end_index = int(quantile[1] * sorted_view.shape[-1])
            return np.mean(sorted_view[:, :, start_index:end_index], axis=-1)

        else:
            raise TypeError(f"Type of averaging: {average_type} not recognised")

    def readImageList(self,
                      image_list: List[tuple],  # a list of (filepath, tiff_frame)
                      return_fullarray: bool = True,
                      verbose: bool = False,
                      ):
        """
        Read the images into a y by x by n_images array
        :returns ndarray
        """

        first_image = self._readImageFromFilename(image_list[0])

        if first_image.ndim == 3:

            zdim, ydim, xdim = first_image.shape
            img_array = None

        elif first_image.ndim == 2:

            ydim, xdim = first_image.shape
            img_array = np.zeros((ydim, xdim, len(image_list)))

        else:

            raise ValueError(f"Image array has {first_image.ndim} dimensions."
                             f"Should be either 2 or 3.")

        for image_num, filename in enumerate(image_list):

            image_temp = self._readImageFromFilename(filename)
            # check that the dimensions of the image match the first image
            assert image_temp.shape[-2] == ydim, (f"image {image_num}: {filename}"
                                                  f"has y dimension {image_temp.shape[-2]}"
                                                  f"which does not match {ydim}")
            assert image_temp.shape[-1] == xdim, (f"image {image_num}: {filename}"
                                                  f"has y dimension {image_temp.shape[-1]}"
                                                  f"which does not match {xdim}")

            if image_temp.ndim == 3:
                if img_array is None:
                    img_array = np.moveaxis(image_temp, 0, -1)
                else:
                    img_array = np.concatenate((img_array,
                                                np.moveaxis(image_temp, 0, -1)),
                                               axis=-1)

            elif image_temp.ndim == 2:
                img_array[:, :, image_num] = image_temp

            if verbose:
                print(f"Read: image {image_num}: {filename}")

        num_images = img_array.shape[-1]

        # subtract zero-light term
        img_array -= self.zero_light

        print(f"New block dimensions: {self.block_dims+(num_images,)}")

        downsampled_array = self._blockReduce(
            img_array,
            self.block_dims + (num_images,),
            average_type=self.average_type,
        )

        if return_fullarray:
            return img_array, downsampled_array, num_images, ydim, xdim
        else:
            del img_array
            return None, downsampled_array, num_images, ydim, xdim

    def _recordDims(self, ydim, xdim):
        """
        if ydim and xdim are not recorded, record dimensions
        to self.ydim and self.xdim
        if not check if ydim and xdim match recorded versions
        """
        # record dimensions and check that all types have same dimensions
        if self.ydim is None:
            self.ydim = ydim
        else:
            assert ydim == self.ydim, (f"y dimension given does not "
                                       f"match recorded y-dimension {self.ydim}")
        if self.xdim is None:
            self.xdim = xdim
        else:
            assert xdim == self.xdim, (f"x dimension given does not "
                                       f"match recorded x-dimension {self.ydim}")

    def readImagesToDict(self,
                         # the dictionary keyed by types,
                         # containing (filenames, tiff_file) for each type
                         verbose=False,
                         ):
        """
        Read the images into a y by x by n_images array
        assumes all images are the same size
        :returns a dictionary of images {fov: image ndarray, ...}
        """
        images_dict, downsampled_dict = {}, {}

        # ____ cycle through images from different FOVs and read them ____
        for basetype in self.filenames_dict:
            if verbose:
                print(f"Reading images of type: {basetype}")

            (img_array,
             downsampled_array,
             num_images,
             ydim,
             xdim) = self.readImageList(self.filenames_dict[basetype],
                                        return_fullarray=True)

            self._recordDims(ydim, xdim)

            images_dict[basetype] = img_array
            downsampled_dict[basetype] = downsampled_array

        return images_dict, downsampled_dict

    def processAll(self,
                   savepath: str,
                   verbose: bool = False,
                   show_fit: bool = True,
                   ) -> str:
        """
        Process all images, estimating each basetype's correction mask.
        Save each correction mask into a hdf5 file.
        Then Delete any image arrays used to estimate the correction.
        :returns full path of the hdf5 file used to save masks
        """
        h5path = os.path.join(
            savepath,
            f"correctionmasks_{'_'.join(self.basetypes)}"
            f"_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.hdf5"
        )

        print(f"...Saving correction masks in:\n   {h5path}")

        with h5py.File(h5path) as f:

            for basetype in self.filenames_dict:

                (_, downsampled_array,
                 num_images,
                 ydim,
                 xdim) = self.readImageList(self.filenames_dict[basetype],
                                            return_fullarray=False)
                self._recordDims(ydim, xdim)

                (correction_mask,
                 params_fullres) = self.correctionMask(downsampled_array,
                                                       show_fit=show_fit,
                                                       plot_title=basetype,
                                                       )

                # save the mask as a dataset of hdf5 file
                dataset = f.create_dataset(basetype,  # key (string)
                                           data=correction_mask,  # data array
                                           )
                dataset.attrs["params"] = params_fullres

                if verbose:
                    print(f"   saved {basetype} as {f[basetype]}")

                # remove array when done
                del downsampled_array

        return h5path

    def dictToHdf5(self,
                   correctionmask_dict,
                   savepath,
                   ):
        h5path = os.path.join(
            savepath, f"correctionmasks_{'_'.join(self.basetypes)}.hdf5"
        )

        try:

            with h5py.File(h5path) as h5file_fieldcorr:
                for basetype in correctionmask_dict:
                    h5file_fieldcorr.create_dataset(
                        basetype, data=correctionmask_dict[basetype]
                    )
            print(f"...Saved hdf file in:\n   {h5path}")
            return h5path

        except:

            print(f"Warning: Failed to save Hdf5 File: {h5path}")
            return None

    def masksFromDict(self,
                      downsampled_dict):
        """
        :return: dictionaries of correction masks and parameters,
                 keyed by basetype (colour)
        """
        correctionmask_dict, params_dict = {}, {}

        for basetype in self.basetypes:
            (
                correctionmask_dict[basetype],
                params_dict[basetype]
            ) = self.correctionMask(downsampled_dict[basetype], )

        return correctionmask_dict, params_dict

    #
    # ------------------------------------------------------------------------------
    #                       Gaussian fitting functions
    # ------------------------------------------------------------------------------
    # Note: a lot of this is adapted from scipy cookbook:
    # https://scipy-cookbook.readthedocs.io/items/FittingData.html
    #
    def correctionMask(self,
                       downsampled_img,
                       show_fit: bool = False,
                       plot_title: str = "",
                       ) -> tuple:
        """
        :returns
        (1) full-resolution correction mask
        (2) parameters for gaussian model on the full-resolution image
        """
        # Gaussian Fit
        params = self.fitgaussian(downsampled_img)
        params_fullres = self.paramsToFullRes(params)

        # _____________  Plot the fit  __________________
        if show_fit:
            self.plotFit(params,
                         downsampled_img,
                         self.gaussian(*params)(
                             *np.indices((downsampled_img.shape[0],
                                          downsampled_img.shape[1]))
                         ),
                         num_slices=6,
                         title=plot_title,
                         save=True, )

        fit_fullres = self.gaussian(*params_fullres)
        correction_mask = fit_fullres(
            *np.indices((self.ydim, self.xdim))
        )

        return correction_mask, params_fullres

    def paramsToFullRes(self, params):
        """
        Convert parameters of gaussian estimated on a
        downsampled image to fit the full image resolution
        Sets the height to 1 since this is only used to scale the image
        """
        (height, center_x, center_y, width_x, width_y) = params

        return (1,
                center_x * self.block_dims[0],
                center_y * self.block_dims[1],
                width_x * self.block_dims[0],
                width_y * self.block_dims[1],)

    @classmethod
    def gaussian(cls, height,
                 center_x, center_y,
                 width_x, width_y,
                 ):
        """
        Returns a gaussian function with the given parameters
        """
        width_x = float(width_x)
        width_y = float(width_y)

        return lambda x, y: height * np.exp(
            -0.5 *
            (((center_x - x) / width_x) ** 2 +
             ((center_y - y) / width_y) ** 2)
        )

    def estimateParams(self,
                       image,
                       fix_center=True,
                       ):

        """
        Rough estimate of (height, x, y, width_x, width_y)
        used as starting guess for least-squares optimization
        """

        height = image.max()

        if fix_center:
            # assume the center of gaussian is dead center of image
            center_x = int(image.shape[1] / 2)
            center_y = int(image.shape[0] / 2)
        else:
            # use moments method to find center
            total = image.sum()
            X, Y = np.indices(image.shape)
            print(f"indicesX =\n{X},\n indices Y =\n{Y}\n")
            center_x = (X * image).sum() / total
            center_y = (Y * image).sum() / total

        col = image[:, int(center_y)]
        width_x = np.sqrt(
            np.abs(
                (np.arange(col.size) - center_y) ** 2 * col
            ).sum()
            / col.sum()
        )

        row = image[int(center_x), :]
        width_y = np.sqrt(
            np.abs(
                (np.arange(row.size) - center_x) ** 2 * row
            ).sum()
            / row.sum()
        )

        return height, center_x, center_y, width_x, width_y

    def fitgaussian(self,
                    image):
        """
        Estimates 2D gaussian
        """
        # x0 :
        # ----
        # starting estimate of the parameters
        params = self.estimateParams(image)
        print(f"\nStarting params: {params}\n")

        # Error Function :
        # ----------------
        # a 1D array of pixel-wise intensity differences
        # between gaussian model and image
        errorfunction = lambda p: np.ravel(
            self.gaussian(*p)(*np.indices(image.shape)) - image
        )

        # Optimize :
        # ----------
        # use Scipy's least-squares optimization to get
        # the best parameters for the gaussian model
        params_optimized, success = optimize.leastsq(errorfunction, params)

        return params_optimized

    #
    # ------------------------------------------------------------------------------
    #                       Plotting functions
    # ------------------------------------------------------------------------------
    #

    def plotFit(self,
                p,  # gaussian parameters
                img,  # combined averaged image
                fit_img,  # image from fitted gaussian params
                num_slices=5,  # number of slices in x/y to show
                title="",  # name of the basetype used
                save=True,
                ):
        """
        plot (1) slices in x and y showing fit
             (2) Contours of fitted gaussian overlaid on image
        """
        gs = gridspec.GridSpec(num_slices, 4)
        fig_fit = plt.figure(figsize=(16, 9))

        # __________  Plot Slices  __________
        sns.set_style("darkgrid")

        x_slice_interval = img.shape[0] // (num_slices + 1)
        y_slice_interval = img.shape[1] // (num_slices + 1)

        # plotting params
        fitline_color = "red"

        # Horizontal slices (y = ...)
        for slice_num in range(num_slices):
            row = int(x_slice_interval * (slice_num + 1))
            ax_temp = fig_fit.add_subplot(gs[slice_num, 0])
            ax_temp.plot(img[row, :])
            ax_temp.plot(fit_img[row, :], color=fitline_color)
            ax_temp.set_title(f"Horizontal slice at y={row:d}")

        # Vertical slices (x = ...)
        for slice_num in range(num_slices):
            col = int(y_slice_interval * (slice_num + 1))
            ax_temp = fig_fit.add_subplot(gs[slice_num, 1])
            ax_temp.plot(img[:, col])
            ax_temp.plot(fit_img[:, col], color=fitline_color)
            ax_temp.set_title(f"Vertical slice at x={col:d}")

        # Plot 2D contours of fit
        # -----------------------

        sns.set_style("white")

        ax2d = fig_fit.add_subplot(gs[:, 2:])

        # plot downsampled image
        ax2d.imshow(img, cmap="hot", )

        # plot contours of gaussian fit
        ax2d.contour(fit_img, cmap="Blues")

        (height, center_x, center_y, width_x, width_y) = p

        ax2d.text(
            0.95, 0.05,
            f"x : {center_x:.1f}\n"
            f"y : {center_y:.1f}\n"
            f"width_x : {width_x:.1f}\n"
            f"width_y : {width_y:.1f}\n"
            f"height : {height:.1f}",
            fontsize=16, fontname='Arial',
            color="w",
            horizontalalignment='right',
            verticalalignment='bottom',
            transform=ax2d.transAxes,
        )

        # Overall Title
        # -------------

        fig_fit.suptitle(
            f"Fit for {title}", size=18, weight="bold",
        )
        fig_fit.tight_layout(rect=(0, 0, 1, 0.95))

        if save_path is not None:
            date_str = datetime.datetime.now().strftime('%Y%m%d_%H%M')

            fig_fit.savefig(
                os.path.join(
                    save_path, f"plot_fit_{title}_{date_str}.png"
                ),
                dpi=400,
            )

    @classmethod
    def plotCorrected(cls,
                      original_img: np.ndarray,
                      corrected_img: np.ndarray,
                      percentile: Tuple[float, float] = (10, 99.5),
                      cmap: str = "hot",
                      ) -> None:
        """
        compare an original and corrected image
        """
        fig_corr, ax_corr = plt.subplots(
            nrows=1, ncols=2, figsize=(12, 8),
        )
        vmax = np.percentile(original_img, percentile[1])
        vmin = np.percentile(original_img, percentile[0])

        # Original
        # --------

        ax_corr[0].imshow(
            original_img, vmin=vmin, vmax=vmax, cmap=cmap,
        )
        ax_corr[0].set_title("Original Image")

        # Corrected
        # ---------

        ax_corr[1].imshow(
            corrected_img, vmin=vmin, vmax=vmax, cmap=cmap,
        )
        ax_corr[1].set_title("Corrected Image")


#
#
# ======================================================================================
#                               Test Script
# ======================================================================================
#
#

if __name__ == "__main__":
    #
    # ask for output directory containing .dax files from all fields of view
    # ----------------------------------------------------------------------

    root = tk.Tk()
    root.withdraw()
    data_path = filedialog.askdirectory(title="Please enter data directory")
    root.destroy()
    save_path = os.path.join(data_path, "field_correction_masks")
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    # Define parameters
    # -----------------

    params = "2"

    if params == "1":
        microscope_type = "Dory"
        basetypes = ["Cy5_Bleach", "Cy7_Bleach"]
        basehyb = 1
        blockdim_x = 32
    else:
        microscope_type = "Dory"
        basetypes = ["Cy5", "Cy7"]
        basehyb = 1
        blockdim_x = 32

    # Set up Field estimation object
    # ------------------------------

    fieldcorr = EstimateFieldCorrection(
        microscope_type=microscope_type,
        data_path=data_path,
        basetypes=basetypes,
        basehyb=basehyb,
        block_dims=(blockdim_x, blockdim_x),
        # average_type="mean_quantile",
        average_type="median",
        max_images=200,
        zero_light=0,
    )

    # images_dict, downsampled_dict = fieldcorr.readImagesToDict()
    #
    # full_img = images_dict[basetypes[0]]
    # downsampled_img = downsampled_dict[basetypes[0]]
    #
    # print(f"Shape of image array: {full_img.shape}\n")
    # print(f"Shape of downsampled image array: "
    #       f"{downsampled_img.shape}\n")

    h5file = fieldcorr.processAll(
        save_path,
        show_fit=True,
    )

    # # test image correction
    # test_img = full_img[:, :, 14]
    #
    # correction_masks, params_dict = fieldcorr.masksFromDict(downsampled_dict)
    # h5file = fieldcorr.saveHdf5(correction_masks, save_path)
    #
    # with h5py.File(h5file, "r") as f:
    #     correction_mask = np.array(f[basetypes[0]])
    #
    # corrected_img = test_img / correction_mask
    #
    # fieldcorr.plotCorrected(test_img,
    #                         corrected_img,
    #                         percentile=(15, 99.99))

    # Use ApplyFieldCorrection to display field variation
    # ---------------------------------------------------

    apply = ApplyFieldCorrection(
        h5file,
        zero_light=0,
    )
    apply.plotMasks()

    plt.show()
