"""
Functions that accept 2 ImageData objects
and perform comparison or subtraction

nigel 12 Nov 19

License and readme found in https://github.com/khchenLab/split-fish
"""

import os
import numpy as np

from typing import Tuple, List, Dict, Union

# from skimage.feature import register_translation
from utils.registrationFunctions import register_translation
from scipy import ndimage

from data_objects.imageData import ImageData

#
# Define custom types
# -------------------
#
RegerrorDict = Dict[int, Dict[str, float]]


def subtractBackground(imgdata1: ImageData,
                       imgdata2: ImageData,
                       stage_to_compare: Union[str, None] = None,
                       subtract_only_bits: List[int] = None,
                       dont_subtract_bits: List[int] = None,
                       upsampling_for_registration: int = 50,
                       ) -> Tuple[np.ndarray,
                                  RegerrorDict]:
    """
    subtract background of second array from first array,
    after registering images from each bit

    assigns subtracted output to
    first ImageData's 'background_removed' array
    """

    imgdata1.checkDims(imgdata2.frames,
                       imgdata2.y_pix,
                       imgdata2.x_pix,
                       imgdata2.num_bits,
                       info_str="second image array")

    # choose the most corrected stage present in both
    # -----------------------------------------------

    if stage_to_compare is None:
        if imgdata1.getFlag("chrcorr") and imgdata2.getFlag("chrcorr"):
            stage_to_compare = "chrcorr"
        elif imgdata1.getFlag("fieldcorr") and imgdata2.getFlag("fieldcorr"):
            stage_to_compare = "fieldcorr"
        elif imgdata1.getFlag("raw") and imgdata2.getFlag("raw"):
            stage_to_compare = "raw"
        else:
            raise ValueError("no raw or corrected images found in both ImageData")

    array1 = imgdata1.copyArray(stage_to_compare)
    array2 = imgdata2.copyArray(stage_to_compare)

    # Set up arrays to store shifts
    # -----------------------------

    # number of dimensions for registration.
    # should be 2 for 2D images. May be 3 in future with image stacks
    ndims_register = 2
    shifts = np.zeros((imgdata1.num_bits, ndims_register))

    regerror_dict = {}

    if subtract_only_bits is not None:

        valid_bits = [bit for bit in subtract_only_bits
                      if bit not in imgdata1.dropped_bits]

    elif dont_subtract_bits is not None:

        valid_bits = [bit for bit in range(imgdata1.num_bits)
                      if bit not in imgdata1.dropped_bits
                      and bit not in dont_subtract_bits]

    else:

        valid_bits = [bit for bit in range(imgdata1.num_bits)
                      if bit not in imgdata1.dropped_bits]

    # Register and align each bit
    # ---------------------------

    for bit in valid_bits:
        # 2D slice from ImageData 1 array
        array1_slice = array1[0, :, :, bit]
        array1_slice_fourier = np.fft.fftn(array1_slice)

        # 2D slice from ImageData 2 array
        array2_slice = array2[0, :, :, bit]
        array2_slice_fourier = np.fft.fftn(array2_slice)

        (shift,
         fine_error,
         pixel_error) = register_translation(array1_slice_fourier,
                                             array2_slice_fourier,
                                             upsample_factor=upsampling_for_registration,
                                             space="fourier")

        shifts[bit, :] = shift
        regerror_dict[bit] = {"fine error": fine_error,
                              "pixel error": pixel_error}

        # Align and subtract slices
        # -------------------------

        array2_registered_slice = np.fft.ifftn(
            ndimage.fourier_shift(array2_slice_fourier, shift)
        )

        array1[0, :, :, bit] -= array2_registered_slice.real

    # Set the Arrays in first ImageData object
    # ----------------------------------------

    imgdata1.setArray("background_removed", array1)

    return shifts, regerror_dict


#
#
#                                               Script
# ------------------------------------------------------------------------------------------------------------
#


if __name__ == "__main__":

    # imports for test
    # ----------------
    from fileparsing.filesClasses import getFileParser
    import tkinter as tk
    from tkinter import filedialog

    #
    #                                 Test main class
    # --------------------------------------------------------------------------------
    #

    root = tk.Tk()
    root.withdraw()
    data_path = filedialog.askdirectory(title="Please select data directory")
    root.destroy()

    # Set microscope type and choice of roi
    # -------------------------------------

    microscope_type = "Dory"

    params = {}

    if microscope_type == "Dory":

        fovs = [0, 1, 2]

        stage_pixel_matrix = 8 * np.array([[0, -1], [-1, 0]])

        # params["num_bits"] = 16
        params["num_bits"] = 26
        params["hyb_list"] = list(range(9)) * 2 + list(range(8))
        params["type_list"] = ["Cy7", ] * 9 + ["Cy5", ] * 9 + ["Cy3", ] * 8
        params["hyb_list_background"] = list(range(9)) * 2 + list(range(8))
        params["type_list_background"] = ["Cy7_Bleach", ] * 9 \
                                         + ["Cy5_Bleach", ] * 9 \
                                         + ["Cy3_Bleach", ] * 8
        params["roi"] = None

    # Print some of the params
    # ------------------------
    print(f"Hyb list: {params['hyb_list']}\n",
          f"Type list: {params['type_list']}\n")

    # get the file parser
    # -------------------
    myparser = getFileParser(data_path,
                             microscope_type,
                             use_existing_filesdata=True)

    files_dict = myparser.dfToDict(
        fovs,
        roi=params["roi"],
        num_bits=params["num_bits"],  # number of bits to go until
        hyb_list=params["hyb_list"],  # list of hyb numbers for each bit
        type_list=params["type_list"],  # list of filename types
        verbose=True,
    )
    files_dict_background = myparser.dfToDict(
        fovs,
        roi=params["roi"],
        num_bits=params["num_bits"],  # number of bits to go until
        hyb_list=params["hyb_list_background"],  # list of hyb numbers for each bit
        type_list=params["type_list_background"],  # list of filename types
        verbose=True,
    )

    ydim = int(myparser.files_df["ydim"].values[0])
    xdim = int(myparser.files_df["xdim"].values[0])

    save_dir = os.path.join(data_path, "imagedata_test")
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    for fov in files_dict:
        with ImageData(fov, save_dir, x_pix=xdim, y_pix=ydim) as imagedata:
            imagedata.readFiles(data_path, files_dict[fov],
                                microscope_type=microscope_type,
                                )
            imagedata.showImages("raw",
                                 fig_savepath=save_dir,
                                 figure_grid=(4, 7))
            imagedata.printStagesStatus("after raw input")

            with ImageData(fov, save_dir, x_pix=xdim, y_pix=ydim) as imagedata_background:
                imagedata_background.readFiles(
                    data_path, files_dict_background[fov],
                    microscope_type=microscope_type,
                )
                imagedata_background.showImages("raw",
                                                fig_savepath=save_dir,
                                                figure_grid=(4, 7),
                                                additional_info_str=" background")

                shifts, error = subtractBackground(imagedata,
                                                   imagedata_background)

                print(f"Shifts:\n{shifts}\n\n"
                      f"Error:\n{error}")

                imagedata.showImages("background_removed",
                                     fig_savepath=save_dir,
                                     figure_grid=(4, 7))
                imagedata.printStagesStatus("after background removed")
