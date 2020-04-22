"""
Functions for correcting: 
(1) Chromatic distortion
(2) uneven illumination field
refactored from old ImageData class

Nigel 4 Nov 19

License and readme found in https://github.com/khchenLab/split-fish
"""

import os
import csv
import re

from typing import Dict

import numpy as np

from skimage.transform import warp, AffineTransform  # for chromatic distortion correction

from fieldCorrMaskGenerator import ApplyFieldCorrection

from data_objects.imageData import ImageData


#
#
# ----------------------------------------------------------------------------------------------------
#                                     Chromatic Distortion Correction
# ----------------------------------------------------------------------------------------------------
#
#

def correctDistortion(imgdata: ImageData,
                      colour_mapping: Dict[str, str],
                      reference_colour: str,
                      calibration_path: str,
                      verbose: bool = True,
                      ) -> None:
    """
    correct chromatic distortion of all colour channels to a given reference colour

    corrected images are saved as self.data["fieldcorr"]["array"]

    Parameters
    ----------
    colour_mapping: dictionary of string : string
        mapping of colour or type names from the image files
        to colours in the correction parameters files
        e.g. {"Cy3": "558", "Cy5": "684", "Cy7": "776"}
    reference_colour: str
        warp all other colours to match this colour
    calibration_path: str
        full path of folder containing chromatic calibration files (in csv format)
    """

    assert reference_colour in colour_mapping.values(), (
        f"Reference colour {reference_colour} not found "
        f"in type-to-colour dictionary:\n{colour_mapping}"
    )

    # Set up dictionary of pair-wise transforms
    # -----------------------------------------
    # keyed by (colour, reference colour) tuples

    params_pattern1 = re.compile("[a-zA-Z]+(\d+)_[a-zA-Z]+(\d+)")
    params_pattern2 = re.compile("([a-zA-Z0-9]+)to([a-zA-Z0-9]+)")

    pairs_tf_dict = {}

    for file in os.listdir(calibration_path):

        if os.path.splitext(file)[-1] == ".csv":

            match1 = re.search(params_pattern1, file)
            match2 = re.search(params_pattern2, file)

            if match1:
                match = match1
            elif match2:
                match = match2
            else:
                continue

            if verbose:
                print(
                    "\nFound chromatic correction params file:", match.group(0)
                )

            params_csv_fullpath = os.path.join(calibration_path, file)
            colour_pair = (match.group(1), match.group(2))

            pairs_tf_dict[colour_pair] = parseCsv(params_csv_fullpath)

    if verbose:
        print(f"Pairs transform dictionary:\n{pairs_tf_dict}")

    # Set up dictionary of colour -> reference transforms
    # ---------------------------------------------------
    # keyed by individual colours

    colours_tf_dict = {reference_colour: None}

    for pair in pairs_tf_dict:
        if pair[0] == reference_colour:
            colours_tf_dict[pair[1]] = pairs_tf_dict[pair]
        if pair[1] == reference_colour:
            colours_tf_dict[pair[0]] = pairs_tf_dict[pair].inverse

    if verbose:
        print(
            f"\nDictionary of affine parameters for colours:\n{colours_tf_dict}\n"
        )

    # Correct each slice of raw image array according to its colour
    # -------------------------------------------------------------

    imgdata.checkDownstream("chrcorr")

    chrcorr_array = imgdata.copyMostCorrectedRaw()

    for bit, colour in enumerate(imgdata.colour_list):

        print(
            f"Correcting chromatic distortion for bit {bit} with colour {colour}...\n"
        )

        if bit not in imgdata.dropped_bits:
            transform = colours_tf_dict[colour_mapping[colour]]
            if transform is not None:
                for frame in range(imgdata.frames):
                    slice_to_warp = chrcorr_array[frame, :, :, bit]
                    chrcorr_array[frame, :, :, bit] = warp(slice_to_warp, transform)

    imgdata.setArray("chrcorr", chrcorr_array)


def parseCsv(filepath: str,
                 verbose: bool = True,
                 ) -> AffineTransform:
    """
    parse chromatic shift parameters from a .csv format file
    and generate affine matrix

    returns affine matrix ready for warping with skimage
    """
    title_row = None
    values_row = None

    with open(filepath) as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=';')

        for row in csv_reader:
            if "a_affine_parameter" in row:
                title_row = [title.strip() for title in row]
            # go to next row. then stop reading
            elif title_row is not None:
                values_row = [float(value) for value in row]
                break

    assert title_row is not None, "title row for param values not found in csv file"
    assert values_row is not None, "row containing param values not found in csv file"

    matrix = np.zeros((3, 3), dtype=np.float64)
    matrix[0, 0] = values_row[title_row.index("a_affine_parameter")]
    matrix[0, 1] = values_row[title_row.index("b_affine_parameter")]
    matrix[1, 0] = values_row[title_row.index("c_affine_parameter")]
    matrix[1, 1] = values_row[title_row.index("d_affine_parameter")]
    matrix[0, 2] = values_row[title_row.index("x_translation_affine_parameter")]
    matrix[1, 2] = values_row[title_row.index("y_translation_affine_parameter")]
    matrix[2, 2] = 1

    if verbose:
        print("____ Matrix: ____\n", matrix)

    return AffineTransform(matrix)


#
#
# ----------------------------------------------------------------------------------------------------
#                                          Field Correction
# ----------------------------------------------------------------------------------------------------
#
#

def correctField(imgdata: ImageData,
                 hdf5_filepath: str,
                 img_to_mask: Dict[str, str] = None,
                 ) -> None:
    """
    Correct field distortion of all colour channels
    using provided correction masks file
    corrected images are saved as raw_im_corrected

    Parameters
    ----------
    hdf5_filepath: str
        full filepath containing field correction masks
    """

    imgdata.checkDownstream("fieldcorr")

    print(f"Correcting field using {hdf5_filepath} ...")

    fc = ApplyFieldCorrection(
        hdf5_filepath,
        verbose=True,
    )

    # check if all the types in the dataset have matching correction masks
    # --------------------------------------------------------------------

    if imgdata.colour_list is None or len(imgdata.colour_list) == 0:
        raise ValueError(
            f"No info on colour of hybs found in imagedata"
        )
    else:
        print(f"Imagedata colour list: {imgdata.colour_list}\n")

    for colour in set(imgdata.colour_list):
        if colour in fc.basetypes:
            print(
                f"{colour} found in correctionmasks hdf5 file.\n"
            )
        elif img_to_mask is not None and img_to_mask[colour] in fc.basetypes:
            print(
                f"{colour}->{img_to_mask[colour]} found in correctionmasks hdf5 file.\n"
            )
        else:
            raise ValueError(
                f"{colour} not found in hdf5 file:\n{hdf5_filepath}\n"
            )

    # correct image array slice by slice
    # ----------------------------------

    fieldcorr_array = imgdata.copyArray("raw")

    for bit, colour in enumerate(imgdata.colour_list):

        if img_to_mask is not None:
            # convert colour reference from the imagedata
            # to corresponding colour reference in the masks hdf5 file
            colour = img_to_mask[colour]

        for frame in range(imgdata.frames):
            print(
                f"Correcting field for bit {bit} ({colour}) frame {frame}...\n"
            )
            slice_to_correct = fieldcorr_array[frame, :, :, bit]
            fieldcorr_array[frame, :, :, bit] = fc.applyFC(
                slice_to_correct, colour,
            )

    imgdata.setArray("fieldcorr", fieldcorr_array)
