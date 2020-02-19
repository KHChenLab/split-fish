"""
ImageData: holds raw and processed images and parameters for these images

Modified from imagedataClasses
for use with functional style processing

30 Oct 19

License and readme found in https://github.com/khchenLab/split-fish
"""

import os
import h5py
import copy

import warnings
from collections import OrderedDict
from typing import Union, Dict, Tuple, List

import numpy as np

from scipy.spatial import cKDTree

from data_objects.geneData import GeneData
from utils.writeClasses import DaxWriter

# _____ Imports for plotting _____
from utils.readImagesFunctions import readImages
from utils.imagesQC import showImagesByBit
from utils.registrationQC import showRegistrationByBit
from utils.printFunctions import ReportTime


class ImageData(object):
    """
    Class for holding image data from a single FOV
    including all registered, and filtered outputs
    also has attributes that provide info on fovs, hybs, image dimensions

    this should be used as a context manager,
    to ensure that all references to its arrays are deleted after the FOV is processed

    dimensions of the typical arrays are:
    (frames, y_pix, x_pix, number_of_hybs)
    """

    # Attributes to save to the hdf5 File
    # -----------------------------------

    attrs_for_h5 = [
        "fov",
        "y_pix", "x_pix", "frames", "num_bits",
        "dropped_bits", "valid_bits",
        "upper_border", "lower_border",
        "left_border", "right_border",
        "normalization_vector",
    ]

    # Standard lists of stages for various purposes
    # ---------------------------------------------

    # full list of stages in standard order of processing
    process_order = [
        "raw",  # original images
        "background_removed",
        "fieldcorr",  # field correction (MUST precede distortion correction)
        "chrcorr",  # chromatic/distortion correction
        "registered",  # image registration across bits
        "filtered",  # after filtering of image
        "filtered_clipped",  # same as filtered, but -ve values clipped
        "normalized",  # normalization across bits
        "normalized_clipped",  # clip values above/below a certain value, usually 0 and 1
        "normalized_magnitude",  # magnitude of normalized, no bits dimension
        "binarized",  # binarized image (optional)
        "unitnormalized",  # unit-normalized image for vector comparison
        "closestdistance",
        "decoded",  # decoded image (no bits dimension)
        "decoded_sizefiltered"  # decoded with small regions removed
    ]

    # which stages have only Z, X and Y dimensions
    stages_with_no_bit_dimension = [
        "normalized_magnitude",
        "closestdistance",
        "decoded",
        "decoded_sizefiltered"
    ]

    stages_to_save_hdf5_default = [
        "raw",
        "fieldcorr",
        "chrcorr",
        "background_removed",
        "registered",
        "filtered",
        "normalized_clipped",
        "normalized_magnitude",
        "closestdistance",
        "decoded_sizefiltered"
    ]

    stages_to_save_hdf5_minimal = [
        "raw",
        "fieldcorr",
        "chrcorr",
        "background_removed",
        # "filtered",
        "filtered_clipped",
    ]

    def __init__(self,
                 iteration: int,
                 fov: str,
                 output_path: str,
                 genedata: GeneData = None,
                 y_pix: int = 1024,
                 x_pix: int = 1024,
                 frames: int = 1,
                 num_bits: int = 1,
                 datatype: np.dtype = np.float64,
                 stages_to_save_hdf5: Union[List[str], str, None] = "default",
                 border_padding: int = 30,
                 dropped_bits: List[int] = (),
                 ) -> None:
        """
        Parameters
        ----------
        fov: str
            FOV reference for the ImageData object
        y_pix, x_pix: int
            image dimensions
        frames: int
            number of frames / image dimension in z
        num_bits: int
            number of bits
        dtype: numpy datatype (default float64)
            datatype in which all operations will take place
        border_padding: int

            add extra padding on borders of image
            added during registration step when borders
            are calcuated from misalignment
        dropped_bits: list of integers
            bits that you intend to drop
            should only be defined at ImageData intialization
        """

        # main parameters
        # ---------------

        self.iteration = iteration
        self.fov = fov
        self.output_path = output_path
        self.y_pix = y_pix
        self.x_pix = x_pix
        self.frames = frames

        self.num_bits = num_bits
        self.dropped_bits = dropped_bits
        self.valid_bits = [bit for bit in range(num_bits) if bit not in dropped_bits]

        self.datatype = datatype
        self.stages_to_save_hdf5 = stages_to_save_hdf5
        self.border_padding = border_padding

        # set hdf5 filepaths
        # ------------------

        self.h5_filepath = os.path.join(
            self.output_path,
            f"FOV_{fov}_imagedata_iter{iteration}.hdf5"
        )

        # the base hdf5 file (for use in higher iterations)
        self.h5_filepath_iter0 = os.path.join(
            self.output_path,
            f"FOV_{fov}_imagedata_iter0.hdf5",
        )

        if genedata is None:
            self.genedata = None
        else:
            self.setGeneData(genedata)

        # Lists to record colours associated with each bit
        # ------------------------------------------------
        # for storing the colour (e.g. "Cy5") of the image associated with each bit

        self.colour_list = None

        self.data = OrderedDict()

        for process_stage in self.process_order:
            self.data[process_stage] = {
                "flag": False,
                "array": None,
                "info": "",
            }

        # NOTE: all flags should be boolean
        #       all arrays are numpy arrays with dimension:
        #       (frames, y_pix, x_pix, num_bits)

        self.printStagesStatus(f"at initialization")

        # Registration params
        # -------------------

        self.reference_bit = None  # an integer
        self.registration_shifts = None  # a numpy array
        self.registration_error_dict = None

        # image borders
        self.upper_border = 0
        self.lower_border = 0
        self.left_border = 0
        self.right_border = 0

        # Normalization params
        # --------------------

        self.normalization_vector = None

    def __enter__(self):
        """
        create a h5py File object and save as an attribute (h5)
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        save valid image arrays to the h5py File object (h5),
        then delete all references to the arrays
        """
        print("\nSaving all arrays and attributes to Hdf5 file ...")

        with ReportTime("save to hdf5") as _:

            with h5py.File(self.h5_filepath, "a") as h5file:
                # save arrays
                if self.stages_to_save_hdf5 is not None:
                    self.saveH5Arrays(h5file, self.stages_to_save_hdf5)

                # save attributes
                self.saveH5Attrs(h5file)

        print("\nDeleting all arrays from current ImageData instance ...")

        for process_stage in self.data:
            del self.data[process_stage]["array"]
            del self.data[process_stage]["flag"]
            del self.data[process_stage]["info"]
        del self.data

        del self.colour_list
        del self.reference_bit

    #
    #                       hdf5 file reading and writing methods
    # -----------------------------------------------------------------------------------------------
    #

    def readH5Arrays(self,
                     h5file: h5py.File,
                     stage_list: Union[List[str], str],
                     ) -> None:
        """
        read arrays from a h5 file
        """

        trimmed_list = self._filterStages(stage_list)

        print(f"Stages to read from hdf5 file: {stage_list}")

        for stage in trimmed_list:

            if stage in h5file:

                print(f"Reading {stage} array from hdf5 file.\n")
                self.setArray(stage, np.array(h5file[stage]))

            else:

                print("Could not find {stage} array in hdf5 file.\n")

    def saveH5Arrays(self,
                     h5file: h5py.File,
                     stage_list: Union[List[str], str],
                     ) -> None:
        """
        save arrays from a selection of stages to the h5 file object
        """

        trimmed_list = self._filterStages(stage_list)

        for stage in trimmed_list:

            if stage in h5file:

                print(f"{stage} already present in hdf5 file. Not saving.\n")

            elif not self.getFlag(stage):

                print(f"{stage} could not be saved to hdf5 file. "
                      f"The ImageData object does not contain this array.\n")

            else:

                try:
                    h5file.create_dataset(stage, data=self.viewArray(stage))
                except:
                    print(f"Failed to create dataset for {stage} with array:\n"
                          f"{self.viewArray(stage)}")
                    raise

    def _filterStages(self,
                      stage_list: Union[str, List[str]],
                      ) -> List[str]:
        """
        return a list of stages containing only the ones
        that are valid stages and have been completed (i.e. contain arrays)
        """

        # Check if we are provided one of the standard lists or a custom list
        # -------------------------------------------------------------------

        if stage_list is "all":
            # list all possible stages
            stage_list = self.process_order

        elif stage_list is "default":
            stage_list = self.stages_to_save_hdf5_default

        elif stage_list is "minimal":
            stage_list = self.stages_to_save_hdf5_minimal

        elif isinstance(stage_list, list):
            pass

        else:
            raise TypeError(
                f"List of stages given was {stage_list}.\n"
                f"Must be a list of strings, 'all', 'default' or 'minimal'."
            )

        # Check that all elements of the list are valid stages
        # ----------------------------------------------------

        return [stage for stage in stage_list
                if stage in self.process_order]

    def saveH5Attrs(self,
                    h5file: h5py.File,
                    ) -> None:
        """
        save relevant attributes of ImageData object
        as attributes of the hdf5 file
        """
        for attr in self.attrs_for_h5:

            attr_value = getattr(self, attr)

            if attr_value is None:
                print(f"{attr} has no value and "
                      f"was not saved to hdf5 file.\n")
            else:
                try:
                    h5file.attrs[attr] = attr_value
                except:
                    raise ValueError(f"{attr}: {attr_value}\n"
                                     f"could not be saved as "
                                     f"a hdf5 dataset attribute.\n")

    def readH5Attrs(self,
                    h5file: h5py.File,
                    ) -> None:
        """
        read the attributes of hdf5 object
        don't need to filter since hdf5 attributes are always
        a subset of ImageData's attributes
        """
        for attr in h5file.attrs:
            setattr(self, attr, h5file.attrs[attr])

    #
    #                            Methods for saving arrays
    # -------------------------------------------------------------------------------
    #

    def saveArray(self,
                  current_stage: str,
                  file_ext: str = ".dax",
                  project_method: Union[str, None] = None,
                  scale_to_max: bool = False,
                  ) -> None:
        """
        save the array from one of the stages to a .dax or .npy file

        Parameters
        ----------
        file_ext: str ".dax" or ".npy"
            file extension, either dax or numpy
        sacle_to_max_value: bool
            if True, the maximum value of the whole array will be 1
            (or 65535 if saving to .dax which uses uint16 type)
            and all other values are scaled accordingly.
            Useful for visualizing normalized arrays where
            maximum value is 1 or slightly more than 1
        project_method: None, "mip" or "mean"
            If given, project the images along both frames and bits
            using maximum intensity projection (mip)
            or    mean projection (mean)
        """

        print(f"Saving {current_stage} array as {file_ext} file...")

        save_filename = (f"FOV_{self.fov}_{current_stage}_"
                         f"{self.frames}x{self.y_pix}x{self.x_pix}")

        assert self.getFlag(current_stage), f"No {current_stage} array found"

        if project_method is None and not scale_to_max:
            # Array will not be modified, so no need to copy the array
            array_to_save = self.viewArray(current_stage)
        else:
            array_to_save = self.copyArray(current_stage)

        # Project Array
        # -------------

        if array_to_save.ndim == 4:

            if project_method is None:
                save_filename += f"x{self.num_bits}"

            elif project_method == "mip":
                print("\nProjecting by Maximum Intensity ...")
                array_to_save = np.amax(array_to_save, axis=3)
                save_filename += "_maxintprojected"

            elif project_method == "mean":
                print("\nProjecting by mean ...")
                array_to_save = np.mean(array_to_save, axis=3)
                save_filename += "_meanprojected"

            else:
                raise ValueError(f"Projection method: {project_method} not recognised.\n"
                                 f"Must be either None, 'mip' or 'mean'.")

        # Scale Array
        # -----------

        if scale_to_max:
            array_to_save /= array_to_save.max()
            if file_ext == ".dax":
                array_to_save *= 65535  # highest integer value for uint16
            save_filename += "_scaledtomax"

        # Save the array to the appropriate type
        # --------------------------------------

        full_savepath = os.path.join(self.output_path,
                                     save_filename + file_ext)

        if file_ext == ".npy":
            np.save(full_savepath, array_to_save)

        elif file_ext == ".dax":

            with DaxWriter(full_savepath) as daxwriter:

                if array_to_save.ndim == 4:
                    for bit in range(self.num_bits):
                        for frame in range(self.frames):
                            daxwriter.addFrame(array_to_save[frame, :, :, bit])

                elif array_to_save.ndim == 3:
                    for frame in range(self.frames):
                        daxwriter.addFrame(array_to_save[frame, :, :])
                else:
                    raise IndexError(f"array has {array_to_save.ndim} dimensions.\n"
                                     f"Must be either 3 or 4")

        else:
            raise ValueError(f"File extension {file_ext} not recognised.\n"
                             f"Must be '.npy' or '.dax'.")

    #
    #                            Print info on status for all stages
    # -------------------------------------------------------------------------------------------
    #
    #

    def printStagesStatus(self,
                          text: str,
                          ) -> str:
        """
        print out the dictionary of flags and arrays
        """
        dotted_line = "-" * 60

        status_str = (dotted_line +
                      f"\nFOV {self.fov} Data and flags {text}:\n" +
                      dotted_line + "\n\n")

        for stage_num, stage in enumerate(self.data):

            stage_title_str = f"Stage {stage_num} : {stage}\n"
            status_str += stage_title_str + "-" * len(stage_title_str)

            # flag
            # ----
            status_str += f"\n\tFlag  : {self.data[stage]['flag']}"

            # array
            # -----
            if self.data[stage]['array'] is None:
                array_str = "-"
            if isinstance(self.data[stage]['array'], np.ndarray):
                array_str = f"{self.data[stage]['array'].shape}"
            status_str += f"\n\tArray : {array_str}"

            # info
            # ----
            status_str += f"\n\tInfo  : {self.data[stage]['info']}\n\n"

        status_str += dotted_line + "\n"

        print(status_str)

        return status_str

    #
    #                 Getting and setting flags / arrays / info for each stage
    # -------------------------------------------------------------------------------------------
    #

    def setGeneData(self,
                    genedata: GeneData,
                    ) -> None:
        """
        Assign a GeneData object to the ImageData object
        Check if the kDTree is present, and make sure number of bits match.
        """
        assert isinstance(genedata.codebook_tree, cKDTree), (
            f"GeneData object does not have a valid kD tree"
        )
        assert genedata.num_bits == self.num_bits, (
            f"Number of bits in GeneData: {genedata.num_bits}\n"
            f"does not match number of bits: {self.num_bits} in ImageData.\n")
        self.genedata = genedata

    def raiseIfNoGeneData(self) -> None:
        """
        return true if the object has a GeneData assigned
        """

        if not isinstance(self.genedata, GeneData):
            raise TypeError(f"ImageData's genedata attribute "
                            f"is of type {self.genedata}.\n"
                            f"Should be an instance of GeneData.")

    def getFlag(self,
                current_stage: str,
                ) -> bool:
        """
        return the flag of the current stage
        """
        return self.data[current_stage]["flag"]

    def setFlag(self,
                current_stage: str,
                flag: bool,
                ) -> None:
        """
        set the flag of the current stage
        """
        self.data[current_stage]["flag"] = flag

    def getInfo(self,
                current_stage: str,
                ) -> bool:
        """
        return the flag of the current stage
        """
        return self.data[current_stage]["info"]

    def setInfo(self,
                current_stage: str,
                info: str,
                ) -> None:
        """
        set the flag of the current stage
        """
        self.data[current_stage]["info"] = info

    def copyArray(self,
                  current_stage: str,
                  ) -> Union[np.ndarray, None]:
        """
        COPY the array from the current stage
        """
        return copy.copy(
            self.viewArray(current_stage)
        ).astype(self.datatype)

    def viewArray(self,
                  current_stage: str,
                  ) -> Union[np.ndarray, None]:
        """
        retrieve a reference to the current stage array
        """
        assert self.getFlag(current_stage), (
            f"No array for {current_stage} present."
        )
        return self.data[current_stage]["array"]

    def setArray(self,
                 current_stage: str,
                 array: np.ndarray,
                 info: Union[str, None] = None,
                 ) -> None:
        """
        set the current stages's array reference to the provided numpy array,
        NOTE: deletes the provided ndarray reference and sets the current stage's flag
        """

        # check that current stage array not already filled
        # -------------------------------------------------

        assert self.data[current_stage]["array"] is None, (
            f"Already have a {current_stage} array, unable to set this array."
        )

        # check dimensions
        # ----------------

        array_shape = array.shape

        # some stages 3D arrays with no bits dimension
        if current_stage in self.stages_with_no_bit_dimension:
            self.checkDims(array_shape[0],  # frames
                           array_shape[1],  # y
                           array_shape[2],  # x
                           None,  # no bits dimension
                           info_str="decoded image array")

        # the rest have a full 4D array with bits as the last dimension
        else:
            self.checkDims(array_shape[0],  # frames
                           array_shape[1],  # y
                           array_shape[2],  # x
                           array_shape[3],  # number of bits
                           info_str="modified array")

        self.data[current_stage]["array"] = array
        del array

        self.setFlag(current_stage, True)

        if info is not None:
            self.setInfo(current_stage, info)

    def copyMostCorrectedRaw(self) -> np.ndarray:
        """
        copy the most corrected raw image array
        """
        return copy.copy(
            self.viewMostCorrectedRaw()
        ).astype(self.datatype)

    def viewMostCorrectedRaw(self,
                             verbose: bool = True
                             ) -> np.ndarray:
        """
        Get the most corrected raw image array
        available in the ImageData object
        """

        array_order = [
            "chrcorr",
            "fieldcorr",
            "background_removed",
            "raw",
        ]  # reverse order starting from most-corrected

        for array_name in array_order:

            if self.getFlag(array_name):

                if verbose:
                    print(
                        f"\nMost corrected raw array found: {array_name}\n"
                    )

                return self.viewArray(array_name)

        raise ValueError("No raw images loaded!")

    #
    #                            Checking Functions
    # ------------------------------------------------------------------------------
    #

    def checkDownstream(self,
                        current_stage: str,
                        start_from_next_stage: bool = False,
                        ) -> None:
        """
        Check that current and downstream stages have not been completed
        by checking the flags at each part of the process
        If start_from_next_stage is True, start checking from the subsequent stage.
        """

        current_index = self.process_order.index(current_stage)

        if start_from_next_stage:
            current_index += 1

        for process_stage in self.process_order[current_index:]:
            assert self.getFlag(process_stage) is False, (
                f"Downstream stage: {process_stage} already completed.\n"
                f"Aborting processing for current stage: {current_stage}\n"
            )

    def checkDims(self,
                  frames: Union[int, None],
                  y_pix: Union[int, None],
                  x_pix: Union[int, None],
                  num_bits: Union[int, None],
                  info_str: str = "",
                  ) -> None:
        """
        used to check each dimension of a particular image or array
        against the expected dimensions.
        Raises exception if there is a mismatch in any of the dimensions.
        Can skip checking dimensions by setting as None
        """
        img_str = f" for {info_str}\n"

        if frames is not None:
            assert self.frames == frames, (f"Z/frames dimension {frames} "
                                           f"does not match expected"
                                           f"Z dimension {self.frames}"
                                           + img_str)
        if y_pix is not None:
            assert self.y_pix == y_pix, (f"Y dimension {y_pix} does not match "
                                         f"expected Y dimension {self.y_pix}"
                                         + img_str)
        if x_pix is not None:
            assert self.x_pix == x_pix, (f"X dimension {x_pix} does not match "
                                         f"expected X dimension {self.x_pix}"
                                         + img_str)
        if num_bits is not None:
            assert self.num_bits == num_bits, (f"bits dimension {num_bits} does not match "
                                               f"expected bits dimension {self.num_bits}"
                                               + img_str)

    #
    #                       Registration related methods
    # -------------------------------------------------------------------------------
    #
    #

    def setRegParams(self,
                     reference_bit: int,
                     registration_shifts: np.ndarray,
                     registration_error_dict: Dict[int, Dict[str, float]],
                     ) -> None:
        """
        set the registration-related parameters
        """
        self.reference_bit = reference_bit
        self.registration_shifts = registration_shifts
        self.registration_error_dict = registration_error_dict
        (self.upper_border,
         self.lower_border,
         self.left_border,
         self.right_border) = self._bordersFromRegistrationShifts(registration_shifts,
                                                                  verbose=True)

    def getRegParams(self):
        """
        get the registration params as a tuple
        """
        return (self.reference_bit, self.registration_shifts, self.registration_error_dict)

    def _bordersFromRegistrationShifts(self,
                                       registration_shifts: np.ndarray,
                                       verbose: bool = True,
                                       ) -> Tuple[int, int, int, int]:
        """
        Find border-size to use based on maximum registration registration_shifts
        Add extra padding provided by padding paramter
        """

        # upper and left max shift (+ve registration_shifts for y & x)
        upper_left_shift = np.max(registration_shifts, axis=0)

        # lower and right max shift (-ve registration_shifts for y & x)
        lower_right_shift = -1 * np.min(registration_shifts, axis=0)

        padding = self.border_padding
        upper_border = int(upper_left_shift[0]) + padding
        lower_border = int(lower_right_shift[0]) + padding
        left_border = int(upper_left_shift[1]) + padding
        right_border = int(lower_right_shift[1]) + padding

        if verbose:
            print(f"--- Maximum registration_shifts for FOV {self.fov} --- : \n"
                  f"Upper: {upper_border}, Lower: {lower_border},\n"
                  f"Left:  {left_border}, Right: {right_border},\n"
                  f"Padding = {padding}\n")

        return upper_border, lower_border, left_border, right_border

    #
    #                     Normalization across bits
    # ---------------------------------------------------------------------------------
    #
    #

    def setNormalizationVector(self,
                               normalization_vector: np.ndarray,
                               ) -> None:
        """
        Check dimension and length of normalization vector before saving it
        """

        assert normalization_vector.ndim == 1, (
            f"Normalization vector has {normalization_vector.ndim} dimensions. "
            f"Should have only 1 dimension"
        )

        assert normalization_vector.shape[0] == self.num_bits, (
            f"Normalization vector length {normalization_vector.shape[0]} "
            f"does not match number of bits {self.num_bits}"
        )

        self.normalization_vector = normalization_vector

    #
    #                     Reading of images (from list of files or HDF5)
    # ---------------------------------------------------------------------------------
    #
    #

    def readFromH5(self,
                   h5_filepath: Union[str, None] = None,
                   stage_list: Union[List[str], str, None] = "minimal",
                   ) -> None:
        """
        read arrays and attributes from an imagedata hdf5 file.

        Parameters
        ----------
        h5_file: h5 file object, h5 filename, or None
            if None, will use the h5 file associated with the ImageData object
        stage_list: str ("all" or "default), list of str or None
            "all" - read all arrays
            "default" - check default list
            "minimal" - a few arrays up till filtered
            None - don't read any arrays
        """

        if h5_filepath is None:
            # default to the 0th iteration hdf5 file
            # which should contain saved raw and/or filtered images
            h5_filepath = self.h5_filepath_iter0

        with h5py.File(h5_filepath, "a") as h5file:

            self.readH5Attrs(h5file)

            if stage_list is None:

                print(
                    f"Read attributes from <{h5_filepath}>, "
                    f"but did not read any arrays."
                )

            else:

                self.readH5Arrays(h5file, stage_list)

    def readFiles(self,
                  data_path: str,
                  img_list: list,
                  dory_projection_method: str = "maxIntensityProjection",
                  microscope_type: str = "Dory",
                  verbose: bool = False,
                  ) -> None:
        """
        Read a list of image files from the data_path folder,
        then populate the raw image array with these images.
        The list of image files should already be in the correct bit order for decoding.
        (i.e. you should rearrange them if hybs were not run in order)

        Parameters
        ----------
        data_path: str
            data folder to read images from
        img_list: list
            list of image files and other relevant data (e.g. tiff frame)
        dory_projection_method: str
            type of 3D -> 2D projection to do if using "Dory" microscope
            available optiions (from ReadDax class):
            1) loadDaxFrame
            2) meanProjection
            3) maxIntensityProjection (default)
        plot_images: bool = True,
            (optional) plot all the raw images
        microscope_type: str
            either "Dory" (using HAL) or "Triton" (using micromanager)
        **kwargs
            extra keyword argument options for plotting (see showRawImages)

        also records the type (e.g. "Cy5") of each bit
        and records it under attribute "type_list"
        """

        num_imgs = len(img_list)

        if num_imgs != self.num_bits:
            print(
                f"...changing original number of bits: {self.num_bits}\n"
                f"   to the number of images in the image-file list: {num_imgs}\n"
            )
            self.num_bits = num_imgs

        raw_array, colour_list = readImages(
            img_list, data_path,
            self.y_pix, self.x_pix,
            microscope_type,
            dory_projection_method=dory_projection_method,
            verbose=verbose,
        )

        if self.colour_list is not None:
            warnings.warn(
                f"Overwriting colour_list of imagedata.\n"
                f"Original colour_list: {colour_list}\n"
            )

        self.colour_list = colour_list

        self.setArray("raw", raw_array)


#
#                              Plotting Functions
# -------------------------------------------------------------------------------
#

def showImages(imgdata: ImageData,
               current_stage: str,
               additional_info_str: str = "",
               **kwargs,
               ) -> None:
    """
    show images from a particular stage of processing
    """

    if not imgdata.getFlag(current_stage):
        warnings.warn(
            f"Unable to plot images. No array for {current_stage}."
        )
        return

    showImagesByBit(imgdata.viewArray(current_stage),
                    fov_str=imgdata.fov,
                    image_info=current_stage + additional_info_str,
                    **kwargs)


def showRegistration(imgdata: ImageData,
                     current_stage: str = "registered",
                     **kwargs,
                     ) -> None:
    """
    show registration and registration accuracy
    """

    assert current_stage not in imgdata.stages_with_no_bit_dimension, (
        f"{current_stage} does not have a bits dimension\n"
        f"and cannot be used to visualize registration."
    )

    assert imgdata.getFlag(current_stage), (
        f"Unable to plot registered images from {current_stage}.\n"
        f"No image array found."
    )

    showRegistrationByBit(
        imgdata.viewArray(current_stage),
        imgdata.reference_bit,
        imgdata.registration_shifts,
        registration_error_dict=imgdata.registration_error_dict,
        fov_str=imgdata.fov,
        dropped_bits=imgdata.dropped_bits,
        **kwargs,
    )


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

    # microscope_type = "Triton"
    microscope_type = "Dory"

    params = {}

    if microscope_type == "Dory":

        fovs = [0, 1, 2]

        stage_pixel_matrix = 8 * np.array([[0, -1], [-1, 0]])

        # params["num_bits"] = 16
        params["num_bits"] = 26
        params["hyb_list"] = list(range(9)) * 2 + list(range(8))
        params["type_list"] = ["Cy7", ] * 9 + ["Cy5", ] * 9 + ["Cy3", ] * 8
        params["roi"] = None

    elif microscope_type == "Triton":

        fovs = [(0, 0), (0, 1), (0, 2)]

        # stage_pixel_matrix = 14.1 * np.array([[1, 0], [0, 1]])
        stage_pixel_matrix = 8 * np.array([[1, 0], [0, 1]])

        params["num_bits"] = 16
        params["hyb_list"] = list(range(4)) * 4
        params["type_list"] = ["Cy7"] * 4 + ["Cy5"] * 4 + ["Alexa594"] * 4 + ["Cy3"] * 4
        params["roi"] = 2
        # params["roi"] = 1
        # params["roi"] = None

    # Print some of the params
    # ------------------------
    print(f"Hyb list: {params['hyb_list']}\n",
          f"Type list: {params['type_list']}\n")

    # get the file parser
    # -------------------
    myparser = getFileParser(
        data_path,
        microscope_type,
        # stage_pixel_matrix=stage_pixel_matrix,
        use_existing_filesdata=True,
    )

    files_dict = myparser.dfToDict(
        fovs,
        roi=params["roi"],
        num_bits=params["num_bits"],  # number of bits to go until
        hyb_list=params["hyb_list"],  # list of hyb numbers for each bit
        type_list=params["type_list"],  # list of filename types
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
            showImages(imagedata, "raw",
                       fig_savepath=save_dir,
                       figure_grid=(4, 7))
            imagedata.printStagesStatus("after raw input")
