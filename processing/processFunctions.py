"""
Attempt at doing the pre-processing "functional style"

Includes functions for
(1) Registration
(2) Filtering

Work on ImageData class objects

nigel 30 Oct 19

License and readme found in https://github.com/khchenLab/split-fish
"""

import copy
import numpy as np

from typing import Tuple, Dict, Union

# from skimage.feature import register_translation
from utils.registrationFunctions import register_translation
from scipy import ndimage
import scipy.ndimage.interpolation as interpolation

from data_objects.imageData import ImageData

#
# Define custom types
# -------------------
#
RegerrorDict = Dict[int, Dict[str, float]]


def _printMinMax(img_slice: np.ndarray,
                 fov: str,
                 bit: int,
                 text: str = "") -> None:
    """
    print minimum and maximum value of an image slice
    """
    print(f"Minimum value of FOV {fov} bit {bit:02d} {text} : "
          f"{img_slice.min():,.2f}\n"
          f"max value of FOV {fov} bit {bit:02d} {text} : "
          f"{img_slice.max():,.2f}\n")


def registerPhaseCorr(imgdata: ImageData,
                      upsampling: int = 50,
                      consecutive: bool = False,
                      reference_bit: int = 0,
                      align_images_after_estimation: bool = True,
                      freq_filter: Union[np.ndarray, None] = None,
                      stage_to_register: Union[str, None] = None,
                      verbose: bool = True,
                      ) -> Tuple[np.ndarray, RegerrorDict]:
    """
    register the images with respect to the previous frame or to a fixed reference,
    then filter them using a frequency filter (if provided),
    (this can also be done seperately but is
     more efficient together as we don't need to fourier transform twice)

    Parameters
    ----------
    upsampling: int
        upsampling parameter for register_translation
        higher will lead to higher resoultion for shift estimate
        (but from what i've seen not neccearily more accurate)
    consecutive: bool
        if False, register all to a single frame
        if True, register each frame to the previous bit
    reference_bit: int,
        if consecutive is set False, register to this bit
        if consecutive is set True, start from this bit
        (using ORIGINAL BIT NUMBERS even if dropping bits)
    freq_filter: ndarray or None
        frequency domain filter of same size as image dimensions for filtering
        If none, image will not be filtered
    align_images_after_estimation:bool
        if True, align images and put result in ImageData's registered array
        If False, just estimate registration error and save in
        ImageData's registration_shifts attribute

    returns
    -------
        bits x 2 array of (y, x) shifts
        also modifies registered_im attribute of the image data object
    """

    assert reference_bit not in imgdata.dropped_bits, (
        f"Reference bit {reference_bit} is a dropped bit.\n"
        f"Cannot be used as reference."
    )

    imgdata.checkDownstream("registered")

    # Copy arrays from ImageData object
    # ---------------------------------

    if stage_to_register is None:
        reg_array = imgdata.copyMostCorrectedRaw()
    else:
        reg_array = imgdata.copyArray(stage_to_register)
    # NOTE: forces conversion from raw array datatype (usually uint16)
    # to ImageData's datatype attribute

    # filtered and registered images array
    filt_array = np.zeros_like(reg_array)

    # zero out slices of array from dropped bits
    reg_array[..., imgdata.dropped_bits] = 0

    # Set up arrays to store translation shifts
    # -----------------------------------------

    # number of dimensions for registration.
    # should be 2 for 2D images. May be 3 in future with image stacks
    ndims_register = 2
    registration_shifts = np.zeros((imgdata.num_bits, ndims_register))
    cumulative_shift = np.zeros((ndims_register,))

    if freq_filter is not None:
        #
        # check that freq filter dimensions match images
        # ----------------------------------------------

        assert freq_filter.ndim == ndims_register, (
            f"Frequency filter has {freq_filter.ndim} dimensions."
            f"Needs to be a {ndims_register} dimension array."
        )

        filter_ydim, filter_xdim = freq_filter.shape

        imgdata.checkDims(
            None,  # frames dimension
            filter_ydim,  # y dimension
            filter_xdim,  # x dimension
            None,  # bits dimension
            "of frequency filter"
        )

        filter_shifted = np.fft.fftshift(freq_filter)

    registration_error_dict = {}

    # Set up a list of valid bits
    # ---------------------------

    valid_bits = [bit for bit in range(imgdata.num_bits)
                  if bit not in imgdata.dropped_bits]

    # roll the array so that we start from the reference bit
    reference_bit_index = valid_bits.index(reference_bit)
    valid_bits = valid_bits[reference_bit_index:] + valid_bits[:reference_bit_index]

    if verbose:
        print(f"Valid bits starting from reference bit:\n{valid_bits}\n")

    # Register and align each bit
    # ---------------------------

    for bit in valid_bits:

        current_slice = copy.copy(reg_array[0, :, :, bit])
        current_slice_fourier = np.fft.fftn(current_slice)

        if bit == reference_bit:
            # This should be the first bit reached

            ref_slice = current_slice
            ref_slice_fourier = current_slice_fourier

            registration_error_dict[bit] = {"fine error": None,
                                            "pixel error": None}

            # filter the reference slice
            # --------------------------
            if freq_filter is not None:
                ref_filtered_slice = np.fft.ifftn(ref_slice_fourier * filter_shifted)
                filt_array[0, :, :, bit] = ref_filtered_slice.real

                _printMinMax(ref_filtered_slice.real,
                             imgdata.fov, bit,
                             text="(reference bit)")

            continue

        elif consecutive:
            # set the previous bit as the reference
            ref_slice_fourier = previous_slice_fourier

        # Estimate shifts between current slice and reference slice
        # ---------------------------------------------------------

        (shifts,
         fine_error,
         pixel_error) = register_translation(ref_slice_fourier,
                                             current_slice_fourier,
                                             upsample_factor=upsampling,
                                             space="fourier")

        assert len(shifts) == ndims_register, (
            f"Number of dimensions {len(shifts)} of "
            f"register_translation shift ouput {shifts}\n"
            f"does not match expected number of dimensions {ndims_register}."
        )

        if consecutive:
            cumulative_shift += shifts
        else:
            cumulative_shift = shifts

        registration_shifts[bit, :] = cumulative_shift

        registration_error_dict[bit] = {"fine error": fine_error,
                                        "pixel error": pixel_error}

        if align_images_after_estimation:

            # Align current slice with reference/previous slice
            # -------------------------------------------------

            registered_slice = np.fft.ifftn(
                ndimage.fourier_shift(current_slice_fourier, cumulative_shift)
            )

            reg_array[0, :, :, bit] = registered_slice.real

            if freq_filter is not None:
                registered_filtered_slice = np.fft.ifftn(
                    ndimage.fourier_shift(current_slice_fourier * filter_shifted,
                                          cumulative_shift)
                )

                filt_array[0, :, :, bit] = registered_filtered_slice.real

                _printMinMax(registered_filtered_slice.real, imgdata.fov, bit)

            previous_slice_fourier = current_slice_fourier

    # Set the Arrays in the ImageData object
    # --------------------------------------

    imgdata.setRegParams(
        reference_bit, registration_shifts, registration_error_dict
    )

    if align_images_after_estimation:

        imgdata.setArray(
            "registered", reg_array,
            info=f"Registered by Phase Correlation (using "
                 f"Scikit-Image implementation).  Ref bit = {reference_bit}"
        )

        if freq_filter is not None:
            imgdata.setArray(
                "filtered", filt_array,
                info=f"Filtered by 2D frequency domain butterworth "
                     f"filter. Ref bit = {reference_bit}"
            )

    return registration_shifts, registration_error_dict


def skipRegistration(imgdata: ImageData,
                     datatype: np.dtype = np.float64,
                     ) -> None:
    """
    skip the registration step and make the
    registered array match the raw array
    """
    imgdata.setArray(
        "registered", imgdata.copyMostCorrectedRaw(), info="registration was skipped",
    )


def skipFiltering(imgdata: ImageData,
                  ) -> None:
    """
    skip the filtering step and make the
    filtered array match the registered array
    """
    imgdata.setArray(
        "filtered", imgdata.copyArray("registered"), info="filtering was skipped",
    )


def registerByShifts(imgdata: ImageData,
                     shifts: np.ndarray,
                     reference_bit: int,
                     registration_error_dict: RegerrorDict = None,
                     stage_to_register: Union[str, None] = None,
                     method: str = "interpolation",
                     freq_filter: Union[np.ndarray, None] = None,
                     ) -> None:
    """
    Register images following a set of shifts
    reference bit must be provided (even though not used by this function)
    Available methods:
        interpolation : use scipy's ndimage interpolation.shift
        fourier : use fourier shift in freqency domain
                  (probably slower but matches the way shifts are done
                  in registerPhaseCorr)
    """

    if freq_filter is not None:
        filter_shifted = np.fft.fftshift(freq_filter)
    else:
        filter_shifted = None

    # Record registration-related data
    # --------------------------------

    imgdata.shifts = shifts
    imgdata.reference_bit = reference_bit
    if registration_error_dict is not None:
        imgdata.registration_error_dict = registration_error_dict

    # Choose which array to use
    # -------------------------

    if stage_to_register is None:
        # if no stage provided, default to most corrected raw images
        array = imgdata.copyMostCorrectedRaw()
    else:
        array = imgdata.copyArray(stage_to_register)

    registered_array = np.zeros_like(array, dtype=imgdata.datatype)

    # Register each slice
    # -------------------

    valid_bits = [bit for bit in range(imgdata.num_bits)
                  if bit not in imgdata.dropped_bits]

    for bit in valid_bits:

        raw_slice = array[0, :, :, bit]
        shift = shifts[bit, :]

        if method == "fourier" or freq_filter is not None:

            if freq_filter is None:

                registered_slice = np.fft.ifftn(
                    ndimage.fourier_shift(np.fft.fftn(raw_slice), shift)
                )

            else:

                registered_slice = np.fft.ifftn(
                    ndimage.fourier_shift(np.fft.fftn(raw_slice) * filter_shifted,
                                          shift)
                )

        elif method == "interpolation":

            registered_slice = interpolation.shift(raw_slice, shift)

        else:
            raise ValueError(f"Method {method} not recognised.\n"
                             f"Should be 'interpolation' or 'fourier'.")

        registered_array[0, :, :, bit] = registered_slice.real

    # Set the appropriate array
    # -------------------------

    if freq_filter is None:
        imgdata.setArray(
            "registered", registered_array,
            info=f"Registered with given shifts by method: {method}."
        )
    else:
        imgdata.setArray(
            "filtered", registered_array,
            info=f"Registered with given shifts in fourier domain and filtered."
        )


def clipBelowZero(imgdata: ImageData,
                  # iteration: int,
                  clip_below: float = 0.,
                  ) -> None:
    """
    Clip a filtered image array, ususally to remove values below 0,
    but allows for setting a different clip value.
    """
    assert imgdata.iteration == 0, (
        f"On iteration {imgdata.iteration}. "
        f"clipBelowZero should only be used on iteration 0"
    )

    filtered_clipped_array = imgdata.copyArray("filtered")

    np.putmask(
        filtered_clipped_array, filtered_clipped_array < clip_below, clip_below
    )

    imgdata.setArray(
        "filtered_clipped", filtered_clipped_array,
        info=f"negative values of filtered images clipped"
    )
