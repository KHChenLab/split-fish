"""
Functions to normalize across bits by
combining image intensity information from multiple FOVs

nigel 11 dec 19

License and readme found in https://github.com/khchenLab/split-fish
"""

import h5py
import numpy as np

from typing import List, Dict

from processing.plotCountsFunctions import listH5PathsByFov
import matplotlib.pyplot as plt


def _getNumBits(files_byfov: Dict[str, str]) -> int:
    """
    count the number of bits in a dictionary of file-lists.
    takes the count from the first key
    """

    try:
        first_fov = next(iter(files_byfov))
    except StopIteration:
        print(f"Dictionary provided: {files_byfov} is Empty.")
        raise

    print(f"\nGetting number of bits from FOV {first_fov}.\n")

    try:
        with h5py.File(files_byfov[first_fov], "a") as h5file:
            num_bits = h5file.attrs["num_bits"]

        print(
            f"Detected that there are {num_bits} bits from FOV {first_fov}.\n"
            f"Assuming that other FOVs have same number of bits.\n"
        )

        return num_bits

    except ValueError:

        print("could not find 'num_bits' attribute in hdf5 imagedata file.")
        raise

    except:

        print("could not read number of bits from hdf5 imagedata file.")
        raise


def globalNormalize(data_path: str,
                    per_image_sample_method: str = "highest",
                    downsample: int = 10,
                    method: str = "median",
                    per_image_percentile_cut: float = 99.9,
                    pooled_percentile: float = 50,
                    fov_subset: List[str] = None,
                    verbose: bool = True,
                    ) -> np.ndarray:
    """

    :param data_path:
    :param per_image_sample_method:
    :param downsample:
    :param method:
    :param per_image_percentile_cut:
    :param pooled_percentile:
    :param verbose:
    :return:
    """

    # Get the valid imagedata hdf5 files from folder
    # ----------------------------------------------

    h5filepaths_byfov = listH5PathsByFov(
        data_path,
        h5filetype="imagedata",
        verbose=True,
    )

    assert len(h5filepaths_byfov) > 0, (
        f"No imagedata hdf5 files found in {data_path}"
    )

    if fov_subset is None:
        fovs = list(h5filepaths_byfov.keys())
    else:
        fovs = [
            fov for fov in h5filepaths_byfov if fov in fov_subset
        ]

    assert len(fovs) > 0, (
        f"FOVs selected do not match folder FOVs."
    )

    num_bits = _getNumBits(h5filepaths_byfov)

    normalization_vector = []

    for bit in range(num_bits):

        print(f"Processing image intensities from bit {bit}\n" + "-" * 45 + "\n")

        intensity_pooled = []

        for fov in fovs:

            # Read one slice of filtered array from hdf5 file
            # -----------------------------------------------

            with h5py.File(h5filepaths_byfov[fov], "a") as h5file:

                try:
                    raw_array = np.array(h5file["filtered_clipped"][..., bit])

                except KeyError:
                    print(f"could not find filtered array in data")
                    raise

                except:
                    print(f"Could not read hdf5 file")
                    raise

            # Sample a subset of pixels from the image
            # ----------------------------------------

            if per_image_sample_method == "highest":

                cutoff = np.percentile(
                    raw_array, per_image_percentile_cut
                )

                subsampled_array = raw_array[raw_array > cutoff]

            elif per_image_sample_method == "subsample":

                subsampled_array = raw_array[:, ::downsample, ::downsample]

            else:
                raise ValueError(f"method {per_image_sample_method} not recognised.")

            intensity_pooled.append(subsampled_array.flat)

        # Concatenate lists of arrays into single 1D array
        intensity_pooled = np.concatenate(intensity_pooled, axis=None)

        # plt.hist(intensity_pooled, bins=500, histtype="step")

        # Calculate some representative value from pooled pixel intensities
        # -----------------------------------------------------------------

        if method == "median":

            bit_normalization_value = np.percentile(
                intensity_pooled, pooled_percentile
            )


        elif method == "histogram_max":

            # FIXME: the histogram method probably doesn't work. Remove?

            hist, bin_edges = np.histogram(intensity_pooled, bins=500)
            max_bin_index = np.argmax(hist)
            max_bin_value = bin_edges[max_bin_index + 1] - bin_edges[max_bin_index]

            bit_normalization_value = max_bin_value

        else:
            raise ValueError(f"Method {method} not recognised."
                             f"Must be 'median' or 'histogram_max'.")

        normalization_vector.append(bit_normalization_value)

        if verbose:
            print(f"\nNormalization vector:\n{normalization_vector}\n")

    return np.array(normalization_vector)


if __name__ == "__main__":

    # imports for testing
    # -------------------

    import tkinter as tk
    from tkinter import filedialog
    from fileparsing.filesClasses import getFileParser

    # choose data path
    # ----------------

    root = tk.Tk()
    root.withdraw()
    data_path = filedialog.askdirectory(title="Please select data directory")
    root.destroy()

    # Set parameters
    # --------------

    microscope_type = "Dory"

    params = {}

    if microscope_type == "Dory":
        # params["fovs_to_process"] = [0, 1, 2]
        params["fovs_to_process"] = list(range(25))
        params["fovstr_length"] = 2
        params["num_bits"] = 26
        params["hyb_list"] = list(range(13)) * 2
        params["type_list"] = ["Cy7"] * 13 + ["Cy5"] * 13
        params["roi"] = None
    else:
        raise ValueError("No parameters provided")

    # Get list of files for each FOV
    # ------------------------------

    myparser = getFileParser(
        data_path,
        microscope_type,
        fovstr_length=params["fovstr_length"],
        use_existing_filesdata=True,
    )

    filelist_byfov = myparser.dfToDict(
        params["fovs_to_process"],
        roi=params["roi"],
        num_bits=params["num_bits"],  # number of bits to go until
        hyb_list=params["hyb_list"],  # list of hyb numbers for each bit
        type_list=params["type_list"],  # list of filename types
        verbose=True,
    )

    ydim = int(myparser.files_df["ydim"].values[0])
    xdim = int(myparser.files_df["xdim"].values[0])

    normalization_vector = globalNormalize(
        filelist_byfov, data_path,
        # ydim, xdim,
        method="median",
        per_image_percentile_cut=99.9,
        pooled_percentile=50,
        verbose=True,
    )

    plt.plot(normalization_vector, "r-")
    plt.show()
