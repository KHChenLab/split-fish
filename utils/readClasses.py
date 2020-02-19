"""
All the classes for reading/parsing files and images

readDax is a heavily modified version of one of the older Dax readers found in
https://github.com/ZhuangLab/storm-control

Nigel Jan 2019

License and readme found in https://github.com/khchenLab/split-fish
"""

import os
import re
import copy
import numpy as np

from typing import Union, List, Dict, Tuple

import warnings
import skimage

# for registration
from skimage import io


#
# =====================================================================================================================
#   Functions for reading specific types of image data
# =====================================================================================================================
#

def readDoryImg(file_path: str,
                project: bool = False,
                ) -> np.ndarray:
    """
    read Dory/Nemo file format (individual .dax files for each FOV/colour/hyb-round)
    squeezes the image to 2D array from 3D output of DaxRead if only 1 frame is detected
    :returns 2D or 3D numpy array with the dimensions (z?, y,x)
    """
    daxreader = DaxRead(file_path)
    if project:
        img = daxreader.maxIntensityProjection()
    else:
        img = daxreader.loadAllFrames()

    frames = daxreader.frames
    del daxreader

    if frames == 1:
        return np.squeeze(img, axis=0)
    else:
        return img


def readSpongebobImg(file_path: str,
                     frame: int = None, ) -> np.ndarray:
    """
    read Spongebob file format (multiframe ome.tif files, one for each FOV and hyb round containing all colours)

    :returns 2D or 3D numpy array with the dimensions (z?, y,x)
    """

    if frame is None:
        raise ValueError(
            "Cannot read Spongebob format image.\n"
            "Frame of multiframe ome.tif file not specified"
        )

    with warnings.catch_warnings():
        # filter out 'not an ome-tiff master file' UserWarning
        warnings.simplefilter("ignore", category=UserWarning)
        img = skimage.io.imread(file_path)[:, :, frame]

    return img


#
# =====================================================================================================================
#   Classes for reading .dax files ----- (1) DaxRead
# =====================================================================================================================
#

class DaxRead(object):
    """
    class to read a SINGLE dax file

    in this class, we assume that each dax file is a 3D image from a single time point
    (if multiple hybs, fovs or times are combined into a single dax file, dont use this)
    all data should be represented by the 3 dimensions:
    dim1 = frame (should be a set of z-stacks)
    dim2 = y axis
    dim3 = x axis
    keeps the frame dimension as a singleton dimension even if there is only one frame (e.g. after projection),
    for compatiblity with the other parts of the pipeline
    """

    def __init__(self,
                 filename: str = None,
                 frames: int = 1,  # z dimension
                 x_pix: int = 1024,
                 y_pix: int = 1024,
                 **kwargs):
        super(DaxRead, self).__init__(**kwargs)

        self.filename = filename
        self.frames = frames
        self.x_pix = x_pix
        self.y_pix = y_pix
        self.readInfFile()
        # this will edit the y_pix, x_pix and frames values (if available)
        # based on the associated .inf file. Othewise, default values will be used

    def readInfFile(self):
        """
        query the associated .inf file for dimensions and frames info.
        update the class attributes (y_pix, x_pix and frames) if such info is found.
        complains if the .inf file could not be found or read
        """
        dim_pattern = re.compile(r"frame\sdimensions\s=\s(\d+)\sx\s(\d+)")
        frames_pattern = re.compile(r"number\sof\sframes\s=\s(\d+)")

        try:
            with open(os.path.splitext(self.filename)[0] + ".inf", "r") as file:
                filetxt = file.read()
                match_dim = re.search(dim_pattern, filetxt)
                if match_dim:
                    self.y_pix, self.x_pix = int(match_dim.group(1)), int(match_dim.group(2))
                match_frames = re.search(frames_pattern, filetxt)
                if match_frames:
                    self.frames = int(match_frames.group(1))

        except FileNotFoundError:
            print(
                f".inf file for {self.filename} could not be found."
            )
        except OSError:
            print(
                f"Unable to open {self.filename} .inf file"
            )
        except:
            print(
                f"Could not read {self.filename} .inf file for some reason"
            )

    def loadSingleDaxFrame(self) -> np.ndarray:
        """
        load the first frame from the dax file

        probably shouldn't use this since it may get the wrong z-slice (possibly the one on top)
        """
        with open(self.filename, "rb") as daxfile:
            image_data = np.fromfile(
                daxfile, dtype=np.uint16,
                count=self.x_pix * self.y_pix,
            )

            image_data = np.reshape(
                image_data, (1, self.y_pix, self.x_pix),
            )

        return image_data

    def loadAllFrames(self,
                      subset: List[int] = None,  # must be a list of frames
                      ):
        """
        loads all the frames in the dax
        can use the given number of frames (in self.frames)
        or calculate it based on single-frame size
        """
        # first read the whole file
        with open(self.filename, "rb") as daxfile:
            image_data = np.fromfile(daxfile, dtype=np.uint16)

        # if we haven't got the number of frames or
        # the given dimensions don't match up,
        # recalculate number of frames

        if self.frames is None or (self.frames * self.y_pix * self.x_pix) != image_data.size:
            frames, remainder = divmod(image_data.size, self.y_pix * self.x_pix)
            if remainder == 0:
                self.frames = frames
            else:
                raise ValueError("Error: dax file element length is not a multiple of frame size")

        # reshape the numpy array
        image_data = image_data.reshape((self.frames, self.y_pix, self.x_pix))

        # get subset of frames if that option is given
        if subset is not None:
            subset = [frame for frame in subset if frame < self.frames]
            image_data = image_data[subset, :, :]

        return image_data

    def meanProjection(self):
        """
        average over all frames (not recommended. maximum intensity is usually better)
        """
        image_data = self.loadAllFrames()
        mp = image_data.sum(0, keepdims=True) / self.frames

        self.frames = 1  # change back to 1 since we have collapsed z dimension

        return mp

    def maxIntensityProjection(self):
        """
        maximum intensity projection i.e. highest pixel value over frames
        """
        image_data = self.loadAllFrames()
        mip = np.nanmax(image_data, axis=0, keepdims=True)
        # print("max intensity projection of dimensions", mip.shape)

        self.frames = 1  # change back to 1 since we have collapsed z dimension

        return mip
