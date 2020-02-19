"""
                   STITCH
                   ------

using MINIMUM SPANNING TREE for mosaic assembly

Class for stitching a regularly-spaced square grid of images
built on Mike's stitching code,
the window-based covariance checking is replaced with
phase correlation over the full overlapping region

REQUIRES stage positions to be provided for every image being stitched
Currently, this is found in the .xml or .inf files associated with each image

other useful parameters for our microscopes:
scalefactor = 7.678  # nemo & dory
theta = 0.0275 # nemo
theta = -0.0435  # dory

nigel 10 apr 2019
added minimum spanning tree and hdf5 image storage - 15 aug 19 nigel

License and readme found in https://github.com/khchenLab/split-fish
"""

import numpy as np
import os
import json
import h5py
import re
import pprint as pp
from collections import defaultdict
import datetime

from typing import Union, Tuple

import pandas as pd

# plotting
import seaborn as sns
import matplotlib.pyplot as plt

# SciPy
from scipy.spatial import cKDTree
import scipy.ndimage.interpolation as interpolation
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

# SciKit-Image
# from skimage.feature import register_translation
from utils.registrationFunctions import register_translation

# for File Dialog Window
import tkinter as tk
from tkinter import filedialog

from utils.readClasses import readDoryImg, readSpongebobImg
from utils.writeClasses import DaxWriter
from fileparsing.filesClasses import getFileParser
from fileparsing.fovGridFunctions import generateFovGrid


class Stitch(object):

    def __init__(self,
                 microscope_type: str = "Dory",  # either "Dory" or "spongebob", case insensitive
                 data_path: str = None,  # path where all the image files are stored
                 output_subdir: str = "output",  # subfolder where output (hdf5file, shifts.csv etc.) will be stored
                 processed_path: str = None,
                 basebit: Union[int, None] = None,
                 basetype: Union[str, None] = None,  # the base type to use for stitching
                 basehyb: Union[int, None] = None,  # the base hyb to use for stitching
                 roi: Union[int, None] = None,
                 stage_pixel_matrix: np.ndarray = np.array([[0, -8], [-8, 0]]),
                 # scaling factors for Nemo & Dory are in num pixels per micron - equivalent to 130 nm / pixel
                 theta: float = 0,  # camera angle: -0.016 dory default, 0.0275 for nemo
                 include_fovs: list = None,  # only use these fovs (if given, exclude fovs are ignored)
                 exclude_fovs: list = None,  # exclude these fovs but use all others
                 ):

        self.microscope_type = microscope_type.lower()
        self.data_path = data_path
        self.processed_path = processed_path
        self.roi = roi
        self.stage_pixel_matrix = stage_pixel_matrix
        self.theta = theta

        # Specify which bit or hyb/type to stitch by
        # ------------------------------------------

        self.basebit = basebit
        self.basetype = basetype
        self.basehyb = basehyb

        if basebit is not None:
            bit_str = f"bit{basebit}"
        elif basetype is not None and basehyb is not None:
            bit_str = f"hyb{basehyb}_{basetype}_"
        else:
            raise ValueError(
                f"Need to choose which image to stitch by. "
                f"To specify this, provide either\n"
                f"  (1) bit \n"
                f"  (2) hyb + type\n"
            )

        # Set stitched mosaic output path
        # -------------------------------

        self.script_time = datetime.datetime.now()
        time_str = self.script_time.strftime("%Y%m%d_%H%M")

        self.stitched_path = os.path.join(
            self.data_path, output_subdir, f"stitched_" + bit_str + time_str
        )
        if not os.path.isdir(self.stitched_path):
            os.mkdir(self.stitched_path)

        # initialize h5py file
        h5_filepath = os.path.join(self.stitched_path, "stitched.hdf5")
        self.h5file = h5py.File(h5_filepath, "a")

        # initialize attributes that will be filled later
        # -----------------------------------------------

        self.have_read_images = False
        self.edge_dict = None
        self.min_span_tree = None
        self.mst_by_row = None
        self.stitched_fovcoord_df = None  # dataframe of FOV coordinates

        print(
            "-" * 90 +
            f"\nInitializing stitch object ({bit_str})...\n"
            + "-" * 90 + "\n"
        )

        # Parser object to parse filenames in directory
        # ---------------------------------------------

        self.parser = getFileParser(data_path, self.microscope_type, )

        self.files_df_roi = self.parser.roiSubsetOfFilesDF(self.roi)

        self.files_dict = {}

        if basebit is not None:

            # Find all imagedata.hdf5 files in data folder (if using bit)
            # -----------------------------------------------------------

            imagedata_pattern = re.compile(
                r"fov_(\d+|\d+_\d+)_imagedata_iter(\d+).hdf5", flags=re.IGNORECASE,
            )

            for file in os.listdir(processed_path):
                match = re.match(imagedata_pattern, file)
                if match:
                    imagedata_filepath = os.path.join(processed_path, file)
                    self.files_dict[match.group(1)] = (imagedata_filepath, None)

            # check if we found the files
            assert len(self.files_dict) > 0, (
                f"No valid imagedata hdf5 files found in folder"
            )

            # set base hyb and type for grid generation
            # (choose first entry of dataframe if not provided)
            if self.basehyb is None:
                self.basehyb = int(self.files_df_roi["hyb"].values[0])
            if self.basetype is None:
                self.basetype = str(self.files_df_roi["type"].values[0])

        else:

            # Filter the dataframe (if using hyb/type combo)
            # ----------------------------------------------

            # include just the rows matching the basetype and hyb we want
            files_df_stitching_subset = self.files_df_roi[
                (self.files_df_roi["type"] == self.basetype) &
                (self.files_df_roi["hyb"] == self.basehyb)
                ]

            print(f"Truncated files dictionary:\n {files_df_stitching_subset}")

            # format: {FOV: (relative filepath, tiff frame), ...}
            for index, row in files_df_stitching_subset.iterrows():
                self.files_dict[f"{row['fov']}"] = (
                    os.path.join(data_path, row["file_name"]),
                    row["tiff_frame"],
                )

            # check if we found the files
            assert len(self.files_dict) > 0, (
                f"No Files matching type {self.basetype} and hyb {self.basehyb} "
                f"found in data folder!\n\n"
                f"Alternatives to try: \n"
                f"types:\t{self.parser.files_df['type'].unique()}\n"
                f"hybs:\t{self.parser.files_df['hyb'].unique()}\n"
            )

        # Get the subset of FOVs to be stitched
        # -------------------------------------

        # list of all FOVs in data path
        self.fovs = list(self.files_dict.keys())

        if include_fovs is not None:
            self.fov_subset = include_fovs
        elif exclude_fovs is not None:
            self.fov_subset = [
                fov for fov in self.fovs if fov not in exclude_fovs
            ]
        else:
            self.fov_subset = self.fovs

        print(
            f"List of FOVs in data folder:\n{self.fovs}\n"
            f"List of FOVs to stitch:\n{self.fov_subset}\n"
        )

        assert len(self.fov_subset) > 0, (
            f"FOVs selected do not match any in folder"
        )

        print(
            "Files to process:\n", json.dumps(self.files_dict, indent=2), "\n"
        )

        # Get FOV grid and stage/image coordinates
        # ----------------------------------------

        (self.fov_grid,
         self.image_coords,
         self.stage_coords) = generateFovGrid(
            self.files_df_roi,
            stage_pixel_matrix,
            fov_subset=self.fov_subset,
            hyb=self.basehyb,
            colour_type=self.basetype,
            plot_grid=False,
        )

        # flat version of fov grid for edge matrix
        self.fov_grid_flat = self.fov_grid.flat

        # maximum extents in x and y of the grid
        self.y_gridmax = self.fov_grid.shape[0]
        self.x_gridmax = self.fov_grid.shape[1]

        # Read the images and record image dimensions
        # -------------------------------------------

        self.imgs = self.readImages()

        # (assume all images have same dimensions)
        self.ydim = self.imgs[self.fovs[0]].shape[0]  # y dimensions of images
        self.xdim = self.imgs[self.fovs[0]].shape[1]  # x dimensions of images
        print(f"Images have dimensions {self.ydim} x {self.xdim}\n")

    def __enter__(self):
        """
        return the instance
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        close hdf5 file and save stitched FOV coordinates as a .csv file.
        """

        self.h5file.close()

        # save the calculated FOV coordinates to a csv file
        if self.stitched_fovcoord_df is not None:
            self.saveFovCoords()

    def readImages(self,
                   verbose: bool = True,
                   ) -> dict:
        """
        returns a dictionary of references
        to newly-created h5py datasets for each FOV
        {FOV: h5py dataset, ...}
        """
        images_dict = {}

        for fov in self.fovs:

            filepath = self.files_dict[fov][0]
            dataset_found = ""

            if self.basebit is not None:

                # Read basebit of imagedatafile
                with h5py.File(filepath, "a") as imgdata_h5:

                    img = None

                    datasets_to_check = ["fieldcorr", "raw"]

                    for dataset in datasets_to_check:

                        if dataset in imgdata_h5:
                            img = np.array(imgdata_h5[dataset][..., self.basebit])
                            # max intensity projection along all frames
                            # (currently should only have 1 frame)
                            img = np.nanmax(img, axis=0)
                            dataset_found = dataset
                            break

                    if img is None:
                        raise FileNotFoundError(
                            f"Could not find any of {datasets_to_check} arrays "
                            f"in {filepath}.\n"
                        )
            else:

                # For Dory/Nemo | .dax file format images
                # ---------------------------------------

                if self.microscope_type in ["dory", "nemo"]:
                    img = readDoryImg(filepath)

                # For spongebob | .tiff multiframe file format images
                # ------------------------------------------------

                elif self.microscope_type == "spongebob":

                    tiff_frame = self.files_dict[fov][1]
                    img = readSpongebobImg(filepath, tiff_frame)

                else:

                    raise ValueError(
                        f"Microscope type {self.microscope_type} not recognised. "
                        f"Must be dory, nemo or spongebob.\n"
                    )

            # write image to hdf5 file
            # ------------------------

            images_dict[fov] = self.h5file.create_dataset(
                fov, data=img,
            )

            if verbose:
                print(f"Read: {self.files_dict[fov]}\n"
                      f"      at FOV {fov} {dataset_found} which has\n"
                      f"      -stage coords {self.stage_coords[fov]}\n"
                      f"      -image coords {self.image_coords[fov]}\n")

        assert len(images_dict) > 0, "Unable to load images"

        self.have_read_images = True

        return images_dict

    #
    # ---------------------------------------------------------------------------------------
    #                              Image pairs Alignment
    # ---------------------------------------------------------------------------------------
    #

    def getShift(self,
                 fov1: str, fov2: str,  # the FOVs to be aligned
                 vertical: bool = False,
                 verbose: bool = True,
                 check_alignment: bool = False,
                 ):
        """
        get the relative shift between 2 fovs
        starting with shifts derived from stage movement
        and performing registration to correct for stage errors
        """
        return self._correctShift(
            np.array(self.imgs[fov1]),
            np.array(self.imgs[fov2]),
            *self._findShifts(fov1, fov2),
            check_alignment=check_alignment,
            vertical=vertical,
            verbose=verbose,
        )

    def _findShifts(self,
                    fov1: str, fov2: str,  # the FOVs to be aligned
                    verbose: bool = True,
                    ) -> Tuple[float, float]:
        """
        Find expected shift from STAGE POSITIONS *before* image registration
        calculates the shift between 2 fovs,
        with correction for slight camera rotation (theta)
        returns a tuple of (y-shift, x-shift)
        """
        yshift = self.image_coords[fov2][0] - self.image_coords[fov1][0]
        xshift = self.image_coords[fov2][1] - self.image_coords[fov1][1]

        # correct for rotation
        # --------------------
        # (positive theta is anticlockwise)

        xshift_corrected = int(round(xshift * np.cos(self.theta) - yshift * np.sin(self.theta)))
        yshift_corrected = int(round(xshift * np.sin(self.theta) + yshift * np.cos(self.theta)))

        if verbose:
            print(f"___ Finding shifts between {fov1} and FOV {fov2}: ___\n"
                  f"raw shifts:\t\tY = {yshift}, X = {xshift}\n"
                  f"corrected shifts:\tY = {yshift_corrected}, X = {xshift_corrected}")

        return yshift_corrected, xshift_corrected

    def _getOvelapRegions(self,
                          img1, img2,
                          yshiftstart, xshiftstart,
                          vertical=False,
                          check_dims_match=True,
                          ):
        """
          _______________
         |      |#|      |
         | img1 |#| img2 |
         |      |#|      |
         |______|_|______|  vertical = False
          ______
         | img1 |
         |______|
         |######|
         |------|
         | img2 |
         |______|  vertical = True

        """

        # This is a special case which is the same for vertical and horizontal
        if xshiftstart > 0 and yshiftstart > 0:
            overlap_region_1 = img1[yshiftstart:, xshiftstart:]
            overlap_region_2 = img2[:-yshiftstart, :-xshiftstart]

        else:

            if vertical:  # img2 BELOW img 1
                assert yshiftstart > 0, "image2 must be below image1"
                if xshiftstart == 0:
                    overlap_region_1 = img1[yshiftstart:, :]
                    overlap_region_2 = img2[:-yshiftstart, :]
                elif xshiftstart < 0:
                    overlap_region_1 = img1[yshiftstart:, 0:xshiftstart]
                    overlap_region_2 = img2[:-yshiftstart, -xshiftstart:]

            else:  # img2 to the RIGHT of img1
                assert xshiftstart > 0, "image2 must be to the right of image1"
                if yshiftstart == 0:
                    overlap_region_1 = img1[:, xshiftstart:]
                    overlap_region_2 = img2[:, :-xshiftstart]
                elif yshiftstart < 0:
                    overlap_region_1 = img1[0:yshiftstart, xshiftstart:]
                    overlap_region_2 = img2[-yshiftstart:, :-xshiftstart]

        if check_dims_match:
            assert np.array_equal(overlap_region_1.shape,
                                  overlap_region_2.shape), (
                f"Overlapping region dimensions {overlap_region_1.shape} "
                f"and {overlap_region_2.shape} do not match")

        return overlap_region_1, overlap_region_2

    def _correctShift(self,
                      img1, img2,  # the 2 images
                      yshiftstart, xshiftstart,  # the estimated y and x shifts to start with
                      vertical=False,
                      verbose=True,
                      check_alignment=True,
                      display_vmax=20000,
                      ):
        """
        refines the xshift and yshift between two images
        using some image registration algorithm
        currently implemented:
         (1) Phase correlation (Scikit image version, slightly modded)
         
        starts with an initial shift estimate (from stage shifts)
        image2 must be to the right of image1
                  ________
          _______|_       |
         |       || img2  |
         |  img1 ||_______|
         |________|         | y shift
          -------> x shift  v
        """
        (overlap_region_1,
         overlap_region_2) = self._getOvelapRegions(img1, img2,
                                                    yshiftstart, xshiftstart,
                                                    vertical=vertical,
                                                    check_dims_match=True)

        shift, error, diffphase = register_translation(overlap_region_1,
                                                       overlap_region_2,
                                                       50)

        if verbose:
            print("Shift between overlap regions: {}, {}".format(*shift))

        if check_alignment:

            # _____ original regions _____
            if vertical:
                figure_overlap, axes_overlap = plt.subplots(4, 1,
                                                            figsize=(9, 9))
            else:
                figure_overlap, axes_overlap = plt.subplots(1, 4,
                                                            figsize=(11, 9))
            axes_overlap.flat[0].imshow(overlap_region_1, vmax=display_vmax)
            axes_overlap.flat[1].imshow(overlap_region_2, vmax=display_vmax)

            # _____ corrected regions _____
            yshift_corrected = yshiftstart + int(shift[0])
            xshift_corrected = xshiftstart + int(shift[1])
            (overlap_region_1_updated,
             overlap_region_2_updated) = self._getOvelapRegions(img1, img2,
                                                                yshift_corrected,
                                                                xshift_corrected,
                                                                vertical=vertical,
                                                                check_dims_match=True)

            axes_overlap.flat[2].imshow(overlap_region_1_updated, vmax=display_vmax)
            axes_overlap.flat[3].imshow(overlap_region_2_updated, vmax=display_vmax)
            figure_overlap.suptitle("Original (1/2) and Corrected (3/4) overlap regions")

        return shift + (yshiftstart, xshiftstart), error

    #
    # ---------------------------------------------------------------------------------------
    #                              Mosaic Assembly methods
    # ---------------------------------------------------------------------------------------
    #

    def _getNeighbours(self,
                       grid_position,  # array-like, length 2
                       ):
        """
        get surrounding filled grid positions
        for a FOV in the FOV grid
        returns: (top, bottom, left, right)
                 FOV reference str, or
                 None if at the edge or if no FOV present at that position
        """
        assert len(grid_position) == 2, (
            f"Grid position was given {len(grid_position)} arguments."
            f"Must be (y position, x position)"
        )

        neighbours = [None, ] * 4

        # Top
        # ---
        if grid_position[0] != 0:
            neighbours[0] = self.fov_grid[grid_position[0] - 1,
                                          grid_position[1]]
        # Bottom
        # ------
        if grid_position[0] != self.y_gridmax - 1:
            neighbours[1] = self.fov_grid[grid_position[0] + 1,
                                          grid_position[1]]
        # Left
        # ----
        if grid_position[1] != 0:
            neighbours[2] = self.fov_grid[grid_position[0],
                                          grid_position[1] - 1]
        # Right
        # -----
        if grid_position[1] != self.x_gridmax - 1:
            neighbours[3] = self.fov_grid[grid_position[0],
                                          grid_position[1] + 1]

        # if no-image detected in adjacent position, change id to None
        return [nb if nb != "noimage" else None for nb in neighbours]

    def _getAllNeighbours(self,
                          verbose=True,
                          ):
        """
        generate a dictionary of 
        connected neighbours for every filled FOV
        format: { fov : [top, bottom, left, right], ... }
        """
        nb_dict = {}
        for y_pos in range(self.y_gridmax):
            for x_pos in range(self.x_gridmax):
                fov = self.fov_grid[y_pos, x_pos]
                if fov != "noimage":
                    nb_dict[fov] = self._getNeighbours((y_pos, x_pos))

        if verbose:
            print("____ Neighbours dictionary: ____")
            pp.pprint(nb_dict)

        return nb_dict

    def _plotEdges(self,
                   edge_dict,  # dictionary of edges
                   pairs: list,  # list of pairs
                   ax,  # axes to plot on
                   weight_norm=1,  # normalize weights by this
                   linecolour="blue",
                   linestyle="-",
                   alpha=0.8,
                   ):
        """
        Plot registration error metric as edges between FOVs
        on a given axes (ax)
        """
        for pair in pairs:
            coord1 = self.image_coords[pair[0]]
            coord2 = self.image_coords[pair[1]]
            if pair in edge_dict:
                weight = edge_dict[pair][1] / weight_norm * 2
            elif (pair[1], pair[0]) in edge_dict:
                weight = edge_dict[(pair[1], pair[0])][1] / weight_norm * 2
            else:
                raise ValueError(f"could not find pair {pair} in edge dictionary")
            ax.plot([coord1[1], coord2[1]],  # x start, end
                    [coord1[0], coord2[0]],  # y start, end
                    linestyle=linestyle,
                    color=linecolour,
                    linewidth=weight,
                    marker="o",
                    markersize=10,
                    markeredgecolor="red",
                    markeredgewidth=2,
                    alpha=alpha,
                    )

    def getAllShiftPairs(self,
                         plot_edges=True,
                         verbose=True,
                         ):
        """
        get all shifts between valid pairs
        in the form of a dictionary of FOV pair : (shift, error_metric)
        example entry:
          ('000_000', '000_001'): (array([2654.24,  -24.64]), 3860283169.9943023)
        """
        edge_dict = {}
        nb_dict = self._getAllNeighbours(verbose=True, )

        for fov in nb_dict:

            fov_below = nb_dict[fov][1]
            if fov_below is not None:
                edge_dict[(fov, fov_below)] = self.getShift(fov, fov_below,
                                                            vertical=True)
            fov_right = nb_dict[fov][3]
            if fov_right is not None:
                edge_dict[(fov, fov_right)] = self.getShift(fov, fov_right,
                                                            vertical=False)

        errors = np.array([edge_dict[pair][1] for pair in edge_dict])
        print(errors)
        min_err = np.amin(errors)
        max_err = np.amax(errors)
        row, col, weight = [], [], []
        for pair in edge_dict:
            row.append(np.where(self.fov_grid_flat == pair[0])[0][0])
            col.append(np.where(self.fov_grid_flat == pair[1])[0][0])
            weight.append(-1 * edge_dict[pair][1])

        if verbose:
            print(f"Rows:\n{row}"
                  f"Columns:\n{col}"
                  f"Weight:\n{weight}\n")

        num_fovs = self.fov_grid.size
        self.weight_graph = csr_matrix(
            (np.array(weight), (np.array(row), np.array(col))),
            shape=(num_fovs, num_fovs),
        )
        # print(f"Full weight graph = \n{self.weight_graph.toarray()}")
        self.min_span_tree = minimum_spanning_tree(self.weight_graph)

        # symmetrize the min_span_graph,
        # then split into rows
        # for easy referencing when calculating coordinates
        # FIXME: Seems to be not sparse efficient

        msg_rows, msg_cols = self.min_span_tree.nonzero()
        self.min_span_tree[msg_cols, msg_rows] = self.min_span_tree[msg_rows, msg_cols]
        self.mst_by_row = np.split(self.min_span_tree.indices,
                                   self.min_span_tree.indptr)[1:-1]
        print(f"min span tree by row:\n{self.mst_by_row}")

        fig_graph, ax_graph = plt.subplots(nrows=1, ncols=2,
                                           figsize=(12, 8))
        ax_graph[0].imshow(self.weight_graph.toarray())
        ax_graph[1].imshow(self.min_span_tree.toarray())

        mingraph_rows, mingraph_cols = self.min_span_tree.nonzero()
        mingraph_pairs = [
            (self.fov_grid_flat[row], self.fov_grid_flat[col])
            for (row, col)
            in zip(mingraph_rows, mingraph_cols)
        ]
        firstvalue = self.min_span_tree[mingraph_rows[0], mingraph_cols[0]]
        print(f"Full weight graph num edges= {self.weight_graph.nnz}\n"
              f"Min graph num edges = {self.min_span_tree.nnz}\n"
              f"data:\n{self.min_span_tree.data}\n"
              f"rows = {mingraph_rows}\n"
              f"cols = {mingraph_cols}\n"
              f"firstvalue = {firstvalue}\n")

        if plot_edges:
            sns.set_style("darkgrid")
            fig, ax = plt.subplots()
            self._plotEdges(edge_dict, list(edge_dict.keys()),
                            ax,
                            weight_norm=min_err,
                            )
            self._plotEdges(edge_dict, mingraph_pairs,
                            ax,
                            weight_norm=min_err,
                            linecolour="red", linestyle="-",
                            alpha=0.9,
                            )

            ax.set_aspect('equal', 'box')

            # flip the y axis
            # ---------------
            limits = ax.axis()
            ax.axis([limits[0], limits[1],  # x limits stay the same
                     limits[3], limits[2]],  # y limits swapped
                    )

        self.edge_dict = edge_dict

        return edge_dict

    def getAllCoords(self,
                     verbose=True,
                     ):
        """
        get the coords of all FOVs
        as a pandas dataframe:
         index : the FOV reference string
         col 1 : the y pixel-coordinate position
         col 2 : the x pixel-coordinate position

        zeros all positions according to the top-left FOV
        """

        # get starting FOV by finding the best-registered edge
        # and using one of the 2 FOVs from that edge
        fov_start_idx = self.min_span_tree.argmin() // self.fov_grid.size
        fov_start = self.fov_grid_flat[fov_start_idx]

        coord_list = [[fov_start, 0.0, 0.0], ]

        if verbose:
            print(f"\nStarting with FOV {fov_start} "
                  f"with index {fov_start_idx} ...\n"
                  f"coord list intialized as {coord_list}\n")

        # recursively run through all branches of the spanning tree
        coord_list = self._runVertex(coord_list,
                                     fov_start_idx,
                                     (0.0, 0.0),
                                     verbose=verbose,
                                     )
        if verbose:
            print(f"Final coord list:")
            pp.pprint(coord_list)

        # Convert list of lists to Dataframe
        stitched_fovcoord_df = pd.DataFrame(coord_list,  # list of [fov, y ,x] lists
                                            columns=["fov", "y_coord", "x_coord"],
                                            )
        if verbose:
            print(f"\nCoord dataframe original:\n{stitched_fovcoord_df}\n")

        # zero all coordinates by the top left FOV
        # i.e. top left of canvas is 0
        #             no image here
        #         __|_____________0
        #           |     #######
        #           |     ####### |
        # no image  |############ |
        # here      |########     V
        #           |########
        #           0  ---->

        stitched_fovcoord_df["y_coord"] -= stitched_fovcoord_df["y_coord"].min()
        stitched_fovcoord_df["x_coord"] -= stitched_fovcoord_df["x_coord"].min()

        if verbose:
            print(f"Coord dataframe zeroed:\n{stitched_fovcoord_df}")

        self.stitched_fovcoord_df = stitched_fovcoord_df

        return stitched_fovcoord_df

    def _runVertex(self,
                   coord_list,  # list of [fov,y-coord, x-coord]
                   current_fov_idx,
                   current_fov_coord,
                   previous_fov_idx=None,
                   verbose=False,
                   ):
        """
        Note: each vertex is an FOV
        A recursive function that
        either:
        returns the same coord list if it has reached a terminal branch
        or:
        checks any new vertices attached to the current vertex,
        adds the cumulative position of the new vertex
        and runs the same function on that vertex
        """
        current_fov = self.fov_grid_flat[current_fov_idx]
        current_y, current_x = current_fov_coord

        if previous_fov_idx is not None:
            previous_fov = self.fov_grid_flat[previous_fov_idx]
        else:
            previous_fov = None

        # list/numpy array of vertices connected to the current vertex
        vertices = self.mst_by_row[
            np.where(self.fov_grid_flat == current_fov)[0][0]
        ]

        if verbose:
            print(f"Current FOV: {current_fov}\t|\t"
                  f"Connected vertices: {vertices}\n")

        # decide whether to check vertices 
        # or just return coord_list (i.e. reached end of branch)
        check_new = (vertices.shape[0] > 1
                     or
                     (previous_fov == None
                      and vertices.shape[0] > 0))

        if check_new:
            for fov_idx in vertices:
                if fov_idx != previous_fov_idx:
                    next_fov = self.fov_grid_flat[fov_idx]
                    if (current_fov, next_fov) in self.edge_dict:
                        y_shift, x_shift = tuple(
                            self.edge_dict[(current_fov, next_fov)][0]
                        )
                    elif (next_fov, current_fov) in self.edge_dict:
                        y_shift, x_shift = tuple(
                            self.edge_dict[(next_fov, current_fov)][0] * -1
                        )
                    else:
                        raise KeyError(f"Pair ({current_fov}, {next_fov})"
                                       f" not found in edge dictionary")

                    fov_y = current_y + y_shift
                    fov_x = current_x + x_shift

                    coord_list.append([next_fov, fov_y, fov_x])

                    coord_list = self._runVertex(coord_list,
                                                 fov_idx, (fov_y, fov_x),
                                                 previous_fov_idx=current_fov_idx,
                                                 verbose=verbose,
                                                 )

        return coord_list

    def saveFovCoords(self,
                      filename: str = "FOV_globalcoords.tsv",
                      ) -> None:
        """
        save a list of global coords for each FOV as a .tsv file
        """

        assert isinstance(self.stitched_fovcoord_df, pd.DataFrame), (
            "FOV stitched coordinates dataframe is not a pandas dataframe.\n"
            "Unable to save."
        )

        self.stitched_fovcoord_df.to_csv(
            os.path.join(self.stitched_path, filename), sep="\t"
        )

    def showStitched(self,
                     h5_dataset_name: str = "stitched",  # h5py dataset reference
                     downsample: int = 4,
                     display_vmax: Union[int, float] = 40000,
                     title="Stitched image",
                     ) -> None:
        """
        show stitched image
        """
        full_canvas = self.h5file[h5_dataset_name]

        sns.set_style("white")
        fig_canvas, ax_canvas = plt.subplots(figsize=(8, 8))
        ax_canvas.imshow(np.array(full_canvas[::downsample, ::downsample]),
                         vmax=display_vmax)
        fig_canvas.tight_layout(rect=(0, 0, 1, 0.95))
        fig_canvas.suptitle(title)

    def assembleCanvas(self,
                       subpixel: bool = True,
                       subpixel_interp_order: int = 1,
                       verbose: bool = True,
                       ):
        """
        Using shifts from coords dataframe
        Insert the images in approprate place in an empty matrix

        Parameters
        ----------
        subpixel:
          whether to do subpixel interpolation
        subpixel_interp_order:
          default to linear interpolation (0-NN, 1-linear, 2-cubic)
        """
        assert self.stitched_fovcoord_df is not None, (
            f"Coords Dataframe not found.\n"
            f"need to calculate shifts first."
        )

        max_shift_y = self.stitched_fovcoord_df["y_coord"].max()
        max_shift_x = self.stitched_fovcoord_df["x_coord"].max()

        # Initialize large empty hdf5 dataset as a canvas to place images in
        # ------------------------------------------------------------------

        full_canvas = self.h5file.create_dataset(
            "stitched",
            (int(max_shift_y) + 1 + self.ydim,
             int(max_shift_x) + 1 + self.xdim),
            dtype="f",
        )

        if verbose:
            print(f"Full canvas\t-\tshape: {full_canvas.shape}, "
                  f"datatype: {full_canvas.dtype}\n")

        for df_row in range(len(self.stitched_fovcoord_df.index)):
            fov, ycoord, xcoord = self.stitched_fovcoord_df.loc[
                df_row, ["fov", "y_coord", "x_coord"]
            ]

            top_index = int(ycoord)
            left_index = int(xcoord)
            bottom_index = top_index + self.ydim
            right_index = left_index + self.xdim

            if verbose:
                print(f"FOV {fov}:\n"
                      f"top={top_index}, left={left_index}, "
                      f"bottom={bottom_index}, right={right_index}\n")

            # Insert into canvas
            # ------------------

            img = np.array(self.imgs[fov])
            if subpixel:  # apply subpixel shift
                img = interpolation.shift(img,
                                          (ycoord % 1, xcoord % 1),
                                          order=subpixel_interp_order,
                                          mode="constant",
                                          cval=0.0,
                                          prefilter=True)

            full_canvas[
            top_index: bottom_index, left_index:right_index
            ] = np.maximum(
                img,
                full_canvas[
                top_index: bottom_index, left_index:right_index
                ]
            )
            # FIXME: the above is inefficent (lots of maximizing over zero regions)
            # FIXME: but it should work for now

        return full_canvas

    def saveDax(self,
                full_canvas: h5py.Dataset = None,  # h5py dataset reference
                filename: str = "stitched_",
                add_timestr: bool = True,
                ):
        """
        save the stitched image as a .dax file
        """
        if full_canvas is None:
            full_canvas = self.h5file["stitched"]

        timestr = ""
        if add_timestr:
            timestr += datetime.datetime.now().strftime("_%Y%m%d_%H%M")

        full_savepath = os.path.join(self.stitched_path,
                                     filename + timestr + ".dax")

        with DaxWriter(full_savepath) as writer:
            writer.addFrame(np.array(full_canvas))

    #
    # ---------------------------------------------------------------------------------------
    #                              Meging of Spot coordinates Across FOVs
    # ---------------------------------------------------------------------------------------
    # These methods merge coordinates across FOVs
    # according to the shifts calculated for each FOV.
    #

    def combineSpots(self,
                     combined_filename: str = "coords_combined",
                     iteration: int = 0,
                     remove_overlap: bool = True,
                     overlap_distance: float = 1.4,
                     fov_list: list = None,
                     verbose: bool = True,
                     ) -> dict:
        """
        uses stitched global FOV coordinates to combine spots coordinates
        from individual FOVs into combined set of spot coordinates
        New coordinates are referenced to the top left of the stitched canvas

        OK if coords hdf5 files are a subset of the FOVs you want to stitch.
        This will just stitch those FOVs for which coords hdf5 files are found.

        Parameters
        ----------
        outputpath: str
            path where all the individual FOV coordinate hdf5 files are stored
            Coordinates files must have the following signature:
            coords_FOV_([_0-9]+)_iter([0-9]+).hdf5
        combined_filename: str
            starting string for savename of the combined coordinates hdf5 file
        iteration: int
            iteration over which to combine spots, since there may
            be multiple iterations in the same folder
        remove_overlap: bool
            whether to merge overlapping gene spots in overlap regions
        overlap_distance: float
            maximum euclidean distance between genes (in pixels)
            for merging spots
        fov_list: list
            list of FOVs to combine.
            If not given, defaults to instances' fov_subset

        :returns
            dictionary of combined coordinates arrays keyed by each gene
        """

        assert self.stitched_fovcoord_df is not None, (
            f"Coords Dataframe not found.\nNeed to calculate shifts first."
        )

        # for each gene, store lists of coord arrays from each coord file
        coord_dict = defaultdict(list)
        # for each gene, store hdf5 attributes from first coord file to contain the gene
        gene_attributes = {}

        filename_pattern = re.compile(
            r"FOV_([_0-9]+)_coord_iter([0-9]+).hdf5", flags=re.IGNORECASE
        )

        if fov_list is None:
            # If FOV list is not provided,
            # default to the instances' FOV subset
            fov_list = self.fov_subset
        else:
            # check that all elements of provided list have been stitched
            fov_list = [fov for fov in fov_list if fov in self.fov_subset]

        # Find coords files, add shifts and concatenate all coords
        # --------------------------------------------------------

        matches_found_counter = 0

        for filename in os.listdir(self.processed_path):
            # NOTE: Assumes there is only one file for each FOV/iteration

            match = re.match(filename_pattern, filename)

            if match and int(match.group(2)) == iteration:

                matches_found_counter += 1

                fov = match.group(1)

                if fov in fov_list:

                    shift = self.stitched_fovcoord_df.loc[
                                self.stitched_fovcoord_df["fov"] == fov,
                                ["y_coord", "x_coord"]
                            ].values[0, :]

                    print(f"Shift for FOV {fov} = {shift}")

                    full_filepath = os.path.join(self.processed_path, filename)

                    with h5py.File(full_filepath, 'r') as f:

                        for gene in f.keys():

                            # copy attributes of dataset to gene_attributes dictionary
                            # --------------------------------------------------------

                            if gene not in gene_attributes:

                                attributes_dict = {}
                                for attribute in f[gene].attrs:
                                    attributes_dict[attribute] = f[gene].attrs[attribute]

                                gene_attributes[gene] = attributes_dict

                            # Shift and record coordinates
                            # ----------------------------

                            print(
                                f"dataset size for {gene}, FOV {fov}: {f.get(gene).size}"
                            )

                            if f[gene].shape[0] != 0:

                                coords = np.array(f[gene])

                                # add FOV global shift to the y and x coordinates
                                # (which are in the 2nd and 3rd columns) of all spots
                                coords[:, 1:3] += shift[np.newaxis, :]

                                coord_dict[gene].append(coords)

                                if verbose:
                                    print(f"shifting coords of gene:{gene}")

        if matches_found_counter == 0:
            raise FileNotFoundError(
                f"No coord hdf5 files found for iteration {iteration}"
            )

        # if verbose:
        #     print("coord dict:", coord_dict)

        for gene in coord_dict:
            coord_dict[gene] = np.vstack(coord_dict[gene])

        # if verbose:
        #     print("coord dict concatenated:", coord_dict)

        if remove_overlap:
            combined_filename += "_merged"

        timestr = datetime.datetime.now().strftime("_%Y%m%d_%H%M")

        combined_filepath = os.path.join(
            self.processed_path,
            combined_filename + f"_iter_{iteration}_{timestr}.hdf5"
        )

        # Generate combined hdf5 file while removing the overlapping spots
        # ----------------------------------------------------------------

        with h5py.File(combined_filepath) as combined_h5:

            for gene in coord_dict:

                gene_array = coord_dict[gene]

                if verbose:
                    print(f"{gene}:\n{gene_array}\n")

                if remove_overlap:

                    if verbose:
                        print(f"Original shape of coord array for "
                              f"{gene}: {gene_array.shape}")

                    tree = cKDTree(gene_array[:, 1:3])
                    rows_to_fuse = tree.query_pairs(r=overlap_distance)

                    rows_to_delete = []

                    for pair in list(rows_to_fuse):
                        # average spot data between the 2 overlapping spots
                        mean_coord = (gene_array[pair[0], :] + gene_array[pair[1], :]) / 2

                        # set first entry to mean, assign second entry to be deleted
                        gene_array[pair[0], :] = mean_coord
                        rows_to_delete.append(pair[1])

                    gene_array = np.delete(gene_array, rows_to_delete, axis=0)

                    if verbose:
                        print(
                            f"Shape of trimmed coord array for {gene}: {gene_array.shape}\n"
                        )

                combined_h5.create_dataset(gene, data=gene_array)

                # re-record attributes
                # --------------------

                for attribute in gene_attributes[gene]:
                    combined_h5[gene].attrs[attribute] = gene_attributes[gene][attribute]
                combined_h5[gene].attrs["pixel_count"] = np.sum(gene_array[:, 3])
                combined_h5[gene].attrs["spot_count"] = gene_array.shape[0]

        return coord_dict


#
#
#                                            Script
# =====================================================================================================
#
#
#

if __name__ == "__main__":

    #
    # Ask for output directory containing image files from all fields of view
    #

    root = tk.Tk()
    root.withdraw()
    data_path = filedialog.askdirectory(
        title="Choose main data directory"
    )
    # NOTE: if you just want to stich the image, click "cancel" on the second dialog box
    processed_path = filedialog.askdirectory(
        title="Choose directory with processed images and spots data"
    )
    root.destroy()

    # Select parameter set
    # --------------------

    datatype = "jolene"

    #
    # Define the different sets of parameters
    # ------------------------------------------------------------
    #

    if datatype == "jolene":
        microscope_type = "Dory"
        basebit = 14
        basetype = "Cy5"
        # basehyb = 3
        basehyb = 0  # for ovary data
        display_vmax = 25000
        stage_pixel_matrix = 8 * np.array([[0, -1], [-1, 0]])
    else:
        raise ValueError("No params provided")

    #
    # --------------------------------------------------------------
    #

    with Stitch(
            microscope_type=microscope_type,
            data_path=data_path,
            processed_path=processed_path,
            basebit=basebit,
            basetype=basetype,
            basehyb=basehyb,
            stage_pixel_matrix=stage_pixel_matrix,
            # exclude_fovs=["002_002", "002_003"],
            # include_fovs=["001_000","002_000",
            #               "001_001","002_001","003_001"]
    ) as stitcher:

        pp.pprint(stitcher.getAllShiftPairs())
        stitcher.getAllCoords()

        fullcanvas = stitcher.assembleCanvas(subpixel=True)
        stitcher.showStitched(
            downsample=4,
            display_vmax=display_vmax,
        )

        # if you want to save an additional .dax file
        # -------------------------------------------

        # stitcher.saveDax()

        # Test combine spots
        # ------------------

        if processed_path != "":
            stitcher.combineSpots(
                remove_overlap=True,
                overlap_distance=4,
                iteration=0,
                verbose=True,
            )

    plt.show()
