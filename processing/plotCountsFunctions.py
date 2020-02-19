"""
Calculates and displays correlation plots of gene counts to FPKM data,
by reading files that list spot/pixel counts per gene.

- initialized by passing in the path of the output folder.
  - finds the relevant data (optionally crosstalk data) files in the output folder
    and processes them accordingly
- contains methods to calculate correlations from the area/number-of-regions counts

NOTE:
There are 2 types of dataframe among these functions
counts dataframes (df, summed_df, combined df):
    contains counts per gene for single FOV or over multiple FOVs
results datafrmeas (results_df):
    stores result of running various statistics on each/all the FOVs

nigel chou Dec 2018
updated Mar 2019

License and readme found in https://github.com/khchenLab/split-fish
"""

import os
import time
import re  # used to parse the column headings
import h5py

from typing import Tuple, Union, Dict, List

import numpy as np

import matplotlib.pyplot as plt
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigCanvas
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigCanvas

from matplotlib import gridspec
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import seaborn as sns

# from Scipy
from scipy.stats.stats import pearsonr  # for fpkm correlation


def calcLogCorrelation(array1: np.ndarray,
                       array2: np.ndarray,
                       ) -> Tuple[float, float]:
    """
    calculate log-correlation between 2 arrays using scipy's pearsonr
     - usually a FPKM value array and some kind of count
    returns (correlation, p_value) same as scipy's pearsonr
    """
    print(f"array1 type = {array1.dtype}"
          f"array2 type = {array2.dtype}")

    # mask out 0 and non-finite values of array
    combined_mask = np.logical_and(np.logical_and(np.isfinite(array1), array1 > 0),
                                   np.logical_and(np.isfinite(array2), array2 > 0))

    return pearsonr(np.log10(array1[combined_mask]), np.log10(array2[combined_mask]))


def _calcCorrAndCountFromDF(df: pd.DataFrame,
                            x_column: str,
                            y_column: str,
                            ) -> Dict[str, float]:
    """
    Calculate the correlation, p_value and total_count
    from 2 columns of a dataframe to correlate
    (usually x column is FPKM and y column is count)
    returns as dictionary
    """
    corr, pval = calcLogCorrelation(df[x_column].values,
                                    df[y_column].values)
    total_count = df[y_column].sum()

    return {"correlation": corr,
            "p_value": pval,
            "total_count": total_count}


def _sortAndCalcConfidence(df: pd.DataFrame,
                           col_to_sort: str = "count",
                           verbose: bool = True,
                           ) -> Tuple[pd.DataFrame, dict]:
    """
    Modify dataframe and get gene/blank confidence ratios

    returns:
    (1) Dataframe that is:
     - sorted
     - includes an additional "genes_to_blank" column

    (2) dictionary with:
     - percent_above_blank
     - gene_blank_ratio
    """

    sorted_df = df.sort_values(by=[col_to_sort],
                               ascending=False, inplace=False)

    # Add extra column indicating if each row is a gene or a blank
    # ------------------------------------------------------------

    sorted_df["gene_or_blank"] = "gene"
    sorted_df.loc[
        sorted_df["gene_names"].str.contains("blank", case=False, regex=False),
        "gene_or_blank"
    ] = "blank"

    if verbose:
        print("Sorted dataframe for barplot:\n", sorted_df)

    # set up groupby object
    # ---------------------

    groupby_gene_or_blank = sorted_df.groupby("gene_or_blank")

    #
    # get the percentage of genes that are above blanks
    # -------------------------------------------------
    #
    group_counts = groupby_gene_or_blank.count()
    gene_count, blank_count = group_counts.loc[["gene", "blank"],
                                               col_to_sort]

    try:
        # get index of largest blank
        sorted_geneblank_list = sorted_df["gene_or_blank"].tolist()
        first_blank_idx = sorted_geneblank_list.index("blank")
        percent_above_blank = first_blank_idx / gene_count
    except ValueError:
        first_blank_idx = None
        percent_above_blank = 1

    percent_above_blank *= 100

    #
    # get ratio of median gene over median blank
    # ------------------------------------------
    #

    group_medians = groupby_gene_or_blank.median()
    genes_median = group_medians.loc["gene", col_to_sort]
    try:
        blanks_median = group_medians.loc["blank", col_to_sort]
        gene_blank_ratio = genes_median / blanks_median
    except KeyError:
        gene_blank_ratio = np.inf

    if verbose:
        print(f"\nNumber of genes:  {gene_count}\n"
              f"Number of blanks: {blank_count}\n\n"
              f"Group counts:\n{groupby_gene_or_blank.count()}\n\n"
              f"Gene/blank list:\n{sorted_geneblank_list}\n\n"
              f"First blank index: {first_blank_idx}\n"
              f"% above blank: {percent_above_blank:0.3f}\n\n"
              f"Median groups:\n{group_medians}\n\n"
              f"Gene-to-blank median ratio: {gene_blank_ratio:0.3f}\n")

    return sorted_df, {"percent_above_blank": percent_above_blank,
                       "gene_blank_ratio": gene_blank_ratio, }


def countSpotsInH5CoordsFiles(hdf5_filepath_list: Union[str, List[str]],
                              verbose: bool = True,
                              ) -> pd.DataFrame:
    """
    parse a list of hdf5 coordinates files into a dataframe
    NOTE: Gene names and FPKM come from the first entry found in any of the hdf5 files
    """

    def _H5DatasetToDict(dataset: h5py.Dataset,
                         ) -> Dict[str, Union[str, int, np.ndarray]]:
        """
        Generate a dictionary of FPKM-data, spot-counts, and 
        parameters (excluding coordinates) from a gene dataset of a h5py coordinates file.
        """

        try:
            FPKM_data = dataset.attrs["FPKM_data"]
        except KeyError:
            print(f"hdf5 file does not contain FPKM value"
                  f"for gene {gene}")
            raise

        if "spot_count" in dataset.attrs:
            spot_count = dataset.attrs["spot_count"]
        else:
            # 1st dimension of array - each row is a spot
            spot_count = dataset.shape[0]

        if spot_count != 0:
            # Sum everything after along spots dimension
            params_array = np.sum(dataset, axis=0)
        else:
            # equal length array with all zeros
            params_array = np.zeros((dataset.shape[1],))

        return {"FPKM_data": FPKM_data,
                "spot_count": spot_count,
                "params_array": params_array}

    if isinstance(hdf5_filepath_list, str):
        hdf5_filepath_list = [hdf5_filepath_list, ]

    # Read data from coords files into a dictionary
    # ---------------------------------------------

    all_gene_counts = {}

    for hdf5_filepath in hdf5_filepath_list:

        with h5py.File(hdf5_filepath, "r") as hdf5_file:

            for gene in hdf5_file:

                gene_dataset = hdf5_file[gene]
                gene_dict = _H5DatasetToDict(gene_dataset)

                if gene not in all_gene_counts:
                    all_gene_counts[gene] = gene_dict
                else:
                    all_gene_counts[gene]["spot_count"] += gene_dict["spot_count"]
                    # add up the params [number of pixels, closest distance etc..]
                    all_gene_counts[gene]["params_array"] += gene_dict["params_array"]

    # Generate dataframe
    # ------------------

    df_columns = ["FPKM_data",
                  "spot_count",
                  "pixel_count",
                  "mean_pixels",
                  "mean_closest_distance",
                  "mean_onbit_intensity",
                  ]

    df = pd.DataFrame(columns=df_columns, index=list(all_gene_counts.keys()), dtype=np.float64)

    for gene in df.index:

        df.loc[gene, "FPKM_data"] = all_gene_counts[gene]["FPKM_data"]

        spot_count = all_gene_counts[gene]["spot_count"]
        df.loc[gene, "spot_count"] = spot_count

        if spot_count == 0:
            df.loc[gene, "pixel_count":"mean_onbit_intensity"] = [0., 0., np.inf, 0.]
        else:
            params_array = all_gene_counts[gene]["params_array"]

            df.loc[gene, "pixel_count"] = params_array[3]
            df.loc[gene, "mean_pixels"] = params_array[3] / spot_count
            df.loc[gene, "mean_closest_distance"] = params_array[4] / spot_count
            df.loc[gene, "mean_onbit_intensity"] = np.mean(params_array[5:]) / spot_count

    df.index.name = "gene_names"
    df.reset_index(inplace=True)

    if verbose:
        print(f"df of counts from:\n{hdf5_filepath_list}\n\n{df}\n"
              f"Summary:\n{df.info()}")

    return df


def listH5PathsByFov(path: str,
                     return_fullpath: bool = True,
                     h5filetype: str = "coord",
                     iteration: int = 0,
                     verbose: bool = True,
                     ) -> Dict[str, str]:
    """
    get a list of hdf5 coordinates or imagedata files within a directory
    
    regex: compiled regular expression pattern
        (optional) regular expression for filename pattern.
        If None, will use default pattern.
    """

    if h5filetype == "coord":
        regex = re.compile(
            r"FOV_?([0-9]+|[0-9]+[_x][0-9]+)_coord_iter([0-9]+)\.hdf5"
        )
    elif h5filetype == "imagedata":
        regex = re.compile(
            r"FOV_?([0-9]+|[0-9]+[_x][0-9]+)_imagedata_iter([0-9]+)\.hdf5"
        )
    else:
        raise ValueError(f"h5 filetype: {h5filetype} not recognised!\n")

    all_files = os.listdir(path)

    print(all_files)

    hdf5_coord_file_dict = {}

    for file in all_files:

        match = regex.search(file)

        if match and int(match.group(2)) == iteration:

            fov = match.group(1)

            filepath = match.group(0)
            if return_fullpath:
                filepath = os.path.join(path, filepath)

            hdf5_coord_file_dict[fov] = filepath

    if verbose:
        print("\nFound the following hdf5 coord files:\n" + "-" * 37)
        [print(f"\tFOV {fov} : {hdf5_coord_file_dict[fov]}") for fov in hdf5_coord_file_dict]
        print("")

    return hdf5_coord_file_dict


#
#                                           Main Functions
# ------------------------------------------------------------------------------------------------------
#


def calcAllFovsFromHdf5(hdf5_filepath_or_df_dict: Dict[str,
                                                       Union[str, pd.DataFrame]],
                        count_type: str = "spot_count",
                        return_combined_fovs_results: bool = True,
                        iteration: int = 0,
                        drop_genes: List[str] = None,
                        df_savepath: str = None,
                        verbose: bool = True,
                        ) -> pd.DataFrame:
    """
    calculates the pearson correlation and p_values of all the fovs being processed
    returns (and saves) the summary data in another dataframe
    """

    def _resultsTupleFromDF(df: pd.DataFrame,
                            count_type: str,
                            drop_genes: List[str],
                            results_order: List[str],
                            ) -> tuple:
        """
        calculate the results parameters:
        correlation, p_value, total_count, percent_above_blank, gene_blank_ratio
        from a dataframe of gene counts / FPKM values
        """

        if isinstance(drop_genes, list):
            df.drop(drop_genes, axis=0, inplace=True)

        corr_count_dict = _calcCorrAndCountFromDF(df,
                                                  "FPKM_data",
                                                  count_type)

        _, confidence_dict = _sortAndCalcConfidence(df,
                                                    col_to_sort=count_type)

        # concatenate correlation and confidence dictionaries
        results_dict = {**corr_count_dict, **confidence_dict}
        results_keys = list(results_dict.keys())

        results_list = [results_dict[result]
                        for result in results_order
                        if result in results_keys]

        return tuple(results_list)

    # set up empty 'results' dataframe to record calculated results from each FOV
    # ---------------------------------------------------------------------------

    df_columns = ["fov",
                  "correlation", "p_value", "total_count",
                  "percent_above_blank", "gene_blank_ratio"]
    num_df_columns = len(df_columns)
    results_df = pd.DataFrame(columns=df_columns)

    # Calculate for each FOV
    # ----------------------

    for fov in hdf5_filepath_or_df_dict:

        fov_value = hdf5_filepath_or_df_dict[fov]
        if isinstance(fov_value, str):
            df = countSpotsInH5CoordsFiles(fov_value, verbose=verbose)
        elif isinstance(fov_value, pd.DataFrame):
            df = fov_value
        else:
            raise TypeError(f"hdf5_filepath_or_df_dict value for "
                            f"FOV {fov} is type <{type(fov_value)}>.\n"
                            f"Must be either str (filepath) or pandas dataframe.")

        results_tuple = _resultsTupleFromDF(df, count_type, drop_genes, df_columns[1:])

        # append a row to the bottom of the correlation dataframe
        results_df.loc[len(results_df)] = (fov,) + results_tuple

    # Combined results from all FOVs
    # ------------------------------

    if return_combined_fovs_results:
        df = countSpotsInH5CoordsFiles(list(hdf5_filepath_or_df_dict.values()),
                                       verbose=verbose)

        combined_results_tuple = _resultsTupleFromDF(df, count_type, drop_genes, df_columns[1:])

        results_df.loc[len(results_df)] = ("all",) + combined_results_tuple

    # Save dataframe
    # --------------

    if df_savepath is not None:
        results_df_name = (f"correlations_summary_iter{iteration}"
                           f"_{time.strftime('%Y%m%d_%H%M%S')}.tsv")
        results_df.to_csv(os.path.join(df_savepath, results_df_name), sep="\t")

    if verbose:
        print("Results Dataframe: \n" + "-" * 24)
        print(results_df.tail(15), "\n")

    return results_df


def generateAndPlotResultsGrid(results_df: pd.DataFrame,
                               fov_grid: np.ndarray,
                               columns_to_plot: Dict[str,
                                                     Tuple[np.dtype,
                                                           int, int,
                                                           str,
                                                           int]] = None,
                               iteration: int = 0,
                               plot_grids: bool = True,
                               fig_savepath: str = "",
                               verbose: bool = True,
                               ) -> Dict[str, np.ndarray]:
    """
    generate and return a grid of correlations and total_counts
    only looks at 'regions' rows
    """

    if columns_to_plot is None:
        # use defaults
        columns_to_plot = {"correlation": (np.float64, 0, 1, "RdYlGn", None, 12),
                           "total_count": (np.int32, None, None, "hot", None, 8),
                           "percent_above_blank": (np.float64, 0, 100, "RdYlGn", None, 14),
                           "gene_blank_ratio": (np.float64, None, None, "RdYlGn", 1., 12),
                           }

    if fov_grid.ndim != 2:
        raise NotImplementedError("FOV grid has {fov_grid.ndim} dimensions.\n"
                                  "Grids for more than 2 dimensions not yet implemented.")

    assert all([column in results_df.columns for column in columns_to_plot]), (
        f"Columns to plot:\n{columns_to_plot}\n"
        f"are not a subset of results dataframe "
        f"columns:\n{results_df.columns}"
    )

    # Generate the empty grids
    # ------------------------

    results_grid_dict = {}

    for column in columns_to_plot:
        results_grid_dict[column] = np.zeros_like(
            fov_grid, dtype=columns_to_plot[column][0]
        )

    # fov_list = results_df["fov"].tolist()
    results_df.set_index("fov", inplace=True)

    for fov in results_df.index:

        # find index of this FOV in the FOV grid
        # --------------------------------------

        grid_coordinates = np.argwhere(fov_grid == fov)
        num_coordinates_found, grid_dimensions = grid_coordinates.shape

        assert grid_dimensions == 2, (f"Grid dimensions is {grid_dimensions}.\n"
                                      f"Should be 2 for 2D grids.")

        if num_coordinates_found == 0:
            print(f"FOV {fov} not found on the FOV grid")

        elif num_coordinates_found > 1:
            raise IndexError(f"Multiple copies of FOV {fov} found on the FOV grid")

        else:

            # Insert values into the approproate grid
            # ---------------------------------------

            grid_y = grid_coordinates[0, 0]  # y-coord of first coordinate-set
            grid_x = grid_coordinates[0, 1]  # x-coord of first coordinate-set

            for column in columns_to_plot:

                value = results_df.loc[fov, column]

                if value in [np.inf, ]:

                    print(
                        f"Invalid value: {value} found in FOV {fov}. "
                        f"Will not be displayed on grid."
                    )

                else:

                    results_grid_dict[column][grid_y, grid_x] = value

                    if verbose:
                        print(f"FOV {fov} {column}: {value}")

    if plot_grids:
        from utils.gridsQC import plotHeatmaps
        # heatmap plot parameters:
        # ( grid, vmin, vmax, colourmap, size-of-annotations )

        grid_paramsets_dict = {}
        for column in columns_to_plot:
            grid_paramsets_dict[column] = (results_grid_dict[column],
                                           ) + columns_to_plot[column][1:]

        plotHeatmaps(grid_paramsets_dict,
                     iteration=iteration,
                     fig_savepath=fig_savepath)

    if verbose:
        for column in results_grid_dict:
            print(f"{column} grid:\n{results_grid_dict[column]}\n")

    return results_grid_dict


def calcAndPlotCountsFromHdf5(hdf5_filepaths: Union[str,
                                                    List[str],
                                                    Dict[str, str]],
                              count_type: str = "spot_count",
                              properties_column: str = "mean_onbit_intensity",
                              external_properties: Tuple[pd.DataFrame, str] = None,
                              iteration: int = None,
                              summed_df_savepath: str = None,
                              plot_correlation: bool = True,
                              verbose: bool = True,
                              **kwargs,
                              ) -> Tuple[Figure, dict]:
    """
    Plot a correlation and bar plot from a coords hdf5 file

    Parameters
    ----------
    hdf5_filepaths: str, list or dict
        hdf5 coordinates file or
        list of hdf5 coordinates files or
        dictionary of hdf5 coordinates files keyed by FOV
    count_type:
        Type of count to use for correlation plot.
        Is also the name of the column in the counts dataframe
        Can be "spot_count" or "pixel_count"
    properties_column:
        column of counts dataframe to use as property (affects colour of spots)
        should be "mean_onbit_intensity" or "mean_closest_distance"
    external_properties:
        A tuple of:
        (1) dataframe for properties for every gene,
            must have a "gene_names" column
        (2) column-name indicating which column of properties df to use
    iteration:
        analysis iteration (only used to generate plot title)
    **kwargs:
        arguments for plotting e.g.
        - style: seaborn style
    """

    # generate the counts dataframe from the hdf coordinates files
    # ------------------------------------------------------------

    if isinstance(hdf5_filepaths, (list, str)):
        df = countSpotsInH5CoordsFiles(hdf5_filepaths, verbose=verbose)
    elif isinstance(hdf5_filepaths, dict):
        df = countSpotsInH5CoordsFiles(list(hdf5_filepaths.values()),
                                       verbose=verbose)
    else:
        raise TypeError(f"hdf5 file list has type: {type(hdf5_filepaths)}."
                        f"Must be either a dictionary, string or list of strings.")

    # Add properties
    # --------------

    if external_properties is not None:

        # external_properties should be a tuple of datframe, column name
        properties_df, properties_df_column = external_properties

        assert "gene_names" in properties_df.columns, (
            f"Properties dataframe provided "
            f"does not have a 'gene_names' column.\n"
        )

        try:
            properties_df.rename({properties_df_column: "property"},
                                 axis="columns", inplace=True)
        except KeyError:
            print(f"No column: {properties_df_column} in properties dataframe")
            raise

        df = df.merge(properties_df[["gene_names", "property"]],
                      left_on="gene_names", right_on="gene_names")

    else:
        df.rename({properties_column: "property"}, axis="columns", inplace=True)

    # Sort genes from blank and do calculations
    # -----------------------------------------

    corr_count_dict = _calcCorrAndCountFromDF(
        df, "FPKM_data", count_type,
    )

    (sorted_df,
     confidence_dict) = _sortAndCalcConfidence(
        df, col_to_sort=count_type,
    )

    # concatenate correlation and confidence dictionaries
    results_dict = {**corr_count_dict, **confidence_dict}

    # Plot the figure
    # ---------------

    if count_type not in ["spot_count", "pixel_count"]:
        raise ValueError(f"Count type is {count_type}\n"
                         f"Must be 'spot_count' or 'pixel_count'.")

    sorted_df.set_index("gene_names", inplace=True)

    if summed_df_savepath is not None:
        # Save the sorted combined counts dataframe
        iteration_str = ""
        if iteration is not None:
            iteration_str += f"_iter{iteration}"
        summed_df_name = (
            f"allFOVs_counts_df{iteration_str}_{time.strftime('%Y%m%d_%H%M%S')}.tsv"
        )
        sorted_df.to_csv(
            os.path.join(summed_df_savepath, summed_df_name), sep="\t"
        )

    # Plot counts vs FPKM
    # -------------------

    if plot_correlation:

        fig = plotCorrAndCounts(sorted_df,
                                results_dict,
                                iteration=iteration,
                                x_column="FPKM_data",
                                y_column=count_type,
                                **kwargs)

    else:

        fig = None

    return fig, results_dict


#
#                                    Plotting Functions
# -----------------------------------------------------------------------------------------------
#
#


def plotCorrAndCounts(df: pd.DataFrame,
                      results_dict: dict,
                      iteration: Union[int, None] = None,
                      x_column: str = "FPKM_data",
                      y_column: str = "spot_counts",
                      annotate: bool = True,
                      figsize: Tuple[float, float] = (13.5, 9),
                      dpi: int = 400,
                      style: str = "darkgrid",
                      base_markersize: int = 32,
                      fig_savepath: str = None,
                      use_pyplot: bool = True,
                      verbose: bool = True,
                      ) -> plt.Figure:
    """
    Plot: (1) correlation scatterplot with colormap (above)
          (2) count barplot (below)

    Parameters
    ----------
    df: pandas dataframe:
        MUST satisfy the following:
         - SORTED along y_column
         - index as labels (genes)
         - x and y data columns must be present as specified in x_column and y_column
         - "gene_or_blank" column to indicate if gene or blank
         - (optional) "size" column to control spot size
         - (optional) "property" column controls colour of spots
    results_dict: dict
        dictionary of results parameters.
        Should contain:
        correlation, p_value, total_count, percent_above_blank, gene_blank_ratio
    annotate:bool
        whether to annotate the correlation plot with gene labels
    x_column,y_column:
        x and y columns of dataframe to use for correlation plot
        y_column is the column of counts to use for both plots
    style: str
        seaborn style to use
    base_markersize: int
        size for scatterplot markers.
        If no property given, all markers will be this size.
        If not, this will be size of smallest marker
    """
    from matplotlib.cm import ScalarMappable

    def _formatPlotTitlesFromDict(results_dict: dict,
                                  count_type: str,
                                  iteration: Union[int, None],
                                  ) -> Tuple[str, str]:
        """
        use calculated values from a dictionary containing:
        (1) log-correlation (2) p_value (3) total_count
        (4) confidence percentage (5) gene to blank ratio

        to generate plot titles for both the
        (1) scatter plot and (2) bar plot
        """

        iteration_str = ""
        if iteration is not None:
            iteration_str += f" for iteration {iteration}"

        count_type = count_type.replace("_", " ").title()

        scatterplot_title = (f"{count_type} vs FPKM{iteration_str}\n"
                             f"log correlation = {results_dict['correlation']:0.3f}, "
                             f"p = {results_dict['p_value']:0.2e}"
                             f"  |  Total RNA count: {results_dict['total_count']:0.0f}")

        barplot_title = (f"Genes, descending order  |  "
                         f"{results_dict['percent_above_blank']:0.1f} % above blank  |  "
                         f"median-gene to median-blank ratio : "
                         f"{results_dict['gene_blank_ratio']:0.2f}")

        return scatterplot_title, barplot_title

    def _scatterPlot(ax,
                     df: pd.DataFrame,
                     title: str,
                     x_column: str,
                     y_column: str,
                     annotate: bool = True,
                     base_markersize: int = 32,

                     verbose: bool = True,
                     ) -> None:
        """
        plot scatterplot of y_column values vs x_column values on given axes
        """

        df_genes = df[df["gene_or_blank"] == "gene"]
        df_blank = df[df["gene_or_blank"] == "blank"]

        s = base_markersize
        s_blank = base_markersize

        if "mean_pixels" in df.columns:
            s *= df_genes["mean_pixels"].values
            s_blank *= df_blank["mean_pixels"].values
            print(f"s: {s}")

        if "property" in df.columns:
            min_property_value = df["property"].min()
            max_property_value = df["property"].max()
            c = df_genes["property"].values
            c_blank = df_blank["property"].values
            plot_colorbar = True
        else:
            min_property_value = None
            max_property_value = None
            c = sns.xkcd_rgb["cerulean blue"]
            c_blank = sns.xkcd_rgb["pale red"]
            plot_colorbar = False

        # df.plot.scatter(x=x_column, y=y_column,
        #                 s=s, c=c,
        #                 alpha=0.8,
        #                 colormap='viridis',
        #                 ax=ax_scatter,
        #                 title=scatter_title,
        #                 colorbar=colorbar,
        #                 )

        cmap = 'viridis'

        scatter_gene = ax.scatter(x=df_genes[x_column].values,
                                  y=df_genes[y_column].values,
                                  s=s, c=c,
                                  alpha=0.6,
                                  cmap=cmap,
                                  vmin=min_property_value,
                                  vmax=max_property_value,
                                  )

        scatter_blank = ax.scatter(x=df_blank[x_column].values,
                                   y=df_blank[y_column].values,
                                   s=s_blank, c=c_blank,
                                   alpha=0.6,
                                   cmap=cmap,
                                   vmin=min_property_value,
                                   vmax=max_property_value,
                                   edgecolor=sns.xkcd_rgb["pale red"],
                                   )

        if plot_colorbar:
            divider = make_axes_locatable(ax_scatter)
            cbar_ax = divider.append_axes("right", size="3%", pad=0.2)
            mappable = ScalarMappable(cmap=cmap)
            mappable.set_clim(vmin=min_property_value,
                              vmax=max_property_value)
            mappable.set_array([])
            fig.colorbar(mappable, cax=cbar_ax)

        ax_scatter.set_title(title)

        # configure y and x axis
        # ----------------------

        def _findLogLowerBound(array: np.ndarray,
                               ) -> float:
            """
            calculate the lower bound on a log plot for the given array of values
            """
            # remove 0s from array (log of 0 undefined), also reject negative values
            array = array[array > 0.]
            min_val = np.amin(array)

            return 10 ** (np.floor(np.log10(min_val)))

        ax_scatter.set_ylabel("count")

        # find value of linthreshy that is closest to
        # the lowest nonzero counts value
        count_values = df_genes[y_column].values
        linthreshy = _findLogLowerBound(count_values)
        print(f"linthreshy (counts axis) : {linthreshy}\n")

        ax_scatter.set_yscale(
            "symlog", linthreshy=linthreshy, linscaley=0.2,
        )

        ax_scatter.set_xlabel("FPKM value")

        # find value of linthreshx that is closest to
        # the lowest nonzero FPKM value
        fpkm_values = df_genes[x_column].values
        linthreshx = _findLogLowerBound(fpkm_values)
        print(f"linthreshx (FPKM axis): {linthreshx}\n")

        ax_scatter.set_xscale(
            "symlog", linthreshx=linthreshx, linscalex=0.2,
        )

        # set default axes limits
        ax_scatter.set_xlim(-linthreshx / 10, None)
        ax_scatter.set_ylim(-linthreshy / 10, None)

        # Annotate the plot with gene_names (from dataframe index)
        # --------------------------------------------------------

        if annotate:
            for label in df.index:
                x = df.loc[label, x_column]
                y = df.loc[label, y_column]

                if verbose:
                    print(f"{label}: y={y}, x={x}")

                ax_scatter.annotate(label, xy=(x, y),
                                    xytext=(2, 1), textcoords='offset points',
                                    fontname='Arial', fontsize=12,
                                    alpha=0.8, )

    def _barPlot(ax,
                 df: pd.DataFrame,
                 y_column: str,
                 title: str,
                 ) -> None:
        """
        plot a bar plot on given axes
        """
        barplot = sns.barplot(
            x=df.index, y=y_column,
            hue="gene_or_blank",
            data=df, ax=ax,
            dodge=False,
            palette=[sns.xkcd_rgb["cerulean blue"],
                     sns.xkcd_rgb["pale red"], ],
        )
        barplot.set_ylabel('count')
        barplot.set_yscale('log')
        barplot.set_xticklabels(barplot.get_xticklabels(),
                                fontdict={'fontsize': 6, },
                                rotation=85,
                                # ha='right',
                                )
        barplot.xaxis.label.set_visible(False)
        barplot.set_title(title)
        ax_bar.legend().set_visible(False)

    #
    # Main plotting function
    # ----------------------
    #

    (scatterplot_title,
     barplot_title) = _formatPlotTitlesFromDict(results_dict, y_column, iteration)

    sns.set_style(style)

    if use_pyplot:

        fig = plt.figure(figsize=figsize)

    else:

        fig = Figure(figsize=figsize, dpi=dpi)
        canvas = FigCanvas(fig)
        fig.set_canvas(canvas)

    gs = gridspec.GridSpec(4, 1)

    ax_scatter = fig.add_subplot(gs[:-1, :])

    _scatterPlot(
        ax_scatter, df, scatterplot_title,
        x_column, y_column,
        annotate=annotate, base_markersize=base_markersize,
        verbose=verbose,
    )

    ax_bar = fig.add_subplot(gs[-1, :])
    _barPlot(ax_bar, df, y_column, barplot_title)

    fig.subplots_adjust(
        left=0.08, right=0.94,
        bottom=0.08, top=0.94,
        wspace=0.1, hspace=.8,
    )

    # save the figure
    # ---------------
    if fig_savepath is not None:
        savefilename = f"correlation_confidence_iter{iteration}.png"
        fig.savefig(os.path.join(fig_savepath, savefilename), dpi=dpi)
        print(f"   saved correlation plot in <{fig_savepath}>\n")

    return fig


#
#
# -------------------------------------------------------------------------------------------------------
#
#

if __name__ == "__main__":
    #
    # ____ for testing ____
    import tkinter as tk
    from tkinter import filedialog

    #
    # Test plotting function
    # ----------------------
    #

    root = tk.Tk()
    root.withdraw()

    # get the directory where the hdf5 coord files are stored
    data_path = filedialog.askdirectory(
        title="Please select the main data directory")

    # get the filepath for the hdf5 coordinates file
    # hdf5_path = filedialog.askopenfilename(
    #     title="Please select coordinates hdf5 file")

    # get the directory where the hdf5 coord files are stored
    hdf5_path = filedialog.askdirectory(
        title="Please select directory for coordinates hdf5 file")

    # get the filepath for the properties .tsv file
    properties_path = filedialog.askopenfilename(
        title="Please select file containing properties info")
    root.destroy()

    if properties_path:
        properties_df = pd.read_csv(properties_path, sep="\t")
    else:
        properties_df = None
    print("properties dataframe:", properties_df)

    if hdf5_path:

        hdf5_dict = listH5PathsByFov(hdf5_path,
                                     verbose=True)

        results_df = calcAllFovsFromHdf5(hdf5_dict, df_savepath=hdf5_path)

        calcAndPlotCountsFromHdf5(hdf5_dict,
                                  # properties_df=properties_df,
                                  # properties_df_column="mean_brightness",
                                  style="darkgrid",
                                  limits=((None, None), (None, None)),
                                  annotate=True,
                                  summed_df_savepath=hdf5_path,
                                  )

        from fileparsing.filesClasses import getFileParser
        from fileparsing.fovGridFunctions import generateFovGrid

        # Set microscope type and choice of roi
        # -------------------------------------

        # microscope_type = "Triton"
        microscope_type = "Dory"

        params = {}

        if microscope_type == "Dory":

            stage_pixel_matrix = 8 * np.array([[0, -1], [-1, 0]])

            params["reference_bit"] = 1
            params["roi"] = None

        elif microscope_type == "Triton":

            # stage_pixel_matrix = 14.1 * np.array([[1, 0], [0, 1]])
            stage_pixel_matrix = 8 * np.array([[1, 0], [0, 1]])
            params["reference_bit"] = 1
            params["roi"] = 2

        # get the file parser
        # -------------------

        parser = getFileParser(data_path,
                               microscope_type,
                               use_existing_filesdata=True)

        fov_grid, _, _ = generateFovGrid(parser.roiSubsetOfFilesDF(params["roi"]),
                                         stage_pixel_matrix,
                                         fov_subset=list(hdf5_dict.keys()),
                                         plot_grid=True,
                                         # hyb=hyb_for_grid,
                                         # colour_type=colour_for_grid,
                                         )

        generateAndPlotResultsGrid(results_df, fov_grid,
                                   fig_savepath=hdf5_path)

    plt.show()
