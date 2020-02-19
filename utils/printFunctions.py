"""
Miscellaneous functions for printing info onto
the console or writing to text files

typically used for:
 - printing parmaeters
 - recording time taken to run a script or part of code

nigel Aug 2019

License and readme found in https://github.com/khchenLab/split-fish
"""

import os
import json

import numpy as np
import pandas as pd

from typing import List, Dict, Union, Any

import timeit


def formatParamSetName(params_dict: Dict[str, Any],
                       main_name: str,
                       ) -> str:
    """
    format a name for a parameter set
    incorporating some of the parameter values,
    such as magnitude threshold, small spot threshold and low cut of filter
    """

    def floatToStr(value: float, ) -> str:
        return f"{value:.2f}".replace(".", "_")

    num_fov_str = f"_{len(params_dict['fovs_to_process']):d}FOVs"

    magnitude_str = "_mag" + floatToStr(params_dict["magnitude_threshold"])

    if params_dict["small_spot_threshold"] is not None:
        small_spot_threshold_str = "_ssth" + floatToStr(params_dict["small_spot_threshold"])
    else:
        small_spot_threshold_str = ""

    if params_dict["large_spot_threshold"] is not None:
        large_spot_threshold_str = "_lsth" + floatToStr(params_dict["large_spot_threshold"])
    else:
        large_spot_threshold_str = ""

    if params_dict['low_cut'] is not None:
        low_cut_str = f"_lc{params_dict['low_cut']:d}"
    else:
        low_cut_str = f"_nolc"

    if params_dict["subtract_background"]:
        background_str = "_bgrm"
    else:
        background_str = ""

    return (main_name
            + num_fov_str
            + magnitude_str
            + small_spot_threshold_str
            + large_spot_threshold_str
            + low_cut_str
            + background_str)


def dictOfDictToDf(dict_of_dict: Dict[str, Dict[str, Any]],
                   save_path: str,
                   filename: str,
                   sort_by_column: Union[str, List[str], None] = "correlation",
                   datatype: np.dtype = None,
                   sep: str = "\t",
                   ) -> pd.DataFrame:
    """
    Convert a dictionary of dictionaries to a dataframe and save it.
    The keys of the main dictionary become the columns of the dataframe,
    while the keys of the nested dictionaries (which should all be the same)
    become the columns of the dataframe
    """

    main_keys = list(dict_of_dict.keys())
    nested_keys = list(dict_of_dict[main_keys[0]].keys())

    df = pd.DataFrame(
        columns=nested_keys, index=main_keys, dtype=datatype,
    )

    for main_key in dict_of_dict:
        for nested_key in dict_of_dict[main_key]:
            if nested_key in dict_of_dict[main_key]:
                df.loc[main_key, nested_key] = dict_of_dict[main_key][nested_key]
            else:
                print(f"Warning: could not find {main_key} : {nested_key} entry!")
                df.loc[main_key, nested_key] = np.nan

    if sort_by_column is not None:
        df.sort_values(by=sort_by_column, inplace=True)

    if sep == "\t":
        file_extension = ".tsv"
    elif sep == ",":
        file_extension = ".csv"
    else:  # default to saving as csv
        file_extension = ".csv"

    df.to_csv(
        os.path.join(save_path, filename + file_extension), sep=sep,
    )

    return df


def printParams(params_dict: dict,
                title_string: str,
                ) -> None:
    """
    print out parameters nicely from a parameters dictionary

    Parameters
    ----------
    params_dict: dict
        a dictionary of parameters
    title_string: str
        a description of the type of parameters being printed
    """
    print("\n" + "-" * 25 +
          f" {title_string} Parameters: "
          + "-" * 25)

    for key in params_dict:
        if (isinstance(params_dict[key], list)
                and len(params_dict[key]) > 20):
            print(f"{key}\t:\n{json.dumps(params_dict[key], indent=2)}")
        else:
            print(f"{key}\t:\t{params_dict[key]}")

    print("-" * 70 + "\n")


def printParamsToFile(params_dict_list: List[dict],
                      savepath: str,
                      timestr: str = "",
                      ) -> None:
    """
    Print parameters to a text file in savepath

    Parameters
    ----------
    params_dict_list: list of dictionaries
        a list of parameter dictinoaries
        (e.g. display parameters, user paramters etc.)
    savepath: str
        full path to folder where you want save the text file
    timestr: str
        (optional) string indicating the time the script was run
    """
    fullpath = os.path.join(savepath,
                            "parameters_" + timestr + ".txt")

    with open(fullpath, "w") as params_file:
        for param_type in params_dict_list:
            for key in param_type:
                params_file.write(key + "\t:\t"
                                  + str(param_type[key])
                                  + "\n")


def printTime(elapsed_time: float,
              text: str,
              ) -> None:
    """
    print to console the time-interval in hrs/min/seconds
    given a floating-point time interval in seconds
    (this is usually the output of timeit.default_timer)

    Parameters
    ----------
    elapsed_time: float
        time elapsed for the task (in seconds)
    text: str
        a description of what you were doing during that time
    """
    if elapsed_time > 3600:
        hours, elapsed_time = divmod(elapsed_time, 3600)
        hours_str = f"{hours:0.0f} hrs "
    else:
        hours_str = ""

    if elapsed_time > 60:
        minutes, elapsed_time = divmod(elapsed_time, 60)
        minutes_str = f"{minutes:0.0f} min "
    else:
        minutes_str = ""

    seconds_str = f"{elapsed_time:0.1f} s"

    print("\n" + "-" * 50 +
          f"\nTime taken to {text}: "
          f"{hours_str}{minutes_str}{seconds_str}\n"
          + "-" * 50)


class ReportTime(object):
    """
    report (to console) the time taken for a block of code to execute
    to be used like a context manager
    """

    def __init__(self,
                 text: str = "do something",
                 ):
        self.text = text

    def __enter__(self):
        self.start_time = timeit.default_timer()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        when exiting, print the total time taken to
        run the block of code
        """
        printTime(timeit.default_timer() - self.start_time,
                  self.text)
