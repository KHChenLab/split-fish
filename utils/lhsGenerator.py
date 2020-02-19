"""
Functions to generate latin hyper-something sampling

makes use of pyDOE's lhs function,
which is stored in the "external" folder
so user doesn't have to intall pyDOE

nigel sep 2019

License and readme found in https://github.com/khchenLab/split-fish
"""

import numpy as np

import pprint as pp

from typing import Dict, List, Tuple

from external.doe_lhs import lhs
from scipy.stats import norm


def lhsGen(range_dict: Dict[str, Tuple[float, float]],
           samples: int = 10,
           criterion: str = "corr",
           verbose=True,
           ) -> List[Dict[str, float]]:
    """
    generate a list of parameter values, given ranges for those parameters
    
    Parameters
    ----------
    range_dict: dictionary of 2-tuples
        dictionary of the form:
        {parameter 1 : (lower-bound1 , upper-bound1),
         parameter 2 : (lower-bound2 , upper-bound2),
         ... }
    samples: int
        number of samples to generate
    criterion: str
        sampling criterion passed to pyDOE's lhs function
    
    returns
    -------
    a samples-long list of dictionaries, of the form:
    [{parameter1: value1, parameter2: value2, ... }, ... ]
    """

    def _scale_range(input, min, max):
        input -= np.min(input)
        input /= np.max(input) / (max - min)
        input += min

    # the number of parameters to search over
    num_params = len(range_dict)

    lhsarray = lhs(num_params,
                   samples=samples,
                   criterion=criterion)

    # NOTE: output of lhs function has the form:
    #
    #            parameters ->
    # samples  [ .  .  .  . ,
    #    |       .  .  .  . ,
    #    v       .  .  .  .  ]

    # scale the array
    # ---------------

    for param_num, param in enumerate(range_dict):
        _scale_range((lhsarray[:, param_num]),
                     range_dict[param][0],
                     range_dict[param][1])

    # convert to list of dicts form
    # -----------------------------

    params_list = range_dict.keys()
    values_list = []
    for sample in range(lhsarray.shape[0]):
        values_list.append(
            dict(zip(params_list, lhsarray[sample, :]))
        )

    if verbose:
        print(f"\nLHS samples chosen:\n{'-' * 19}")
        pp.pprint(values_list)

    return values_list


if __name__ == "__main__":
    #
    # define parameter ranges
    # -----------------------

    parameter_ranges = {"cut_threshold": (0.46, 0.5176),
                        "magnitude_threshold": (0.14, 0.6),
                        "low_cut": (200, 600), }

    #
    # run lhs
    # -------

    values = lhsGen(parameter_ranges,
                    samples=12,
                    verbose=True, )
