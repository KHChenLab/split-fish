"""
GENE DATA object -stores all gene-related and codeword-related info
Note that GeneData has 2 read methods to read codebook fasta and matlab fpkm files

split into a separate file - nigel 18 jul 19

License and readme found in https://github.com/khchenLab/split-fish
"""

import os
import warnings

from typing import Union, Tuple, List, Dict

import numpy as np
import pandas as pd
from itertools import combinations

import scipy.io  # for loading .mat format FPKM files
# (should be removed? Noboby uses .mat FPKM files anymore)
from scipy.spatial import cKDTree


class GeneData(object):
    """
    The class for holding data related to the genes being detected
    this object is created at the beginning of analysis,
    and is shared with all FOVs being processed
    e.g. gene names, id, FPKM etc.

    has methods to read FPKM data from various formats
    and create a kdtree for querying pixel vectors
    """

    def __init__(self,
                 codebook_filepath: str,
                 fpkm_filepath: str = None,
                 num_bits: int = None,
                 dropped_bits: Union[list, np.ndarray] = (),
                 fpkm_structname: str = None,
                 fpkm_column: int = 1,
                 print_dataframe: bool = False,
                 ) -> None:
        """
        Parameters
        ----------
        codebook_filepath: str
            filepath for the .csv file holding codewords for genes
        fpkm_filepath: str
            filepath for the .mat or .csv file holding FPKM values per genes
        num_bits: int
            number of bits in the codeword.
        dropped_bits:
             list of bits to drop from codebook
        fpkm_structname: str
            name of internal variable name of the 'struct'
            in the .mat file to reference FPKM data
            (Note: may not be the same as the name of the .mat file!)
        fpkm_column: int
            starting from 1 (column 0 must be gene names)
            column of the .tsv file with the appropriate fpkm data
        print_dataframe:
            print the pandas dataframe after reading in codebook/fpkm
        """
        self.codebook_filepath = codebook_filepath
        self.dropped_bits = dropped_bits

        self.duplicated_genes_found = False

        self.num_bits = num_bits
        self.bit_list = list(range(num_bits))

        # Read in the CODEBOOK and FPKM, and get number of genes
        # ------------------------------------------------------

        (self.codebook_data,
         self.num_genes) = self._readCodebook(codebook_filepath)

        if print_dataframe:
            self._printTable("after reading codebook")

        # if dropped bits are requested, process dataframe
        # to update bits, genes and number of genes
        if len(dropped_bits) > 0:
            self.dropBits(dropped_bits)

        if fpkm_filepath is not None:
            self.readFpkm(fpkm_filepath,
                          fpkm_structname=fpkm_structname,
                          fpkm_column=fpkm_column,
                          verbose=print_dataframe)

        self.dropDuplicates()

        # save the full codebook/fpkm dataframe as .csv file
        # in same folder as codebook for troubleshooting
        codebook_dir = os.path.dirname(codebook_filepath)
        checkcodebook_filepath = os.path.join(codebook_dir,
                                              "check_fullcodebook.tsv")
        self.codebook_data.to_csv(checkcodebook_filepath, sep="\t")

        # Genereate Tree
        # --------------

        # kDtree of UNIT NORMALIZED codewords, used to search for closest codeword
        self.codebook_tree = self.generateTree(self.codebook_data)

    def _raiseIfInvalidCodebookDf(self) -> None:
        """
        check if codebook dataframe is present and is a pandas dataframe
        """
        if self.codebook_data is None:
            raise ValueError("Codebook dataframe not provided")

        elif not isinstance(self.codebook_data, pd.DataFrame):
            raise TypeError("'Codebook dataframe' is not a pandas Dataframe")

    def _printTable(self,
                    txt: str,
                    dashes: int = 20,
                    head_only: bool = True,
                    ) -> None:
        """
        print out the codebook dataframe
        """
        self._raiseIfInvalidCodebookDf()

        if head_only:
            df = self.codebook_data.head()
        else:
            df = self.codebook_data

        print(f"\nCodebook Dataframe {txt}:"
              f"\n{'-' * (len(txt) + dashes)}\n"
              f"{df}\n"
              f"Summary:\n{self.codebook_data.describe()}\n"
              f"\nNumber of genes: {self.num_genes}\n"
              f"{'-' * 70}")

    def replaceCodebook(self,
                        **kwargs,
                        ) -> None:
        """
        replace the existing codebook with a new one
        """
        if self.codebook_data is not None:
            print("Overwriting current codebook data")

        self.codebook_data = self._readCodebook(**kwargs)

    def _readCodebook(self,
                      filepath: str,
                      header: int = None,
                      sep: str = "\t",
                      name_col: int = -1,
                      ) -> Tuple[pd.DataFrame, int]:
        """
        read a codebook .tsv file into a pandas dataframe
        
        Parameters
        ----------
        filepath: str
            filepath for the codebook .csv file to read
        header, sep:
            keyword arguments for pandas read_csv
        name_col: int
            the index of the column of the dataframe
            indicating the gene names (last column by default)
        """

        try:
            codebook_df = pd.read_csv(filepath,
                                      header=header,
                                      sep=sep)
        except FileNotFoundError:
            print("Could not find codebook file")
            raise
        except Exception:
            print("Could not open codebook file")
            raise

        # Make gene-names column the dataframe index
        # ------------------------------------------

        codebook_df.rename({codebook_df.columns[name_col]: "gene_names"},
                           axis="columns", inplace=True)
        codebook_df.set_index("gene_names", inplace=True)

        num_genes = len(codebook_df.index)

        return codebook_df, num_genes

    def readFpkm(self,
                 filepath: str,
                 overwrite: bool = True,
                 fpkm_structname: str = "FPKMData",
                 fpkm_column: int = 1,
                 verbose: bool = True,
                 ) -> None:
        """
        must be done after reading in the codebook

        read FPKM data from a matlab .mat file or .tsv file

        Parameters
        ----------
        filepath: str
            filepath for the FPKM .mat or .tsv file
        overwrite: bool
            whether to overwrite existing dataframe if already present
        fpkm_structname: str
            name of internal variable name of the 'struct'
            in the .mat file to reference FPKM data
        fpkm_column: int
            the column in the .tsv for the desired set of FPKM data
            when we have multiple sets of FPKM data
            from different organs or cell-lines e.g.
            gene name |  liver_FPKM     |  brain_FPKM  |  A549_FPKM
                            ...               ...            ...
                        col 1 (default)      col 2          col 3
        """
        self._raiseIfInvalidCodebookDf()

        if "FPKM_data" in self.codebook_data.columns:
            if overwrite:
                warnings.warn("Overwriting current FPKM data")
            else:
                warnings.warn("Already have FPKM data. No FPKM data read")
                return

        FPKM_file_extension = os.path.splitext(filepath)[-1]

        # matlab .mat format
        # ------------------

        if FPKM_file_extension == ".mat":
            fpkm_raw = scipy.io.loadmat(filepath)
            fpkm_num_genes = fpkm_raw[fpkm_structname][0][0][0].ravel().shape[0]
            fpkm_df = pd.Series(data=fpkm_raw[fpkm_structname][0][0][0].ravel(),
                                index=[fpkm_raw[fpkm_structname][0][0][1].ravel()[x][0]
                                       for x in
                                       range(fpkm_num_genes)],
                                name="FPKM_data")
            self.codebook_data = pd.concat([self.codebook_data, fpkm_df],
                                           axis=1, join="inner", sort=False)

        # .tsv format
        # -----------
        # in this format the first column must be the gene names

        elif FPKM_file_extension in [".tsv", ".txt", ".csv"]:

            if FPKM_file_extension == ".csv":
                sep = ","
            else:  # if .tsv or .txt assume tab separators
                sep = "\t"

            fpkm_df = pd.read_csv(filepath, header=None, sep=sep,
                                  usecols=[0, fpkm_column], )

            # rename gene names column as "gene_names"
            # and set it as the index of codebook_data dataframe
            genes_col = "gene_names"
            fpkm_df.rename({fpkm_df.columns[0]: genes_col,
                            fpkm_df.columns[1]: "FPKM_data"},
                           axis="columns", inplace=True)

            self.codebook_data = self.codebook_data.merge(fpkm_df,
                                                          how="inner",
                                                          left_index=True,
                                                          right_on=genes_col,
                                                          sort=False)
            self.codebook_data.set_index(genes_col, inplace=True)
            # NOTE: using 'INNER' join method.
            # only genes in both codebook and fpkm files will be retained

        else:
            raise TypeError(f"FPKM file extension is {FPKM_file_extension}.\n"
                            f"Should be .mat or .tsv/.csv/.txt")

        if verbose:
            self._printTable("after reading FPKM")

    def dropBits(self,
                 dropped_bits: Union[list, np.ndarray],  # a list of bit numbers to be dropped
                 ) -> None:
        """
        Trims the codebook dataframe based on a list/array of integers such that:
            - columns in dropped_bits list are trimmed
            - genes with on-bits in the bist_to_drop list are trimmed
        """

        def _printBits(txt: str, dashes: int):
            print(f"{txt} Dropping bits:\n{'-'*(len(txt)+dashes)}\n"
                  f"Bit list: {self.bit_list}\n"
                  f"Number of bits: {self.num_bits}\n")

        self._raiseIfInvalidCodebookDf()
        _printBits("BEFORE", 15)

        # check if all the bits to be dropped are valid dataframe column headers
        # ----------------------------------------------------------------------

        if not all([bit in self.codebook_data.columns for bit in dropped_bits]):
            raise ValueError(f"Bits-to-drop list: {dropped_bits}\nnot a subset of\n"
                             f"Dataframe columns: {self.codebook_data.columns}")

        # drop all rows with at least 1 on-bit in the dropped_bits list
        # -------------------------------------------------------------

        self.codebook_data = self.codebook_data[
            self.codebook_data[dropped_bits].sum(axis=1) == 0
            ]

        # drop all columns in the dropped_bits list
        # -----------------------------------------

        self.codebook_data.drop(dropped_bits, axis=1, inplace=True)

        # update the number of genes, bits and bit-list
        # ---------------------------------------------

        self.num_genes = self.codebook_data.shape[0]
        self.bit_list = [bit for bit in self.bit_list if bit in dropped_bits]
        self.num_bits = len(self.bit_list)
        _printBits("AFTER", 15)

    def dropDuplicates(self,
                       keep_type: str = "first",
                       ) -> None:
        """
        check for duplicate gene entries and remove extra ones

        Parameters
        ----------
        keep_type: str
            which of the duplicated entries to keep.
            keyword passed to pandas dataframe index.duplicated method
            should be 'first' or 'last'
        """
        self._raiseIfInvalidCodebookDf()

        self.check_duplicates = self.codebook_data.index.duplicated(keep=keep_type)

        if True in self.check_duplicates:
            warnings.warn("Duplicates found in gene names!")
            self.duplicated_genes_found = True

        self.codebook_data = self.codebook_data[~self.check_duplicates]

        # update the number of genes
        self.num_genes = self.codebook_data.shape[0]

    def generateTree(self,
                     dataframe: pd.DataFrame,
                     ) -> cKDTree:
        """
        parses the codebook to get a Ncodeword x Nbits array,
        normalizes each codeword,
        then generates a KDTree to use as reference when decoding
        """

        def _printCodebook(array: np.ndarray,
                           txt: str,
                           dashes: int = 16,
                           ) -> None:
            print(f"\n{txt} codebook array:\n"
                  f"{'-'*(len(txt) + dashes)}\n"
                  f"{array}\n( shape = {array.shape} )\n")

        # Get Binary codes from dataframe
        # -------------------------------

        codebook_array = dataframe.loc[:, self.bit_list].values
        _printCodebook(codebook_array, "Original")

        # Normalize codewords to unit vectors
        # -----------------------------------

        codebook_array = codebook_array / np.linalg.norm(
            codebook_array, axis=1, keepdims=True,
        )
        _printCodebook(codebook_array, "Normalized")

        # Generate tree
        # -------------

        return cKDTree(codebook_array)


class CrosstalkGeneData(GeneData):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.codebook_data = self.crosstalkCodebook(self.codebook_data)

    def crosstalkCodebook(self,
                          codebook_df: pd.DataFrame,
                          ) -> pd.DataFrame:
        """
        generate a crosstalk codebook from the original codebook
        """

        crosstalk_df = codebook_df.drop(["FPKM_data"],
                                        axis=1)

        gene_pairs = list(combinations(crosstalk_df.index, 2))
        print(f"list of combinations: {gene_pairs} \n"
              f"with length: {len(gene_pairs)}")

        return pd.DataFrame([crosstalk_df.loc[pair, :].max(axis=0)
                             for pair in gene_pairs],
                            index=gene_pairs)
