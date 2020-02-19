"""
quick and dirty script to convert legacy .fasta codebook files to .tsv files

you will be asked for the file you want to convert

some fasta parsing code is adapted from: https://www.biostars.org/p/710/

Nigel Dec 2018

License and readme found in https://github.com/khchenLab/split-fish
"""
import os
import csv
import re
from itertools import groupby

import pandas as pd

import tkinter as tk
from tkinter import filedialog

# ask user for file to convert
root = tk.Tk()
root.withdraw()
filepath = filedialog.askopenfilename()
# inhouseseq_filepath = filedialog.askopenfilename(title="choose in house sequencing text file")
root.destroy()  # need to do this otherwise will hang when closing matplotlib window
directory, file = os.path.split(filepath)
filename, fileext = os.path.splitext(file)

# Parameters
# ----------

checkfile = True  # print the data to console to check if everything ok
codebook_filename = "codebook_data"  # if same, follow original codebook name
id_filename = "id_data"  # gene names with corresponding gene or variant ids
overwrite = False

read_fpkm = True
fpkm_filename = "fpkm_data"
overwrite_fpkm = True

# ---------------------------------------------------------------------------------

genename_pattern = re.compile(r"^(\w|-)+(\s|$)", re.IGNORECASE)
liver_pattern = re.compile(r"(?<=Liver)\s?(\d+.\d+|0|\d+)(?=\s|$)")
brain_pattern = re.compile(r"(?<=Brain)\s?(\d+.\d+|0|\d+)(?=\s|$)")
kidney_pattern = re.compile(r"(?<=Kidney)\s?(\d+.\d+|0|\d+)(?=\s|$)")
AML_pattern = re.compile(r"(?<=AML)\s?(\d+.\d+|0|\d+)(?=\s|$)")
ENSMUSG_pattern = re.compile(r"ENSMUSG(\d+.\d+|\d+)(?=\s|$)")

idpattern_list = [ENSMUSG_pattern]
pattern_list = [liver_pattern]

if codebook_filename == "same":
    codebook_filename = filename

# the output file
codebook_file = os.path.join(directory, codebook_filename + ".tsv")
id_file = os.path.join(directory, id_filename + ".tsv")
if read_fpkm:
    fpkm_file = os.path.join(directory, fpkm_filename + ".tsv")

# check for existing file
keep_codebook_file = os.path.isfile(codebook_file) and not overwrite
keep_fpkm_file = os.path.isfile(fpkm_file) and not overwrite_fpkm

if keep_codebook_file and keep_fpkm_file:
    print("already have: {}".format(codebook_file))
    print("already have: {}".format(fpkm_file))
else:
    if overwrite:
        print("overwriting {}".format(codebook_file))
    if overwrite_fpkm:
        print("overwriting {}".format(fpkm_file))

    all_data = []  # list of lists of bits followed by gene name
    fpkm_data = []  # gene name followed by FPKMs from various tissue
    id_data = []  # gene name followed by various IDs

    # parse fasta file
    with open(filepath, "r") as fr:
        fasta_iter = (x[1] for x in groupby(fr, lambda line: line[0] == ">"))
        for codeword in fasta_iter:
            bit_list = codeword.__next__()[1:].strip().split()
            info = "".join(s.strip("\r\n") for s in fasta_iter.__next__())
            print(info)
            genename = re.search(genename_pattern, info).group(0).strip()
            print("gene name:", genename)
            bit_list.append(genename)
            all_data.append(bit_list)

            # _______ deal with gene identifiers _______
            id_list = [genename, ]
            if genename[0:5].lower() == "blank":
                for pattern in idpattern_list:
                    id_list.append(genename)
            else:
                for pattern in idpattern_list:
                    id_list.append(re.search(pattern, info).group(0))
            id_data.append(id_list)

            # _______ read the FPKM values _______
            if read_fpkm:
                fpkm_list = [genename, ]
                if genename[0:5].lower() == "blank":
                    for pattern in pattern_list:
                        fpkm_list.append(0)
                else:
                    for pattern in pattern_list:
                        fpkm_list.append(re.search(pattern, info).group(0).strip())
                fpkm_data.append(fpkm_list)

    # write to id file
    with open(id_file, 'w', newline='\n') as fw:
        tsv_writer = csv.writer(fw, delimiter='\t')
        for line in id_data:
            tsv_writer.writerow(line)

    # write to tsv file
    with open(codebook_file, 'w', newline='\n') as fw:
        tsv_writer = csv.writer(fw, delimiter='\t')
        for line in all_data:
            tsv_writer.writerow(line)

    # write to fpkm tsv file
    if read_fpkm:
        with open(fpkm_file, 'w', newline='\n') as fw:
            tsv_writer = csv.writer(fw, delimiter='\t')
            for line in fpkm_data:
                tsv_writer.writerow(line)

# optional: check if tsv file is ok by opening it with pandas
if checkfile:
    print("opening {}".format(codebook_file))
    df = pd.read_table(codebook_file, header=None)
    print(df)

    print("opening {}".format(fpkm_file))
    df_fpkm = pd.read_table(fpkm_file, header=None, index_col=0)
    print(df_fpkm)

    print("opening {}".format(id_file))
    df_id = pd.read_table(id_file, header=None)
    print(df_id)

    # print("opening {}".format(inhouseseq_filepath))
    # df_ihs = pd.read_table(inhouseseq_filepath)
    # df_ihs['avg'] = df_ihs.filter(like="RMC").mean(axis=1)
    #
    # short_ids = df_ihs["location"].str.split('.', expand=True)
    # df_ihs["short_ids"] = short_ids[0]
    #
    # df_newfpkm = pd.merge(df_id, df_ihs, left_on=1, right_on="short_ids", how="left")
    # df_newfpkm = df_newfpkm[[0, "avg"]]
    # df_newfpkm.fillna(0, inplace=True)
    #
    # # re-write FPKM file
    # df_newfpkm.to_csv(fpkm_file, sep="\t",
    #                   index=False, header=False)
    #
    # print(df_ihs)
    # print(df_newfpkm)
