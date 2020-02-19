"""
Simple script to read any hdf5 file that is one-layer deep.
Will also read attributes if present.

If reading a normalization vector file, will read a second layer of data
if first-layer key is "0"

Nigel - 17 Jul 19

License and readme found in https://github.com/khchenLab/split-fish
"""
import h5py
import numpy as np

import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt

root = tk.Tk()
root.withdraw()
data_path = filedialog.askopenfilename(title="Please select hdf file")
root.destroy()

plot = False

with h5py.File(data_path, 'r') as f:
    # print a list of the keys
    print(f"\nKeys:\n {list(f.keys())}\n"
          f"Number of datasets = {len(list(f.keys()))}")

    # print attributes if present
    if len(f.attrs) > 0:
        print(f"Root attributes:")
        for attr in f.attrs:
            print(f"  -{attr} : {f.attrs[attr]}")

    for key in f:

        print("_" * 20 + f" {key} " +
              "_" * 20 + "\n")

        if key == "0":
            for group in f.get(key):
                print(f[f"{key}/{group}"])
                print(np.array(f[f"{key}/{group}"]))

        else:
            # print attributes if present
            if len(f[key].attrs) > 0:
                print(f"Attributes:")
                for attr in f[key].attrs:
                    print(f"  -{attr} : {f[key].attrs[attr]}")

            # print the data array with properties
            print(f"\nShape = {f[key].shape}\n"
                  f"Type = {f[key].dtype}\n"
                  f"\nArray for {f[key]}:\n\n"
                  f"{np.array(f[key])}\n")

            if plot:
               fig,ax =plt.subplots()
               plt.imshow(f[key])

    plt.show()

