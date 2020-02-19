"""
Dax writing functionality adapted from
https://github.com/ZhuangLab/storm-analysis/blob/master/storm_analysis/sa_library/datawriter.py

It is under the MIT license:
"Copyright (c) 2018 Zhuang Lab / Babcock Lab
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software."

It has been modified to use as a Python3 context manager
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import math

import tkinter as tk
from tkinter import filedialog


class Writer(object):
    def __init__(self, width=None, height=None, **kwds):
        super(Writer, self).__init__(**kwds)
        self.w = width
        self.h = height

    def frameToU16(self, frame):
        frame = frame.copy()
        frame[(frame < 0)] = 0
        frame[(frame > 65535)] = 65535
        return np.round(frame).astype(np.uint16)


class DaxWriter(Writer):
    """
    from https://github.com/ZhuangLab/storm-analysis/blob/master/storm_analysis/sa_library/datawriter.py

    has been modified to work as a python 3 context manager - nigel
    """

    def __init__(self, name, **kwds):
        super(DaxWriter, self).__init__(**kwds)
        self.name = name
        if len(os.path.dirname(name)) > 0:
            self.root_name = os.path.dirname(name) + "/" + os.path.splitext(os.path.basename(name))[0]
        else:
            self.root_name = os.path.splitext(os.path.basename(name))[0]
        self.fp = open(self.name, "wb")
        self.l = 0

    def __enter__(self):
        return self

    def addFrame(self, frame):
        frame = self.frameToU16(frame)
        if (self.w is None) or (self.h is None):
            [self.h, self.w] = frame.shape
        else:
            assert (self.h == frame.shape[0]), "height of frame incompatible"
            assert (self.w == frame.shape[1]), "width of frame incompatible"
        frame.tofile(self.fp)
        self.l += 1

    def close(self):
        self.fp.close()
        self.w, self.h = int(self.w), int(self.h)
        with open(self.root_name + ".inf", "w") as inf_fp:
            inf_fp.write("binning = 1 x 1\n")
            inf_fp.write("data type = 16 bit integers (binary, little endian)\n")
            inf_fp.write("frame dimensions = " + str(self.h) + " x " + str(self.w) + "\n")
            inf_fp.write("number of frames = " + str(self.l) + "\n")
            inf_fp.write("Lock Target = 0.0\n")
            if True:
                inf_fp.write("x_start = 1\n")
                inf_fp.write("x_end = " + str(self.w) + "\n")
                inf_fp.write("y_start = 1\n")
                inf_fp.write("y_end = " + str(self.h) + "\n")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    # this script
    root = tk.Tk()
    root.withdraw()
    data_path = filedialog.askopenfilename(title="Please select image")
    save_path = filedialog.asksaveasfilename(title="where to save output dax file")
    root.destroy()  # need to do this otherwise will hang when closing matplotlib window

    img = np.load(data_path)
    plt.imshow(img[0, :, :, 0], vmin=np.percentile(img[0, :, :, 0], 40), vmax=np.percentile(img[0, :, :, 0], 99), )
    plt.show()

    print("-" * 50 + "\n Saving dax...\n" + "-" * 50)
    with DaxWriter(save_path + ".dax", width=2048, height=2048) as writer:
        for n in range(img.shape[3]):
            writer.addFrame(img[0, :, :, n])
            print("Max of frame {}: {}".format(n, np.amax(img[0, :, :, n])))
