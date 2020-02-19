"""
Do Mutual Information registration on 2 images
Incomplete

19 Jul 19 - started nigel
"""

import numpy as np

import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt

from scipy.optimize import minimize
import scipy.ndimage.interpolation as interpolation

from sklearn.metrics import mutual_info_score

from utils.readClasses import readDoryImg


def plotOverlay(img1, img2,
                pct=(10, 99),  # percentile of image to scale to
                ):
    """
    simple function to plot overlay of 2 images (1st in red and 2nd in green)
    overlap regions will be yellow

    returns rgb image with 1st channel img1, 2nd channel img2, nothing on 3rd channel
    """
    assert img1.shape == img2.shape, "2 images have different dimensions!"
    assert img1.ndim == 2, "not a 2-dimensional image!"

    rgb_temp = np.zeros((img1.shape[0], img1.shape[1], 3), dtype=np.float64)

    # high_pct = min(np.percentile(img1, pct[1]), np.percentile(img2, pct[1]))
    # low_pct = max(np.percentile(img1, pct[0]), np.percentile(img2, pct[0]))

    low_pct1, high_pct1 = np.percentile(img1, pct[0]), np.percentile(img1, pct[1])
    low_pct2, high_pct2 = np.percentile(img2, pct[0]), np.percentile(img2, pct[1])

    rgb_temp[:, :, 0] = (img1 - low_pct1) / (high_pct1 - low_pct1)
    rgb_temp[:, :, 1] = (img2 - low_pct2) / (high_pct2 - low_pct2)

    fig_overlap, ax_overlap = plt.subplots(figsize=(8, 9))
    ax_overlap.imshow(np.clip(rgb_temp, 0, 1), vmax=10000)


def calcImageMI(x, y, bins):
    """
    adapted from stackexchange
    """
    assert x.shape == y.shape, (f"Dimensions of image 1 {x.shape} "
                                f"do not match image 2 {y.shape}")

    c_xy = np.histogram2d(x.ravel(), y.ravel(), bins)[0]

    return mutual_info_score(None, None, contingency=c_xy)


class EstimateDistortion(object):
    """
    Class to estimate the parameters for an affine transform (with just scaling and translation)
    to correct for chromatic distortion between different colour channels
    """

    def __init__(self,
                 img1, img2,  # the images to estimate
                 bins=100,  # number of bins to use for Mutual Information calculation
                 method="COBYLA",
                 initial_guess=(0, 0),
                 bounds=((-50, 50), (-50, 50)),
                 borders="max",  # or "exact"
                 ):

        assert img1.shape == img2.shape, (f"Dimensions of image 1 {img1.shape} "
                                          f"do not match image 2 {img2.shape}")
        self.ndim = img1.ndim

        if self.ndim == 3:
            self.objective_fn = self._objectiveFn3D
        elif self.ndim == 2:
            self.objective_fn = self._objectiveFn2D
        else:
            raise ValueError(f"image dimensions are {self.ndim}!"
                             f"\nMust be either 2 or 3.")

        self.img1, self.img2 = img1, img2
        self.bins = bins
        self.borders = borders
        self.method = method

        # starting guess (default to 0)
        self.initial_guess = np.array(initial_guess)

        # bounds
        self.bounds = list(bounds)

    def _objectiveFn3D(self,
                       params,  # array of [tx, ty]
                       ):
        return calcImageMI(self.img1,
                           interpolation.shift(self.img2,
                                               (0, params[0], params[1]),
                                               order=3),
                           self.bins,
                           ) * -1

    def _objectiveFn2D(self,
                       params,  # array of [tx, ty]
                       ):
        return calcImageMI(self.img1,
                           interpolation.shift(self.img2,
                                               (params[0], params[1]),
                                               order=3),
                           self.bins,
                           ) * -1

    def optimizeTranslation(self, ):
        """
        optimize
        """
        result = minimize(self.objective_fn,
                          self.initial_guess,
                          method=self.method,
                          bounds=self.bounds,
                          options={"disp": True})
        print(result.message)

        return result.x


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    img1_path = filedialog.askopenfilename(title="Please select first image")
    img2_path = filedialog.askopenfilename(title="Please select second image")
    root.destroy()  # need to do this otherwise will hang when closing matplotlib window

    img1 = readDoryImg(img1_path)
    img2 = readDoryImg(img2_path)
    print(f"image 1 dimensions: {img1.shape}\n"
          f"image 2 dimensions: {img2.shape}")

    # --------------------------------  Test out estimator  -----------------------------------------
    # MI estimator
    for method in ["SLSQP", "Powell", "COBYLA"]:
        try:
            estimator = EstimateDistortion(img1,
                                           img2,
                                           bins=1000,  # number of bins to use for Mutual Information calculation
                                           method=method,
                                           initial_guess=(0, 0),
                                           bounds=((-50, 50), (-50, 50)),
                                           borders="max",  # or "exact"
                                           )
            tx_estimate = estimator.optimizeTranslation()
            print(f"Optimized params using {method}: {tx_estimate}.\n")

            plotOverlay(img1, interpolation.shift(img2, tx_estimate))

        except:
            print(f"method {method} failed")

    # Phase correlation: register_translation
    from skimage.feature import register_translation

    pc_estimate, _, _ = register_translation(img1, img2, upsample_factor=20)
    mi_pc = calcImageMI(img1, interpolation.shift(img2, pc_estimate), bins=300)
    print(f"Phase correlation params : {pc_estimate}.\n"
          f"Has MI of {mi_pc} with these params")

    plt.show()
