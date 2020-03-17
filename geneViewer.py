"""
Gene Viewer
-----------

A simple GUI to display and explore the gene coordinates
(single FOV or merged) produced by the processing pipeline

nigel jan 2019

added density maps, greatly increased speed by
using matplotlib plot instead of patches
 - nigel sep 2019

License and readme found in https://github.com/khchenLab/split-fish
"""

import os
import sys
import re

import h5py

import numpy as np
import pandas as pd

from PyQt5.QtWidgets import QApplication, QPushButton, QFileDialog, QComboBox
from PyQt5.QtWidgets import QDialog, QMessageBox, QSplashScreen
from PyQt5.QtGui import QPixmap, QIcon
import PyQt5.QtCore as QtCore
from PyQt5.QtCore import pyqtSignal, Qt, QSize
import PyQt5.QtWidgets as QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.cm as cm

import seaborn as sns

from skimage.measure import label, regionprops

from utils.readClasses import DaxRead


# from spatialComparisonClasses import AnalyzeSpots


def trap_exc_during_debug(*args):
    """
    when app raises uncaught exception, print info on console
    """
    print(args)


# install exception hook: without this, uncaught exception would cause application to exit
sys.excepthook = trap_exc_during_debug


class ReadNpThread(QtCore.QObject):
    """
    tried to create a thread to read .npy file. couldn't get it to work...
    """
    finished = pyqtSignal()
    fileread = pyqtSignal(object)

    def __init__(self, file_name: str):
        super().__init__()
        self.file_name = file_name

    @QtCore.pyqtSlot()
    def readFile(self):
        nparray = np.load(self.file_name)
        print("shape of input array:", self.nparray.shape)
        self.fileread.emit(nparray)
        self.finished.emit()


class LoadMsg(QMessageBox):
    """
    loading screen
    prints the path of the file being loaded
    also shows an image while waiting for file to load
    """

    def __init__(self):
        super().__init__()

    @QtCore.pyqtSlot()
    def closeLoadingWindow(self):
        self.close()

    def displayFile(self, filepath):
        self.setText(f"Loading:\n{filepath}...")

        # class GeneViewerMainWindow(QMainWindow):
        #     """
        #     the Main Window
        #     """
        #
        #     def __init__(self, parent=None):
        #         super(GeneViewerMainWindow, self).__init__(parent)
        #         self.main_widget = GeneViewerMainWidget(self)
        #         self.setCentralWidget(self.main_widget)


class GeneViewerMainWidget(QDialog):
    """
    the Main GUI widget
    """
    finishedLoading = pyqtSignal()

    def __init__(self, parent=None):
        super(QDialog, self).__init__(parent)

        # scale the size of the window in case
        # user's screen is really big or really small
        # must be set in script. cannot be re-set from GUI
        self.windowscale = .95
        self.save_dpi = 900

        #
        # Window Properties
        # -----------------
        #

        self.setMinimumSize(self.scale(980),
                            self.scale(1020))

        # set window background color:
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), Qt.white)
        self.setPalette(p)

        # set general stylesheet settings
        self.setStyleSheet("QLabel {font-size: 12pt; "
                           "font-family: Arial }")

        self.setWindowTitle('GeneViewer')

        #
        # variables
        # ---------
        #

        self.temp_image_ref = None
        self.gene_data_type = None  # either ".npy" or ".hdf5"
        self.gene_filepath = None
        self.gene_df = None

        # dimensions of the raw image
        self.image_dims = None

        self.markersize = 6  # default value

        # AnalyseSpots object from spatialComparisonClasses
        # to be used for density plotting
        self.analysespots = None

        #
        # log row
        # -------
        # physically at the bottom of the window
        #

        log_row = QtWidgets.QHBoxLayout()

        # log text field
        self.log = QtWidgets.QTextEdit()
        self.log.setMaximumHeight(self.scale(50))
        log_row.addWidget(self.log)

        # save button
        self.saveButton = QPushButton('Save\nHi-res')
        self.saveButton.clicked.connect(self.handleSave)
        self.saveButton.setStyleSheet(
            "font-size: 12pt; "
            "font-weight: bold; "
            "font-family: Arial"
        )
        self.saveButton.setMinimumHeight(self.scale(50))
        self.saveButton.setFixedWidth(self.scale(80))
        log_row.addWidget(self.saveButton)

        #
        # File entry rows
        # ---------------
        #

        self.input_filetypes = [
            "Raw images file",
            "Genes file"
        ]
        self.input_titles = [
            "please select Raw images file",
            "please select Gene mask file"
        ]

        # Initialize dictionaries, keyed by input_filetypes.
        self.data = {}
        self.image_file_ext = {}

        # Set all entries to None
        for filetype in self.input_filetypes:
            self.data[filetype] = None
            self.image_file_ext[filetype] = None

        # Set up the labels and buttons for file loading
        self.fileLabels = []
        self.fileButtons = []
        file_rows = []
        for n, filetext in enumerate(self.input_filetypes):
            self.fileLabels.append(QtWidgets.QLabel(""))
            self.fileLabels[n].setStyleSheet("font-size: 9pt; "
                                             "font-family: Arial; "
                                             "background-color: rgb(200,200,200)")
            self.fileLabels[n].setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            self.fileButtons.append(QtWidgets.QPushButton())
            self.fileButtons[n].setText(filetext)
            self.fileButtons[n].setStyleSheet("font-size: 11pt; "
                                              "font-family: Arial; "
                                              "font-weight: 450")
            self.fileButtons[n].setMinimumHeight(self.scale(26))
            self.fileButtons[n].setFixedWidth(self.scale(160))
            self.fileButtons[n].clicked.connect(self.handleFileButtons)
            file_rows.append(QtWidgets.QHBoxLayout())
            file_rows[n].addWidget(self.fileLabels[n])
            file_rows[n].addWidget(self.fileButtons[n])

        #
        # Gene Lists Comboboxes
        # ---------------------
        #

        genelist_row = QtWidgets.QHBoxLayout()

        self.comboboxColors = [
            sns.xkcd_rgb["pale red"],
            sns.xkcd_rgb["medium green"],
            sns.xkcd_rgb["cerulean blue"],
            sns.xkcd_rgb["amber"],
        ]
        # the total number of gene comboboxes
        self.num_combobox = len(self.comboboxColors)

        # surface colormaps (for density plot)
        # that match each combobox
        self.surfCmap = [
            "Reds",
            "Greens",
            "PuBu",
            "YlOrBr",
        ]

        assert self.num_combobox == len(self.surfCmap), (
            f"number of colormaps ({self.num_combobox}) "
            f"does not match "
            f"number of comoboxes ({len(self.surfCmap)})"
        )

        # tuples of gene name and index
        # in the gene mask/dataframe
        # (index not relevant for .hdf5 gene coord format)
        self.current_genes = []
        for _ in range(len(self.comboboxColors)):
            self.current_genes.append((None, None))

        # list of references to gene comboboxes
        self.comboboxList = []

        for box_num in range(self.num_combobox):
            gene_box = QComboBox()
            gene_box.addItem("no gene selected")

            stylesheet = (
                f"font-size: 12pt; "
                f"font-family: Arial; "
                f"color: {self.comboboxColors[box_num]}; "
                f"border: 2px outset rgb(100,100,100); "
                f"border-radius: 2px; "
                f"box-shadow:0 0 10px;"
                f"background-color: rgb(35,35,35)")
            print(f"stylesheet for combobox "
                  f"{box_num}:\n{stylesheet}")
            gene_box.setStyleSheet(stylesheet)

            # gene_box.setInsertPolicy(QComboBox.InsertAlphabetically)
            gene_box.setAutoFillBackground(True)
            pal = gene_box.palette()
            pal.setColor(gene_box.backgroundRole(),
                         Qt.blue)
            gene_box.setPalette(pal)

            self.comboboxList.append(gene_box)

            genelist_row.addWidget(gene_box)

        #
        # "Show Genes" and "Show ALL Genes" button row
        # --------------------------------------------
        #

        show_row = QtWidgets.QHBoxLayout()

        # a set of tuples of (button text, method to connect to)
        stylesheet_plain = ("font-size: 12pt; "
                            "font-weight: bold; "
                            "font-family: Arial;")

        # include style with xkcd dark coral colour
        stylesheet_red = stylesheet_plain + "background-color: #cf524e;"

        show_button_list = [
            ("Show Genes", self.showGenes, stylesheet_plain),
            # ("Show Density", self.showDensity, stylesheet_plain),
            ("Show ALL Genes", self.showAllGenes, stylesheet_plain),
            ("geneviewer_images/leftarrow.png", self.handlePrevious, stylesheet_plain),
            ("geneviewer_images/rightarrow.png", self.handleNext, stylesheet_plain),
            ("geneviewer_images/eraser.png", self.erase, stylesheet_red)
        ]
        # Note: disabled show density as it requires spatialComparisonClasses
        # which we are not releasing at this time.

        self.showRowButtons = {}

        for button_num, (buttontext,
                         method,
                         stylesheet) in enumerate(show_button_list):
            showbutton = QtWidgets.QPushButton()

            showbutton.setStyleSheet(stylesheet)
            if buttontext.endswith((".jpg", ".png", ".tiff")):
                icon = QIcon(buttontext)
                showbutton.setIcon(icon)
                showbutton.setIconSize(QSize(75, 25))
            else:
                showbutton.setText(buttontext)
            showbutton.setMinimumHeight(self.scale(35))
            show_row.addWidget(showbutton)
            showbutton.clicked.connect(method)
            self.showRowButtons[button_num] = showbutton

        #
        # Canvas Row
        # ----------
        # contains the matplotlib figure
        #

        canvas_row = QtWidgets.QHBoxLayout()

        self.figure = Figure(
            figsize=(7 * self.windowscale, 7 * self.windowscale)
        )

        # create a single set of axes in the figure
        # that extends to the figure borders
        self.image_ax = self.figure.add_axes([0, 0, 1, 1])
        self.image_ax.axis('off')

        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(self.scale(760))

        canvas_row.addWidget(self.canvas)
        # canvas_row.addStretch(1)

        #
        # Toolbar Row
        # -----------
        # this is physically above the canvas row in the window,
        # but we define it after since toolbar requires
        # reference to the Figure canvas
        #

        toolbar_row = QtWidgets.QHBoxLayout()

        # Navigation widget (from matplotlib)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setMaximumHeight(self.scale(30))

        # self.toolbar._actions["save_figure"].clicked.connect(self.handleSave)

        # spinbox for setting spot/marker size
        self.spotsize_spinbox = QtWidgets.QSpinBox()
        self.spotsize_spinbox.setRange(1, 15)
        self.spotsize_spinbox.setMaximumWidth(self.scale(80))
        self.spotsize_spinbox.setMaximumHeight(self.scale(30))
        self.spotsize_label = QtWidgets.QLabel("SPOT SIZE")
        self.spotsize_label.setStyleSheet("font-size: 11pt; "
                                          "font-family: Arial")
        self.spotsize_spinbox.setStyleSheet("font-size: 14pt; "
                                            "font-weight: bold; "
                                            "font-family: Arial")
        self.spotsize_spinbox.setValue(self.markersize)
        self.spotsize_spinbox.valueChanged.connect(self.setMarkerSize)

        # combine widgets horizontally
        for widget in [self.toolbar,
                       self.spotsize_label,
                       self.spotsize_spinbox]:
            toolbar_row.addWidget(widget)

        #
        # Set overall Window Layout
        # -------------------------
        #

        layout = QtWidgets.QVBoxLayout()

        main_rows = [
            show_row,
            genelist_row,
            toolbar_row,
            canvas_row,
            log_row
        ]

        for row in file_rows + main_rows:
            layout.addLayout(row)

        self.setLayout(layout)

    def scale(self, winsize: int):
        """
        scale window sizes
        """
        return int(winsize * self.windowscale)

    def setMarkerSize(self, new_size):
        """
        update the gene spot marker size
        """
        self.log.append(f"Marker size set to {new_size}")
        self.markersize = new_size
        # self.spotsize_label.setText(str(new_size))

    def showImage(self):
        """
        display the image
        """

        if self._checkLoaded("Raw images file"):

            self.image_ax.clear()
            self.image_ax.axis("off")

            # check image filetype
            # --------------------
            if self.image_file_ext["Raw images file"] == ".npy":
                # just use the first frame
                image = self.data["Raw images file"][0, :, :, 0]
            elif self.image_file_ext["Raw images file"] == ".dax":
                print(self.data["Raw images file"].shape)
                image = self.data["Raw images file"][0, :, :]
            elif self.image_file_ext["Raw images file"] == ".hdf5":
                image = self.data["Raw images file"]
            else:
                raise ValueError("Image file type not recognised")

            # plot image
            # ----------
            self.image_ax.imshow(image, cmap="gray",
                                 vmin=max(np.percentile(image, 10), 0),
                                 vmax=3 * np.percentile(image, 99), )
            self.figure.set()
            self.canvas.draw()

            # record image dimensions
            # ----------------------
            self.image_dims = image.shape

    def erase(self):
        """
        erase any spots or contour plots,
        leaving the raw image
        """
        self.image_ax.lines = []
        self.image_ax.collections = []
        self.canvas.draw()

    def showGenes(self):
        """
        plot the selected genes (as coloured dots) on top of the image
        """
        # replace axes list of line2D objects with empty list
        self.image_ax.lines = []
        self._showGenesOrDensity("spots")
        self.canvas.draw()

    def showDensity(self, ):
        """
        plot a density / KDE plot on top of the image
        """
        # replace axes list of collections objects with empty list
        self.image_ax.collections = []
        self._showGenesOrDensity("density")
        self.canvas.draw()

    def _showGenesOrDensity(self, plot_type: str):
        """
        plot the selected genes as
         (1) spots (plot_type = "spots")
         (2) density plot (plot_type = "density")
        on top of the image
        """
        if not self._checkImgAndSpotsLoaded():
            return False

        if plot_type not in ("spots", "density"):
            raise ValueError(f"Plot type given was {plot_type}.\n"
                             f"Must be 'spots' or 'density'.")

        #
        # a list of all the collections objects to plot
        # ---------------------------------------------
        # only used for density plot
        #

        all_quadcollections = []

        #
        # create plots from chosen genes in each combobox
        # -----------------------------------------------
        #

        for box_num in range(self.num_combobox):

            current_gene = self.comboboxList[box_num].currentText().split("(")[0].strip()

            if current_gene != "no gene selected":

                #
                # extract y and x coords given datatype
                # -------------------------------------
                #

                if self.gene_data_type == ".hdf5":
                    if self.gene_filepath is None:
                        self.log.append("Gene file not loaded!!!")
                    else:
                        with h5py.File(self.gene_filepath, 'r') as f:

                            self.log.append(f"Trying to load gene {current_gene}. "
                                            f"Data has shape {f[current_gene].shape}")

                            if f[current_gene].shape[0] == 0:
                                continue

                            y_coord = f[current_gene][:, 2]
                            x_coord = f[current_gene][:, 1]

                elif self.gene_data_type == ".npy":

                    gene_mask_index = self.current_genes[box_num][1]

                    spot_coords = []
                    for region in regionprops(
                            label(self.data["Genes file"][gene_mask_index, 0, :, :])
                    ):
                        y, x = region.centroid
                        r_spot = region.equivalent_diameter
                        spot_coords.append([y, x])
                    spot_coords = np.array(spot_coords)

                    y_coord = spot_coords[:, 1]
                    x_coord = spot_coords[:, 0]

                else:
                    raise ValueError(f"gene data type {self.gene_data_type}"
                                     f"not recognised")

                #
                # plot spots or density
                # ---------------------
                #

                if plot_type == "spots":
                    current_plot = self.image_ax.plot(
                        y_coord, x_coord, ".",
                        markersize=self.markersize,
                        color=self.comboboxColors[box_num],
                        alpha=0.5,
                    )

                    self.log.append(f"Plotted {current_gene} "
                                    f"with color: "
                                    f"{self.comboboxColors[box_num]}")

                if plot_type == "density":
                    # # clip parts outside plot
                    # clip_area = ((0, self.image_dims[0]),
                    #              (0, self.image_dims[1]))
                    # maxdim = max(self.image_dims)
                    # grid_interval = 40
                    # gridsize = maxdim // grid_interval
                    #
                    # self.log.append(f"Set gridsize to {gridsize}\n"
                    #                 f"based on max dimension of {maxdim} "
                    #                 f"and grid interval of {grid_interval}.\n")
                    #
                    # sns.kdeplot(
                    #     y_coord, x_coord,
                    #     cmap=self.surfCmap[box_num],
                    #     shade=True,
                    #     gridsize=gridsize,
                    #     clip=clip_area,
                    #     alpha=0.5,
                    #     n_levels=10,
                    #     shade_lowest=False,
                    #     ax=self.image_ax,
                    # )

                    density_img = self.analysespots._readH5toImgSingle(0, current_gene)

                    y = np.arange(density_img.shape[0]) * self.analysespots.bin_size[0]
                    x = np.arange(density_img.shape[1]) * self.analysespots.bin_size[1]
                    # print(f"x: {x}\ny:{y}\n")

                    self.log.append(f"size of density img: {density_img.shape}\n"
                                    f"size of x: {x.shape}, y: {y.shape}")

                    num_levels = 30

                    # colours = sns.color_palette(self.surfCmap[box_num],
                    # num_levels)

                    c = self.image_ax.contourf(x, y, density_img,
                                               num_levels,
                                               # colors=colours,
                                               cmap=self.surfCmap[box_num],
                                               alpha=0.3,
                                               )
                    all_quadcollections.append(c)

                    self.log.append(f"Plotted {current_gene} "
                                    f"density plot with "
                                    f"colormap: {self.surfCmap[box_num]}\n"
                                    f"final collections:\n{all_quadcollections}\n"
                                    )

        # if plot_type == "density":

        # remove lowest contour(s) from density plot
        # ------------------------------------------
        ax_collections = []  # final list of collections objects to plot
        for c in all_quadcollections:
            ax_collections += c.collections[2:]
        self.image_ax.collections = ax_collections

    def showAllGenes(self, ):
        """
        show all genes in the gene list,
        cycling through a set of colours
        (currently derived from rainbow)
        """
        if not self._checkImgAndSpotsLoaded():
            return False

        #
        # Plot points from all genes
        # --------------------------
        #

        # replace axes list of line2D objects with empty list
        self.image_ax.lines = []

        with h5py.File(self.gene_filepath, 'r') as f:

            gene_list = f.keys()
            num_genes = len(gene_list)

            # colors = sns.color_palette("bright", num_genes)
            colors = cm.rainbow(np.linspace(0, 1, num_genes))

            for gene, color in zip(gene_list, colors):
                if not (gene.lower().startswith("blank")
                        or f[gene].shape[0] == 0):
                    current_plot = self.image_ax.plot(f[gene][:, 2],
                                                      f[gene][:, 1],
                                                      ".",
                                                      markersize=self.markersize,
                                                      color=color,
                                                      alpha=0.8,
                                                      )
                    self.log.append(f"plotted {gene} with color: {color}")

        self.canvas.draw()
        self.log.append(f"plotted all genes ({num_genes} in total)")

    def _changeIndexAndPlot(self, increment: int):

        last_combobox = self.comboboxList[-1]

        current_gene_index = last_combobox.currentIndex()
        last_combobox.setCurrentIndex(current_gene_index + increment)

        self.showGenes()

    def handleNext(self):

        self._changeIndexAndPlot(1)

    def handlePrevious(self):

        self._changeIndexAndPlot(-1)

    def _checkImgAndSpotsLoaded(self):
        """
        Check if images AND spots are loaded
        typically used before plotting
        """

        # check for raw image file
        if not self._checkLoaded(self.input_filetypes[0]):
            self.log.append(f"{self.input_filetypes[0]} not loaded. "
                            f"Cannot plot genes.")
            return False

        # check for gene spots file
        elif not self._checkLoaded(self.input_filetypes[1]):
            self.log.append(f"{self.input_filetypes[1]} not loaded. "
                            f"Cannot plot genes.")
            return False

        else:
            return True

    def _checkLoaded(self, imagetype):
        """
        check if we have an image loaded
        typically used before plotting the image
        """
        if self.image_file_ext[imagetype] == ".hdf5":
            # img array from hdf5 is not loaded to memory
            # so self.data[...] value will be None
            return True

        elif self.data[imagetype] is None:
            msg = QMessageBox()
            msg.setText(f"No {imagetype} loaded!")
            msg.exec_()
            return False

        else:
            return True

    def handleSave(self):
        """
        save figure
        """
        options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Choose folder to save file", "",
            "Text Files (*.png);;Text Files (*.jpg);;All File types... (*)",
            options=options,
        )

        if filename:
            self.figure.savefig(filename, dpi=self.save_dpi)

    def handleFileButtons(self):

        sender = self.sender()

        for n, filetype in enumerate(self.input_filetypes):

            if sender is self.fileButtons[n]:

                print("handleFileButtons recieved signal from", self.input_filetypes[n])
                image_filepath = QFileDialog.getOpenFileName(self, self.input_titles[n])[0]
                print("Image File:", image_filepath)
                self.fileLabels[n].setText(image_filepath)

                # split filepath into filename and extension
                image_file_name, file_ext = os.path.splitext(image_filepath)
                self.image_file_ext[filetype] = file_ext

                # loading screen
                # --------------

                pixmap = QPixmap(
                    "geneviewer_images/Fishes_loading_screen.png",
                )
                splash = QSplashScreen(pixmap, Qt.WindowStaysOnTopHint)
                splash.show()
                splash.showMessage(
                    f"loading\n{image_filepath}\n...\n",
                    Qt.AlignBottom | Qt.AlignCenter,
                    Qt.red
                )
                app.processEvents()

                # loadbox = QMessageBox()
                # # self.finishedLoading.connect(loadbox.closeLoadingWindow)
                # loadbox.setWindowTitle("loading {}...".format(image_filepath))
                # loadbox.setText(image_filepath)
                # loadbox.setMinimumWidth(600)
                # loadbox.setMaximumHeight(20)
                # loadbox.open()
                #
                # self.data[filetype] = np.load(image_filepath)
                # loadbox.close()
                # self.finishedLoading.emit()

                #
                # Directly load image (no threads)
                # --------------------------------
                #

                if file_ext == ".npy":

                    self.data[filetype] = np.load(image_filepath)
                    print(f"shape of loaded numpy array: {self.data[filetype].shape}")
                    self.showImage()

                elif file_ext == ".dax":

                    self.log.append(f"opening dax file: {image_filepath}")
                    with open(image_file_name + ".inf", "r") as inf_file:
                        inf_text = inf_file.read()
                        dims = re.search(r"frame\s+dimensions\s+=\s+(\d+)\s+x\s+(\d+)",
                                         inf_text, re.IGNORECASE).group(1, 2)
                    daxreader = DaxRead(filename=image_filepath, x_pix=int(dims[1]), y_pix=int(dims[0]))
                    self.data[filetype] = daxreader.loadAllFrames()
                    self.log.append(f"{self.image_file_ext[filetype]} image "
                                    f"loaed with dimensions "
                                    f"{daxreader.y_pix} x {daxreader.x_pix} "
                                    f"x {daxreader.frames}\n"
                                    f"image has shape: {self.data[filetype].shape}")
                    self.showImage()

                elif file_ext == ".hdf5":

                    #
                    # if it's a COORDINATES .hdf5 file
                    # --------------------------------
                    #

                    self.log.append(f"opening hdf5 file: {image_filepath}")
                    # spots will be parsed in next section

                    #
                    # if it's a .hdf5 file containing a stitched IMAGE
                    # ------------------------------------------------
                    #

                    if image_file_name.lower().endswith("stitched"):
                        print("hdf5 image file found")
                        with h5py.File(image_filepath, 'r') as f:
                            self.data[filetype] = np.array(f["stitched"])
                            # self.data[filetype] = f["stitched"]
                            # print(f"Raw images file:{self.data['Raw images file']}")
                            self.log.append(f"hdf5 image has shape: "
                                            f"{self.data[filetype].shape}")
                        self.showImage()

                else:
                    print("Could not read image file")

                splash.finish(self)

                # _________ Thread method (does not work!)__________

                # readObj = ReadNpThread(np_file[0])
                # print("Read object initialized")
                #
                # thread = QThread()
                # print("thread initialized")
                # # thread.setObjectName("read_thread")
                # # print("thread name:", thread_name)
                #
                # readObj.moveToThread(thread)
                # print("object moved")
                #
                # readObj.finished.connect(self.handleThreadFinish)
                # readObj.fileread.connect(self.handleReadFinish)
                #
                # thread.started.connect(readObj.readFile)
                # # thread.finished.connect(self.handleThreadFinished)
                # print("signals connected")
                #
                #
                # thread.start()

                #
                # Deal with files containing spot coordinates or info
                # ---------------------------------------------------
                #

                if filetype == "Genes file":
                    print("gene data file extension:", file_ext)
                    self.gene_filepath = image_filepath

                    if file_ext == ".hdf5":
                        self.gene_data_type = file_ext
                        try:
                            fov_num = re.search(r"FOV_(\d+)", image_filepath, re.IGNORECASE).group(1)
                            self.log.append(f"loaded gene coordinates for FOV {fov_num}")
                        except:
                            self.log.append("loaded combinded gene coordinates")

                        # if hdf file loaded, populate the comoboxes
                        with h5py.File(image_filepath, 'r') as f:
                            for combobox in self.comboboxList:
                                combobox.clear()
                                combobox.addItem("no gene selected")
                                for gene_name in f.keys():
                                    gene_fpkm = f[gene_name].attrs["FPKM_data"]
                                    combobox.addItem(f"{gene_name} ({gene_fpkm:0.2f})")
                                # combobox.model().sort(0)
                                combobox.update()
                                combobox.currentIndexChanged[str].connect(self.handleGeneChoice)

                        self.analysespots = AnalyzeSpots(
                            image_filepath,
                            bin_size=(25, 25),
                            smooth=True,
                            sigma=(250, 250),
                        )

                    elif file_ext == ".npy":
                        self.gene_data_type = file_ext
                        fov_num = re.search(r"FOV_(\d+)", image_filepath, re.IGNORECASE).group(1)
                        self.log.append(f"found gene mask from FOV {fov_num}")
                        self.gene_df_file = os.path.join(os.path.split(image_filepath)[0],
                                                         "FOV" + fov_num + "_area_counts.tsv")
                        self.log.append(f"opening tsv file: {self.gene_df_file}")
                        try:
                            self.gene_df = pd.read_table(self.gene_df_file)
                            print(self.gene_df)

                            # if dataframe sucessfully loaded, refresh and populate the comboboxes with gene names
                            for combobox in self.comboboxList:
                                combobox.clear()
                                combobox.addItem("no gene selected")
                                for gene_name, fpkm in zip(self.gene_df["gene_names"],
                                                           self.gene_df["FPKM_data"]):
                                    combobox.addItem(f"{gene_name} ({fpkm})")
                                combobox.model().sort(0)
                                combobox.update()
                                combobox.currentIndexChanged[str].connect(self.handleGeneChoice)

                        except:
                            self.log.append(f"Error: Could not find or open "
                                            f"gene dataframe with expected filename:\n"
                                            f"{self.gene_df_file}")

    def handleGeneChoice(self, box_txt):
        """
        handle a change in selected gene in one of the gene comboboxes
        """

        # Find out which combobox has been changed
        comboListIndex = self.comboboxList.index(self.sender())

        # Separate the gene name from the FPKM value
        # (may or may not be present) by cutting off characters behind left bracket
        gene_str = box_txt.split("(")[0].strip()
        self.log.append(f"New gene selected in "
                        f"box {comboListIndex+1}: {gene_str}")

        # Find index of gene in dataframe, if present
        if self.gene_df is None:
            gene_dfindex = None
        else:
            gene_dfindex = pd.Index(self.gene_df["gene_names"]).get_loc(gene_str)
            self.log.append(f"Dataframe index of "
                            f"gene {gene_str}: {gene_dfindex}")

        # Update record of the current gene
        # and its corresponding dataframe index
        self.current_genes[comboListIndex] = (gene_str, gene_dfindex)

    @QtCore.pyqtSlot(object)
    def handleReadFinish(self, nparray):
        self.temp_image_ref = nparray
        print("shape of temp array:", self.temp_image_ref.shape)

    @QtCore.pyqtSlot()
    def handleThreadFinish(self):
        print("Thread Finished!")


#
# -----------------------------------------------------------------------------------
#                                     Run App
# -----------------------------------------------------------------------------------
#

if __name__ == "__main__":
    app = QApplication(sys.argv)

    main = GeneViewerMainWidget()
    main.show()

    sys.exit(app.exec_())
