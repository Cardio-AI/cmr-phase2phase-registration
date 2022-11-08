# define logging and working directory
from ProjectRoot import change_wd_to_project_root

change_wd_to_project_root()

# import helper functions
import numpy as np
# import pandas as pd
import SimpleITK as sitk
import logging
from logging import info as INFO
from src_julian.utils.skhelperfunctions import Console_and_file_logger
from src_julian.utils.myhelperfunctions import *
from src_julian.utils.myclasses import *
from stl import mesh
# from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import math
import matplotlib.cm as cm
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
import matplotlib as mpl

# set up logging
Console_and_file_logger('mvfviz/dmd_temp', logging.INFO)

# set the colormap and centre the colorbar
# https://github.com/mne-tools/mne-python/issues/5693
# http://chris35wills.github.io/matplotlib_diverging_colorbar/
class MidpointNormalize(mpl.colors.Normalize):
    """Normalise the colorbar."""

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

# ----------------------------------------------------------------------------------------------#
# --------PARAMS---------------------#
PERCENTAGE = 0.1  # state value in decimal
FLOW_VALUE = 0  # which values shall be masked at plotting
QUIVER_VALUE = 0.01  # the value to which arrows under the threshold will be set
QUIVER_SCALE = 0.2
QUIVER_WIDTH = 0.75
QUIVER_COLORMAP = 'y'
NTH_VECTOR_PLOTTED = 4  # quiverplots
SLICE0 = 32
NTIMESTEPS = 5
Z_SPACING = 1
# PATH_TO_TODAYSFOLDER = '/mnt/ssd/julian/data/raw/flowfields/v3_firstDMDpredictions/mypatient/hb_20191227/'
PATH_TO_TODAYSFOLDER = '/mnt/ssd/julian/data/raw/flowfields/v5/pred/hh_20190621/'
SEG_ONLY_COLORMAP = 'Reds'
naming_volume = '_cmr_full_'
naming_segmentation = '_mask_'
# --------PARAMS---------------------#


x = ['ED', 'MS', 'ES', 'PF', 'MD']
mpl.style.use('seaborn-bright')
fig, ax = plt.subplots(1, len(x), figsize=(20, 8), sharey=True)
plt.subplots_adjust(left=0.05, bottom=0.1)
fig.suptitle('flowfield analyzer GUI', fontsize=25)

ff = stack_nii_flowfield(PATH_TO_TODAYSFOLDER, NTIMESTEPS)
ff_whole = mvf(ff, '4Dt', Z_SPACING)

vol = stack_nii_volume(PATH_TO_TODAYSFOLDER, naming_volume, NTIMESTEPS)
vol_whole = volume(vol, '4Dt', Z_SPACING)
vol_whole.Data = append_array_entries(vol_whole.Data)

seg_array = get_segmentationarray(PATH_TO_TODAYSFOLDER, naming_segmentation, NTIMESTEPS, Z_SPACING) # is 5,64,128,128

z_array = mvf(ff[..., 0], '4Dt', Z_SPACING)
z_array.Data = append_array_entries(z_array.Data)
y_array = mvf(ff[..., 1], '4Dt', Z_SPACING)
y_array.Data = append_array_entries(y_array.Data)
x_array = mvf(ff[..., 2], '4Dt', Z_SPACING)
x_array.Data = append_array_entries(x_array.Data)
mag_array = mvf(ff[..., 0], '4Dt', Z_SPACING)
mag_array.Data = np.sqrt(ff_whole.Data[..., 0] * ff_whole.Data[..., 0]
                         + ff_whole.Data[..., 1] * ff_whole.Data[..., 1]
                         + ff_whole.Data[..., 2] * ff_whole.Data[..., 2])
mag_array.Data = append_array_entries(mag_array.Data)

seg_array = mvf(seg_array[..., 0], '4Dt', Z_SPACING)
seg_array.Data = append_array_entries(seg_array.Data)

def get_arraycolormapnorm(radioval, chxbox):
    # check, if thresholding is wanted
    THRESHOLDING = chxbox.lines[3][0].get_visible()

    # check, if masking is wanted
    MASKING = chxbox.lines[4][0].get_visible()

    # check, if masking by segmentation data is wanted
    # if yes, only segmented areas will be printed
    SEGMENTATION = chxbox.lines[5][0].get_visible()

    # if threshold, modify arrays
    # if yes, set values within a given percentage range to a specified MASK_VALUE
    # i.e. set motion below 10% to 0
    if THRESHOLDING:
        # get array Data
        z_array_thresh = np.copy(z_array.Data)
        y_array_thresh = np.copy(y_array.Data)
        x_array_thresh = np.copy(x_array.Data)
        mag_array_thresh = np.copy(mag_array.Data)

        # thresholding
        z_array_thresh[(z_array_thresh > PERCENTAGE * z_array_thresh.min()) & (
                z_array_thresh < PERCENTAGE * z_array_thresh.max())] = FLOW_VALUE
        y_array_thresh[(y_array_thresh > PERCENTAGE * y_array_thresh.min()) & (
                y_array_thresh < PERCENTAGE * y_array_thresh.max())] = FLOW_VALUE
        x_array_thresh[(x_array_thresh > PERCENTAGE * x_array_thresh.min()) & (
                x_array_thresh < PERCENTAGE * x_array_thresh.max())] = FLOW_VALUE
        mag_array_thresh[mag_array_thresh < (mag_array_thresh.min() + PERCENTAGE * (mag_array_thresh.max() - mag_array_thresh.min()))] = FLOW_VALUE

        # preparing outputs
        if radioval == 'Motion in z':
            array = mvf(z_array_thresh, '4Dt', Z_SPACING)
            colormap = 'seismic'
            norm = MidpointNormalize(array.Data.min(), array.Data.max(), 0.0)
        if radioval == 'Motion in y':
            array = mvf(y_array_thresh, '4Dt', Z_SPACING)
            colormap = 'seismic'
            norm = MidpointNormalize(array.Data.min(), array.Data.max(), 0.0)
        if radioval == 'Motion in x':
            array = mvf(x_array_thresh, '4Dt', Z_SPACING)
            colormap = 'seismic'
            norm = MidpointNormalize(array.Data.min(), array.Data.max(), 0.0)
        if radioval == 'Magnitude':
            array = mvf(mag_array_thresh, '4Dt', Z_SPACING)
            colormap = 'jet'
            norm = MidpointNormalize(array.Data.min(), array.Data.max(), 0.3 * (array.Data.max() - array.Data.min()))

        # check if masking, because masking makes sense when combined with thresholding
        # then, values set to a MASK_VALUE because they did not fulfill the thresholding conditions,
        # will be shifted to the background
        if MASKING:
            array_thresh = np.copy(array.Data)
            array_thresh_masked = np.ma.masked_where(array.Data == FLOW_VALUE, array_thresh)
            array = mvf(array_thresh_masked, '4Dt', Z_SPACING)

            if SEGMENTATION:
                mask_masked_segmentation = np.logical_or((array == FLOW_VALUE), (seg_array.Data == False))
                array_thresh_masked = np.copy(array.Data)
                array_thresh_masked_segmentation = np.ma.masked_where(mask_masked_segmentation, array_thresh_masked)
                array = mvf(array_thresh_masked_segmentation, '4Dt', Z_SPACING)

        if SEGMENTATION:
            array_thresh = np.copy(array.Data)
            array_thresh_segmentation = np.ma.masked_where(seg_array.Data==False, array_thresh)
            array = mvf(array_thresh_segmentation, '4Dt', Z_SPACING)

    else:
        # without thresholding, original arrays
        if radioval == 'Motion in z':
            array = z_array
            colormap = 'seismic'
            norm = MidpointNormalize(array.Data.min(), array.Data.max(), 0.0)
        if radioval == 'Motion in y':
            array = y_array
            colormap = 'seismic'
            norm = MidpointNormalize(array.Data.min(), array.Data.max(), 0.0)
        if radioval == 'Motion in x':
            array = x_array
            colormap = 'seismic'
            norm = MidpointNormalize(array.Data.min(), array.Data.max(), 0.0)
        if radioval == 'Magnitude':
            array = mag_array
            colormap = 'jet'
            norm = MidpointNormalize(array.Data.min(), array.Data.max(), 0.3 * (array.Data.max() - array.Data.min()))

        if SEGMENTATION:
            # masking or segmentation
            array = np.copy(array.Data)
            array = np.ma.masked_where(seg_array.Data == False, array)
            array = mvf(array, '4Dt', Z_SPACING)

    return array, colormap, norm


def plot_axislabels():
    for t, num in enumerate(x):
        # plot the labels
        ax[0].set_ylabel('y', fontsize=20)
        for idx, num in enumerate(x):
            ax[idx].set_xlabel('x', fontsize=20)
            ax[idx].set_title(num, fontsize=20)


def get_plotcase(chxbox):
    # select the data to be plotted from CheckBox
    vis_Patient = chxbox.lines[0][0].get_visible()
    vis_Flowfield = chxbox.lines[1][0].get_visible()
    vis_Quiver = chxbox.lines[2][0].get_visible()
    vis_Thresholding = chxbox.lines[3][0].get_visible()
    vis_Masking = chxbox.lines[4][0].get_visible()
    vis_Segmentation = chxbox.lines[5][0].get_visible()

    # standard case, not defined
    plot_case = 999

    # plotcase definitions
    if vis_Patient: plot_case = 1
    if vis_Flowfield: plot_case = 2
    if vis_Quiver: plot_case = 3

    if vis_Patient & vis_Flowfield: plot_case = 12
    if vis_Patient & vis_Quiver: plot_case = 13
    if vis_Patient & vis_Segmentation: plot_case = 16

    if vis_Flowfield & vis_Quiver: plot_case = 23
    if vis_Flowfield & vis_Thresholding: plot_case = 24
    if vis_Flowfield & vis_Segmentation: plot_case = 26

    if vis_Quiver & vis_Thresholding: plot_case = 34
    if vis_Quiver & vis_Segmentation: plot_case = 36

    if vis_Patient & vis_Flowfield & vis_Quiver: plot_case = 123
    if vis_Patient & vis_Flowfield & vis_Thresholding: plot_case = 124
    if vis_Patient & vis_Flowfield & vis_Masking: plot_case = 125
    if vis_Patient & vis_Flowfield & vis_Segmentation: plot_case = 126
    if vis_Patient & vis_Quiver & vis_Thresholding: plot_case = 134
    if vis_Patient & vis_Quiver & vis_Segmentation: plot_case = 136

    if vis_Flowfield & vis_Quiver & vis_Thresholding: plot_case = 234
    if vis_Flowfield & vis_Quiver & vis_Segmentation: plot_case = 236

    if vis_Patient & vis_Flowfield & vis_Quiver & vis_Thresholding: plot_case = 1234
    if vis_Patient & vis_Flowfield & vis_Quiver & vis_Masking: plot_case = 1235
    if vis_Patient & vis_Flowfield & vis_Quiver & vis_Segmentation: plot_case = 1236
    if vis_Patient & vis_Flowfield & vis_Thresholding & vis_Masking: plot_case = 1245
    if vis_Patient & vis_Quiver & vis_Thresholding & vis_Masking: plot_case = 1345
    if vis_Patient & vis_Quiver & vis_Thresholding & vis_Segmentation: plot_case = 1346

    if vis_Flowfield & vis_Quiver & vis_Thresholding & vis_Masking: plot_case = 2345

    if vis_Patient & vis_Flowfield & vis_Quiver & vis_Thresholding & vis_Masking: plot_case = 12345
    if vis_Patient & vis_Flowfield & vis_Quiver & vis_Thresholding & vis_Segmentation: plot_case = 12346
    if vis_Patient & vis_Flowfield & vis_Thresholding & vis_Masking & vis_Segmentation: plot_case = 12456

    if vis_Patient & vis_Quiver & vis_Thresholding & vis_Masking & vis_Segmentation: plot_case = 13456

    if vis_Patient & vis_Flowfield & vis_Quiver & vis_Thresholding & vis_Masking & vis_Segmentation: plot_case = 123456

    return plot_case


def plot_colorbar(im):
    global CB_AVAILABLE
    global cb
    cb_fig = fig.add_axes([0.95, 0.25, 0.015, 0.5])
    cb = fig.colorbar(im, cax=cb_fig, extend='both')
    CB_AVAILABLE = True

def append_array_entries_triple(array_target, array_toappend, dtype):
    '''
    array_target: ndarray i.e. of shape 5,64,128,128,3
    array_toappend: ndarray i.e. of shape 5,64,128,128,1 which will be tripled to result in array_target shape
    '''
    result = np.ndarray((array_target.shape), dtype=dtype)
    result[..., 0] = array_toappend[..., 0]
    result[..., 1] = array_toappend[..., 0]
    result[..., 2] = array_toappend[..., 0]
    return result

def plot_Quiver(t, chxbox):
    THRESHOLDING = chxbox.lines[3][0].get_visible()
    MASKING = chxbox.lines[4][0].get_visible()
    SEGMENTATION = chxbox.lines[5][0].get_visible()

    # prepare segmentation arrays
    quiver_array_original = np.copy(ff_whole.Data)
    seg_array_original = np.copy(seg_array.Data)
    seg_array_shapeofquiver = append_array_entries_triple(quiver_array_original, seg_array_original, dtype=bool)

    if THRESHOLDING:
        # get the current vector flowfield
        quiver_array_thresh = np.copy(ff_whole.Data) # this is a ndarray

        # get locations where the magnitude is too low
        idxs = mag_array.Data < (mag_array.Data.min() + PERCENTAGE * (mag_array.Data.max() - mag_array.Data.min()))

        # create an array with the same shape as the quiver array and fill it with the booleans
        idxs_shapeofquiver = append_array_entries_triple(quiver_array_thresh, idxs, dtype=bool)

        # set the array direction to a small value where the bool is True
        quiver_array_thresh[idxs_shapeofquiver] = QUIVER_VALUE

        # set the result to the resulting array
        quiver_array_result = quiver_array_thresh

        # check if masking, because masking makes sense when combined with thresholding
        # then, values set to a MASK_VALUE because they did not fulfill the thresholding conditions,
        # will be shifted to the background
        if MASKING:
            quiver_array_thresh_masked = np.ma.masked_where(quiver_array_thresh == QUIVER_VALUE,
                                                            quiver_array_thresh)
            quiver_array_result = quiver_array_thresh_masked
            if SEGMENTATION:
                mask_thresh_masked_segmentation = np.logical_or((quiver_array_thresh == QUIVER_VALUE),
                                                                (seg_array_shapeofquiver==False))
                quiver_array_thresh_masked_segmentation = np.ma.masked_where(mask_thresh_masked_segmentation,
                                                                             quiver_array_thresh)
                quiver_array_result = quiver_array_thresh_masked_segmentation
                xx, yy, Fx, Fy = mvf(quiver_array_result[t], '4D', Z_SPACING).plot_Grid2D_MV2Dor3D(slider1.val, NTH_VECTOR_PLOTTED)
                ax[t].quiver(xx, yy, Fx, Fy, units='xy', angles='xy', scale=QUIVER_SCALE, color=QUIVER_COLORMAP, width=QUIVER_WIDTH)
                plt.show()
                return


        if SEGMENTATION:
            quiver_array_thresh_segmentation = np.ma.masked_where(seg_array_shapeofquiver==False, quiver_array_thresh)
            quiver_array_result = quiver_array_thresh_segmentation

        # needs to access: self.Data[slice, :, :, 0]
        xx, yy, Fx, Fy = mvf(quiver_array_result[t], '4D', Z_SPACING).plot_Grid2D_MV2Dor3D(slider1.val, NTH_VECTOR_PLOTTED)
        ax[t].quiver(xx, yy, Fx, Fy, units='xy', angles='xy', scale=QUIVER_SCALE, color=QUIVER_COLORMAP, width=QUIVER_WIDTH)

    elif SEGMENTATION:
        quiver_array_segmentation = np.ma.masked_where(seg_array_shapeofquiver==False, quiver_array_original)
        xx, yy, Fx, Fy = mvf(quiver_array_segmentation[t], '4D', Z_SPACING).plot_Grid2D_MV2Dor3D(slider1.val, NTH_VECTOR_PLOTTED)
        ax[t].quiver(xx, yy, Fx, Fy, units='xy', angles='xy', scale=QUIVER_SCALE, color=QUIVER_COLORMAP, width=QUIVER_WIDTH)

    else:
        # quiverplots without any additional filters
        xx, yy, Fx, Fy = mvf(ff[t], '4D', Z_SPACING).plot_Grid2D_MV2Dor3D(slider1.val, NTH_VECTOR_PLOTTED)
        ax[t].quiver(xx, yy, Fx, Fy, units='xy', angles='xy', scale=QUIVER_SCALE, color=QUIVER_COLORMAP, width=QUIVER_WIDTH)

    plt.show()

def plot_patientdata(t, *args, **kwargs):
    alpha = kwargs.get('alpha', None)
    ax[t].imshow(vol_whole.Data[t, slider1.val, ..., 0], cmap='gray', interpolation='bilinear', alpha=alpha)

def plot_flowfield(t, array, colormap, *args, **kwargs):
    alpha = kwargs.get('alpha', None)
    im = ax[t].imshow(array.Data[t, slider1.val, ..., 0],
                      cmap=colormap,
                      interpolation='bilinear',
                      alpha=alpha)
    return im

def figure_update():
    # get values, colormaps and norms from RadioButtons
    array, colormap, norm = get_arraycolormapnorm(radio1.value_selected, chxbox)

    # get plotcase
    plot_case = get_plotcase(chxbox)

    # clear the axes to care for non-intended overlays
    clear_axes()

    # plot
    if plot_case == 1:
        for t, num in enumerate(x):
            plot_patientdata(t)
    if plot_case == 16:
        for t, num in enumerate(x):
            plot_patientdata(t)
            plot_flowfield(t, array, colormap=SEG_ONLY_COLORMAP)
    if plot_case in (12, 124, 126):
        for t, num in enumerate(x):
            im = plot_flowfield(t, array, colormap, alpha=1.0)
            plot_patientdata(t, alpha=0.5)
        plot_colorbar(im)
    if plot_case in (1245, 12456):
        for t, num in enumerate(x):
            plot_patientdata(t)
            im = plot_flowfield(t, array, colormap)
        plot_colorbar(im)
    if plot_case in (13, 134, 136, 1345, 1346, 13456):
        for t, num in enumerate(x):
            plot_patientdata(t)
            plot_Quiver(t, chxbox)
    if plot_case in (123, 1234, 1235, 1236, 12345, 12346, 123456):
        for t, num in enumerate(x):
            plot_patientdata(t, alpha=0.5)
            im = plot_flowfield(t, array, colormap)
            plot_Quiver(t, chxbox)
        plot_colorbar(im)
    if plot_case in (2, 24, 26):
        for t, num in enumerate(x):
            im = plot_flowfield(t, array, colormap)
        plot_colorbar(im)
    if plot_case in (23, 234, 236, 2345):
        for t, num in enumerate(x):
            im = plot_flowfield(t, array, colormap)
            plot_Quiver(t, chxbox)
        plot_colorbar(im)
    if plot_case in (3, 34, 36):
        for t, num in enumerate(x):
            plot_Quiver(t, chxbox)

    plot_axislabels()

    plt.show()

def clear_axes():
    global CB_AVAILABLE
    # clear colorbars
    if CB_AVAILABLE:
        cb.remove()
        CB_AVAILABLE = False

    # clear imshow axes
    for t, num in enumerate(x):
        ax[t].cla()


def slice_update(val):
    figure_update()
    fig.canvas.draw()


def reset_sliders(event):
    slider1.reset()


def parameter_update(label):
    figure_update()
    fig.canvas.draw()


def initialize_figure():
    global CB_AVAILABLE
    CB_AVAILABLE = False
    figure_update()


def visibility_update(label):
    figure_update()
    fig.canvas.draw()


axSlider1 = plt.axes([0.1, 0.15, 0.8, 0.025])
slider1 = Slider(ax=axSlider1,
                 label='Apex to Base',
                 valmin=0,
                 valmax=63,
                 valinit=SLICE0,
                 valstep=1,
                 valfmt='%i',
                 color='black')
slider1.on_changed(slice_update)

axButton1 = plt.axes([0.3, 0.75, 0.1, 0.1])
button1 = Button(axButton1,
                 'Reset to the mid slice',
                 color='white',
                 hovercolor='green')
button1.on_clicked(reset_sliders)

raxButton1 = plt.axes([0.45, 0.75, 0.1, 0.15])
radio1 = RadioButtons(raxButton1,
                      ['Motion in z', 'Motion in y', 'Motion in x', 'Magnitude'],
                      active=3,
                      activecolor='red')
radio1.on_clicked(parameter_update)

axCheckButton = plt.axes([0.6, 0.75, 0.1, 0.15])
chxbox = CheckButtons(axCheckButton,
                      ['patient data', 'flowfield', 'quiverplot', 'thresholding', 'masking', 'segmentation'],
                      [False, False, False, False, False, False])
chxbox.on_clicked(visibility_update)

initialize_figure()

INFO('Hallo')


# def original_arrays():
#     ff = stack_nii_flowfield(PATH_TO_TODAYSFOLDER, N_TIMESTEPS)
#     ff_whole = mvf(ff, '4Dt', Z_SPACING)
#
#     vol = stack_nii_volume(PATH_TO_TODAYSFOLDER, N_TIMESTEPS)
#     vol_whole = volume(vol, '4Dt', Z_SPACING)
#     vol_whole.Data = append_array_entries(vol_whole.Data)
#
#     z_array = mvf(ff[..., 0], '4Dt', Z_SPACING)
#     z_array.Data = append_array_entries(z_array.Data)
#     y_array = mvf(ff[..., 1], '4Dt', Z_SPACING)
#     y_array.Data = append_array_entries(y_array.Data)
#     x_array = mvf(ff[..., 2], '4Dt', Z_SPACING)
#     x_array.Data = append_array_entries(x_array.Data)
#     mag_array = mvf(ff[..., 0], '4Dt', Z_SPACING)
#     mag_array.Data = np.sqrt(ff_whole.Data[..., 0] * ff_whole.Data[..., 0]
#                              + ff_whole.Data[..., 1] * ff_whole.Data[..., 1]
#                              + ff_whole.Data[..., 2] * ff_whole.Data[..., 2])
#     mag_array.Data = append_array_entries(mag_array.Data)
#
#     return ff, vol_whole, z_array, y_array, x_array, mag_array

# l.set_clim([new_img.min(), new_img.max()])


# ff_whole        = mvf(ff, '4Dt', Z_SPACING)
# ff_base         = mvf(ff[:, 42:64, ...], '4Dt', Z_SPACING)
# ff_midcavity    = mvf(ff[:, 20:42, ...], '4Dt', Z_SPACING)
# ff_apex         = mvf(ff[:, 0:20, ...], '4Dt', Z_SPACING)
#
#
# base = calculate_parameters(ff_base)
# midcavity = calculate_parameters(ff_midcavity)
# apex = calculate_parameters(ff_apex)
#
#
#
# x = ['ED', 'MS', 'ES', 'PF', 'MD']
# parameters = ['global average displacement z', 'global average displacement y', 'global average displacement x', 'global average magnitude']
# fig, ax = plt.subplots(1, 3, figsize=(15, 7), sharey=True)
#
# for idx, num in enumerate(base):
#     ax[0].plot(num, label=parameters[idx])
# for idx, num in enumerate(midcavity):
#     ax[1].plot(num, label=parameters[idx])
# for idx, num in enumerate(apex):
#     ax[2].plot(num, label=parameters[idx])
#
# ax[0].set_ylabel('Values')
# ax[0].set_title('Base')
# ax[1].set_title('Mid-Cavity')
# ax[2].set_title('Apex')
#
# for xc in x: ax[0].axvline(x=xc, color='k', linestyle='--', linewidth=0.5), \
#              ax[1].axvline(x=xc, color='k', linestyle='--', linewidth=0.5), \
#              ax[2].axvline(x=xc, color='k', linestyle='--', linewidth=0.5)
#
# ax[1].legend(parameters, loc="lower center", bbox_to_anchor=(0.5, -0.3))
# fig.subplots_adjust(bottom=0.25)
#
# plt.show()
# INFO('Hi')


# box = ax.get_position()
# ax.set_position([box.x0, box.y0 + box.height * 0.1,
#                  box.width, box.height * 0.9])
#
# # Put a legend below current axis
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
#           fancybox=True, shadow=True, ncol=5)


# code


# load mvf
# data = np.load(PATH_TO_TODAYSFOLDER + 'flow.npz.npy')
# mvf_ED = data[0,0,...]
# mymvf = mvf(mvf_ED, '4D', Z_SPACING)
# samplearray = np.random.rand(15, 128, 128, 2)
# samplemvf = mvf(samplearray, '4D', Z_SPACING)

# plot mvf
# N = 1
# slice = 0
# fig, ax = plt.subplots()
# xx,yy,Fx,Fy = mymvf.plot_Grid2D_MV2Dor3D(slice, N)
# plt.quiver(xx, yy, Fx, Fy, units='xy', angles='xy')
# ax.set_title('MVF plot')
# plt.show()

# test = mymesh.x
# mymesh = mesh.Mesh.from_file(PATH_TO_TODAYSFOLDER + 'mysegmentation_Villard_simple.stl')
# fig = pyplot.figure()
# ax = mplot3d.Axes3D(fig)
# ax.add_collection3d(mplot3d.art3d.Poly3DCollection(mymesh.vectors))
# scale = mymesh.points.flatten()
# ax.auto_scale_xyz(scale, scale, scale)


# filename = 'coloredmesh'
# values = np.linspace(0, mymesh.data.size, mymesh.data.size)
# x=mymesh.x[:,0]
# y=mymesh.y[:,0]
# z=mymesh.z[:,0]
# df = pd.DataFrame({"x": x, "y": y, "z": z, "value": values})
# df.to_csv(PATH_TO_TODAYSFOLDER + filename + ".csv", index=False)

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(x, y, z, s=20, c=values)
# ax.set_xlabel('X'), ax.set_ylabel('Y'), ax.set_zlabel('Z')
# plt.show()


# PATH_TO_NEWVOL = '/mnt/ssd/julian/data/raw/flowfields/flowinputtarget2/example_flow_0/_cmr_0_.nii'
# PATH_TO_NEWFF = '/mnt/ssd/julian/data/raw/flowfields/flowinputtarget2/example_flow_0/_flow_0_.nii'


# img_ff=sitk.ReadImage(PATH_TO_FF)
# data_ff=sitk.GetArrayFromImage(img_ff)
# data_ff=np.einsum('cxyz->zyxc',data_ff)
# ff=mvf(data_ff,'4D',Z_SPACING)
# filename_ff = 'test_ff_'
# ff.save_as_csv(filename_ff, PATH_TO_OUTPUT_FOLDER)

# img_ff = sitk.ReadImage(PATH_TO_NEWFF)
# ff = sitk.GetArrayFromImage(img_ff)
# ff = np.einsum('cxyz->zyxc', ff)
#
# img_volume=sitk.ReadImage(PATH_TO_VOL)
# data_volume=sitk.GetArrayFromImage(img_volume)
# data_volume=np.einsum('xyz->zyx',data_volume)
#
# # we have to append the pixel values in the volume data
# volume_enlarged = np.ndarray((64,128,128,1))
# volume_enlarged[...,0] = data_volume
# data_volume = volume_enlarged
#
# vol=volume(data_volume,'4D',Z_SPACING)
# filename_vol = 'test_vol_'
# vol.save_as_csv(filename_vol, PATH_TO_OUTPUT_FOLDER)

# N = 1
# slice = 15
# fig, ax = plt.subplots()
# xx, yy, Fx, Fy = ff.plot_Grid2D_MV2Dor3D(slice, N)
# plt.quiver(xx, -yy, Fx, Fy, units='xy', angles='xy')
# ax.set_title('MVF plot')
# plt.show()


# load mvf
# data = np.load(PATH_TO_TODAYSFOLDER + 'flow.npz.npy')
# mvf_ED = data[0,0,...]
# mymvf = mvf(mvf_ED, '4D', Z_SPACING)
# samplearray = np.random.rand(15, 128, 128, 2)
# samplemvf = mvf(samplearray, '4D', Z_SPACING)

# plot mvf
# N = 1
# slice = 0
# fig, ax = plt.subplots()
# xx,yy,Fx,Fy = mymvf.plot_Grid2D_MV2Dor3D(slice, N)
# plt.quiver(xx, yy, Fx, Fy, units='xy', angles='xy')
# ax.set_title('MVF plot')
# plt.show()

# test = mymesh.x
# mymesh = mesh.Mesh.from_file(PATH_TO_TODAYSFOLDER + 'mysegmentation_Villard_simple.stl')
# fig = pyplot.figure()
# ax = mplot3d.Axes3D(fig)
# ax.add_collection3d(mplot3d.art3d.Poly3DCollection(mymesh.vectors))
# scale = mymesh.points.flatten()
# ax.auto_scale_xyz(scale, scale, scale)


# filename = 'coloredmesh'
# values = np.linspace(0, mymesh.data.size, mymesh.data.size)
# x=mymesh.x[:,0]
# y=mymesh.y[:,0]
# z=mymesh.z[:,0]
# # df = pd.DataFrame({"x": x, "y": y, "z": z, "value": values})
# # df.to_csv(PATH_TO_TODAYSFOLDER + filename + ".csv", index=False)
#
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(x, y, z, s=20, c=values)
# ax.set_xlabel('X'), ax.set_ylabel('Y'), ax.set_zlabel('Z')
# plt.show()
#
#
# PATH_TO_NEWVOL = '/mnt/ssd/julian/data/raw/flowfields/flowinputtarget2/example_flow_0/_cmr_0_.nii'
# PATH_TO_NEWFF = '/mnt/ssd/julian/data/raw/flowfields/flowinputtarget2/example_flow_0/_flow_0_.nii'
#
# img_ff = sitk.ReadImage(PATH_TO_NEWFF)
# ff = sitk.GetArrayFromImage(img_ff)
# ff = np.einsum('cxyz->zyxc', ff)
#
#
# INFO('Hi')


# PATH_TO_FF_OUTPUT_FOLDER = '/mnt/ssd/julian/data/outputs/flowfields/'
# PATH_TO_VTK_OUTPUT_FOLDER = '/mnt/ssd/julian/data/outputs/vtk/'
# PATH_TO_CSV_OUTPUT_FOLDER = '/mnt/ssd/julian/data/raw/paraview/examples/myexamplefiles/'
# PATH_TO_REALFF = '/mnt/ssd/julian/data/raw/flowfields/es_ed_myo/flowfield_3d_gcn_3HQQHA3N.npy'
# PATH_TO_NEWFF = '/mnt/ssd/julian/data/raw/flowfields/exampleflow.npz.npy'
# PATH_TO_FLOW = '/mnt/ssd/julian/data/raw/flowfields/flowinputtarget/flow.npz.npy'
# PATH_TO_INPUT = '/mnt/ssd/julian/data/raw/flowfields/flowinputtarget/input.npz.npy'
# PATH_TO_NRRD_MASK = '/mnt/ssd/julian/data/interim/2021.04.20_allcopyrotationfalse/DMD/sax/aa_20180710_volume_mask.nrrd'
# PATH_TO_NRRD_CLEAN = '/mnt/ssd/julian/data/interim/2021.04.20_allcopyrotationfalse/DMD/sax/aa_20180710_volume_clean.nrrd'
# PATH_TO_FILTER_SCRIPT = '/home/julian/meshlab/2021.05.meshlabfilterscript_smoothingsubdivisiondecimation.mlx'
# PATH_TO_STL = '/mnt/ssd/julian/data/raw/paraview/examples/myexamplefiles/segmentation_0.stl'
# PATH_TO_STL_OUTPUT_FOLDER = '/mnt/ssd/julian/data/raw/paraview/examples/myexamplefiles/'
# PATH_TO_MY_SAMPLE_SEGMENTATION = '/mnt/ssd/julian/data/raw/paraview/examples/myexamplefiles/mysegmentation.nrrd'
# PATH_TO_INPUT_TARGET_FLOW_PATIENT_FOLDER = '/mnt/ssd/julian/data/raw/paraview/examples/myexamplefiles/raw npy nrrd input target/'
# PATH_TO_MYSEGMENTATION = '/mnt/ssd/julian/data/raw/paraview/examples/myexamplefiles/raw npy nrrd input target/mysegmentation.nrrd'
# PATH_TO_MYSEGMENTATION2 = '/mnt/ssd/julian/data/raw/paraview/examples/myexamplefiles/raw npy nrrd input target/mysegmentation2.nrrd'
# PATH_TO_NRRD_INPUT = '/mnt/ssd/julian/data/raw/paraview/examples/myexamplefiles/raw npy nrrd input target/input.nrrd'
# PATH_TO_SEG = '/mnt/ssd/julian/data/raw/paraview/examples/myexamplefiles/raw npy nrrd input target/2021.05.21 tests/mysegmentation.nrrd'
# PATH_TO_OUTPUT_FOLDER = '/mnt/ssd/julian/data/raw/paraview/examples/myexamplefiles/raw npy nrrd input target/2021.05.21 tests/'
# PATH_TO_NRRD_INPUT = '/mnt/ssd/julian/data/raw/paraview/examples/myexamplefiles/raw npy nrrd input target/2021.05.21 tests/input.nrrd'
# PATH_TO_NPY_INPUT = '/mnt/ssd/julian/data/raw/paraview/examples/myexamplefiles/raw npy nrrd input target/2021.05.21 tests/input.npz.npy'
# PATH_TO_NPY_FF = '/mnt/ssd/julian/data/raw/paraview/examples/myexamplefiles/raw npy nrrd input target/2021.05.21 tests/flow.npz.npy'

# ----------------------------------------------------------------------------- #
# Arrays for testing
# mytest1Dff = np.random.rand(5, 2)               # 1D flowfield means grid is a line with vectors at some points
# mytest2Dff = np.random.rand(224, 224, 3)        # a 2D image with x by y pixels including one 2D vector each
# mytest3Dff = np.random.rand(15, 20, 20, 3)      # a 3D volume with x by y by z voxels including one 3D vector each
# mytestff = np.random.rand(16, 224, 224, 3)      # a realistic 3D ff
