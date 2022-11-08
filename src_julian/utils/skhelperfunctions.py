import logging
import os
import sys
from collections import Counter
import SimpleITK as sitk
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import confusion_matrix
from src_julian.utils.Utils_io import save_plot, ensure_dir
import cv2

def get_ip_from_2dmask(nda, debug=False, rev=False):
    """
    Find the RVIP on a 2D mask with the following labels
    RV (0), LVMYO (1) and LV (2) mask

    Parameters
    ----------
    nda : numpy ndarray with one hot encoded labels
    debug :
    rev: (bool), return the coordinates as x,y tuples (for comparison with matrix based indexing)

    Returns a tuple of two points anterior IP, inferior IP, each with (y,x)-coordinates
    -------

    """
    if debug: print('msk shape: {}'.format(nda.shape))
    # initialise some values
    first, second = None, None
    # find first and second insertion points
    myo_msk = (nda == 2).astype(np.uint8)
    comb_msk = ((nda == 1) | (nda == 2) | (nda == 3)).astype(np.uint8)
    if np.isin(1,nda) and np.isin(2, nda):
        myo_contours, hierarchy = cv2.findContours(myo_msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        comb_contours, hierarchy = cv2.findContours(comb_msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(myo_contours) > 0 and len(comb_contours) > 0: # we just need to search for IP if there are two contours
            # some lambda helpers
            # transform and describe contour lists to pythonic list which makes "elem in" syntax possible
            clean_contour = lambda cont: list(map(lambda x: (x[0][0], x[0][1]), cont[0]))
            descr_cont = lambda cont: print(
                'len: {}, first elem: {}, type of one elem: {}'.format(len(cont), cont[0], type(cont[0])))

            # clean/describe both contours
            myo_clean = clean_contour(myo_contours)
            if debug: descr_cont(myo_clean)
            comb_clean = clean_contour(comb_contours)
            if debug: descr_cont(comb_clean)

            # initialise some values
            septum_visited = False
            border_visited = False
            memory_first = None
            for p in myo_clean:
                if debug: print('p {} in {}'.format(p, p in comb_clean))
                # we are at the border,
                # moving anti-clockwise,
                # we dont know if we are in the septum
                # no second IP found so far.

                if p in comb_clean:
                    border_visited = True
                    if septum_visited and not second:
                        # take the first point after the septum as second IP
                        # we are at the border
                        # we have been at the septum
                        # no second defined so far
                        second = p
                        if debug: print('second= {}'.format(second))

                    # we are at the border
                    if not first:
                        # if we haven't been at the septum, update/remember this point
                        # use the last visited point before visiting the septum as first IP
                        memory_first = p
                        if debug: print('memory= {}'.format(memory_first))
                else:
                    septum_visited = True  # no contour points matched --> we are at the septum
                    if border_visited and not first:
                        first = memory_first
            if second and not first: # if our contour started at the first IP
                first = memory_first
            #assert first and second, 'missed one insertion point: first: {}, second: {}'.format(first, second)
            if debug: print('first IP: {}, second IP: {}'.format(first, second))
        if rev and (first is not None) and (second is not None): first, second = (first[1], first[0]), (second[1], second[0])

    return first, second

def get_ip_from_mask_3d(msk_3d, debug=False, keepdim=False, rev=False):
    '''
    Returns two lists of RV insertion points (y,x)-coordinates
    For a standard SAX orientation:
    the first list belongs to the anterior IP and the second to the inferior IP
    Parameters
    ----------
    msk_3d : (np.ndarray) with z,y,x
    debug : (bool) print additional info
    keepdim: (bool) returns two lists of the same length as z, slices where no RV IPs were found are represented by an tuple of None

    Returns tuple of lists with points (y,x)-coordinates
    -------
    '''

    first_ips = []
    second_ips = []
    for msk2d in msk_3d:

        try:
            first, second = get_ip_from_2dmask(msk2d, debug=debug, rev=rev)
            if (first and second) or keepdim:
                first_ips.append(first)
                second_ips.append(second)
        except Exception as e:
            print(str(e))
            pass

    return first_ips, second_ips

def my_autopct(pct):
    """
    Helper to filter % values of a pie chart, which are smaller than 1%
    :param pct:
    :return:
    """
    return ('%1.0f%%' % pct) if pct > 1 else ''


def get_metadata_maybe(sitk_img, key, default='not_found'):
    # helper for unicode decode errors
    try:
        value = sitk_img.GetMetaData(key)
    except Exception as e:
        logging.debug('key not found: {}, {}'.format(key, e))
        value = default
    # need to encode/decode all values because of unicode errors in the dataset
    if not isinstance(value, int):
        value = value.encode('utf8', 'backslashreplace').decode('utf-8').replace('\\udcfc', 'ue')
    return value


def show_2D_or_3D(img=None, mask=None, save=False, file_name='reports/figure/temp.png', dpi=200, f_size=(5, 5),
                  interpol='bilinear'):
    """
    Wrapper for 2D/3D or 4D image/mask visualisation.
    Single point for plotting, as this wrapper will delegate the plotting according to the input dim
    Works with image,mask tuples but also with image,None internal it calls show_transparent, plot_3d or plot_4d
    Parameters
    ----------
    img : np.ndarray - with 2 <= dim <= 4
    mask : np.ndarray - with 2 <= dim <= 4
    save : bool - save this figure, please provide a file_name
    file_name : string - full-path-and-file-name-with-suffix
    dpi : int - dpi of the saved figure, will be used by matplotlib
    f_size : tuple of int - define the figure size
    interpol : enumaration - interpolation method for matplotlib e.g.: 'linear', 'bilinear', None

    Returns matplotlib.figure
    -------

    """

    if isinstance(img, sitk.Image):
        img = sitk.GetArrayFromImage(img).astype(np.float32)

    if isinstance(mask, sitk.Image):
        mask = sitk.GetArrayFromImage(mask).astype(np.float32)

    # dont print anything if no images nor masks are given
    if img is None and mask is None:
        logging.error('No image data given')
        return
    if img is not None:
        dim = img.ndim
    else:
        dim = mask.ndim

    if dim == 2:
        return show_slice_transparent(img, mask)
    elif dim == 3 and img.shape[-1] == 1:  # data from the batchgenerator
        return show_slice_transparent(img, mask)
    elif dim == 3:
        return plot_3d_vol(img, mask, save=save, path=file_name, dpi=dpi, fig_size=f_size, interpol=interpol)
    elif dim == 4 and img.shape[-1] == 1:  # data from the batchgenerator
        return plot_3d_vol(img, mask, save=save, path=file_name, dpi=dpi, fig_size=f_size, interpol=interpol)
    elif dim == 4 and img.shape[-1] in [3, 4]:  # only mask
        return plot_3d_vol(img, save=save, path=file_name, dpi=dpi, fig_size=f_size, interpol=interpol)
    elif dim == 4:
        return plot_4d_vol(img, mask)
    else:
        logging.error('Unsupported dim: {}, shape: {}'.format(img.ndim, img.shape))
        raise NotImplementedError('Wrong shape Exception in: {}'.format('show_2D_or_3D()'))


def create_eval_plot(df_dice, df_haus=None, df_hd=None, df_vol=None, eval=None):
    """
    Create a violinplot with an integrated bland altmann plot
    Nobs = median
    Expects the following dataframe structure (created in notebooks/Evaluate/Evaluate_create_plots.ipynb):
        Name 	Dice LV 	Volume LV 	Err LV(ml) 	Hausdorff LV 	Dice RV 	Volume RV 	Err RV(ml) 	Hausdorff RV 	Dice MYO 	Volume MYO 	Err MYO(ml) 	Hausdorff MYO
    0 	0000-0HQQW4ZN_2007-05-23_ED_msk 	0.897436 	110.887500 	-7.106250 	5.744563 	0.868490 	149.231250 	-30.600000 	7.211103 	0.619342 	57.768750 	-2.925000 	10.000000
    1 	0000-0HQQW4ZN_2007-05-23_ES_msk 	0.850738 	43.443750 	4.921875 	4.123106 	0.830049 	82.743750 	-3.862500 	10.816654 	0.695500 	51.993750 	2.325000 	5.830952
    Parameters
    ----------
    df_dice : pd.dataframe - melted dice dataframe
    df_haus : pd.dataframe - melted dataframe with the hausdorff
    df_hd : pd.dataframe - melted dataframe with the difference (pred-gt) of the volumes
    df_vol : pd.dataframe - melted dataframe with the predicted volume
    eval : pd.dataframe - full dataframe as shown in the fn description

    Returns
    -------

    """

    import seaborn as sns
    outliers = False
    # make sure the color schema reflects the RGB schema of show_slice_transparent
    my_pal_1 = {"Dice LV": "dodgerblue", "Dice MYO": "g", "Dice RV": "darkorange"}
    my_pal_2 = {"Err LV(ml)": "dodgerblue", "Err MYO(ml)": "g", "Err RV(ml)": "darkorange"}
    my_pal_3 = {"Volume LV": "dodgerblue", "Volume MYO": "g", "Volume RV": "darkorange"}
    hd_pal = {"Hausdorff LV": "dodgerblue", "Hausdorff MYO": "g", "Hausdorff RV": "darkorange"}

    plt.rcParams.update({'font.size': 20})
    if df_haus is not None:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(25, 8), sharey=False)
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25, 8), sharey=False)

    ax1 = sns.violinplot(x='variable', y='value', data=df_dice, order=["Dice LV", "Dice MYO", "Dice RV"],
                         palette=my_pal_1, showfliers=outliers, ax=ax1)
    mean = df_dice.groupby(['variable'])['value'].mean().round(2)
    sd = df_dice.groupby(['variable'])['value'].std().round(2)
    nobs = ['{}+/-{}'.format(m, s) for m, s in zip(mean, sd)]

    for tick, label in zip(range(len(ax1.get_xticklabels())), ax1.get_xticklabels()):
        _ = ax1.text(tick, mean[tick], nobs[tick], horizontalalignment='center', size='x-small', color='black',
                     weight='semibold')
    plt.setp(ax1, ylim=(0, 1))
    plt.setp(ax1, ylabel=('DICE'))
    plt.setp(ax1, xlabel='')
    ax1.set_xticklabels(['LV', 'MYO', 'RV'])

    # create bland altmannplot from vol diff
    ax2 = bland_altman_metric_plot(eval, ax2)

    # create violin plot for the volume
    ax3 = sns.violinplot(x='variable', y='value', order=["Volume LV", "Volume MYO", "Volume RV"], palette=my_pal_3,
                         showfliers=outliers, data=df_vol, ax=ax3)

    mean = df_vol.groupby(['variable'])['value'].mean().round(2)
    sd = df_vol.groupby(['variable'])['value'].std().round(2)
    nobs = ['{}+/-{}'.format(m, s) for m, s in zip(mean, sd)]

    for tick, label in zip(range(len(ax3.get_xticklabels())), ax3.get_xticklabels()):
        _ = ax3.text(tick, mean[tick], nobs[tick], horizontalalignment='center', size='x-small', color='black',
                     weight='semibold')
    # plt.setp(ax3, ylim=(0,500))
    plt.setp(ax3, ylabel=('Vol size in ml'))
    plt.setp(ax3, xlabel='')
    ax3.set_xticklabels(['LV', 'MYO', 'RV'])

    ax4 = sns.violinplot(x='variable', y='value', order=["Hausdorff LV", "Hausdorff MYO", "Hausdorff RV"],
                         palette=hd_pal,
                         showfliers=outliers, data=df_haus, ax=ax4)

    mean = df_haus.groupby(['variable'])['value'].mean().round(2)
    sd = df_haus.groupby(['variable'])['value'].std().round(2)
    nobs = ['{}+/-{}'.format(m, s) for m, s in zip(mean, sd)]

    for tick, label in zip(range(len(ax4.get_xticklabels())), ax4.get_xticklabels()):
        _ = ax4.text(tick, mean[tick], nobs[tick], horizontalalignment='center', size='x-small', color='black',
                     weight='semibold')
    plt.setp(ax4, ylim=(0, 50))
    plt.setp(ax4, ylabel=('Hausdorff distance'))
    plt.setp(ax4, xlabel='')
    ax4.set_xticklabels(['LV', 'MYO', 'RV'])
    plt.tight_layout()
    return fig

def show_slice_transparent(img=None, mask=None, show=True, f_size=(5, 5), ax=None, interpol='none'):
    """
    Plot image + masks in one figure
    Parameters
    ----------
    img : np.ndarray - image with the shape x,y
    mask :  np.ndarray - mask with the shape x,y,channel --> one channel per label with bool values
    show : bool - this is necessary for the tf.keras callbacks, true returns the ax, otherwise we return the figure
    f_size : tuple of int - specify the figure size
    ax : matplotlib.axes object - plots into that ax, if given, creates a new one otherwise

    Returns ax or figure object
    -------

    """

    # If mask has int values (0 - #of_labels) instead of channeled bool values, define the labels of interest
    # not provided as fn-parameter to reduce the complexity
    mask_values = [1, 2, 3]
    # define a threshold if we have a mask from a sigmoid/softmax output-layer which is not binary
    mask_threshold = 0.5

    if isinstance(img, sitk.Image):
        img = sitk.GetArrayFromImage(img).astype(np.float32)

    if isinstance(mask, sitk.Image):
        mask = sitk.GetArrayFromImage(mask).astype(np.float32)

    # dont print anything if no images nor masks are given
    if img is None and mask is None:
        logging.error('No image data given')
        return

    # replace mask with empty slice if none is given
    if mask is None:
        shape = img.shape
        mask = np.zeros((shape[0], shape[1], 3))

    # replace image with empty slice if none is given
    if img is None:
        shape = mask.shape
        img = np.zeros((shape[0], shape[1], 1))

    # check image shape
    if len(img.shape) == 2:
        # image already in 2d shape take it as it is
        x_ = (img).astype(np.float32)
    elif len(img.shape) == 3:
        # take only the first channel, grayscale - ignore the others
        x_ = (img[..., 0]).astype(np.float32)
    else:
        logging.error('invalid dimensions for image: {}'.format(img.shape))
        return

    # check masks shape, handle mask without channel per label
    if len(mask.shape) == 2:  # mask with int as label values
        y_ = transform_to_binary_mask(mask, mask_values=mask_values).astype(np.float32)
    elif len(mask.shape) == 3 and mask.shape[2] == 1:  # handle mask with empty additional channel
        mask = mask[..., 0]
        y_ = transform_to_binary_mask(mask, mask_values=mask_values).astype(np.float32)

    elif len(mask.shape) == 3 and mask.shape[2] == 3:  # handle mask with three channels
        y_ = (mask).astype(np.float32)
    elif len(mask.shape) == 3 and mask.shape[2] == 4:  # handle mask with 4 channels (backround = first channel)
        # ignore background channel for plotting
        y_ = (mask[..., 1:] > mask_threshold).astype(np.float32)
    else:
        logging.error('invalid dimensions for masks: {}'.format(mask.shape))
        return

    if not ax:  # no axis given
        fig = plt.figure(figsize=f_size)
        ax = fig.add_subplot(1, 1, 1, frameon=False)
    else:  # axis given get the current fig
        fig = plt.gcf()

    fig.tight_layout(pad=0)
    ax.axis('off')

    # normalise image, avoid interpolation by matplotlib to have full control
    x_ = (x_ - x_.min()) / (x_.max() - x_.min() + sys.float_info.epsilon)
    ax.imshow(x_, 'gray', vmin=0, vmax=0.4)
    ax.imshow(y_, interpolation=interpol, alpha=.3)

    if show:
        return ax
    else:
        return fig


def bland_altman_metric_plot(metric, ax=None):
    '''
    Plots a Bland Altmann plot for a evaluation dataframe from the eval scripts
    :param metric: pd.Dataframe
    :return: plt.ax
    Parameters
    ----------
    metric :
    ax :

    Returns
    -------

    '''
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 15))

    my_colors = {"LV": "dodgerblue", "MYO": "g", "RV": "darkorange"}

    def bland_altman_plot(data1, data2, shifting, *args, **kwargs):
        """
        Create a single bland altmann plot into the ax object of the surounding wrapper function,
        this functions will be called for each label
        :param data1: list - prediction
        :param data2: list - ground truth
        :param shifting: float - relative y-axis shift from 0
        :param args:
        :param kwargs:
        :return: None
        """
        from scipy.stats import ttest_ind
        # stat, p = wilcoxon(data1, data2)
        # print('wilcoxon rank test: {}, {}'.format(stat, p))
        stat, p = ttest_ind(data1, data2)
        print('T-test - stats: {}, p: {}'.format(stat, p))

        data1 = np.asarray(data1)
        data2 = np.asarray(data2)
        mean = np.mean([data1, data2], axis=0)
        diff = data1 - data2  # Difference between data1 and data2
        md = np.mean(diff)  # Mean of the difference
        sd = np.std(diff, axis=0)  # Standard deviation of the difference

        # plot points and lines
        line_size = 4
        ax.scatter(mean, diff, alpha=0.4, s=200, *args, **kwargs)
        ax.axhline(md, **kwargs, linestyle='-', alpha=0.5, lw=line_size)
        ax.axhline(md + 1.96 * sd, **kwargs, linestyle='--', alpha=0.5, lw=line_size)
        ax.axhline(md - 1.96 * sd, **kwargs, linestyle='--', alpha=0.5, lw=line_size)

        # calculate Properties for text
        trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
        limitOfAgreement = 1.96
        limitOfAgreementRange = (md + (limitOfAgreement * sd)) - (md - limitOfAgreement * sd)
        offset = (limitOfAgreementRange / 100.0) * 1.5
        text_size = plt.rcParams['font.size']
        ax.annotate(f'{md:.2f}' + ' ± ' + f'{sd:.2f}', xy=(0.05, shifting), xycoords='axes fraction',
                    fontsize=text_size, fontname='Cambria', weight='semibold', **kwargs)

    # plot 3 different metrics into the same plot
    pred = metric['Volume LV']
    gt = pred - metric['Err LV(ml)']
    bland_altman_plot(pred, gt, shifting=0.95, color=my_colors['LV'])

    pred = metric['Volume MYO']
    gt = pred - metric['Err MYO(ml)']
    bland_altman_plot(pred, gt, shifting=0.90, color=my_colors['MYO'])

    pred = metric['Volume RV']
    gt = pred - metric['Err RV(ml)']
    bland_altman_plot(pred, gt, shifting=0.85, color=my_colors['RV'])

    # set labels
    label_size = plt.rcParams['font.size']
    ax.set_ylabel('Vol diff \n(pred - gt) in ml', fontsize=label_size, fontname='Cambria')
    ax.set_xlabel('Mean vol in ml', fontsize=label_size, fontname='Cambria')

    # set legend
    legend_size = plt.rcParams['font.size']
    LV_patch = mpatches.Patch(color=my_colors['LV'], label='LV')
    MYO_patch = mpatches.Patch(color=my_colors['MYO'], label='MYO')
    RV_patch = mpatches.Patch(color=my_colors['RV'], label='RV')
    ax.legend(handles=[LV_patch, MYO_patch, RV_patch], prop={'size': legend_size})

    # set axis
    ax.tick_params(axis='both', labelsize=plt.rcParams['font.size'])
    # to test if fixed axes is ok just comment the next line and add back, if it didn't get bigger
    ax.set_ylim(-250, 250)
    ax.set_xlim(0, 550)

    return ax


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    '''
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Parameters
    ----------
    y_true :
    y_pred :
    classes :
    normalize :
    title :
    cmap :

    Returns
    -------

    '''
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    logging.info(cm)

    fig, ax = plt.subplots(figsize=(15, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def plot_4d_vol(img_4d, timesteps=[0], save=False, path='temp/', mask_4d=None, f_name='4d_volume.png'):
    '''
    Creates a grid with # timesteps * z-slices
    #saves all slices as fig
    expects nda with t, z, x, y
    Parameters
    ----------
    img_4d :
    timesteps : list of int defining the timesteps which should be print
    save : bool, save the plot or not
    path : path, where this fig should be saved to
    mask_4d :
    f_name :

    Returns
    -------

    '''

    if isinstance(img_4d, sitk.Image):
        img_4d = sitk.GetArrayFromImage(img_4d)

    if len(timesteps) <= 1:  # add first volume if no timesteps found
        logging.info('No timesteps given for: {}, use img.shape[0]'.format(path))
        timesteps = list(range(0, img_4d.shape[0]))
    assert (len(timesteps) == img_4d.shape[0]), 'timeteps does not match'

    if img_4d.shape[-1] == 4:
        img_4d = img_4d[..., 1:]  # ignore background if 4 channels are given

    elif img_4d.shape[-1] == 1:
        img_4d = img_4d[..., 0]  # ignore single channels at the end, matpotlib cant plot this shape

    if mask_4d is not None:  # if images and masks are provided
        if mask_4d.shape[-1] in [3, 4]:
            mask_4d = mask_4d[..., -3:]  # ignore background for masks if 4 channels are given

    # define the number of subplots
    # timesteps * z-slices
    z_size = min(int(2 * img_4d.shape[1]), 30)
    t_size = min(int(2 * len(timesteps)), 20)
    logging.info('figure: {} x {}'.format(z_size, t_size))

    # long axis volumes have only one z slice squeeze=False is necessary to avoid squeezing the axes
    fig, ax = plt.subplots(len(timesteps), img_4d.shape[1], figsize=[z_size, t_size], squeeze=False)
    for t_, img_3d in enumerate(img_4d):  # traverse trough time

        for z, slice in enumerate(img_3d):  # traverse through the z-axis
            # show slice and delete ticks
            if mask_4d is not None:
                ax[t_][z] = show_slice_transparent(slice, mask_4d[t_, z, ...], show=True, ax=ax[t_][z])
            else:
                ax[t_][z] = show_slice_transparent(slice, show=True, ax=ax[t_][z])
            ax[t_][z].set_xticks([])
            ax[t_][z].set_yticks([])
            # ax[t_][z].set_aspect('equal')
            if t_ == 0:  # set title before first row
                ax[t_][z].set_title('z-axis: {}'.format(z), color='r')
            if z == 0:  # set y-label before first column
                ax[t_][z].set_ylabel('t-axis: {}'.format(timesteps[t_]), color='r')

    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    plt.tight_layout()

    if save:
        ensure_dir(path)
        save_plot(fig, path, f_name, override=True, tight=False)
    else:
        return fig


def plot_3d_vol(img_3d, mask_3d=None, timestep=0, save=False, path='reports/figures/temp.png',
                fig_size=None, dpi=200, interpol='none'):
    '''
    plots a 3D nda, if a mask is given combine mask and image slices
    Parameters
    ----------
    img_3d :
    mask_3d :
    timestep :
    save :
    path :
    fig_size :
    dpi :
    interpol :

    Returns
    -------

    '''

    if fig_size is None:
        fig_size = [25, 8]
    max_number_of_slices = 12

    if isinstance(img_3d, sitk.Image):
        img_3d = sitk.GetArrayFromImage(img_3d)

    if isinstance(mask_3d, sitk.Image):
        mask_3d = sitk.GetArrayFromImage(mask_3d)

    # use float as dtype for all plots
    if img_3d is not None:
        img_3d = img_3d.astype(np.float32)
    if mask_3d is not None:
        mask_3d = mask_3d.astype(np.float32)

    if img_3d.max() == 0:
        logging.debug('timestep: {} - no values'.format(timestep))
    else:
        logging.debug('timestep: {} - plotting'.format(timestep))

    if img_3d.shape[-1] in [3, 4]:  # this image is a mask
        img_3d = img_3d[..., -3:]  # ignore background
        mask_3d = img_3d  # handle this image as mask
        img_3d = np.zeros((mask_3d.shape[:-1]))

    elif img_3d.shape[-1] == 1:
        img_3d = img_3d[..., 0]  # matpotlib cant plot this shape

    if mask_3d is not None:
        if mask_3d.shape[-1] in [3, 4]:
            mask_3d = mask_3d[..., -3:]  # ignore background if 4 channels are given
        elif mask_3d.shape[-1] > 4:
            mask_3d = transform_to_binary_mask(mask_3d)

    slice_n = 1
    # slice very huge 3D volumes, otherwise they are too small on the plot
    if (img_3d.shape[0] > max_number_of_slices) and (img_3d.ndim == 3):
        slice_n = img_3d.shape[0] // max_number_of_slices

    img_3d = img_3d[::slice_n]
    mask_3d = mask_3d[::slice_n] if mask_3d is not None else mask_3d

    # number of subplots = no of slices in z-direction
    fig = plt.figure(figsize=fig_size, dpi=dpi)

    for idx, slice in enumerate(img_3d):  # iterate over all slices
        ax = fig.add_subplot(1, img_3d.shape[0], idx + 1)

        if mask_3d is not None:
            ax = show_slice_transparent(img=slice, mask=mask_3d[idx], show=True, ax=ax, interpol=interpol)
        else:
            ax = show_slice_transparent(img=slice, mask=None, show=True, ax=ax, interpol=interpol)

        ax.set_xticks([])
        ax.set_yticks([])
        # ax.set_title('z-axis: {}'.format(idx), color='r', fontsize=plt.rcParams['font.size'])

    fig.subplots_adjust(wspace=0, hspace=0)
    if save:
        save_plot(fig, path, str(timestep), override=False)

    else:
        return fig


def plot_dice_per_slice_line(gt, pred, save_path='reports/figures/error_per_labelandslice.png'):
    '''
        Calculate the dice per slice, create a lineplot per label
        This is necessary to figure out if we under- or over-segmented a stack of 2D slices
        Parameters
        ----------
        gt : ground truth - 3D np.ndarray with 1,2,3 ... at the voxel-position of the label
        pred : prediction - 3D np.ndarray with 1,2,3 ... at the voxel-position of the label
        save_path : False = No savefig, str with full file and pathname otherwise

        Returns
        -------

        '''
    import src_julian.utils.Loss_and_metrics as metr

    myos = [metr.dice_coef_myo(g, p).numpy() for g, p in zip(gt, pred)]
    lvs = [metr.dice_coef_lv(g, p).numpy() for g, p in zip(gt, pred)]
    rvs = [metr.dice_coef_rv(g, p).numpy() for g, p in zip(gt, pred)]

    plt.plot(myos, label='myo')
    plt.plot(lvs, label='lv')
    plt.plot(rvs, label='rv')
    plt.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()


def plot_dice_per_slice_bar(gt, pred, save_path='reports/figures/error_per_labelandslice.png'):
    '''
    Calculate the dice per slice, create a stacked barchart for this plot
    This is necessary to figure out if we under- or over-segmented a stack of 2D slices
    we expect the input masks to have the following shape: [z,x,y,labels],
    Inverse indexing of the labels allow to work with [...,background, rv, myo, lv] or [..., rv, myo, lv]
    Parameters
    ----------
    gt : ground truth - 3D np.ndarray with 1,2,3 ... at the voxel-position of the label
    pred : prediction - 3D np.ndarray with 1,2,3 ... at the voxel-position of the label
    save_path : False = No savefig, str with full file and pathname otherwise

    Returns
    -------

    '''
    import src_julian.utils.Loss_and_metrics as metr

    # calculate the dice per label and per slice
    myos = [metr.dice_coef_myo(g, p).numpy() for g, p in zip(gt, pred)]
    # for each slice
    # if we have at least one voxel in this slice write 0 into the *_gt
    myos_gt = [int(not g.max()) for g in gt[..., -2]]

    lvs = [metr.dice_coef_lv(g, p).numpy() for g, p in zip(gt, pred)]
    lvs_gt = [int(not g.max()) for g in gt[..., -1]]
    rvs = [metr.dice_coef_rv(g, p).numpy() for g, p in zip(gt, pred)]
    rvs_gt = [int(not g.max()) for g in gt[..., -3]]

    # zip the scores together --> a list of tuples: [(lv,lv_gt,myo,myo_gt,rv,rv_gt)...(...)]
    scores = list(zip(lvs, lvs_gt, myos, myos_gt, rvs, rvs_gt))

    import matplotlib
    plt.rcParams.update({'font.size': 25})

    cmap = matplotlib.cm.get_cmap('RdYlBu')

    def custom_map(value):
        colors = []
        for v in value:
            # start with white
            color = (1, 1, 1, 0)
            # reverse 1 for slices with no label
            if v < 1:
                # use black for the gt bars
                if v == 0:
                    color = (0, 0, 0, 1)
                # use the colorbar for the dice
                else:
                    color = cmap(v)
            colors.append(color)

        return colors

    fig, ax = plt.subplots(figsize=(10, 10))

    bottom = 0
    # draw 6 bars (lv,lv_gt,myo,myo_gt,rv,rv_gt) for each slice position
    # each bar has the hight 1 but a different color (according to the custom color mapping)
    # increase the buttom edge of the bar by one for the next bar
    # by this we get a stacked bar plot
    for v in scores:
        rects = ax.bar([0, 1, 2, 3, 4, 5], 1, bottom=bottom, color=custom_map(v))
        bottom += 1
    fig.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap), ax=ax)
    plt.xticks([0, 1, 2, 3, 4, 5], ['LV', 'LV GT', 'MYO', 'MYO GT', 'RV', 'RV GT'], rotation=90)
    plt.ylabel('Slice position, \n 0 = lower part (1 slice = 1.5 mm)')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        return fig


def transform_to_binary_mask(mask_nda, mask_values=None):
    '''
    Transform the int labels to binary channel masks
    Parameters
    ----------
    mask_nda : 2D or 3D np.ndarray with one value per label,
    mask_values : list of int

    Returns np.ndarray with ndim +1 and one binary channel per label
    -------

    '''

    if mask_values is None:
        mask_values = [0, 1, 2, 3]
    mask = np.zeros((*mask_nda.shape, len(mask_values)), dtype=np.bool)
    for ix, mask_value in enumerate(mask_values):
        mask[..., ix] = mask_nda == mask_value
    return mask


def plot_value_histogram(nda, f_name='histogram.jpg', image=True, reports_path='reports/figures/4D_description'):
    '''
    plot 4 histograms for a numpy array of any shape
    1st plot with all values (100 buckets)
    2nd plot with .999 quantile (20 buckets)
    3rd plot with .75 quantile (20 buckets)
    4th plot with .5 quantile (bucketsize = Counter.most_common())
    y axis is percentual scaled
    x axis linear spaced - logspaced buckets are possible
    Parameters
    ----------
    nda :
    f_name :
    image :
    reports_path :

    Returns
    -------

    '''

    ensure_dir(reports_path)
    nda_img_flat = nda.flatten()
    plt.close('all')

    if not image:
        fig = plt.figure(figsize=[6, 6])
        ax1 = fig.add_subplot(111)
        nda_img_flat_filter = nda_img_flat[nda_img_flat > 0]
        c = Counter(nda_img_flat_filter)
        ax1.hist(nda_img_flat_filter, weights=np.ones(len(nda_img_flat_filter)) / len(nda_img_flat_filter), bins=3)
        ax1.set_title("Mask with  = {0:.2f} values".format(len(c)))
        ax1.yaxis.set_major_formatter(PercentFormatter(1))
    else:
        fig = plt.figure(figsize=[20, 6])
        ax1 = fig.add_subplot(141)
        ax1.hist(nda_img_flat, weights=np.ones(len(nda_img_flat)) / len(nda_img_flat), bins=100)
        ax1.set_title("1. quantile = {0:.2f}".format(nda_img_flat.max()))
        ax1.yaxis.set_major_formatter(PercentFormatter(1))

        ax2 = fig.add_subplot(142)
        ninenine_q = np.quantile(nda_img_flat, .999)
        nda_img_flat_nine = nda_img_flat[nda_img_flat <= ninenine_q]
        ax2.hist(nda_img_flat_nine, weights=np.ones(len(nda_img_flat_nine)) / len(nda_img_flat_nine), bins=20)
        ax2.set_title("0.999 quantile = {0:.2f}".format(ninenine_q))
        ax2.yaxis.set_major_formatter(PercentFormatter(1))

        ax3 = fig.add_subplot(143)
        seven_q = np.quantile(nda_img_flat, .75)
        nda_img_flat_seven = nda_img_flat[nda_img_flat <= seven_q]
        ax3.hist(nda_img_flat_seven, weights=np.ones(len(nda_img_flat_seven)) / len(nda_img_flat_seven), bins=20)
        ax3.set_title("0.75 quantile = {0:.2f}".format(seven_q))
        ax3.yaxis.set_major_formatter(PercentFormatter(1))

        ax4 = fig.add_subplot(144)
        mean_q = np.quantile(nda_img_flat, .5)
        nda_img_flat_mean = nda_img_flat[nda_img_flat <= mean_q]
        c = Counter(nda_img_flat_mean)
        ax4.hist(nda_img_flat_mean, weights=np.ones(len(nda_img_flat_mean)) / len(nda_img_flat_mean),
                 bins=len(c.most_common()))
        ax4.set_title("0.5 quantile = {0:.2f}".format(mean_q))
        ax4.set_xticks([key for key, _ in c.most_common()])
        ax4.yaxis.set_major_formatter(PercentFormatter(1))

    fig.suptitle(f_name, y=1.08)
    fig.tight_layout()
    plt.savefig(os.path.join(reports_path, f_name))
    plt.show()


def create_quiver_plot(flowfield_2d=None, ax=None, N=5, scale=0.3, linewidth=.5):
    """
    Function to create an flowfield for a deformable vector-field
    Needs a 2D flowfield, function can handle 2D or 3D vectors as channels
    :param N: take only every n vector
    :param flowfield_2d: numpy array with shape x, y, vectors
    :param ax: matplotlib ax object which should be used for plotting,
    create a new ax object if none is given
    :return: ax to plot or save
    """
    from matplotlib import cm
    import matplotlib

    if not ax:
        fig, ax = plt.subplots(figsize=(15, 15))

    # extract flowfield for x and y
    if flowfield_2d.shape[-1] == 3:  # originally a 3d flowfield
        Z_ = flowfield_2d[..., 0]
        X_ = flowfield_2d[..., 1]
        Y_ = flowfield_2d[..., 2]
    elif flowfield_2d.shape[-1] == 2:  # 2d flowfield
        X_ = flowfield_2d[..., 0]
        Y_ = flowfield_2d[..., 1]

    border = 0
    start_x = border
    start_y = border
    nz = Z_.shape[0] - border
    nx = X_.shape[0] - border  # define ticks in x
    ny = Y_.shape[1] - border  # define ticks in y

    # slice flowfield, take every N value
    Fz = Z_[::N, ::N]
    Fx = X_[::N, ::N]
    Fy = Y_[::N, ::N]
    nrows, ncols = Fx.shape

    # create a grid with the size nx/ny and ncols/npatients_lgetable
    z_ = np.linspace(start_x, nz, ncols)
    x_ = np.linspace(start_x, nx, ncols)
    y_ = np.linspace(start_y, ny, nrows)
    xi, yi = np.meshgrid(x_, y_, indexing='xy')
    zi, _ = np.meshgrid(z_, nx, indexing='xy')

    # working, use z as color
    # this way is not as clear as the test 3
    # norm = normalise_image(Fz)
    # colors = cm.copper(norm)
    # colors = colors.reshape(-1, 4)

    # test 3
    occurrence = Fz.flatten() / np.sum(Fz)
    norm = matplotlib.colors.Normalize()
    norm.autoscale(occurrence)
    cm = matplotlib.cm.copper
    sm = matplotlib.cm.ScalarMappable(cmap=cm, norm=norm)
    sm.set_array([])
    colors = cm(norm(Fz)).reshape(-1, 4)

    # plot
    ax.set_title('Flowfield')
    # ax.quiver(xi, -yi, Fx, Fy, units='xy', scale=.5, alpha=.5)
    ax.quiver(xi, -yi, Fx, Fy, color=colors, units='xy', angles='xy', scale=scale, linewidth=linewidth, minshaft=2,
              headwidth=6, headlength=7)
    # plt.colorbar(sm)
    return ax



class Console_and_file_logger():
    def __init__(self, logfile_name='Log', log_lvl=logging.INFO, path='./logs/'):
        """
        Create your own logger
        log debug messages into a logfile
        log info messages into the console
        log error messages into a dedicated *_error logfile
        :param logfile_name:
        :param log_dir:
        """

        # Define the general formatting schema
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        logger = logging.getLogger()

        # define a general logging level,
        # each handler has its own logging level
        # the console handler ist selectable by log_lvl
        logger.setLevel(logging.DEBUG)

        log_f = os.path.join(path, logfile_name + '.log')
        ensure_dir(os.path.dirname(os.path.abspath(log_f)))

        # delete previous handlers and overwrite with given setup
        logger.handlers = []
        if not logger.handlers:
            # Define debug logfile handler
            hdlr = logging.FileHandler(log_f)
            hdlr.setFormatter(formatter)
            hdlr.setLevel(logging.DEBUG)

            # Define info console handler
            hdlr_console = logging.StreamHandler()
            hdlr_console.setFormatter(formatter)
            hdlr_console.setLevel(log_lvl)

            # write error messages in a dedicated logfile
            log_f_error = os.path.join(path, logfile_name + '_errors.log')
            ensure_dir(os.path.dirname(os.path.abspath(log_f_error)))
            hdlr_error = logging.FileHandler(log_f_error)
            hdlr_error.setFormatter(formatter)
            hdlr_error.setLevel(logging.ERROR)

            # Add all handlers to our logger instance
            logger.addHandler(hdlr)
            logger.addHandler(hdlr_console)
            logger.addHandler(hdlr_error)

        cwd = os.getcwd()
        logging.info('{} {} {}'.format('--' * 10, 'Start', '--' * 10))
        logging.info('Working directory: {}.'.format(cwd))
        logging.info('Log file: {}'.format(log_f))
        logging.info('Log level for console: {}'.format(logging.getLevelName(log_lvl)))

def extract_segments(segments_str, segments=16):
    nda = np.zeros(segments)
    try:
        numbers = [int(i) - 1 for i in segments_str.split(',')]
    except Exception as e:
        # print(e)
        if type(segments_str) == type(1) and (int(segments_str) != 0):
            numbers = [int(segments_str) - 1]
        else:
            # print('cant find a sequence: {}'.format(segments_str))
            numbers = []
    nda[numbers] = 1
    return nda
