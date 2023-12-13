# define logging and working directory
from ProjectRoot import change_wd_to_project_root
change_wd_to_project_root()

# import packages
import logging
from logging import info as INFO
from src_julian.utils.skhelperfunctions import Console_and_file_logger
import numpy as np
import math
from scipy.interpolate import splprep, splev, interp1d
import cv2 as cv
import matplotlib.pyplot as plt

# set up logging
Console_and_file_logger('mvfviz/dmd_temp', logging.INFO)

###########################################################################################

def STRAIN_plot_composedflowfields(masks_rot, Radial_stack, Circumferential_stack, labels, colors):
    ax_labels_ED_relative = ['ED ' + '$\longrightarrow$' + ' MS',
                             'ED ' + '$\longrightarrow$' + ' ES',
                             'ED ' + '$\longrightarrow$' + ' PF',
                             'ED ' + '$\longrightarrow$' + ' MD',
                             'ED ' + '$\longrightarrow$' + ' ED']
    for i in range(Radial_stack.shape[0]):
        rr_array = np.zeros(5)
        cc_array = np.zeros(5)
        for t in range(Radial_stack.shape[1]):
            rr_array[t] = Radial_stack[i,t,...][masks_rot[t] == 1].mean()
            cc_array[t] = Circumferential_stack[i,t,...][masks_rot[t] == 1].mean()
        plt.plot(ax_labels_ED_relative[0:5], 100*rr_array, label='Err_' + labels[i], color=colors[i])
        plt.plot(ax_labels_ED_relative[0:5], 100*cc_array, label='Ecc_' + labels[i], linestyle='dashed', color=colors[i])
    plt.xlabel('Phase')
    plt.ylabel('Mean Strain in %')
    plt.title('Flowfield compositions and Strain Values (Morales et al.)')
    plt.legend()
    plt.grid()
    plt.show()

def get_phantom_masks(N_TIMESTEPS=5, N_ZREPEATS=64, xdim_img=128, ydim_img=128):
    from skimage.draw import disk
    import matplotlib.pyplot as plt
    ED_outter_radius = 30
    ED_inner_radius = 25
    ED_outter_center = (np.round(xdim_img/2),np.round(ydim_img/2))
    ED_inner_center = (np.round(xdim_img/2),np.round(ydim_img/2))

    ES_outter_radius = 20
    ES_inner_radius = 10
    ES_outter_center = (np.round(xdim_img/2),np.round(ydim_img/2))
    ES_inner_center = (np.round(xdim_img/2),np.round(ydim_img/2))

    ED_outter = np.zeros((xdim_img, ydim_img, 1))
    rr, cc = disk(ED_outter_center, ED_outter_radius)
    ED_outter[rr,cc,:] = 1
    ED_inner = np.zeros((xdim_img, ydim_img, 1))
    rr, cc = disk(ED_inner_center, ED_inner_radius)
    ED_inner[rr,cc,:] = 1
    ED_mask = ED_outter-ED_inner

    ES_outter = np.zeros((xdim_img, ydim_img, 1))
    rr, cc = disk(ES_outter_center, ES_outter_radius)
    ES_outter[rr, cc, :] = 1
    ES_inner = np.zeros((xdim_img, ydim_img, 1))
    rr, cc = disk(ES_inner_center, ES_inner_radius)
    ES_inner[rr, cc, :] = 1
    ES_mask = ES_outter - ES_inner

    # calculate the ED and ES output arrays
    mask_ED = np.tile(ED_mask, N_ZREPEATS)
    mask_ED = mask_ED[..., np.newaxis]
    mask_ED = np.einsum('xyzc -> zyxc', mask_ED)
    mask_ES = np.tile(ES_mask, N_ZREPEATS)
    mask_ES = mask_ES[..., np.newaxis]
    mask_ES = np.einsum('xyzc -> zyxc', mask_ES)

    # construct the output mask array
    masks_stack = np.ndarray((N_TIMESTEPS, N_ZREPEATS, xdim_img, ydim_img, 1), dtype='uint8')
    masks_stack[0] = mask_ED
    masks_stack[1] = mask_ED
    masks_stack[2] = mask_ES
    masks_stack[3] = mask_ED
    masks_stack[4] = mask_ED

    return masks_stack

def Ferdian2020_calculate_radial_strain(endo_batch, epi_batch, endo_ED, epi_ED, use_linear_strain=False, use_maskbased_L0=False):
    """
        epi_ED and endo_ED have to be of the form: zslices,1,2,npoints
        Calculate rr strain for a batch of image sequences
        flattened_coords => [batch_size, nr_frames, 2, 168]
    """
    # point 0 is epi, point 6 is endo, do this for all the 'radials'
    # endo_batch = coords_batch[:, :, :, ::7]
    # epi_batch = coords_batch[:, :, :, 6::7]

    # batch x time x 2 x 24 radials
    diff = (epi_batch - endo_batch) ** 2
    # print('diff', diff.shape)

    # batch x time x 24 sqrdiff
    summ = diff[:, :, 0, :] + diff[:, :, 1, :]  # x^2 + y^2
    # print('summ', summ.shape)

    if use_linear_strain:
        # use L instead of L^2
        summ = np.sqrt(summ)

    # grab the frame 0 (ED) for all data, and 24 RR strains
    # original calculating procedure by Ferdian
    summ_ed = summ[:, 0, :]

    # prepare ED status
    if use_maskbased_L0:
        # repeat all calculating steps for my ED reference now
        diff_ED = (epi_ED - endo_ED) ** 2
        my_summ_ED = diff_ED[:,:,0,:] + diff_ED[:,:,1,:]
        if use_linear_strain:
            my_summ_ED = np.sqrt(my_summ_ED)

        # overwrite the ED values
        summ_ed = my_summ_ED[:,0,:]



    # division through a certain column, without np.split
    # batch x time x 24 rr strains
    divv = summ / summ_ed[:, np.newaxis]  # this is the trick, add new axis

    if use_linear_strain:
        rr_strains = divv - 1
    else:
        rr_strains = (divv - 1) / 2

    rr_strains = np.mean(rr_strains, axis=2)

    # batch x time x strain
    rr_strains = np.expand_dims(rr_strains, axis=2)
    return rr_strains


def STRAIN_apply_ff_and_get_radial_lengths(contour_cube, flowfield, centroid_cube, Z_SLICES, N_TIMESTEPS, N_EVALUATION_RADIAL):
    # applies the flowfield over and over
    # trajectory principle
    # for every timestep the distance to the centroid will be calculated and written in l_cube

    target_cube = np.zeros_like(contour_cube)
    ff_l_cube = np.ndarray((len(Z_SLICES), N_TIMESTEPS, N_EVALUATION_RADIAL))

     # get flowfield extract
    flowfield_extract=flowfield[:,Z_SLICES,...]

    for idx, z in enumerate(Z_SLICES):
        for eval in range(N_EVALUATION_RADIAL):
            for t in range(N_TIMESTEPS):

                # we follow each point over time through the fields now
                # prepare inputs
                if t > 0:
                    input_coord_x = target_cube[idx, t - 1, eval, 0]
                    input_coord_y = target_cube[idx, t - 1, eval, 1]
                elif t == 0:  # ED case, first timestep
                    input_coord_x = contour_cube[idx, t, eval, 0]
                    input_coord_y = contour_cube[idx, t, eval, 1]

                # get idxs out of exact coordinate values
                input_idx_x = input_coord_x.round(decimals=0).astype(int)
                input_idx_y = input_coord_y.round(decimals=0).astype(int)
                # get flowfield values at those idxs
                ff_x = flowfield_extract[idx, t, input_idx_x, input_idx_y, 2]
                ff_y = flowfield_extract[idx, t, input_idx_x, input_idx_y, 1]
                # apply transform
                target_coord_x = input_coord_x + ff_x
                target_coord_y = input_coord_y + ff_y

                # calculate the radial distance to the new target point
                centroid_to_target = calc_distance_2D(centroid_cube[idx, t, 0], centroid_cube[idx, t, 1],
                                                              target_coord_x, target_coord_y)
                ff_l_cube[idx, t, eval] = centroid_to_target

                # plt.figure()
                # plt.imshow(vol_cube[t,z,:,:,0], cmap='gray')
                # plt.imshow(seg_cube[t+1,z,:,:,0])
                # plt.scatter(input_coord_x, input_coord_y, c='g')
                # plt.scatter(target_coord_x, target_coord_y, c='r')
                # plt.show()

                # save coordinates after flowfield application
                # the idx coordinates dont have to be saved because they can be
                # recalculated at any time out of the exact positions
                target_cube[idx, t, eval, 0] = target_coord_x
                target_cube[idx, t, eval, 1] = target_coord_y

    return ff_l_cube, target_cube


def get_l0(contour_cube, centroid_cube, Z_SLICES, N_TIMESTEPS, N_EVALUATION_RADIAL):
    # create centroid cube with the same shape as contour cube for subtraction
    centroid_cube = centroid_cube[..., 0:2]  # only the x and y components of the centroid cube
    centroid_cube = centroid_cube[..., np.newaxis]
    centroid_cube = np.repeat(centroid_cube, contour_cube.shape[2], axis=-1)
    centroid_cube = np.einsum('ztcv->ztvc', centroid_cube)

    # create an empty l0 array where i can write the distances
    l0 = np.zeros_like(contour_cube[..., -1])
    l0 = l0[..., np.newaxis]

    # pointwise distance calculation
    for idx, z in enumerate(Z_SLICES):
        for t in range(N_TIMESTEPS):
            for eval in range(N_EVALUATION_RADIAL):
                x0 = centroid_cube[idx, t, eval, 0]
                y0 = centroid_cube[idx, t, eval, 1]
                x1 = contour_cube[idx, t, eval, 0]
                y1 = contour_cube[idx, t, eval, 1]
                l0[idx, t, eval, 0] = calc_distance_2D(x0, y0, x1, y1)

    l0 = np.squeeze(l0)

    return l0

def STRAIN_apply_ff_and_calc_targetlengths_pairwise(flowfield, contour_cube_i, contour_cube_o, centroid_cube, Z_SLICES, N_TIMESTEPS, N_EVALUATION_RADIAL):
    # take the inner contour and round it, so that we have index values for every contour point which we can
    # access in the flowfield cube and take a motion vector from it
    # 22, 5, 1000, 2
    contour_cube_i_idx = np.stack((contour_cube_i[..., 0],
                                   contour_cube_i[..., 1]), axis=-1)
    contour_cube_o_idx = np.stack((contour_cube_o[..., 0],
                                   contour_cube_o[..., 1]), axis=-1)

    # round so that we have integer indexes and convert to int so that we have indexes
    contour_cube_i_idx = contour_cube_i_idx.round(decimals=0).astype(int)
    contour_cube_o_idx = contour_cube_o_idx.round(decimals=0).astype(int)

    # flowfield shape = 5,64,128,128,3
    flowfield_extract = np.copy(flowfield.Data[:, Z_SLICES, ...])  # 5,22,128,128,2
    flowfield_extract = np.einsum('tzyxc->ztxyc', flowfield_extract)  # 22,5,128,128,2

    # inits
    target_array_i = np.zeros_like(contour_cube_i_idx)
    target_array_o = np.zeros_like(contour_cube_o_idx)
    ff_l_cube = np.ndarray((len(Z_SLICES), N_TIMESTEPS, N_EVALUATION_RADIAL))

    for idx, z in enumerate(Z_SLICES):
        for t in range(N_TIMESTEPS):
            for eval in range(N_EVALUATION_RADIAL):
                # get the current indices we are after
                idx_xi = contour_cube_i_idx[idx, t, eval, 0]
                idx_yi = contour_cube_i_idx[idx, t, eval, 1]
                idx_xo = contour_cube_o_idx[idx, t, eval, 0]
                idx_yo = contour_cube_o_idx[idx, t, eval, 1]

                # CAVE: flowfield_extract is of shape ztxyc where c = zyx
                ffxi = flowfield_extract[idx, t, idx_xi, idx_yi, 2]
                ffyi = flowfield_extract[idx, t, idx_xi, idx_yi, 1]
                ffxo = flowfield_extract[idx, t, idx_xo, idx_yo, 2]
                ffyo = flowfield_extract[idx, t, idx_xo, idx_yo, 1]

                # add mask based spline and flowfield displacement coordinates
                # target[0] will be the following MS spline
                # target[1] will be the following ES spline
                # target[4] will be the following ED spline
                target_array_i[idx, t, eval, 0] = contour_cube_i[idx, t, eval, 0] + ffxi
                target_array_i[idx, t, eval, 1] = contour_cube_i[idx, t, eval, 1] + ffyi
                target_array_o[idx, t, eval, 0] = contour_cube_o[idx, t, eval, 0] + ffxo
                target_array_o[idx, t, eval, 1] = contour_cube_o[idx, t, eval, 1] + ffyo

                ff_l_cube[idx, t, eval] = get_displacement(target_array_o,
                                                           target_array_i,
                                                           np.roll(centroid_cube, 1, axis=1),
                                                           method='radial projected', # 'radial cumulative', 'radial projected'
                                                           idx=idx, t=t, eval=eval)
    return ff_l_cube, target_array_i, target_array_o

def plot_1x3distdiffstrain(distance_cube, diff_cube, strain_cube, patientname):
# plot mean distance, difference, strain curves for all slices in the cubes
# cubes have to have the same dimensions
# cubes have to be ordered: z,t,values where values i.e. = 1000
# when there are 15 z slices but only 4 curves show, this is because the masks upon which the curves are based
# appear multiple times within one stack
    PHASE_LABELS = ['ED', 'MS', 'ES', 'PF', 'MD']
    fig,ax=plt.subplots(1, 3, figsize=(20,5))
    for slice in range(distance_cube.shape[0]):
        ax[0].plot(PHASE_LABELS, distance_cube.mean(axis=-1)[slice,:])
        ax[0].set_ylabel('Myocardium Thickness in Voxels')
        ax[0].set_title('Distances')
        ax[1].plot(PHASE_LABELS, diff_cube.mean(axis=-1)[slice,:])
        ax[1].set_ylabel('Myocardium Thickness Change in Voxels')
        ax[1].set_title('Differences')
        ax[2].plot(PHASE_LABELS, 100*strain_cube.mean(axis=-1)[slice,:])
        ax[2].set_ylabel('Radial Strain in %')
        ax[2].set_title('Radial Strain')
    fig.suptitle(str(patientname))
    plt.show()

def plot_2x3distdiffstrain_AHA(mask_distance_cube, mask_diff_cube, mask_strain_cube,
                           ff_distance_cube, ff_diff_cube, ff_strain_cube,
                           N_MIDCAVITY_AHASEGMENTS, PHASE_LABELS, AHA_LABELS):
    plot_labels = ['distance', 'difference', 'strain']

    plt.style.use('ggplot')
    nrows, ncols = (2, 3)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 15))
    fig.suptitle('distances, differences and radial strains wrt AHA segments', fontsize=20)
    for aha in range(N_MIDCAVITY_AHASEGMENTS):
        ax[0, 0].plot(PHASE_LABELS[0:5], mask_distance_cube[aha], label=AHA_LABELS[aha])
        ax[1, 0].plot(PHASE_LABELS[0:5], ff_distance_cube[aha], label=AHA_LABELS[aha])
        ax[0, 1].plot(PHASE_LABELS[0:5], mask_diff_cube[aha], label=AHA_LABELS[aha])
        ax[1, 1].plot(PHASE_LABELS[0:5], ff_diff_cube[aha], label=AHA_LABELS[aha])
        ax[0, 2].plot(PHASE_LABELS[0:5], mask_strain_cube[aha], label=AHA_LABELS[aha])
        ax[1, 2].plot(PHASE_LABELS[0:5], ff_strain_cube[aha], label=AHA_LABELS[aha])

    ax[0, 0].set_title('mask-based distances')
    ax[0, 0].set_xlabel('phase')
    ax[0, 0].set_ylabel('distance')
    ax[0, 0].legend()
    ax[1, 0].set_title('flowfield-based distances')
    ax[1, 0].set_xlabel('phase')
    ax[1, 0].set_ylabel('distance')
    ax[1, 0].legend()

    ax[0, 1].set_title('mask-based difference')
    ax[0, 1].set_xlabel('phase')
    ax[0, 1].set_ylabel('difference')
    ax[0, 1].legend()
    ax[1, 1].set_title('flowfield-based difference')
    ax[1, 1].set_xlabel('phase')
    ax[1, 1].set_ylabel('difference')
    ax[1, 1].legend()

    ax[0, 2].set_title('mask-based strain')
    ax[0, 2].set_xlabel('phase')
    ax[0, 2].set_ylabel('strain')
    ax[0, 2].legend()
    ax[1, 2].set_title('flowfield-based strain')
    ax[1, 2].set_xlabel('phase')
    ax[1, 2].set_ylabel('strain')
    ax[1, 2].legend()
    plt.show()

def sort_radians_into_aha(evaluation_angles, distancecube, nevaluationradialaha):
    aha11_distance_array = []
    aha10_distance_array = []
    aha9_distance_array = []
    aha8_distance_array = []
    aha7_distance_array = []
    aha12_distance_array = []

    # get the distance values with respect to the radial angle with high resolution
    # we are travelling over all evaluated angles and check to which segment these values belong
    # then we assign these values into the corresponding array
    for idx, angle in enumerate(evaluation_angles):
        if -np.pi / 3 < angle <= 0: aha11_distance_array.append(distancecube[..., idx])  # AHA11
        if -2 * np.pi / 3 < angle <= -np.pi / 3: aha10_distance_array.append(distancecube[..., idx])  # AHA10
        if -3 * np.pi / 3 < angle <= -2 * np.pi / 3: aha9_distance_array.append(distancecube[..., idx])  # AHA9
        if 2 * np.pi / 3 < angle <= 3 * np.pi / 3: aha8_distance_array.append(distancecube[..., idx])  # AHA8
        if np.pi / 3 < angle <= 2 * np.pi / 3: aha7_distance_array.append(distancecube[..., idx])  # AHA7
        if 0 < angle <= np.pi / 3: aha12_distance_array.append(distancecube[..., idx])  # AHA12

    # transpose the lists and create nparrays
    aha11_distance_array = np.transpose(np.array(aha11_distance_array))
    aha10_distance_array = np.transpose(np.array(aha10_distance_array))
    aha9_distance_array = np.transpose(np.array(aha9_distance_array))
    aha8_distance_array = np.transpose(np.array(aha8_distance_array))
    aha7_distance_array = np.transpose(np.array(aha7_distance_array))
    aha12_distance_array = np.transpose(np.array(aha12_distance_array))

    # to refactor the multiple high-resolution lists, we take every Nth value to gain a np.ndarray with equal length
    # per aha segment, i.e. 10 evaluation points per segment
    aha11_distance_array = aha11_distance_array[...,
                                                np.round(np.linspace(0, len(aha11_distance_array) - 1,
                                                                     nevaluationradialaha)).astype(
                                                    int)]
    aha10_distance_array = aha10_distance_array[...,
                                                np.round(np.linspace(0, len(aha10_distance_array) - 1,
                                                                     nevaluationradialaha)).astype(
                                                    int)]
    aha9_distance_array = aha9_distance_array[...,
                                              np.round(np.linspace(0, len(aha9_distance_array) - 1,
                                                                   nevaluationradialaha)).astype(
                                                  int)]
    aha8_distance_array = aha8_distance_array[...,
                                              np.round(np.linspace(0, len(aha8_distance_array) - 1,
                                                                   nevaluationradialaha)).astype(
                                                  int)]
    aha7_distance_array = aha7_distance_array[...,
                                              np.round(np.linspace(0, len(aha7_distance_array) - 1,
                                                                   nevaluationradialaha)).astype(
                                                  int)]
    aha12_distance_array = aha12_distance_array[...,
                                                np.round(np.linspace(0, len(aha12_distance_array) - 1,
                                                                     nevaluationradialaha)).astype(
                                                    int)]

    # stack the slim arrays
    ahadistancecube = np.stack([aha11_distance_array, aha10_distance_array,
                                aha9_distance_array, aha8_distance_array,
                                aha7_distance_array, aha12_distance_array])

    return ahadistancecube


def get_displacement(target_array_o, target_array_i, centroid_cube, method, idx, t, eval):
    if method == 'radial cumulative':
        # radial and circumferential vector parts will be cumulated;
        # no partial evaluations will be made
        projected_length = calc_distance_2D(target_array_i[idx, t, eval, 0],
                             target_array_i[idx, t, eval, 1],
                             target_array_o[idx, t, eval, 0],
                             target_array_o[idx, t, eval, 1])

    elif method == 'radial projected':
        # the resulting vector will be projected on the radial vector, then only the
        # true radial length will be taken into account
        # we assume that the points do not "switch positions"
        distance_centroid_to_outter = calc_distance_2D(target_array_o[idx, t, eval, 0],
                                                       target_array_o[idx, t, eval, 1],
                                                       centroid_cube[idx, t, 0],
                                                       centroid_cube[idx, t, 1])
        distance_centroid_to_inner = calc_distance_2D(target_array_i[idx, t, eval, 0],
                                                      target_array_i[idx, t, eval, 1],
                                                      centroid_cube[idx, t, 0],
                                                      centroid_cube[idx, t, 1])
        if distance_centroid_to_outter > distance_centroid_to_inner:
            projection_base_point = (target_array_o[idx, t, eval, 0], target_array_o[idx, t, eval, 1])
        elif distance_centroid_to_outter < distance_centroid_to_inner:
            projection_base_point = (target_array_i[idx, t, eval, 0], target_array_i[idx, t, eval, 1])
        elif distance_centroid_to_outter == distance_centroid_to_inner:
            INFO('At radial projection two points have been the same')
            projection_base_point = (target_array_i[idx, t, eval, 0], target_array_i[idx, t, eval, 1])

        # projection
        centroid = (centroid_cube[idx, t, 0], centroid_cube[idx, t, 1])
        # the vector where we project on
        vector1 = (projection_base_point[0] - centroid[0], projection_base_point[1] - centroid[1])
        # the vector which we are projecting onto the other
        vector2 = (target_array_o[idx, t, eval, 0] - target_array_i[idx, t, eval, 0],
                   target_array_o[idx, t, eval, 1] - target_array_i[idx, t, eval, 1])

        # project vector2 on vector1 results in the length radial which we will take into account
        projected_length = np.dot(vector2, vector1) / np.linalg.norm(vector1)

    return projected_length

def get_wholeheartvolumeborders(segmentation_array, N_TIMESTEPS):
    '''
    takes the segmentation array "SLICEWISE" which states for every slice if we have segmentations ready
    returns the min and max Z index of the slices where we have them
    these two values i.e. can provide the patient specific heart volume Z indexes
    '''
    seg_minmax = np.ndarray((N_TIMESTEPS, 2))  # 5 timesteps, write local min and max segmented slice indices
    for phase in range(N_TIMESTEPS):
        seg_minmax[phase, 0] = np.where(segmentation_array[phase] == True)[0].min()
        seg_minmax[phase, 1] = np.where(segmentation_array[phase] == True)[0].max()

    global_min = seg_minmax[:, 0].max()
    global_max = seg_minmax[:, 1].min()

    return (int(global_min), int(global_max))

def get_volumeborders(lvmyo_idxs, border =1):
    '''
    takes the indices of the whole heart volume borders i.e. slice 5 and slice 50 define the cardiac volume within
    a total range of z-slices from 0 to 64

    then what will be returned will be the indices of the inner 35% which respectively define the midcavity wrt the aha publication:
    https://www.ahajournals.org/doi/epub/10.1161/hc0402.102975
    '''

    # inits
    # maybe we should remove the most apical and basal slices, they are very likely wrong
    if border>0:
        lvmyo_idxs = lvmyo_idxs[:-border]
    size_heart = len(lvmyo_idxs)


    perc_midcavity = .35
    perc_apex = .30
    end_apex = int(round(size_heart * perc_apex))
    end_midcavity = int(round(size_heart * (perc_apex+perc_midcavity)))
    # we will calculate mid-cavity borders, then from there derive the base and apex ranges
    apex_slices = lvmyo_idxs[0:end_apex]
    midcavity_slices = lvmyo_idxs[end_apex:end_midcavity]
    base_slices = lvmyo_idxs[end_midcavity:]

    # make sure that we have at least two slices per area,
    # otherwise a spatial gradient will fail
    if len(apex_slices) < 2:
        apex_slices = lvmyo_idxs[0:2]

    if len(midcavity_slices) < 2:
        midcavity_slices = lvmyo_idxs[end_apex:end_apex+2]

    if len(base_slices) < 2:
        base_slices = lvmyo_idxs[-2:]


    return base_slices, midcavity_slices, apex_slices

def calc_distance_2D(x0,y0,x1,y1): # simple function, I hope you are more comfortable
    return math.sqrt((x0-x1)**2+(y0-y1)**2)

def cart2pol(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return(theta, r)

def pol2cart(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return(x, y)

def deviaton_from_line(x,y, x0,y0, x1,y1):
    # https: // forum.image.sc / t / intersection - coordinates - between - line - and -spline - fitted - curve / 26658 / 3
    # if x,y are on line formed by (x1,y0) and (x1,y1)
    # this returns 0
    a = (y1-y0)
    b = (x1-x0)
    return abs(a*(x-x0)-b*(y-y0))/math.sqrt(a*a+b*b)

def get_idx_of_starting_point(centroid,x0,y0,x1,y1,x_spline,y_spline):
    # calculate the deviations of all spline points from the horizontal line points
    dev = np.array([ deviaton_from_line(x,y, x0,y0, x1,y1)  for x,y in zip(x_spline, y_spline) ])

    # find the index of the point with minimal deviation which lies right to the centroid
    dev0 = dev[0]
    for idx, num in enumerate(dev):
        if num < dev0 and x_spline[idx] > centroid[0]: # this states that the idx of the point should be right to the centroid
            dev0 = num
            index_min = idx

    return index_min

def get_centroid_dynamic(x_whole_i, y_whole_i, x_whole_o, y_whole_o):
    centroid_i = (sum(x_whole_i) / len(x_whole_i), sum(y_whole_i) / len(y_whole_i))
    centroid_o = (sum(x_whole_o) / len(x_whole_o), sum(y_whole_o) / len(y_whole_o))
    centroid = ((centroid_i[0]+centroid_o[0])/2, (centroid_i[1]+centroid_o[1])/2)
    return centroid

def get_overlay_limits(curr_contour_cube_o, curr_centroid):
    # get centroid coordinates
    x_centroid = curr_centroid[0]
    y_centroid = curr_centroid[1]

    # get max and min values of current contour
    xmin = curr_contour_cube_o[:, 0].min()
    xmax = curr_contour_cube_o[:, 0].max()
    ymin = curr_contour_cube_o[:, 1].min()
    ymax = curr_contour_cube_o[:, 1].max()

    # calculate the distances
    xspan = xmax - xmin
    yspan = ymax - ymin

    # compare the spans to decide which span will be relevant for the square depiction
    if xspan > yspan:
        relevant_span = xmax - x_centroid
    elif yspan > xspan:
        relevant_span = ymax - y_centroid

    xlim = [x_centroid - relevant_span, x_centroid + relevant_span]
    ylim = [y_centroid - relevant_span, y_centroid + relevant_span]

    return xlim, ylim

def STRAIN_get_contour_evaluation_points(mask_of_lv_myo, N_SPLINE_EVALUATION, N_EVALUATION_RADIAL, N_SMOOTHING):
    # get inner contour and get outter contour
    # CHAIN_APPROX_NONE gets absolutely all contour points
    # CHAIN_APPROX_SIMPLE: compresses horizontal, vertical, and diagonal segments and leaves only their end points.
    # For example, an up-right rectangular contour is encoded with 4 points.
    # RETR_METHOD = cv.RETR_CCOMP: retrieves all of the contours and organizes them into a two-level hierarchy.
    # At the top level, there are external boundaries of the components.
    # At the second level, there are boundaries of the holes.
    # If there is another contour inside a hole of a connected component, it is still put at the top level.
    # RETR_METHOD = cv.RETR_EXTERNAL: retrieves only the extreme outer contours
    RETR_METHOD = cv.RETR_CCOMP
    CH_APPROX_METHOD = cv.CHAIN_APPROX_NONE
    contour,_ = cv.findContours(mask_of_lv_myo, RETR_METHOD, CH_APPROX_METHOD)

    # catch holes in the masks and remove their contouring lists
    # remove their contouring lists means that the holes otherwise will receive own contours
    # we will simply remove them, so that only outter and inner contours will be remained in the list
    if len(contour) > 2:
        # more than two contours (inner and outer) have been detected.
        INFO('findcontours: more than two contours detected. intermediate contours will be deleted.')
        del contour[1:int(len(contour)-1)]

    # extract the xy coordinates from the found contours lists
    # get points out of the contours
    # pts_outter and pts_inner will contain a different amount of points
    # i.e. 90,1,2 where 90 points, and xy coordinates
    # contour[0] contains all coordinates of the outter contour
    # contour[1] contains all coordinates of the inner contour
    x_o, y_o = (np.squeeze(contour[0])[:, 0],
                np.squeeze(contour[0])[:, 1])
    x_i, y_i = (np.squeeze(contour[1])[:, 0],
                np.squeeze(contour[1])[:, 1])

    # initial spline calculation
    # fit spline through contour points
    # s = smoothing: Larger s means more smoothing while smaller values of s indicate less smoothing
    # (t,c,k) a tuple containing the vector of knots, the B-spline coefficients, and the degree of the spline.
    # u: An array of the values of the parameter.
    # per = periodic, means closed spline
    # s=10 for approx 100 spline findcontour points
    # Recommended values of s depend on the weights, w:
    # If the weights represent the inverse of the standard-deviation of y, then a good s value should be found in
    # the range (m-sqrt(2*m),m+sqrt(2*m)), where m is the number of data points in x, y, and w.
    # Recommended values of s depend on the weights, w. If the weights represent the inverse of the standard-deviation
    # of y, then a good s value should be found in the range (m-sqrt(2*m),m+sqrt(2*m)), where m is the
    # number of data points in x, y, and w.
    tck_i, _ = splprep([x_i, y_i], s=N_SMOOTHING, per=1)
    tck_o, _ = splprep([x_o, y_o], s=N_SMOOTHING, per=1)

    # initial spline evaluation
    x_whole_i, y_whole_i = splev(np.linspace(0, 1, N_SPLINE_EVALUATION), tck_i)
    x_whole_o, y_whole_o = splev(np.linspace(0, 1, N_SPLINE_EVALUATION), tck_o)

    # calculate center of contours dynamically
    centroid = get_centroid_dynamic(x_whole_i, y_whole_i, x_whole_o, y_whole_o)
    # calculate static centroid here
    # calculate center of contours statically: only calculate ED centroid and use that one for every phase
    # if t == 0: centroid = get_centroid_dynamic(x_whole_i, y_whole_i, x_whole_o, y_whole_o)

    # find the point on the spline which can be the starting point
    # the point shall be outtermost right with minimum distance to a horizontal line through the centroid
    # calculate the deviation for the points in the line and get the index of the closest point
    x0, y0 = centroid # first point of the horizontal line
    x1, y1 = (x0+1, y0) # second point of the horizontal line
    idx_i = get_idx_of_starting_point(centroid, x0, y0, x1, y1, x_whole_i, y_whole_i)
    idx_o = get_idx_of_starting_point(centroid, x0, y0, x1, y1, x_whole_o, y_whole_o)

    # rearrange the arrays for new spline calculation
    # the found starting point will be the first in the array now
    x_i_rearranged, y_i_rearranged = (np.roll(x_whole_i, -idx_i), np.roll(y_whole_i, -idx_i))
    x_o_rearranged, y_o_rearranged = (np.roll(x_whole_o, -idx_o), np.roll(y_whole_o, -idx_o))

    # check, in which direction the points wander if we start at the starting point
    # due to findcontours, sometimes the contours are drawn clockwise and sometimes anticlockwise
    # correct the ordering of the points, shall be ordered always anti-clockwise
    if y_i_rearranged[0] < y_i_rearranged[1]:
        x_i_rearranged = np.flip(x_i_rearranged)
        y_i_rearranged = np.flip(y_i_rearranged)
    if y_o_rearranged[0] < y_o_rearranged[1]:
        x_o_rearranged = np.flip(x_o_rearranged)
        y_o_rearranged = np.flip(y_o_rearranged)

    # coordinates transformation into parameter room
    # every point in the reordered array currently has a x and y coordinate
    # i want radius and angle theta instead
    # global values minus centroid equal local values
    x_centroid, y_centroid = centroid
    xi_local, yi_local = (x_i_rearranged-x_centroid,
                          y_i_rearranged-y_centroid)
    xo_local, yo_local = (x_o_rearranged-x_centroid,
                          y_o_rearranged-y_centroid)

    # the second angle is not needed, because the inner and outter splines work with the same angles
    theta, ri = cart2pol(xi_local, yi_local)
    _, ro = cart2pol(xo_local, yo_local)

    # plt.figure()
    # plt.imshow(slice_for_inner_contour)
    # plt.scatter(x_centroid, y_centroid)
    # plt.scatter(x_i_rearranged[0], y_i_rearranged[0]) # check the starting point of the rolled array
    # plt.scatter(x_i_rearranged[20], y_i_rearranged[20]) # check ordering / direction
    # plt.scatter(x_o_rearranged[0], y_o_rearranged[0])  # check the starting point of the rolled array
    # plt.scatter(x_o_rearranged[20], y_o_rearranged[20])  # check ordering / direction
    # plt.show()

    # plt.figure()
    # plt.plot(thetai, label='thetai')
    # plt.plot(thetao, label='thetao')
    # plt.legend()
    # plt.show()

    # we have to define the extreme values of the angles
    # necessary to not receive the error that the evaluated indices are out of the interpolation range
    # set the border values
    theta[0] = 0
    theta[theta.argmin()] = -np.pi
    theta[theta.argmax()] = np.pi
    theta[-1] = 0

    # interpolation curve of theta angle and radius
    fi, fo = (interp1d(theta, ri), interp1d(theta, ro))

    # define evaluation points
    # looks almost like a step curve
    # first half goes from 0 to minus pi
    # second half goes from pi to 0
    eval_angles_firsthalf = np.linspace(0, -np.pi, int(N_EVALUATION_RADIAL / 2))
    eval_angles_secondhalf = np.linspace(np.pi, 0, int(N_EVALUATION_RADIAL / 2))
    evaluation_angles = np.append(eval_angles_firsthalf, eval_angles_secondhalf)

    # evaluate the points
    r_eval_i, r_eval_o = (fi(evaluation_angles), fo(evaluation_angles))

    # conversion and image domain translation with respect to centroid
    x_eval_cart_i, y_eval_cart_i = pol2cart(r_eval_i, evaluation_angles)
    x_eval_cart_o, y_eval_cart_o = pol2cart(r_eval_o, evaluation_angles)

    # shift around centroid
    x_eval_cart_i, y_eval_cart_i = (x_eval_cart_i + x_centroid, y_eval_cart_i + y_centroid)
    x_eval_cart_o, y_eval_cart_o = (x_eval_cart_o + x_centroid, y_eval_cart_o + y_centroid)

    # INFO('Hi')

    # plt.figure(figsize=(10, 10))
    # plt.imshow(slice_for_inner_contour)
    # plt.plot(x_whole_i, y_whole_i, 'b-', label='inner wholespline')
    # plt.plot(x_whole_o, y_whole_o, 'b-', label='outer wholespline')
    # plt.plot(x_eval_cart_i, y_eval_cart_i, 'g-', label='inner evalspline')
    # plt.plot(x_eval_cart_o, y_eval_cart_o, 'g-', label='outer evalspline')
    # plt.scatter(centroid[0], centroid[1])
    # plt.legend()
    # plt.show()
    #
    # INFO('Hi')

    # import seaborn as sns
    # sns.set_theme()
    # plt.figure(figsize=(12, 8))
    # ax = plt.subplot(polar=True)
    # ax.scatter(-evaluation_angles[100], r_eval_i[100], c='r', marker='o', label='evaluated points inner')
    # plt.show()



    # plt.scatter(pol2cart(ri[50], thetai[50])[0] + centroid[0], pol2cart(ri[50], thetai[50])[1] + centroid[1])

    # ri = np.sqrt(np.add(np.square(xi_local), np.square(yi_local)))
    # ro = np.sqrt(np.add(np.square(xo_local), np.square(yo_local)))
    # thetai, thetao = (np.arctan2(yi_local, xi_local), np.arctan2(yo_local, xo_local))

    # theta unwrapped; set endpoints
    # because of this transpose, we also have to reverse the order of the radii
    # theta_unwrap_i = -np.unwrap(thetai)
    # theta_unwrap_i[0] = 0
    # theta_unwrap_i[-1] = 2*np.pi
    # theta_unwrap_o = -np.unwrap(thetao)
    # theta_unwrap_o[0] = 0
    # theta_unwrap_o[-1] = 2*np.pi
    # ri = np.flip(ri)
    # ro = np.flip(ro)

    # plt.figure
    # plt.imshow(seg_data[slice, ..., 0])
    # plot(pol2cart(ri, theta_unwrap_i)[0]+x_centroid, pol2cart(ri, theta_unwrap_i)[1]+y_centroid)
    # plt.show()

    # evaluation_angles = np.arange(0, 2*np.pi, 2*np.pi/N_EVALUATION_RADIAL)
    # evaluation_angles = [0, 1/4*np.pi, 2/4*np.pi, 3/4*np.pi, np.pi, 5/4*np.pi, 6/4*np.pi, 7/4*np.pi]

    # reorder the theta values. for some reason, they are messed up after evaluating
    # evaluation_angles = np.flip(evaluation_angles)
    # evaluation_angles = np.flip(evaluation_angles) # flip
    # evaluation_angles = evaluation_angles[:-1] # cut last
    # evaluation_angles = np.insert(evaluation_angles, 0, 0) # add first 0; # thetanew = [0, 7/4*np.pi, 6/4*np.pi, 5/4*np.pi, np.pi, 3/4*np.pi, 2/4*np.pi, 1/4*np.pi]

    # plt.figure()
    # plt.imshow(seg_data[slice, ..., 0])
    # # plt.scatter(x_whole_i[100], y_whole_i[100])
    # plt.scatter(x_eval_cart_i[100], y_eval_cart_i[100])
    # plt.show()

    # set up figure
    # import seaborn as sns
    # sns.set_theme()
    # plt.figure(figsize=(12, 8))
    # ax = plt.subplot(polar=True)
    # # ax.plot(cart2pol(xi_local, yi_local)[0],
    # #         cart2pol(xi_local, yi_local)[1], label='whole spline inner')  # whole spline points
    # # ax.plot(cart2pol(xo_local, yo_local)[0],
    # #         cart2pol(xo_local, yo_local)[1], label='whole spline outter')  # whole spline points
    # # plot evaluated points on splines for measurements
    # ax.scatter(evaluation_angles, r_eval_i, c='r', marker='o', label='evaluated points inner')
    # ax.scatter(evaluation_angles, r_eval_o, c='r', marker='o', label='evaluated points outter')
    # # ax.scatter(evaluation_angles[10], r_eval_i[10])
    # # xcart,ycart=pol2cart(r_eval_i[10],evaluation_angles[10])
    # # connect the evaluated points to visualize measurements
    # ax.plot([evaluation_angles, evaluation_angles], [r_eval_i, r_eval_o], linewidth=0.1, markersize=2)
    # ax.legend()
    # plt.show()

    # return spline curves and evaluated points
    return evaluation_angles, \
           centroid, \
           x_whole_i, y_whole_i,\
           x_whole_o, y_whole_o,\
           x_eval_cart_i, y_eval_cart_i, \
           x_eval_cart_o, y_eval_cart_o, \
           r_eval_i, r_eval_o

def plot_segmentation_grid(seg_lvcube, zslices, ntimesteps):
    '''
    takes seg_lvcube (segmentation cube of left ventricle) and zslices to be plotted
    plots 3 x lines and timesteps x columns
    plot grid of segmentations. manually tell which slices should be used
    '''
    PHASE_LABELS=['ED', 'MS', 'ES', 'PF', 'MD']

    fig, ax = plt.subplots(len(zslices), ntimesteps, sharex=True, sharey=True)
    for idx, z in enumerate(zslices):
        for t in range(ntimesteps):
            ax[idx, t].imshow(seg_lvcube[t, z, :, :, 0], cmap='gray')
            ax[idx, t].set_title(PHASE_LABELS[t], fontsize=20)
            ax[idx, t].set_xlim(50, 90)
            ax[idx, t].set_ylim(90, 40)
    plt.show()

    return ax

def plot_segmentation_grid_with_splines(seg_lvcube, vol_cube, zslices, ntimesteps):
    # plot five images of phase ground truth masks and the splines for discussion with the group

    PHASE_LABELS = ['ED', 'MS', 'ES', 'PF', 'MD']
    fig, ax = plt.subplots(1, ntimesteps, figsize=(10, 5), sharey=True)
    slice_absolute = zslices[0] + np.round(len(zslices)/2).astype('int')
    for t in range(ntimesteps):
        ax[t].imshow(seg_lvcube[t, slice_absolute, :, :, 0], cmap='gray')
        ax[t].imshow(vol_cube[t, slice_absolute, :, :, 0], cmap='gray')
        ax[t].set_title(PHASE_LABELS[t])
        ax[t].set_xlim(50, 90)
        ax[t].set_ylim(80, 40)
    plt.tight_layout(True)

    return ax

def plot_strain_per_slice_Radial_Circumferential_side_by_side(Radial_itk, Circumferential_itk, masks_rot_itk, N_TIMESTEPS, Z_SLICES, patientname):
    # plot strain per slice, radial and circumferential side-by-side
    import matplotlib.pylab as pl

    Radial_itk_strainperslice_array=np.zeros((N_TIMESTEPS, len(Z_SLICES)))
    Circumferential_itk_strainperslice_array = np.zeros((N_TIMESTEPS, len(Z_SLICES)))
    for t in range(N_TIMESTEPS):
        for slice in range(len(Z_SLICES)):
            Radial_itk_strainperslice_array[t,slice] = 100 * Radial_itk[t, ..., slice][masks_rot_itk[t, ..., slice] == 1].mean()
            Circumferential_itk_strainperslice_array[t, slice] = 100 * Circumferential_itk[t, ..., slice][masks_rot_itk[t, ..., slice] == 1].mean()
    ax_labels_ED_relative = ['ED ' + '$\longrightarrow$' + ' MS',
                             'ED ' + '$\longrightarrow$' + ' ES',
                             'ED ' + '$\longrightarrow$' + ' PF',
                             'ED ' + '$\longrightarrow$' + ' MD',
                             'ED ' + '$\longrightarrow$' + ' ED']

    colors = pl.cm.jet(np.linspace(0, 1, len(Z_SLICES)))
    fig,ax=plt.subplots(1, 2)

    for slice in range(len(Z_SLICES)):
        ax[0].plot(ax_labels_ED_relative, Radial_itk_strainperslice_array[...,slice], color=colors[slice], label='z_'+str(slice))
        ax[0].set_title('Radial Strain per Slice by Morales et al.')
        ax[0].set_xlabel('Phases')
        ax[0].set_ylabel('Radial Strain in %')
        ax[0].legend()
        ax[0].grid(True)

        ax[1].plot(ax_labels_ED_relative, Circumferential_itk_strainperslice_array[...,slice], color=colors[slice], label='z_'+str(slice))
        ax[1].set_title('Circumferential Strain per Slice by Morales et al.')
        ax[1].set_xlabel('Phases')
        ax[1].set_ylabel('Circumferential Strain in %')
        ax[1].legend()
        ax[1].grid(True)
    fig.suptitle(patientname)
    plt.show()

    return ax

def spoke_get_maskbased_arrays(seg_cube, Z_SLICES, Z_SPACING, N_TIMESTEPS, N_SPLINE_EVALUATION, N_EVALUATION_RADIAL, N_SMOOTHING):
    # takes segmentation cube
    # returns, based on wheelspoke design, distance_cube

    # inits
    contour_cube_i = np.ndarray((len(Z_SLICES), N_TIMESTEPS, N_EVALUATION_RADIAL, 2))
    contour_cube_o = np.ndarray((len(Z_SLICES), N_TIMESTEPS, N_EVALUATION_RADIAL, 2))
    distance_matrix = np.ndarray((N_TIMESTEPS, N_EVALUATION_RADIAL))
    centroid_cube = np.ndarray((len(Z_SLICES), N_TIMESTEPS, 3))
    radius_evaluation_matrix = np.ndarray((N_TIMESTEPS, N_EVALUATION_RADIAL, 2))
    distance_cube = np.ndarray((len(Z_SLICES), N_TIMESTEPS, N_EVALUATION_RADIAL))
    radius_cube = np.ndarray((len(Z_SLICES), N_TIMESTEPS, N_EVALUATION_RADIAL, 2))

    # for every Z slice, go through the cardiac phases and calculate the array cubes
    for idx_slice, slice in enumerate(Z_SLICES):
        for t in range(N_TIMESTEPS):
            # get the current segmentation data
            # the mask nii files only contain the LV myocardium area
            # from this with the contour detection algorithm, we can extract the outter and the inner contour
            mask_of_lv_myo = np.squeeze(seg_cube[t, slice, ...])

            # seg_data_inner_contour = seg_lvcube[t] # segmentation_filename_outter_contour = '_mask_' + str(t) + '_.nii', # get_stacked_masks(path_to_patient_folder, segmentation_filename_inner_contour)
            # get_stacked_masks(path_to_patient_folder, segmentation_filename_outter_contour)

            # get a binary array with 0 everywhere and 1 where idx is (i.e. bloodpool)
            # idx_bloodpool, idx_myo = (3, 2)
            # all_slices_bloodpool = (seg_data == idx_bloodpool).astype(np.uint8)
            # all_slices_bloodmyo = np.logical_or(seg_data==idx_bloodpool, seg_data==idx_myo).astype(np.uint8)

            # slice_bloodpool = seg_data_inner_contour[slice, :, :, 0]

            # slice_myo = slice_bloodmyo
            # CH_APPROX_METHOD = cv.CHAIN_APPROX_NONE
            # RETR_METHOD = cv.RETR_CCOMP #cv.RETR_EXTERNAL
            #
            # contour, hierarchy = cv.findContours(slice_myo,RETR_METHOD,CH_APPROX_METHOD)
            #
            # pts_inner, pts_outter = (contour[0], contour[1])
            # x_i, y_i = (pts_inner[:, 0, 0], pts_inner[:, 0, 1])
            # x_o, y_o = (pts_outter[:, 0, 0], pts_outter[:, 0, 1])
            #
            # plt.figure()
            # plt.imshow(slice_myo)
            # plt.plot(x_i, y_i, x_o, y_o)
            # plt.show()
            #
            # INFO('Hi')

            evaluation_angles, \
            centroid, \
            x_whole_i, y_whole_i, \
            x_whole_o, y_whole_o, \
            x_eval_cart_i, y_eval_cart_i, \
            x_eval_cart_o, y_eval_cart_o, \
            r_eval_i, r_eval_o = STRAIN_get_contour_evaluation_points(mask_of_lv_myo,
                                                                      N_SPLINE_EVALUATION,
                                                                      N_EVALUATION_RADIAL,
                                                                      N_SMOOTHING)
            # plt.figure()
            # plt.imshow(seg_data_inner_contour[slice,...,0])
            # # plt.imshow(seg_data_outter_contour[slice, ..., 0])
            # plot(x_whole_i, y_whole_i, 'b-')
            # plot(x_whole_o, y_whole_o, 'b-')
            # plot(x_eval_cart_i, y_eval_cart_i, 'g-')
            # plot(x_eval_cart_o, y_eval_cart_o, 'g-')
            # plt.show()

            # INFO('Hi')

            # plt.figure()
            # plt.imshow(seg.Data[xx, :, :, 0])
            # plt.scatter(x_eval_cart_i, y_eval_cart_i)
            # plt.scatter(x_eval_cart_o, y_eval_cart_o)
            # plt.show()
            # plt.figure()
            # plt.imshow(seg.Data[slice, :, :, 0])
            # plot(x_whole_i, y_whole_i)
            # plot(x_whole_o, y_whole_o)
            # plt.show()

            # saving the evaluated spline contours for later plotting
            # stack the evaluated contour points into a contour cube
            # contour_cube[...,0] will contain the xy coordinates
            contour_cube_i[idx_slice, t, ...] = np.stack((x_eval_cart_i, y_eval_cart_i), axis=-1)
            contour_cube_o[idx_slice, t, ...] = np.stack((x_eval_cart_o, y_eval_cart_o), axis=-1)

            # saving the calculated centroid values for this slice for later plotting
            # saving the x,y,z coordinates where z = slice
            centroid_cube[idx_slice, t, 0], centroid_cube[idx_slice, t, 1] = centroid
            centroid_cube[idx_slice, t, 2] = Z_SPACING * (Z_SLICES[0] + idx_slice)

            # here, we have the xy coordinates of the contours ready
            # we want to calculate the distances of the contour points
            # creating the distance matrix and the radius evaluation matrix
            # i.e. 5x1000 where we have 5 timesteps and 1000 radial evaluation lines
            # in the radius_evaluation_matrix, we will have two entries: inner and outter radius
            for idx_rad in range(N_EVALUATION_RADIAL):
                distance_matrix[t, idx_rad] = calc_distance_2D(x_eval_cart_i[idx_rad],
                                                               y_eval_cart_i[idx_rad],
                                                               x_eval_cart_o[idx_rad],
                                                               y_eval_cart_o[idx_rad])
                radius_evaluation_matrix[t, idx_rad, 0] = r_eval_i[idx_rad]  # inner radius
                radius_evaluation_matrix[t, idx_rad, 1] = r_eval_o[idx_rad]  # outter radius

        distance_cube[idx_slice] = distance_matrix
        radius_cube[idx_slice] = radius_evaluation_matrix

    return distance_cube, radius_cube, radius_evaluation_matrix, contour_cube_i, contour_cube_o, centroid_cube, evaluation_angles

def calculate_diff_and_strain_curves(distance_cube, opt_l0_ED_cube=None):
    # takes distance_cube_array of shape i.e. 15,5,1000 where 15=zslices, 5=timesteps, 1000=evaluation points
    # returns diff_cube and strain_cube of same shape
    # takes optional l0_ED_values if different from timestep=[0] of distance_cube
    # l0_ED_cube then contains l0 ED values in all the 5 timesteps

    # fetch l0_ED_array
    # standard way
    l0_ED_cube = distance_cube[:, 0, :]

    # kwargs optional way if there is an explicit input
    if opt_l0_ED_cube is not None:
        l0_ED_cube = opt_l0_ED_cube[:, 0, :]

    # calculate difference cube
    diff_cube = np.zeros_like(distance_cube)
    for z in range(distance_cube.shape[0]):
        for t in range(distance_cube.shape[1]):
            diff_cube[z, t, :] = distance_cube[z, t, :] - l0_ED_cube[z]  # wrt ED; L-L0

    # calculate strain cube
    strain_cube = np.zeros_like(distance_cube)
    for z in range(distance_cube.shape[0]):
        for t in range(distance_cube.shape[1]):
            strain_cube[z, t, :] = diff_cube[z, t, :] / l0_ED_cube[z]  # (L-L0)/L0

    return diff_cube, strain_cube

def plot_Ferdian_straincube_mean(rr_strain):
    # quick plot of Ferdian result
    plt.figure()
    plt.plot(['ED', 'MS', 'ES', 'PF', 'MD'], 100 * np.squeeze(rr_strain).mean(axis=0))
    plt.show()

def plot_segmentations_and_polarplots_sidebyside(seg_cube, vol_cube, contour_cube_i, contour_cube_o,
                                                 evaluation_angles, radius_cube, strain_cube, Z_SLICES, slice_relative, phase_left, phase_right):
    # plot segmentations and polar plots side-by-side
    # plot one slice only
    # patient data, segmentation mask, contours, thickness and max strain in polar plot
    # plot 2x2 figure where we have in the top row two phases, their patient data and the splines overlay
    # the second row shows two polar plots with the respective splines, where the zone of maximum strain is marked
    # configure the slice to be plotted in the function call as well as the phases left and right
    labels=['ED', 'MS', 'ES', 'PF', 'MD']
    slice_relative = slice_relative
    slice_absolute = Z_SLICES[0] + slice_relative
    # xlim = (30, 100)
    # ylim = (10, 80)
    # radius_slice_relative = int(len(Z_SLICES)/2)

    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot(221)
    ax1.imshow(seg_cube[phase_left, slice_absolute, :, :, 0], origin='upper', cmap='gray')
    ax1.plot(contour_cube_i[slice_relative, phase_left, :, 0],
             contour_cube_i[slice_relative, phase_left, :, 1], 'r-')
    ax1.plot(contour_cube_o[slice_relative, phase_left, :, 0],
             contour_cube_o[slice_relative, phase_left, :, 1], 'r-')
    ax1.imshow(vol_cube[phase_left, slice_absolute, :, :, 0], cmap='gray', alpha=0.5)
    # ax1.set_xlim(xlim), ax1.set_ylim(ylim)
    ax1.set_title(labels[phase_left], fontsize=25)
    ax2 = plt.subplot(222)
    ax2.imshow(seg_cube[phase_right, slice_absolute, :, :, 0], origin='upper', cmap='gray')
    ax2.plot(contour_cube_i[slice_relative, phase_right, :, 0],
             contour_cube_i[slice_relative, phase_right, :, 1], 'r-')
    ax2.plot(contour_cube_o[slice_relative, phase_right, :, 0],
             contour_cube_o[slice_relative, phase_right, :, 1], 'r-')
    ax2.imshow(vol_cube[phase_right, slice_absolute, :, :, 0], cmap='gray', alpha=0.5)
    # ax2.set_xlim(xlim), ax2.set_ylim(ylim)
    ax2.set_title(labels[phase_right], fontsize=25)
    plt.setp(ax2.get_yticklabels(), visible=False)
    ax3 = plt.subplot(223, projection='polar')
    ax3.plot([-evaluation_angles, -evaluation_angles],
             [radius_cube[slice_relative, phase_left, :, 0],
              radius_cube[slice_relative, phase_left, :, 1]],
             'go-',
             linewidth=2,
             markersize=1)
    ax4 = plt.subplot(224, projection='polar')
    ax4.plot([-evaluation_angles, -evaluation_angles],
             [radius_cube[slice_relative, phase_right, :, 0],
              radius_cube[slice_relative, phase_right, :, 1]],
             'go-',
             linewidth=2,
             markersize=1)
    # mark the biggest strain between the phases
    idxmaxdiff = strain_cube[slice_relative, phase_right].argmax()
    ax4.plot([-evaluation_angles[idxmaxdiff], -evaluation_angles[idxmaxdiff]],
             [radius_cube[slice_relative, phase_right, idxmaxdiff, 0],
              radius_cube[slice_relative, phase_right, idxmaxdiff, 1]],
             'ro-',
             linewidth=2,
             markersize=1)
    plt.show()

def plot_closeup_of_patientdata_splines_with_polarplot_overlayed(vol_cube, contour_cube_o,
                                                                 centroid_cube, evaluation_angles,
                                                                 radius_cube, Z_SLICES, slice_relative, phase):
    # this will generate one single figure with one plot where we overlay the polarplot for a closeup view
    # overlay polar plot and patient data
    # the first half will plot the patient data with focus on the spline area
    # the second half then overlays the polar plot
    # contour cube o is only needed to get the overlay limits - spline points will be plotted by radius_cube
    slice_relative = slice_relative
    slice_absolute = Z_SLICES[0] + slice_relative

    coords = [0.1, 0.1, 0.8, 0.8]
    fig = plt.figure(figsize=(10, 10))
    ax_image = fig.add_axes(coords, label='ax_image')
    ax_image.imshow(vol_cube[phase, slice_absolute, :, :, 0], alpha=0.5, origin='upper', cmap='gray')
    # ax_image.scatter(centroid_cube[slice_relative, phase, 0], centroid_cube[slice_relative,phase, 1], 'k')
    # ax_image.plot(contour_cube_i[slice_relative, phase, :, 0], contour_cube_i[slice_relative, phase, :, 1], 'k')
    # ax_image.plot(contour_cube_o[slice_relative, phase, :, 0], contour_cube_o[slice_relative, phase, :, 1], 'k')
    ax_image.axis('on')  # don't show the axes ticks/lines/etc. associated with the image
    ax_image.autoscale(False)
    curr_contour_cube_o = contour_cube_o[slice_relative, phase, ...]
    curr_centroid = centroid_cube[slice_relative, phase, ...]
    xlim, ylim = get_overlay_limits(curr_contour_cube_o, curr_centroid)
    ax_image.set_xlim(xlim[0], xlim[1])
    ax_image.set_ylim(ylim[1], ylim[0])
    ax_image.set_aspect('equal')
    ax_image.grid(False)

    ####################################################

    # plotting the polar plot on top now
    n_rticks = 4  # one less will be displayed
    # fig = plt.figure(figsize=(10,10))
    polar_coords = [0.1, 0.1, 0.8, 0.8]
    ax_polar = fig.add_axes(coords, projection='polar', label='ax_polar')
    ax_polar.set_ylim([0, radius_cube[slice_relative, phase, ...].max()])
    ax_polar.patch.set_alpha(0)  # necessary to see the patient data
    ax_polar.plot(-evaluation_angles, radius_cube[slice_relative, phase, :, 0], color='r', linewidth=2)  # inner radius
    ax_polar.plot(-evaluation_angles, radius_cube[slice_relative, phase, :, 1], color='r', linewidth=2)  # outer radius
    ax_polar.grid(True)  # necessary to see the polar grid radial lines
    ax_polar.set_rticks(np.linspace(0, radius_cube[slice_relative, phase, ...].max(), n_rticks))
    ax_polar.set_thetagrids([0, 60, 120, 180, 240, 300])
    ax_polar.set_yticklabels([])
    ax_polar.spines['polar'].set_visible(False)
    plt.rc('grid', color='k', linewidth=1.5, linestyle='-')
    plt.show()

def plot_two_phases_as_polarplots_and_mark_biggest_differences(evaluation_angles, radius_cube,
                                                               distance_cube, diff_cube, slice, phases):
    # set up figure
    phase = ['ED', 'MS', 'ES', 'PF', 'MD']
    fig, ax = plt.subplots(1, 2, figsize=(12, 8), subplot_kw={'projection': 'polar'})

    for idx, t in enumerate(phases):
        # plot evaluated points on splines for measurements
        # ax[t].scatter(evaluation_angles, radius_cube[slice,t,:,0], c='r', marker='o', s=10, label='evaluated points inner')
        # ax[t].scatter(evaluation_angles, radius_cube[slice,t,:,1], c='r', marker='o', s=10, label='evaluated points outter')

        # connect the evaluated points to visualize measurements
        ax[idx].plot([-evaluation_angles, -evaluation_angles],
                   [radius_cube[slice, t, :, 0], radius_cube[slice, t, :, 1]],
                   'go-',
                   linewidth=2,
                   markersize=1)

        # mark the biggest distances in the phase
        idxmaxdist = distance_cube[slice, t, :].argmax()
        ax[idx].plot([-evaluation_angles[idxmaxdist], -evaluation_angles[idxmaxdist]],
                   [radius_cube[slice, t, idxmaxdist, 0], radius_cube[slice, t, idxmaxdist, 1]],
                   'ro-',
                   linewidth=2,
                   markersize=1)

        # mark the biggest difference between the phases
        # diff_cube = distance_cube[slice, 2, :]-distance_cube[slice, 0, :]
        idxmaxdiff = diff_cube[slice, t, :].argmax()
        ax[idx].plot([-evaluation_angles[idxmaxdiff], -evaluation_angles[idxmaxdiff]],
                   [radius_cube[slice, t, idxmaxdiff, 0], radius_cube[slice, t, idxmaxdiff, 1]],
                   'bo-',
                   linewidth=2,
                   markersize=1)

        # title
        ax[idx].set_title(phase[t])
    plt.show()

def plot_3x5grid_maskbasedcontours_and_targetcontours(vol_cube, input_array_i, input_array_o,
                                                      target_array_i, target_array_o, Z_SLICES, N_TIMESTEPS, N=10):
    # plot three lines where we have five images for each line showing input and target splines
    # plot 3x5 figure where 3 lines plot different zslices of the stack and 5 are the timesteps
    # in each subplot, in green will be plotted the maskbased contours and in red will be plotted the moved masks

    PHASE_LABELS=['ED', 'MS', 'ES', 'PF', 'MD', 'ED']

    # calculate the slices
    # most basal, midcavity, apical of the stack
    slices=[len(Z_SLICES)-1, int(np.round(len(Z_SLICES)/2)), 0]
    fig, ax = plt.subplots(nrows=len(slices), ncols=5, figsize=(15, 5), sharey=True, sharex=True)
    z_start = Z_SLICES[0]

    for idx, z in enumerate(slices):
        for phase in range(N_TIMESTEPS):
            ax[idx,phase].imshow(vol_cube[phase, z+z_start, ...], cmap='gray')

            ax[idx,phase].scatter(input_array_i[z,phase,:,0], input_array_i[z,phase,:,1], s=0.8, c='g', label='input spline')
            ax[idx,phase].scatter(input_array_o[z,phase,:,0], input_array_o[z,phase,:,1], s=0.8,  c='g', label='input spline')
            ax[idx,phase].plot([input_array_i[z,phase,::N,0], input_array_o[z,phase,::N,0]],
                               [input_array_i[z,phase,::N,1], input_array_o[z,phase,::N,1]], c='g')

            ax[idx,phase].scatter(target_array_i[z,phase,:,0], target_array_i[z,phase,:,1], s=0.8, c='r', label='target spline')
            ax[idx,phase].scatter(target_array_o[z,phase,:,0], target_array_o[z,phase,:,1], s=0.8, c='r', label='target spline')
            ax[idx,phase].plot([target_array_i[z,phase,::N,0], target_array_o[z,phase,::N,0]],
                               [target_array_i[z,phase,::N,1], target_array_o[z,phase,::N,1]], c='r')

            ax[0,phase].set_title(str(PHASE_LABELS[phase] + '$\longrightarrow$' + str(PHASE_LABELS[phase+1])), fontsize=20)
            ax[idx,phase].grid(False)
            ax[idx,phase].set_xlim(30, 90)
            ax[idx,phase].set_ylim(80, 20)
    plt.show()

def plot_1x3_distdiffstrain_per_slice(distance_cube, difference_cube, strain_cube, evaluation_angles, Z_SLICES, phase=2, NTHCURVE=1):
    # mean strain curves over angle to identify areas which move too severe/too low
    # figure 1x3 with distances, differences, strain per slice
    # all curves of all slices will be plotted, but may overlay because we have the same mask multiple times
    # x-axis is polar angle in degrees

    fig, ax = plt.subplots(1, 3)

    for z in range(len(Z_SLICES[::NTHCURVE])):
        ax[0].plot(np.linspace(0, 360, len(evaluation_angles)),
                   distance_cube[z, phase, :],  # inspect ED slicewise
                   label="slice" + str(int(np.linspace(0, len(Z_SLICES), len(Z_SLICES))[z])))
        ax[0].set_xlabel('Polar Angle in Degrees')
        ax[0].set_ylabel('Distance')
        ax[0].set_title('Distances in the Slice')
        ax[0].legend()
        ax[0].grid('True')

        ax[1].plot(np.linspace(0, 360, len(evaluation_angles)),
                   difference_cube[z, phase, :],  # inspect ED slicewise
                   label="slice" + str(int(np.linspace(0, len(Z_SLICES), len(Z_SLICES))[z])))
        ax[1].set_xlabel('Polar Angle in Degrees')
        ax[1].set_ylabel('Difference')
        ax[1].set_title('Differences in the Slice')
        ax[1].legend()
        ax[1].grid('True')

        ax[2].plot(np.linspace(0, 360, len(evaluation_angles)),
                   strain_cube[z, phase, :],  # inspect ES slicewise
                   label="slice" + str(int(np.linspace(0, len(Z_SLICES), len(Z_SLICES))[z])))
        ax[2].set_xlabel('Polar Angle in Degrees')
        ax[2].set_ylabel('Radial Strain')
        ax[2].set_title('Radial Strain per Slice')
        ax[2].legend()
        ax[2].grid('True')

def plot_1x3_distdiffstrain_all_slices_per_phase(distance_cube, difference_cube, strain_cube, Z_SLICES):
    '''
    1x3 plot with distance, difference, strain per slice and all phases
    x-axis is Phases
    lines are rainbow colored so that one can visually see directly if basal or apical areas influence
    mean values too severe
    '''
    fig, ax = plt.subplots(1,3,figsize=(20,8))
    phaselabels=['ED', 'MS', 'ES', 'PF', 'MD']
    colormap = plt.cm.gist_ncar  # nipy_spectral, Set1,Paired

    for idx, z in enumerate(Z_SLICES):
        ax[0].plot(phaselabels, distance_cube[idx].mean(axis=-1), label='z='+str(idx))
        ax[0].set_xlabel('Phase')
        ax[0].set_ylabel('Mean distance per slice in Voxels')
        ax[0].set_title('Flowfield based distances per slice')
        ax[0].legend()
        # color the lines in rainbow colors to differentiate better
        colors = [colormap(i) for i in np.linspace(0, 1, len(ax[0].lines))]
        for i, j in enumerate(ax[0].lines):
            j.set_color(colors[i])

        ax[1].plot(phaselabels, difference_cube[idx].mean(axis=-1), label='z=' + str(idx))
        ax[1].set_xlabel('Phase')
        ax[1].set_ylabel('Mean difference per slice in Voxels')
        ax[1].set_title('Flowfield based difference per slice')
        ax[1].legend()
        # color the lines in rainbow colors to differentiate better
        colors = [colormap(i) for i in np.linspace(0, 1, len(ax[1].lines))]
        for i, j in enumerate(ax[1].lines):
            j.set_color(colors[i])

        ax[2].plot(phaselabels, 100*strain_cube[idx].mean(axis=-1), label='z=' + str(idx))
        ax[2].set_xlabel('Phase')
        ax[2].set_ylabel('Mean radial strain per slice in %')
        ax[2].set_title('Flowfield based radial strain per slice')
        ax[2].legend()
        # color the lines in rainbow colors to differentiate better
        colors = [colormap(i) for i in np.linspace(0, 1, len(ax[2].lines))]
        for i, j in enumerate(ax[2].lines):
            j.set_color(colors[i])
    plt.show()

def plot_5xZSLICES_alltargetcontours_for_all_slices(vol_cube, distance_cube, movedpoints_i, movedpoints_o, Z_SLICES, N_TIMESTEPS, NTHPOINT=20):
    '''
    plot 5xZSLICES grid where we can see all flowfield based contours
    headings of the small images will be the mean distance of the contour, so that thick contours get high values
    plot all target contours for all slices
    '''
    fig, ax = plt.subplots(N_TIMESTEPS, len(Z_SLICES), figsize=(10, 10), sharex=True, sharey=True)
    for phase in range(N_TIMESTEPS):
        for idx, z in enumerate(np.arange(Z_SLICES[0],Z_SLICES[-1]+1,1)):
            ax[phase, idx].imshow(vol_cube[phase, z, ...], cmap='gray')
            ax[phase, idx].plot([movedpoints_i[idx,phase,::NTHPOINT, 0], movedpoints_o[idx,phase,::NTHPOINT, 0]],
                                [movedpoints_i[idx,phase,::NTHPOINT, 1], movedpoints_o[idx,phase,::NTHPOINT, 1]],
                                c='r')
            ax[phase, idx].set_title(str(np.round(distance_cube[idx, phase,:].mean(axis=0), decimals=1)), fontsize=10)
            ax[phase, idx].set_xlim(30, 90)
            ax[phase, idx].set_ylim(80, 20)
            ax[phase, idx].grid(False)
    plt.show()


def get_borders_from_list(RVIP_list):
    for z in range(len(RVIP_list)):
        if np.sum(RVIP_list[z] != None): # if there are None entries
            z_start = z
            break
    z_end = z_start + sum(x is not None for x in RVIP_list)-1
    return [z_start, z_end]


def get_wholeheartvolumeborders_by_RVIP(seg_array, N_TIMESTEPS):
    '''
    might be the more exact method than by only one segmentation label, especially when we use base and apex data
    takes seg_array which has to contain RV and LV myo mask such that RVIP can be found
    whenever only one or no RVIP is identified in a slice, None will be returned
    returns two z values which state the smallest area of RVIP identified
    '''

    from src_julian.utils.skhelperfunctions import get_ip_from_mask_3d

    # get whole heart volume borders from RVIP detection
    heartborders = np.ndarray((N_TIMESTEPS, 2))

    for t in range(N_TIMESTEPS):
        mask3d = seg_array[t]
        RVIPup_glob, _ = get_ip_from_mask_3d(mask3d, debug=False, keepdim=True)
        heartborders[t, 0] = get_borders_from_list(RVIPup_glob)[0]
        heartborders[t, 1] = get_borders_from_list(RVIPup_glob)[1]

    # get the smallest array area where both RVIP are identified in each slice
    wholeheartvolumeborders = [np.max(heartborders[:, 0]), np.min(heartborders[:, 1])]

    return wholeheartvolumeborders

