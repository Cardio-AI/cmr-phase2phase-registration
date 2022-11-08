# define logging and working directory
from ProjectRoot import change_wd_to_project_root
change_wd_to_project_root()

# import helper functions
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot
import SimpleITK as sitk
import numpy as np
import logging
from logging import info as INFO
from src_julian.utils.skhelperfunctions import Console_and_file_logger
from src_julian.utils.myhelperfunctions import *
from src_julian.utils.myclasses import *
from src_julian.utils.mystrainfunctions import *

# set up logging
Console_and_file_logger('mvfviz/dmd_temp', logging.INFO)

#######################################################################################################
# fixed values
N_TIMESTEPS = 5
N_SPLINE_EVALUATION = 1000
N_EVALUATION_RADIAL = 1000
N_SMOOTHING = 10
Z_SPACING = 1
N_EVALUATION_RADIAL_AHA = 10
PHASE_LABELS = ['ED', 'MS', 'ES', 'PF', 'MD', 'ED', 'MS', 'ES', 'PF', 'MD']
AHA_LABELS = ('AHA11', 'AHA10', 'AHA9', 'AHA8', 'AHA7', 'AHA12')
N_MIDCAVITY_AHASEGMENTS = 6

# create segmentation phantom
phantom = get_phantom_masks(N_TIMESTEPS=N_TIMESTEPS)

# t = 2
# z = 32
# plt.figure()
# plt.imshow(phantom[t,z,:,:,0], cmap='gray')
# plt.show()

# INFO('Hi')


# get patient specific whole-heart-volume-range which means:
# the slices where we do have segmentations available
# as a basis to calculate the inner 35% midcavity slices
# in order to see the range from where we will pick the midcavity slices, we have to check where we do have segmentations
# i.e. we have 5 timesteps where we do have segmentations in the slices 5...50, 2...40, ...
# then we will choose 2...50 as the final whole-heart-volume-range from where we will take the midcavity slices
data_masks = volume(phantom, '4Dt', Z_SPACING)
seg_available = data_masks.get_segmentationarray(resolution='slicewise')
# the wholeheartvolumeborders numbers are idx numbers and refer equally to the mitk slices
wholeheartvolumeborders = get_wholeheartvolumeborders(seg_available, N_TIMESTEPS)
# from the whole heart volume indices, calculate the inner 35% aha midcavity indices
z_start, z_end = get_midcavityvolumeborders(wholeheartvolumeborders)
# we have to extend the z_end value by 1 so that the last index will be included in the Z_SLICES array
Z_SLICES = np.arange(z_start, z_end+1, 1)

seg_cube = phantom

# init
distance_matrix = np.ndarray((N_TIMESTEPS, N_EVALUATION_RADIAL))
distance_cube = np.ndarray((len(Z_SLICES), N_TIMESTEPS, N_EVALUATION_RADIAL))
radius_evaluation_matrix = np.ndarray((N_TIMESTEPS, N_EVALUATION_RADIAL, 2))
radius_cube = np.ndarray((len(Z_SLICES), N_TIMESTEPS, N_EVALUATION_RADIAL, 2))
contour_cube_i = np.ndarray((len(Z_SLICES), N_TIMESTEPS, N_EVALUATION_RADIAL, 2))
contour_cube_o = np.ndarray((len(Z_SLICES), N_TIMESTEPS, N_EVALUATION_RADIAL, 2))
centroid_cube = np.ndarray((len(Z_SLICES), N_TIMESTEPS, 3))

# for every Z slice, go through the cardiac phases and calculate the array cubes
for idx_slice, slice in enumerate(Z_SLICES):
    for t in range(N_TIMESTEPS):
        # get the current segmentation data
        seg_data_outter_contour = seg_cube[t] # get_stacked_masks(path_to_patient_folder, segmentation_filename_outter_contour)

        slice_myo = seg_data_outter_contour[slice, :, :, 0]


        evaluation_angles,\
        centroid, \
        x_whole_i, y_whole_i,\
        x_whole_o, y_whole_o,\
        x_eval_cart_i, y_eval_cart_i,\
        x_eval_cart_o, y_eval_cart_o, \
        r_eval_i, r_eval_o = STRAIN_get_contour_evaluation_points(slice_myo,
                                                                  N_SPLINE_EVALUATION,
                                                                  N_EVALUATION_RADIAL,
                                                                  N_SMOOTHING,
                                                                  t)


        # saving the evaluated spline contours for later plotting
        contour_cube_i[idx_slice, t, ...] = np.stack((x_eval_cart_i, y_eval_cart_i), axis=1)
        contour_cube_o[idx_slice, t, ...] = np.stack((x_eval_cart_o, y_eval_cart_o), axis=1)

        # saving the calculated centroid values for this slice for later plotting
        # saving the x,y,z coordinates where z = slice
        centroid_cube[idx_slice, t, 0], centroid_cube[idx_slice, t, 1] = centroid
        centroid_cube[idx_slice, t, 2] = Z_SPACING * (Z_SLICES[0] + idx_slice)

        # creating the distance matrix and the radius evaluation matrix
        # i.e. 5x1000 where 5 timesteps and 1000 radial evaluation lines
        for idx_rad in range(N_EVALUATION_RADIAL):
            distance_matrix[t, idx_rad] = calc_distance_2D(x_eval_cart_i[idx_rad],
                                                           y_eval_cart_i[idx_rad],
                                                           x_eval_cart_o[idx_rad],
                                                           y_eval_cart_o[idx_rad])
            radius_evaluation_matrix[t, idx_rad, 0] = r_eval_i[idx_rad] # inner radius
            radius_evaluation_matrix[t, idx_rad, 1] = r_eval_o[idx_rad] # outter radius

    distance_cube[idx_slice] = distance_matrix
    radius_cube[idx_slice] = radius_evaluation_matrix

#######################################################################################################

# Ferdian tests with phantom data
FerdianStrain = Ferdian2020_calculate_radial_strain(endo_batch=np.einsum('ztvc->ztcv',contour_cube_i),
                                                    epi_batch=np.einsum('ztvc->ztcv',contour_cube_o),
                                                    use_linear_strain=True)



# calculate distance differences cube
diff_cube = np.ndarray((len(Z_SLICES), N_TIMESTEPS, N_EVALUATION_RADIAL))
for z in range(len(Z_SLICES)):
    for t in range(N_TIMESTEPS):
        diff_cube[z,t,:] = distance_cube[z, t, :] - distance_cube[z, 0, :] # wrt ED

# calculate strain cube
strain_cube = np.ndarray((len(Z_SLICES), N_TIMESTEPS, N_EVALUATION_RADIAL))
for z in range(len(Z_SLICES)):
    for t in range(N_TIMESTEPS):
        strain_cube[z, t, :] = diff_cube[z, t, :] / distance_cube[z, 0, :]

# inspections
# calculate the positions of the min max values
ind_dist_min = np.unravel_index(np.argmin(distance_cube, axis=None), distance_cube.shape)
ind_dist_max = np.unravel_index(np.argmax(distance_cube, axis=None), distance_cube.shape)
ind_diff_min = np.unravel_index(np.argmin(diff_cube, axis=None), diff_cube.shape)
ind_diff_max = np.unravel_index(np.argmax(diff_cube, axis=None), diff_cube.shape)
ind_strain_min = np.unravel_index(np.argmin(strain_cube, axis=None), strain_cube.shape)
ind_strain_max = np.unravel_index(np.argmax(strain_cube, axis=None), strain_cube.shape)

INFO('distance_cube.min(): ' + str(round(distance_cube.min(), 2)) + ' at ' +  str(ind_dist_min))
INFO('distance_cube.max(): ' + str(round(distance_cube.max(), 2)) + ' at ' +  str(ind_dist_max))
INFO('diff_cube.min(): ' + str(round(diff_cube.min(), 2)) + ' at ' +  str(ind_diff_min))
INFO('diff_cube.max(): ' + str(round(diff_cube.max(), 2)) + ' at ' +  str(ind_diff_max))
INFO('strain_cube.min(): ' + str(round(strain_cube.min(), 2)) + ' at ' +  str(ind_strain_min))
INFO('strain_cube.max(): ' + str(round(strain_cube.max(), 2)) + ' at ' +  str(ind_strain_max))

########################################automatic reporting##############################################################

# create sample flowfield
sample_ff = np.ndarray((5,64,128,128,3))
sample_ff[...,0] = 0 #z
sample_ff[...,1] = 10 #y
sample_ff[...,2] = 10 #x
flowfield = mvf(data=sample_ff, format='4Dt', zspacing=Z_SPACING)

contour_cube_i_idx = np.stack((contour_cube_i[..., 0],
                                   contour_cube_i[..., 1]), axis=-1)
contour_cube_o_idx = np.stack((contour_cube_o[..., 0],
                               contour_cube_o[..., 1]), axis=-1)

# round so that we have integer indexes
# convert to int so that we have indexes
contour_cube_i_idx = contour_cube_i_idx.round(decimals=0).astype(int)
contour_cube_o_idx = contour_cube_o_idx.round(decimals=0).astype(int)

# flowfield shape = 5,64,128,128,3
flowfield_extract = np.copy(flowfield.Data[:,Z_SLICES,...]) # 5,22,128,128,2
flowfield_extract = np.einsum('tzyxc->ztxyc' , flowfield_extract) # 22,5,128,128,2

# move contours by displacement field
# add the flowfield to the contour points now
# we take the contours that have been created by the masked based approach
input_array_i = contour_cube_i # nslices,5,1000,2
target_array_i = np.zeros_like(contour_cube_i) # nslices,5,1000,2
input_array_o = contour_cube_o
target_array_o = np.zeros_like(contour_cube_o)
ff_l_cube = np.ndarray((len(Z_SLICES), N_TIMESTEPS, N_EVALUATION_RADIAL))


for idx, z in enumerate(Z_SLICES):
    for t in range(N_TIMESTEPS):
        for eval in range(N_EVALUATION_RADIAL):
            # get the current indices we are after
            idx_xi = contour_cube_i_idx[idx, t, eval, 0]
            idx_yi = contour_cube_i_idx[idx, t, eval, 1]
            idx_xo = contour_cube_o_idx[idx, t, eval, 0]
            idx_yo = contour_cube_o_idx[idx, t, eval, 1]

            # CAVE: flowfield channel = z,y,x
            ffxi = flowfield_extract[idx, t, idx_xi, idx_yi, 2]
            ffyi = flowfield_extract[idx, t, idx_xi, idx_yi, 1]
            ffxo = flowfield_extract[idx, t, idx_xo, idx_yo, 2]
            ffyo = flowfield_extract[idx, t, idx_xo, idx_yo, 1]

            # add mask based spline and flowfield displacement coordinates
            # target[0] will be the following MS spline
            # target[1] will be the following ES spline
            # target[4] will be the following ED spline
            target_array_i[idx, t, eval, 0] = input_array_i[idx, t, eval, 0] + ffxi
            target_array_i[idx, t, eval, 1] = input_array_i[idx, t, eval, 1] + ffyi
            target_array_o[idx, t, eval, 0] = input_array_o[idx, t, eval, 0] + ffxo
            target_array_o[idx, t, eval, 1] = input_array_o[idx, t, eval, 1] + ffyo

            ff_l_cube[idx, t, eval] = get_displacement(target_array_o,
                                                       target_array_i,
                                                       np.roll(centroid_cube, 1, axis=1),
                                                       method='radial projected', #'radial cumulative', 'radial projected'
                                                       idx=idx, t=t, eval=eval)









############aha############
maskbased_distance_cube = distance_cube
ff_distance_cube = np.roll(ff_l_cube, 1, axis=1)
maskbased_distances_aha = sort_radians_into_aha(evaluation_angles, maskbased_distance_cube, N_EVALUATION_RADIAL_AHA)
ff_distances_aha = sort_radians_into_aha(evaluation_angles, ff_distance_cube, N_EVALUATION_RADIAL_AHA)

# calculate difference cubes aha based
# distances are 6,5,14,10 as aha,t,z,value
method='mean'
if method=='mean':
    maskbased_distances_aha_mean = maskbased_distances_aha.mean(axis=-1)
    ff_distances_aha_mean = ff_distances_aha.mean(axis=-1)

    l0 = maskbased_distances_aha_mean[:,0,:]
    l0 = l0[:,np.newaxis,:]
    l0 = np.repeat(l0, 5, axis=1)

    # calculate differences and strains
    maskbased_differences_aha_mean = maskbased_distances_aha_mean-l0
    ff_differences_aha_mean = ff_distances_aha_mean-l0
    maskbased_strain_aha_mean = maskbased_differences_aha_mean / l0 * 100
    ff_strain_aha_mean = ff_differences_aha_mean / l0 * 100

    # mean over all Z slices
    maskbased_distances_aha_mean = maskbased_distances_aha_mean.mean(axis=-1)
    maskbased_differences_aha_mean = maskbased_differences_aha_mean.mean(axis=-1)
    maskbased_strain_aha_mean = maskbased_strain_aha_mean.mean(axis=-1)
    ff_distances_aha_mean = ff_distances_aha_mean.mean(axis=-1)
    ff_differences_aha_mean = ff_differences_aha_mean.mean(axis=-1)
    ff_strain_aha_mean = ff_strain_aha_mean.mean(axis=-1)

plot_2x3distdiffstrain(maskbased_distances_aha_mean,
                           maskbased_differences_aha_mean,
                           maskbased_strain_aha_mean,
                           ff_distances_aha_mean,
                           ff_differences_aha_mean,
                           ff_strain_aha_mean,
                           N_MIDCAVITY_AHASEGMENTS, PHASE_LABELS, AHA_LABELS)




##################################################################################################################################

# plot grid with basal, midcavity, apical slice of the midcavity stack and all the five phases
N = 10
slices = [len(Z_SLICES) - 1, int(np.round(len(Z_SLICES) / 2)), 0]
fig, ax = plt.subplots(nrows=len(slices), ncols=5, figsize=(15, 5), sharey=True, sharex=True)
for idx, z in enumerate(slices):
    for phase in range(N_TIMESTEPS):
        # ax[idx, phase].imshow(vol_cube[phase, z + z_start, ...], cmap='gray')

        ax[idx, phase].scatter(input_array_i[z, phase, :, 0], input_array_i[z, phase, :, 1], s=0.8, c='g',
                               label='input spline')
        ax[idx, phase].scatter(input_array_o[z, phase, :, 0], input_array_o[z, phase, :, 1], s=0.8, c='g',
                               label='input spline')
        ax[idx, phase].plot([input_array_i[z, phase, ::N, 0], input_array_o[z, phase, ::N, 0]],
                            [input_array_i[z, phase, ::N, 1], input_array_o[z, phase, ::N, 1]], c='g')

        ax[idx, phase].scatter(target_array_i[z, phase, :, 0], target_array_i[z, phase, :, 1], s=0.8, c='r',
                               label='target spline')
        ax[idx, phase].scatter(target_array_o[z, phase, :, 0], target_array_o[z, phase, :, 1], s=0.8, c='r',
                               label='target spline')
        ax[idx, phase].plot([target_array_i[z, phase, ::N, 0], target_array_o[z, phase, ::N, 0]],
                            [target_array_i[z, phase, ::N, 1], target_array_o[z, phase, ::N, 1]], c='r')

        ax[0, phase].set_title(str(PHASE_LABELS[phase] + '$\longrightarrow$' + str(PHASE_LABELS[phase + 1])),
                               fontsize=20)
        ax[idx, phase].grid(False)
        # ax[idx, phase].set_xlim(30, 90)
        # ax[idx, phase].set_ylim(80, 20)
# plt.tight_layout()
plt.show()




INFO('Hi')
