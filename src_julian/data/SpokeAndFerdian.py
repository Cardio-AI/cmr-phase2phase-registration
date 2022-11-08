# define logging and working directory
from ProjectRoot import change_wd_to_project_root
change_wd_to_project_root()

# import helper functions
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot
import numpy as np
import logging
from logging import info as INFO
from src_julian.utils.skhelperfunctions import Console_and_file_logger
from src_julian.utils.myhelperfunctions import *
from src_julian.utils.myclasses import *
from src_julian.utils.mystrainfunctions import *
import seaborn as sns
import pandas as pd

# set up logging
Console_and_file_logger('mvfviz/dmd_temp', logging.INFO)

#######################################################################################################
path_to_segmentation_folder = '/mnt/ssd/julian/data/raw/flowfields/v10_nn011_nopostprocess/'
dirlist = [ item for item in os.listdir(path_to_segmentation_folder) if os.path.isdir(os.path.join(path_to_segmentation_folder, item)) ]
results = []

# specify only one patient if you would like to inspect
dirlist = ['hh_20190621']
# hh_20190621

for patient_name in dirlist:

    path_to_patient_folder = path_to_segmentation_folder + patient_name + '/'
    N_TIMESTEPS = 5
    Z_SPACING = 1
    N_SPLINE_EVALUATION = 1000
    N_EVALUATION_RADIAL = 1000
    N_SMOOTHING = 10
    N_EVALUATION_RADIAL_AHA = 10
    PHASE_LABELS = ['ED', 'MS', 'ES', 'PF', 'MD', 'ED', 'MS', 'ES', 'PF', 'MD']
    AHA_LABELS = ('AHA11', 'AHA10', 'AHA9', 'AHA8', 'AHA7', 'AHA12')
    N_MIDCAVITY_AHASEGMENTS = 6

    # get patient specific whole-heart-volume-range which means:
    # the slices where we do have segmentations available
    # as a basis to calculate the inner 35% midcavity slices
    # in order to see the range from where we will pick the midcavity slices, we have to check where we do have segmentations
    # i.e. we have 5 timesteps where we do have segmentations in the slices 5...50, 2...40, ...
    # then we will choose 2...50 as the final whole-heart-volume-range from where we will take the midcavity slices
    masks = get_segmentationarray(path_to_nii_folder=path_to_patient_folder, naming='_mask_', Ntimesteps=N_TIMESTEPS, Z_SPACING=Z_SPACING)
    data_masks = volume(masks, '4Dt', Z_SPACING)
    seg_available = data_masks.get_segmentationarray(resolution='slicewise')
    # the wholeheartvolumeborders numbers are idx numbers and refer equally to the mitk slices
    wholeheartvolumeborders = get_wholeheartvolumeborders(seg_available, N_TIMESTEPS)

    # from the whole heart volume indices, calculate the inner 35% aha midcavity indices
    z_start, z_end = get_midcavityvolumeborders(wholeheartvolumeborders)
    # we have to extend the z_end value by 1 so that the last index will be included in the Z_SLICES array
    Z_SLICES = np.arange(z_start, z_end+1, 1)

    # get segmentation cube
    naming_lvmasks = '_lvmask_'
    seg_lvcube = stack_nii_masks(path_to_patient_folder, naming_lvmasks, N_TIMESTEPS)
    seg_lvcube = np.roll(seg_lvcube, 1, axis=0)
    naming_masks = '_mask_' # the mask contains only the LV myo area
    seg_cube = stack_nii_masks(path_to_patient_folder, naming_masks, N_TIMESTEPS)
    seg_cube = np.roll(seg_cube, 1, axis=0)

    # get patients data as volume cube
    naming_cmr = '_cmr_full_'
    vol_cube = stack_nii_volume(path_to_patient_folder, naming_cmr, N_TIMESTEPS)
    vol_cube = append_array_entries(vol_cube)
    vol_cube = np.roll(vol_cube, 1, axis=0)

    # get flowfield
    # dont roll the flow!
    naming_flow = '_flow_full_'
    ff = stack_nii_flowfield(path_to_patient_folder, naming_flow, N_TIMESTEPS)
    flowfield = mvf(ff, '4Dt', Z_SPACING)


    ########################MASKBASED RESULT ARRAYS########################
    # for every slice, then for everytimestep within this slice, the pointwise distances will be returned
    distance_cube_maskbased, radius_cube_maskbased, \
    radius_evaluation_matrix_maskbased, contour_cube_i_maskbased, \
    contour_cube_o_maskbased, centroid_cube_maskbased, evaluation_angles_maskbased = spoke_get_maskbased_arrays(seg_cube, Z_SLICES, Z_SPACING,
                                                N_TIMESTEPS, N_SPLINE_EVALUATION, N_EVALUATION_RADIAL, N_SMOOTHING)

    # calculate distance cube simple
    # distance_cube_mean_maskbased = distance_cube_maskbased.mean(axis=0)

    # from spoke results, calculate difference and strain curves
    difference_cube_maskbased, strain_cube_maskbased = calculate_diff_and_strain_curves(distance_cube_maskbased)

    # plot the three cubes if wanted
    # distace_cube, diff_cube, strain_cube
    # plot_1x3distdiffstrain(distance_cube=distance_cube_maskbased, diff_cube=difference_cube_maskbased,
    #                        strain_cube=strain_cube_maskbased, patientname=patient_name)


    # plot the maskbased result for the patient
    # whole mean of stack; this means that this is the radial strain mean curve for all slices
    # plt.figure()
    # plt.plot(['ED', 'MS', 'ES', 'PF', 'MD'], 100*strain_cube.mean(axis=(2, 0)), label='(L-L0)/L0')
    # plt.xlabel('Phase')
    # plt.ylabel('Radial Strain in %')
    # plt.title('Maskbased Radial Strain wrt ED')
    # plt.legend()

    # inspections
    # calculate the positions of the min max values
    # ind_dist_min = np.unravel_index(np.argmin(distance_cube, axis=None), distance_cube.shape)
    # ind_dist_max = np.unravel_index(np.argmax(distance_cube, axis=None), distance_cube.shape)
    # ind_diff_min = np.unravel_index(np.argmin(diff_cube, axis=None), diff_cube.shape)
    # ind_diff_max = np.unravel_index(np.argmax(diff_cube, axis=None), diff_cube.shape)
    # ind_strain_min = np.unravel_index(np.argmin(strain_cube, axis=None), strain_cube.shape)
    # ind_strain_max = np.unravel_index(np.argmax(strain_cube, axis=None), strain_cube.shape)
    #
    # INFO('distance_cube.min(): ' + str(round(distance_cube.min(), 2)) + ' at ' +  str(ind_dist_min))
    # INFO('distance_cube.max(): ' + str(round(distance_cube.max(), 2)) + ' at ' +  str(ind_dist_max))
    # INFO('diff_cube.min(): ' + str(round(diff_cube.min(), 2)) + ' at ' +  str(ind_diff_min))
    # INFO('diff_cube.max(): ' + str(round(diff_cube.max(), 2)) + ' at ' +  str(ind_diff_max))
    # INFO('strain_cube.min(): ' + str(round(strain_cube.min(), 2)) + ' at ' +  str(ind_strain_min))
    # INFO('strain_cube.max(): ' + str(round(strain_cube.max(), 2)) + ' at ' +  str(ind_strain_max))

    # plot five images of phase ground truth masks and the splines for discussion with the group
    # fig,ax=plt.subplots(1, 5, figsize=(10,5), sharey=True)
    # slice_absolute = Z_SLICES[0]+np.round(len(Z_SLICES)/2).astype('int')
    # slice_relative = np.round(len(Z_SLICES)/2).astype('int')
    # for t in range(5):
    #     ax[t].imshow(seg_cube[t,slice_absolute,:,:,0],cmap='gray')
    #     # ax[t].imshow(vol_cube[t,slice_absolute,:,:,0],cmap='gray')
    #     ax[t].plot(contour_cube_i[slice_relative,t,:,0],contour_cube_i[slice_relative,t,:,1])
    #     ax[t].plot(contour_cube_o[slice_relative,t,:,0],contour_cube_o[slice_relative,t,:,1])
    #     ax[t].set_title(PHASE_LABELS[t])
    #     ax[t].set_xlim(50,90)
    #     ax[t].set_ylim(80,40)
    # plt.tight_layout(True)



    ########################FLOWFIELDBASED RESULT ARRAYS########################
    # first approach:
    # CAVE: we are not sure if pointwise applying and rounding the flowfield entries is valid!
    # pairwise calculation
    # contour_cube_i and contour_cube_o are the mask-based contours for every phase and have shape 15,5,1000,2
    # for every phase, the mask-based contour will be moved by the flowfield
    # then, the new distance between the moved points will be stored in the target array
    # this is what we call the ff_l_cube, because it contains lengths L and not L0
    # ff_l_cube will be shifted by 1, because for phase 0 it will contain the target lengths of the MS
    ff_l_cube, \
    movedpoints_i, \
    movedpoints_o = STRAIN_apply_ff_and_calc_targetlengths_pairwise(flowfield, contour_cube_i_maskbased, contour_cube_o_maskbased,
                                                                    centroid_cube_maskbased, Z_SLICES, N_TIMESTEPS,
                                                                    N_EVALUATION_RADIAL)

    # shift ff_l_cube by one, because we have previously calculated the lengths of the next phase for each step
    ff_l_cube = np.roll(ff_l_cube, 1, axis=1)

    # calculate difference and strain cube from points moved by flowfield
    # take explicitly maskbased L0 values
    # opt_l0_ED_cube parameter means that these values will be taken as l0 ED values
    l0_ED_cube = np.repeat(distance_cube_maskbased[:, 0, :][:, np.newaxis,:], 5, axis=1)
    diff_cube_pairwise, strain_cube_pairwise = calculate_diff_and_strain_curves(ff_l_cube, opt_l0_ED_cube=l0_ED_cube)

    # plot the distance, difference and strain cubes if wanted
    # plot_1x3distdiffstrain(distance_cube=ff_l_cube, diff_cube=diff_cube_pairwise, strain_cube=strain_cube_pairwise, patientname=patient_name)



    ######FERDIAN STRAINS######
    # METHOD1: give Ferdian always the ED mask and the current deformed masks by flowfield
    # Ferdian2020; Alistair Young Paper
    # output: z, t, 1 where 1 = radial strain value for slice over time
    # calculate Ferdian for target contours moved by flowfield
    # for every phase the maskbased contour was taken
    # Ferdian strains will be of the form 15,5,1
    endo_batch = np.roll(np.einsum('ztvc->ztcv', movedpoints_i), 1, axis=1) # mask moved by flowfield
    epi_batch = np.roll(np.einsum('ztvc->ztcv', movedpoints_o), 1, axis=1)  # mask moved by flowfield
    endo_ED = np.einsum('ztvc->ztcv', contour_cube_i_maskbased) # maskbased
    epi_ED = np.einsum('ztvc->ztcv', contour_cube_o_maskbased)  # maskbased
    Ferdian_method1_rr_strains = Ferdian2020_calculate_radial_strain(endo_batch, epi_batch,
                                                        endo_ED, epi_ED,
                                                        use_linear_strain=True,
                                                        use_maskbased_L0=True)

    # we can plot the two resulting curves here
    # plt.figure()
    # plt.plot(100*strain_cube_pairwise.mean(axis=(-1, 0)), label='(L-L0)/L0')
    # plt.plot(['ED', 'MS', 'ES', 'PF', 'MD'], 100*np.squeeze(Ferdian_method1_rr_strains).mean(axis=0), label='Ferdian')
    # plt.xlabel('Phase')
    # plt.ylabel('Radial Strain in %')
    # plt.title('Masks moved by flowfield - Radial Strain wrt ED')
    # plt.legend()


    # METHOD2: give Ferdian the ground truth masks splines without any flowfield application
    endo_batch = np.einsum('ztvc->ztcv', contour_cube_i_maskbased)
    epi_batch = np.einsum('ztvc->ztcv', contour_cube_o_maskbased)
    endo_ED = np.einsum('ztvc->ztcv', contour_cube_i_maskbased)
    epi_ED = np.einsum('ztvc->ztcv', contour_cube_o_maskbased)
    Ferdian_method2_rr_strains = Ferdian2020_calculate_radial_strain(endo_batch, epi_batch, endo_ED, epi_ED,
                                                               use_linear_strain=True, use_maskbased_L0=True)


    # METHOD3: apply flowfield pointwise and get the radial distances for every time step with respect to the centroid
    # we have used this method to check what the curves look like if we measure the distance to an arbitrary point in space such
    # as the centroid. the problem is, that then the radial strain will get negative, as the distance to the centroid shortens
    # also, the values over time are hard to compare.
    # ff_l_i, target_coordinates_i = STRAIN_apply_ff_and_get_radial_lengths(contour_cube_i_maskbased, flowfield=ff, centroid_cube_maskbased,
    #                                                                       Z_SLICES, N_TIMESTEPS, N_EVALUATION_RADIAL)
    # ff_l_o, target_coordinates_o = STRAIN_apply_ff_and_get_radial_lengths(contour_cube_o_maskbased, flowfield=ff, centroid_cube_maskbased,
    #                                                                       Z_SLICES, N_TIMESTEPS, N_EVALUATION_RADIAL)
    # here, the Ferdian strain calculation for this approach can be entered and plotted.
    # method 3 will not be used anymore.

    ###IMPORTANT PLOTS###
    # plot the Ferdian calculated strain results
    # plot_Ferdian_straincube_mean(Ferdian_method1_rr_strains)
    # plot_Ferdian_straincube_mean(Ferdian_method2_rr_strains)
    ###IMPORTANT PLOTS###

    # plot grid with basal, midcavity, apical slice of the midcavity stack and all the five phases
    # N = 10
    # slices = [len(Z_SLICES) - 1, int(np.round(len(Z_SLICES) / 2)), 0]
    # fig, ax = plt.subplots(nrows=len(slices), ncols=5, figsize=(15, 5), sharey=True, sharex=True)
    # for idx, z in enumerate(slices):
    #     for phase in range(N_TIMESTEPS):
    #         ax[idx, phase].imshow(vol_cube[phase, z + z_start, ...], cmap='gray')
    #
    #         ax[idx, phase].scatter(input_array_i[z, phase, :, 0], input_array_i[z, phase, :, 1], s=0.8, c='g',
    #                                label='input spline')
    #         ax[idx, phase].scatter(input_array_o[z, phase, :, 0], input_array_o[z, phase, :, 1], s=0.8, c='g',
    #                                label='input spline')
    #         ax[idx, phase].plot([input_array_i[z, phase, ::N, 0], input_array_o[z, phase, ::N, 0]],
    #                             [input_array_i[z, phase, ::N, 1], input_array_o[z, phase, ::N, 1]], c='g')
    #
    #         ax[idx, phase].scatter(target_array_i[z, phase, :, 0], target_array_i[z, phase, :, 1], s=0.8, c='r',
    #                                label='target spline')
    #         ax[idx, phase].scatter(target_array_o[z, phase, :, 0], target_array_o[z, phase, :, 1], s=0.8, c='r',
    #                                label='target spline')
    #         ax[idx, phase].plot([target_array_i[z, phase, ::N, 0], target_array_o[z, phase, ::N, 0]],
    #                             [target_array_i[z, phase, ::N, 1], target_array_o[z, phase, ::N, 1]], c='r')
    #
    #         ax[0, phase].set_title(str(PHASE_LABELS[phase] + '$\longrightarrow$' + str(PHASE_LABELS[phase + 1])),
    #                                fontsize=20)
    #         ax[idx, phase].grid(False)
    #         ax[idx, phase].set_xlim(30, 90)
    #         ax[idx, phase].set_ylim(80, 20)
    # # plt.tight_layout()
    # plt.show()

    # plot contour points, idx points, seg_data, etc.
    # quiver plot!
    # phase = 2 #ED
    # slice_relative = 0 # zero means we pick the most apical slice in the automated selected cardiac volume slices
    # slice_absolute = z_start + slice_relative
    # markersize=.5
    # N=1
    # plt.figure()
    # plt.imshow(vol_cube[phase,slice_absolute, :, :, 0], cmap='gray') #ED patient
    # plt.scatter(contour_cube_o[slice_relative, phase, :, 0], contour_cube_o[slice_relative, phase, :, 1], s=markersize)
    # plt.scatter(contour_cube_i[slice_relative, phase, :, 0], contour_cube_i[slice_relative, phase, :, 1], s=markersize)
    # plt.scatter(contour_cube_o_idx[slice_relative, phase, :, 0], contour_cube_o_idx[slice_relative, phase, :, 1], s=markersize)
    # plt.scatter(contour_cube_i_idx[slice_relative, phase, :, 0], contour_cube_i_idx[slice_relative, phase, :, 1], s=markersize)
    # # plt.imshow(seg_cube[phase,slice_absolute,:,:,0])
    # myinput = np.einsum('ztxyc->tzyxc', flowfield_extract)
    # ff = mvf(myinput[phase], format='4D', zspacing=Z_SPACING)
    # xx, yy, Fx, Fy = ff.plot_Grid2D_MV2Dor3D(slice=slice_relative, N=N)
    # plt.quiver(xx, yy, Fx, Fy, units='xy', angles='xy', color='y', scale=1)
    # plt.show()


    # ########################################################
    # plot myocardium thickness curves for every slice and all phases
    # fig1 = plt.figure(figsize=(12, 8))
    # ax1 = fig1.add_subplot(111)
    # for idx, z in enumerate(Z_SLICES):
    #     # plt.plot(PHASE_LABELS[0:5], distance_cube[idx].mean(axis=-1), label='l0')
    #     plt.plot(PHASE_LABELS[0:5], np.roll(ff_l_cube[idx].mean(axis=-1), 1, axis=0), label='z='+str(idx))
    # # plt.plot(PHASE_LABELS[0:5], ff_l0_cube.mean(axis=-1).mean(axis=0), label='l0')
    # # plt.plot(PHASE_LABELS[0:5], ff_l_cube.mean(axis=-1).mean(axis=0), label='l')
    #
    # # color the lines in rainbow colors to differentiate better
    # colormap = plt.cm.gist_ncar  # nipy_spectral, Set1,Paired
    # colors = [colormap(i) for i in np.linspace(0, 1, len(ax1.lines))]
    # for i, j in enumerate(ax1.lines):
    #     j.set_color(colors[i])
    # plt.title("Flowfield based distances per slice")
    # plt.xlabel('Phase')
    # plt.ylabel('Mean distance per slice')
    # plt.legend(loc='upper right')
    # plt.show()


    # create stack of arrays to be written as a dataframe
    # mean approach will be followed, not the median approach
    val = np.stack([l0_ED_cube.mean(axis=(-1,0)),
                    ff_l_cube.mean(axis=(-1, 0)),
                    diff_cube_pairwise.mean(axis=(-1, 0)),
                    strain_cube_pairwise.mean(axis=(-1, 0))],
                   axis=-1)

    # create df for one patient with all 5 phases
    df = pd.DataFrame(data=val, columns=['maskbased l0 ED', 'ff target distances', 'ff differences', 'ff strain'])

    # name the patient id as column
    df['id'] = patient_name
    # name the phase as string
    df['phase'] = PHASE_LABELS[0:5]
    # append to list
    results.append(df)

###RESULTS###
# merge all dataframes and plot the violinplots for Ferdian approaches etc.
df = pd.concat(results)
fig, ax = plt.subplots(1, 3, figsize=(17, 7))
sns.violinplot(x='phase', y='ff target distances', inner='quart', ax=ax[0], data=df)
sns.violinplot(x='phase', y='ff differences', inner='quart', ax=ax[1], data=df)
sns.violinplot(x='phase', y='ff strain', inner='quart', ax=ax[2], data=df)
plt.show()

##########################
###ADVANCEDPLOTTINGZONE###
##########################

# plot 2x2 figure where we have in the top row two phases, their patient data and the splines overlay
# the second row shows two polar plots with the respective splines, where the zone of maximum strain is marked
# configure the slice to be plotted in the function call as well as the phases left and right
# plot_segmentations_and_polarplots_sidebyside(seg_cube, vol_cube, contour_cube_i_maskbased, contour_cube_o_maskbased,
#                                                  evaluation_angles_maskbased, radius_cube_maskbased, strain_cube_maskbased, Z_SLICES,
#                                              slice_relative=12, phase_left=0, phase_right=2)


# plot 1x1 figure where we have a closeup of patient data, splines and polarplot for angles
# plot_closeup_of_patientdata_splines_with_polarplot_overlayed(vol_cube, contour_cube_o_maskbased,
#                                                                  centroid_cube_maskbased, evaluation_angles_maskbased,
#                                                                  radius_cube_maskbased, Z_SLICES, slice_relative=12, phase=2)


# plot 1x2 figure where we see side-by-side two polarplots with marks where we have biggest distance and differences
# plot_two_phases_as_polarplots_and_mark_biggest_differences(evaluation_angles_maskbased, radius_cube_maskbased,
#                                                            distance_cube_maskbased, difference_cube_maskbased,
#                                                            slice=7, phases=[0, 2])


# plot three lines where we have five images for each line showing input and target splines
# plot 3x5 figure where 3 lines plot different zslices of the stack and 5 are the timesteps
# in each subplot, in green will be plotted the maskbased contours and in red will be plotted the moved masks
# plot_3x5grid_maskbasedcontours_and_targetcontours(vol_cube, input_array_i=contour_cube_i_maskbased, input_array_o=contour_cube_o_maskbased,
#           target_array_i=movedpoints_i, target_array_o=movedpoints_o, Z_SLICES=Z_SLICES, N_TIMESTEPS=N_TIMESTEPS)


# mean strain curves over angle to identify areas which move too severe/too low
# figure 1x3 with distances, differences, strain per slice
# all curves of all slices will be plotted, but may overlay because we have the same mask multiple times
# x-axis is polar angle in degrees
# plot_1x3_distdiffstrain_per_slice(distance_cube_maskbased, difference_cube_maskbased, strain_cube_maskbased,
#                                   evaluation_angles_maskbased, Z_SLICES, phase=2, NTHCURVE=1)


# 1x3 plot with distance, difference, strain per slice and all phases
# x-axis is Phases
# lines are rainbow colored so that one can visually see directly if basal or apical areas influence
# mean values too severe
# plot_1x3_distdiffstrain_all_slices_per_phase(distance_cube=ff_l_cube, difference_cube=diff_cube_pairwise,
#                                              strain_cube=strain_cube_pairwise, Z_SLICES=Z_SLICES)

# plot 5xZSLICES grid where we can see all flowfield based contours
# headings of the small images will be the mean distance of the contour, so that thick contours get high values
# plot all target contours for all slices
# plot_5xZSLICES_alltargetcontours_for_all_slices(vol_cube, ff_l_cube, movedpoints_i, movedpoints_o, Z_SLICES, N_TIMESTEPS)

