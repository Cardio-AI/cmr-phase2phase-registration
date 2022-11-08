####################################################################################################################


#######################################################################################################

# define logging and working directory
from ProjectRoot import change_wd_to_project_root
# import helper functions
import matplotlib.pyplot as plt

import pandas as pd
import logging
from logging import info as INFO
from src_julian.utils.skhelperfunctions import Console_and_file_logger, extract_segments
from src_julian.utils.myhelperfunctions import *
from src_julian.utils.myclasses import *
from src_julian.utils.mystrainfunctions import *
from src_julian.utils.myAHAfunctions import *
# New import
import glob


def calculate_strain():
    patient_folders = sorted(glob.glob(os.path.join(data_root, '*/*/pred/*/')))
    sorting_lambda = lambda x: os.path.basename(os.path.dirname(x))
    patient_folders = sorted(patient_folders, key=sorting_lambda)
    patients = [os.path.basename(os.path.dirname(f)) for f in patient_folders]

    ###initial inits###
    df_patients = []  # df where we will store our results
    path_to_metadata_xls = '/mnt/ssd/julian/data/metadata/DMDTarique_2.0.xlsx'
    path_to_segmentation_folder = '/mnt/ssd/julian/data/raw/flowfields/v16_smoothmyo_maskloss0_001_compose_reg_dmd/'

    # v16_smoothmyo_maskloss0_001_compose_reg_dmd_control
    # v16_smoothmyo_maskloss0_001_compose_reg_dmd
    # v15_smooth_myo_maskloss0_001_dmd_control
    # v15_smooth_myo_maskloss0_001
    # v14_refactored
    # v13_added_z_motion_solved_stack_bugfix
    # get patients names list
    patients = [item for item in sorted(os.listdir(path_to_segmentation_folder)) if
                os.path.isdir(os.path.join(path_to_segmentation_folder, item))]
    patients = ['hh_20190621']
    # hh_20190621
    # cf_20191009
    # dv_20190624
    # ct_20191108
    # me_20190816
    # ac_20181203
    # wr_20170905
    # nb_20180628
    # rc_20181207
    # aa_20180710
    # pm_20190801
    # dc_20180516
    # ci_20190513
    # ht_20191007
    ####################################
    # methods
    df_style = 'time'
    composition_direction = 'reversed'
    RVIP_method = 'staticED'  # dynamically
    com_method = 'staticED'  # dynamically
    spacing = 1.5
    N_TIMESTEPS = 5
    Z_SPACING = 1
    label_bloodpool = 3
    label_lvmyo = 1
    sheet_name_soalge = 'clean DMD'  # clean Control, clean DMD
    sheet_name_ahastrain = 'DMD AHA Strain'  # Control AHA Strain, DMD AHA Strain
    ####################################
    # loop here
    for i, patient_name in enumerate(patients):
        path_to_patient_folder = path_to_segmentation_folder + patient_name + '/'


        vol_cube = stack_nii_volume(path_to_patient_folder, 'cmr_target_', N_TIMESTEPS)  # refactored
        # vol_cube = stack_nii_volume(path_to_patient_folder, '_targetcmr_', N_TIMESTEPS) # old
        vol_cube = vol_cube[..., np.newaxis]
        # vol_cube = np.roll(vol_cube, 1, axis=0)

        # LV MYO MASKS
        # targetmask doesnt need to be rolled
        # previously "mask" files were used here
        mask_lvmyo = stack_nii_masks(path_to_patient_folder, 'myo_target_', N_TIMESTEPS)  # refactored
        # mask_lvmyo = stack_nii_masks(path_to_patient_folder, '_targetmask_', N_TIMESTEPS) # old
        # mask_lvmyo = np.roll(mask_lvmyo, 1, axis=0)

        # WHOLE MASKS
        # lvtargetmask doesnt need to be rolled
        mask_whole = stack_nii_masks(path_to_patient_folder, 'fullmask_target_', N_TIMESTEPS)  # refactored
        # mask_whole = stack_nii_masks(path_to_patient_folder, '_lvtargetmask_', N_TIMESTEPS) # old
        # mask_whole = np.roll(mask_whole, 1, axis=0)

        # FLOWFIELD FULL
        # dont roll the flow!
        # originally from Svens output, ff is of shape cxyzt with c=zyx
        ff = mvf(data=stack_nii_flowfield(path_to_patient_folder, 'flow_', N_TIMESTEPS), format='4Dt',
                 zspacing=Z_SPACING)  # refactored
        # ff = mvf(data=stack_nii_flowfield(path_to_patient_folder, '_flow_', N_TIMESTEPS), format='4Dt', zspacing=Z_SPACING) # old
        # ff is now of shape tzyxc with c=zyx
        # ff_switched = mvf(data=ff.switch_channel(), format='4Dt', zspacing=Z_SPACING)
        # get whole composed itk flowfield
        # forward = standard
        # reversed means that the displacement fields are added in reverse order
        # ff_whole_itk = ff.compose_sitk(np.arange(0, ff.nz), method=composition_direction)
        # ff_whole_itk is tzyxc with c=zyx

        # IDXs FROM (SPARSE) LVMYOMASKS
        # get all indexes of phases where all timesteps contain lv myo segmentations
        lvmyo_idxs = np.argwhere(
            np.all(volume(mask_lvmyo, '4Dt', 1).get_segmentationarray(resolution='slicewise')[..., 0],
                   axis=0)).flatten()

        # IDXs FROM RVIP DETECTION IN WHOLE MASKS
        # get lowest and highest index of z where all timesteps have RVIP identified
        rvip_range = calculate_wholeheartvolumeborders_by_RVIP(mask_whole)

        # define from where we take the identified heart volume borders
        wholeheartvolumeborders_lvmyo = [lvmyo_idxs[0], lvmyo_idxs[-1]]  # from LVMYOMASKS range
        wholeheartvolumeborders_rviprange = [rvip_range[0], rvip_range[-1]]  # from RVIP range

        # level ranges
        # 2021.10.06: lvmyo more accurate when not-sparse
        base_slices, midcavity_slices, apex_slices = get_volumeborders(wholeheartvolumeborders_lvmyo)  # by lvmyo-range
        # base_slices, midcavity_slices, apex_slices = get_volumeborders(wholeheartvolumeborders_rviprange)  # by lvmyo-range

        # zsliceswholeheart = np.arange(wholeheartvolumeborders[0], wholeheartvolumeborders[1] + 1)

        # SPARSE
        # sort lvmyo sparse idxs into level ranges
        # base_idxs, midcavity_idxs, apex_idxs = get_volumeborders(wholeheartvolumeborders_lvmyo)
        # base_slices = np.intersect1d(lvmyo_idxs, base_idxs)
        # midcavity_slices = np.intersect1d(lvmyo_idxs, midcavity_idxs)
        # apex_slices = np.intersect1d(lvmyo_idxs, apex_idxs)

        x = 0

        # CF
        # base_slices=[35]
        # midcavity_slices=[21, 28]
        # apex_slices=[14]

        # HH
        # base_slices=[35, 42]
        # midcavity_slices=[21, 28]
        # apex_slices=[6, 14]

        # ME
        # base_slices=[35,42]
        # midcavity_slices=[21,28]
        # apex_slices=[6,14]

        # z-array for the whole volume
        # zsliceswholevolume = np.arange(0, ff_whole_itk.shape[1])

        ############COMPOSED FLOWFIELDS############
        ###### get my own manually composed ff ######
        # INFO('Calculating my composition now. Takes time.')
        # ff_sum, targets = ff.compose_myimplementation() # my algorithm
        # ff_sum = mvf(data=ff_sum, format='4Dt', zspacing=1)
        # ff_comp_mine = ff_sum.switch_channel() # switch c=channel from xyz (after stackniiflowfield) to zyx

        ###### get svens composed ff ######
        # dont roll the flow!
        # originally from Svens output, ff is of shape cxyzt with c=zyx
        naming_flow_comp = 'flow_composed_'
        ff_whole_Sven = stack_nii_flowfield(path_to_patient_folder, naming_flow_comp, N_TIMESTEPS)
        # ff_whole_Sven is tzyxc with c=zyx

        ###### get laliths just adding composed ff ######
        # ff_comp_add = ff_switched.compose_justadding() # order stays the same
        # slice it
        # ff_comp_add = ff_comp_add[:, Z_SLICES, ...]
        # ax_labels_ED_relative = ['ED ' + '$\longrightarrow$' + ' MS',
        #                          'ED ' + '$\longrightarrow$' + ' ES',
        #                          'ED ' + '$\longrightarrow$' + ' PF',
        #                          'ED ' + '$\longrightarrow$' + ' MD',
        #                          'ED ' + '$\longrightarrow$' + ' ED']
        # # input = np.copy(ff.Data)
        # input_raw = np.linalg.norm(np.copy(ff.Data), axis=-1)
        # input_comp = np.linalg.norm(np.copy(ff_whole_Sven), axis=-1)
        # plt.figure()
        # plt.plot(ax_labels_ED_relative, input_raw[mask_lvmyo==1].mean(axis=(1, 2, 3)), label='mag raw')
        # # plt.plot(ax_labels_ED_relative, input.min(axis=(1, 2, 3)), label='min')
        # # plt.plot(ax_labels_ED_relative, input.max(axis=(1, 2, 3)), label='max')
        # plt.plot(ax_labels_ED_relative, input_comp.mean(axis=(1, 2, 3)), label='mag comp')
        # # plt.plot(ax_labels_ED_relative, ff_whole_Sven.mean(axis=(1, 2, 3))[:, 0], label='Sven_z')
        # # plt.plot(ax_labels_ED_relative, ff_whole_Sven.mean(axis=(1, 2, 3))[:, 1], label='Sven_y')
        # # plt.plot(ax_labels_ED_relative, ff_whole_Sven.mean(axis=(1, 2, 3))[:, 2], label='Sven_x')
        # plt.legend()
        # plt.suptitle('Comparison of composition axes')
        # plt.show()
        #
        #
        # plt.figure()
        # for t in range(5):
        #     plt.scatter(t, input_raw[t][mask_lvmyo[t][..., 0] == 1].mean(), label='ff raw masked')
        # plt.legend()

        # y_raw_z = calculate_ff_magnitude_masked_over_time_and_z(wholeheartvolumeborders=wholeheartvolumeborders_lvmyo,
        #                     ff_raw=np.copy(ff.Data),
        #                     mask_lvmyo=mask_lvmyo)
        # y_raw = y_raw_z.mean(axis=1)
        #
        # y_comp_z_Sven = calculate_ff_magnitude_masked_over_time_and_z(wholeheartvolumeborders=wholeheartvolumeborders_lvmyo,
        #                     ff_raw=np.copy(ff_whole_Sven),
        #                     mask_lvmyo=mask_lvmyo)
        # y_comp_Sven = y_comp_z_Sven.mean(axis=1)
        #
        # y_comp_z_sitk = calculate_ff_magnitude_masked_over_time_and_z(wholeheartvolumeborders=wholeheartvolumeborders_lvmyo,
        #                     ff_raw=np.copy(ff_whole_itk),
        #                     mask_lvmyo=mask_lvmyo)
        # y_comp_sitk = y_comp_z_sitk.mean(axis=1)
        #
        # plt.figure(figsize=(8,8))
        # plt.plot(y_raw, label='Raw')
        # plt.plot(y_comp_Sven, label='Composed_Sven')
        # plt.plot(y_comp_sitk, label='Composed_sitk')
        # plt.ylim(0, 3)
        # plt.suptitle('Masked Displacement Field Magnitude over Phase in LVMYO Range')
        # plt.xlabel('Phases')
        # plt.ylabel('Mean Vector Magnitude')
        # plt.legend()
        # plt.show()

        # plot mean channels over time
        # plot_mean_channel_overtime(ff_raw=np.copy(ff.Data), ff_comp_Sven=ff_whole_Sven, ff_comp_itk=ff_whole_itk)

        # plot
        # plot_motioncurve_per_z_slice_perphase(y=y_comp_z_Sven)

        x = 0

        # plot quiver one slice
        # t = 0
        # slice = 24
        # N = 2
        # whole=ff_whole_Sven
        # # whole has dimensions tzyxc with c.ndim=3 zyx
        # test = mvf(data=whole[t], format='4D', zspacing=1)
        # xx, yy, Fx, Fy = test.plot_Grid2D_MV2Dor3D(slice=slice, N=N)
        # fig, ax = plt.subplots()
        # plt.imshow(mask_whole[t, slice, ..., 0])
        # plt.quiver(xx, yy, Fx, Fy, units='xy', angles='xy', scale=1, color='k')
        # ax.set_title('MVF plot')
        # ax.set_aspect('equal')
        # plt.show()

        # calculate RVIP cube
        # contains anterior and inferior mean RVIP coordinates for LVMYO masks slices range
        # contains mean RVIP coordinates for base, mid cavity, apex ranges!
        # c = y,x
        # dynamically: every timestep gets different mean RVIPs for base mid cavity apex
        # staticED: every timestep contains the same mean RVIPs for base mid cavity apex of ED
        RVIP_cube = calculate_RVIP_cube(mask_whole, base_slices, midcavity_slices, apex_slices, method=RVIP_method)

        # method 2 cannot handle None RVIP entries in between volume
        # RVIP_cube = calculate_RVIP_cube2(mask_whole, rvip_range, base_slices, midcavity_slices, apex_slices)

        # CALCULATE COM CUBE 5x3x3
        # base, midcavity, apex = axis=1
        # calulate com cube
        # c = z,y,x
        com_cube = calculate_center_of_mass_cube(mask_whole, label_bloodpool=label_bloodpool,
                                                 base_slices=base_slices, midcavity_slices=midcavity_slices,
                                                 apex_slices=apex_slices, method=com_method)

        # move manually x+50
        # com_cube[:,1,2]=com_cube[:,1,2]+10

        x = 0

        # validation plot
        # t=0
        # fig,ax=plt.subplots(1,len(midcavity_slices),sharey=True)
        # for zrel, zabs in enumerate(midcavity_slices):
        #     ax[zrel].imshow(mask_whole[t,zabs], cmap='gray')
        #     ax[zrel].scatter(com_cube[t,1,1], com_cube[t,1,2], label='com')
        #     ax[zrel].scatter(RVIP_cube[t, zabs, 0, 0], RVIP_cube[t, zabs, 0, 1], label='ant')
        #     ax[zrel].scatter(RVIP_cube[t, zabs, 1, 0], RVIP_cube[t, zabs, 1, 1], label='inf')
        # plt.legend()
        # plt.show()
        #
        # zrel=7
        # t=0
        # zabs=midcavity_slices[zrel]
        # fig, ax = plt.subplots(1, 1, sharey=True)
        # ax.imshow(mask_whole[t, zabs], cmap='gray')
        # ax.scatter(com_cube[t, 1, 1], com_cube[t, 1, 2], label='com')
        # ax.scatter(RVIP_cube[t, zabs, 0, 0], RVIP_cube[t, zabs, 0, 1], label='ant')
        # ax.scatter(RVIP_cube[t, zabs, 1, 0], RVIP_cube[t, zabs, 1, 1], label='inf')
        # ax.imshow(mask_lvmyo[t, zabs], alpha=.5)
        # plt.legend()
        # plt.show()

        # move manually x+50
        # com_cube[:,1,2]=com_cube[:,1,2]+10

        # x=0

        # validation plot
        # t=0
        # fig,ax=plt.subplots(1,len(midcavity_slices),sharey=True)
        # for zrel, zabs in enumerate(midcavity_slices):
        #     ax[zrel].imshow(mask_whole[t,zabs], cmap='gray')
        #     ax[zrel].scatter(com_cube[t,1,1], com_cube[t,1,2], label='com')
        #     ax[zrel].scatter(RVIP_cube[t, zabs, 0, 0], RVIP_cube[t, zabs, 0, 1], label='ant')
        #     ax[zrel].scatter(RVIP_cube[t, zabs, 1, 0], RVIP_cube[t, zabs, 1, 1], label='inf')
        # plt.legend()
        # plt.show()

        # set which composition will be used as composed flowfield for further analyses
        ff_whole = ff_whole_Sven

        # NOTES ON LAYOUTS BEFORE DEEPSTRAIN
        # ff_whole_itk : tzyxc with c=zyx
        # mask_lvmyo : tzyxc
        # com_cube : tc with c=zyx
        # zsliceswholevolume : 0...63
        # N_TIMESTEPS : 5
        # REARRANGE INPUTS FOR DEEPSTRAIN
        # prepare my flowfield for Morales
        ff_Moralesinput = mvf(ff_whole, format='4Dt', zspacing=Z_SPACING)
        ff_Moralesinput = ff_Moralesinput.switch_channel()
        ff_Moralesinput = np.einsum('tzyxc->txyzc', ff_Moralesinput)

        # prepare mask_lvmyo for Morales
        mask_lvmyo_Moralesinput = np.einsum('tzyxc->txyzc', mask_lvmyo)

        # prepare com_cube for Morales
        # for roll_to_center to work, we have to give t,c with c=y,x,z into Morales
        # DeepStrain will not make use of cz, so only the first two entries are relevant
        com_cube_Moralesinput = np.zeros_like(com_cube)
        com_cube_Moralesinput[..., 0] = com_cube[..., 1]
        com_cube_Moralesinput[..., 1] = com_cube[..., 2]
        com_cube_Moralesinput[..., 2] = com_cube[..., 0]

        # validation plot
        # t, z = (2, 45)
        # plt.figure()
        # plt.imshow(mask_lvmyo_Moralesinput[t,:,:,z,0], cmap='gray')
        # plt.scatter(com_cube_Moralesinput[t,0], com_cube_Moralesinput[t,1], label='com')
        # plt.show()

        # DEEPSTRAIN
        # calculate strain curves by Morales
        # Radial_itk, Circumferential_itk, masks_rot_itk = myMorales(ff_Moralesinput, mask_lvmyo_Moralesinput,
        #                                                            com_cube_Moralesinput, zsliceswholevolume, spacing)
        Radial_Sven_base, \
        Circumferential_Sven_base, \
        masks_rot_Sven_base = myMorales(ff_comp=ff_Moralesinput[:, :, :, base_slices],
                                        mask_lvmyo=mask_lvmyo_Moralesinput[:, :, :, base_slices],
                                        com_cube=com_cube_Moralesinput[:, 0, :],
                                        spacing=spacing)
        Radial_Sven_mc, \
        Circumferential_Sven_mc, \
        masks_rot_Sven_mc = myMorales(ff_comp=ff_Moralesinput[:, :, :, midcavity_slices],
                                      mask_lvmyo=mask_lvmyo_Moralesinput[:, :, :, midcavity_slices],
                                      com_cube=com_cube_Moralesinput[:, 1, :],
                                      spacing=spacing)
        Radial_Sven_apex, \
        Circumferential_Sven_apex, \
        masks_rot_Sven_apex = myMorales(ff_comp=ff_Moralesinput[:, :, :, apex_slices],
                                        mask_lvmyo=mask_lvmyo_Moralesinput[:, :, :, apex_slices],
                                        com_cube=com_cube_Moralesinput[:, 2, :],
                                        spacing=spacing)

        # stack the information together
        # Radial_Sven_stacked = np.concatenate([Radial_Sven_base, Radial_Sven_mc, Radial_Sven_apex], axis=-1)
        # Circumferential_Sven_stacked = np.concatenate([Circumferential_Sven_base, Circumferential_Sven_mc, Circumferential_Sven_apex], axis=-1)
        # masks_rot_Sven_stacked = np.concatenate([masks_rot_Sven_base, masks_rot_Sven_mc, masks_rot_Sven_apex], axis=-1)

        # create whole colume arrays with stacked information contained
        Radial_Sven = np.zeros((ff.nt, ff.nx, ff.ny, ff.nz))
        # Radial_Sven[..., apex_slices[0]:base_slices[-1]+1] = Radial_Sven_stacked
        Radial_Sven[..., base_slices] = Radial_Sven_base
        Radial_Sven[..., midcavity_slices] = Radial_Sven_mc
        Radial_Sven[..., apex_slices] = Radial_Sven_apex

        Circumferential_Sven = np.zeros((ff.nt, ff.nx, ff.ny, ff.nz))
        Circumferential_Sven[..., base_slices] = Circumferential_Sven_base
        Circumferential_Sven[..., midcavity_slices] = Circumferential_Sven_mc
        Circumferential_Sven[..., apex_slices] = Circumferential_Sven_apex

        masks_rot_Sven = np.zeros((ff.nt, ff.nx, ff.ny, ff.nz))
        masks_rot_Sven[..., base_slices] = masks_rot_Sven_base
        masks_rot_Sven[..., midcavity_slices] = masks_rot_Sven_mc
        masks_rot_Sven[..., apex_slices] = masks_rot_Sven_apex

        # most important
        # rearrange Strain Tensor for further processing
        Err = np.einsum('txyz->tzyx', Radial_Sven)
        Ecc = np.einsum('txyz->tzyx', Circumferential_Sven)
        masks_rot_lvmyo = np.einsum('txyz->tzyx', masks_rot_Sven)
        # now, Strain Tensor and sector masks do have the shape
        # tzyx = (5,16,128,128)

        x = 0

        # plt.plot(get_mean_strain_values_from_Morales(Err, masks_rot_lvmyo))
        # plt.plot(get_mean_strain_values_from_Morales(Ecc, masks_rot_lvmyo))

        # validation plot
        # t=1
        # z=32
        # plt.figure()
        # plt.imshow(Err[t,z], cmap='inferno')
        # plt.colorbar()
        # plt.imshow(roll_to_center(vol_cube[t, z, ..., 0], com_cube[t, 2], com_cube[t, 1]), cmap='gray', alpha=.5)
        # plt.imshow(masks_rot_lvmyo[t,z], alpha=.2, cmap='gray')
        # # plt.imshow(roll_to_center(mask_whole[t,z,...,0], com_cube[t, 2], com_cube[t, 1]), alpha=.3)
        # plt.show()

        # t = 0
        # z = 32
        # cx, cy = (com_cube[t,0], com_cube[t,1])
        # rvipupx, rvipupy = RVIP_cube[t, z, 0]
        # rviplowx, rviplowy = RVIP_cube[t, z, 1]
        #
        # plt.figure()
        # plt.imshow(vol_cube[t, z, ...], cmap='gray')
        # plt.imshow(mask_whole[t, z, ...], alpha=0.2)
        # plt.scatter(rvipupy, rvipupx, label='up')
        # plt.scatter(rviplowy, rviplowx, label='low')
        # plt.scatter(cy, cx, label='com')
        # plt.legend()
        # plt.show()

        # calculate strain curves by Morales
        # Radial_Sven, Circumferential_Sven, masks_rot_Sven = myMorales(ff_comp_Sven, mask_lvmyo, mask_whole, np.arange(0, ff_whole_itk.shape[1]), N_TIMESTEPS)
        # Radial_itk, Circumferential_itk, masks_rot_itk = myMorales(ff_comp_itk, seg_lvcube, Z_SLICES, N_TIMESTEPS)
        # Radial_add, Circumferential_add, masks_rot_add = myMorales(ff_comp_add, seg_lvcube, Z_SLICES, N_TIMESTEPS)

        # itk_MCMRS = get_mean_strain_values_from_Morales(Radial_itk, masks_rot_itk)

        ######AHA DIVISIONS#####
        # calculate array of sector masks with the same shape as the whole 4DCMR
        # entries only where slices are given
        sector_masks_raw_base = calculate_sector_masks(mask_whole=mask_whole,
                                                       com_cube=com_cube[:, 0, :],
                                                       RVIP_cube=RVIP_cube,
                                                       Z_SLICES=base_slices,
                                                       level='base')
        sector_masks_raw_midcavity = calculate_sector_masks(mask_whole=mask_whole,
                                                            com_cube=com_cube[:, 1, :],
                                                            RVIP_cube=RVIP_cube,
                                                            Z_SLICES=midcavity_slices,
                                                            level='mid-cavity')
        sector_masks_raw_apex = calculate_sector_masks(mask_whole=mask_whole,
                                                       com_cube=com_cube[:, 2, :],
                                                       RVIP_cube=RVIP_cube,
                                                       Z_SLICES=apex_slices,
                                                       level='apex')

        # validation plots for sector masks raw
        # t=3
        # z=20
        # anty, antx = RVIP_cube[t, z, 0]
        # infy, infx = RVIP_cube[t, z, 1]
        # plt.imshow(vol_cube[t,z], cmap='gray')
        # plt.imshow(sector_masks_raw_midcavity[t,z],alpha=.5)
        # plt.scatter(com_cube[t,1], com_cube[t,2], label='com')
        # plt.scatter(anty, antx, label='ant')
        # plt.scatter(infy, infx, label='inf')
        # plt.legend()
        # plt.show()

        # roll sector masks to center for overlay plotting with Strain Maps
        sector_masks_rot_base = roll_sector_mask_to_bloodpool_center(sector_mask_raw=sector_masks_raw_base,
                                                                     com_cube=com_cube[:, 0, :],
                                                                     N_TIMESTEPS=N_TIMESTEPS,
                                                                     Z_SLICES=base_slices)
        sector_masks_rot_midcavity = roll_sector_mask_to_bloodpool_center(sector_mask_raw=sector_masks_raw_midcavity,
                                                                          com_cube=com_cube[:, 1, :],
                                                                          N_TIMESTEPS=N_TIMESTEPS,
                                                                          Z_SLICES=midcavity_slices)
        sector_masks_rot_apex = roll_sector_mask_to_bloodpool_center(sector_mask_raw=sector_masks_raw_apex,
                                                                     com_cube=com_cube[:, 2, :],
                                                                     N_TIMESTEPS=N_TIMESTEPS,
                                                                     Z_SLICES=apex_slices)

        # validation plotting overlay cmr, Strainmap, masksrot, sectorrot
        # t=2
        # z_rel=8
        # z_abs=24
        # plt.figure()
        # plt.imshow()
        # plt.colorbar()
        # plt.imshow()
        # plt.imshow()
        # plt.imshow(sector_masks_rot_midcavity[t, z_rel],alpha=.2)
        # # plt.imshow(roll_to_center(mask_whole[t,z,...,0], com_cube[t, 2], com_cube[t, 1]), alpha=.3)
        # plt.legend()
        # plt.show()

        # calculate the AHA cubes
        # we can read out the Err and Ecc with the sector masks in a desired AHA segment order
        # we can define the order of the array entries by AHA ascending here
        N_AHA_base = [1, 2, 3, 4, 5, 6]
        N_AHA_midcavity = [7, 8, 9, 10, 11, 12]
        N_AHA_apex = [13, 14, 15, 16]
        AHAcube_base = calculate_AHA_cube(Err=Err[:, base_slices], Ecc=Ecc[:, base_slices],
                                          sector_masks_rot=sector_masks_rot_base,
                                          masks_rot=masks_rot_lvmyo[:, base_slices],
                                          Z_SLICES=base_slices,
                                          N_AHA=N_AHA_base)
        AHAcube_midcavity = calculate_AHA_cube(Err=Err[:, midcavity_slices], Ecc=Ecc[:, midcavity_slices],
                                               sector_masks_rot=sector_masks_rot_midcavity,
                                               masks_rot=masks_rot_lvmyo[:, midcavity_slices],
                                               Z_SLICES=midcavity_slices,
                                               N_AHA=N_AHA_midcavity)
        AHAcube_apex = calculate_AHA_cube(Err=Err[:, apex_slices], Ecc=Ecc[:, apex_slices],
                                          sector_masks_rot=sector_masks_rot_apex,
                                          masks_rot=masks_rot_lvmyo[:, apex_slices],
                                          Z_SLICES=apex_slices,
                                          N_AHA=N_AHA_apex)

        x = 0

        # new visalization method ONEVIEW
        plot_4x5_MaskQuiver_Magnitude_Err_Ecc(ff_composed=ff_whole_Sven, mask_whole=mask_whole, Err=Err, Ecc=Ecc, z=24,
                                              N=1)

        # bullsplots
        # Circle
        # get cvi Circle Peak Strain values
        # df_dmdahastrain = pd.read_excel(io=path_to_metadata_xls, sheet_name='DMD AHA Strain', index_col=0, header=0)
        # cvi_prs = get_parameter_series_from_xls(df=df_dmdahastrain, parametername='radial peak strain (%)',
        #                                         patientname=patient_name)
        # cvi_pcs = get_parameter_series_from_xls(df=df_dmdahastrain, parametername='circumferential peak strain (%)',
        #                                         patientname=patient_name)
        # myBullsplot(data=cvi_prs, param='Err')
        # myBullsplot(data=cvi_pcs, param='Ecc')
        #
        # # plot bullsplot for minmax over all phases
        # PRS_values_base = AHAcube_base[..., 0].mean(axis=2).max(axis=1)
        # PRS_values_midcavity = AHAcube_midcavity[..., 0].mean(axis=2).max(axis=1)
        # PRS_values_apex = AHAcube_apex[..., 0].mean(axis=2).max(axis=1)
        # our_prs = 100 * np.concatenate([PRS_values_base, PRS_values_midcavity, PRS_values_apex], axis=0)
        # myBullsplot(data=our_prs, param='Err')
        #
        # PCS_values_base = AHAcube_base[..., 1].mean(axis=2).min(axis=1)
        # PCS_values_midcavity = AHAcube_midcavity[..., 1].mean(axis=2).min(axis=1)
        # PCS_values_apex = AHAcube_apex[..., 1].mean(axis=2).min(axis=1)
        # our_pcs = 100 * np.concatenate([PCS_values_base, PCS_values_midcavity, PCS_values_apex], axis=0)
        # myBullsplot(data=our_pcs, param='Ecc')
        #
        # # plot bullsplot for ES
        # def_phase=1 # ES=1
        # PRS_values_base_at_ES = AHAcube_base[..., 0].mean(axis=2)[:, def_phase]
        # PRS_values_mc_at_ES = AHAcube_midcavity[..., 0].mean(axis=2)[:, def_phase]
        # PRS_values_apex_at_ES = AHAcube_apex[..., 0].mean(axis=2)[:, def_phase]
        # our_phase_rs = 100 * np.concatenate([PRS_values_base_at_ES, PRS_values_mc_at_ES, PRS_values_apex_at_ES], axis=0)
        # myBullsplot(data=our_phase_rs, param='Err')
        #
        # PCS_values_base_at_ES = AHAcube_base[..., 1].mean(axis=2)[:, def_phase]
        # PCS_values_mc_at_ES = AHAcube_midcavity[..., 1].mean(axis=2)[:, def_phase]
        # PCS_values_apex_at_ES = AHAcube_apex[..., 1].mean(axis=2)[:, def_phase]
        # our_phase_cs = 100 * np.concatenate([PCS_values_base_at_ES, PCS_values_mc_at_ES, PCS_values_apex_at_ES], axis=0)
        # myBullsplot(data=our_phase_cs, param='Ecc')

        x = 0

        # output min max mean strain for patient; Err and Ecc
        rs_overtime_base = AHAcube_base.mean(axis=2)[..., 0]
        cs_overtime_base = AHAcube_base.mean(axis=2)[..., 1]
        rs_overtime_mc = AHAcube_midcavity.mean(axis=2)[..., 0]
        cs_overtime_mc = AHAcube_midcavity.mean(axis=2)[..., 1]
        rs_overtime_apex = AHAcube_apex.mean(axis=2)[..., 0]
        cs_overtime_apex = AHAcube_apex.mean(axis=2)[..., 1]
        # 80,1 = 5 timesteps * 16 segments as column
        rs_AHA_overtime = np.concatenate((rs_overtime_base, rs_overtime_mc, rs_overtime_apex), axis=0).reshape(
            (16 * 5, 1))
        cs_AHA_overtime = np.concatenate((cs_overtime_base, cs_overtime_mc, cs_overtime_apex), axis=0).reshape(
            (16 * 5, 1))
        INFO('Err min: {:3.1f}%'.format(100 * rs_AHA_overtime.min()))
        INFO('Err max: {:3.1f}%'.format(100 * rs_AHA_overtime.max()))
        INFO('Err mean: {:3.1f}%'.format(100 * rs_AHA_overtime.mean()))
        INFO('Ecc min: {:3.1f}%'.format(100 * cs_AHA_overtime.max()))
        INFO('Ecc max: {:3.1f}%'.format(100 * cs_AHA_overtime.min()))
        INFO('Ecc mean: {:3.1f}%'.format(100 * cs_AHA_overtime.mean()))

        x = 0

        # plot cmr overlay strain map masked
        # plot_3x5_cmroverlaywithmaskedstrain(ff_whole_itk, Err, Ecc, base_slices, midcavity_slices, apex_slices, vol_cube,
        #                                     com_cube, masks_rot_lvmyo, method='Ecc')
        #
        # # plot curves per AHA and heart base mc apex
        # plot_3x2_AHAStrainMotioncurvesOvertime(AHAcube_base, AHAcube_midcavity, AHAcube_apex,
        #                                        Err_min=-20, Err_max=200, Ecc_min=-15, Ecc_max=20)

        # # # base
        # plot_3x5grid_CMRxStrainxSectormasks(com_cube=com_cube[:,0,:],
        #                                     Radial_Morales=Ecc,
        #                                     masks_rot_Morales=masks_rot_lvmyo,
        #                                     vol_cube=vol_cube,
        #                                     sector_masks_raw=sector_masks_raw_base,
        #                                     N_TIMESTEPS=N_TIMESTEPS,
        #                                     Z_SLICES=base_slices,
        #                                     # minmin=0, maxmax=100, type='Err')
        #                                     minmin = -15, maxmax = 15, type = 'Ecc')
        # # midcavity
        # plot_3x5grid_CMRxStrainxSectormasks(com_cube=com_cube[:,1,:],
        #                                     Radial_Morales=Ecc,
        #                                     masks_rot_Morales=masks_rot_lvmyo,
        #                                     vol_cube=vol_cube,
        #                                     sector_masks_raw=sector_masks_raw_midcavity,
        #                                     N_TIMESTEPS=N_TIMESTEPS,
        #                                     Z_SLICES=midcavity_slices,
        #                                     # minmin=0, maxmax=100, type='Err')
        #                                     minmin=-15, maxmax=15, type='Ecc')

        # # apex
        # plot_3x5grid_CMRxStrainxSectormasks(com_cube=com_cube[:,2,:],
        #                                     Radial_Morales=Ecc,
        #                                     masks_rot_Morales=masks_rot_lvmyo,
        #                                     vol_cube=vol_cube,
        #                                     sector_masks_raw=sector_masks_raw_apex,
        #                                     N_TIMESTEPS=N_TIMESTEPS,
        #                                     Z_SLICES=apex_slices,
        #                                     # minmin=0, maxmax=100, type='Err')
        #                                     minmin=-15, maxmax=15, type='Ecc')

        x = 0
        #
        # t=0
        # fig,ax=plt.subplots(nrows=1, ncols=len(base_slices))
        # for i,z_abs in enumerate(base_slices):
        #     ax[i].imshow(mask_whole[t,z_abs,...,0])
        #
        # plt.figure()
        # plt.imshow(mask_whole[0,44,...,0])
        #
        # t, z = (2, 32)
        # plt.figure()
        # plt.imshow(mask_whole[t,z,...,0], cmap='gray')
        # plt.scatter(RVIP_cube[t,z,0,0], RVIP_cube[t,z,0,1], label='ant')
        # plt.scatter(RVIP_cube[t, z, 1, 0], RVIP_cube[t, z, 1, 1], label='inf')
        # plt.legend()
        # plt.show()

        #
        # plt.figure()
        # plt.imshow(mask_whole[t,26,...,0])

        # validation plotting overlay cmr, Strainmap, masksrot, sectorrot
        # AHA=9
        # label_lvmyo=1
        # t=1
        # cmap_strain='jet'
        # z_rel=8
        # z_abs=24
        # plt.figure()
        # plt.imshow(Err[t,z_abs], cmap=cmap_strain)
        # # plt.colorbar()
        # plt.imshow(roll_to_center(vol_cube[t, z_abs, ..., 0], com_cube[t, 2], com_cube[t, 1]), cmap='gray', alpha=1)
        # # plt.imshow(masks_rot_lvmyo[t, z_abs], alpha=.5)
        # plt.imshow(
        #     Err[t, z_rel] * ((sector_masks_rot_midcavity[t, z_rel] == AHA) & (masks_rot_lvmyo[t, z_abs] == label_lvmyo)),
        #     cmap=cmap_strain,
        #     alpha=.5)
        # plt.imshow(((sector_masks_rot_midcavity[t, z_rel] == AHA) & (masks_rot_lvmyo[t, z_abs] == label_lvmyo)), cmap='gray', alpha=.5)
        # # plt.imshow(sector_masks_rot_midcavity[t, z_rel], alpha=.2)
        # # plt.scatter(com_cube[t, 1], com_cube[t, 2], label='com')
        # # plt.scatter(RVIP_cube[t, z_abs, 0, 0], RVIP_cube[t, z_abs, 0, 1], label='ant')
        # # plt.scatter(RVIP_cube[t, z_abs, 1, 0], RVIP_cube[t, z_abs, 1, 1], label='inf')
        # # plt.imshow(roll_to_center(mask_whole[t, z_abs, ..., 0], com_cube[t, 2], com_cube[t, 1]), alpha=.5, cmap='gray')
        # plt.legend()
        # plt.show()
        #
        # Err[t,z_rel][((sector_masks_rot_midcavity[t, z_rel] == AHA) & (masks_rot_lvmyo[t, z_abs] == label_lvmyo))].mean()
        # Err[t,z_rel,masks_rot_lvmyo[t, z_abs] == label_lvmyo].mean()
        #
        # Err[t, z_abs][(sector_masks_rot_midcavity[t, z_rel] == AHA) & (masks_rot_lvmyo[t, z_abs] == label_lvmyo)].mean()
        # AHAcube_midcavity[2,t,z_rel,0]
        #
        # AHA=1
        # z_abs=31
        # z_rel=0
        # Ecc[t, z_abs][(sector_masks_rot_base[t, z_rel] == AHA) & (masks_rot_lvmyo[t, z_abs] == label_lvmyo)].mean()
        # AHAcube_base[0,t,z_rel,1]
        #
        #
        #

        #

        # t = 0
        # z_abs = 21
        # cy, cx = (com_cube[t, 1], com_cube[t, 2])
        # plt.figure()
        # plt.imshow(roll_to_center(mask_whole[t, z_abs, ..., 0], cx, cy), label='mask_whole_rolledtocenter')
        # plt.imshow(masks_rot_lvmyo[t, z_abs], label='masks_rot_lvmyo', alpha=.5)
        # plt.show()

        #######define dataframe style here#######
        # df_style='time'#########################
        #######define dataframe style here#######

        if df_style == 'time':
            rs_overtime_base = AHAcube_base.mean(axis=2)[..., 0]
            cs_overtime_base = AHAcube_base.mean(axis=2)[..., 1]
            rs_overtime_mc = AHAcube_midcavity.mean(axis=2)[..., 0]
            cs_overtime_mc = AHAcube_midcavity.mean(axis=2)[..., 1]
            rs_overtime_apex = AHAcube_apex.mean(axis=2)[..., 0]
            cs_overtime_apex = AHAcube_apex.mean(axis=2)[..., 1]

            # 80,1 = 5 timesteps * 16 segments as column
            rs_AHA_overtime = np.concatenate((rs_overtime_base, rs_overtime_mc, rs_overtime_apex), axis=0).reshape(
                (16 * 5, 1))
            cs_AHA_overtime = np.concatenate((cs_overtime_base, cs_overtime_mc, cs_overtime_apex), axis=0).reshape(
                (16 * 5, 1))

            patientid = ([patient_name] * 16 * 5)
            ahano = np.repeat(range(1, 17), repeats=5, axis=0)
            phaseno = np.tile(range(0, 5), reps=(1, 16)).squeeze()

            # get soa and lge data
            df_cleandmd = pd.read_excel(io=path_to_metadata_xls, sheet_name=sheet_name_soalge, engine='openpyxl')
            soa = np.repeat(extract_segments(df_cleandmd[df_cleandmd['pat'] == patient_name]['soa'].values[0]),
                            repeats=5, axis=0)
            lge = np.repeat(extract_segments(df_cleandmd[df_cleandmd['pat'] == patient_name]['lgepos'].values[0]),
                            repeats=5, axis=0)

            # write df
            df_patient = pd.DataFrame(columns=['pat', 'aha', 'phase', 'our_rs', 'our_cs', 'soa', 'lge'])
            df_patient['pat'] = patientid
            df_patient['aha'] = ahano
            df_patient['phase'] = phaseno
            df_patient['our_rs'] = rs_AHA_overtime
            df_patient['our_cs'] = cs_AHA_overtime
            df_patient['soa'] = soa
            df_patient['lge'] = lge

        if df_style == 'peaks':
            # get Peak Err/Ecc values for all AHA segments
            # first, mean over all zslices
            # then, max/min value over time
            # get Peak Err/Ecc values for all AHA segments
            # first, mean over all zslices
            # then, max/min value over time
            PRS_values_base = 100 * AHAcube_base[..., 0].mean(axis=2).max(axis=1)
            PRS_values_midcavity = 100 * AHAcube_midcavity[..., 0].mean(axis=2).max(axis=1)
            PRS_values_apex = 100 * AHAcube_apex[..., 0].mean(axis=2).max(axis=1)
            our_prs = np.concatenate([PRS_values_base, PRS_values_midcavity, PRS_values_apex], axis=0)

            PCS_values_base = 100 * AHAcube_base[..., 1].mean(axis=2).min(axis=1)
            PCS_values_midcavity = 100 * AHAcube_midcavity[..., 1].mean(axis=2).min(axis=1)
            PCS_values_apex = 100 * AHAcube_apex[..., 1].mean(axis=2).min(axis=1)
            our_pcs = np.concatenate([PCS_values_base, PCS_values_midcavity, PCS_values_apex], axis=0)

            # pat, ahano
            patientid = ([patient_name] * 16)
            ahano = np.arange(1, 17)

            # get soa and LGE data
            df_cleandmd = pd.read_excel(io=path_to_metadata_xls, sheet_name=sheet_name_soalge, engine='openpyxl')
            soa = extract_segments(df_cleandmd[df_cleandmd['pat'] == patient_name]['soa'].values[0])
            lge = extract_segments(df_cleandmd[df_cleandmd['pat'] == patient_name]['lgepos'].values[0])

            # get cvi Circle Peak Strain values
            df_ahastrain = pd.read_excel(io=path_to_metadata_xls, sheet_name=sheet_name_ahastrain, index_col=0,
                                         header=0)
            cvi_prs = get_parameter_series_from_xls(df=df_ahastrain, parametername='radial peak strain (%)',
                                                    patientname=patient_name)
            cvi_pcs = get_parameter_series_from_xls(df=df_ahastrain, parametername='circumferential peak strain (%)',
                                                    patientname=patient_name)

            # write df
            df_patient = pd.DataFrame(columns=['pat', 'aha', 'cvi_prs', 'cvi_pcs', 'our_prs', 'our_pcs', 'soa', 'lge'])
            df_patient['pat'] = patientid
            df_patient['aha'] = ahano
            df_patient['cvi_prs'] = cvi_prs
            df_patient['cvi_pcs'] = cvi_pcs
            df_patient['our_prs'] = our_prs
            df_patient['our_pcs'] = our_pcs
            df_patient['soa'] = soa
            df_patient['lge'] = lge

        # append the patient df to the whole list of patients df
        df_patients.append(df_patient)

        # iteration info
        INFO(patient_name)

        x = 0
    # concatenate and write
    df_patients = pd.concat(df_patients, axis=0)
    df_filepath = '/home/julian/df_DMD_time.csv'  # df_DMD_peaks, df_DMD_time
    df_patients.to_csv(df_filepath, index=False)
    x = 0
    # # pairplot for peaks file
    import pandas as pd
    import seaborn as sns
    df = pd.read_csv(filepath_or_buffer='/home/julian/df_DMD_peaks.csv')
    g = sns.pairplot(df, hue='lge', hue_order=[0, 1])
    # g.axes[3,4].set_xlim((-15,40))#pcs
    # g.axes[3,4].set_ylim((0,200))#prs
    plt.tight_layout()
    #
    # # heatmap
    import numpy as np
    import matplotlib.pyplot as plt
    corr = df.corr(method='pearson')  # method{‘pearson’, ‘kendall’, ‘spearman’} or callable
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(10, 10))
        ax = sns.heatmap(corr, annot=True, cmap="coolwarm", fmt='.2f', mask=mask, square=True)
    #     # read AHA values from metadata xls table
    #     # i.e. read peak circumferential strain values
    #     # index_col and header have to be set to 0 to enable indexing by these column+line combination
    #     df = pd.read_excel(io=path_to_metadata_xls, sheet_name='DMD AHA Strain', index_col=0, header=0)
    #     Circle_AHA_PCS = get_parameter_series_from_xls(df=df, parametername='circumferential peak strain (%)', patientname=patient_name)
    #
    #     # plot bullseye
    #     myBullsplot(data=Circle_AHA_PCS, param='Ecc')
    #     myBullsplot(data=Our_AHA_PCS, param='Ecc')
    #
    #     # plot LGE series for this patient
    #     plot_1xX_LGE_series(path_to_lge_volume_nrrd='/mnt/ssd/julian/data/interim/2021.04.25_lgecorrected/',
    #                     patientname=patient_name,
    #                     fileending='_volume_clean.nrrd')
    #
    #     x=0
    #
    # plot strain curves per AHA
    # plt.figure(figsize=(8, 6))
    # ax_labels_ED_relative = ['ED ' + '$\longrightarrow$' + ' MS',
    #                          'ED ' + '$\longrightarrow$' + ' ES',
    #                          'ED ' + '$\longrightarrow$' + ' PF',
    #                          'ED ' + '$\longrightarrow$' + ' MD',
    #                          'ED ' + '$\longrightarrow$' + ' ED']
    # for idx, AHA in enumerate(N_AHA_base):
    #     plt.plot(ax_labels_ED_relative, 100 * AHAcube_base.mean(axis=2)[idx, :, 1], label=('AHA' + str(AHA)))
    # plt.ylim(-20, 50)
    # plt.xlabel('Phase')
    # plt.ylabel('Mean Ecc in %')
    # plt.legend()
    # plt.suptitle('Base')
    # plt.show()
    #
    #     # plot 3x5grid for AHA inspection
    #     # base
    #     plot_3x5grid_CMRxStrainxSectormasks(mask_whole=mask_whole,
    #                                         Radial_Morales=Ecc,
    #                                         masks_rot_Morales=masks_rot_lvmyo,
    #                                         vol_cube=vol_cube,
    #                                         sector_masks_raw=sector_masks_raw_base,
    #                                         N_TIMESTEPS=N_TIMESTEPS,
    #                                         Z_SLICES=base_slices,
    #                                         # minmin=0, maxmax=100, type='Err')
    #                                         minmin = -15, maxmax = 15, type = 'Ecc')
    # midcavity
    # plot_3x5grid_CMRxStrainxSectormasks(com_cube=com_cube,
    #                                     Radial_Morales=Ecc,
    #                                     masks_rot_Morales=masks_rot_lvmyo,
    #                                     vol_cube=vol_cube,
    #                                     sector_masks_raw=sector_masks_raw_midcavity,
    #                                     N_TIMESTEPS=N_TIMESTEPS,
    #                                     Z_SLICES=midcavity_slices,
    #                                     # minmin=0, maxmax=100, type='Err')
    #                                     minmin=-15, maxmax=15, type='Ecc')
    #     # apex
    #     plot_3x5grid_CMRxStrainxSectormasks(mask_whole=mask_whole,
    #                                         Radial_Morales=Ecc,
    #                                         masks_rot_Morales=masks_rot_lvmyo,
    #                                         vol_cube=vol_cube,
    #                                         sector_masks_raw=sector_masks_raw_apex,
    #                                         N_TIMESTEPS=N_TIMESTEPS,
    #                                         Z_SLICES=apex_slices,
    #                                         # minmin=0, maxmax=100, type='Err')
    #                                         minmin=-15, maxmax=15, type='Ecc')
    #
    #
    #
    #
    #
    #
    #
    #
    #     x=0
    #
    #
    #
    #
    #
    #     t = 0
    #     # z_rel = 5
    #     z_abs = Z_SLICES[0] + z_rel
    #     fig, axs = plt.subplots(4,4, sharey=True, sharex=True)
    #     ax = axs.flatten()
    #     for z_rel, z_abs in enumerate(Z_SLICES):
    #         cy, cx, _ = center_of_mass(mask_whole[t, z_abs, ...] == label_bloodpool)  # COM of slice
    #         ax[z_rel].imshow(roll_to_center(vol_cube[t, z_abs, :, :, 0], cy, cx), cmap='gray')
    #         ax[z_rel].imshow(sector_masks_rot[t, z_rel], alpha=.5)
    #         ax[z_rel].imshow(Err[t, z_rel], cmap='jet', alpha=.5)
    #         ax[z_rel].set_title('z='+str(z_abs))
    #     plt.show()
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #     t=0
    #     z_rel = 5
    #     cond = (sector_masks_rot==12) & (masks_rot_lvmyo == 1)
    #     plt.figure()
    #     plt.imshow(masks_rot_lvmyo[t, z_rel])
    #     plt.imshow(sector_masks_rot[t, z_rel], alpha=.5)
    #     plt.imshow((cond*Err)[t, z_rel], cmap='jet',alpha=.5)
    #     plt.show()
    #
    #
    #
    #     # get in same shape and right order
    #     t = 0
    #     z_rel = 5
    #     z_abs = Z_SLICES[0] + z_rel
    #     cy, cx, _ = center_of_mass(mask_whole[t, z_abs, ...] == label_bloodpool)  # COM of slice
    #     plt.figure()
    #     # plt.imshow(np.squeeze(vol_cube[t,z_abs,:,:])) #tzyx
    #     plt.imshow(roll_to_center(mask_whole[t, z_abs, :, :, 0], cy, cx), cmap='gray')
    #     plt.imshow(Err[t,z_rel], alpha=.5) #txyz
    #     plt.imshow(sector_masks_rot[t, z_rel],alpha=0.5)  # tzyx
    #     plt.show()
    #
    #
    #
    #
    #
    #     plt.figure()
    #     plt.imshow(Err[t,z_rel]*(sector_masks_rot[t,z_rel]==12))
    #     plt.show()
    #
    #     t = 2
    #     z_rel = 5
    #     z_abs = Z_SLICES[0] + z_rel
    #     cy, cx, _ = center_of_mass(mask_whole[t, z_abs, ...] == label_bloodpool)  # COM of slice
    #     RVIPup_glob, RVIPlow_glob = get_ip_from_mask_3d(mask_whole[t], debug=False, keepdim=True)
    #     fig,ax=plt.subplots(1,3,figsize=(12,5),sharey=True)
    #     ax[0].imshow(roll_to_center(mask_whole[t, z_abs, :, :, 0], cy, cx), cmap='gray')
    #     ax[0].imshow(roll_to_center(vol_cube[t, z_abs, :, :, 0], cy, cx), cmap='gray', alpha=.5)
    #     ax[0].set_title('CMR+Mask')
    #     ax[1].imshow(roll_to_center(vol_cube[t,z_abs,:,:,0],cy,cx), cmap='gray')
    #     ax[1].imshow(np.einsum('xy->yx',Radial_Sven[t,:,:,z_rel]), cmap='jet', alpha=.75)
    #     ax[1].set_title('CMR+Err')
    #     ax[2].imshow(roll_to_center(vol_cube[t, z_abs, :, :, 0], cy, cx), cmap='gray')
    #     ax[2].imshow(roll_to_center(sector_masks_raw[t,z_abs],cy,cx), alpha=.75)
    #     ax[2].set_title('CMR+AHA')
    #     fig.suptitle('roll_to_center',fontsize=20)
    #     plt.show()
    #
    #     t=2
    #     AHAquery=12
    #     plt.figure()
    #     plt.imshow(roll_to_center(vol_cube[t, z_abs, :, :, 0], cy, cx), cmap='gray')
    #     plt.imshow(np.einsum('xy->yx', Radial_Sven[t,...,z_rel]*(sector_masks_rot[t,...,z_rel]==AHAquery)),cmap='jet')
    #     plt.colorbar()
    #     plt.show()
    #
    #
    #     test = np.where(sector_masks_rot[t,...,z_rel]==8)
    #
    #     fig,ax=plt.subplots(1,2, sharey=True)
    #     ax[0].imshow(vol_cube[t,z_abs,...], cmap='gray')
    #     ax[0].scatter(cx,cy,label='COM')
    #     ax[0].set_title('patient volume raw')
    #     ax[1].imshow(roll_to_center(vol_cube[t,z_abs,...],cx,cy), cmap='gray')
    #     ax[1].scatter(64,64, label='COM')
    #     ax[1].set_title('patient volume rolledtocenter')
    #     plt.legend()
    #     plt.show()
    #
    #
    #
    #
    #     plt.figure()
    #     plt.imshow(vol_cube[0,Z_SLICES[0]+5,...])
    #     plt.imshow(sector_masks_raw[0,Z_SLICES[0]+5,...],alpha=.5)
    #     plt.show()
    #
    #     plt.figure()
    #     plt.imshow(sector_masks[0, ..., 5])
    #     plt.show()
    #
    #
    #
    #     # overlay
    #
    #     plt.figure()
    #     plt.imshow(roll_to_center(vol_cube[t, z_abs, ...],cx,cy),cmap='gray')
    #     plt.imshow(Radial_Sven[t, ..., z_rel], cmap='jet', alpha=0.5)
    #     plt.imshow(sector_masks_rot[t, ..., z_rel], alpha=0.5)
    #     plt.show()
    #
    #
    #     z=0
    #
    #
    #
    #     ######PLOTTING######
    #     # plot mean radial/circumferential strain curves wrt ED phase over all slices for both approaches
    #     # STRAIN_plot_composedflowfields(masks_rot_itk,
    #     #                                np.stack([Radial_Sven, Radial_itk, Radial_add]),
    #     #                                np.stack([Circumferential_Sven, Circumferential_itk, Circumferential_add]),
    #     #                                labels=['Sven', 'itk', 'add'],
    #     #                                colors=['k', 'r', 'g'])
    #
    #     INFO('Hi')
    #
    #
    #     # plot strain per slice if wanted
    #     # plot_strain_per_slice_Radial_Circumferential_side_by_side(Radial_itk, Circumferential_itk, masks_rot_itk,
    #     #                                                           N_TIMESTEPS, Z_SLICES, patientname=patients[0])
    #
    #     # plot segmentations with splines if wanted
    #     # plot_segmentation_grid_with_splines(seg_lvcube=seg_lvcube, vol_cube=vol_cube,
    #     #                                     zslices=Z_SLICES, ntimesteps=N_TIMESTEPS)
    #
    #     # plot segmentations if wanted
    #     # zslices = [34, 28, 22]
    #     # plot_segmentation_grid(seg_lvcube=seg_lvcube, zslices=zslices, ntimesteps=N_TIMESTEPS)
    #
    #
    #     itk_MCPRS = get_mean_strain_values_from_Morales(array=Radial_itk, masks=masks_rot_itk)
    #     itk_MCPCS = get_mean_strain_values_from_Morales(array=Circumferential_itk, masks=masks_rot_itk)
    #     add_MCPRS = get_mean_strain_values_from_Morales(array=Radial_add, masks=masks_rot_add)
    #     add_MCPCS = get_mean_strain_values_from_Morales(array=Circumferential_add, masks=masks_rot_add)
    #
    #     ###############WRITE RESULTS###############
    #     # gather ground truth parameter for comparison
    #     # create df which contains patients metadata
    #     df_groundtruth = pd.read_excel(io=path_to_metadata_xls, sheet_name='clean DMD', dtype='object', index_col=0)
    #     groundtruth_MCPRS = get_parameter_from_xls(dataframe=df_groundtruth, patientname=patient_name,
    #                                                parametername='mid-cavity peak radial strain')
    #     groundtruth_MCPCS = get_parameter_from_xls(dataframe=df_groundtruth, patientname=patient_name,
    #                                                parametername='mid-cavity peak circumferential strain')
    #
    #     # current results
    #     tobewritten = [patient_name,
    #                    groundtruth_MCPRS, itk_MCPRS.max(), add_MCPRS.max(),
    #                    groundtruth_MCPCS, itk_MCPCS.min(), add_MCPCS.min()]
    #
    #     # append to big list
    #     results.append(tobewritten)
    #
    #     INFO('patient ' + str(i+1) + ' / ' + str(len(patients)) + ' finished')
    #
    # # write the results list to a file
    # strain_folder = '/mnt/ssd/julian/data/outputs/strainresults/'
    # filename = '2021.08.19_OLDVERSION_' + patients[0] + '.txt'
    # with open(strain_folder + filename, 'w') as f:
    #     for item in results:
    #         f.write("%s\n" % item)
    #
    # # create DATAFRAME
    # # plot the results list as boxplot/violinplot
    # df = pd.DataFrame(results)
    # df.columns = ['ID',
    #               'Circle_MCPRS', 'itk_MCPRS', 'add_MCPRS',
    #               'Circle_MCPCS', 'itk_MCPCS', 'add_MCPCS']
    #
    # import seaborn as sns
    # sns.set_theme()
    # plt.figure()
    # ax = sns.boxplot(data=df)
    # ax = sns.violinplot(data=df)
    #
    # # inspecting the results df
    # import pandas as pd
    # path='/mnt/ssd/julian/data/outputs/strainresults/2021.08.18_flowfullgtMCPRSvsITKvsJUSTADDED.txt'
    # df = pd.read_csv(path)
    # df.columns = ['ID', 'Circle_MCPRS', 'itk_MCPRS', 'add_MCPRS']
    # # cut the square brackets from the data
    # df['ID'] = df['ID'].map(lambda x: x.lstrip('['))
    # df['add_MCPRS'] = df['add_MCPRS'].map(lambda x: x.rstrip(']'))
    # df['add_MCPRS'] = pd.to_numeric(df['add_MCPRS']) # convert to float for consecutive rounding
    # df.round(decimals=1)
    #
    # # plot correlation
    # import seaborn as sns
    # sns.set_theme()
    # ax=sns.scatterplot(data=df, x='Circle_MCPRS', y='itk_MCPRS', label='itk_MCPRS')
    # ax=sns.scatterplot(data=df, x='Circle_MCPRS', y='add_MCPRS', label='add_MCPRS')
    # ax.set_xlabel('Circle_MCPRS')
    # ax.set_ylabel('Second Method')
    # ax.set_title('Scatter Plot')
    #
    # import seaborn as sns
    # sns.set_theme()
    # ax=sns.scatterplot(data=df, x='Circle_MCPCS', y='itk_MCPCS', label='itk_MCPCS')
    # ax=sns.scatterplot(data=df, x='Circle_MCPCS', y='add_MCPCS', label='add_MCPCS')
    # ax.set_xlabel('Circle_MCPCS')
    # ax.set_ylabel('Second Method')
    # ax.set_title('Scatter Plot')
    #
    # sns.lmplot(x='Circle_MCPRS', y='itk_MCPRS', data=df)
    # plt.xlim(0, 50)
    # plt.ylim(0, 120)
    # plt.title('Circle vs itk')
    #
    # sns.lmplot(x='Circle_MCPRS', y='add_MCPRS', data=df)
    # plt.xlim(0, 50)
    # plt.ylim(0, 120)
    # plt.title('Circle vs add')
    #
    # sns.lmplot(x='Circle_MCPCS', y='itk_MCPCS', data=df)
    # plt.xlim(-5, -25)
    # plt.ylim(-15, 0)
    # plt.title('Circle vs itk')
    #
    # sns.lmplot(x='Circle_MCPCS', y='add_MCPCS', data=df)
    # plt.xlim(-5, -25)
    # plt.ylim(-15, 0)
    # plt.title('Circle vs add')
    #
    # # spearman R
    # from scipy import stats
    # stats.spearmanr(df['Circle_MCPRS'], df['itk_MCPRS'])
    # stats.spearmanr(df['Circle_MCPRS'], df['add_MCPRS'])
    #
    # from scipy import stats
    # stats.spearmanr(df['Circle_MCPCS'], df['itk_MCPCS'])
    # stats.spearmanr(df['Circle_MCPCS'], df['add_MCPCS'])
    #
    # # histograms
    # df['Circle_MCPRS'].hist()
    # df['itk_MCPRS'].hist()
    # df['add_MCPRS'].hist()
    #
    # df['Circle_MCPCS'].hist()
    # df['itk_MCPCS'].hist()
    # df['add_MCPCS'].hist()
    #
    # # seaborn qq plot to check if data is normally distributed
    # import statsmodels.api as sm
    # from scipy.stats import norm
    # import pylab
    # sm.qqplot(df['Circle_MCPRS'], line='s')
    # sm.qqplot(df['itk_MCPRS'], line='s')
    # sm.qqplot(df['add_MCPRS'], line='s')
    #
    # # seaborn qq plot to check if data is normally distributed
    # import statsmodels.api as sm
    # from scipy.stats import norm
    # import pylab
    # sm.qqplot(df['Circle_MCPCS'], line='s')
    # sm.qqplot(df['itk_MCPCS'], line='s')
    # sm.qqplot(df['add_MCPCS'], line='s')
    #
    # # pearson R
    # from scipy import stats
    # stats.pearsonr(df['Circle_MCPRS'], df['itk_MCPRS'])
    # stats.pearsonr(df['Circle_MCPRS'], df['add_MCPRS'])
    #
    # from scipy import stats
    # stats.pearsonr(df['Circle_MCPCS'], df['itk_MCPCS'])
    # stats.pearsonr(df['Circle_MCPCS'], df['add_MCPCS'])
    #
    #
    #
    #
    # # plot mean of single slice at given time step
    # t = 0
    # plt.figure()
    # plt.imshow(Radial_itk[t,...].mean(axis=-1), cmap='jet')
    # plt.colorbar()
    # plt.show()
    # plt.figure()
    # plt.imshow(Circumferential[t,...].mean(axis=-1), cmap='jet')
    # plt.colorbar()
    # plt.show()
    #
    # # identify min and max values first
    # vmax = Radial_itk.max()
    # vmin = Radial_itk.min()
    #
    # vmax = 1.5
    # vmin = -0.5
    # ax_labels=['ED', 'MS', 'ES', 'PF', 'MD', 'ED']
    # cmap = plt.cm.jet
    # # cmap.set_under(color='white')
    # fig, ax = plt.subplots(1,5,figsize=(12,8), sharey=True)
    # for t in range(N_TIMESTEPS):
    #     im = ax[t].imshow(Radial_itk.mean(axis=-1)[t], interpolation=None, cmap=cmap, vmin=vmin, vmax=vmax)
    #     ax[t].set_title(str(ax_labels[0]) + '$\longrightarrow$' + str(ax_labels[t+1]))
    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.35, 0.025, 0.3])
    # fig.colorbar(im, cax=cbar_ax)
    # plt.show()
    # ############OPTICAL FLOW FLOWFIELD CALCULATION############
    # # get the full flowfield
    # naming_flow_full = '_flow_full_'
    # ff_full = stack_nii_flowfield(path_to_patient_folder, naming_flow_full, N_TIMESTEPS)
    #
    # # get patient cmr data for opencv flowfield calculation
    # naming_cmr = '_cmr_'
    # vol_cube = stack_nii_volume(path_to_patient_folder, naming_cmr, N_TIMESTEPS)
    # vol_cube = append_array_entries(vol_cube)
    # vol_cube = np.roll(vol_cube, 1, axis=0)
    #
    #
    # vol_for_optflow = volume(data=vol_cube, format='4Dt', zspacing=Z_SPACING)
    # # flowfield_Farneback_phasephase, _ = input_volume_for_optical_flow.get_Farneback_flowfield_of_slice(slice=7, reference='dynamic')
    #
    # stack=np.ndarray((5,len(Z_SLICES),128,128,2))
    # bgr_stack=np.ndarray((5,len(Z_SLICES),128,128,3))
    #
    #
    # # pyr_scale: parameter, specifying the image scale (<1) to build pyramids for each image; pyr_scale=0.5
    # # means a classical pyramid, where each next layer is twice smaller than the previous one.
    # # levels: number of pyramid layers including the initial image; levels=1 means that
    # # no extra layers are created and only the original images are used.
    # # winsize: averaging window size; larger values increase the algorithm robustness to image noise and give more
    # # chances for fast motion detection, but yield more blurred motion field.
    # # iterations: number of iterations the algorithm does at each pyramid level.
    # # poly_n/poly_sigma = 5/1.1 or 7/1.5
    #
    # # pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags
    # parameters = [0.5, 10, 5, 1, 10, 5, 0]
    # for slice in range(len(Z_SLICES)):
    #     phasephase_Farneback, bgr_array = vol_for_optflow.get_Farneback_flowfield_of_slice(slice=slice, reference='dynamic', parameters=parameters)
    #     stack[:,slice,...] = phasephase_Farneback
    #     bgr_stack[:,slice,...] = bgr_array
    #
    #
    # new_stack = np.zeros((5,len(Z_SLICES),128,128,3))
    # new_stack[...,0] = stack[...,1] #x
    # new_stack[...,1] = stack[...,0] #y
    # new_stack[...,2] = 0 #z
    # # z remains zero
    #
    # nrows = 2
    # ncols = 5
    # fig,ax=plt.subplots(nrows, ncols, figsize=(15, 8), sharex=True, sharey=True)
    # for t in range(N_TIMESTEPS):
    #     ax[0,t].imshow(vol_cube[t, Z_SLICES[0],...], cmap='gray')
    #     ax[1,t].imshow(bgr_stack[t, 0, ...])
    #     # ax[2,t].imshow(new_stack[t,0,...])
    # plt.show()
    #
    # import os, sys
    # # third party imports
    # import numpy as np
    # import tensorflow as tf
    # assert tf.__version__.startswith('2.'), 'This tutorial assumes Tensorflow 2.0+'
    # import voxelmorph as vxm
    # import neurite as ne
    # N=1
    # # 0:2 means take the first and the second entries (order xyz then, if xy plotting wanted)
    # ne.plot.flow(slices_in=[new_stack[1, 0, ::N, ::N, 0:2]], titles=['openCV dense optical flow'], width=5, scale=1)
    # ne.plot.slices(slices_in=[np.squeeze(vol_cube[1, Z_SLICES[0],...])], width=5)
    # # ne.plot.flow(slices_in=[switch_channel(flowfield_unswitched=ff_full)[0, Z_SLICES[0], ::N, ::N, 0:2]], titles=['Sven full flowfield'], width=5, scale=1)
    #
    # plt.figure()
    # plt.imshow(vol_cube[1, Z_SLICES[0],...], cmap='gray')
    # plt.show()
    #
    # ne.plot.slices(slices_in=[np.squeeze(vol_cube[0, Z_SLICES[0],...])], width=5)
    #
    # INFO('Hi')
    #
    #
    #
    #
    #
    #
    #
    # ff_comp_itk_Farneback = compose_ff_withsitk(ff=new_stack, Z_SLICES=Z_SLICES) # itk needs zyxc with c=xyz
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    # #######phantom data############
    # # create sample masks
    # # phantom = get_phantom_masks(N_TIMESTEPS=N_TIMESTEPS)
    #
    # # create sample flowfield
    # # phantom_ff = np.ndarray((5,64,128,128,3))
    # # phantom_ff[...,0] = 0 #x
    # # phantom_ff[...,1] = 0 #y
    # # phantom_ff[...,2] = 0 #z
    #
    # # change one vector at 49/62 which is onto the LVM
    # # phantom_ff[2,23,49,62,1] = 5
    #
    # # create vector field where arrows point outwards
    # # x,y = np.meshgrid(np.linspace(0,127,128),np.linspace(0,127,128))
    #
    # # variant 1
    # # u = x/np.sqrt(x**2 + y**2)
    # # v = y/np.sqrt(x**2 + y**2)
    #
    # # variant 2 doesnt work
    # # u = np.cos(x)
    # # v = np.sin(y)
    #
    # # variant 3 doesnt work
    # # u, v = np.meshgrid(x, y)
    #
    # # phantom_ff[...,0] = u
    # # phantom_ff[...,1] = v
    #
    #
    # # seg_lvcube = phantom
    # # ff_switch = phantom_ff
    # #######phantom data############
    #
    #
    #
    # # we will only take the middle mid cavity slice for the beginning now
    # # mask has to be 128,128,atleast2
    # # flowfield has to be 128,128,atleast2,3


def main(data_root='/mnt/sds-hd/sd20i001/julian/v12', debug=True):
    # global Err, Ecc, corr, ax
    print('a')
    temp = np.zeros(10)
    calculate_strain()


if __name__ == "__main__":
    # set up logging
    Console_and_file_logger('mvfviz/dmd_temp', logging.INFO)
    main()
