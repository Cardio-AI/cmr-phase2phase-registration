# define logging and working directory
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import SimpleITK
from ProjectRoot import change_wd_to_project_root

# import helper functions
import matplotlib.pyplot as plt
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import logging, os

from logging import info as INFO

import src_julian.utils.mystrainfunctions
from src_julian.utils.myAHAfunctions import *
from src_julian.utils.myclasses import mvf, volume
from src_julian.utils.myhelperfunctions import stack_nii_volume, stack_nii_masks, stack_nii_flowfield
from src_julian.utils.mystrainfunctions import get_volumeborders
from src_julian.utils.skhelperfunctions import Console_and_file_logger, extract_segments

def plotff(ff, t, z_abs, N):
    nx, ny = (ff.shape[3], ff.shape[2])
    X, Y = np.mgrid[0:nx, 0:ny]
    X, Y = (X[::N, ::N], Y[::N, ::N])
    U = ff[t, z_abs, ::N, ::N, 2]  # x
    V = ff[t, z_abs, ::N, ::N, 1]
    return X, Y, U, V

def plot_1x5_quiver_flowfield(ff, mask_lvmyo, z_abs, N=1, crop=None, style=None, headwidth=3, headlength=5, width=1):
    xmin, xmax, ymin, ymax = (crop[0], crop[1], crop[2], crop[3])
    nt = ff.shape[0]


    if style == 'p2p':
        xlabels = ['ED-MS', 'MS-ES', 'ES-PF', 'PF-MD', 'MD-ED']
    elif style == 'comp':
        xlabels = ['ED-MS', 'ED-ES', 'ED-PF', 'ED-MD', 'ED-ED']


    fig, ax = plt.subplots(1, nt, figsize=(12, 10))
    for t in range(nt):
        if style == 'p2p':
            ax[t].imshow(mask_lvmyo[t, z_abs, ..., 0], cmap='Wistia')
        elif style == 'comp':
            ax[t].imshow(mask_lvmyo[0, z_abs, ..., 0], cmap='Wistia') # only plot ED mask
        X, Y, U, V = plotff(ff=ff, t=t, z_abs=z_abs, N=N)
        ax[t].quiver(Y, X, U, V, units='xy', angles='xy', scale=.75, color='k', headwidth=headwidth,
                     headlength=headlength, width=width)
        ax[t].set_title(xlabels[t], fontsize=20)
        ax[t].set_xlim(xmin, xmax)
        ax[t].set_ylim(ymax, ymin)
        ax[t].set_aspect('equal')
    plt.setp(ax, xticks=[], yticks=[])
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()

def calculate_strain(data_root='', metadata_path='/mnt/ssd/julian/data/metadata/', debug=False, df_style=None, p2p_style=True, isDMD=True):
    patient_folders = sorted(glob.glob(os.path.join(data_root, 'pred/*/'))) # usual for the DMD crosses
    if len(patient_folders)==0:
        patient_folders = sorted(glob.glob(os.path.join(data_root, 'pred/*/')))  # for the healthy folders

    sorting_lambda = lambda x: os.path.basename(os.path.dirname(x))
    patient_folders = sorted(patient_folders, key=sorting_lambda)
    patient_folders = [p for p in patient_folders]


    # define debug mode for a specific patient only
    # DMD LGE+ and LGE-
    # DMD LGE+
    # hh_20190621
    # cf_20191009
    # ct_20191108
    # rg_20180919
    # me_20190816

    # DMD LGE-
    # ab_20180710
    # dc_20180516
    # ha_20190301
    # la_20190503
    # pm_20190801

    # Healthy Controls:
    # aj_20191018
    # bb_20190627
    # bl_20190122
    # cc_20191029
    # cf_20181119

    debug_patient = 'cf_20181119'
    if debug:
        patient_folders = [p for p in patient_folders if debug_patient in p]

    ###initial inits###
    import json
    try:
        cfg = os.path.join(data_root, 'f0', 'config/config.json')
        print('config given: {}'.format(cfg))
        # load the experiment config
        with open(cfg, encoding='utf-8') as data_file:
            config = json.loads(data_file.read())
    except Exception as e: # maybe no fold subfolders
        print(e)
        print('data_root: {}'.format(data_root))
        cfg = os.path.join(data_root, 'config/config.json')
        print('config given: {}'.format(cfg))
        # load the experiment config
        with open(cfg, encoding='utf-8') as data_file:
            config = json.loads(data_file.read())
    spacing_vol = list(reversed(config.get('SPACING')))
    register_backwards = config.get('REGISTER_BACKWARDS')

    df_patients = []  # df where we will store our results
    metadata_filename = 'DMDTarique_3.0.xlsx'
    RVIP_method = 'dynamically'   # staticED (standard), dynamically
    com_method = 'dynamically'  # dynamically (standard), staticED

    N_TIMESTEPS = 5
    Z_SPACING = spacing_vol[-1]
    label_bloodpool = 3
    label_lvmyo = 1
    executor = ThreadPoolExecutor(max_workers=12)

    if isDMD:
        sheet_name_soalge = 'clean DMD'
        sheet_name_ahastrain = 'DMD AHA Strain'
    else:
        sheet_name_soalge = 'clean Control'
        sheet_name_ahastrain = 'Control AHA Strain'

    if p2p_style:
        ff_style='p2p'
    else:
        ff_style='ed2p'

    # compose metadata xls filepath
    path_to_metadata_xls = os.path.join(metadata_path, metadata_filename)
    df_dmdahastrain = pd.read_excel(io=path_to_metadata_xls, sheet_name=sheet_name_ahastrain, index_col=0, header=0,engine='openpyxl')
    df_cleandmd = pd.read_excel(io=path_to_metadata_xls, sheet_name=sheet_name_soalge, engine='openpyxl')

    pats = len(patient_folders)
    params = [N_TIMESTEPS, RVIP_method, Z_SPACING, com_method, df_style, ff_style,
                                               label_bloodpool, p2p_style, path_to_metadata_xls,
                                               df_dmdahastrain, df_cleandmd, spacing_vol, register_backwards]
    params = [[elem] * pats for elem in params]

    for result in executor.map(calc_strain4singlepatient, patient_folders, *params):
        df_patients.append(result)
    df_patients = pd.concat(df_patients, axis=0, ignore_index=True)
    df_patients.sort_values(by=['pat', 'aha'], inplace=True)

    return df_patients


def calc_strain4singlepatient(path_to_patient_folder, N_TIMESTEPS, RVIP_method, Z_SPACING, com_method, df_style, ff_style, label_bloodpool,
                              p2p_style, path_to_metadata_xls, df_dmdahastrain,
                              df_cleandmd, spacing, register_backwards):
    '''

    Parameters
    ----------
    path_to_patient_folder :
    N_TIMESTEPS :
    RVIP_method :
    Z_SPACING :
    com_method :
    df_style :
    ff_style :
    label_bloodpool :
    p2p_style :
    path_to_metadata_xls :
    df_dmdahastrain :
    df_cleandmd :
    spacing :
    register_backwards :

    Returns
    -------

    '''
    patient_name = os.path.basename(os.path.dirname(path_to_patient_folder))
    if ff_style=='p2p':
        RVIP_method = 'dynamically'  # staticED (standard), dynamically
        com_method = 'dynamically'  # dynamically (standard), staticED
    elif ff_style=='ed2p':
        RVIP_method = 'staticED'  # staticED (standard), dynamically
        com_method = 'staticED'  # dynamically (standard), staticED
    else:
        raise NotImplementedError('ffstyle : {} not supported'.format(ff_style))

    RVIP_method = 'dynamically'  # staticED (standard), dynamically
    com_method = 'dynamically'  # dynamically (standard), staticED


    # patient_name = os.path.basename(path_to_patient_folder) #test21.10.21
    # iteration info
    INFO('now processing: ' + patient_name)
    # overwrite the config spacing
    import SimpleITK as sitk
    temp = sitk.ReadImage(sorted(glob.glob(os.path.join(path_to_patient_folder, '*fullmask*.nii')))[0])
    spacing = temp.GetSpacing()
    Z_SPACING = spacing[-1] # (1.5,1.5,2.5) x,y,z sitk representation
    # CMR of patient
    #vol_cube = stack_nii_volume(path_to_patient_folder, 'cmr_moving_', N_TIMESTEPS)  # refactored
    # LV MYO MASKS
    mask_lvmyo = stack_nii_masks(path_to_patient_folder, 'myo_moving_', N_TIMESTEPS)  # moving: ED,MS, ES, PF, MD
    mask_lvmyo = mask_lvmyo>0.1
    #mask_lvmyo = np.roll(mask_lvmyo, shift=1, axis=0)
    # WHOLE MASKS
    mask_whole = stack_nii_masks(path_to_patient_folder, 'fullmask_moving_', N_TIMESTEPS)  # ED,MS, ES, PF, MD
    #mask_whole = np.roll(mask_whole, shift=1, axis=0)
    # FULL FLOWFIELD PHASE-PHASE
    # ff stored is of shape t x cxyz with c=zyx <-- need to verify if this is now still the case
    idx_ed = 0
    if p2p_style:
        ff_whole = stack_nii_flowfield(path_to_patient_folder, 'flow_', N_TIMESTEPS)
    else:
        ff_whole = stack_nii_flowfield(path_to_patient_folder, 'flow_composed_', N_TIMESTEPS)

    # Mask: MD,ED,MS,ES,PF - TZYXC
    # Flow: MD-ED,ED-MS,MS-ES,ES-PF,PF-MD - TZYXC
    # Flow composed: ED-ED,ED-MS,ED-ES,ED-PF,ED-MD - TZYXC

    shape_ = ff_whole.shape
    dim_ = ff_whole.ndim
    nt, nz, nx, ny, nc = shape_

    # remove the most apical and basal slices, as they often are wrong
    # use different borders depending on t, as the heart size changes
    border = 1
    for t in range(mask_lvmyo.shape[0]):
        mask_given = np.argwhere(mask_lvmyo[t].sum(axis=(1, 2)) > 0)
        mask_lvmyo[t, 0:int(mask_given[border])] = 0
        mask_lvmyo[t, int(mask_given[-border]):] = 0
        mask_whole[t,0:int(mask_given[border])] = 0
        mask_whole[t, int(mask_given[-border]):] = 0

    # get the indexes showing the lvmyo on the ED keyframe
    lvmyo_idxs = np.argwhere(mask_lvmyo[idx_ed].sum(axis=(1,2)) > 0)[:,0]
    if len(lvmyo_idxs) == 0:
        print(patient_name)

    # the most basal and most apical myom ask is often not reliable, we zero them out
    base_slices, midcavity_slices, apex_slices = get_volumeborders(lvmyo_idxs)  # by lvmyo-range

    if len(base_slices)==0 or len(midcavity_slices)==0 or len(apex_slices)==0:
        raise NotImplementedError('some areas are empty')

    # c = y,x
    # dynamically: every timestep gets different mean RVIPs for base mid cavity apex
    # staticED: every timestep contains the same mean RVIPs for base mid cavity apex of ED
    # c = z,y,x
    RVIP_cube = calculate_RVIP_cube(mask_whole, base_slices, midcavity_slices, apex_slices, method=RVIP_method, idx_ed=idx_ed)
    # CALCULATE COM CUBE 5x3x3
    # base, midcavity, apex : axis=1
    # c = z,y,x
    com_cube = calculate_center_of_mass_cube(mask_whole, label_bloodpool=label_bloodpool,
                                             base_slices=base_slices, midcavity_slices=midcavity_slices,
                                             apex_slices=apex_slices, method=com_method, idx_ed=idx_ed)
    # Morales expects a shape of: x,y,z, also for the channels
    # C of ff should be z,y,x such as the axis, invert them
    ff_Moralesinput = np.stack([ff_whole[..., 2],
                         ff_whole[..., 1],
                         ff_whole[..., 0]], axis=-1)

    ff_Moralesinput = np.einsum('tzyxc->txyzc', ff_Moralesinput)
    mask_lvmyo_Moralesinput = np.einsum('tzyxc->txyzc', mask_lvmyo[...,None])

    # for roll_to_center to work, we have to give t,c with c=x,y,z into Morales
    # DeepStrain will not make use of cz, so only the first two entries are relevant
    com_cube_Moralesinput = np.zeros_like(com_cube)
    com_cube_Moralesinput[..., 0] = com_cube[..., 2]
    com_cube_Moralesinput[..., 1] = com_cube[..., 1]
    com_cube_Moralesinput[..., 2] = com_cube[..., 0]
    # validation plot: plot mask and center of mass before DeepStrain call
    # t, z = (2, 24)
    # plt.figure()
    # plt.imshow(mask_lvmyo_Moralesinput[t,:,:,z,0], cmap='gray')
    # plt.scatter(com_cube_Moralesinput[t,1,1], com_cube_Moralesinput[t,1,0], label='com')
    # plt.show()
    # DEEPSTRAIN
    # we call the algorithm three times; for base, mc, apex, then put the results together
    Radial_Sven_base, \
    Circumferential_Sven_base, \
    masks_rot_Sven_base = myMorales(ff=ff_Moralesinput[:, :, :, base_slices],
                                    mask_lvmyo=mask_lvmyo_Moralesinput[:, :, :, base_slices],
                                    com_cube=com_cube_Moralesinput[:, 0, :],
                                    spacing=spacing,
                                    method=ff_style,
                                    reg_backwards=register_backwards,
                                    idx_ed=idx_ed)
    Radial_Sven_mc, \
    Circumferential_Sven_mc, \
    masks_rot_Sven_mc = myMorales(ff=ff_Moralesinput[:, :, :, midcavity_slices],
                                  mask_lvmyo=mask_lvmyo_Moralesinput[:, :, :, midcavity_slices],
                                  com_cube=com_cube_Moralesinput[:, 1, :],
                                  spacing=spacing,
                                  method=ff_style,
                                  reg_backwards=register_backwards,
                                    idx_ed=idx_ed)
    Radial_Sven_apex, \
    Circumferential_Sven_apex, \
    masks_rot_Sven_apex = myMorales(ff=ff_Moralesinput[:, :, :, apex_slices],
                                    mask_lvmyo=mask_lvmyo_Moralesinput[:, :, :, apex_slices],
                                    com_cube=com_cube_Moralesinput[:, 2, :],
                                    spacing=spacing,
                                    method=ff_style,
                                    reg_backwards=register_backwards,
                                    idx_ed=idx_ed)
    # stack the information together
    # create whole volume arrays with stacked information contained
    Err = np.zeros((nt, nx, ny, nz))
    Err[..., base_slices] = Radial_Sven_base
    Err[..., midcavity_slices] = Radial_Sven_mc
    Err[..., apex_slices] = Radial_Sven_apex
    Ecc = np.zeros((nt, nx, ny, nz))
    Ecc[..., base_slices] = Circumferential_Sven_base
    Ecc[..., midcavity_slices] = Circumferential_Sven_mc
    Ecc[..., apex_slices] = Circumferential_Sven_apex
    masks_rot_lvmyo = np.zeros((nt, nx, ny, nz))
    masks_rot_lvmyo[..., base_slices] = masks_rot_Sven_base
    masks_rot_lvmyo[..., midcavity_slices] = masks_rot_Sven_mc
    masks_rot_lvmyo[..., apex_slices] = masks_rot_Sven_apex

    if max(base_slices)==Err.shape[-1]:
        print(base_slices)

    # Rearrange Strain Tensor for further processing
    Err = np.einsum('txyz->tzyx', Err)
    Ecc = np.einsum('txyz->tzyx', Ecc)
    masks_rot_lvmyo = np.einsum('txyz->tzyx', masks_rot_lvmyo)

    ######AHA DIVISIONS#####
    # calculate array of sector masks with the same shape as the whole 4DCMR
    # entries only where slices are given
    sector_masks_raw_base = calculate_sector_masks(mask_whole=mask_whole[:,base_slices],
                                                   com_cube=com_cube[:, 0, :],
                                                   RVIP_cube=RVIP_cube[:,base_slices],
                                                   level='base')
    sector_masks_raw_midcavity = calculate_sector_masks(mask_whole=mask_whole[:,midcavity_slices],
                                                        com_cube=com_cube[:, 1, :],
                                                        RVIP_cube=RVIP_cube[:,midcavity_slices],
                                                        level='mid-cavity')
    sector_masks_raw_apex = calculate_sector_masks(mask_whole=mask_whole[:,apex_slices],
                                                   com_cube=com_cube[:, 2, :],
                                                   RVIP_cube=RVIP_cube[:,apex_slices],
                                                   level='apex')

    # validation plots for sector masks raw
    # t=3
    # z=24
    # anty, antx = RVIP_cube[t, z, 0]
    # infy, infx = RVIP_cube[t, z, 1]
    # plt.imshow(vol_cube[t,z], cmap='gray')
    # plt.imshow(sector_masks_raw_midcavity[t,z],alpha=.5)
    # plt.scatter(com_cube[t,1,1], com_cube[t,1,2], label='com')
    # plt.scatter(anty, antx, label='ant')
    # plt.scatter(infy, infx, label='inf')
    # plt.legend()
    # plt.show()
    # roll sector masks to center for overlay plotting with Strain Maps
    sector_masks_rot_base = roll_sector_mask_to_bloodpool_center(sector_mask_raw=sector_masks_raw_base,
                                                                 com_cube=com_cube[:, 0, :])
    sector_masks_rot_midcavity = roll_sector_mask_to_bloodpool_center(sector_mask_raw=sector_masks_raw_midcavity,
                                                                      com_cube=com_cube[:, 1, :])
    sector_masks_rot_apex = roll_sector_mask_to_bloodpool_center(sector_mask_raw=sector_masks_raw_apex,
                                                                 com_cube=com_cube[:, 2, :])
    # it seems that the secor mask has a shape of: t,z,x,y
    # roll_sector_mask_to_bloodpool_center uses the function "roll_to_center" from morales
    # This method expects an axis order of x,y
    # we keep that part and change the axis afterwards.

    """sector_masks_rot_base = np.einsum('tzxy->tzyx', sector_masks_rot_base)
    sector_masks_rot_midcavity = np.einsum('tzxy->tzyx', sector_masks_rot_midcavity)
    sector_masks_rot_apex = np.einsum('tzxy->tzyx', sector_masks_rot_apex)"""

    quantile = .95
    msk_heart = (masks_rot_lvmyo == 1)
    Err[msk_heart] = clip_quantile(Err[msk_heart], q=quantile)
    Ecc[msk_heart] = clip_quantile(Ecc[msk_heart], q=quantile)


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
                                      N_AHA=N_AHA_base, p2p_style=p2p_style)
    AHAcube_midcavity = calculate_AHA_cube(Err=Err[:, midcavity_slices], Ecc=Ecc[:, midcavity_slices],
                                           sector_masks_rot=sector_masks_rot_midcavity,
                                           masks_rot=masks_rot_lvmyo[:, midcavity_slices],
                                           Z_SLICES=midcavity_slices,
                                           N_AHA=N_AHA_midcavity, p2p_style=p2p_style)
    AHAcube_apex = calculate_AHA_cube(Err=Err[:, apex_slices], Ecc=Ecc[:, apex_slices],
                                      sector_masks_rot=sector_masks_rot_apex,
                                      masks_rot=masks_rot_lvmyo[:, apex_slices],
                                      Z_SLICES=apex_slices,
                                      N_AHA=N_AHA_apex, p2p_style=p2p_style)


    rs_overtime_base = AHAcube_base[..., 0]
    cs_overtime_base = AHAcube_base[..., 1]
    rs_overtime_mc = AHAcube_midcavity[..., 0]
    cs_overtime_mc = AHAcube_midcavity[..., 1]
    rs_overtime_apex = AHAcube_apex[..., 0]
    cs_overtime_apex = AHAcube_apex[..., 1]

    # runtime outputs
    # output min max mean strain for patient; Err and Ecc
    rs_overtime_base = np.nanmean(rs_overtime_base, axis=2)
    cs_overtime_base = np.nanmean(cs_overtime_base, axis=2)
    rs_overtime_mc = np.nanmean(rs_overtime_mc, axis=2)
    cs_overtime_mc = np.nanmean(cs_overtime_mc, axis=2)
    rs_overtime_apex = np.nanmean(rs_overtime_apex, axis=2)
    cs_overtime_apex = np.nanmean(cs_overtime_apex, axis=2)
    # 80,1 = 5 timesteps * 16 segments as column

    rs_AHA_overtime = np.concatenate((rs_overtime_base, rs_overtime_mc, rs_overtime_apex), axis=0).reshape(
        (16 * 5, 1))
    cs_AHA_overtime = np.concatenate((cs_overtime_base, cs_overtime_mc, cs_overtime_apex), axis=0).reshape(
        (16 * 5, 1))



    #
    # rs_AHA_overtime = np.nan_to_num(rs_AHA_overtime)
    # cs_AHA_overtime = np.nan_to_num(cs_AHA_overtime)
    if (np.isnan(rs_AHA_overtime).any() or np.isnan(cs_AHA_overtime).any()):
        print('Some AHA segments have NaN values, please check!')
        #raise NotImplementedError('Some AHA segments have NaN values, please check!')
    INFO('Strain method: {}'.format(ff_style))
    INFO('Err min: {:3.1f}%, max: {:3.1f}%, mean: {:3.1f}%'.format(100 * rs_AHA_overtime.min(),
                                                                   100 * rs_AHA_overtime.max(),
                                                                   100 * rs_AHA_overtime.mean()))
    INFO('Ecc min: {:3.1f}%, max: {:3.1f}%, mean: {:3.1f}%'.format(100 * cs_AHA_overtime.min(),
                                                                   100 * cs_AHA_overtime.max(),
                                                                   100 * cs_AHA_overtime.mean()))

    ######PLOTTING######
    x = 0
    # plot cmr overlay strain map masked
    # plot_3x5_cmroverlaywithmasked_strainormagnitude(ff_whole_itk, Err, Ecc, base_slices, midcavity_slices,
    #                                                 apex_slices, vol_cube, com_cube, masks_rot_lvmyo, method='Ecc')
    # new visalization method ONEVIEW
    # plot_4x5_MaskQuiver_Magnitude_Err_Ecc(ff_composed=ff_whole, mask_whole=masks_rot_lvmyo, Err=Err, Ecc=Ecc, z=24,N=1)
    # ff_composed=ff_whole_Sven
    # N=1
    # nx,ny=128,128
    # t,z = 2,24
    # plt.figure()
    # plt.imshow(mask_whole[t,z]==2)
    # X, Y = np.mgrid[0:nx, 0:ny]
    # X, Y = (X[::N, ::N], Y[::N, ::N])
    # U = ff_composed[t, z, ::N, ::N, 2]  # x
    # V = ff_composed[t, z, ::N, ::N, 1]  # y
    # # ax[0, t].imshow(mask_whole[0, z, ..., 0])
    # plt.quiver(Y, X, U, V, units='xy', angles='xy', scale=1, color='k')
    # plot strain per slice, side by side Err and Ecc, rainbow plot
    # src_julian.utils.mystrainfunctions.plot_strain_per_slice_Radial_Circumferential_side_by_side(Radial_itk=Radial_Sven,
    #                                                                                       Circumferential_itk=Circumferential_Sven,
    #                                                                                       masks_rot_itk=masks_rot_Sven,
    #                                                                                       N_TIMESTEPS=5,
    #                                                                                       Z_SLICES=lvmyo_idxs,
    #                                                                                       patientname=patient_name)
    # plot masks and strain by side to inspect masks orientation and placements
    # import matplotlib
    # t,z=(1,24)
    # fig, ax=plt.subplots(1,5)
    # ax[0].imshow(mask_whole[t,z], cmap='gray')
    # ax[0].set_title('mask_whole')
    # ax[1].imshow(mask_lvmyo[t, z], cmap='gray')
    # ax[1].set_title('mask_lvmyo')
    # ax[2].imshow(masks_rot_lvmyo[t, z], cmap='gray')
    # ax[2].set_title('masks_rot_lvmyo')
    # ax[3].imshow(Err[t,z],cmap='inferno')
    # ax[3].set_title('Err')
    # matplotlib.cm.spring.set_bad(color='white')
    # ax[4].imshow(np.ma.masked_where(Err[t,z] == 0, Err[t,z]), cmap='spring')
    # ax[4].set_title('Err masked')
    x=0
    # # base
    # plot_3x5grid_CMRxStrainxSectormasks(com_cube=com_cube[:,0,:],
    #                                     Radial_Morales=Err,
    #                                     masks_rot_Morales=masks_rot_lvmyo,
    #                                     vol_cube=vol_cube,
    #                                     sector_masks_raw=sector_masks_raw_base,
    #                                     N_TIMESTEPS=N_TIMESTEPS,
    #                                     Z_SLICES=base_slices,
    #                                     minmin=0, maxmax=100, type='Err')
    # minmin = -15, maxmax = 15, type = 'Ecc')
    # # # midcavity
    # plot_3x5grid_CMRxStrainxSectormasks(com_cube=com_cube[:,1,:],
    #                                     Radial_Morales=Err,
    #                                     masks_rot_Morales=masks_rot_lvmyo,
    #                                     vol_cube=vol_cube,
    #                                     sector_masks_raw=sector_masks_raw_midcavity,
    #                                     N_TIMESTEPS=N_TIMESTEPS,
    #                                     Z_SLICES=midcavity_slices,
    #                                     minmin=0, maxmax=100, type='Err')
    #                                     # minmin=-15, maxmax=15, type='Ecc')
    # # # apex
    # plot_3x5grid_CMRxStrainxSectormasks(com_cube=com_cube[:,2,:],
    #                                     Radial_Morales=Err,
    #                                     masks_rot_Morales=masks_rot_lvmyo,
    #                                     vol_cube=vol_cube,
    #                                     sector_masks_raw=sector_masks_raw_apex,
    #                                     N_TIMESTEPS=N_TIMESTEPS,
    #                                     Z_SLICES=apex_slices,
    #                                     minmin=0, maxmax=100, type='Err')
    #                                     # minmin=-15, maxmax=15, type='Ecc')
    # get cvi Circle Peak Strain values for current patient
    # the arrays contain 16 values each; for every AHA segment
    cvi_given = False
    try:
        cvi_given = True
        INFO('metadata loaded, cvi_given={}'.format(cvi_given))
    except Exception as e:
        print(e)
        cvi_given = False

    if cvi_given:cvi_prs = get_parameter_series_from_xls(df=df_dmdahastrain, parametername='radial peak strain (%)',
                                            patientname=patient_name)
    if cvi_given: cvi_pcs = get_parameter_series_from_xls(df=df_dmdahastrain, parametername='circumferential peak strain (%)',
                                            patientname=patient_name)
    # BULLSPLOTS
    # CIRCLE
    # myBullsplot(data=cvi_prs, param='Err')
    # myBullsplot(data=cvi_pcs, param='Ecc')
    # plot curves per AHA and heart base mc apex
    # plot_3x2_AHAStrainMotioncurvesOvertime(AHAcube_base, AHAcube_midcavity, AHAcube_apex,
    # Err_min=-10, Err_max=100, Ecc_min=-15, Ecc_max=20)
    # Bullsplot MINMAX OVER ALL PHASES
    # plot_Bullsplots_minmax_over_all_phases(AHAcube_base, AHAcube_midcavity, AHAcube_apex, cvi_prs, cvi_pcs)
    x = 0
    # Bullsplot SPECIFIC PHASE
    # specific_phase = 1 # ES=1
    # plot_Bullsplots_minmax_at_specific_phase(AHAcube_base, AHAcube_midcavity, AHAcube_apex, specific_phase)
    # plot LGE series for current patient
    # plot_1xX_LGE_series(path_to_lge_volume_nrrd='/mnt/ssd/julian/data/interim/2021.04.25_lgecorrected/',
    #                 patientname=patient_name,
    #                 fileending='_volume_clean.nrrd',
    #                 crop=None)
    # plot all segmentations on a grid for specified levels
    # zslices = [34, 28, 22]
    # src_julian.utils.mystrainfunctions.plot_segmentation_grid(seg_lvcube=mask_lvmyo, zslices=zslices, ntimesteps=N_TIMESTEPS)
    # write data to dataframe
    if df_style == 'time':
        # get soa and lge data for current patient from metadata xls
        if cvi_given:
            #df_cleandmd = pd.read_excel(io=path_to_metadata_xls, sheet_name=sheet_name_soalge, engine='openpyxl')
            soa = np.repeat(extract_segments(df_cleandmd[df_cleandmd['pat'] == patient_name]['soa'].values[0]),
                        repeats=5, axis=0)
            lge = np.repeat(extract_segments(df_cleandmd[df_cleandmd['pat'] == patient_name]['lgepos'].values[0], lge=True),
                        repeats=5, axis=0)
            lgeef = np.repeat(extract_segments(df_cleandmd[df_cleandmd['pat'] == patient_name]['lgepos_'].values[0]),
                            repeats=5, axis=0)
            try:
                first_split = df_cleandmd[df_cleandmd['pat'] == patient_name]['first_split'].values[0]
            except Exception as e:
                pass

        patientid = ([patient_name] * 16 * 5)
        ahano = np.repeat(range(1, 17), repeats=5, axis=0)
        phaseno = np.tile(range(0, 5), reps=(1, 16)).squeeze()

        # write df
        df_patient = pd.DataFrame(columns=['pat', 'aha', 'phase', 'our_rs', 'our_cs', 'soa', 'lge'])
        df_patient['pat'] = patientid
        df_patient['aha'] = ahano
        df_patient['phase'] = phaseno
        df_patient['our_rs'] = rs_AHA_overtime
        df_patient['our_cs'] = cs_AHA_overtime
        if cvi_given: df_patient['soa'] = soa
        if cvi_given: df_patient['lge'] = lge
        if cvi_given: df_patient['lgeef'] = lgeef
        try:
            df_patient['first_split'] = first_split
        except Exception as e:
            pass
    if df_style == 'peaks':
        # get soa and lge data for current patient from metadata xls
        if cvi_given:
            #df_cleandmd = pd.read_excel(io=path_to_metadata_xls, sheet_name=sheet_name_soalge, engine='openpyxl')
            soa = extract_segments(df_cleandmd[df_cleandmd['pat'] == patient_name]['soa'].values[0])
            lge = extract_segments(df_cleandmd[df_cleandmd['pat'] == patient_name]['lgepos'].values[0])
            lgeef = extract_segments(df_cleandmd[df_cleandmd['pat'] == patient_name]['lgepos_'].values[0])

            try:
                first_split = df_cleandmd[df_cleandmd['pat'] == patient_name]['first_split'].values[0]
            except Exception as e:
                pass

        # get Peak Err/Ecc values for all AHA segments
        # first, mean over all zslices
        # then, max/min value over time
        # get Peak Err/Ecc values for all AHA segments
        # first, mean over all zslices
        # then, max/min value over time
        PRS_values_base = np.nanmean(AHAcube_base[..., 0], axis=2).max(axis=1)
        PRS_values_midcavity = np.nanmean(AHAcube_midcavity[..., 0], axis=2).max(axis=1)
        PRS_values_apex = np.nanmean(AHAcube_apex[..., 0], axis=2).max(axis=1)
        our_prs = 100 * np.concatenate([PRS_values_base, PRS_values_midcavity, PRS_values_apex], axis=0)

        PCS_values_base = np.nanmean(AHAcube_base[..., 1], axis=2).min(axis=1)
        PCS_values_midcavity = np.nanmean(AHAcube_midcavity[..., 1], axis=2).min(axis=1)
        PCS_values_apex = np.nanmean(AHAcube_apex[..., 1], axis=2).min(axis=1)
        our_pcs = 100 * np.concatenate([PCS_values_base, PCS_values_midcavity, PCS_values_apex], axis=0)

        # pat, ahano
        patientid = ([patient_name] * 16)
        ahano = np.arange(1, 17)

        # write df
        df_patient = pd.DataFrame(columns=['pat', 'aha', 'cvi_prs', 'cvi_pcs', 'our_prs', 'our_pcs', 'soa', 'lge'])
        df_patient['pat'] = patientid
        df_patient['aha'] = ahano
        if cvi_given: df_patient['cvi_prs'] = cvi_prs
        if cvi_given: df_patient['cvi_pcs'] = cvi_pcs
        df_patient['our_prs'] = our_prs
        df_patient['our_pcs'] = our_pcs
        if cvi_given: df_patient['soa'] = soa
        if cvi_given: df_patient['lge'] = lge
        if cvi_given: df_patient['lgeef'] = lgeef
        try:
            df_patient['first_split'] = first_split
        except Exception as e:
            pass
    # append the patient df to the whole list of patients df
    return df_patient


def plot_pairplot_from_dfpatients(df_patients_peaks):
    # df = pd.read_csv(filepath_or_buffer='/home/julian/df_DMD_peaks.csv')
    sns.pairplot(df_patients_peaks, hue='lge', hue_order=[0, 1])
    plt.tight_layout()
    plt.show()

def plot_heatmap_from_dfpatients(df_patients_peaks):
    corr = df_patients_peaks.corr(method='pearson')  # method{‘pearson’, ‘kendall’, ‘spearman’} or callable
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(10, 10))
        ax = sns.heatmap(corr, annot=True, cmap="coolwarm", fmt='.2f', mask=mask, square=True)
    plt.show()

if __name__ == "__main__":
    # set up logging
    Console_and_file_logger('mvfviz/dmd_temp', logging.INFO)

    import argparse

    parser = argparse.ArgumentParser(description='Calculate Strain values')

    # usually these parameters should encapsulate all experiment parameters
    parser.add_argument('-exp', action='store', default=None)
    parser.add_argument('-metadata', action='store', default=None)
    parser.add_argument('-iscontrol', choices=['True','False','true','false'],action='store', default='false')
    parser.add_argument('-debug', action='store', default=None)

    results = parser.parse_args()
    print('given parameters: {}'.format(results))
    is_dmd = True
    if results.iscontrol.lower() == 'true':
        is_dmd = False
        print('*'*30)
        print('Inference on DMD control patients')
        print('*' * 30)

    df_patients_p2p = calculate_strain(data_root=results.exp, metadata_path=results.metadata,
                                   debug=results.debug=='debug', df_style='time', p2p_style=True, isDMD=is_dmd)
    df_patients_ed2p = calculate_strain(data_root=results.exp, metadata_path=results.metadata,
                                   debug=results.debug=='debug', df_style='time', p2p_style=False, isDMD=is_dmd)

    x=0
    df_patients_p2p.to_csv(os.path.join(results.exp, 'df_DMD_time_p2p.csv'), index=False)
    df_patients_ed2p.to_csv(os.path.join(results.exp, 'df_DMD_time_ed2p.csv'), index=False)
    print('calculate strain done!')

    # continue with peaks data
    # plot_pairplot_from_dfpatients(df_patients_peaks=df_patients)
    # plot_heatmap_from_dfpatients(df_patients_peaks=df_patients)




