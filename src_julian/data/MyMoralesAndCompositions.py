####################################################################################################################
# define logging and working directory
from ProjectRoot import change_wd_to_project_root
change_wd_to_project_root()

# import helper functions
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
from logging import info as INFO
from src_julian.utils.skhelperfunctions import Console_and_file_logger
from src_julian.utils.myhelperfunctions import *
from src_julian.utils.myclasses import *
from src_julian.utils.mystrainfunctions import *
from src_julian.utils.MoralesFast import *

# set up logging
Console_and_file_logger('mvfviz/dmd_temp', logging.INFO)

#######################################################################################################
def myMorales(ff_comp, seg_lvcube, Z_SLICES, N_TIMESTEPS, lv_label=1):
    '''
    input: tzyxc with c=xyz
    but inside then einsums in the first step to xyz order
    '''
    Radial = np.zeros((N_TIMESTEPS,ff_comp.shape[3],ff_comp.shape[2],len(Z_SLICES)))
    Circumferential = np.zeros((N_TIMESTEPS,ff_comp.shape[3],ff_comp.shape[2],len(Z_SLICES)))
    masks_rot = np.ndarray((N_TIMESTEPS,ff_comp.shape[3],ff_comp.shape[2],len(Z_SLICES)))

    for t in range(N_TIMESTEPS):
        # prepare Morales inputs
        # for ED->X defines flowfields only take the ED mask
        # thus when we have composed flowfields here we will only take the ED mask
        mask = np.einsum('zyx->xyz', np.squeeze(seg_lvcube[0, ...]))[..., Z_SLICES]
        flow = np.einsum('zyxc->xyzc', ff_comp[t, ...])

        # strain calculation
        strain = MyocardialStrain(mask=mask, flow=flow)
        strain.calculate_strain(dx=1.4, dy=1.4, dz=1.4, lv_label=lv_label)

        # set values outside of input mask to zero
        strain.Err[strain.mask_rot != lv_label] = 0.0
        strain.Ecc[strain.mask_rot != lv_label] = 0.0

        # save mask_rot
        masks_rot[t] = strain.mask_rot

        # save strain values
        Radial[t,...] += strain.Err
        Circumferential[t,...] += strain.Ecc

        GRS = strain.Err[strain.mask_rot==1].mean()
        GCS = strain.Ecc[strain.mask_rot==1].mean()
        INFO('iteration: ' + str(t+1) + ' / ' + str(N_TIMESTEPS))
        print(GRS, GCS)

    return Radial, Circumferential, masks_rot

def get_parameter_from_xls(dataframe, patientname, parametername):
    return dataframe.loc[patientname, parametername]

def get_mean_strain_values_from_Morales(array, masks):
    MeanStrains = np.zeros(5)
    for t in range (5):
        MeanStrains[t] = 100*array[t][masks[t]==1].mean()
    return MeanStrains

###initial inits###
results = [] # df where we will store our results
path_to_metadata_xls = '/mnt/ssd/julian/data/metadata/DMDTarique_1.7.xlsx'

path_to_segmentation_folder = '/mnt/ssd/julian/data/raw/flowfields/v10_nn011_nopostprocess/'
# v10_nn011_nopostprocess
# v9_nn_01_1
# v8_NNvsLI

# read patients names list
patients = [ item for item in os.listdir(path_to_segmentation_folder) if os.path.isdir(os.path.join(path_to_segmentation_folder, item)) ]

patients=['hh_20190621']
# ab_20180710
# ej_20191126
# fa_20180626
# cm_20180608
# dc_20180516

# loop here
for i, patient_name in enumerate(patients):

    path_to_patient_folder = path_to_segmentation_folder + patient_name + '/'
    # path_to_patient_folder = path_to_segmentation_folder
    N_TIMESTEPS = 5
    Z_SPACING = 1

    # get patient specific whole-heart-volume-range which means:
    # the slices where we do have segmentations available
    # as a basis to calculate the inner 35% midcavity slices
    # in order to see the range from where we will pick the midcavity slices, we have to check where we do have segmentations
    # i.e. we have 5 timesteps where we do have segmentations in the slices 5...50, 2...40, ...
    # then we will choose 2...50 as the final whole-heart-volume-range from where we will take the midcavity slices
    # masks = get_segmentationarray(path_to_nii_folder=path_to_patient_folder, naming='_mask_', Ntimesteps=N_TIMESTEPS, Z_SPACING=Z_SPACING)
    # data_masks = volume(masks, '4Dt', Z_SPACING)
    # seg_available = data_masks.get_segmentationarray(resolution='slicewise')
    # # the wholeheartvolumeborders numbers are idx numbers and refer equally to the mitk slices
    # wholeheartvolumeborders = get_wholeheartvolumeborders(seg_available, N_TIMESTEPS)
    # # from the whole heart volume indices, calculate the inner 35% aha midcavity indices
    # z_start, z_end = get_midcavityvolumeborders(wholeheartvolumeborders)
    # # we have to extend the z_end value by 1 so that the last index will be included in the Z_SLICES array
    # Z_SLICES = np.arange(z_start, z_end+1, 1)

    # get whole segmentation cube
    naming_lv = '_lvmask_'
    mask_whole = stack_nii_masks(path_to_patient_folder, naming_lv, N_TIMESTEPS)
    mask_whole = np.roll(mask_whole, 1, axis=0)

    label_bloodpool = 3
    # get whole heart volume borders by RVIP method
    wholeheartvolumeborders = get_wholeheartvolumeborders_by_RVIP(seg_array=mask_whole, N_TIMESTEPS=N_TIMESTEPS)
    # get mid cavity heart volume borders from whole heart volume border detection; AHA definition
    Z_SLICES = get_midcavityvolumeborders(wholeheartvolumeborders)

    # modify the range to save computation time
    # slices 24 and 25
    # Z_SLICES = Z_SLICES[np.round(len(Z_SLICES)/2).astype(int):np.round(len(Z_SLICES)/2).astype(int)+2]

    # get segmentation cube
    # mask contains only the LV myocardium segmentations
    naming_lvm = '_mask_'
    seg_lvcube = stack_nii_masks(path_to_patient_folder, naming_lvm, N_TIMESTEPS)
    seg_lvcube = np.roll(seg_lvcube, 1, axis=0)

    # get flowfield
    # dont roll the flow!
    naming_flow = '_flow_full_'
    ff = mvf(data=stack_nii_flowfield(path_to_patient_folder, naming_flow, N_TIMESTEPS),
             format='4Dt',
             zspacing=1)
    ff_switched = mvf(data=ff.switch_channel(),
                      format='4Dt',
                      zspacing=1)

    ############COMPOSED FLOWFIELDS############
    ###### get my own manually composed ff ######
    # INFO('Calculating my composition now. Takes time.')
    # ff_sum, targets = ff.compose_myimplementation() # my algorithm
    # ff_sum = mvf(data=ff_sum, format='4Dt', zspacing=1)
    # ff_comp_mine = ff_sum.switch_channel() # switch channel from zyx to xyz

    ###### get svens composed ff ######
    naming_flow_comp = '_flow_comp_'
    ff_comp = stack_nii_flowfield(path_to_patient_folder, naming_flow_comp, N_TIMESTEPS)
    ff_comp = ff_comp[:, Z_SLICES, ...]
    ff_comp = mvf(data=ff_comp, format='4Dt', zspacing=1)
    ff_comp_Sven = ff_comp.switch_channel() # switch c=channel from zyx to xyz

    ###### get laliths just adding composed ff ######
    ff_comp_add = ff_switched.compose_justadding() # order stays the same
    # slice it
    ff_comp_add = ff_comp_add[:, Z_SLICES, ...]

    ###### get sitk composed ff ######
    ff_comp_itk = ff_switched.compose_sitk(Z_SLICES) # itk needs zyxc with c=xyz

    # calculate strain curves by Morales
    Radial_Sven, Circumferential_Sven, masks_rot_Sven = myMorales(ff_comp_Sven, seg_lvcube, Z_SLICES, N_TIMESTEPS)
    Radial_itk, Circumferential_itk, masks_rot_itk = myMorales(ff_comp_itk, seg_lvcube, Z_SLICES, N_TIMESTEPS)
    Radial_add, Circumferential_add, masks_rot_add = myMorales(ff_comp_add, seg_lvcube, Z_SLICES, N_TIMESTEPS)

    ######PLOTTING######
    # plot mean radial/circumferential strain curves wrt ED phase over all slices for both approaches
    STRAIN_plot_composedflowfields(masks_rot_itk,
                                   np.stack([Radial_Sven, Radial_itk, Radial_add]),
                                   np.stack([Circumferential_Sven, Circumferential_itk, Circumferential_add]),
                                   labels=['Sven', 'itk', 'add'],
                                   colors=['k', 'r', 'g'])

    INFO('Hi')


    # plot strain per slice if wanted
    # plot_strain_per_slice_Radial_Circumferential_side_by_side(Radial_itk, Circumferential_itk, masks_rot_itk,
    #                                                           N_TIMESTEPS, Z_SLICES, patientname=patients[0])

    # plot segmentations with splines if wanted
    # plot_segmentation_grid_with_splines(seg_lvcube=seg_lvcube, vol_cube=vol_cube,
    #                                     zslices=Z_SLICES, ntimesteps=N_TIMESTEPS)

    # plot segmentations if wanted
    # zslices = [34, 28, 22]
    # plot_segmentation_grid(seg_lvcube=seg_lvcube, zslices=zslices, ntimesteps=N_TIMESTEPS)


    itk_MCPRS = get_mean_strain_values_from_Morales(array=Radial_itk, masks=masks_rot_itk)
    itk_MCPCS = get_mean_strain_values_from_Morales(array=Circumferential_itk, masks=masks_rot_itk)
    add_MCPRS = get_mean_strain_values_from_Morales(array=Radial_add, masks=masks_rot_add)
    add_MCPCS = get_mean_strain_values_from_Morales(array=Circumferential_add, masks=masks_rot_add)

    ###############WRITE RESULTS###############
    # gather ground truth parameter for comparison
    # create df which contains patients metadata
    df_groundtruth = pd.read_excel(io=path_to_metadata_xls, sheet_name='clean DMD', dtype='object', index_col=0)
    groundtruth_MCPRS = get_parameter_from_xls(dataframe=df_groundtruth, patientname=patient_name,
                                               parametername='mid-cavity peak radial strain')
    groundtruth_MCPCS = get_parameter_from_xls(dataframe=df_groundtruth, patientname=patient_name,
                                               parametername='mid-cavity peak circumferential strain')

    # current results
    tobewritten = [patient_name,
                   groundtruth_MCPRS, itk_MCPRS.max(), add_MCPRS.max(),
                   groundtruth_MCPCS, itk_MCPCS.min(), add_MCPCS.min()]

    # append to big list
    results.append(tobewritten)

    INFO('patient ' + str(i+1) + ' / ' + str(len(patients)) + ' finished')

# write the results list to a file
strain_folder = '/mnt/ssd/julian/data/outputs/strainresults/'
filename = '2021.08.19_OLDVERSION_' + patients[0] + '.txt'
with open(strain_folder + filename, 'w') as f:
    for item in results:
        f.write("%s\n" % item)

# create DATAFRAME
# plot the results list as boxplot/violinplot
df = pd.DataFrame(results)
df.columns = ['ID',
              'Circle_MCPRS', 'itk_MCPRS', 'add_MCPRS',
              'Circle_MCPCS', 'itk_MCPCS', 'add_MCPCS']

import seaborn as sns
sns.set_theme()
plt.figure()
ax = sns.boxplot(data=df)
ax = sns.violinplot(data=df)

# inspecting the results df
import pandas as pd
path='/mnt/ssd/julian/data/outputs/strainresults/2021.08.18_flowfullgtMCPRSvsITKvsJUSTADDED.txt'
df = pd.read_csv(path)
df.columns = ['ID', 'Circle_MCPRS', 'itk_MCPRS', 'add_MCPRS']
# cut the square brackets from the data
df['ID'] = df['ID'].map(lambda x: x.lstrip('['))
df['add_MCPRS'] = df['add_MCPRS'].map(lambda x: x.rstrip(']'))
df['add_MCPRS'] = pd.to_numeric(df['add_MCPRS']) # convert to float for consecutive rounding
df.round(decimals=1)

# plot correlation
import seaborn as sns
sns.set_theme()
ax=sns.scatterplot(data=df, x='Circle_MCPRS', y='itk_MCPRS', label='itk_MCPRS')
ax=sns.scatterplot(data=df, x='Circle_MCPRS', y='add_MCPRS', label='add_MCPRS')
ax.set_xlabel('Circle_MCPRS')
ax.set_ylabel('Second Method')
ax.set_title('Scatter Plot')

import seaborn as sns
sns.set_theme()
ax=sns.scatterplot(data=df, x='Circle_MCPCS', y='itk_MCPCS', label='itk_MCPCS')
ax=sns.scatterplot(data=df, x='Circle_MCPCS', y='add_MCPCS', label='add_MCPCS')
ax.set_xlabel('Circle_MCPCS')
ax.set_ylabel('Second Method')
ax.set_title('Scatter Plot')

sns.lmplot(x='Circle_MCPRS', y='itk_MCPRS', data=df)
plt.xlim(0, 50)
plt.ylim(0, 120)
plt.title('Circle vs itk')

sns.lmplot(x='Circle_MCPRS', y='add_MCPRS', data=df)
plt.xlim(0, 50)
plt.ylim(0, 120)
plt.title('Circle vs add')

sns.lmplot(x='Circle_MCPCS', y='itk_MCPCS', data=df)
plt.xlim(-5, -25)
plt.ylim(-15, 0)
plt.title('Circle vs itk')

sns.lmplot(x='Circle_MCPCS', y='add_MCPCS', data=df)
plt.xlim(-5, -25)
plt.ylim(-15, 0)
plt.title('Circle vs add')

# spearman R
from scipy import stats
stats.spearmanr(df['Circle_MCPRS'], df['itk_MCPRS'])
stats.spearmanr(df['Circle_MCPRS'], df['add_MCPRS'])

from scipy import stats
stats.spearmanr(df['Circle_MCPCS'], df['itk_MCPCS'])
stats.spearmanr(df['Circle_MCPCS'], df['add_MCPCS'])

# histograms
df['Circle_MCPRS'].hist()
df['itk_MCPRS'].hist()
df['add_MCPRS'].hist()

df['Circle_MCPCS'].hist()
df['itk_MCPCS'].hist()
df['add_MCPCS'].hist()

# seaborn qq plot to check if data is normally distributed
import statsmodels.api as sm
from scipy.stats import norm
import pylab
sm.qqplot(df['Circle_MCPRS'], line='s')
sm.qqplot(df['itk_MCPRS'], line='s')
sm.qqplot(df['add_MCPRS'], line='s')

# seaborn qq plot to check if data is normally distributed
import statsmodels.api as sm
from scipy.stats import norm
import pylab
sm.qqplot(df['Circle_MCPCS'], line='s')
sm.qqplot(df['itk_MCPCS'], line='s')
sm.qqplot(df['add_MCPCS'], line='s')

# pearson R
from scipy import stats
stats.pearsonr(df['Circle_MCPRS'], df['itk_MCPRS'])
stats.pearsonr(df['Circle_MCPRS'], df['add_MCPRS'])

from scipy import stats
stats.pearsonr(df['Circle_MCPCS'], df['itk_MCPCS'])
stats.pearsonr(df['Circle_MCPCS'], df['add_MCPCS'])




# plot mean of single slice at given time step
t = 0
plt.figure()
plt.imshow(Radial_itk[t,...].mean(axis=-1), cmap='jet')
plt.colorbar()
plt.show()
plt.figure()
plt.imshow(Circumferential[t,...].mean(axis=-1), cmap='jet')
plt.colorbar()
plt.show()

# identify min and max values first
vmax = Radial_itk.max()
vmin = Radial_itk.min()

vmax = 1.5
vmin = -0.5
ax_labels=['ED', 'MS', 'ES', 'PF', 'MD', 'ED']
cmap = plt.cm.jet
# cmap.set_under(color='white')
fig, ax = plt.subplots(1,5,figsize=(12,8), sharey=True)
for t in range(N_TIMESTEPS):
    im = ax[t].imshow(Radial_itk.mean(axis=-1)[t], interpolation=None, cmap=cmap, vmin=vmin, vmax=vmax)
    ax[t].set_title(str(ax_labels[0]) + '$\longrightarrow$' + str(ax_labels[t+1]))
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.35, 0.025, 0.3])
fig.colorbar(im, cax=cbar_ax)
plt.show()






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
