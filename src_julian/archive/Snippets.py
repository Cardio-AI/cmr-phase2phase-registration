#---------------------snippet to read dcm folder to plot sax slices with show_2Dor3D-------------------------------------------#
# reader = sitk.ImageSeriesReader()
# dicom_names = reader.GetGDCMSeriesFileNames(PATH_TO_DICOM_SAX_CA)
#
# reader.SetFileNames(dicom_names)
# image = reader.Execute()
# image_array = sitk.GetArrayFromImage(image) # z, y, x
# origin = image.GetOrigin() # x, y, z
# spacing = image.GetSpacing() # x, y, z
#
# sk.show_2D_or_3D(img=image_array, mask=None, save=False, file_name='egal.png', dpi=200, f_size=(5,5), interpol='bilinear')
#
# plt.show()
#---------------------snippet to read dcm folder to plot sax slices with show_2Dor3D-------------------------------------------#


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