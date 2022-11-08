####################################################################################################################
# define logging and working directory
from ProjectRoot import change_wd_to_project_root
change_wd_to_project_root()
# import helper functions
import matplotlib.pyplot as plt
from src_julian.utils.myclasses import *
import logging
from logging import info as INFO
from src_julian.utils.skhelperfunctions import Console_and_file_logger
from src_julian.utils.Morales import *
# set up logging
Console_and_file_logger('mvfviz/dmd_temp', logging.INFO)

#######################################################################################################
import numpy as np
import cv2 as cv
import SimpleITK as sitk

# metadata
phase_labels = [29,6,13,19,25]

# load nrrd
img = sitk.ReadImage('/mnt/ssd/julian/data/interim/2021.04.20_allcopyrotationfalse/DMD/sax/hh_20190621_volume_clean.nrrd')
array_unsorted = sitk.GetArrayFromImage(img)
# array_unsorted is of shape tzyx
# array = np.einsum('tzyx->txyz', array_unsorted)

# at the beginning now, we will only take the mid cavity slice
# 5 because 11 in total
# array = array[..., 5]

# roll, so that ED is frame 0
# we roll by 2 because ED frame in this patient is 29 in the reference 1-30, that means 28 in the reference 0-29
# which is the very last one in the series of this patient
array = np.roll(array_unsorted, 2, axis=0)

# call Farneback implementation
# volume 4Dt means nt nz ny nx nc
# append one axis to get the values
vol = volume(data=array[..., np.newaxis], format='4Dt', zspacing=1)

# the returned flow will be of shape nt, ny, nx, nc=2
slice = 5
flow = vol.get_Farneback_flowfield_of_slice(slice=slice, reference='dynamic')

# code to visually inspect the rolled volume for one slice out of the mid.cavity ie set z=5 (out of 11)
# z=5
# N=2
# Nplots=np.round((flow.shape[0]-1)/2).astype(int)
# fig,ax=plt.subplots(1,Nplots+1, sharey=True, sharex=True)
# for idx, t in enumerate(np.arange(0, 29, N)): ax[idx].imshow(vol.Data[t,z,:,:,0])
# plt.show()


# rearrange the Farneback result
# Farneback result will contain y,x in this order
# 30,288,288,2
# construction of the 3D Farneback flowfield
flow_Farneback = np.ndarray((30,288,288,3))
flow_Farneback[...,0] = flow[...,1] #x
flow_Farneback[...,1] = flow[...,0] #y
flow_Farneback[...,2] = 0 #z


# INFO('Hi')

####flow evaluation#######
# push into morales approach

# load masks
mask = sitk.ReadImage('/mnt/ssd/julian/data/interim/2021.04.20_allcopyrotationfalse/DMD/sax/hh_20190621_volume_mask.nrrd')
mask = sitk.GetArrayFromImage(mask) #30,11,288,288
mask = mask[:, slice, ...]
mask = np.roll(mask, 2, axis=0)

# inits
# txyz
zlayers=2
masks_rot = np.ndarray((30,288,288,zlayers))
Radial = np.zeros((30,288,288,zlayers))
Circumferential = np.zeros((30,288,288,zlayers))
N_TIMESTEPS = 30

# Morales needs at least two slices, so we copy the info
mask = mask[:, np.newaxis, ...] # now tzyx
mask = np.repeat(mask, zlayers, axis=1)
flow_Farneback = flow_Farneback[:, np.newaxis, ...] # now tzyx
flow_Farneback = np.repeat(flow_Farneback, zlayers, axis=1)

# apply Morales strain algorithm
for t in range(N_TIMESTEPS):
    m = np.einsum('zyx->xyz', mask[0]) # ED mask
    f = np.einsum('zyxc->xyzc', flow_Farneback[t])
    strain = MyocardialStrain(mask=m, flow=f)
    strain.calculate_strain(lv_label=3)  # label=3 means bloodpool, used for centroid calc, takes time
    strain.Err[strain.mask_rot != 2] = 0.0 #  label=2 means LVM
    strain.Ecc[strain.mask_rot != 2] = 0.0
    masks_rot[t] = strain.mask_rot
    Radial[t, ...] += strain.Err
    Circumferential[t, ...] += strain.Ecc
    INFO('iteration: ' + str(t + 1) + ' out of ' + str(N_TIMESTEPS))

INFO('Hi')

plt.figure()
for t in range(30): plt.scatter(t,100*Radial[t,...,2][masks_rot[0,...,2]==2].mean())
plt.ylabel('Err in %')
plt.xlabel('Frame')
plt.show()


plt.figure()
for t in range(30): plt.scatter(t,100*Circumferential[t,...,2][masks_rot[0,...,2]==2].mean())
plt.ylabel('Ecc in %')
plt.xlabel('Frame')
plt.show()


