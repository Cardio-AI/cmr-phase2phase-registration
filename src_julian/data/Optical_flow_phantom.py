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
from src_julian.utils.mystrainfunctions import *
from src_julian.utils.myhelperfunctions import *
from src_julian.utils.MoralesFast import *
# set up logging
Console_and_file_logger('mvfviz/dmd_temp', logging.INFO)

#######################################################################################################
import numpy as np
import cv2 as cv
from skimage.draw import disk
import matplotlib.pyplot as plt

# create phantom mask and volume
nt, nz, ny, nx, nc = (10,11,288,288,1)
mask = np.ndarray((nt, nz, ny, nx, nc))
r_o_start = 100
r_i_start = 80
r_i_min = 10 # this inner radius will be reached at ES, max. contraction

def draw_mask(ED_inner_radius, ED_outter_radius, lv_label=1):

    ED_outter_center = (np.round(nx/2),np.round(ny/2))
    ED_inner_center = (np.round(nx/2),np.round(ny/2))

    ED_outter = np.zeros((nx, ny, 1))
    rr, cc = disk(ED_outter_center, ED_outter_radius)
    ED_outter[rr,cc,:] = lv_label
    ED_inner = np.zeros((nx, ny, 1))
    rr, cc = disk(ED_inner_center, ED_inner_radius)
    ED_inner[rr,cc,:] = lv_label
    ED_mask = ED_outter-ED_inner

    return ED_mask

for t in range(nt):
    for z in range(nz):
        # calculate new radii; we will keep the outer radius constant for now
        delta_i = t*(r_i_start-r_i_min)/nt
        inner_radius = r_i_start-delta_i
        outer_radius = r_o_start

        # draw mask and save
        mask[t,z,...] = draw_mask(inner_radius, outer_radius)

#####TBD#######
# mirror the volume dataset so that we have thickening and then thinning
############

# plot one slice over time to check plausibility
# z = 0
# fig,ax=plt.subplots(1,nt, sharey=True)
# for t in range(nt): ax[t].imshow(mask[t,z,:,:,0])
# plt.show()

# apply Gaussian kernel to the clean binary mask
# blur = np.zeros_like(mask)
# ksize = 31 #odd!
# borderType=cv.BORDER_CONSTANT
# for t in range(nt):
#     for z in range(nz):
#         blur[t, z, ...] = cv.GaussianBlur(src_julian=mask[t,z,...], ksize=(ksize,ksize), sigmaX=0)[..., np.newaxis]

# blur = mask

# plot one slice over time to check plausibility
z = 0
fig,ax=plt.subplots(1,nt, sharey=True)
for t in range(nt): ax[t].imshow(mask[t,z,:,:,0])
plt.show()

# round the blur, this is necessary if we later want to filter by the blurred image
# blur = np.round(a=blur, decimals=1, out=None)

# array has to be of type tzyxc where c=1
phantom = volume(data=mask, format='4Dt', zspacing=1)

# the returned flow will be of shape nt, ny, nx, nc=2
# calc flow on blurred masks
slice = 5
flow, bgr_array = phantom.get_Farneback_flowfield_of_slice(slice=slice, reference='dynamic')


# prep = np.zeros((nt,ny,nx,3))
# prep[...,1] = np.copy(flow[...,1]) #y
# prep[...,2] = np.copy(flow[...,0]) #x
# prep = prep[:,np.newaxis,...] #tzyxc
# prep = np.repeat(a=prep, repeats=nz, axis=1)
# t = 2
# slice = 5
# N = 5
# test = mvf(data=prep[t, ...], format='4D', zspacing=1)
# xx, yy, Fx, Fy = test.plot_Grid2D_MV2Dor3D(slice=slice, N=N)
# fig, ax = plt.subplots()
# plt.quiver(xx, yy, Fx, Fy, units='xy', angles='xy', scale=1, color='y')
# ax.set_title('MVF plot')
# ax.set_aspect('equal')
# plt.show()


# plot bgr array
# fig,ax = plt.subplots(1, nt, sharey=True)
# for t in range(nt): ax[t].imshow(bgr_array[t, ...])
# plt.show()

# rearrange the Farneback result
# Farneback result will contain y,x in this order
# 30,288,288,2
# construction of the 3D Farneback flowfield
flow_Farneback = np.ndarray((nt,ny,nx,3))
flow_Farneback[...,0] = flow[...,1] #x
flow_Farneback[...,1] = flow[...,0] #y
flow_Farneback[...,2] = 0 #z

# inits
# txyz
# we have to double the layers again to make use of Morales algorithm
zlayers=2
masks_rot = np.ndarray((nt,ny,nx,zlayers))
Radial = np.zeros((nt,ny,nx,zlayers))
Circumferential = np.zeros((nt,ny,nx,zlayers))

# Morales needs at least two slices, so we copy the info
# mask_Morales = np.copy(np.squeeze(mask[:,0:2,...]))
flow_Farneback = flow_Farneback[:, np.newaxis, ...] # now tzyx
flow_Farneback = np.repeat(flow_Farneback, zlayers, axis=1)

# apply Morales strain algorithm
label = 1
for t in range(nt):
    # m = np.einsum('zyx->xyz', mask_Morales[0]) # ED mask
    m = np.einsum('zyx->xyz', phantom.Data[t,0:2,:,:,0])  # ED mask
    f = np.einsum('zyxc->xyzc', flow_Farneback[t])
    strain = MyocardialStrain(mask=m, flow=f)
    strain.calculate_strain(lv_label=label)  # the label used for centroid calc
    # strain.Err[strain.mask_rot != 0] = 0.0 #  label=2 means LVM
    # strain.Ecc[strain.mask_rot != 0] = 0.0
    strain.Err[strain.mask_rot != label] = 0.0
    strain.Ecc[strain.mask_rot != label] = 0.0
    masks_rot[t] = strain.mask_rot
    Radial[t, ...] += strain.Err
    Circumferential[t, ...] += strain.Ecc
    INFO('iteration: ' + str(t + 1) + ' out of ' + str(nt))




# inspect strain curves
# INFO('Hi')

prep = np.zeros((nt,ny,nx,3))
prep[...,1] = np.copy(flow[...,1]) #y
prep[...,2] = np.copy(flow[...,0]) #x
prep = prep[:,np.newaxis,...] #tzyxc
prep = np.repeat(a=prep, repeats=nz, axis=1)
t = 0
slice = 5
N = 3
test = mvf(data=prep[t, ...], format='4D', zspacing=1)
xx, yy, Fx, Fy = test.plot_Grid2D_MV2Dor3D(slice=slice, N=N)
fig, ax = plt.subplots()
plt.quiver(xx, yy, Fx, Fy, units='xy', angles='xy', scale=1, color='r')
plt.imshow(masks_rot[0,:,:,0])
ax.set_title('MVF plot')
ax.set_aspect('equal')
plt.show()

plt.figure()
for t in range(nt-1): plt.scatter(t,100*Radial[t,...,0][masks_rot[t,...,0]==1].mean())
plt.ylabel('Err in %')
plt.xlabel('Frame')
plt.show()

plt.figure()
for t in range(nt-1): plt.scatter(t,100*Circumferential[t,...,0][masks_rot[t,...,0]==1].mean())
plt.ylabel('Ecc in %')
plt.xlabel('Frame')
plt.show()

fig,ax=plt.subplots(1,nt,sharey=True)
# for t in range(nt): ax[t].imshow(blur[t,0,...,0])
for t in range(nt): ax[t].imshow(Radial[t,...,0])
plt.show()