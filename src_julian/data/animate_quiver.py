import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from src_julian.utils.myhelperfunctions import *

# https://stackoverflow.com/questions/19329039/plotting-animated-quivers-in-python
def update_quiver(num, t, Q):
    """
    updates the horizontal and vertical vector components by a
    fixed increment on each frame
    """
    N=1
    U = ff_comp[num, z, ::N, ::N, 2]  # x
    V = ff_comp[num, z, ::N, ::N, 1]  # y
    Q.set_UVC(U,V)
    return Q

nt = 5
patient_name='hh_20190621'
path_to_segmentation_folder = '/mnt/ssd/julian/data/raw/flowfields/v16_smoothmyo_maskloss0_001_compose_reg_dmd/'
path_to_patient_folder = path_to_segmentation_folder + patient_name + '/'
naming_flow = 'flow_composed_'
ff_comp = stack_nii_flowfield(path_to_patient_folder, naming_flow, Ntimesteps=nt)

mask_whole = stack_nii_masks(path_to_patient_folder, 'fullmask_target_', N_TIMESTEPS=nt)  # refactored
nt, nz, ny, nx, _ = ff_comp.shape
title=['ED-MS', 'ED-ES', 'ED-PF', 'ED-MD', 'ED-ED']
z = 24
N=1
xmin, xmax, ymin, ymax = (30, 90, 20, 80)
for t in range(nt):
    X, Y = np.mgrid[0:nx, 0:ny]
    X, Y = (X[::N, ::N], Y[::N, ::N])
    U = ff_comp[t,z,::N,::N,2] #x
    V = ff_comp[t,z,::N,::N,1] #y
    fig, ax = plt.subplots(1,1)
    ax.imshow(mask_whole[t, z, ..., 0], cmap='gray')
    Q = ax.quiver(Y, X, V, U, units='xy', angles='xy', scale=1, color='y')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)
    anim = animation.FuncAnimation(fig, update_quiver, fargs=(t, Q), interval=2000, repeat=True)
    plt.show()

x=0