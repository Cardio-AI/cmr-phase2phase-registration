import os, logging
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage.measurements import center_of_mass
from src_julian.utils.skhelperfunctions import get_ip_from_mask_3d
from src_julian.utils.MoralesFast import roll_to_center
from src_julian.utils.myclasses import mvf

# set up logging
from src_julian.utils.skhelperfunctions import Console_and_file_logger
#Console_and_file_logger('mvfviz/dmd_temp', logging.INFO)

def calculate_sector_masks(mask_whole, com_cube, RVIP_cube, Z_SLICES, level):
    '''
    calculates an array of same shape as mask_whole which contains labels of AHA segments
    mask_whole has to be of form tzyxc
    '''

    # inits
    sector_masks = np.zeros_like(mask_whole[..., 0])
    nt, nz, ny, nx = mask_whole[..., 0].shape

    for t in range(nt):
        # COM_glob needs order y,x
        # ant and inf RVIP need order y,x
        COM_glob = [com_cube[t, 1], com_cube[t, 2]]
        for z in Z_SLICES:
            ant = [RVIP_cube[t,z,0,0], RVIP_cube[t,z,0,1]]
            inf = [RVIP_cube[t,z,1,0], RVIP_cube[t,z,1,1]]
            if level == 'base' or level == 'mid-cavity':
                sector_masks[t, z] = get_AHA_6_sector_mask(ant=ant, inf=inf, COM_glob=COM_glob, N_AHA=6, nx=nx, ny=ny,
                                                           level=level)
            elif level == 'apex':
                sector_masks[t, z] = get_AHA_4_sector_mask(ant=ant, inf=inf, COM_glob=COM_glob, N_AHA=4, nx=nx, ny=ny)

    return sector_masks


def get_AHA_6_sector_mask(ant, inf, COM_glob, N_AHA, nx, ny, level):
    '''
    takes insertion points, center of mass and image dimensions and returns an aha sector plot
    of size i.e. 6, 128, 128
    CAVE: the resulting mask may miss some points. should not influence further vector field processing
    '''

    # inits
    RVIPup_loc = np.zeros_like(ant)
    RVIPlow_loc = np.zeros_like(inf)

    # shift insertion points around mass center for angle calculation
    RVIPup_loc[0] = ant[0] - COM_glob[0]
    RVIPup_loc[1] = ant[1] - COM_glob[1]
    RVIPlow_loc[0] = inf[0] - COM_glob[0]
    RVIPlow_loc[1] = inf[1] - COM_glob[1]

    # calculate the two angles of the RVIP
    angles = np.zeros(N_AHA)
    angles[0], _ = np.mod(cart2pol(RVIPup_loc[0], RVIPup_loc[1]), 2 * np.pi)
    angles[2], _ = np.mod(cart2pol(RVIPlow_loc[0], RVIPlow_loc[1]), 2 * np.pi)

    # now we have to take care of the two cases: standard and flipped case
    if angles[0] > angles[2]:
        # standard
        # we are always walking counterclockwise!
        angles[1] = (angles[0] + angles[2]) / 2
        max_angle = angles[0]
        min_angle = angles[2]
        step = (2 * np.pi - (max_angle - min_angle)) / 4
        angles[3] = min_angle - step
        angles[4] = min_angle - 2 * step
        angles[5] = min_angle - 3 * step

    elif angles[0] < angles[2]:
        # almost the same here
        # flipped case, this means that in the image the RVIP_up lies below the RVIP_low
        # np.mod catches negative values and adds 2pi
        # we are always walking counterclockwise!
        angles[1] = np.mod(angles[0]-(angles[0]+2*np.pi-angles[2])/2, 2*np.pi)
        max_angle = angles[2]
        min_angle = angles[0]
        step = (max_angle - min_angle) / 4
        angles[3] = max_angle - step
        angles[4] = max_angle - 2 * step
        angles[5] = max_angle - 3 * step

    # this should not happen in the flipped case!
    angles -= (angles > 2 * np.pi) * 2 * np.pi
    angles += (angles < 0) * 2 * np.pi
    degrees = np.degrees(angles)

    # create sector masks
    AHA_sector_masks = np.ndarray((N_AHA, nx, ny))
    for idx, degree in enumerate(degrees):
        angle_left = degree

        # if we inspect the last entry we want to compare it to the first again
        if idx < 5:
            angle_right = degrees[idx + 1]
        else:
            angle_right = degrees[0]

        # we check if the values are falling, when they rise, we have passed the 0/2pi border
        # radius = 2*nx to draw the spoke wheel for the whole image dimensions
        # function sector_mask needs the angle_range ordered min to max
        # centre coordinates have to be switched here.
        if angle_right < angle_left:
            AHA_sector_masks[idx] = sector_mask(shape=(nx, ny), centre=(COM_glob[1], COM_glob[0]), radius=2*nx, # (COM_glob[1], COM_glob[0])
                                                angle_range=(angle_right, angle_left))
        else:
            AHA_sector_masks[idx] = sector_mask(shape=(nx, ny), centre=(COM_glob[1], COM_glob[0]), radius=2*nx, # (COM_glob[1], COM_glob[0])
                                                angle_range=(0, angle_left)) \
                                    + sector_mask(shape=(nx, ny), centre=(COM_glob[1], COM_glob[0]), radius=2*nx, # (COM_glob[1], COM_glob[0])
                                                  angle_range=(angle_right, 360))

    # write AHA labels to sector masks instead of zeros and ones
    if level == 'base':
        AHA_sector_masks[0][AHA_sector_masks[0] != 0] = 2
        AHA_sector_masks[1][AHA_sector_masks[1] != 0] = 3
        AHA_sector_masks[2][AHA_sector_masks[2] != 0] = 4
        AHA_sector_masks[3][AHA_sector_masks[3] != 0] = 5
        AHA_sector_masks[4][AHA_sector_masks[4] != 0] = 6
        AHA_sector_masks[5][AHA_sector_masks[5] != 0] = 1
    elif level == 'mid-cavity':
        AHA_sector_masks[0][AHA_sector_masks[0] != 0] = 8
        AHA_sector_masks[1][AHA_sector_masks[1] != 0] = 9
        AHA_sector_masks[2][AHA_sector_masks[2] != 0] = 10
        AHA_sector_masks[3][AHA_sector_masks[3] != 0] = 11
        AHA_sector_masks[4][AHA_sector_masks[4] != 0] = 12
        AHA_sector_masks[5][AHA_sector_masks[5] != 0] = 7

    return np.sum(AHA_sector_masks, axis=0)

def get_AHA_4_sector_mask(ant, inf, COM_glob, N_AHA, nx, ny):
    '''
    takes insertion points, center of mass and image dimensions and returns an aha sector plot
    of size i.e. 4, 128, 128
    CAVE: the resulting mask may miss some points. should not influence further vector field processing
    '''

    # inits
    RVIPup_loc = np.zeros_like(ant)
    RVIPlow_loc = np.zeros_like(inf)

    # shift insertion points around mass center for angle calculation
    # this means coordinate calculation from global to local coordinates
    RVIPup_loc[0] = ant[0] - COM_glob[0]
    RVIPup_loc[1] = ant[1] - COM_glob[1]
    RVIPlow_loc[0] = inf[0] - COM_glob[0]
    RVIPlow_loc[1] = inf[1] - COM_glob[1]

    # calculate the two angles of the RVIP
    # with np.mod approaching, we prevent negative values
    angles = np.zeros(N_AHA)
    angles[0], _ = np.mod(cart2pol(RVIPup_loc[0], RVIPup_loc[1]), 2 * np.pi)
    angles[2], _ = np.mod(cart2pol(RVIPlow_loc[0], RVIPlow_loc[1]), 2 * np.pi)

    # now we have to take care of the two cases: standard and flipped case
    step = (2 * np.pi) / 4
    if angles[0] > angles[2]:
        # standard
        # we are always walking counterclockwise!
        # calculate the true/corrected first and second angles of the segments 14 and 15
        angles[0] = ((angles[0]+angles[2]) / 2) + step/2
        angles[1] = angles[0] - step
        angles[2] = angles[0] - 2 * step
        angles[3] = angles[0] - 3 * step

    elif angles[0] < angles[2]:
        # almost the same here
        # flipped case, this means that in the image the RVIP_up lies below the RVIP_low
        # np.mod catches negative values and adds 2pi
        # we are always walking counterclockwise!
        angles[0] = ((angles[0] + angles[2]) / 2) - step / 2
        angles[1] = angles[0] + step
        angles[2] = angles[0] + 2 * step
        angles[3] = angles[0] + 3 * step

    # this should not happen in the flipped case!
    angles -= (angles > 2 * np.pi) * 2 * np.pi
    angles += (angles < 0) * 2 * np.pi
    degrees = np.degrees(angles)

    # create sector masks
    AHA_sector_masks = np.ndarray((N_AHA, nx, ny))
    for idx, degree in enumerate(degrees):
        angle_left = degree

        # if we inspect the last entry we want to compare it to the first again
        if idx < (N_AHA-1):
            angle_right = degrees[idx + 1]
        else:
            angle_right = degrees[0]

        # we check if the values are falling, when they rise, we have passed the 0/2pi border
        # radius = 2*nx to draw the spoke wheel for the whole image dimensions
        # function sector_mask needs the angle_range ordered min to max
        # centre coordinates have to be switched here.
        if angle_right < angle_left:
            AHA_sector_masks[idx] = sector_mask(shape=(nx, ny), centre=(COM_glob[1], COM_glob[0]), radius=2*nx,
                                                angle_range=(angle_right, angle_left))
        else:
            AHA_sector_masks[idx] = sector_mask(shape=(nx, ny), centre=(COM_glob[1], COM_glob[0]), radius=2*nx,
                                                angle_range=(0, angle_left)) \
                                    + sector_mask(shape=(nx, ny), centre=(COM_glob[1], COM_glob[0]), radius=2*nx,
                                                  angle_range=(angle_right, 360))

    # write AHA labels to sector masks instead of zeros and ones
    AHA_sector_masks[0][AHA_sector_masks[0] != 0] = 14
    AHA_sector_masks[1][AHA_sector_masks[1] != 0] = 15
    AHA_sector_masks[2][AHA_sector_masks[2] != 0] = 16
    AHA_sector_masks[3][AHA_sector_masks[3] != 0] = 13

    return np.sum(AHA_sector_masks, axis=0)

# https://stackoverflow.com/questions/18352973/mask-a-circular-sector-in-a-numpy-array/41743189
def sector_mask(shape, centre, radius, angle_range):
    """
    Return a boolean mask for a circular sector. The start/stop angles in
    `angle_range` should be given in clockwise order.
    """

    x, y = np.ogrid[:shape[0],:shape[1]]
    cx, cy = centre
    tmin, tmax = np.deg2rad(angle_range)

    # ensure stop angle > start angle
    if tmax < tmin:
            tmax += 2*np.pi

    # convert cartesian --> polar coordinates
    r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
    theta = np.arctan2(x-cx,y-cy) - tmin

    # wrap angles between 0 and 2*pi
    theta %= (2*np.pi)

    # circular mask
    circmask = r2 <= radius*radius

    # angular mask
    anglemask = theta <= (tmax-tmin)

    return circmask*anglemask

def cart2pol(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return(theta, r)

def plot_AHA_6_sector_mask_with_RVIP(vol, mask, RVIPup_glob, RVIPlow_glob):
    plt.figure()
    plt.imshow(vol, cmap='gray')
    plt.imshow(mask, alpha=0.5, cmap='jet')
    plt.scatter(RVIPup_glob[0], RVIPup_glob[1], c='r')
    plt.scatter(RVIPlow_glob[0], RVIPlow_glob[1], c='y')
    plt.show()

def plot_3x5grid_CMRxStrainxSectormasks(com_cube, Radial_Morales, masks_rot_Morales, vol_cube, sector_masks_raw, N_TIMESTEPS, Z_SLICES, minmin, maxmax, type):
    '''
    minmin and maxmax are for colorbar strain range

    '''

    fig, ax = plt.subplots(3, 5, sharex=True, sharey=True, figsize=(15, 10))
    ax_labels_ED_relative = ['ED ' + '$\longrightarrow$' + ' MS',
                             'ED ' + '$\longrightarrow$' + ' ES',
                             'ED ' + '$\longrightarrow$' + ' PF',
                             'ED ' + '$\longrightarrow$' + ' MD',
                             'ED ' + '$\longrightarrow$' + ' ED']

    # set the cropping zone for every image
    axmin = 30
    axmax = 100

    # set the interpolation method for the Strain values if wanted
    interpol = 'bilinear'

    # define labels
    # label_bloodpool = 3
    label_lvmyo = 1

    # define range
    # will be first, last and middle z of stack
    # has to be big to small, because these are indexes here
    # if we then access Z_SLICES(z_rel), we will take base to apex slices
    z_selection = [len(Z_SLICES)-1, int(len(Z_SLICES)/2), 0]

    for t in range(N_TIMESTEPS):
        for idx, z_rel in enumerate(z_selection):
            # calculate the center of mass
            # cy, cx, _ = center_of_mass(mask_whole[t, Z_SLICES[z_rel], ...] == label_bloodpool)  # COM of slice
            cx, cy = (com_cube[t,2], com_cube[t,1])

            # plot the strain map
            im = ax[idx,t].imshow(100*Radial_Morales[t, Z_SLICES[z_rel]], cmap='jet', vmin=minmin, vmax=maxmax, interpolation=interpol)

            # plt.imshow(roll_to_center(vol_cube[t, Z_SLICES[z_rel], :, :, 0], cy, cx), cmap='gray')

            # plot the CMR
            # roll to center cx and cy are switched?
            ax[idx,t].imshow(np.ma.masked_where(condition=(masks_rot_Morales[t, Z_SLICES[z_rel]] == label_lvmyo),
                                                a=roll_to_center(vol_cube[t, Z_SLICES[z_rel], :, :, 0], cx, cy)),
                             cmap='gray')

            # plot the sector mask
            ax[idx,t].imshow(np.ma.masked_where(condition=(masks_rot_Morales[t, Z_SLICES[z_rel]] == label_lvmyo),
                                                a=roll_to_center(sector_masks_raw[t, Z_SLICES[z_rel]], cx, cy)),
                             cmap='gray', alpha=.4)

            # plt.imshow(roll_to_center(sector_masks_raw[t, Z_SLICES[z_rel]], cy, cx), cmap='gray', alpha=.5)
            # plt.imshow(np.einsum('xy->yx', masks_rot_Sven[t,...,z_rel]), alpha=.5)
            ax[idx,t].set_xlim(axmin, axmax)
            ax[idx,t].set_ylim(axmax, axmin)
        ax[0,t].set_title(ax_labels_ED_relative[t], fontsize=20)

    # add space for colour bar
    plt.tight_layout()
    fig.subplots_adjust(right=0.90)
    cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
    cb = fig.colorbar(im, cax=cbar_ax, extend='both')
    cb.set_label(str(type) + ' in %')
    plt.show()

def roll_sector_mask_to_bloodpool_center(sector_mask_raw, com_cube, N_TIMESTEPS, Z_SLICES):
    # roll sector masks to center for overlay
    sector_masks_rot = np.zeros_like(sector_mask_raw[:, Z_SLICES, ...])
    for t in range(N_TIMESTEPS):
        cx, cy = (com_cube[t, 2], com_cube[t, 1])
        for idx, val in enumerate(Z_SLICES):
            sector_masks_rot[t, idx] = roll_to_center(sector_mask_raw[t, val], cx, cy)
    return sector_masks_rot

# https://stackoverflow.com/questions/53629554/add-some-numbers-in-a-figure-generated-with-python
def bullseye_plot(ax, data, segBold=None, cmap=None, norm=None, labels=[], labelProps={}):
    import matplotlib as mpl

    fig = plt.gcf()
    from matplotlib import cm

    if segBold is None:
        segBold = []

    linewidth = 2
    data = np.array(data).ravel()

    if cmap is None:
        cmap = plt.cm.viridis

    if norm is None:
        norm = mpl.colors.Normalize(vmin=data.min(), vmax=data.max())

    theta = np.linspace(0, 2*np.pi, 768)
    r = np.linspace(0, 1, 4)

    # Create the bound for the segment 17
    for i in range(r.shape[0]):
        ax.plot(theta, np.repeat(r[i], theta.shape), '-k', lw=linewidth)

    # Create the bounds for the segments  1-12
    for i in range(6):
        theta_i = i*60*np.pi/180
        ax.plot([theta_i, theta_i], [r[1], 1], '-k', lw=linewidth)

    # Create the bounds for the segments 13-16
    for i in range(4):
        theta_i = i*90*np.pi/180 - 45*np.pi/180
        ax.plot([theta_i, theta_i], [r[0], r[1]], '-k', lw=linewidth)

    # Fill the segments 1-6
    r0 = r[2:4]
    r0 = np.repeat(r0[:, np.newaxis], 128, axis=1).T
    for i in range(6):
        # First segment start at 60 degrees
        theta0 = theta[i*128:i*128+128] + 60*np.pi/180
        theta0 = np.repeat(theta0[:, np.newaxis], 2, axis=1)
        z = np.ones((128, 2))*data[i]
        ax.pcolormesh(theta0, r0, z, shading='gouraud', cmap=cmap, norm=norm)
        if labels:
            ax.annotate(labels[i], xy=(theta0[0,0]+30*np.pi/180,np.mean(r[2:4])), ha='center', va='center', **labelProps)
        if i+1 in segBold:
            ax.plot(theta0, r0, '-k', lw=linewidth+2)
            ax.plot(theta0[0], [r[2], r[3]], '-k', lw=linewidth+1)
            ax.plot(theta0[-1], [r[2], r[3]], '-k', lw=linewidth+1)

    # Fill the segments 7-12
    r0 = r[1:3]
    r0 = np.repeat(r0[:, np.newaxis], 128, axis=1).T
    for i in range(6):
        # First segment start at 60 degrees
        theta0 = theta[i*128:i*128+128] + 60*np.pi/180
        theta0 = np.repeat(theta0[:, np.newaxis], 2, axis=1)
        z = np.ones((128, 2))*data[i+6]
        ax.pcolormesh(theta0, r0, z, shading='gouraud', cmap=cmap, norm=norm)
        if labels:
            ax.annotate(labels[i+6], xy=(theta0[0,0]+30*np.pi/180,np.mean(r[1:3])), ha='center', va='center', **labelProps)
        if i+7 in segBold:
            ax.plot(theta0, r0, '-k', lw=linewidth+2)
            ax.plot(theta0[0], [r[1], r[2]], '-k', lw=linewidth+1)
            ax.plot(theta0[-1], [r[1], r[2]], '-k', lw=linewidth+1)

    # Fill the segments 13-16
    r0 = r[0:2]
    r0 = np.repeat(r0[:, np.newaxis], 192, axis=1).T
    for i in range(4):
        # First segment start at 45 degrees
        theta0 = theta[i*192:i*192+192] + 45*np.pi/180
        theta0 = np.repeat(theta0[:, np.newaxis], 2, axis=1)
        z = np.ones((192, 2))*data[i+12]
        ax.pcolormesh(theta0, r0, z, shading='gouraud', cmap=cmap, norm=norm)
        if labels:
            ax.annotate(labels[i+12], xy=(theta0[0,0]+45*np.pi/180,np.mean(r[0:2])), ha='center', va='center', **labelProps)
        if i+13 in segBold:
            ax.plot(theta0, r0, '-k', lw=linewidth+2)
            ax.plot(theta0[0], [r[0], r[1]], '-k', lw=linewidth+1)
            ax.plot(theta0[-1], [r[0], r[1]], '-k', lw=linewidth+1)

    ax.set_ylim([0, 1])
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='horizontal')
    return ax


def myBullsplot(data, lge=None, cmap=None, norm=None, ax=None):
    import matplotlib as mpl

    # create labelling of the AHA sectors
    # for the labelling, we will use rounded values
    # .51 will be rounded up, .49 will be rounded down, .50 will be rounded down
    # plotted as label will be the integer values then

    labels = ['{}\n({})'.format(int(round(data[i]*100)), i+1) for i in range(len(data))]
    if lge is not None:
        labels = [l + '*' if lg else l for l,lg in zip(labels, lge)]
    # Make a figure and axes with dimensions as desired.
    if ax==None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw=dict(projection='polar'))
    else:
        fig=plt.gcf()
    #from matplotlib import cm
    #fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax , orientation='horizontal')
    # Create the 16 segment model
    return bullseye_plot(ax=ax, data=data, cmap=cmap, norm=norm, labels=labels, labelProps={'size':7, "weight":'bold'})


def myMorales(ff_comp, mask_lvmyo, com_cube, spacing, method, reg_backwards):
    '''
    tbd
    '''
    from src_julian.utils.MoralesFast import MyocardialStrain

    # define the lv label for the mask_lvmyo (should always be 1!!!)
    lv_label = 1

    # inits
    Radial = np.zeros((ff_comp.shape[:-1]))
    Circumferential = np.zeros((ff_comp.shape[:-1]))
    masks_rot = np.zeros((ff_comp.shape[:-1]))

    for t in range(ff_comp.shape[0]):
        # take data from timestep
        # for ED->X defined composed flowfields only take the ED mask
        # masklvmyo = mask_lvmyo[0, ..., 0]
        if method == 'p2p':
            masklvmyo = mask_lvmyo[t, ..., 0] # for p2p defined flowfields take the mask dynamically
        elif method == 'ed2p':
            if reg_backwards:
                masklvmyo = mask_lvmyo[0, ..., 0] # for composed and backwards take the ED mask fix, targetmask: MD,ED,MS,ES,PF
            else:
                masklvmyo = mask_lvmyo[t, ..., 0]  # for composed and forward reg we sample from a dynamic target
        else:
            raise NotImplementedError('invalid method: {}, valid methods: {}'.format(method, ['p2p', 'ed2p']))
        com = com_cube[t]
        flow = ff_comp[t]

        # strain calculation
        dx, dy, dz = spacing
        # TEST

        strain = MyocardialStrain(masklvmyo=masklvmyo, com=com, flow=flow)
        strain.calculate_strain(dx=dx, dy=dy, dz=dz)

        # set values outside of input mask to zero
        # comment out following two lines to remain all entries for whole strain map
        # TODO
        #strain.Err[strain.mask_rot != lv_label] = 0.0
        #strain.Ecc[strain.mask_rot != lv_label] = 0.0
        #strain.Err = np.clip(a=strain.Err,a_min=np.quantile(a=strain.Err,q=0.001), a_max=np.quantile(a=strain.Err,q=0.999))
        #strain.Ecc = np.clip(a=strain.Ecc, a_min=np.quantile(a=strain.Ecc, q=0.001),
        #                     a_max=np.quantile(a=strain.Ecc, q=0.999))

        # save mask_rot
        masks_rot[t] = strain.mask_rot

        # save strain values
        Radial[t] = strain.Err
        Circumferential[t] = strain.Ecc

        # GRS = strain.Err[strain.mask_rot==1].mean()
        # GCS = strain.Ecc[strain.mask_rot==1].mean()
        # INFO('iteration: ' + str(t+1) + ' / ' + str(N_TIMESTEPS))
        # print(GRS, GCS)

    return Radial, Circumferential, masks_rot


def get_single_parameter_from_xls(df, parametername, patientname):
    return df.loc[patientname, parametername]


def get_mean_strain_values_from_Morales(array, masks):
    MeanStrains = np.zeros(5)
    for t in range(5):
        MeanStrains[t] = 100 * array[t][masks[t] == 1].mean()
    return MeanStrains


def calculate_AHA_cube(Err, Ecc, sector_masks_rot, masks_rot, Z_SLICES, N_AHA):
    '''
    the last index of AHAcube contains Err (0) and Ecc (1) strain values
    AHAcube is of shape i.e. 6,5,15,2
    where 6=AHA segments, 5=TIMESTEPS, 15=zslices, 2=strain entries
    '''
    nt = Err.shape[0]
    label_lvmyo = 1
    AHA_cube = np.ndarray((len(N_AHA), nt, len(Z_SLICES), 2))

    # here we derive the mean strain per segment mask and slice, every voxel is weighted equally,
    # for compatibility reasons with the per-slice average approach we repeat the mean strain value per slice
    # and average them later
    for t in range(nt):
        for idx, AHA in enumerate(N_AHA): # check if this segment is visible in this slice
            mask = ((sector_masks_rot[t] == AHA) & (masks_rot[t] == label_lvmyo))
            #mask = (sector_masks_rot[t, z_rel] == AHA) # here we dont mask by the smoothed LV myo
            if mask.sum()>0:
                err = np.ma.mean(np.ma.array(Err[t], mask=~mask))
                ecc = np.ma.mean(np.ma.array(Ecc[t], mask=~mask))
            else: # ignore slices where no segment is visible
                err = np.NaN
                ecc = np.NaN
            AHA_cube[idx, t, ..., 0] = err
            AHA_cube[idx, t, ..., 1] = ecc

    # here we derive the mean strain value per slice and average them later,
    # slices with only few voxels are equally weighted as slices with many voxels
    """for t in range(nt):
        for z_rel, z_abs in enumerate(Z_SLICES):
            for idx, AHA in enumerate(N_AHA): # check if this segment is visible in this slice
                mask = ((sector_masks_rot[t, z_rel] == AHA) & (masks_rot[t, z_rel] == label_lvmyo))
                #mask = (sector_masks_rot[t, z_rel] == AHA) # here we dont mask by the smoothed LV myo
                if mask.sum()>0:
                    err = np.ma.mean(np.ma.array(Err[t, z_rel], mask=~mask))
                    ecc = np.ma.mean(np.ma.array(Ecc[t, z_rel], mask=~mask))
                else: # ignore slices where no segment is visible
                    err = np.NaN
                    ecc = np.NaN
                AHA_cube[idx, t, z_rel, 0] = err
                AHA_cube[idx, t, z_rel, 1] = ecc"""

    return AHA_cube


def get_parameter_series_from_xls(df, parametername, patientname):
    import pandas as pd
    return np.squeeze(df.loc[[patientname], [parametername]].to_numpy())


def sort_values_AHA_ascending(values_base, values_midcavity, values_apex, order_base, order_midcavity, order_apex):
    # sort the values by AHA segment ascending order

    order_unsorted = []
    order_unsorted.extend(order_base)
    order_unsorted.extend(order_midcavity)
    order_unsorted.extend(order_apex)

    values_unsorted = []
    values_unsorted.extend(values_base)
    values_unsorted.extend(values_midcavity)
    values_unsorted.extend(values_apex)

    together_unsorted = np.zeros((2, 16))
    together_unsorted[0] = np.array(order_unsorted)
    together_unsorted[1] = np.array(values_unsorted)

    # sort by first row, maintaining columns
    # https://stackoverflow.com/questions/49374253/sort-a-numpy-2d-array-by-1st-row-maintaining-columns
    together_sorted = together_unsorted[:, together_unsorted[0].argsort()]

    # row=1 contains the values; row=0 constains AHA 0-15 ascending
    return together_sorted[1]


def plot_1xX_LGE_series(path_to_lge_volume_nrrd, patientname, fileending, crop=None):
    # crop in order xmin,xmax,ymin,ymax

    import SimpleITK as sitk

    # read volume
    img = sitk.ReadImage(path_to_lge_volume_nrrd + patientname + fileending)

    # only take first slice as there we have previously arranged the true LGE data
    arr = sitk.GetArrayFromImage(img)[0]  # 12,288,288
    nz, ny, nx = arr.shape

    z = int(nz / 2)
    midslice = arr[z, :, :]  # 288,288

    Z_SLICES = np.arange(1, nz - 1)

    fig, ax = plt.subplots(nrows=1, ncols=len(Z_SLICES), figsize=(20, 5), sharex=True, sharey=True)
    for idx, z in enumerate(Z_SLICES):
        im = ax[idx].imshow(arr[z, :, :], cmap='gray')
        if crop:
            ax[idx].set_xlim(crop[0], crop[1])
            ax[idx].set_ylim(crop[3], crop[2])
        ax[idx].set_title('z=' + str(z))
    fig.suptitle('z_total=' + str(nz))
    fig.tight_layout()
    fig.subplots_adjust(right=0.93)
    cbar_ax = fig.add_axes([0.95, 0.2, 0.015, 0.5])
    cb = fig.colorbar(im, cax=cbar_ax, extend='both')
    cb.set_label('pixel value')
    plt.show()


def calculate_center_of_mass_cube(mask_whole, label_bloodpool, base_slices, midcavity_slices, apex_slices, method):
    '''
    calculate center of mass (COM) based on masks on a predefined level
    mask_whole: tzyxc
    com_cube output: tc with c=zyx
    possible options:
    1) statically for ED = "staticED", repeated for all timesteps
    2) dynamically for all timesteps = "dynamically"; this means we have for every timestep a different COM
    '''

    # inits
    nt = mask_whole.shape[0]
    com_cube = np.ndarray((nt, 3, 3))

    # dynamically
    for t in range(nt):
        # calculate com at level
        com_base = center_of_mass((mask_whole[t, base_slices, ..., 0] == label_bloodpool).astype(int))
        com_mc = center_of_mass((mask_whole[t, midcavity_slices, ..., 0] == label_bloodpool).astype(int))
        com_apex = center_of_mass((mask_whole[t, apex_slices, ..., 0] == label_bloodpool).astype(int))

        # providing zyx mask coordinates returns zxy center of mass coordinates; reordering
        com_cube[t, 0, 0], com_cube[t, 0, 1], com_cube[t, 0, 2] = (com_base[0], com_base[2], com_base[1])
        com_cube[t, 1, 0], com_cube[t, 1, 1], com_cube[t, 1, 2] = (com_mc[0], com_mc[2], com_mc[1])
        com_cube[t, 2, 0], com_cube[t, 2, 1], com_cube[t, 2, 2] = (com_apex[0], com_apex[2], com_apex[1])

        """com_cube[t, 0, 0], com_cube[t, 0, 1], com_cube[t, 0, 2] = (com_mc[0], com_mc[2], com_mc[1])
        com_cube[t, 1, 0], com_cube[t, 1, 1], com_cube[t, 1, 2] = (com_mc[0], com_mc[2], com_mc[1])
        com_cube[t, 2, 0], com_cube[t, 2, 1], com_cube[t, 2, 2] = (com_mc[0], com_mc[2], com_mc[1])"""

    # if statically, then overwrite all lines with the first timestep ED
    if method == 'staticED':
        com_cube = np.repeat([com_cube[0]], repeats=nt, axis=0)

    return com_cube

def calculate_RVIP_cube(mask_whole, base_slices, midcavity_slices, apex_slices, method):
    '''
    RVIP_cube = t,z,p,c where p=anterior,inferior and c=y,x
    contains mean RVIP coordinates for the three heart areas base, midcavity, apex
    calculate RVIP cube
    contains anterior and inferior mean RVIP coordinates for LVMYO masks slices range
    contains mean RVIP coordinates for base, mid cavity, apex ranges!
    c = y,x
    dynamically: every timestep gets different mean RVIPs for base mid cavity apex
    staticED: every timestep contains the same mean RVIPs for base mid cavity apex of ED
    '''

    nt, nz = (mask_whole.shape[0], mask_whole.shape[1])
    RVIP_cube = np.zeros((nt, nz, 2, 2))


    for t in range(nt):
        # when mask_whole is provided zyx, the returned tuples are y,x
        if method=='dynamically':
            ant, inf = get_ip_from_mask_3d(mask_whole[t], debug=False, keepdim=True, rev=False)
        elif method=='staticED':
            ant, inf = get_ip_from_mask_3d(mask_whole[0], debug=False, keepdim=True, rev=False)

        # validation plot
        # t, z = (0, 32)
        # cx, cy = center_of_mass(mask_whole[t, z, ..., 0] == label_bloodpool) # provided zyx returns xy because of image domain and np array domain
        # plt.figure()
        # plt.imshow(mask_whole[t, z], cmap='gray')
        # plt.scatter(inf[z][0], inf[z][1], label='inf')
        # plt.scatter(ant[z][0], ant[z][1], label='ant')
        # plt.scatter(cy, cx, label='com')
        # plt.legend()
        # plt.show()


        # write RVIP mean coordinates for all slices
        # remember: when we provide mask_whole as tzyx, the returned RVIP tuples are y,x above.
        # This method fails if we dont find RVIPS at the apical or basal area
        # here we should fallback to the RVIP of the midcavity
        # anterior RVIPs
        base = ant[base_slices[0]:base_slices[-1] + 1]
        midcavity = ant[midcavity_slices[0]:midcavity_slices[-1] + 1]
        apex = ant[apex_slices[0]:apex_slices[-1] + 1]
        if all(i is None for i in apex):
            apex = midcavity[:len(apex)]
        if all(i is None for i in base):
            base = midcavity[:len(base)]
        RVIP_cube[t, base_slices, 0] = np.array([x for x in base if x != None]).mean(axis=0)
        RVIP_cube[t, midcavity_slices, 0] = np.array([x for x in midcavity if x != None]).mean(axis=0)
        RVIP_cube[t, apex_slices, 0] = np.array([x for x in apex if x != None]).mean(axis=0)

         # inferior RVIPs
        base = inf[base_slices[0]:base_slices[-1] + 1]
        midcavity = inf[midcavity_slices[0]:midcavity_slices[-1] + 1]
        apex = inf[apex_slices[0]:apex_slices[-1] + 1]
        if all(i is None for i in apex):
            apex = midcavity[:len(apex)]
            #print('use midcav RVIP for apex')
            #print(midcavity)
        if all(i is None for i in base):
            base = midcavity[:len(base)]
        if all(i is None for i in apex):
            print('None')
        RVIP_cube[t, base_slices, 1] = np.array([x for x in base if x != None]).mean(axis=0)
        RVIP_cube[t, midcavity_slices, 1] = np.array([x for x in midcavity if x != None]).mean(axis=0)
        RVIP_cube[t, apex_slices, 1] = np.array([x for x in apex if x != None]).mean(axis=0)

    return RVIP_cube

def calculate_RVIP_cube2(mask_whole, rvip_range, base_slices, midcavity_slices, apex_slices):
    '''
    calculate RVIP cube
    c = y,x
    first dimension is anterior
    second dimension is inferior
    '''

    nt, nz = (mask_whole.shape[0], mask_whole.shape[1])
    RVIP_cube = np.zeros((nt, nz, 2, 2))

    # whole heart by lvmyo
    lvmyo_base_indices = base_slices
    lvmyo_mc_indices = midcavity_slices
    lvmyo_apex_indices = apex_slices

    # rvip array
    rvip_indices = np.arange(rvip_range[0], rvip_range[-1] + 1)

    # sort rvip array into whole heart indices
    # this equals the overlap zones
    rvip_base_indices = np.intersect1d(lvmyo_base_indices, rvip_indices)
    rvip_mc_indices = np.intersect1d(lvmyo_mc_indices, rvip_indices)
    rvip_apex_indices = np.intersect1d(lvmyo_apex_indices, rvip_indices)

    for t in range(nt):
        # when mask_whole is provided zyx, the returned tuples are y,x
        ant, inf = get_ip_from_mask_3d(mask_whole[t], debug=False, keepdim=True)  # calculate ranges

        # from the intersection arrays, calculate the mean RVIP coordinates
        # anterior RVIPs
        ant_mean_coordinates_base = np.array(ant[rvip_base_indices[0]:rvip_base_indices[-1] + 1]).mean(axis=0)
        ant_mean_coordinates_mc = np.array(ant[rvip_mc_indices[0]:rvip_mc_indices[-1] + 1]).mean(axis=0)
        ant_mean_coordinates_apex = np.array(ant[rvip_apex_indices[0]:rvip_apex_indices[-1] + 1]).mean(axis=0)
        # inferior RVIPs
        inf_mean_coordinates_base = np.array(inf[rvip_base_indices[0]:rvip_base_indices[-1] + 1]).mean(axis=0)
        inf_mean_coordinates_mc = np.array(inf[rvip_mc_indices[0]:rvip_mc_indices[-1] + 1]).mean(axis=0)
        inf_mean_coordinates_apex = np.array(inf[rvip_apex_indices[0]:rvip_apex_indices[-1] + 1]).mean(axis=0)

        # write the mean RVIP into the whole heart range of LVMYO masks
        # anterior RVIPs
        RVIP_cube[t, base_slices, 0] = ant_mean_coordinates_base
        RVIP_cube[t, midcavity_slices, 0] = ant_mean_coordinates_mc
        RVIP_cube[t, apex_slices, 0] = ant_mean_coordinates_apex
        # inferior RVIPs
        RVIP_cube[t, base_slices, 1] = inf_mean_coordinates_base
        RVIP_cube[t, midcavity_slices, 1] = inf_mean_coordinates_mc
        RVIP_cube[t, apex_slices, 1] = inf_mean_coordinates_apex

        # x=0

    return RVIP_cube

def calculate_wholeheartvolumeborders_by_RVIP(mask_whole):
    '''
    get whole heart volume borders by RVIP method
    '''

    nt = mask_whole.shape[0]

    heartborders = np.zeros((nt, 2))
    for t in range(nt):
        # calculate list or IPs
        RVIPup_glob, _ = get_ip_from_mask_3d(mask_whole[t], debug=False, keepdim=True)
        res = [i for i in range(len(RVIPup_glob)) if RVIPup_glob[i] != None]
        heartborders[t, 0] = res[0]
        heartborders[t, 1] = res[-1]

    wholeheartvolumeborders = [int(np.max(heartborders[:, 0])), int(np.min(heartborders[:, 1]))]

    return wholeheartvolumeborders

def plot_3x5_cmroverlaywithmasked_strainormagnitude(ff_composed, Err, Ecc, base_slices, midcavity_slices, apex_slices, vol_cube, com_cube, masks_rot_lvmyo, method):
    # 3x5 plot to overlay CMR, strain map and lvmyomask
    # CMR volume, Strain, LVmyomask
    # the top line will be base
    # the lowest row will be apex
    # for every stage, the middle slice will be taken
    if method == 'Err': Strain = Err
    elif method == 'Ecc': Strain = Ecc

    Displacement_Magnitude0 = ff_composed[...,0]**2+ff_composed[...,1]**2+ff_composed[...,2]**2
    Displacement_Magnitude1 = Displacement_Magnitude0[..., np.newaxis]
    Displacement_Magnitude = np.sqrt(Displacement_Magnitude1[..., 0])

    ax_labels_ED_relative = ['ED ' + '$\longrightarrow$' + ' MS',
                             'ED ' + '$\longrightarrow$' + ' ES',
                             'ED ' + '$\longrightarrow$' + ' PF',
                             'ED ' + '$\longrightarrow$' + ' MD',
                             'ED ' + '$\longrightarrow$' + ' ED']
    idx_abs = [base_slices[int(np.round(len(base_slices)/2))],
               midcavity_slices[int(np.round(len(midcavity_slices)/2))],
               apex_slices[int(np.round(len(apex_slices)/2))]]

    nrows, ncols = (3, 5)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 10), sharex=True, sharey=True)
    for z in range(nrows):
        for t in range(ncols):
            ax[z, t].imshow(roll_to_center(vol_cube[t, idx_abs[z], ..., 0], com_cube[t, z, 2], com_cube[t, z, 1]),
                            cmap='gray', alpha=1, interpolation=None)
            # ax[z, t].imshow(np.ma.masked_where(condition=(masks_rot_lvmyo[t, idx_abs[z]] != 1), a=Strain[t, idx_abs[z]]),
                            # cmap='seismic', alpha=1, interpolation=None)
            ax[z, t].imshow(np.ma.masked_where(condition=(masks_rot_lvmyo[t, idx_abs[z]] != 1), a=Displacement_Magnitude[t, idx_abs[z]]),
                            cmap='seismic', alpha=1, interpolation=None)

            # ax[z, t].imshow(Strain[t, idx_abs[z]], cmap='jet', alpha=1)
            # ax[z, t].imshow(masks_rot_lvmyo[t, idx_abs[z]], alpha=.5, cmap='gray')
            ax[0, t].set_title(ax_labels_ED_relative[t], fontsize=20)
    plt.tight_layout()
    plt.show()

def plot_3x2_AHAStrainMotioncurvesOvertime(AHAcube_base, AHAcube_midcavity, AHAcube_apex, Err_min, Err_max, Ecc_min, Ecc_max):
    '''
    plot curves per AHA and heart base mc apex
    '''
    # Err_min, Err_max = (-20,200)
    # Ecc_min, Ecc_max = (-20,100)
    N_AHA_base = [1, 2, 3, 4, 5, 6]
    N_AHA_midcavity = [7, 8, 9, 10, 11, 12]
    N_AHA_apex = [13, 14, 15, 16]
    ax_labels_ED_relative = ['ED ' + '$\longrightarrow$' + ' MS',
                             'ED ' + '$\longrightarrow$' + ' ES',
                             'ED ' + '$\longrightarrow$' + ' PF',
                             'ED ' + '$\longrightarrow$' + ' MD',
                             'ED ' + '$\longrightarrow$' + ' ED']
    fig, ax = plt.subplots(nrows=3, ncols=2, sharex='col', figsize=(10, 15))
    for idx, AHA in enumerate(N_AHA_base):
        ax[0, 0].plot(ax_labels_ED_relative, 100 * np.nanmean(AHAcube_base, axis=2)[idx, :, 0], label=('AHA' + str(AHA)))
        ax[0, 1].plot(ax_labels_ED_relative, 100 * np.nanmean(AHAcube_base, axis=2)[idx, :, 1], label=('AHA' + str(AHA)))
    for idx, AHA in enumerate(N_AHA_midcavity):
        ax[1, 0].plot(ax_labels_ED_relative, 100 * np.nanmean(AHAcube_midcavity, axis=2)[idx, :, 0], label=('AHA' + str(AHA)))
        ax[1, 1].plot(ax_labels_ED_relative, 100 * np.nanmean(AHAcube_midcavity, axis=2)[idx, :, 1], label=('AHA' + str(AHA)))
    for idx, AHA in enumerate(N_AHA_apex):
        ax[2, 0].plot(ax_labels_ED_relative, 100 * np.nanmean(AHAcube_apex, axis=2)[idx, :, 0], label=('AHA' + str(AHA)))
        ax[2, 1].plot(ax_labels_ED_relative, 100 * np.nanmean(AHAcube_apex, axis=2)[idx, :, 1], label=('AHA' + str(AHA)))
    ax[0, 0].set_title('Mean Err per AHA per phase in %', fontsize=15)
    ax[0, 1].set_title('Mean Ecc per AHA per phase in %', fontsize=15)
    ax[0, 0].set_ylim(Err_min, Err_max)
    ax[0, 1].set_ylim(Ecc_min, Ecc_max)
    ax[1, 0].set_ylim(Err_min, Err_max)
    ax[1, 1].set_ylim(Ecc_min, Ecc_max)
    ax[2, 0].set_ylim(Err_min, Err_max)
    ax[2, 1].set_ylim(Ecc_min, Ecc_max)
    ax[0, 0].legend()
    ax[0, 1].legend()
    ax[1, 0].legend()
    ax[1, 1].legend()
    ax[2, 0].legend()
    ax[2, 1].legend()
    plt.show()

def calculate_ff_magnitude_masked_over_time_and_z(wholeheartvolumeborders, ff_raw, mask_lvmyo):
    '''
    returns an array of shape i.e. 5,40 where 5=timesteps and 40=zslices in wholeheartvolumeborders range
    '''

    ff_raw_mag = np.linalg.norm(ff_raw, axis=-1)
    nt = ff_raw.shape[0]
    label_lvmyo = 1
    z_valid_range = np.arange(wholeheartvolumeborders[0], wholeheartvolumeborders[-1] + 1)
    y = np.ndarray((nt, len(z_valid_range)))

    for t in range(nt):
        for z_rel, z_abs in enumerate(z_valid_range):
            y[t, z_rel] = ff_raw_mag[t, z_abs][mask_lvmyo[t, z_abs, ..., 0] == label_lvmyo].mean()

    return y

def plot_mean_channel_overtime(ff_raw, ff_comp_Sven, ff_comp_itk):
    plt.figure(figsize=(8, 8))
    plt.plot(ff_raw[..., 0].mean(axis=(1, 2, 3)), label='raw z')
    plt.plot(ff_raw[..., 1].mean(axis=(1, 2, 3)), label='raw y')
    plt.plot(ff_raw[..., 2].mean(axis=(1, 2, 3)), label='raw x')
    plt.plot(ff_comp_Sven[..., 0].mean(axis=(1, 2, 3)), label='comp Sven z')
    plt.plot(ff_comp_Sven[..., 1].mean(axis=(1, 2, 3)), label='comp Sven y')
    plt.plot(ff_comp_Sven[..., 2].mean(axis=(1, 2, 3)), label='comp Sven x')
    plt.plot(ff_comp_itk[..., 0].mean(axis=(1, 2, 3)), label='comp itk z')
    plt.plot(ff_comp_itk[..., 1].mean(axis=(1, 2, 3)), label='comp itk y')
    plt.plot(ff_comp_itk[..., 2].mean(axis=(1, 2, 3)), label='comp itk x')
    plt.ylim(-0.3, 0.3)
    plt.legend()
    plt.show()

def plot_motioncurve_per_z_slice_perphase(y):
    '''
    array = of shape i.e. 5, 40 where 5 timesteps and 40 slices
    '''

    nz = y.shape[1]

    # create the colormap for nz slices
    colormap = plt.cm.gist_rainbow
    colors = [colormap(i) for i in np.linspace(0, 1, nz)]

    fig, ax = plt.subplots(1, 1)
    for z in range(nz): ax.plot(y[:, z], label=str(z), color=colors[z])

    plt.suptitle('Masked Displacement Field Magnitude over Phase in LVMYO Range')
    plt.xlabel('Phases')
    plt.ylabel('Mean Vector Magnitude')
    plt.legend(ncol=5)
    plt.show()
    
def get_RVIP_list_from_ACDC_masks(path_to_acdc_patient_folders):
    '''
    this method was developed for BVM submission
    takes path to ACDC patient folder
    returns List of length (number of masks) where sorted as follows:
    ED, ES
    anterior, inferior RVIP
    y,x coordinates

    sample call:
    path_to_acdc_patient_folders = '/mnt/sds-hd/sd20i001/julian/acdc/all/'
    data = get_RVIP_list_from_ACDC_masks(path_to_acdc_patient_folders)
    '''

    import numpy as np
    import glob
    import nibabel as nib
    import matplotlib.pyplot as plt

    # get list of paths to masks
    masks_list = sorted(glob.glob(path_to_acdc_patient_folders + '*/*frame*gt.nii.gz'))
    # cmrs_list = sorted(glob.glob(path_to_acdc_patients + '*/*frame*[!gt].nii.gz'))

    # initializing
    results_list = []

    for idx_patient, mask_path in enumerate(masks_list):

        # read mask image
        img_msk = nib.load(mask_path)
        # img_cmr = nib.load(cmr_path)

        # convert image to array
        # arr is of shape xyz
        arr_msk = np.array(img_msk.dataobj)
        # arr_cmr = np.array(img_cmr.dataobj)

        # reorder array to zyx; dim order of rvip method
        arr_msk_reordered = np.einsum('xyz->zyx', arr_msk)
        # arr_cmr_reordered = np.einsum('xyz->zyx', arr_cmr)

        # get RVIP coordinates for current phase, patient
        # coordinates are y,x
        anterior, inferior = get_ip_from_mask_3d(msk_3d=arr_msk_reordered, keepdim=True)

        # visual inspection
        # z = 5
        # plt.figure()
        # # plt.imshow(arr_cmr_reordered[z], cmap='gray')
        # plt.imshow(arr_msk_reordered[z], alpha=.5)
        # plt.scatter(ant[z][0], ant[z][1], label='ant')
        # plt.scatter(inf[z][0], inf[z][1], label='inf')
        # plt.legend()
        # plt.show()

        # save data
        RVIP_list_for_current_phase = []
        RVIP_list_for_current_phase.append(anterior)
        RVIP_list_for_current_phase.append(inferior)
        results_list.append(RVIP_list_for_current_phase)

    return results_list

def plot_4x5_MaskQuiver_Magnitude_Err_Ecc(ff_composed, mask_whole, Err, Ecc, z, N):
    nt, nz, ny, nx, _ = ff_composed.shape
    title = ['ED-MS', 'ED-ES', 'ED-PF', 'ED-MD', 'ED-ED']
    xmin, xmax, ymin, ymax = (0, 96, 0, 96)
    offset = 0
    cmap_strain='inferno'
    interpol_method = 'bilinear'
    fig, ax = plt.subplots(4, 5, figsize=(12,10))
    for t in range(nt):
        X, Y = np.mgrid[0:nx, 0:ny]
        X, Y = (X[::N, ::N], Y[::N, ::N])
        U = ff_composed[t, z, ::N, ::N, 2]  # x
        V = ff_composed[t, z, ::N, ::N, 1]  # y
        ax[0, t].imshow(mask_whole[0, z, ..., 0])
        ax[0, t].quiver(Y, X, U, V, units='xy', angles='xy', scale=1, color='k')
        ax[0, t].set_title(title[t], fontsize=20)
        ax[0, t].set_xlim(xmin, xmax)
        ax[0, t].set_ylim(ymax, ymin)
        ax[1, t].imshow(np.linalg.norm(ff_composed, axis=-1)[t, z], cmap=cmap_strain, interpolation=interpol_method)
        ax[1, t].set_xlim(xmin, xmax)
        ax[1, t].set_ylim(ymax, ymin)
        ax[2, t].imshow(Err[t,z],cmap=cmap_strain, interpolation=interpol_method)
        ax[2, t].set_xlim(offset, nx-offset)
        ax[2, t].set_ylim(ny-offset, offset)
        ax[3, t].imshow(Ecc[t, z], cmap=cmap_strain, interpolation=interpol_method)
        ax[3, t].set_xlim(offset, nx - offset)
        ax[3, t].set_ylim(ny - offset, offset)
        ax[0, 0].set_ylabel('FF', fontsize=20)
        ax[1, 0].set_ylabel('Mag', fontsize=20)
        ax[2, 0].set_ylabel('Err', fontsize=20)
        ax[3, 0].set_ylabel('Ecc', fontsize=20)
    plt.setp(ax, xticks=[], yticks=[])
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def plot_three_ComposedFlowfields_against_each_other(ff, ff_whole_Sven, ff_whole_itk,
                                                     wholeheartvolumeborders_lvmyo, mask_lvmyo):
    y_raw_z = calculate_ff_magnitude_masked_over_time_and_z(wholeheartvolumeborders=wholeheartvolumeborders_lvmyo,
                                                            ff_raw=np.copy(ff.Data),
                                                            mask_lvmyo=mask_lvmyo)
    y_raw = y_raw_z.mean(axis=1)

    y_comp_z_Sven = calculate_ff_magnitude_masked_over_time_and_z(wholeheartvolumeborders=wholeheartvolumeborders_lvmyo,
                                                                  ff_raw=np.copy(ff_whole_Sven),
                                                                  mask_lvmyo=mask_lvmyo)
    y_comp_Sven = y_comp_z_Sven.mean(axis=1)

    y_comp_z_sitk = calculate_ff_magnitude_masked_over_time_and_z(wholeheartvolumeborders=wholeheartvolumeborders_lvmyo,
                                                                  ff_raw=np.copy(ff_whole_itk),
                                                                  mask_lvmyo=mask_lvmyo)
    y_comp_sitk = y_comp_z_sitk.mean(axis=1)

    plt.figure(figsize=(8, 8))
    plt.plot(y_raw, label='Raw')
    plt.plot(y_comp_Sven, label='Composed_Sven')
    plt.plot(y_comp_sitk, label='Composed_sitk')
    plt.ylim(0, 3)
    plt.suptitle('Masked Displacement Field Magnitude over Phase in LVMYO Range')
    plt.xlabel('Phases')
    plt.ylabel('Mean Vector Magnitude')
    plt.legend()
    plt.show()


def plot_Quiver_onetime_oneslice(t, slice, N, ff_whole, mask_whole, scale=1):
    # whole has dimensions tzyxc with c.ndim=3 zyx
    test = mvf(data=ff_whole[t], format='4D', zspacing=1)
    xx, yy, Fx, Fy = test.plot_Grid2D_MV2Dor3D(slice=slice, N=N)
    fig, ax = plt.subplots()
    plt.imshow(mask_whole[t, slice, ..., 0])
    plt.quiver(xx, yy, Fx, Fy, units='xy', angles='xy', scale=scale, color='k')
    ax.set_title('MVF plot')
    ax.set_aspect('equal')
    plt.show()

def plot_Bullsplots_minmax_over_all_phases(AHAcube_base, AHAcube_midcavity, AHAcube_apex, cvi_prs, cvi_pcs):
    import matplotlib as mpl

    RS_values_base = np.nanmean(AHAcube_base[..., 0], axis=2)
    RS_values_midcavity = np.nanmean(AHAcube_midcavity[..., 0], axis=2)
    RS_values_apex = np.nanmean(AHAcube_apex[..., 0], axis=2)
    our_rs = 100 * np.concatenate([RS_values_base, RS_values_midcavity, RS_values_apex], axis=0)

    CS_values_base = np.nanmean(AHAcube_base[..., 1], axis=2)
    CS_values_midcavity = np.nanmean(AHAcube_midcavity[..., 1], axis=2)
    CS_values_apex = np.nanmean(AHAcube_apex[..., 1], axis=2)
    our_cs = 100 * np.concatenate([CS_values_base, CS_values_midcavity, CS_values_apex], axis=0)

    fig, axs = plt.subplots(2, 7, figsize=(30, 20), subplot_kw=dict(projection='polar'))

    # Set the colormap and norm to correspond to the data for which the colorbar will be used.
    # for Err we revert the colormap
    cmap_rr = mpl.cm.autumn_r
    cmap_cc = mpl.cm.autumn

    # ColorbarBase derives from ScalarMappable and puts a colorbar
    # in a specified axes, so it has everything needed for a
    # standalone colorbar.  There are many more kwargs, but the
    # following gives a basic continuous colorbar with ticks
    # and labels.
    for i in range(5):
        x0 = .04
        cb_width = .07
        cb_height = .02
        bp_delta = .142
        x = x0 + i * bp_delta
        y_rr = .55
        y_cc = .16

        # bp_labels = ['ED-MS', 'MS-ES', 'ES-PF', 'PF-MD', 'MD-ED']

        axl_rr = fig.add_axes([x, y_rr, cb_width, cb_height])
        norm_rr = mpl.colors.Normalize(vmin=our_rs[:, i].min(), vmax=our_rs[:, i].max())
        cb_rr = mpl.colorbar.ColorbarBase(axl_rr, cmap=cmap_rr, norm=norm_rr, orientation='horizontal')
        cb_rr.ax.tick_params(labelsize=10)
        cb_rr.set_label('Mean Err [%]', fontsize=10)
        myBullsplot(data=our_rs[:, i], cmap=cmap_rr, norm=norm_rr, ax=axs[0, i])
        axs[0, i].set_title('Phase '+str(i), fontsize=20)

        axl_cc = fig.add_axes([x, y_cc, cb_width, cb_height])
        norm_cc = mpl.colors.Normalize(vmin=our_cs[:, i].min(), vmax=our_cs[:, i].max())
        cb_cc = mpl.colorbar.ColorbarBase(axl_cc, cmap=cmap_cc, norm=norm_cc, orientation='horizontal')
        cb_cc.ax.tick_params(labelsize=10)
        cb_cc.set_label('Mean Ecc [%]', fontsize=10)
        myBullsplot(data=our_cs[:, i], cmap=cmap_cc, norm=norm_cc, ax=axs[1, i])

    # plot Circle Peak Bullsplots besides for comparison
    idx = 5
    x = x0 + idx * bp_delta
    axl_rr = fig.add_axes([x, y_rr, cb_width, cb_height])
    norm_rr = mpl.colors.Normalize(vmin=cvi_prs.min(), vmax=cvi_prs.max())
    cb_rr = mpl.colorbar.ColorbarBase(axl_rr, cmap=cmap_rr, norm=norm_rr, orientation='horizontal')
    cb_rr.ax.tick_params(labelsize=10)
    cb_rr.set_label('Mean Err [%]', fontsize=10)
    myBullsplot(data=cvi_prs, cmap=cmap_rr, norm=norm_rr, ax=axs[0, idx])
    axs[0, idx].set_title('cvi Peaks', fontsize=20)

    axl_cc = fig.add_axes([x, y_cc, cb_width, cb_height])
    norm_cc = mpl.colors.Normalize(vmin=cvi_pcs.min(), vmax=cvi_pcs.max())
    cb_cc = mpl.colorbar.ColorbarBase(axl_cc, cmap=cmap_cc, norm=norm_cc, orientation='horizontal')
    cb_cc.ax.tick_params(labelsize=10)
    cb_cc.set_label('Mean Ecc [%]', fontsize=10)
    myBullsplot(data=cvi_pcs, cmap=cmap_cc, norm=norm_cc, ax=axs[1, idx])

    # plot Our MinMax Peak Bullsplots for comparison
    idx = 6
    x = x0 + idx * bp_delta
    axl_rr = fig.add_axes([x, y_rr, cb_width, cb_height])
    norm_rr = mpl.colors.Normalize(vmin=np.amax(our_rs, axis=1).min(), vmax=np.amax(our_rs, axis=1).max())
    cb_rr = mpl.colorbar.ColorbarBase(axl_rr, cmap=cmap_rr, norm=norm_rr, orientation='horizontal')
    cb_rr.ax.tick_params(labelsize=10)
    cb_rr.set_label('Mean Err [%]', fontsize=10)
    myBullsplot(data=np.amax(our_rs, axis=1), cmap=cmap_rr, norm=norm_rr, ax=axs[0, idx])
    axs[0, idx].set_title('Our Peaks', fontsize=20)

    axl_cc = fig.add_axes([x, y_cc, cb_width, cb_height])
    norm_cc = mpl.colors.Normalize(vmin=np.amin(our_cs, axis=1).min(), vmax=np.amin(our_cs, axis=1).max())
    cb_cc = mpl.colorbar.ColorbarBase(axl_cc, cmap=cmap_cc, norm=norm_cc, orientation='horizontal')
    cb_cc.ax.tick_params(labelsize=10)
    cb_cc.set_label('Mean Ecc [%]', fontsize=10)
    myBullsplot(data=np.amin(our_cs, axis=1), cmap=cmap_cc, norm=norm_cc, ax=axs[1, idx])

    # plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    plt.show()



def plot_overlay_CMRwithMaskedStrainBySectorAHA(AHA, label_lvmyo, t, z_rel, z_abs, Err, vol_cube, com_cube,
                                                sector_masks_rot_midcavity, masks_rot_lvmyo):
    cmap_strain = 'jet'
    plt.figure()
    plt.imshow(Err[t, z_abs], cmap=cmap_strain)
    plt.imshow(roll_to_center(vol_cube[t, z_abs, ..., 0], com_cube[t, 1, 2], com_cube[t, 1, 1]), cmap='gray',
               alpha=1)
    # plt.imshow(masks_rot_lvmyo[t, z_abs], alpha=.5)
    plt.imshow(
        Err[t, z_rel] * (
                    (sector_masks_rot_midcavity[t, z_rel] == AHA) & (masks_rot_lvmyo[t, z_abs] == label_lvmyo)),
        cmap=cmap_strain,
        alpha=.5)
    plt.imshow(((sector_masks_rot_midcavity[t, z_rel] == AHA) & (masks_rot_lvmyo[t, z_abs] == label_lvmyo)),
               cmap='gray', alpha=.5)
    # plt.imshow(sector_masks_rot_midcavity[t, z_rel], alpha=.2)
    # plt.scatter(com_cube[t, 1], com_cube[t, 2], label='com')
    # plt.scatter(RVIP_cube[t, z_abs, 0, 0], RVIP_cube[t, z_abs, 0, 1], label='ant')
    # plt.scatter(RVIP_cube[t, z_abs, 1, 0], RVIP_cube[t, z_abs, 1, 1], label='inf')
    # plt.imshow(roll_to_center(mask_whole[t, z_abs, ..., 0], com_cube[t, 2], com_cube[t, 1]), alpha=.5, cmap='gray')
    plt.legend()
    plt.show()

