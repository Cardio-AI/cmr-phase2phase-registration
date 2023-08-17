import os, logging
from os import listdir
from os.path import isfile, join
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from src_julian.utils.myclasses import *

# set up logging
from src_julian.utils.skhelperfunctions import Console_and_file_logger
Console_and_file_logger('mvfviz/dmd_temp', logging.INFO)


def plot_volumecurve(x, counter_array):
    fig = plt.figure()
    ax = plt.axes()
    plt.plot(x, counter_array)
    for xc in x: ax.axvline(x=xc, color='k', linestyle='--', linewidth=0.5)
    ax.set_ylabel('Volume [Voxels]')
    ax.set_xlabel('Phase')
    ax.set_title('Sample Volume Curve')
    plt.show()

def stack_nii_flowfield(PATH_TO_NII_FILES, naming, Ntimesteps):
    # naming of nii files has to be _flow_0_.nii, _flow_1_.nii...
    list = []
    for t in range(Ntimesteps):
        name_ff = naming + str(t) + '_.nii'
        path_to_file = os.path.join(PATH_TO_NII_FILES, name_ff)
        data_ff = sitk.GetArrayFromImage(sitk.ReadImage(path_to_file))
        # data_ff=czxy
        data_ff = np.einsum('czxy->zxyc', data_ff) # should be the same shape as implemented by julian
        list.append(data_ff)

    array = np.array(list)

    return array

def stack_nii_masks(PATH_TO_NII_MASKS, naming, N_TIMESTEPS, ):
    # can then be saved to csv with the class "volume" and method "save as csv"
    list = []
    for t in range(N_TIMESTEPS):
        filename = naming + str(t) + '_.nii'
        path_to_file = os.path.join(PATH_TO_NII_MASKS, filename)
        arrayraw = sitk.GetArrayFromImage(sitk.ReadImage(path_to_file))
        list.append(arrayraw)

    maskstack = np.array(list)

    return maskstack

def stack_nii_volume(PATH_TO_NII_FILES, naming, Ntimesteps):
    # naming of nii files has to be _flow_0_.nii, _flow_1_.nii...
    # naming can be stated a an input to the method
    # i.e. naming = '_cmr_full_'
    list = []
    for t in range(Ntimesteps):
        name_vol = naming + str(t) + '_.nii'
        path_to_file = os.path.join(PATH_TO_NII_FILES, name_vol)
        img_vol = sitk.ReadImage(path_to_file)
        data_vol = sitk.GetArrayFromImage(img_vol)
        data_vol = np.einsum('xyz->zyx', data_vol)
        list.append(data_vol)

    volume = np.array(list)

    return volume[..., np.newaxis]

def Grid2D_getMVlengths(MVF, N, lengthdim):
    # given MVF must have 3D MV
    # lets assume input was 224x224x3
    # output is of same shape but contains the length in the three MVF entry locations
    # so, this function calculates the vector lengths in 2D and 3D space

    # slicing the field, take every Nth value
    MVF = MVF[::N, ::N]
    MVFlengths = np.zeros_like(MVF)  # create empty array which looks like sliced MVF

    for i in range(MVF.shape[0]):  # column-wise to the right
        for j in range(MVF.shape[1]):  # row-wise downwards
            z = MVF[j, i, 0]
            y = MVF[j, i, 1]
            x = MVF[j, i, 2]
            if lengthdim == 3:
                MVFlengths[j, i] = np.linalg.norm(MVF[j, i])  # calculate MV length in 2D
            elif lengthdim == 2:
                MVFlengths[j, i] = np.linalg.norm(MVF[j, i])  # calculate MV length in 3D
            else:
                print("Invalid dimension argument for MV length calculation.")

    return MVFlengths

def calculate_vector_magnitude(vector):
    magnitude = np.linalg.norm(vector)
    return magnitude

def calculate_parameters(ff):
    average_z = []
    for t in range(ff.nt): average_z.append(np.average(ff.Data[t, ..., 0]))
    average_z = np.array(average_z)

    average_y = []
    for t in range(ff.nt): average_y.append(np.average(ff.Data[t, ..., 1]))
    average_y = np.array(average_y)

    average_x = []
    for t in range(ff.nt): average_x.append(np.average(ff.Data[t, ..., 2]))
    average_x = np.array(average_x)

    average_magnitudes = []
    mag_array = np.sqrt(ff.Data[..., 0]**2 + ff.Data[..., 1]**2 + ff.Data[..., 2]**2)
    mag_array = mag_array[..., np.newaxis]
    for t in range(ff.nt): average_magnitudes.append(np.average(mag_array[t, ..., 0]))  # calculate average value for every phase
    average_magnitudes = np.array(average_magnitudes)

    return average_z, average_y, average_x, average_magnitudes

def plot1(PATH_TO_TODAYSFOLDER):
    # (1,3) with parameters and base, midcavity, apex
    # set
    Ntimesteps = 5
    Z_SPACING = 1
    # mpl.style.use('seaborn')

    ff = stack_nii_flowfield(PATH_TO_TODAYSFOLDER, Ntimesteps)

    ff_base = mvf(ff[:, 42:64, ...], '4Dt', Z_SPACING)
    ff_midcavity = mvf(ff[:, 20:42, ...], '4Dt', Z_SPACING)
    ff_apex = mvf(ff[:, 0:20, ...], '4Dt', Z_SPACING)

    base = calculate_parameters(ff_base)
    midcavity = calculate_parameters(ff_midcavity)
    apex = calculate_parameters(ff_apex)

    x = ['ED', 'MS', 'ES', 'PF', 'MD']
    parameters = ['global average displacement z', 'global average displacement y', 'global average displacement x',
                  'global average magnitude']
    fig, ax = plt.subplots(1, 3, figsize=(15, 7), sharey=True)

    for idx, num in enumerate(base):
        ax[0].plot(num, label=parameters[idx])
    for idx, num in enumerate(midcavity):
        ax[1].plot(num, label=parameters[idx])
    for idx, num in enumerate(apex):
        ax[2].plot(num, label=parameters[idx])

    ax[0].set_ylabel('Values')
    ax[0].set_title('Base')
    ax[1].set_title('Mid-Cavity')
    ax[2].set_title('Apex')

    for xc in x: ax[0].axvline(x=xc, color='k', linestyle='--', linewidth=0.5), \
                 ax[1].axvline(x=xc, color='k', linestyle='--', linewidth=0.5), \
                 ax[2].axvline(x=xc, color='k', linestyle='--', linewidth=0.5)

    ax[1].legend(parameters, loc="lower center", bbox_to_anchor=(0.5, -0.3))
    fig.subplots_adjust(bottom=0.25)

    plt.show()
    INFO('Hi')

def get_segmentationarray(path_to_nii_folder, naming, Ntimesteps):
    '''
    CAVE: check naming of nii segmentation files

    takes multidimensional volume i.e. 5,64,128,128,1
    checks for every element if other than 0
    returns boolean array containing True or False for every element
    True == there is a segmented object
    False == value here is 0, thus nothing was segmented here
    '''

    # stack nii segmentations
    list = []
    for t in range(Ntimesteps):
        name_vol = naming + str(t) + '_.nii'
        img_vol = sitk.ReadImage(path_to_nii_folder + name_vol)
        data_vol = sitk.GetArrayFromImage(img_vol)
        data_vol = np.einsum('xyz->zyx', data_vol)
        list.append(data_vol)

    # append the segmentation value to the array so that we have an array with seg values: [...,0]
    segmentation_stacked = np.array(list)[..., np.newaxis]

    # check, in which slices we do have segmentations
    # get a boolean array where anything was segmented
    boolarray = (segmentation_stacked != 0)

    return boolarray

def myffgenerator():
    """
        before quiver plotting, this method may return 224x224 resolution sample flowfield with single vectors
        only for testing
    """

    start = 0
    end = 224
    nsteps = end-start
    x = np.linspace(start, end, nsteps)
    y = np.linspace(start, end, nsteps)
    X, Y = np.meshgrid(x, y)

    # horizontal arrows
    # u = 10*np.zeros_like(X)
    # v = 10*np.ones_like(Y)

    # diagonal arrows
    # u = 10*np.ones_like(X)
    # v = 10*np.ones_like(Y)

    # circular2
    # u = -Y / np.sqrt(X ** 2 + Y ** 2)
    # v = X / np.sqrt(X ** 2 + Y ** 2)

    # varying length
    # u = np.zeros_like(X)
    # v = np.zeros_like(Y)
    # v[80, :110] = 20
    # v[80, 110:] = 100
    # v[80, :110] = 20
    # v[80, 110:] = 100
    # v[100, :110] = 20
    # v[100, 110:] = 100

    # u, v = np.meshgrid(x, y)
    # u = 0.01*u
    # v = 0.01*v

    # empty
    # u = 10*np.ones_like(X)
    # v = np.zeros_like(Y)

    # four circles
    # x = np.arange(0,2*np.pi+2*np.pi/20,2*np.pi/20)
    # y = np.arange(0, 2*np.pi + 2*np.pi / 20, 2*np.pi/20)
    # X, Y = np.meshgrid(x, y)
    # u = np.sin(X)*np.cos(Y)
    # v = -np.cos(X)*np.sin(Y)

    # now testing
    u = np.zeros_like(X)
    v = np.zeros_like(Y)

    v[200,10]=10

    # v[110,110] = 10
    #
    # v[130,110] = 10

    return X,Y,u,v


# def save_ff_as_nii(ff, format, PATH_TO_OUTPUT_FOLDER, name):
#     """
#     flowfield convention:
#     2D: x,c
#     2Dt: t,x,c
#     3D: y,x,c
#     3Dt: t,y,x,c
#     4D: z,y,x,c
#     4Dt: t,z,y,x,c
#     flowfield gets rearranged corresponding for correct mitk reading
#     """
#
#     if format == '3D' and ff.ndim == 3:
#         ff_rearranged = np.einsum('yxc->cxy', ff)
#         INFO('shape of ff after rearrangement ready for nii export (c, x, y): {}'.format(ff_rearranged.shape))
#     elif format == '3Dt' and ff.ndim == 4:
#         ff_rearranged = np.einsum('tyxc->cxyt', ff)
#         INFO('shape of ff after rearrangement ready for nii export (c, x, y, t): {}'.format(ff_rearranged.shape))
#     elif format == '4D' and ff.ndim == 4 :
#         ff_rearranged = np.einsum('zyxc->cxyz', ff)
#         INFO('shape of ff after rearrangement ready for nii export (c, x, y, z): {}'.format(ff_rearranged.shape))
#     elif format == '4Dt' and ff.ndim == 5 :
#         ff_rearranged = np.einsum('tzyxc->cxyzt', ff)
#         INFO('shape of ff after rearrangement ready for nii export (c, x, y, z, t): {}'.format(ff_rearranged.shape))
#         INFO('WARNING: 4Dt array display is not possible in MITK Diffusion')
#
#     back_ff2img = sitk.GetImageFromArray(ff_rearranged, isVector=False)  # isVector=False important for adequate writing of sitk
#     sitk.WriteImage(back_ff2img, PATH_TO_OUTPUT_FOLDER + name + '.nii')
#     INFO('ff {} has been successfully saved.'.format(name))


# def getarrayspecs(array, format, isFlowfield):
#     if isFlowfield:
#         if format == '4D' and array.ndim == 4:
#             z, y, x, c = array.shape
#             return z, y, x, c
#         if format == '4Dt' and array.ndim == 5:
#             t, z, y, x, c = array.shape
#             return t, z, y, x, c
#     elif not isFlowfield:
#         if format == '3D' and array.ndim == 4:
#             z, y, x, value = array.shape
#             return z, y, x, value
#         if format == '3Dt' and array.ndim == 4:
#             t, z, y, x = array.shape
#             return t, z, y, x


def getMDcineSAXbyMDmatrix(PATH_TO_FOLDER_NRRD_SAX_CRFALSE, PATH_TO_FOLDER_NRRD_LGE_CRFALSE, PATH_TO_OUTPUTFOLDER, PATH_TO_XLS_FILE):
    """
    for registration visual inspection between sax cine MRI and LGE images, we want MD timestep of cine MRI sax in timestep 0 nrrd file.
    """

    # small preprocessing, preparing
    df = pd.read_excel(PATH_TO_XLS_FILE, 'lge matrices', dtype='object')
    npy = df.to_numpy()
    npatients_lgetable, ncols = npy.shape

    # get lge patients. for those, we want the sax files changed.
    lgepatientslist = [f for f in listdir(PATH_TO_FOLDER_NRRD_LGE_CRFALSE) if isfile(join(PATH_TO_FOLDER_NRRD_LGE_CRFALSE, f))]

    for i in range(len(lgepatientslist)):
        if os.path.splitext(lgepatientslist[i])[0][-5:] == 'clean': # filename without path and file extension
            patientname_nrrdfile_lge = lgepatientslist[i][:-18] # get patientname; cut file extension and _volume_clean from string of filename
            # check where in excel sheet this patient stands
            for j in range(npatients_lgetable):
                patientname_matrixtable = npy[j, 0]
                if patientname_nrrdfile_lge.lower() == patientname_matrixtable.lower():
                    MD_matrix = npy[:, 2] # get MD labels from Tarique
                    # validation
                    if npatients_lgetable != MD_matrix.size:
                        print('WARNING: number of mid-diastolic labels and number of patients do not match.')

                    imgsax = sitk.ReadImage(PATH_TO_FOLDER_NRRD_SAX_CRFALSE + lgepatientslist[i])
                    arraysax = sitk.GetArrayFromImage(imgsax)
                    arraysax[0, :, :, :] = arraysax[MD_matrix[j] - 1, :, :, :]

                    # back converting and writing
                    newimg = sitk.GetImageFromArray(arraysax, isVector=False)
                    newimg.CopyInformation(imgsax)
                    sitk.WriteImage(newimg, PATH_TO_OUTPUTFOLDER + lgepatientslist[i])
                    break


def getcorrectedLGEbymatrix(inputfolderpath, outputfolderpath, excelsheetpath):
    """
        inputfolderpath: PATH_TO_FOLDER_NRRD_LGE_CRFALSE
        create nrrd files for LGE patients which contain corrected images in timestep 0.
        the input folder path can contain the volume_mask nrrd files also, but they will not be handled.
        the excelsheet given here must contain patient IDs in the first column and lgematrices in the second column.

        i.e. function calls:
        path_to_lgecorrected = '/mnt/ssd/julian/data/interim/2021.04.25_lgecorrected/'
        path_to_saxmd0 = '/mnt/ssd/julian/data/interim/2021.04.25_saxmd0/'
        jk.getcorrectedLGEbymatrix(PATH_TO_FOLDER_NRRD_LGE_CRFALSE, path_to_lgecorrected, PATH_TO_XLS_FILE)
        jk.getMDcineSAXbyMDmatrix(PATH_TO_FOLDER_NRRD_SAX_CRFALSE, PATH_TO_FOLDER_NRRD_LGE_CRFALSE, path_to_saxmd0, PATH_TO_XLS_FILE)
    """

    # folder paths
    PATH_TO_FOLDER_NRRD_LGE_CRFALSE = inputfolderpath
    PATH_TO_OUTPUTFOLDER = outputfolderpath
    PATH_TO_XLS_FILE = excelsheetpath

    # small preprocessing, preparing
    df = pd.read_excel(PATH_TO_XLS_FILE, 'lge matrices', dtype='object')
    npy = df.to_numpy()
    npatients_lgetable, ncols = npy.shape

    # read all files that are in the input folder
    allfilenames = [f for f in listdir(PATH_TO_FOLDER_NRRD_LGE_CRFALSE) if isfile(join(PATH_TO_FOLDER_NRRD_LGE_CRFALSE, f))]

    for i in range(len(allfilenames)):
        if os.path.splitext(allfilenames[i])[0][-5:] == 'clean': # filename without path and file extension
            patientname_nrrdfile = allfilenames[i][:-18] # get patientname; cut file extension and _volume_clean from string of filename
            # check where in excel sheet this patient stands
            for j in range(npatients_lgetable):
                patientname_matrixtable = npy[j, 0]
                if patientname_nrrdfile.lower() == patientname_matrixtable.lower():
                    str_matrix = str(npy[j, 1])
                    lge_matrix = np.array([int(x) for x in str_matrix]) # get integer numbers from parsed string numbers
                    imglge = sitk.ReadImage(PATH_TO_FOLDER_NRRD_LGE_CRFALSE + allfilenames[i])
                    img = getcorrectedLGE(imglge, lge_matrix) # sorting function

                    # writing new file
                    sitk.WriteImage(img, PATH_TO_OUTPUTFOLDER + allfilenames[i])
                    break

def getcorrectedLGE(img_original_lge, lge_index):
    """
    this method takes an existing lge img read from a nrrd file and returns the image, but in timestep 0
    written will be the corrected lge images from the patient acquisition.
    the lge_index array was acquired by hand as the lge images in different timesteps seem to be in diffuse order.
    """

    arraylge = sitk.GetArrayFromImage(img_original_lge)
    ntlge, nzlge, nylge, nxlge = arraylge.shape

    # validation
    if nzlge != lge_index.size :
        print('WARNING: image slices amount and matrix entries do not match.')

    # create new lge with corrected images in timestep channel 0
    for i in range(nzlge):
        arraylge[0, i, :, :] = arraylge[lge_index[i], i, :, :]

    img_correctedimages_lge = sitk.GetImageFromArray(arraylge, isVector=False)
    img_correctedimages_lge.CopyInformation(img_original_lge)

    return img_correctedimages_lge


# https://terbium.io/2017/12/matplotlib-3d/
def normalize(arr):
    arr_min = np.min(arr)
    return (arr - arr_min) / (np.max(arr) - arr_min)


# https://terbium.io/2017/12/matplotlib-3d/
def show_histogram(values):
    n, bins, patches = plt.hist(values.reshape(-1), 50, density=True)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    for c, p in zip(normalize(bin_centers), patches):
        plt.setp(p, 'facecolor', cm.viridis(c))

    plt.show()


# https://terbium.io/2017/12/matplotlib-3d/
def make_ax(grid=False):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.grid(grid)
    return ax


# https://terbium.io/2017/12/matplotlib-3d/
def explode(data):
    shape_arr = np.array(data.shape)
    size = shape_arr[:3] * 2 - 1
    exploded = np.zeros(np.concatenate([size, shape_arr[3:]]), dtype=data.dtype)
    exploded[::2, ::2, ::2] = data
    return exploded


# https://terbium.io/2017/12/matplotlib-3d/
def expand_coordinates(indices):
    x, y, z = indices
    x[1::2, :, :] += 1
    y[:, 1::2, :] += 1
    z[:, :, 1::2] += 1
    return x, y, z


# https://terbium.io/2017/12/matplotlib-3d/
def plot_cube(cube, angle=320):
    cube = normalize(cube)

    facecolors = cm.viridis(cube)
    facecolors[:, :, :, -1] = cube
    facecolors = explode(facecolors)

    filled = facecolors[:, :, :, -1] != 0
    x, y, z = expand_coordinates(np.indices(np.array(filled.shape) + 1))

    fig = plt.figure(figsize=(30 / 2.54, 30 / 2.54))
    ax = fig.gca(projection='3d')
    ax.view_init(30, angle)
    # ax.set_xlim(right=IMG_DIM * 2)
    # ax.set_ylim(top=IMG_DIM * 2)
    # ax.set_zlim(top=IMG_DIM * 2)

    ax.voxels(x, y, z, filled, facecolors=facecolors, shade=False)
    plt.show()


def get_test_data(delta=0.05):
    """
    Return a tuple X, Y, Z with a test data set.
    """
    x = y = np.arange(-3.0, 3.0, delta)
    X, Y = np.meshgrid(x, y)

    Z1 = np.exp(-(X**2 + Y**2) / 2) / (2 * np.pi)
    Z2 = (np.exp(-(((X - 1) / 1.5)**2 + ((Y - 1) / 0.5)**2) / 2) /
          (2 * np.pi * 0.5 * 1.5))
    Z = Z2 - Z1

    X = X * 10
    Y = Y * 10
    Z = Z * 500
    return X, Y, Z


def getZZarray(nx, ny, desired_z, zshift, zspacing):
    """
    the val array can contain multiple values. then, for every value, an array will be returned.
    returns 2D arrays of size nx, ny with values filled as val
    the first dimension of the returned array states how many of these are returned.
    function originally created to return one ZZarray for all questioned SAX and one for all questioned LGE arrays.
    """
    n_returnedarrays = desired_z.size

    # create empty array
    ZZ = np.zeros((n_returnedarrays, nx, ny))

    for i in range(n_returnedarrays):
        ZZ[i] = zspacing*np.full((nx, ny), desired_z[i]) + zshift

    return ZZ

def normalizecolors(array, min, max):
    """
    takes a color array and desired min and max value of new colorranges
    """

    array_maxvalue = np.max(array)
    array_minvalue = np.min(array)
    array = max * (array/(array_maxvalue-array_minvalue)) + min
    return array


def create2Dmesh(xmin, xmax, xspacing, ymin, ymax, yspacing):
    """
    creates a 2D mesh out of boundary values
    """

    X = np.arange(xmin, xmax, xspacing)
    Y = np.arange(ymin, ymax, yspacing)
    XX, YY = np.meshgrid(X, Y)
    return XX, YY

# https://www.programcreek.com/python/example/123389/SimpleITK.AffineTransform
def get(self, **kwargs):
    """
    Returns the sitk transform based on the given parameters.
    :param kwargs: Must contain either 'image', or 'input_size' and 'input_spacing', which define the input image physical space.
    :return: The sitk.AffineTransform().
    """
    input_size, input_spacing, _, _ = self.get_image_size_spacing_direction_origin(**kwargs)
    current_scale = []
    for i in range(self.dim):
        if self.output_size[i] is None or self.output_spacing[i] is None:
            continue
        else:
            current_scale.append((input_size[i] * input_spacing[i]) / (self.output_size[i] * self.output_spacing[i]))
    max_scale = max(current_scale)
    current_scale = []
    for i in range(self.dim):
        if i in self.ignore_dim:
            current_scale.append(1.0)
        else:
            current_scale.append(max_scale)
    return self.get_scale_transform(self.dim, current_scale)



# def Grid2D_MV2Dor3D(MVF, N):
#     # MVF vector ordering has to be z,x,y
#     # N = slicing, take every Nth value
#     # a function that handles 2D grids such as 224x224 and plots motion vectors 2D on the grid in the xy plane.
#     # it is not yet stated how z values of 3D MV should be visualized here.
#
#     # example call:
#     # N = 1
#     # slice = 15
#     # fig, ax = plt.subplots()
#     # xx, yy, Fx, Fy = Grid2D_MV2Dor3D(mvf_ED[slice], N)
#     # plt.quiver(xx, yy, Fx, Fy, units='xy', angles='xy')
#     # ax.set_title('MVF plot')
#     # plt.show()
#
#     # define starting values for axes
#     starting_value = 0
#     zstart = starting_value
#     xstart = starting_value
#     ystart = starting_value
#
#     # checking grid and motion vector dimensions; plausibility of input
#     if MVF.ndim != 3:
#         print("Error: Grid is not 2D")
#         return
#     if MVF.ndim == 3:  # the grid is an image, MVF given as x,y,v
#         print("Grid: 2D")
#     if MVF.shape[-1] == 2:  # MVs are 2D
#         # if the motion vectors input are in 2D coordinates, we are filling with 0 z values
#         print("MVs: 2D")
#         print("Vectors are now filled with zeros as z-values")
#         xs, ys, vs = MVF.shape
#         vss = vs + 1 # increase MVs dimensions by 1, resp. z-values
#         b = np.zeros((xs, ys, vss))
#         b[:, :, 1:] = MVF
#         MVF = b
#     elif MVF.shape[-1] == 3:  # MVs are 3D
#         print("MVs: 3D")
#
#     # calculation of grid coordinates begins
#     print("Calculating now.")
#
#     zdir = MVF[..., 0]  # motion vectors z directions
#     ydir = MVF[..., 1]  # motion vectors y directions
#     xdir = MVF[..., 2]  # motion vectors x directions
#
#     # checking how many values we need for the grid
#     nz = MVF.shape[0]  # grid is 2D here, so we do not have z-values!
#     nx = MVF.shape[0]
#     ny = MVF.shape[1]
#
#     # calculation of the last axes values on the grid
#     zend = MVF.shape[0] - 1  # define ticks. as we go from 0 on, the last value of z has to be reduced by 1 for the axes
#     xend = MVF.shape[0] - 1
#     yend = MVF.shape[1] - 1
#
#     # slicing the field, take every Nth value
#     Fz = zdir[::N, ::N]
#     Fx = xdir[::N, ::N]
#     Fy = ydir[::N, ::N]
#     nrows, ncols = Fx.shape
#
#     zlin = np.linspace(zstart, zend, nz)  # these linspace values are later used for meshgrid
#     xlin = np.linspace(xstart, xend, ncols)
#     ylin = np.linspace(ystart, yend, nrows)
#
#     # calculating the meshgrid which equals the vectors origins
#     xx, yy = np.meshgrid(xlin, ylin, indexing='xy')
#     zz = np.meshgrid(zlin, nz, indexing='xy')
#
#     #-----------------------------------#
#     # here we have to state what should happen with the z values of the MVs on a 2D grid!
#     # -----------------------------------#
#
#     # plotting
#     # method 1: mlab.quiver3d by Mayavi
#     # fig = plt.figure()
#     # mlab.quiver3d(xx, yy, zz, xdir, ydir, zdir)
#     # mlab.show()
#
#     # method 2: matplotlib
#     # fig, ax = plt.subplots()
#     # ax.quiver(xx, -yy, Fx, Fy, units='xy', angles='xy')  # negative yy sets origin at the top left
#     # ax.set_title('MVF plot')
#     # plt.grid(which='major', axis='both')
#     # plt.show()
#
#     return xx,yy,Fx,Fy


def Grid3D_MV2Dor3D(MVF, N):
    from mayavi import mlab

    # define starting values for axes
    zstart = 0
    xstart = 0
    ystart = 0

    if MVF.ndim != 4:
        print("Error: Grid is not 3D")
        return
    if MVF.ndim == 4:  # the grid is a volume given by x,y,z,v
        print("Grid: 3D")
    if MVF.shape[-1] == 2:  # MVs are 2D
        print("MVs: 2D")
        print("Vectors are now filled with zeros as z-values")
        zs, xs, ys, vs = MVF.shape
        vss = vs + 1  # increase MVs dimensions by 1, resp. z-values
        b = np.zeros((zs, xs, ys, vss))
        b[:, :, :, 1:] = MVF
        MVF = b
    elif MVF.shape[-1] == 3:  # MVs are 3D
        print("MVs: 3D")

    print("Calculating now.")

    zdir = MVF[..., 0]  # motion vectors z directions
    xdir = MVF[..., 2]
    ydir = MVF[..., 1]

    nz = MVF.shape[0]  # how many z values do we have originally?
    nx = MVF.shape[2]
    ny = MVF.shape[1]

    # axes end values
    zend = MVF.shape[0] - 1  # as we go from 0 on, the last value of z has to be reduced by 1
    xend = MVF.shape[2] - 1
    yend = MVF.shape[1] - 1

    # slicing the field in the planes of x and y where the many vectors of the motion are
    zdirsliced = zdir[:, ::N, ::N]  # we will keep the z values at first because the slicer works in the planes
    xdirsliced = xdir[:, ::N, ::N]
    ydirsliced = ydir[:, ::N, ::N]
    nzsliced, nxsliced, nysliced = xdirsliced.shape  # the new grid

    zlin = np.linspace(zstart, zend, nzsliced)  # these linspace values are later used for meshgrid
    xlin = np.linspace(xstart, xend, nxsliced)
    ylin = np.linspace(ystart, yend, nysliced)

    # calculating the meshgrid which equals the vectors origins
    zz, xx, yy = np.meshgrid(zlin, xlin, ylin, indexing='xy')

    # reshape the meshgrid output to the original shape (where the last entry has to be excluded because its the vectors)
    zz = np.reshape(zz, xdirsliced.shape)
    xx = np.reshape(xx, xdirsliced.shape)
    yy = np.reshape(yy, xdirsliced.shape)

    # plotting
    fig = plt.figure()
    # ----------------------------------------------------------------------------- #
    # method 1: ax.quiver
    # ax = fig.gca(projection='3d')
    # ax.quiver(xi, yi, zi, xdir, ydir, zdir, length=0.1, normalize=True)
    # ax.set_zlabel('Z'), ax.set_xlabel('X'), ax.set_ylabel('Y')
    # ax.set_title('MVF plot')
    # plt.grid(which='major', axis='both')
    # plt.show()
    # ----------------------------------------------------------------------------- #
    # method 2: mlab.quiver3d by Mayavi
    mlab.quiver3d(zz, xx, yy, zdirsliced, xdirsliced, ydirsliced)
    mlab.show()
    # ----------------------------------------------------------------------------- #

    # def savevolumeascsv(data, i, PATH_TO_CSV_OUTPUT_FOLDER, Z_SPACING, name):
    #    """
    #            takes data of form data[30,128,128,1] and writes this as a data frame into a csv file
    #            with six columns resp. x,y,z,value for further processing in ParaView
    #            i is the time step
    #
    #            CAVE: if data is not z,y,x,c but i.e. x,y,z you can transpose the values to make einsum obsolete to order the array!
    #
    #            FUNCTION CALL:
    #
    # PATH_TO_CSV_OUTPUT_FOLDER = '/mnt/ssd/julian/data/raw/paraview/examples/myexamplefiles/'
    #
    # data = np.load(PATH_TO_INPUT)
    # patient = data[1] # first patient: 5,30,128,128,3
    #
    # ntimesteps = 5
    # name = "volume"
    # for i in range(ntimesteps):
    #    phase = patient[i] # if the original dataset contains more than one patient, we have an array called patient
    #    savevolumeascsv(phase, i, PATH_TO_CSV_OUTPUT_FOLDER, 1, name)
    #
    #    """
    #
    #    z_data, y_data, x_data, c_data = getarrayspecs(data, "3D", isFlowfield=False)
    #
    #    x_ = np.linspace(0, x_data, x_data)
    #    y_ = np.linspace(0, y_data, y_data)
    #    z_ = np.linspace(0, z_data, z_data)
    #    X, Y, Z = np.meshgrid(x_, y_, z_)
    #    X = X.T
    #    Y = Y.T
    #    Z = Z.T
    #    x = X.ravel()
    #    y = Y.ravel()
    #    z = Z.ravel()
    #
    #    z = np.multiply(z, Z_SPACING)
    #
    #    values = data[:, :, :, 0]
    #    # values = values.T if data is not "einsummed" to have the order z,y,x,c
    #    values = values.ravel()
    #
    #
    #    df = pd.DataFrame({"x": x, "y": y, "z": z, "value": values})
    #    df.to_csv(PATH_TO_CSV_OUTPUT_FOLDER + name + ".csv." + str(i), index=False)
    #
    #    INFO('volume successfully saved as csv.')

    # def saveffascsv(data, i, PATH_TO_CSV_OUTPUT_FOLDER, Z_SPACING, name):
    #     """
    #             takes data of form data[30,128,128,3] and writes this as a data frame into a csv file
    #             with six columns resp. x,y,z,Fx,Fy,Fz for further processing in ParaView
    #             i is the time step
    #             FUNCTION CALL:
    #             ntimesteps = 5
    #             for i in range(ntimesteps):
    #                 phase = patient[i] # if the original dataset contains more than one patient, we have an array called patient
    #                 saveffascsv(phase, i, PATH_TO_CSV_OUTPUT_FOLDER, Z_SPACING, name)
    #     """
    #
    #     z_data, y_data, x_data, c_data = getarrayspecs(data, "4D", isFlowfield=True)
    #
    #     x_ = np.linspace(0, x_data, x_data)
    #     y_ = np.linspace(0, y_data, y_data)
    #     z_ = np.linspace(0, z_data, z_data)
    #     X, Y, Z = np.meshgrid(x_, y_, z_)
    #     X = X.T
    #     Y = Y.T
    #     Z = Z.T
    #     x = X.ravel()
    #     y = Y.ravel()
    #     z = Z.ravel()
    #
    #     z = z * Z_SPACING
    #
    #     z_direction = data[:, :, :, 0]
    #     y_direction = data[:, :, :, 1]
    #     x_direction = data[:, :, :, 2]
    #     Fz = z_direction.ravel()
    #     Fy = y_direction.ravel()
    #     Fx = x_direction.ravel()
    #
    #     df = pd.DataFrame({"x": x, "y": y, "z": z, "Fx": Fx, "Fy": Fy, "Fz": Fz})
    #
    #     df.to_csv(PATH_TO_CSV_OUTPUT_FOLDER + name + ".csv." + str(i), index=False)
    #
    #     INFO('csv successfully saved.')

# https://gist.github.com/LyleScott/e36e08bfb23b1f87af68c9051f985302
def rotate_around_point_lowperf(point, radians, origin=(0, 0)):
    import math
    """Rotate a point around a given point.

    I call this the "low performance" version since it's recalculating
    the same values more than once [cos(radians), sin(radians), x-ox, y-oy).
    It's more readable than the next function, though.
    """
    x, y = point
    ox, oy = origin

    qx = ox + math.cos(radians) * (x - ox) + math.sin(radians) * (y - oy)
    qy = oy + -math.sin(radians) * (x - ox) + math.cos(radians) * (y - oy)

    return qx, qy

# https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point
def rotate(image, angle, center = None, scale = 1.0):
    import cv2
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated

def my_save_list_as_txt(path, name, data):
    with open(path+name+'.txt','w') as f:
        for item in data: f.write("%s\n" % item)