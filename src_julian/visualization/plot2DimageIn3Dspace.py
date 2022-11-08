# define logging and working directory
from ProjectRoot import change_wd_to_project_root
change_wd_to_project_root()

# import helper functions
import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
import logging
from logging import info as INFO
from src_julian.utils.skhelperfunctions import Console_and_file_logger
from matplotlib import cm
import src_julian.utils.myhelperfunctions as jk

# set up logging
Console_and_file_logger('mvfviz/dmd_temp', logging.INFO)

# define relevant paths to data for these tests
PATH_TO_FOLDER_DCM_SAX = '/mnt/ssd/julian/data/raw/DMD/AC_20181203_DMD201812031039101/SAx_AC_20181203_DMD201812031039101.2.276.0.7230010.3.1.2.15116645.3956.1578350918.427299/series0301-unknown/'
PATH_TO_DICOM_SAMPLE_1 = '/mnt/ssd/julian/data/raw/DMD/AC_20181203_DMD201812031039101/SAx_AC_20181203_DMD201812031039101.2.276.0.7230010.3.1.2.15116645.3956.1578350918.427299/series0301-unknown/img0001-126.491.dcm'
PATH_TO_DICOM_SAMPLE_2 = '/mnt/ssd/julian/data/raw/DMD/AC_20181203_DMD201812031039101/SAx_AC_20181203_DMD201812031039101.2.276.0.7230010.3.1.2.15116645.3956.1578350918.427299/series0301-unknown/img0202-66.4912.dcm'
PATH_TO_NRRD_SAX_CRFALSE = '/mnt/ssd/julian/data/interim/2021.04.20_allcopyrotationfalse/DMD/sax/ac_20180301_volume_clean.nrrd'
PATH_TO_NRRD_LGE_CRFALSE = '/mnt/ssd/julian/data/interim/2021.04.20_allcopyrotationfalse/DMD/lge/ac_20180301_volume_clean.nrrd'

# get data, get arrays, plot shapes
imgsax = sitk.ReadImage(PATH_TO_NRRD_SAX_CRFALSE)
imglge = sitk.ReadImage(PATH_TO_NRRD_LGE_CRFALSE)
arraysax = sitk.GetArrayFromImage(imgsax)
arraylge = sitk.GetArrayFromImage(imglge)
INFO('shape of nrrd sax (t,z,y,x): {}'.format(arraysax.shape))
INFO('shape of nrrd lge (t,z,y,x): {}'.format(arraylge.shape))

# print meta data
INFO('sax origin: {}'.format(imgsax.GetOrigin()))
INFO('lge origin: {}'.format(imglge.GetOrigin()))
INFO('sax spacing: {}'.format(imgsax.GetSpacing()))
INFO('lge spacing: {}'.format(imglge.GetSpacing()))
INFO('sax direction: {}'.format(imgsax.GetDirection()))
INFO('sax direction: {}'.format(imglge.GetDirection()))

# get origins
xoriginsax = imgsax.GetOrigin()[0]
yoriginsax = imgsax.GetOrigin()[1]
zoriginsax = imgsax.GetOrigin()[2]
xoriginlge = imglge.GetOrigin()[0]
yoriginlge = imglge.GetOrigin()[1]
zoriginlge = imglge.GetOrigin()[2]

# get spacings
xspacingsax = imgsax.GetSpacing()[0]
yspacingsax = imgsax.GetSpacing()[1]
zspacingsax = imgsax.GetSpacing()[2]
xspacinglge = imglge.GetSpacing()[0]
yspacinglge = imglge.GetSpacing()[1]
zspacinglge = imglge.GetSpacing()[2]
if xspacingsax != xspacinglge or yspacingsax != yspacinglge or xspacingsax != yspacingsax or xspacinglge != yspacinglge:
    print('WARNING: intra-patient x- and y-spacings differ.')

# calculate shifts. will be applied to LGE data
xshift = abs(xoriginsax - xoriginlge)
yshift = abs(yoriginsax - yoriginlge)
zshift = abs(zoriginsax - zoriginlge)

# read number of entries from array, assign to time, dimensions
# inplane resolution should be the same here!
ntsax, nzsax, nysax, nxsax = arraysax.shape
ntlge, nzlge, nylge, nxlge = arraylge.shape
if nxsax != nxlge:
    print('WARNING: inplane resolutions differ.')

# create mesh. in-plane resolutions are the same for sax and lge.
# the grids are the same for comparing sax and lge intra-patient
xmin = 0
xmax = nxsax
xspacing = 1 # by adding the spacing relatio as a factor here should be able to catch intra-patient x-y-spacing differences
ymin = 0
ymax = nysax
yspacing = 1
XXsax, YYsax = jk.create2Dmesh(xmin, xmax, xspacing, ymin, ymax, yspacing)

# shiftings in xy-plane, to align the sax images at the 0,0 origin and shift the LGE according to this.
if xoriginsax < xoriginlge :
    XXlge = XXsax + xshift
    YYlge = YYsax + yshift
elif xoriginsax > xoriginlge :
    XXlge = XXsax - xshift
    YYlge = YYsax - yshift

# shifting in zplane
zshiftsax = 0
if zoriginsax < zoriginlge :
    zshiftlge = zshift
if zoriginsax > zoriginlge :
    zshiftlge = -1 * zshift

# wishlist for christmas
stride = 5
t_desired_sax = np.array([0])
slice_desired_sax = np.array([5])
t_desired_lge = np.array([0])
slice_desired_lge = np.array([5])
if t_desired_sax.size != t_desired_sax.size or t_desired_lge.size != t_desired_lge.size :
    print('WARNING: your input is invalid. not every timestep is a slice assigned.')

# get the number of images to be plotted, convenience
nsax = slice_desired_sax.size
nlge = slice_desired_lge.size

# write the Z values in arrays for every image
ZZsax = jk.getZZarray(nxsax, nysax, slice_desired_sax, zshiftsax, zspacingsax)
ZZlge = jk.getZZarray(nxlge, nylge, slice_desired_lge, zshiftlge, zspacinglge)

# create colormaps
colors_sax = np.zeros((nsax, nxsax, nysax))
colors_lge = np.zeros((nlge, nxlge, nylge))

for i in range(nsax):
    colors_sax[i] = arraysax[t_desired_sax[i], slice_desired_sax[i], :, :]
for i in range(nlge):
    colors_lge[i] = arraylge[t_desired_lge[i], slice_desired_lge[i], :, :]

colors_sax = jk.normalizecolors(colors_sax, 0, 1)
colors_lge = jk.normalizecolors(colors_lge, 0, 1)

# plotting
ax = jk.make_ax(True)
for i in range(nsax):
    ax.plot_surface(XXsax, YYsax, ZZsax[i], rstride=stride, cstride=stride, cmap=cm.gray, facecolors=cm.gray(colors_sax[i]))
for i in range(nlge):
    ax.plot_surface(XXlge, YYlge, ZZlge[i], rstride=stride, cstride=stride, cmap=cm.gray, facecolors=cm.gray(colors_lge[i]))
# plt.show()


# drawing target
target_x, target_y, target_z = [100,100,51] #nx/2, ny/2, nz
ax.scatter3D(target_x, target_y, target_z, linewidth=5)
plt.show()