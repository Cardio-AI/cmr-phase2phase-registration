import SimpleITK as sitk
import numpy as np
from nibabel.pydicom_compat import pydicom

img_path = '/mnt/ssd/julian/data/raw/DMD/AC_20180301_DMD201803010859111/LGE_AC_20180301_DMD201803010859111.2.276.0.7230010.3.1.2.15116645.3956.1578350395.421534/series1201-unknown/img0001-104.388.dcm'

# reader = sitk.ImageSeriesReader()
# img_names = reader.GetGDCMSeriesFileNames(img_path)
# reader.SetFileNames(img_names)
# image = reader.Execute()
# image_array = sitk.GetArrayFromImage(image)  # z, y, x


ds = pydicom.filereader.dcmread(img_path)


x=0