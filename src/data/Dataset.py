import glob
import logging
import os
import random
from logging import debug as debug
# import yaml
from time import time

import SimpleITK as sitk
import numpy as np
import pandas as pd
import skimage
import skimage.exposure
import yaml
from sklearn.model_selection import KFold

from src.utils.Utils_io import ensure_dir
from src.visualization.Visualize import plot_value_histogram


#from sklearn.model_selection import KFold

def copy_meta(new_image, reference_sitk_img):

    """
    copy sitk metadata tags from one image to another
    :param new_image:
    :param reference_sitk_img:
    :return:
    """
    t1 = time()
    if isinstance(new_image, np.ndarray):
        new_image = sitk.GetImageFromArray(new_image)

    if reference_sitk_img is not None:
        assert (isinstance(reference_sitk_img, sitk.Image)), 'no reference image given'
        assert (isinstance(new_image, sitk.Image)), 'only np.ndarrays and sitk images could be stored'

        # copy metadata
        for key in reference_sitk_img.GetMetaDataKeys():
            new_image.SetMetaData(key, get_metadata_maybe(reference_sitk_img, key))
        #logging.debug('Metadata_copied: {:0.3f}s'.format(time() - t1))

        # copy all structural information to image with same dimension and size
        if (reference_sitk_img.GetDimension() == new_image.GetDimension()) and (reference_sitk_img.GetSize() == new_image.GetSize()):
            new_image.CopyInformation(reference_sitk_img)

        # same dimension (e.g. 4) but different size per dimension
        elif (reference_sitk_img.GetDimension() == new_image.GetDimension()):

            # copy spacing and origin but keep size as it is
            new_image.SetOrigin(reference_sitk_img.GetOrigin())
            new_image.SetSpacing(reference_sitk_img.GetSpacing())
            new_image.SetDirection(reference_sitk_img.GetDirection())

        # copy structural informations to smaller images e.g. 4D to 3D
        elif reference_sitk_img.GetDimension() > new_image.GetDimension():
            shape_ = len(new_image.GetSize())
            reference_shape = len(reference_sitk_img.GetSize())
            new_image.SetOrigin(reference_sitk_img.GetOrigin()[:shape_])
            new_image.SetSpacing(reference_sitk_img.GetSpacing()[:shape_])
            # copy direction to smaller images
            # 1. extract the direction, 2. create a matrix, 3. slice by the new shape, 4. flatten
            if new_image.GetDimension() > 2:  # only for volumes
                direction = np.array(reference_sitk_img.GetDirection())
                dir_ = direction.reshape(reference_shape, reference_shape)
                direction = dir_[:shape_, :shape_].flatten()
                new_image.SetDirection(direction)

        # copy structural informations to bigger images e.g. 3D to 4D, fill with 1.0
        else:
            ones = [1.0] * (new_image.GetDimension() - reference_sitk_img.GetDimension())
            new_image.SetOrigin((*reference_sitk_img.GetOrigin(), *ones))
            new_image.SetSpacing((*reference_sitk_img.GetSpacing(), *ones))

        new_image.SetDirection(reference_sitk_img.GetDirection())
        #logging.debug('spatial data_copied: {:0.3f}s'.format(time() - t1))

        return new_image




def copy_meta_and_save(new_image, reference_sitk_img, full_filename=None, overwrite_spacing=None, copy_direction=True):
    """
    Copy metadata, UID and structural information from one image to another
    Works also for different dimensions, returns new_image with copied structural info
    :param new_image: sitk.Image
    :param reference_sitk_img: sitk.Image
    :param path: full file path as str
    :return:
    """

    t1 = time()
    try:
        # make sure this method works with nda and sitk images
        if isinstance(new_image, np.ndarray):
            if len(new_image.shape) == 4:
                # 4D needs to be built from a series
                new_image = [sitk.GetImageFromArray(img) for img in new_image]
                new_image = sitk.JoinSeries(new_image)
            else:
                new_image = sitk.GetImageFromArray(new_image)
        if full_filename:
            ensure_dir(os.path.dirname(os.path.abspath(full_filename)))

        if reference_sitk_img is not None:
            assert (isinstance(reference_sitk_img, sitk.Image)), 'no reference image given'
            assert (isinstance(new_image, sitk.Image)), 'only np.ndarrays and sitk images could be stored'

            # copy metadata
            for key in reference_sitk_img.GetMetaDataKeys():
                new_image.SetMetaData(key, get_metadata_maybe(reference_sitk_img, key))
            #logging.debug('Metadata_copied: {:0.3f}s'.format(time() - t1))

            # copy structural informations to image with same dimension and size
            if (reference_sitk_img.GetDimension() == new_image.GetDimension()) and (reference_sitk_img.GetSize() == new_image.GetSize()):
                new_image.CopyInformation(reference_sitk_img)

            # same dimension (e.g. 4) but different size per dimension
            elif (reference_sitk_img.GetDimension() == new_image.GetDimension()):

                # copy spacing, origin and rotation but keep size as it is
                if copy_direction:
                    new_image.SetDirection(reference_sitk_img.GetDirection())
                new_image.SetOrigin(reference_sitk_img.GetOrigin())
                new_image.SetSpacing(reference_sitk_img.GetSpacing())

            # copy structural information to smaller images e.g. 4D to 3D
            elif reference_sitk_img.GetDimension() > new_image.GetDimension():
                shape_ = len(new_image.GetSize())
                reference_shape = len(reference_sitk_img.GetSize())

                # copy direction to smaller images
                # 1. extract the direction, 2. create a matrix, 3. slice by the new shape, 4. flatten
                if copy_direction:
                    direction = np.array(reference_sitk_img.GetDirection())
                    dir_ = direction.reshape(reference_shape, reference_shape)
                    direction = dir_[:shape_, :shape_].flatten()
                    new_image.SetDirection(direction)

                new_image.SetOrigin(reference_sitk_img.GetOrigin()[:shape_])
                new_image.SetSpacing(reference_sitk_img.GetSpacing()[:shape_])

            # copy structural information to bigger images e.g. 3D to 4D, fill with 1.0 spacing
            else:
                ones = [1.0] * (new_image.GetDimension() - reference_sitk_img.GetDimension())
                new_image.SetOrigin((*reference_sitk_img.GetOrigin(), *ones))
                new_image.SetSpacing((*reference_sitk_img.GetSpacing(), *ones))
                # we cant copy the direction from smaller images to bigger ones

            #logging.debug('spatial data_copied: {:0.3f}s'.format(time() - t1))

            if overwrite_spacing:
                new_image.SetSpacing(overwrite_spacing)

        if full_filename:
            # copy uid
            writer = sitk.ImageFileWriter()
            # writer.KeepOriginalImageUIDOn()
            writer.SetFileName(full_filename)
            writer.Execute(new_image)
            logging.debug('image saved: {:0.3f}s'.format(time() - t1))
            return True
        else:
            return new_image
    except Exception as e:
        logging.error('Error with saving file: {} - {}'.format(full_filename, str(e)))
        return False


def create_4d_volumes_from_4d_files(img_f, mask_f, full_path='data/raw/GCN/3D/', slice_threshold=2):
    """
    Expects an 4d-image and -mask file name and a target path
    filter mask and image volumes by contoured time-steps
    copy all metadata
    save them to the destination path
    :param img_f: 4D image filepath as str
    :param mask_f: 4D mask filepath as str
    :param full_path: export path as str
    :param slice_threshold: minimal masks per timestep as int
    :return:
    """

    logging.info('process file: {}'.format(img_f))
    # get sitk images
    mask_4d_sitk = sitk.ReadImage(mask_f)
    img_4d_sitk = sitk.ReadImage(img_f)

    # filter 4d image nda according to given mask nda
    mask_4d_nda, masked_t = filter_4d_vol(mask_4d_sitk, slice_threshold=slice_threshold)
    img_4d_nda = sitk.GetArrayFromImage(img_4d_sitk)[masked_t]

    # write filtered 4d image to disk
    patient_name = os.path.basename(img_f).split('.')[0].replace('volume_clean', '')
    img_file = '{}_{}{}'.format(patient_name, 'img', '.nrrd')
    mask_file = '{}_{}{}'.format(patient_name, 'msk', '.nrrd')

    copy_meta_and_save(img_4d_nda, img_4d_sitk, os.path.join(full_path, img_file))
    copy_meta_and_save(mask_4d_nda, img_4d_sitk, os.path.join(full_path, mask_file))

    return [masked_t, list(img_4d_nda.shape)]


def match_hist_(nda, avg):
    # this method takes 3 times as long as the looping version below, check if time is given
    t0 = time()
    shape_ = nda.shape
    c = shape_[0] * shape_[1]
    s_len = avg.ndim
    logging.info('first: {:0.3f} s'.format(time() - t0))
    t0 = time()
    if s_len == 3:
        c_avg = avg.shape[0]
        avg = np.reshape(avg, (avg.shape[-2], avg.shape[-1] * c_avg))

    elif s_len == 4:
        c_avg = avg.shape[0] * avg.shape[1]
        avg = np.reshape(avg, (avg.shape[-2], avg.shape[-1] * c_avg))
    logging.info('second: {:0.3f} s'.format(time() - t0))
    t0 = time()
    temp = np.reshape(nda, (shape_[-2], shape_[-1] * c))
    logging.info('third: {:0.3f} s'.format(time() - t0))
    t0 = time()
    temp = skimage.exposure.match_histograms(temp, avg, multichannel=False)
    logging.info('fourth: {:0.3f} s'.format(time() - t0))
    return np.reshape(temp, shape_)


def match_hist_any_dim(nda, ref):
    """
    math the histgram from one ndarray to any other
    works with any shape
    Parameters
    ----------
    nda :
    ref :

    Returns
    -------

    """
    shape_ = nda.shape
    shape_ref = ref.shape
    nda = skimage.exposure.match_histograms(nda.flatten(), ref.flatten(), channel_axis=None)
    return nda.reshape(shape_)


def match_hist(nda,ref, prob_per_z=40):

    for z in range(nda.shape[1]):
        # apply hist matching only on some 2d slices +t, this is more realistic, as some series have different scanner settings
        if random.randint(0,100) <= prob_per_z:
            for t in range(nda.shape[0]):
                nda[t,z] = skimage.exposure.match_histograms(nda[t,z], ref, channel_axis=None)
    return nda

def split_one_4d_sitk_in_list_of_3d_sitk(img_4d_sitk, axis=None, prob=0.8):
    """
    Splits a 4D dicom image into a list of 3D sitk images, copy alldicom metadata
    Parameters
    ----------
    img_4d_sitk :
    mask_4d_sitk :
    slice_treshhold :

    Returns list of 3D-sitk objects
    -------
    """

    img_4d_nda = sitk.GetArrayFromImage(img_4d_sitk)

    if axis: # if we want to split by any other axis than 0, split by this axis and rearange the spacing in the reference sitk
        img_4d_nda = np.split(img_4d_nda,indices_or_sections=img_4d_nda.shape[axis], axis=axis)
        img_4d_nda = [np.squeeze(n) for n in img_4d_nda]
        if axis==1:
            # copy_meta takes the values from spacing, need to swap t and z if we split along the z axis
            old_spacing = img_4d_sitk.GetSpacing()
            img_4d_sitk.SetSpacing((old_spacing[0], old_spacing[1], old_spacing[3], old_spacing[2]))

    # create t 3d volumes
    list_of_3d_sitk = [copy_meta_and_save(new_image=img_3d, reference_sitk_img=img_4d_sitk, full_filename = None, overwrite_spacing = None, copy_direction = True) for img_3d in img_4d_nda]

    return list_of_3d_sitk


def create_3d_volumes_from_4d_files(img_f, mask_f, full_path='data/raw/tetra/3D/', slice_treshhold=2):
    """
    Expects an 4d-image and -mask file name and a target path
    filter mask and image volumes with segmentation
    copy all metadata
    save them to the destination path
    :param img_f:
    :param mask_f:
    :param full_path:
    :return:
    """

    logging.info('process file: {}'.format(img_f))
    # get sitk images
    mask_4d_sitk = sitk.ReadImage(mask_f)
    img_4d_sitk = sitk.ReadImage(img_f)

    # filter 4d image nda according to given mask nda
    mask_4d_nda, masked_t = filter_4d_vol(mask_4d_sitk, slice_threshold=slice_treshhold)
    img_4d_nda = sitk.GetArrayFromImage(img_4d_sitk)[masked_t]

    # create t 3d volumes
    for img_3d, mask_3d, t in zip(img_4d_nda, mask_4d_nda, masked_t):
        # write 3d image
        patient_name = os.path.basename(img_f).split('.')[0].replace('volume_clean', '')
        img_file = '{}_t{}_{}{}'.format(patient_name, str(t), 'img', '.nrrd')
        mask_file = '{}_t{}_{}{}'.format(patient_name, str(t), 'msk', '.nrrd')

        copy_meta_and_save(img_3d, img_4d_sitk, os.path.join(full_path, img_file))
        copy_meta_and_save(mask_3d, img_4d_sitk, os.path.join(full_path, mask_file))

    return [masked_t, list(img_4d_nda.shape)]


def create_2d_slices_from_4d_volume_files(img_f, mask_f, export_path, filter_by_mask=True, slice_threshold=2):
    """
        Expects an 4d-image and -mask file name and a target path
    filter mask and image volumes with segmentation
    copy all metadata
    save them to the destination path
    :param img_f:
    :param mask_f:
    :param export_path: str
    :param filter_by_mask: bool
    :param slice_threshold: int
    :return:
    """

    logging.info('process file: {}'.format(img_f))

    # get sitk images
    mask_4d_sitk = sitk.ReadImage(mask_f)
    img_4d_sitk = sitk.ReadImage(img_f)

    # filter 4d image nda according to given mask nda
    if filter_by_mask:
        mask_4d_nda, masked_t = filter_4d_vol(mask_4d_sitk,slice_threshold=slice_threshold)
        img_4d_nda = sitk.GetArrayFromImage(img_4d_sitk)[masked_t]
    else:
        img_4d_nda = sitk.GetArrayFromImage(img_4d_sitk)
        masked_t = list(range(img_4d_nda.shape[0]))
        mask_4d_nda = sitk.GetArrayFromImage(mask_4d_sitk)

    # create t x 3d volumes
    for img_3d, mask_3d, t in zip(img_4d_nda, mask_4d_nda, masked_t):

        # get patient_name
        patient_name = os.path.basename(img_f).split('.')[0].replace('volume_clean', '')

        # create z x 2d slices
        for z, slice_2d in enumerate(zip(img_3d, mask_3d)):
            # create filenames with reference to t and z position
            img_file = '{}_t{}_z{}_{}{}'.format(patient_name, str(t), str(z), 'img', '.nrrd')
            mask_file = '{}_t{}_z{}_{}{}'.format(patient_name, str(t), str(z), 'msk', '.nrrd')

            # save nrrd file with metadata
            copy_meta_and_save(slice_2d[0], img_4d_sitk, os.path.join(export_path, img_file),copy_direction=False)
            copy_meta_and_save(slice_2d[1], img_4d_sitk, os.path.join(export_path, mask_file),copy_direction=False)

    return [masked_t, list(img_4d_nda.shape)]


def create_2d_slices_from_3d_volume_files_any_filename(img_f, mask_f, export_path):
    """
    Helper to split a GCN 3D dicom file into z x 2D slices

    Expects an 3d-image and -mask file name and a target path
    filter mask and image volumes with segmentation
    copy all metadata
    save them to the destination path
    :param img_f:
    :param mask_f:
    :param full_path:
    :return:
    """
    import re
    logging.info('process file: {}'.format(img_f))

    # get sitk images
    mask_3d_sitk = sitk.ReadImage(mask_f)
    img_3d_sitk = sitk.ReadImage(img_f)

    # filter 4d image nda according to given mask nda
    mask_3d = sitk.GetArrayFromImage(mask_3d_sitk)
    img_3d = sitk.GetArrayFromImage(img_3d_sitk)

    # get file names
    _, img_f = os.path.split(img_f)
    _, mask_f = os.path.split(mask_f)


    def get_new_name(f_name, z):
        match = ''
        # check if image or mask
        m = re.search('_img|_msk', f_name)
        if m:
            match = m.group(0)
        # extend filename at the very last position before 'img' or 'msk'
        return re.sub('{}.nrrd'.format(match), '_{}{}.nrrd'.format(z, match), f_name)

    # create z x 2d slices
    for z, slice_2d in enumerate(zip(img_3d, mask_3d)):
        # create filenames with reference to t and z position
        # extend the filename

        img_file = get_new_name(img_f, z)
        mask_file = get_new_name(mask_f, z)

        # save nrrd file with metadata
        copy_meta_and_save(slice_2d[0], img_3d_sitk, os.path.join(export_path, img_file))
        copy_meta_and_save(slice_2d[1], img_3d_sitk, os.path.join(export_path, mask_file))

    return list(img_3d.shape)

def create_2d_slices_from_3d_volume_files(img_f, mask_f, export_path):
    """
    Helper for ACDC data
    
    Expects an 3d-image and -mask file name and a target path
    filter mask and image volumes with segmentation
    copy all metadata
    save them to the destination path
    :param img_f:
    :param mask_f:
    :param full_path:
    :return:
    """

    logging.info('process file: {}'.format(img_f))

    # get sitk images
    if not mask_f:
        masks_given = False
        mask_f = img_f

    mask_3d_sitk = sitk.ReadImage(mask_f)
    img_3d_sitk = sitk.ReadImage(img_f)

    mask_3d = sitk.GetArrayFromImage(mask_3d_sitk)
    img_3d = sitk.GetArrayFromImage(img_3d_sitk)

    # get patient_name
    patient_name = os.path.basename(img_f).split('_')[0]
    frame = os.path.basename(img_f).split('frame')[1][:2]
    # create z x 2d slices
    for z, slice_2d in enumerate(zip(img_3d, mask_3d)):
        # create filenames with reference to t and z position
        img_file = '{}__t{}_z{}_{}{}'.format(patient_name, str(frame), str(z), 'img', '.nrrd')
        mask_file = '{}__t{}_z{}_{}{}'.format(patient_name, str(frame), str(z), 'msk', '.nrrd')

        # save nrrd file with metadata
        copy_meta_and_save(slice_2d[0], img_3d_sitk, os.path.join(export_path, img_file))
        copy_meta_and_save(slice_2d[1], img_3d_sitk, os.path.join(export_path, mask_file))

    return [frame, list(img_3d.shape)]


def get_patient(filename_to_2d_nrrd_file):
    """
    Split the nrrd filename and returns the patient id
    split the filename by '_' returns the first two elements of that list
    If the filename contains '__' it returns the part before
    """
    import re
    m = re.search('__', filename_to_2d_nrrd_file)
    if m: # nrrd filename with '__'
        return os.path.basename(filename_to_2d_nrrd_file).split('__')[0]
    if os.path.basename(filename_to_2d_nrrd_file).startswith('patient'): # acdc file
        return os.path.basename(filename_to_2d_nrrd_file).split('_')[0]
    else: # gcn filename
        return '_'.join(os.path.basename(filename_to_2d_nrrd_file).split('_')[:2])


def get_trainings_files(data_path, fold=0, path_to_folds_df=None):
    """
    Load CMR images and masks for a given data path according to different suffixes
    If we path_to_folds (dataframe) is provided, use this dataframe and the parameter fold
    to split our x and y into train, val.
    If no path_to_folds is given use all files for train/validation (usually for inference with another dataset)
    :param data_path:
    :param fold:
    :param path_to_folds_df:
    :return: x_train, y_train, x_val, y_val
    """

    single_file = sorted(glob.glob(os.path.join(data_path,'*')))[0]
    if 'nii.gz' in os.path.basename(single_file):
        ftype= '.nii.gz'
    elif '.nrrd' in os.path.basename(single_file):
        ftype = '.nrrd'
    else:
        print('no nii and nrrd files found: {}'.format(single_file))

    img_suffix = '*img{}'.format(ftype)
    mask_suffix = '*msk{}'.format(ftype)

    # load the nrrd files with given pattern from the data path
    x = sorted(glob.glob(os.path.join(data_path, img_suffix)))
    y = sorted(glob.glob(os.path.join(data_path, mask_suffix)))
    if len(x) == 0:
        logging.info('no files found, try to load with clean.nrrd/mask.nrrd pattern')
        logging.info('searched in: {}'.format(data_path))
        img_suffix = '*clean{}'.format(ftype)
        mask_suffix = '*mask{}'.format(ftype)
        x = sorted(glob.glob(os.path.join(data_path, img_suffix)))
        y = sorted(glob.glob(os.path.join(data_path, mask_suffix)))

    if len(x) == 0:
        logging.info('no files found, try to load with acdc file pattern')
        x, y = load_acdc_files(data_path)
        if len(x) == 0:
            logging.error('no files found in: {}, try to list all files in this directory:'.format(data_path))
            files_ = os.listdir(data_path)
            logging.error(files_)
            x = sorted(glob.glob(os.path.join(data_path, '*')))
            y = x
    if path_to_folds_df and os.path.isfile(path_to_folds_df):
        df = pd.read_csv(path_to_folds_df)
        patients = df[df.fold.isin([fold])]
        # make sure we count each patient only once
        patients_train = patients[patients['modality'] == 'train']['patient'].str.lower().unique()
        patients_test = patients[patients['modality'] == 'test']['patient'].str.lower().unique()
        logging.info('Found {} images/masks in {}'.format(len(x), data_path))
        logging.info('Patients train: {}'.format(len(patients_train)))

        def filter_files_for_fold(list_of_filenames, list_of_patients):
            """Helper to filter one list by a list of substrings"""
            from src.data.Dataset import get_patient
            return [str for str in list_of_filenames
                    if get_patient(str).lower() in list_of_patients]

        x_train = sorted(filter_files_for_fold(x, patients_train))
        y_train = sorted(filter_files_for_fold(y, patients_train))
        x_test = sorted(filter_files_for_fold(x, patients_test))
        y_test = sorted(filter_files_for_fold(y, patients_test))
        logging.info('Selected {} of {} files with {} of {} patients for training fold {}'.format(len(x_train), len(x),
                                                                                                  len(patients_train),
                                                                                                  len(df.patient.unique()),
                                                                                                  fold))

    else: # no dataframe given for splitting
        logging.info('no dataframe for splitting provided. Will use all files for train and validation')
        x_train = sorted(x)
        y_train = sorted(y)
        x_test = sorted(x)
        y_test = sorted(y)
        logging.info('Selected {} of {} files'.format(len(x_train), len(x)))

    assert (len(x_train) == len(y_train)), 'len(x_train != len(y_train))'


    return x_train, y_train, x_test, y_test

def get_kfolded_data(kfolds=4, path_to_data='data/raw/tetra/2D/', extract_patient_id=get_patient):
    """
    filter all image files by patient names defined in fold n
    functions expects subfolders, collects all image, mask files
    and creates a list of unique patient ids

    create k folds of this patient ids
    filter the filenames containing the patient ids from each kfold split
    returns
    :param kfolds: number of splits
    :param path_to_data: path to root of split data e.g. 'data/raw/tetra/2D/'
    :param extract_patient_id: function which returns the patient id for each filename in path_to_data
    :return: a dataframe with the following columns:
        fold (kfolds-1),
        x_path (full filename to image as nrrd),
        y_path (full filename to mask as nrrd),
        modality(train or test)
        patient (patient id)
    """

    img_pattern = '*img.nrrd'
    columns = ['fold', 'x_path', 'y_path', 'modality', 'patient']
    modality_train = 'train'
    modality_test = 'test'
    seed = 42

    # get all images, masks from given directory
    # get all img files in all subdirs
    x = sorted(glob.glob(os.path.join(path_to_data, '**/*{}'.format(img_pattern))))

    # if no subdirs given, search in root
    if len(x) == 0:
        x = sorted(glob.glob(os.path.join(path_to_data, '*{}'.format(img_pattern))))
    logging.info('found: {} files'.format(len(x)))
    # create a unique list of patient ids
    patients = sorted(list(set([extract_patient_id(f) for f in x])))
    logging.info('found: {} patients'.format(len(patients)))
    # create a k-fold instance with k = kfolds
    kfold = KFold(n_splits=kfolds, shuffle=True,random_state=seed)

    def filter_x_by_patient_ids_(x, patient_ids, modality, columns, f):
        # create a dataframe from x (list of filenames) filter by patient ids
        # returns a dataframe
        df = pd.DataFrame(columns=columns)
        df['x_path'] = [elem for elem in x if extract_patient_id(elem) in patient_ids]
        df['y_path'] = [elem.replace('img', 'msk') for elem in df['x_path']]
        df['fold'] = [f] * len(df['x_path'])
        df['modality'] = [modality] * len(df['x_path'])
        df['patient'] = [extract_patient_id(elem) for elem in df['x_path']]
        logging.debug(len(df['x_path']))
        logging.debug(patient_ids)
        logging.debug(len(x))
        logging.debug(extract_patient_id(x[0]))
        return df

    # split patients k times
    # use the indexes to get the patient ids from x
    # use the patient ids to filter train/test from the complete list of files
    df_folds = pd.DataFrame(columns=columns)
    for f, idx in enumerate(
            kfold.split(patients)):  # f = fold, idx = tuple with all indexes to split the patients in this fold
        train_idx, test_idx = idx
        # create a list of train and test indexes
        logging.debug("TRAIN: {}, TEST: {}".format(train_idx, test_idx))
        # slice the filenames by the given indexes 
        patients_train, patients_test = [patients[i] for i in train_idx], [patients[i] for i in test_idx]

        df_train = filter_x_by_patient_ids_(x, patients_train, modality_train, columns, f)
        df_test = filter_x_by_patient_ids_(x, patients_test, modality_test, columns, f)

        # merge train and test
        df_fold = pd.concat([df_train, df_test])
        # merge fold into folds dataset
        df_folds = pd.concat([df_fold, df_folds])

    return df_folds


def filter_x_by_patient_ids(x, patient_ids, modality='test', columns=['x_path', 'y_path', 'fold', 'modality', 'patient', 'pathology'], fold=0, pathology=None, filter=True):
    """
    Create a df from a given list of files
    and a list of patient which are used to filter the file names

    :param x:
    :param patient_ids:
    :param modality:
    :param columns:
    :param f:
    :param pathology:
    :return:
    """
    # create a dataframe from x (list of filenames) filter by patient ids
    # returns a dataframe
    df = pd.DataFrame(columns=columns)
    if filter:
        df['x_path'] = [elem for elem in x if get_patient(elem) in patient_ids]
    else:
        df['x_path'] = [elem for elem in x]
    df['y_path'] = [elem.replace('img', 'msk') for elem in df['x_path']]
    df['fold'] = [fold] * len(df['x_path'])
    df['modality'] = [modality] * len(df['x_path'])
    df['patient'] = [get_patient(elem) for elem in df['x_path']]
    df['pathology'] = [pathology] * len(df['x_path'])

    return df


def get_n_patients(df, n=1):
    """
    Select n random patients
    Filter the data frame by this patients
    Use the Fold 0 split as default
    Override the modality for all random selected patients to "train"
    return filtered df
    :param df:
    :param n:
    :param fold:
    :return:
    """

    # fold is not important,
    # because we return patients from train and test modality
    fold = 0

    # make random.choice idempotent
    np.random.seed(42)
    # select random patients
    patients = np.random.choice(sorted(df['patient'].unique()), size=n)
    logging.info('Added patients: {} from the GCN dataset'.format(patients))
    # filter data frame by fold and by random selected patients ids, make sure to make a copy to avoid side effects
    df_temp = df[(df['fold'] == fold) & (df['patient'].isin(patients))].copy()
    # make sure all selected images will be used during training, change modality to train for this images
    # train_kfolded will only use images with modality == train, override the modality for all selected patients/rows
    df_temp.loc[:, 'modality'] = 'train'
    df_temp.reset_index(inplace=True)
    return df_temp


def get_train_data_from_df(first_df='reports/kfolds_data/2D/acdc/df_kfold.csv', second_df=None,
                           n_second_df=0, n_first_df=None, fold=0,):
    """
    load one df and select n patients, default: use all
    load a second df, if given
    select n patients from second df,
    merge first df into second df
    return x_train, y_train, x_val, y_val as list of files
    :param df_fname: full file/pathname to first df (str)
    :param second_df_fname: full file/pathname to second df (str)
    :param n_second_df: number of patients from second df, that should be merged
    :param n_patients_first_df:  int - number of patients to load from the first dataframe
    :param fold: select a fold from df
    :return:
    """
    extend = dict()
    extend['GCN_PATIENTS'] = list()
    extend['GCN_IMAGES'] = 0
    df = pd.read_csv(first_df)

    # take only n patients from the first dataframe
    if n_first_df:
        df = get_n_patients(df, n_first_df)

    # if second dataframe given, load df, select m patients, and concat this dataframe with the first one
    if second_df:  # extend dataframe with n patients from second dataframe
        df_second = pd.read_csv(second_df)
        df_second = get_n_patients(df_second, n_second_df)
        df = pd.concat([df, df_second], sort=False)
        extend['GCN_PATIENTS'] = sorted(df_second['patient'].unique())
        extend['GCN_IMAGES'] = len(df_second)

    logging.info('loaded df from {} with shape: {}'.format(first_df, df.shape))
    logging.info('available folds: {}, selected fold: {}'.format(df.fold.unique(), fold))
    if 'fold' in df:
        df = df[df['fold'] == fold]

    if 'pathology' in df:
        logging.info(
            'available modalities: {}\n available pathologies: {}'.format(df.modality.unique(), df.pathology.unique()))
    # get a trainings and a test dataframe
    df_train = df[df['modality'] == 'train']
    df_test = df[df['modality'] == 'test']

    return sorted(df_train['x_path'].values), sorted(df_train['y_path'].values), sorted(df_test['x_path'].values), sorted(df_test[
        'y_path'].values), extend


def create_acdc_dataframe_for_cv(path_to_data='data/raw/ACDC/2D/', kfolds=4,
                                 original_acdc_dir='data/raw/ACDC/original/all/', img_pattern='*img.nrrd'):
    """
    Creates a dataframe of all 2D ACDC cmr filenames,
    splits the data in n folds with respect to the pathologies
    0. Load all 2D nrrd files in the given directory
    1. Create a dataframe with all original ACDC files, patient ids, pathology
    2. Create a list of all pathologies
    3. Create n (# of pathologies) subgroups of patients, filtered by pathology
    4. Create k splits (# of splits) of each patient subgroup
    5. For each pathology split: collect all 2D files
    6. Merge 2D files according to their split number and modality (train/val)
    return a df with all files for each split
    f_name, patient, split, train_modality, pathology

    :param img_pattern:
    :param path_to_data: path to 2D ACDC files
    :param kfolds:
    :return: dataframe with all splits
    """

    from sklearn.model_selection import KFold

    seed = 42
    #img_pattern = '*img.nrrd'
    columns = ['fold', 'x_path', 'y_path', 'modality', 'patient', 'pathology']
    modality_train = 'train'
    modality_test = 'test'

    # list all nrrd image files within the subdirs of path_to_data
    acdc_x_files = sorted(glob.glob(os.path.join(path_to_data, '**/{}'.format(img_pattern))))
    logging.info('Found: {} files in {}'.format(len(acdc_x_files), path_to_data))

    # get all ACDC files + pathology as df
    # provide a seed to make shuffle idempotent
    df = get_acdc_dataset_as_df(original_acdc_dir)
    logging.info('Created a dataframe with shape: {}'.format(df.shape))
    pathologies = df['pathology'].unique()
    kfold = KFold(kfolds, shuffle=True, random_state=seed)

    # create a df to merge all splits into
    df_folds = pd.DataFrame(columns=columns)

    # for each pathology, create k folds of train/test splits
    for pathology in pathologies:
        # collect all patients with this pathology
        patients = df[df['pathology'] == pathology]['patient'].unique()
        logging.debug('{} Patients found for pathology: {}'.format(len(patients), pathology))

        # create k fold of train/test splits
        # split with the patient ids
        # to make sure that one patient occurs either as train or validation data
        # f = fold, idx = tuple with all indexes to split the patients in this fold
        # kfold.split returns the patient indexes for each fold
        for fold, idx in enumerate(kfold.split(patients)):
            train_idx, test_idx = idx
            #logging.debug("TRAIN:", train_idx, "TEST:", test_idx)

            # create one list for the train and test patient ids from the kfold split indexes
            patients_train, patients_test = [patients[i] for i in train_idx], [patients[i] for i in test_idx]
            logging.debug('Fold: {}, Pathology: {} train: {}'.format(fold, pathology, patients_train))
            logging.debug('Fold: {}, Pathology: {}, test: {}'.format(fold, pathology, patients_test))

            # filter the 2D filenames by the two patient id lists (train/test) for this fold
            # create one df for each split (train/test) with the corresponding 2D nrrd files
            df_train = filter_x_by_patient_ids(acdc_x_files, patients_train, modality_train, columns, fold, pathology)
            df_test = filter_x_by_patient_ids(acdc_x_files, patients_test, modality_test, columns, fold, pathology)
            logging.debug('Files x_train: {}'.format(len(df_train)))
            logging.debug('Files x_test: {}'.format(len(df_test)))

            # merge train and test files of this split
            df_fold = pd.concat([df_train, df_test])
            # merge this fold into folds dataset
            df_folds = pd.concat([df_fold, df_folds], sort=True)

    return df_folds


def describe_acdc_patient_folder(p):
    """
    Create a dataframe of a ACDC patient folder
    :param p:
    :return: df with patient-id, pathology and files
    """
    files = list()

    # get patient name
    patient = os.path.basename(os.path.abspath(p))

    # get all files for this patient
    #files = sorted(glob.glob(os.path.join(p, '*')))

    phases = ['cfg', 'ed', 'ed_gt', 'es', 'es_gt', '4d']

    files.append(sorted(glob.glob(os.path.join(p, '*.cfg')))[0])
    files.append(get_phase_file(p, 'ED', False))
    files.append(get_phase_file(p, 'ED', True))
    files.append(get_phase_file(p, 'ES', False))
    files.append(get_phase_file(p, 'ES', True))
    files.append(sorted(glob.glob(os.path.join(p, '*4d.nii.gz')))[0])

    assert len(phases) == len(files), 'cant find 6 files in {}'.format(p)

    # get pathology from info.cfg file
    pathology = get_pathology_group(p)

    # build dataframe with files, pathology, and patient-id
    df = pd.DataFrame()
    df['pathology'] = [pathology] * len(files)
    df['patient'] = [patient] * len(files)
    df['files'] = files
    df['phase'] = phases
    return df


def read_cfg_file(f):
    """Helper to open cfg files"""
    with open(f, 'r') as yml_file:
        cfg = yaml.load(yml_file)
    return cfg


def get_phase_file(folder, phase='ED', gt=False):
    """Helper to get the patient phase filename"""
    cfg_f = os.path.join(folder, 'Info.cfg')
    cfg = read_cfg_file(cfg_f)
    frame = '{:02}'.format(cfg.get(phase, 'NOPHASE'))
    # get phase file for this patient
    if gt:
        p = os.path.join(folder, '*frame{}_gt.nii.gz'.format(frame))
    else:
        p = os.path.join(folder, '*frame{}.nii.gz'.format(frame))
    return glob.glob(p)[0]



def get_pathology_group(folder):
    """Helper to get the patient pathology from the cfg file"""
    cfg_f = os.path.join(folder, 'Info.cfg')
    cfg = read_cfg_file(cfg_f)
    return cfg.get('Group', 'NOGROUP')

def get_phase_for_patient_timestep(folder, timestep):
    """Helper to get the patient phase for a specific timestep"""
    cfg_f = os.path.join(folder, 'Info.cfg')
    cfg = read_cfg_file(cfg_f)
    ed = cfg.get('ED', 100)
    es = cfg.get('ES', 100)
    phase = 'NOPHASE'
    if timestep == ed:
        phase = 'ED'
    elif timestep == es:
        phase = 'ES'

    return phase




def get_acdc_dataset_as_df(path='data/raw/ACDC/original/all/'):
    """
    Create a df for the ACDC dataset
    columns: patient-id, pathology, files
    :param path:
    :param export:
    :param export_folder_name:
    :return:
    """

    # load all patient folders
    patient_folders = sorted(glob.glob(os.path.join(path, '**/')))

    # create a dataframe with patient-id, pathology, files for this patient
    dfs = list(map(describe_acdc_patient_folder, patient_folders))

    return pd.concat(dfs).reset_index(inplace=False)


def filter_4d_vol(sitk_img_4d, slice_threshold=2):
    """
    filter emtpy slices along axis 0
    expect a 4d numpy array with shape t, z, x, y
    Returns numpy with shape t - emtpy, z, x, y
    :param sitk_img_4d:
    :param slice_threshold: min number of masked slices per volume
    :return:
    """
    if isinstance(sitk_img_4d, sitk.Image):
        nda_4d = sitk.GetArrayFromImage(sitk_img_4d)
    elif isinstance(sitk_img_4d, np.ndarray):
        nda_4d = sitk_img_4d

    logging.info(nda_4d.shape)
    timesteps = []
    # get indexes for masked volumes
    # filter 3d volumes with less masked slices than threshold
    for t, nda_3d in enumerate(nda_4d):
        if nda_3d.max() > 0:  # 3d volume with masks
            masked_slices = 0
            for slice in nda_3d:  # check how many slices are masked
                if slice.max() > 0:
                    masked_slices += 1
            if masked_slices > slice_threshold:
                timesteps.append(t)
            else:
                logging.info('filter volume by masked slices threshold')

    logging.info('filtered timesteps: {}'.format(timesteps))
    filtered_nda = nda_4d[timesteps]

    return filtered_nda, timesteps


def describe_sitk(sitk_img):
    """
    log some basic informations for a sitk image
    :param sitk_img:
    :return:
    """
    if isinstance(sitk_img, np.ndarray):
        sitk_img = sitk.GetImageFromArray(sitk_img.astype(np.float32))

    logging.info('size: {}'.format(sitk_img.GetSize()))
    logging.info('spacing: {}'.format(sitk_img.GetSpacing()))
    logging.info('origin: {}'.format(sitk_img.GetOrigin()))
    logging.info('direction: {}'.format(sitk_img.GetDirection()))
    logging.info('pixel type: {}'.format(sitk_img.GetPixelIDTypeAsString()))
    logging.info('number of pixel components: {}'.format(sitk_img.GetNumberOfComponentsPerPixel()))


def get_metadata_maybe(sitk_img, key, default='not_found'):
    # helper for unicode decode errors
    try:
        value = sitk_img.GetMetaData(key)
    except Exception as e:
        logging.debug('key not found: {}, {}'.format(key, e))
        value = default
    # need to encode/decode all values because of unicode errors in the dataset
    if not isinstance(value, int):
        value = value.encode('utf8', 'backslashreplace').decode('utf-8').replace('\\udcfc', 'ue')
    return value


def get_img_msk_files_from_split_dir(path, img_suffix='*img.nrrd', mask_suffix='*msk.nrrd'):
    """
    Loads all images/masks within a given directory
    returns a tuple of lists: images, masks
    """
    assert (os.path.exists(path)), 'Path: {} does not exist'.format(path)
    
    images = sorted(glob.glob(os.path.join(path, '*img.nrrd')))
    masks = sorted(glob.glob(os.path.join(path, '*msk.nrrd')))
    
    if len(images) == 0:
        logging.info('no nrrd files found, try to load acdc files.')
        images = sorted(glob.glob(os.path.join(path, '**/*frame[0-9][0-9].nii.gz')))
        masks = sorted(glob.glob(os.path.join(path, '**/*frame*_gt.nii.gz')))
    
    return images, masks


def get_z_position_from_filename(f_name):
    return int(f_name.split('_')[-2].replace('z', ''))


def get_t_position_from_filename(f_name):
    try:
        return int(f_name.split('_')[-3].replace('t', ''))
    except Exception as e:
        return f_name.split('_')[-3].replace('t', '')


def is_patient_in_df(row, df, col='patient_unique'):
    """
    Check if this row is in the given 2nd dataframe
    """
    unique_p = row[col]

    if unique_p in df[col].values:
        return True
    else:
        return False


def get_phase(row, df, col='patient_unique'):
    """
    Get the phase for a 3D volume from the export excel sheet
    which is loaded as df_time
    """
    
    # get unique patient id from the current row
    unique_p = row[col]
    
    # check if this patient id is in the given dataframe
    if unique_p in df[col].values:

        df_time_row = df[df[col] == unique_p]

        if int(df_time_row['ED#']) - 1 == int(row['t']):
            return "ED"
        elif int(df_time_row['ES#']) - 1 == int(row['t']):
            return "ES"
        elif int(df_time_row['MS#']) - 1 == int(row['t']):
            return "MS"
        elif int(df_time_row['PF#']) - 1 == int(row['t']):
            return "PF"
        elif int(df_time_row['MD#']) - 1 == int(row['t']):
            return "MD"

        else:
            return "no_phase_fits"

    else:
        return False



def get_patients(path):
    """
    Get a list of all unique patients within a train, val, test directory
    """
    images, masks = get_img_msk_files_from_split_dir(path)
    return sorted(list(set([get_patient(file) for file in images])))


def load_acdc_files(path):
    """
    path: root path for all acdc patient folders
    returns: a tuple (images, masks) with full file names
    """

    assert (os.path.exists(path)), 'Path: {} does not exist'.format(path)

    images = sorted(glob.glob(os.path.join(path, '**/*frame[0-9][0-9].nii.gz')))
    masks = sorted(glob.glob(os.path.join(path, '**/*frame*_gt.nii.gz')))

    return images, masks


def get_3d_img_msk_files(path, img_suffix='images/*img.nrrd', mask_suffix='masks/*msk.nrrd'):
    """
    Returns two file lists of the containing
    nrrd file names with the suffix msk/img
    """
    
    assert (os.path.exists(path)), 'Path: {} does not exist'.format(path)
    
    masks = sorted(glob.glob(os.path.join(path, 'masks/*msk.nrrd')))
    images = sorted(glob.glob(os.path.join(path, 'images/*img.nrrd')))
    
    if len(images) == 0:
        logging.info('no nrrd files found, try to load acdc files.')
        images, masks = load_acdc_files(path)

    return images, masks


def describe_volume(f_name, image=True, plot=False):
    """
    Loads a nnrd file from a given filename
    extracts some metadata if given
    returns a flat json with all properties
    :param f_name:
    :param image:
    :param plot:
    :return:
    """

    # load file, convert and calculate stats
    if isinstance(f_name, sitk.Image):
        sitk_img = f_name
    else:
        sitk_img = sitk.ReadImage(f_name)
    img = sitk.GetArrayViewFromImage(sitk_img)

    def get_metadata_maybe(key, default='not_found'):
        try:
            value = sitk_img.GetMetaData(key)
        except Exception as e:
            logging.debug('key not found: {}, {}'.format(key, e))
            value = default
            pass
        if not isinstance(value, int):
            value = value.encode('utf8', 'backslashreplace').decode('utf-8').replace('\\udcfc', 'ue')
        return value

    ninenine_q = np.quantile(img.flatten(), .99, overwrite_input=True)
    seven_q = np.quantile(img.flatten(), .75, overwrite_input=True)
    mean_q = np.quantile(img.flatten(), .5, overwrite_input=True)

    if plot:
        # get filename, make sure to work with acdc format *.nii.gz
        short = os.path.splitext(os.path.basename(f_name))[0]
        plot_value_histogram(img, os.path.splitext(os.path.basename(short))[0], image)

    def build_patient():
        '''
        create a flat dicom image representation
        works with 2d and 3d nrrd files
        :return:
        '''
        json_representation = {}
        json_representation['f_name'] = f_name
        json_representation['image'] = image
        json_representation['shape'] = img.shape
        spacing = sitk_img.GetSpacing()
        json_representation['spacing'] = spacing
        if len(img.shape) == 4:  # process 4d nrrd file, t, z, x, y
            json_representation['x-axis'] = img.shape[3]
            json_representation['y-axis'] = img.shape[2]
            json_representation['z-axis'] = img.shape[1]
            json_representation['t-axis'] = img.shape[0]
            json_representation['slices'] = img.shape[1] * img.shape[0]
            json_representation['x-spacing'] = spacing[0]
            json_representation['y-spacing'] = spacing[1]
            json_representation['z-spacing'] = spacing[2]
            json_representation['t-spacing'] = spacing[3]
        elif len(img.shape) == 3:  # process 3d nrrd file, z, x, y
            json_representation['x-axis'] = img.shape[2]
            json_representation['y-axis'] = img.shape[1]
            json_representation['z-axis'] = img.shape[0]
            json_representation['t-axis'] = 0
            json_representation['slices'] = img.shape[0]  # z axis
            json_representation['x-spacing'] = spacing[0]
            json_representation['y-spacing'] = spacing[1]
            json_representation['z-spacing'] = spacing[2]
            json_representation['t-spacing'] = 0
        else:  # handle 2d nrrd files, keep same json structure
            json_representation['x-axis'] = img.shape[1]
            json_representation['y-axis'] = img.shape[0]
            json_representation['z-axis'] = 0
            json_representation['t-axis'] = 0
            json_representation['slices'] = 1  # z axis
            json_representation['x-spacing'] = spacing[0]
            json_representation['y-spacing'] = spacing[1]
            json_representation['z-spacing'] = 0
            json_representation['t-spacing'] = 0

        json_representation['min'] = img.min()
        json_representation['max'] = img.max()
        json_representation['mean'] = img.mean()
        json_representation['.99-quantile'] = ninenine_q
        json_representation['.75-quantile'] = seven_q
        json_representation['.50-quantle'] = mean_q
        json_representation['sizes'] = str(sitk_img.GetSize())
        json_representation['dimension'] = int(sitk_img.GetDimension())
        json_representation['row'] = get_metadata_maybe('0028|0010')
        json_representation['column'] = get_metadata_maybe('0028|0011')
        json_representation['seriesinstanceuid'] = get_metadata_maybe('0020|000e')
        json_representation['SeriesDescription'] = get_metadata_maybe('0008|103e')
        json_representation['CardiacNumberOfImages'] = get_metadata_maybe('0018|1090')

        # casting crashes sometimes for these properties
        try:
            json_representation['MagneticFieldStrength'] = float(get_metadata_maybe('0018|0087'))
        except:
            json_representation['MagneticFieldStrength'] = get_metadata_maybe('0018|0087')
        try:
            json_representation['SliceThickness'] = float(get_metadata_maybe('0018|0050'))
        except:
            json_representation['SliceThickness'] = get_metadata_maybe('0018|0050')

        json_representation['PatientPosition'] = get_metadata_maybe('0018|5100')
        json_representation['SliceLocation'] = get_metadata_maybe('0020|1041')
        json_representation['SmallestImagePixelValue'] = int(get_metadata_maybe('0028|0106', 100))
        json_representation['LargestImagePixelValue'] = int(get_metadata_maybe('0028|0107', 0))
        json_representation['Manufacturer'] = get_metadata_maybe('0008|0070')
        json_representation['ManufacturerModelName'] = get_metadata_maybe('0008|1090')
        json_representation['InstitutionName'] = get_metadata_maybe('0008|0080')
        json_representation['InstitutionAddress'] = get_metadata_maybe('0008|0081')
        json_representation['ReferringPhysicianName'] = get_metadata_maybe('0008|0090')

        json_representation['PatientID'] = get_metadata_maybe('0010|0020')
        json_representation['PatientBirthDate'] = get_metadata_maybe('0010|0030')
        json_representation['PatientSex'] = get_metadata_maybe('0010|0040')
        json_representation['PatientAge'] = get_metadata_maybe('0010|1010')
        json_representation['PatientSize'] = get_metadata_maybe('0010|1020')
        json_representation['PatientWeight'] = get_metadata_maybe('0010|1030')
        json_representation['studyinstanceuid'] = get_metadata_maybe('0020|000d')
        json_representation['StudyDescription'] = get_metadata_maybe('0008|1030')

        return json_representation

    return build_patient()


def describe_path(path='data/processed/train/', dataset=['ACDC', 'GCN'], wildcard=None, plot_histogram=True):
    """
    reads all nrrd images/ masks within a train, val, test folder 
    returns a dataframe with all image data
    plots a histogram for every 10th volume
    """

    files = pd.DataFrame()

    if wildcard:
        logging.info('Using wildcard description: {}'.format(wildcard))
        files['images'] = sorted(glob.glob(os.path.join(path, wildcard)))

    elif dataset == 'ACDC':
        logging.info('Using acdc dataset')
        files['images'] = sorted(glob.glob(os.path.join(path, '**/*frame[0-9][0-9].nii.gz'), recursive=True))
        files['masks'] = sorted(glob.glob(os.path.join(path, '**/*frame*_gt.nii.gz'), recursive=True))

    elif dataset == 'GCN':
        logging.info('Using GCN dataset')
        files['images'] = sorted(glob.glob(os.path.join(path, '*clean.nrrd')))
        files['masks'] = sorted(glob.glob(os.path.join(path, '*mask.nrrd')))
        if len(files) == 0:
            files['images'] = sorted(glob.glob(os.path.join(path, '*img.nrrd')))
            files['masks'] = sorted(glob.glob(os.path.join(path, '*msk.nrrd')))
        if len(files) == 0:
            logging.info('search in subfolders ...')
            files['images'] = sorted(glob.glob(os.path.join(path, '**/*img.nrrd')))
            files['masks'] = sorted(glob.glob(os.path.join(path, '**/*msk.nrrd')))
        if len(files) == 0:
            logging.info('no files found, maybe image and mask subfolders, checking ...')
            files['images'] = sorted(glob.glob(os.path.join(path, '**/images/*.nrrd')))
            files['masks'] = sorted(glob.glob(os.path.join(path, '**/masks/*.nrrd')))

    logging.info('describing path: {}'.format(path))

    assert (len(files) > 0), 'No files found!'

    i = 0
    rows = []
    for i, f in enumerate(files['images']):
        logging.debug('Image {} of {}'.format(i, len(files['images'])))
        if i % 10 == 0:
            rows.append(describe_volume(f, plot=plot_histogram))
        else:
            rows.append(describe_volume(f, plot=False))
    if not wildcard:
        for j, f in enumerate(files['masks']):
            logging.debug('Mask {} of {}'.format(j, len(files['masks'])))
            if j % 10 == 0:
                rows.append(describe_volume(f, image=False, plot=plot_histogram))
            else:
                rows.append(describe_volume(f, image=False, plot=False))

    df_train = pd.DataFrame(rows)
    return df_train


def get_min_max_t_per_patient(df_patient, col='vol in ml', target_col='t_norm'):
    """
    Helper to get min/max of one columns
    returns a dict with {patient: 123abc , min: timestep as int, max: timestep as int}"""

    result = dict()

    patient = df_patient['patient'].unique()
    assert len(patient) == 1, 'more than one patient in df'

    idx_min = df_patient[col].idxmin()
    idx_max = df_patient[col].idxmax()

    result['patient'] = patient[0]
    result['min_t'] = df_patient.loc[idx_min, target_col]
    result['max_t'] = df_patient.loc[idx_max, target_col]

    return result


def get_extremas(df, col='vol in ml', target_col='t_norm'):
    """Helper, returns a dataframe with the timesteps describing the min/max of each patient """
    patients = df['patient'].unique()
    return pd.DataFrame([get_min_max_t_per_patient(df[df['patient'] == p], col, target_col) for p in patients])



def get_phases_as_idx_gcn(file_path, df, temporal_sampling_factor, length, label_start_with_1=False):
    """
    load the phase info of a gcn data structure
    and converts it into a onehot vector
    # order of phase classes, learnt by the phase regression model
    # ['ED#', 'MS#', 'ES#', 'PF#', 'MD#']]
    Parameters
    ----------
    file_path :
    df :
    temporal_sampling_factor :
    length :
    weight :

    Returns
    -------

    """
    import re
    str_match = re.search('-(.{8})_', file_path)
    if str_match: # gcn
        patient_str = str_match.group(1).lower()
        assert (len(patient_str) == 8), 'matched patient ID from the phase sheet has a length of: {}'.format(
            len(patient_str))
    else: # indicator
        patient_str = os.path.basename(file_path).split('__')[0].lower()

    assert len(patient_str) > 0, 'len(patient_str) = 0'


    # Returns the indices in the following order: 'ED#', 'MS#', 'ES#', 'PF#', 'MD#'
    # Reduce the indices of the excel sheet by one, as the indexes start at 0, the excel-sheet at 1
    # Transform them into an one-hot representation
    indices = df[df.patient.str.contains(patient_str)][
        ['ed#', 'ms#', 'es#', 'pf#', 'md#']]
    indices = indices.values[0].astype(int)# - 1 # the excel sheet starts with 1, indices needs to start with 0
    if label_start_with_1: indices = indices -1
    # scale the idx as we resampled along t (we need to resample the indicies in the same way)
    indices = np.round(indices * temporal_sampling_factor).astype(int)
    # if we have indicies that are bigger than the length, this is a strong indicator for indicies starting at 1 instead of 0
    # here we need to subtract one in order to match the idx of the arrays
    # the problem: this function only see the indicies of one patient
    #indices = indices - 1

    assert (indices<length).all(), 'invalid indicies, maybe they start with 1 instead of with 0? --> {}, length: {}'.format(indices, length)
    #indices = np.clip(indices, a_min=0, a_max=length - 1)
    return  indices

def get_phases_patient_split_by__(file_path, df, temporal_sampling_factor, length):
    patient_str = os.path.basename(file_path).split('__')[0].lower()
    assert len(
        patient_str) > 0, 'empty patient id found, please check the get_patient_id lambda in fn get_phases_as_idx_dmd()'

    # Returns the indices in the following order: 'ED#', 'MS#', 'ES#', 'PF#', 'MD#'
    # Reduce the indices of the excel sheet by one, as the indexes start at 0, the excel-sheet at 1
    # Transform them into an one-hot representation
    use_gt = False
    if use_gt:
        indices = df[df.patient.str.contains(patient_str, case=False)][
            ['ed_gt', 'ms_gt', 'es_gt', 'pf_gt', 'md_gt']]
    else:
        indices = df[df.patient.str.contains(patient_str, case=False)][
            ['ed#', 'ms#', 'es#', 'pf#', 'md#']]
    if len(indices) == 1:
        indices = indices.values[0].astype(
            int)  # only the GT started with 1. All predictions start with 0- 1 # the excel sheet starts with 1, indices needs to start with 0
    else:
        print('failed to load the key frame indices with patient:', patient_str)
    # scale the idx as we resampled along t (we need to resample the indicies in the same way)
    indices = np.round(indices * temporal_sampling_factor).astype(int)
    assert ((indices >= 0).all()) and ((indices < length).all()), 'indicies are: {}, but we have only {} frames'.format(
        indices, length)
    return indices


def get_phases_as_idx_dmd(file_path, df, temporal_sampling_factor, length):
    """
    load the phase info of a dmd data structure
    and converts it into a onehot vector
    # order of phase classes, learnt by the phase regression model
    # ['ED#', 'MS#', 'ES#', 'PF#', 'MD#']]
    Parameters
    ----------
    file_path :
    df :
    temporal_sampling_factor :
    length :
    weight :

    Returns
    -------

    """
    patient_str = os.path.basename(file_path).split('_volume')[0].lower()
    assert len(patient_str) > 0, 'empty patient id found, please check the get_patient_id lambda in fn get_phases_as_idx_dmd()'

    # Returns the indices in the following order: 'ED#', 'MS#', 'ES#', 'PF#', 'MD#'
    # Reduce the indices of the excel sheet by one, as the indexes start at 0, the excel-sheet at 1
    # Transform them into an one-hot representation
    use_gt = False
    if use_gt:
        indices = df[df.patient.str.contains(patient_str, case=False)][
            ['ed_gt', 'ms_gt', 'es_gt', 'pf_gt', 'md_gt']]
    else:
        indices = df[df.patient.str.contains(patient_str, case=False)][
        ['ed#', 'ms#', 'es#', 'pf#', 'md#']]
    if len(indices) == 1:
        indices = indices.values[0].astype(int) # only the GT started with 1. All predictions start with 0- 1 # the excel sheet starts with 1, indices needs to start with 0
    else:
        print('failed to load the key frame indices with patient:',patient_str)
    # scale the idx as we resampled along t (we need to resample the indicies in the same way)
    indices = np.round(indices * temporal_sampling_factor).astype(int)
    assert ((indices >=0).all()) and ((indices<length).all()), 'indicies are: {}, but we have only {} frames'.format(indices, length)
    # this will hide unplausible in the indicies, avoid this step!
    #indices = np.clip(indices, a_min=0, a_max=length)
    return  indices

def get_phases_as_onehot_gcn(file_path, df, temporal_sampling_factor=1, length=-1, weight=1):
    """
    load the phase info of a gcn data structure
    and converts it into a onehot vector
    # order of phase classes, learnt by the phase regression model
    # ['ED#', 'MS#', 'ES#', 'PF#', 'MD#']]
    the GCN phase labels start with 1, different to indexing,
    we need to subtract 1 from each idx
    Parameters
    ----------
    file_path :
    df :
    temporal_sampling_factor :
    length :
    weight :

    Returns
    -------

    """
    import re

    patient_str, ind, indices = '', '', ''
    patient_str = re.search('-(.{8})_', file_path)
    if patient_str:  # GCN data
        patient_str = patient_str.group(1).upper()
        assert (
                    len(patient_str) == 8), 'matched patient ID from the phase sheet has a length of: {}, expected a length of 8 for GCN data'.format(
            len(patient_str))
    else:  # DMD data
        patient_str = os.path.basename(file_path).split('_volume')[0].upper()

    if 'nii.gz' in patient_str:  # ACDC files e.g.: patient001_4d.nii.gz
        patient_str = re.search('patient(.{3})_', file_path)
        patient_str = patient_str.group(1).upper()

    assert len(
        patient_str) > 0, 'empty patient id found, please check the get_patient_id in fn train_fold(), usually there are path problems'

    # Returns the indices in the following order: 'ED#', 'MS#', 'ES#', 'PF#', 'MD#'
    # Reduce the indices of the excel sheet by one, as the indexes start at 0, the excel-sheet at 1
    # Transform them into an one-hot representation
    indices = df[df.patient.str.upper().str.contains(patient_str.upper())][
        ['ED#', 'MS#', 'ES#', 'PF#', 'MD#']]
    #if np.all(indices): # returns true if there are no zeros in this array (which means that they started counting at 1)
    indices = indices.values[0].astype(int) - 1

    # scale the idx as we resampled along t (we need to resample the indicies in the same way)
    if temporal_sampling_factor!=1:
        indices = np.round(indices * temporal_sampling_factor).astype(int)
        indices = np.clip(indices, a_min=0, a_max=length)

    if np.any(indices>length):
        logging.error('found indicies  greater than length of cardiac cycle, please check: {}'.format(indices))
    onehot = np.zeros((indices.size, length))
    onehot[np.arange(indices.size), indices] = weight
    return onehot

def get_phases_as_idx_acdc(file_path, temporal_sampling_factor, length):
    """
    load the phase info of an acdc data structure
    and converts it into a onehot vector
    # order of phase classes, learnt by the phase regression model
    # ['ED#', 'MS#', 'ES#', 'PF#', 'MD#']]
    Parameters
    ----------
    file_path :
    temporal_sampling_factor :
    weight :

    Returns
    -------

    """
    # load cfg for one file/patient
    temp_p = os.path.dirname(os.path.abspath(file_path))
    temp_cfg_f = os.path.join(temp_p, 'Info.cfg')
    temp_cfg = dict()
    cfg_f = open(temp_cfg_f)
    for l in cfg_f:
        key, value = l.split(':')
        temp_cfg[key] = value.replace('\n', '').replace(' ', '')

    # extract ED/ES timetemp
    temp_ed = int(temp_cfg['ED'])
    temp_es = int(temp_cfg['ES'])
    temp_length = int(temp_cfg['NbFrame'])
    # create onehot vector, set the other phases to zero
    idx = np.zeros(5)
    idx[0] = temp_ed
    idx[2] = temp_es
    # order of phase classes, learnt by the phase regression model
    # ['ED#', 'MS#', 'ES#', 'PF#', 'MD#']]
    indices = np.round(idx * temporal_sampling_factor).astype(int)
    indices = np.clip(indices, a_min=0, a_max=length-1)
    return indices

def get_phases_as_onehot_acdc(file_path, df, temporal_sampling_factor=1, length=-1, weight=1):
    """
    load the phase info of a gcn data structure
    and converts it into a onehot vector
    # order of phase classes, learnt by the phase regression model
    # ['ED#', 'MS#', 'ES#', 'PF#', 'MD#']]
    the ACDC phase labels start with 0, similar to indexing, so no "-1" necessary
    Parameters
    ----------
    file_path :
    df :
    temporal_sampling_factor :
    length :
    weight :

    Returns
    -------

    """
    import re

    patient_str, ind, indices = '', '', ''
    patient_str = re.search('-(.{8})_', file_path)
    if patient_str:  # GCN data
        patient_str = patient_str.group(1).upper()
        assert (
                    len(patient_str) == 8), 'matched patient ID from the phase sheet has a length of: {}, expected a length of 8 for GCN data'.format(
            len(patient_str))
    else:  # DMD data
        patient_str = os.path.basename(file_path).split('_volume')[0].lower()

    if 'nii.gz' in patient_str:  # ACDC files e.g.: patient001_4d.nii.gz
        patient_str = re.search('patient(.{3})_', file_path)
        patient_str = patient_str.group(1).upper()

    assert len(
        patient_str) > 0, 'empty patient id found, please check the get_patient_id in fn train_fold(), usually there are path problems'



    # Returns the indices in the following order: 'ED#', 'MS#', 'ES#', 'PF#', 'MD#'
    # Reduce the indices of the excel sheet by one, as the indexes start at 0, the excel-sheet at 1
    # Transform them into an one-hot representation
    indices = df[df.patient.str.contains(patient_str)][
        ['ED#', 'MS#', 'ES#', 'PF#', 'MD#']]
    indices = indices.values[0].astype(int)

    # scale the idx as we resampled along t (we need to resample the indicies in the same way)
    if temporal_sampling_factor!=1:
        indices = np.round(indices * temporal_sampling_factor).astype(int)
        indices = np.clip(indices, a_min=0, a_max=length)

    if np.any(indices>=length):
        logging.error('found indicies  greater than length of cardiac cycle, please check: {}'.format(indices[indices>length]))
    onehot = np.zeros((indices.size, length))
    onehot[np.arange(indices.size), indices] = weight
    return onehot


def get_phases_as_onehot_acdc_cfg(file_path, temporal_sampling_factor, length, weight=1):
    """
    load the phase info of an acdc data structure
    and converts it into a onehot vector
    # order of phase classes, learnt by the phase regression model
    # ['ED#', 'MS#', 'ES#', 'PF#', 'MD#']]
    Parameters
    ----------
    file_path :
    temporal_sampling_factor :
    weight :

    Returns
    -------

    """
    # load cfg for one file/patient
    temp_p = os.path.dirname(os.path.abspath(file_path))
    temp_cfg_f = os.path.join(temp_p, 'Info.cfg')
    temp_cfg = dict()
    cfg_f = open(temp_cfg_f)
    for l in cfg_f:
        key, value = l.split(':')
        temp_cfg[key] = value.replace('\n', '').replace(' ', '')

    # extract ED/ES timetemp
    temp_ed = int(temp_cfg['ED'])
    temp_es = int(temp_cfg['ES'])
    temp_length = int(temp_cfg['NbFrame'])
    # create onehot vector, set the other phases to zero
    idx = np.zeros(5)
    idx[0] = temp_ed
    idx[2] = temp_es
    # order of phase classes, learnt by the phase regression model
    # ['ED#', 'MS#', 'ES#', 'PF#', 'MD#']]
    indices = np.round(idx * temporal_sampling_factor).astype(int)
    indices = np.clip(indices, a_min=0, a_max=length-1)

    onehot = np.zeros((indices.size, length))
    onehot[np.arange(indices.size), indices] = weight
    return onehot


def get_n_windows_from_single4D(nda4d, idx, window_size=1,register_backwards=True, intermediate=True):
    """
    Split a 4D volume in two lists of 3D volumes
    With list1[n] - list2[n] two 3D ndas which shows the start and endpoint of a timepoint n
    given by idx
    Parameters
    ----------
    nda4d : 4D nda
    idx : np.array of a list of int
    window_size : define the window size idx[n]-window_size --> idx[n]+window_size

    Returns
    -------

    """

    t1 = time()
    debug('nda4d shape: {}'.format(nda4d.shape))
    debug('idx shape: {}'.format(idx.shape))
    y_len = nda4d.shape[0]

    # define the motion window --> [x_k-window,x_k] an w in [1,2,3] depending on the temporal resolution/temporal resampling
    idxs_minus_window = idx - window_size
    #idxs_upper = idx + window_size

    debug('idx: {}'.format(idx))
    # fake ring functionality with mod
    idxs_minus_window = np.mod(idxs_minus_window, y_len) # this is faster in the generator, than the tf functions

    debug('idx lower: {}'.format(idxs_minus_window))
    logging.debug('mod took: {:0.3f} s'.format(time() - t1))
    t1 = time()

    # slice the five timesteps from all batches
    # inputs shape: (8, 36, 8, 64, 64, 1)
    # Indicies shape: 2 (lower,upper) x (8, 5)
    # results in: 2 (lower,upper) x (8, 5, 8, 64, 64, 1) volumes
    # with: (batch,phase,z,x,y,1)
    # we need to fill the dimensions from behind by [...,tf.newaxis]
    # and define the number of leading batch dimensions
    #x_k_minus_w = tf.gather_nd(nda4d, idxs_minus_window[..., tf.newaxis], batch_dims=0)
    #t_upper = tf.gather_nd(nda4d, idxs_upper[..., tf.newaxis], batch_dims=0)
    #t_lower_pre = np.squeeze(np.take(nda4d, indices=idxs_lower_pre[..., np.newaxis], axis=0))
    #t_lower_post = np.squeeze(np.take(nda4d, indices=idxs_lower_post[..., np.newaxis], axis=0))


    x_k= np.squeeze(np.take(nda4d, indices=idx[..., np.newaxis], axis=0))
    x_k_minus_w = np.squeeze(np.take(nda4d, indices=idxs_minus_window[..., np.newaxis], axis=0))
    logging.debug('first vols shape: {}'.format(x_k_minus_w.shape))
    logging.debug('gather nd took: {:0.3f} s'.format(time() - t1))

    # INVERTED REGISTRATION TEST
    # fixed = x_k-w, moving = x_k
    if register_backwards:
        windows = [x_k_minus_w, x_k]
    else:
        windows = [x_k, x_k_minus_w]


    return windows

def get_n_windows_between_phases_from_single4D(nda4d, idx, register_backwards=True, intermediate=True):
    """
    Split a 4D volume in two lists of 3D volumes
    With list1[n] - list2[n] two 3D ndas which shows the start and endpoint of a timepoint n
    given by idx
    Parameters
    ----------
    nda4d : 4D nda
    idx : np.array of a list of int
    register_backwards: (bool), True = register T+1 --> T

    idxs (ED,MS,ES,PF,MD)
    idxs_middle
    idx_shift_to_left (MS,ES,PF,MD,ED)

    returns [nda[idx_shift_to_left], nda[idx_middle], nda[idxs]]
    Returns [vol[t+1], vol[t+0.5], vol[t]]
    -------

    """
    return_sitk = False
    if type(nda4d) == type(sitk.Image):
        sitk_save = nda4d
        return_sitk = True
        nda4d = sitk.GetArrayFromImage(nda4d)

    t1 = time()
    debug('nda4d shape: {}'.format(nda4d.shape))
    debug('idx shape: {}'.format(idx.shape))
    y_len = nda4d.shape[0]

    #
    idxs_phases = idx.copy()

    idxs_shift_to_left = np.roll(idx, 1) # shift to the right
    #shifts
    # np.roll([0,1,2,3], shift=-1)
    # --> array([1, 2, 3, 0])
    # np.roll([0,1,2,3], shift=-1)
    # --> array([3, 0, 1, 2])

    # We add a volume from t/2, which should lie between both phases, for the cycle overflow we need
    # to handle a special case

    # failure cases if we first divide or first apply modulo
    # [ 2  4  8 10 14]
    # [ 4  8 10 14  2]

    # 1. divide by 2 and than mod by length --> last index with "8" is wrong
    # [ 3  6  9 12  8]

    # 2. mod by length and than divide by 2 --> third and fourth index with "1" and "4" are wrong.
    # [3 6 1 4 0]

    # we need a different operation for the cycle case (cf. np.where function below):
    # which yields:
    # array([ 3,  6,  9, 12,  0])

    idxs_middle = np.where(idxs_shift_to_left > idxs_phases, np.mod((idxs_phases + idxs_shift_to_left) // 2, y_len), np.mod((idxs_phases + idxs_shift_to_left), y_len) // 2)

    debug('idx: {}'.format(idxs_phases))
    # fake ring functionality with mod
    idxs_phases = np.mod(idxs_phases, y_len)  # this is faster in the generator, than the tf functions
    idxs_middle = np.mod(idxs_middle, y_len)
    idxs_shift_to_left = np.mod(idxs_shift_to_left, y_len)

    debug('idx lower: {}'.format(idxs_phases))
    debug('idx shift: {}'.format(idxs_shift_to_left))
    logging.debug('mod took: {:0.3f} s'.format(time() - t1))
    t1 = time()

    # slice the five timesteps from all batches
    # inputs shape: (8, 36, 8, 64, 64, 1)
    # Indicies shape: 2 (lower,upper) x (8, 5)
    # results in: 2 (lower,upper) x (8, 5, 8, 64, 64, 1) volumes
    # with: (batch,phase,z,x,y,1)
    # we need to fill the dimensions from behind by [...,tf.newaxis]
    # and define the number of leading batch dimensions

    t_middle = np.squeeze(np.take(nda4d, indices=idxs_middle[..., np.newaxis], axis=0))
    t_phases = np.squeeze(np.take(nda4d, indices=idxs_phases[..., np.newaxis], axis=0))
    t_shift_to_left = np.squeeze(np.take(nda4d, indices=idxs_shift_to_left[..., np.newaxis], axis=0))
    logging.debug('first vols shape: {}'.format(t_phases.shape))
    logging.debug('gather nd took: {:0.3f} s'.format(time() - t1))
    if return_sitk: # return sitk images
        t_phases, t_middle, t_shift_to_left = sitk.GetImageFromArray(t_phases), sitk.GetImageFromArray(t_middle), sitk.GetImageFromArray(t_shift_to_left)
        windows = list(map(lambda x: copy_meta_and_save(new_image=x,
                                                       reference_sitk_img=sitk_save,
                                                       full_filename = None,
                                                       overwrite_spacing = None,
                                                       copy_direction = True),[t_shift_to_left, t_middle, t_phases]))

    else:
        # INVERTED REGISTRATION TEST
        # original: # T=fixed, T+1=moving
        if register_backwards:
            windows= [t_shift_to_left, t_middle, t_phases]
        else:
            windows = [t_phases, t_middle, t_shift_to_left]

        if not intermediate:
            windows = windows[0:1] + windows[-1:] # exclude intermediate timestep

    return windows
    # here: T=moving, T+1=fixed, seems to register worse, need to check compose
    # return [t_phases, t_middle, t_shift_to_left]


def save_3d(nda, fname, isVector=False, cfg=None):
    # save one flowfield
    nda = np.squeeze(nda)
    if nda.ndim == 4 and nda.shape[-1]==3:
        nda = np.einsum('zyxc->czyx', nda)
    sitk_img = sitk.GetImageFromArray(nda, isVector=isVector)
    if cfg is not None:
        spacing = list(reversed(cfg.get('SPACING')))
        if nda.ndim==4 and nda.shape[-1]!=3:
            spacing = (*spacing,1)
        sitk_img.SetSpacing(spacing)
    sitk.WriteImage(sitk_img, fname)

def save_gt_and_pred(gt, pred, exp_path, patient, cfg=None):
    """
    Save the ground truth mask and the deformed mask for a given patient and phase
    Parameters
    ----------
    gt :
    pred :
    exp_path :
    patient :

    Returns
    -------

    """
    cardiac_phases = ['ED', 'MS', 'ES', 'PF', 'MD']

    gt_path = os.path.join(exp_path, 'gt_m')
    pred_path = os.path.join(exp_path, 'pred_m')
    ensure_dir(gt_path)
    ensure_dir(pred_path)

    gt = np.einsum('tzyxc->cxyzt', gt)
    pred = np.einsum('tzyxc->cxyzt', pred)

    for t, phase in enumerate(cardiac_phases):
        gt_file_name = os.path.join(gt_path, "{}_{}.nii".format(patient, phase))
        pred_file_name = os.path.join(pred_path, "{}_{}.nii".format(patient, phase))
        save_3d(gt[...,t], gt_file_name,cfg=cfg)
        save_3d(pred[...,t], pred_file_name, cfg=cfg)


def save_all_3d_vols(inputs, outputs, moved, moved_mask, flow, inputs_mask, outputs_mask, inputs_lvmask, outputs_lvmask, inputs_full, outputs_full, EXP_PATH, exp='example_flows', save2dplus_t=False):
    from logging import info
    experiment_ = '{}/{}'.format(EXP_PATH, exp)
    info(experiment_)
    ensure_dir(experiment_)
    maskname = os.path.join(experiment_, '_mask.nii')
    lvmaskname = os.path.join(experiment_, '_lvmask.nii')
    flowname = os.path.join(experiment_, '_flow.nii')
    firstfilename = os.path.join(experiment_, '_cmr.nii')
    secondfilename = os.path.join(experiment_, '_targetcmr.nii')
    secondmaskname = os.path.join(experiment_, '_targetmask.nii')
    lvsecondmaskname = os.path.join(experiment_, '_lvtargetmask.nii')
    movedfilename = os.path.join(experiment_, '_movedcmr.nii')
    movedmaskfilename = os.path.join(experiment_, '_movedmask.nii')
    firstfilename_full = os.path.join(experiment_, '_cmr_full.nii')
    secondfilename_full = os.path.join(experiment_, '_targetcmr_full.nii')

    # invert the axis
    flow = np.einsum('tzyxc->cxyzt', flow)
    inputs = np.einsum('tzyxc->cxyzt', inputs)
    outputs = np.einsum('tzyxc->cxyzt', outputs)
    moved = np.einsum('tzyxc->cxyzt', moved)
    moved_mask = np.einsum('tzyxc->cxyzt', moved_mask)
    inputs_mask = np.einsum('tzyxc->cxyzt', inputs_mask)
    outputs_mask = np.einsum('tzyxc->cxyzt', outputs_mask)
    inputs_lvmask = np.einsum('tzyxc->cxyzt', inputs_lvmask)
    outputs_lvmask = np.einsum('tzyxc->cxyzt', outputs_lvmask)
    inputs_full = np.einsum('tzyxc->cxyzt', inputs_full)
    outputs_full = np.einsum('tzyxc->cxyzt', outputs_full)

    _ = [save_3d(flow[..., t], flowname.replace('.nii', '_{}_.nii'.format(t))) for t in range(flow.shape[-1])]
    _ = [save_3d(inputs[..., t], firstfilename.replace('.nii', '_{}_.nii'.format(t))) for t in range(inputs.shape[-1])]
    _ = [save_3d(outputs[..., t], secondfilename.replace('.nii', '_{}_.nii'.format(t))) for t in
         range(outputs.shape[-1])]
    _ = [save_3d(moved[..., t], movedfilename.replace('.nii', '_{}_.nii'.format(t))) for t in
         range(moved.shape[-1])]
    _ = [save_3d(moved_mask[..., t], movedmaskfilename.replace('.nii', '_{}_.nii'.format(t))) for t in
         range(moved_mask.shape[-1])]
    _ = [save_3d(inputs_mask[...,t], maskname.replace('.nii', '_{}_.nii'.format(t))) for t in
         range(inputs_mask.shape[-1])]
    _ = [save_3d(outputs_mask[...,t], secondmaskname.replace('.nii', '_{}_.nii'.format(t))) for t in
         range(outputs_mask.shape[-1])]
    _ = [save_3d(inputs_lvmask[...,t], lvmaskname.replace('.nii', '_{}_.nii'.format(t))) for t in
         range(inputs_lvmask.shape[-1])]
    _ = [save_3d(outputs_lvmask[...,t], lvsecondmaskname.replace('.nii', '_{}_.nii'.format(t))) for t in
         range(outputs_lvmask.shape[-1])]
    _ = [save_3d(inputs_full[..., t], firstfilename_full.replace('.nii', '_{}_.nii'.format(t))) for t in range(inputs_full.shape[-1])]
    _ = [save_3d(outputs_full[..., t], secondfilename_full.replace('.nii', '_{}_.nii'.format(t))) for t in range(outputs_full.shape[-1])]

def save_all_3d_vols_new(volumes, vol_suffixes, EXP_PATH, exp='example_flows',cfg=None):

    """
    Parameters
    ----------
    volumes : list of nda
    vol_suffixes : list of filenames
    EXP_PATH : (str), path to export
    exp :

    Returns
    -------

    """

    assert type(volumes) == type([])
    assert type(vol_suffixes) == type([])
    from logging import info
    experiment_ = '{}/{}'.format(EXP_PATH, exp)
    info(experiment_)
    ensure_dir(experiment_)
    # iterate over volumes and suffixes save each tuple
    list(map(lambda x : save_phases(x[0], experiment_, x[1], cfg=cfg),list(zip(volumes, vol_suffixes))))



def save_phases(nda, experiment_, suffix, cfg=None):
    """
    Save each 3D nda of a 4D nda with reversed axis order
    expects an nda with: t,z,y,x,c --> saves t times with nda axis of c,x,y,z
    Note: simpleITK will again invert the axis, so that the dicom files have a order of z,y,x,c
    Parameters
    ----------
    nda :
    experiment_ :
    suffix :

    Returns
    -------

    """
    f_name = os.path.join(experiment_, suffix)
    # invert the axis
    #nda = np.einsum('tzyxc->cxyzt', nda)
    _ = [save_3d(nda[t,...], f_name.replace('.nii', '_{}_.nii'.format(t)), cfg=cfg) for t in range(nda.shape[0])]


def all_files_in_df(METADATA_FILE, x_train_sax, x_val_sax):
    import pandas as pd
    import os, re, logging
    all_present = True

    df = pd.read_csv(METADATA_FILE, dtype={'patient': str, 'ED#': int, 'MS#': int, 'ES#': int, 'PF#': int, 'MD#': int})
    df.columns = df.columns.str.lower()
    df['patient'] = df['patient'].str.lower()
    DF_METADATA = df[['patient', 'ed#', 'ms#', 'es#', 'pf#', 'md#']].copy()
    DF_METADATA['patient'] = DF_METADATA['patient'].str.zfill(3).copy()
    files_ = x_train_sax + x_val_sax
    logging.info('Check if we find the patient ID and phase mapping for all: {} files.'.format(len(files_)))
    for x in files_:
        try:
            patient_str, ind, indices = '', '', ''
            patient_str = re.search('-(.{8})_', x)
            if patient_str:  # GCN data
                patient_str = patient_str.group(1).upper()
                assert (
                        len(patient_str) == 8), 'matched patient ID from the phase sheet has a length of: {}, expected a length of 8 for GCN data'.format(
                    len(patient_str))
            else:  # DMD data
                patient_str = os.path.basename(x).split('_volume')[0].lower()

            if 'nii.gz' in patient_str:  # ACDC files e.g.: patient001_4d.nii.gz
                patient_str = re.search('patient(.{3})_', x)
                patient_str = patient_str.group(1).upper()

            assert len(
                patient_str) > 0, 'empty patient id found, please check the get_patient_id in fn train_fold(), usually there are path problems'
            # returns the indices in the following order: 'ED#', 'MS#', 'ES#', 'PF#', 'MD#'
            # reduce by one, as the indexes start at 0, the excel-sheet at 1
            ind = DF_METADATA[DF_METADATA.patient.str.lower().str.contains(patient_str.lower())][
                ['ed#', 'ms#', 'es#', 'pf#', 'md#']]
            indices = ind.values[0].astype(int)
            if len(indices) == 0:
                all_present = False

        except Exception as e:
            logging.info(e)
            logging.info(patient_str)
            logging.info(ind)
            logging.info('indices: \n{}'.format(indices))
            all_present = False
    logging.info('Check done!')
    return all_present


def load_phase_reg_exp(exp_root):
    """
    Load the predicted numpy files of a 4-fold cross validation experiment

    Parameters
    ----------
    exp_root : (str) path to the experiment root

    Returns (tuple of ndarrays), nda_vects, gt, pred, gt_len, mov, patients
    -------

    """
    pathstovectnpy = sorted(glob.glob(os.path.join(exp_root, 'moved', '*vects_*.npy')))
    print(pathstovectnpy)
    nda_vects = np.concatenate([np.load(path_) for path_ in pathstovectnpy], axis=0)
    print(nda_vects.shape)

    # load the phase gt and pred
    pred_path = os.path.join(exp_root, 'pred')
    pathtsophasenpy = sorted(glob.glob(os.path.join(pred_path, '*gtpred*.npy')))
    print(pathtsophasenpy)
    nda_phase = np.concatenate([np.load(path_) for path_ in pathtsophasenpy], axis=1)
    print(nda_phase.shape)
    gt_, pred_ = np.split(nda_phase, axis=0, indices_or_sections=2)
    print(gt_.shape)
    gt = gt_[0, :, 0]
    pred = pred_[0, :, 0]
    print(gt.shape)
    gt_len = gt_[0, :, 1]

    # load some moved examples for easier understanding of the dimensions
    pathtomoved = sorted(glob.glob(os.path.join(exp_root, 'moved', '*moved*.npy')))
    print(len(pathtomoved))
    mov = np.concatenate([np.load(path_) for path_ in pathtomoved], axis=0)
    print(mov.shape)  # patient,time,spatial-z,y,x,channel, e.g.: 69,40,16,64,64,1

    # load a mapping to the original patient ids
    patients = []
    if os.path.exists(os.path.join(pred_path, 'patients.txt')):
        with open(os.path.join(pred_path, 'patients.txt'), "r") as f_:
            lines = f_.readlines()
            _ = [patients.append(p) for p in lines]

    return nda_vects, gt, pred, gt_len, mov, patients


def load_tof_phase_gt(filename='/mnt/ssd/data/tof/02_imported_4D_unfiltered/SAx_3D_dicomTags_phase.csv'):
    """
    Load a csv file with the ground truth idx per phase
    This method behave different for the ACDC GT
    Parameters
    ----------
    filename : (str) path to the csv file

    Returns pd.DataFrame with ['patient', 'ED#', 'MS#', 'ES#', 'PF#', 'MD#']
    -------

    """

    # tof, this phase2idx mapping starts with idx 1, we need a format where we start with 0
    gt_df = pd.read_csv(filename)
    gt_df['patient'] = gt_df['patient'].str.lower()
    gt_df = gt_df[['patient', 'ED#', 'MS#', 'ES#', 'PF#', 'MD#']]
    print('min\n', gt_df[['ED#', 'MS#', 'ES#', 'PF#', 'MD#']].min())
    gt_df[['ED#', 'MS#', 'ES#', 'PF#', 'MD#']] = gt_df[['ED#', 'MS#', 'ES#', 'PF#', 'MD#']] - 1
    gt_df[['ED#', 'MS#', 'ES#', 'PF#', 'MD#']] = gt_df[['ED#', 'MS#', 'ES#', 'PF#', 'MD#']].astype('int')
    gt_df = gt_df.drop_duplicates(subset='patient')
    return gt_df

def load_acdc_phase_gt(filename='/mnt/ssd/data/acdc/02_imported_4D_unfiltered/SAx_3D_dicomTags_phase.csv'):
    # acdc
    # load gt in the same order as this dataframe, merge on patient
    gt_df = pd.read_csv(filename)
    gt_df['patient'] = gt_df['patient'].apply(lambda x: str(x).zfill(3))
    #print('min\n', gt_df[['ED#', 'MS#', 'ES#', 'PF#', 'MD#']].min())
    return gt_df

def calc_vol_along_t(file_4d, label=3):
    """
    Calc the volume over time for a given 4D CMR filename and a label
    labels are usually encoded as:
    0,1,2,3 = background,RV,MYO,LV
    Parameters
    ----------
    file_4d : (str) filename for a 4D CMR
    label : (int) defining the flat value for a label of interest

    Returns (list) with len(list)== 4D_cmr.shape[0] and the corresponding 3D volumes in ml
    -------

    """
    temp = sitk.ReadImage(file_4d)
    assert temp.GetDimension()==4,'please provide a list of 4D files, got: {}'.format(temp.GetDimension())
    spacing = temp.GetSpacing()
    nda = sitk.GetArrayFromImage(temp)
    lv_voxels = (nda==label).sum(axis=(1,2,3))
    voxel_size = spacing[0]*spacing[1]*spacing[2]
    lv_voxels = (lv_voxels*voxel_size)/1000
    return lv_voxels

def create_lv_vol_df(filenames, dataset='acdc'):
    """
    Create a dataframe with:

    df = pd.DataFrame({'patient_long':patients_long,
                   'patient': patients,
                   'ed': ed_idxs,
                   'es': es_idxs,
                   'volume_change': volumes
                  'cycle_len': cycle_len}

    Parameters
    ----------
    filenames : (list of str) list of full paths to 4D CMR files (should work with nrrd and nifti)
    dataset : (str) either 'acdc' or 'tof'

    Returns pd.DataFrame
    -------

    """

    assert len(filenames)>0,'please provide a list of 4D files'
    assert dataset in ['acdc', 'tof']

    volumes = [calc_vol_along_t(x) for x in filenames]
    ed_idxs = [np.argmax(x) for x in volumes]
    es_idxs = [np.argmin(x) for x in volumes]
    cycle_len = [sitk.ReadImage(x).GetSize()[-1] for x in filenames]
    patients_long = [os.path.basename(x).split('_')[0] for x in filenames]

    # the patient id is different depending on the dataset
    if dataset.lower() == 'acdc':
        patients = [x.split('patient')[1] for x in patients_long]
    else:
        patients = [x.split('-')[1].lower() for x in patients_long]

    return pd.DataFrame({'patient_long': patients_long,
                       'patient': patients,
                       'ed_idxs': ed_idxs,
                       'es_idxs': es_idxs,
                       'volume_change': volumes,
                       'cycle_len': cycle_len})

def predict_phase_from_vol(filenames, dataset):
    """
    Calc the LV volume per 3D volume and predict the ED/ES phase
    Merge the prediction with the gt
    Calc the pFD per ED/ES
    Calc the Accuracy per ED/ES
    Parameters
    ----------
    filenames : list of 4D filenames
    dataset : (str) one of ['acdc', 'tof']

    Returns
    -------

    """
    # predict the LV volume over time, create a dataframe
    df = create_lv_vol_df(filenames=filenames, dataset=dataset)

    # load the gt
    if dataset == 'acdc':
        gt_df = load_acdc_phase_gt()
    else:
        gt_df = load_tof_phase_gt()

    # inner join of pred and gt
    return pd.merge(left=df, right=gt_df, how='inner', on='patient')