import logging
from logging import info, debug
import sys
import os
import SimpleITK as sitk

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from skimage.transform import resize
from src.data.Dataset import describe_sitk, get_metadata_maybe, copy_meta_and_save, get_patient
import numpy as np
from src.visualization.Visualize import plot_3d_vol
from albumentations import GridDistortion, RandomRotate90, Compose, ReplayCompose, Flip, Transpose, OneOf, IAAAdditiveGaussianNoise, \
    MotionBlur, MedianBlur, Blur, OpticalDistortion, IAAPiecewiseAffine, CLAHE, IAASharpen, IAAEmboss, \
    RandomBrightnessContrast, HueSaturationValue, ElasticTransform, CenterCrop, PadIfNeeded, RandomBrightness, Downscale, ShiftScaleRotate
import cv2
from src.data.Dataset import copy_meta
from albumentations.augmentations.transforms import PadIfNeeded, GaussNoise, RandomGamma


def load_masked_img(sitk_img_f, mask=False, masking_values = [1,2,3], replace=('img','msk'), mask_labels=[0,1,2,3]):

    """
    Wrapper for opening a dicom image, this wrapper could also load the corresponding segmentation map and mask the loaded image on the fly
     if mask == True use the replace wildcard to open the corresponding segmentation mask
     Use the values given in mask_labels to transform the one-hot-encoded mask into channel based binary mask
     Mask/cut the CMR image/volume by the given labels in masking_values

    Parameters
    ----------
    sitk_img_f : full filename for a dicom image/volume, could be any format supported by sitk
    mask : bool, if the sitk image loaded should be cropped by any label of the corresponding mask
    masking_values : list of int, defines the area/labels which should be cropped from the original CMR
    replace : tuple of replacement string to get from the image filename to the mask filename
    mask_labels : list of int
    """

    assert os.path.isfile(sitk_img_f), 'no valid image: {}'.format(sitk_img_f)
    img_original = sitk.ReadImage(sitk_img_f, sitk.sitkFloat32)

    if mask:
        sitk_mask_f = sitk_img_f.replace(replace[0], replace[1])
        msk_original = sitk.ReadImage(sitk_mask_f)
        
        img_nda = sitk.GetArrayFromImage(img_original)
        msk_nda = transform_to_binary_mask(sitk.GetArrayFromImage(msk_original), mask_values=mask_labels)
                    
        # mask by different labels, sum up all masked channels
        temp = np.zeros(img_nda.shape)
        for c in masking_values:
            # mask by different labels, sum up all masked channels
            temp += img_nda * msk_nda[..., c].astype(np.bool)
        sitk_img = sitk.GetImageFromArray(temp)

        # copy metadata
        for tag in img_original.GetMetaDataKeys():
            value = get_metadata_maybe(img_original, tag)
            sitk_img.SetMetaData(tag, value)
        sitk_img.SetSpacing(img_original.GetSpacing())
        sitk_img.SetOrigin(img_original.GetOrigin())
                    
        img_original = sitk_img
                
    return img_original


def filter_small_vectors_batch(flowfield_3d, normalize=True, thresh_z=(-0.5, 0.5), thresh_x=(-2.5, 1.5),
                               thresh_y=(-1.5, 1.0)):
    """
    wrapper to detect input shape, works with 3d volume of flows
    Expect a numpy array with shape z,x,y,c, return the same shape
    All vector smaller or bigger than the given thresholds (tuples) will be set to the flowfield minimum
    :param flowfield_3d:
    :return:
    """

    if flowfield_3d.ndim == 4:
        # traverse through the z axis and filter each 2d slice independently
        filtered = [
            filter_small_vectors_2d(f, normalize=normalize, thresh_z=thresh_z, thresh_x=thresh_x, thresh_y=thresh_y)
            for f in flowfield_3d]
        return np.stack(filtered, axis=0)

    elif flowfield_3d.ndim == 3:  # 2d slice with 3d vectors
        return filter_small_vectors_2d(flowfield_3d, normalize=normalize, thresh_z=thresh_z, thresh_x=thresh_x,
                                       thresh_y=thresh_y)

    else:
        # returns the input without changes
        logging.error('dimension: {} not supported'.format(flowfield_3d.ndim))
        return flowfield_3d


def filter_small_vectors_2d(flowfield_2d, normalize=True, thresh_z=(-0.7, 0.7), thresh_x=(-2.5, 1.5),
                            thresh_y=(-1.5, 1.0)):
    """
    Expect a numpy array with shape z,x,y,c, return the same shape
    All vector smaller or bigger than the given thresholds (tuples) will be set to the flowfield minimum
    :param flowfield_3d:
    :return:
    """
    flow_min = flowfield_2d.min()

    if not normalize:
        flow_min = 0

    if flowfield_2d.shape[-1] == 3:  # 3d vectors
        flow_z = flowfield_2d[..., 0].copy()
        flow_x = flowfield_2d[..., 1].copy()
        flow_y = flowfield_2d[..., 2].copy()
    elif flowfield_2d.shape[-1] == 2:
        flow_x = flowfield_2d[..., 0].copy()
        flow_y = flowfield_2d[..., 1].copy()
        # create a fake 3rd dimension to work with rgb, set value to minimal flowfield value
        flow_z = np.full_like(flow_x, flow_min)

    else:
        logging.error('vector shape not supported')
        return flowfield_2d

    # filter small z movements
    flow_z[(flow_z > thresh_z[0]) & (flow_z < thresh_z[1])] = flow_min
    # filter small x movements
    flow_x[(flow_x > thresh_x[0]) & (flow_x < thresh_x[1])] = flow_min
    # filter small y movements
    flow_y[(flow_y > thresh_y[0]) & (flow_y < thresh_y[1])] = flow_min
    flow_ = np.stack([flow_z, flow_x, flow_y], axis=-1)

    if normalize:
        # normalize values in the scale of 0 -1 small values will result as 0
        return normalise_image(flow_)
    else:
        return flow_


def resample_t_of_4d(sitk_img, t_spacing=20, interpolation=sitk.sitkLinear, ismask=False):
    '''

    Parameters
    ----------
    sitk_img :
    t_spacing :
    interpolation :
    ismask :

    Returns
    -------

    '''

    from src.data.Dataset import split_one_4d_sitk_in_list_of_3d_sitk

    debug('before resample: {}'.format(sitk_img.GetSize()))
    debug('before resample: {}'.format(sitk_img.GetSpacing()))
    z_spacing = sitk_img.GetSpacing()[2]
    # unstack 4D volume along t
    images = split_one_4d_sitk_in_list_of_3d_sitk(sitk_img, axis=1)

    temp_slice = images[0]
    t_size = temp_slice.GetSize()
    t_spacing_old = temp_slice.GetSpacing()

    # calculate the sampling factor, and the corresponding new size in t
    sampling_factor = t_spacing_old[-1] / t_spacing
    debug('Sampling factor: {}'.format(sampling_factor))
    new_t_size = int(np.ceil(t_size[-1] * sampling_factor))
    target_size = (t_size[0], t_size[1], new_t_size)
    debug('2D+t target size: {}'.format(target_size))
    target_spacing = (t_spacing_old[0], t_spacing_old[1], t_spacing)

    resampled_cmr = list(
        map(lambda x:
            resample_3D(x, size=target_size, spacing=target_spacing, interpolate=interpolation), images))

    # get array from list of 2d+t sitk images, stack along z, unstack along t --> list of 3D volumes
    resampled_cmr_nda = list(map(sitk.GetArrayFromImage, resampled_cmr))
    debug('Before stack: {}'.format(resampled_cmr_nda[0].shape))
    resampled_cmr_nda = np.stack(resampled_cmr_nda, axis=1)

    if ismask:
        idxs_orig = get_mask_idxs(sitk_img)
        # scale the idx by the resampling factor, use this to filter the doubled resampled masks
        # resampling and filtering of the masks to avoid double time steps/phases
        idxs_scaled = np.round(idxs_orig * sampling_factor).astype(int)
        debug('scaled idx: {}'.format(idxs_scaled))
        idxs_scaled = np.clip(idxs_scaled, a_min=0, a_max=resampled_cmr_nda.shape[0] - 1)
        debug('scaled, clipped idx: {}'.format(idxs_scaled))
        temp = np.zeros_like(resampled_cmr_nda)
        temp[idxs_scaled] = resampled_cmr_nda[idxs_scaled]
        resampled_cmr_nda = temp

        mask_idx = np.max(resampled_cmr_nda, axis=(1, 2, 3))
        debug('max: {}'.format(mask_idx))
        idxs_resampled = np.where(mask_idx >= 3)[0].astype(int)
        debug('after stack: {}'.format(resampled_cmr_nda.shape))
        debug('idxs of resampled 4d vol: {}'.format(idxs_resampled))
        assert len(idxs_resampled) == 5

    # split into t x 3D volumes, which is the format we need for sitk to join a series of 3D volumes
    resampled_cmr_nda = np.split(resampled_cmr_nda, indices_or_sections=resampled_cmr_nda.shape[0], axis=0)
    debug('after split: {}'.format(len(resampled_cmr_nda)))
    resampled_cmr_sitk = list(map(lambda x: sitk.GetImageFromArray(np.squeeze(x)), resampled_cmr_nda))
    debug('after as sitk: {}'.format(resampled_cmr_sitk[0].GetSize()))
    resampled_cmr_sitk = sitk.JoinSeries(resampled_cmr_sitk)

    full_spacing = (t_spacing_old[0], t_spacing_old[1], z_spacing, t_spacing)
    resampled_cmr_sitk = copy_meta_and_save(resampled_cmr_sitk, sitk_img, None, overwrite_spacing=full_spacing)
    debug(resampled_cmr_sitk.GetSize())
    debug(resampled_cmr_sitk.GetSpacing())

    return resampled_cmr_sitk


def get_mask_idxs(sitk_img, slice_threshold=2):
    '''
    get valid temporal indices, with a slice threshold, as we have some slices labelled along all time steps
    Parameters
    ----------
    sitk_img : sitk.Image of a 4D mask

    Returns (list) indices of the temporal axis, where we have a mask volume
    -------

    '''
    # from mask2idx for mask and resampled mask
    #
    timesteps = []
    # get indexes for masked volumes
    # filter 3d volumes with less masked slices than threshold
    for t, nda_3d in enumerate(sitk.GetArrayViewFromImage(sitk_img)):
        if nda_3d.max() > 0:  # 3d volume with masks
            masked_slices = 0
            success_ = False
            for slice_ in nda_3d:  # check how many slices are masked
                if slice_.max() > 0:
                    masked_slices += 1
                if masked_slices > slice_threshold:
                    timesteps.append(t)
                    success_ = True
                    break
            if not success_: logging.debug('filter {}t of volume by masked slices threshold'.format(t))
    idxs_orig = np.array(timesteps).astype(int)
    return idxs_orig


def resample_3D(sitk_img, size=(256, 256, 12), spacing=(1.25, 1.25, 8), interpolate=sitk.sitkNearestNeighbor):
    """
    resamples an 3D sitk image or numpy ndarray to a new size with respect to the giving spacing
    This method expects size and spacing in sitk format: x, y, z
    :param sitk_img: sitk.Image
    :param size: (tuple) of int with the following order x,y,z
    :param spacing: (tuple) of float with the following order x,y,z
    :param interpolate:
    :return: the resampled image in the same datatype as submitted, either sitk.image or numpy.ndarray
    """

    return_sitk = True

    if isinstance(sitk_img, np.ndarray):
        return_sitk = False
        sitk_img = sitk.GetImageFromArray(sitk_img)

    assert (isinstance(sitk_img, sitk.Image)), 'wrong image type: {}'.format(type(sitk_img))

    # minor data type corrections
    size = [int(elem) for elem in size]
    spacing = [float(elem) for elem in spacing]

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(interpolate)
    resampler.SetSize(size)
    resampler.SetOutputDirection(sitk_img.GetDirection())
    resampler.SetOutputSpacing(spacing)
    resampler.SetOutputOrigin(sitk_img.GetOrigin())

    resampled = resampler.Execute(sitk_img)

    # return the same data type as input datatype
    if return_sitk:
        return resampled
    else:
        return sitk.GetArrayFromImage(resampled)

def random_rotate90_2D_or_3D(img, mask, probabillity=0.8):
    logging.debug('random rotate for: {}'.format(img.shape))
    augmented = {'image': None, 'mask': None}

    if isinstance(img, sitk.Image):
        img = sitk.GetArrayFromImage(img).astype(np.float32)

    if isinstance(mask, sitk.Image):
        mask = sitk.GetArrayFromImage(mask).astype(np.float32)

    # dont print anything if no images nor masks are given
    if img is None and mask is None:
        logging.error('No image data given')
        raise ('No image data given in grid dissortion')

    # replace mask with empty slice if none is given
    if mask is None:
        mask = np.zeros(img.shape)

    # replace image with empty slice if none is given
    if img is None:
        img = np.zeros(mask.shape)

    if img.ndim == 2:

        aug = RandomRotate90(p=probabillity)

        params = aug.get_params()
        image_aug = aug.apply(img, **params)
        mask_aug = aug.apply(mask, interpolation=cv2.INTER_NEAREST, **params)

        # apply shift-scale and rotation augmentation on 2d data
        augmented['image'] = image_aug
        augmented['mask'] = mask_aug

    elif img.ndim == 3:
        # apply shif-scale and rotation on 3d data, apply the same transform to all slices
        images = []
        masks = []

        aug = RandomRotate90(p=probabillity)
        params = aug.get_params()
        for z in range(img.shape[0]):
            images.append(aug.apply(img[z, ...], interpolation=cv2.INTER_LINEAR, factor=1,**params))
            masks.append(aug.apply(mask[z, ...], interpolation=cv2.INTER_NEAREST, factor=1,**params))

        augmented['image'] = np.stack(images, axis=0)
        augmented['mask'] = np.stack(masks, axis=0)

    else:
        logging.error('Unsupported dim: {}, shape: {}'.format(img.ndim, img.shape))
        raise ('Wrong shape Exception in: {}'.format('show_2D_or_3D()'))

    return augmented['image'], augmented['mask']

def augmentation_compose_2d_3d_4d(img, mask, probabillity=1, get_params=False):
    """
    Apply an compisition of different augmentation steps,
    either on 2D or 3D image/mask pairs,
    apply
    :param img:
    :param mask:
    :param probabillity:
    :return: augmented image, mask
    """
    #logging.debug('random rotate for: {}'.format(img.shape))
    return_image_and_mask = True
    img_given = True
    mask_given = True


    if isinstance(img, sitk.Image):
        img = sitk.GetArrayFromImage(img).astype(np.float32)

    if isinstance(mask, sitk.Image):
        mask = sitk.GetArrayFromImage(mask).astype(np.float32)

    # dont print anything if no images nor masks are given
    if img is None and mask is None:
        logging.error('No image data given')
        raise ('No image data given in augmentation compose')

    # replace mask with empty slice if none is given
    if mask is None:
        return_image_and_mask = False
        mask_given = False

    # replace image with empty slice if none is given
    if img is None:
        return_image_and_mask = False
        img_given = False

    targets = {}
    data = {}
    img_placeholder = 'image'
    mask_placeholder = 'mask'

    if img.ndim == 2:
        data = {"image": img, "mask": mask}

    if img.ndim == 3:
        middle_z = len(img)//2
        if mask_given:
            m_ = mask[middle_z]
        else:
            m_ = mask
        # take an image, mask pair from the middle part of the volume
        data = {"image": img[middle_z], "mask": m_}

        # add each slice of the image/mask stacks into the data dictionary
        for z in range(img.shape[0]):
            # add the other slices to the data dict
            if img_given: data['{}{}'.format(img_placeholder,z)] = img[z,...]
            if mask_given:data['{}{}'.format(mask_placeholder, z)] = mask[z, ...]
            # define the target group,
            # which slice is a mask and which an image (different interpolation)
            if img_given: targets['{}{}'.format(img_placeholder,z)] = 'image'
            if mask_given: targets['{}{}'.format(mask_placeholder, z)] = 'mask'

    if img.ndim ==4:
        middle_t = img.shape[0] // 2
        middle_z = img.shape[1] // 2
        # take an image, mask pair from the middle part of the volume and time
        if mask_given:
            data = {"image": img[middle_t][middle_z], "mask": m_}
        else:
            data = {"image": img[middle_t][middle_z]}


        for t in range(img.shape[0]):
            # add each slice of the image/mask stacks into the data dictionary
            for z in range(img.shape[1]):
                # add the other slices to the data dict
                if img_given: data['{}_{}_{}'.format(img_placeholder, t, z)] = img[t,z, ...]
                if mask_given:data['{}_{}_{}'.format(mask_placeholder, t, z)] = mask[t,z, ...]
                # define the target group,
                # which slice is a mask and which an image (different interpolation)
                if img_given: targets['{}_{}_{}'.format(img_placeholder, t, z)] = 'image'
                if mask_given: targets['{}_{}{}'.format(mask_placeholder, t,z)] = 'mask'



    # create a callable augmentation composition
    aug = _create_aug_compose(p=probabillity, targets=targets)

    # apply the augmentation
    augmented = aug(**data)
    logging.debug(augmented['replay'])

    if img.ndim == 3:
        images = []
        masks = []
        for z in range(img.shape[0]):
            # extract the augmented slices in the correct order
            if img_given: images.append(augmented['{}{}'.format(img_placeholder,z)])
            if mask_given:masks.append(augmented['{}{}'.format(mask_placeholder, z)])
        if img_given: augmented['image'] = np.stack(images,axis=0)
        if mask_given: augmented['mask'] = np.stack(masks, axis=0)

    if img.ndim == 4:
        img_4d = []
        mask_4d = []
        for t in range(img.shape[0]):
            images = []
            masks = []
            for z in range(img.shape[1]):
                # extract the augmented slices in the correct order
                if img_given: images.append(augmented['{}_{}_{}'.format(img_placeholder,t,z)])
                if mask_given: masks.append(augmented['{}_{}_{}'.format(mask_placeholder,t, z)])
            if img_given: img_4d.append(np.stack(images,axis=0))
            if mask_given: mask_4d.append(np.stack(masks, axis=0))

        if img_given: augmented['image'] = np.stack(img_4d,axis=0)
        if mask_given: augmented['mask'] = np.stack(mask_4d, axis=0)


    if return_image_and_mask:
        return augmented['image'], augmented['mask']
    else:
        # dont return the fake augmented masks if none where given
        return augmented['image']


def _create_aug_compose(p=1, border_mode=cv2.BORDER_REPLICATE, val=0, targets=None):
    if targets is None:
        targets = {}
    return ReplayCompose([
        #RandomRotate90(p=0.2),
        #Flip(0.1),
        #Transpose(p=0.1),
        ShiftScaleRotate(p=p, rotate_limit=45,shift_limit=0.025, scale_limit=0,value=val, border_mode=border_mode),
        #GridDistortion(p=p, value=val,border_mode=border_mode),
        #CenterCrop(height=target_dim[0], width=target_dim[1], p=1),
        # HueSaturationValue(p=1)
        #RandomBrightnessContrast(brightness_limit=0.02,contrast_limit=0.02,brightness_by_max=False, p=0.4),
        #Downscale(scale_min=0.9, scale_max=0.9, p=0.4),
        #RandomGamma(p=p)
        # OneOf([
        # OpticalDistortion(p=1),
        # ], p=1),
    ], p=p,
        additional_targets=targets)

def random_rotate_2D_or_3D(img, mask, probabillity=0.8, shift_limit=0.0625, scale_limit=0.0, rotate_limit=0):

    """
    Rotate, shift and scale an image within a given range
    :param img: numpy.ndarray
    :param mask: numpy.ndarray
    :param probabillity: float, will be interpreted as %-value
    :param shift_limit:
    :param scale_limit:
    :param rotate_limit:
    :return:
    """

    logging.debug('random rotate for: {}'.format(img.shape))
    augmented = {'image': None, 'mask': None}

    if isinstance(img, sitk.Image):
        img = sitk.GetArrayFromImage(img).astype(np.float32)

    if isinstance(mask, sitk.Image):
        mask = sitk.GetArrayFromImage(mask).astype(np.float32)

    # dont print anything if no images nor masks are given
    if img is None and mask is None:
        logging.error('No image data given')
        raise ('No image data given in grid dissortion')

    # replace mask with empty slice if none is given
    if mask is None:
        mask = np.zeros(img.shape)

    # replace image with empty slice if none is given
    if img is None:
        img = np.zeros(mask.shape)

    if img.ndim == 2:

        aug = ShiftScaleRotate(shift_limit=shift_limit, scale_limit=scale_limit, rotate_limit=rotate_limit,
                               border_mode=cv2.BORDER_REFLECT_101, p=probabillity)

        params = aug.get_params()
        image_aug = aug.apply(img, interpolation=cv2.INTER_LINEAR, **params)
        mask_aug = aug.apply(mask, interpolation=cv2.INTER_NEAREST, **params)

        # apply shift-scale and rotation augmentation on 2d data
        augmented['image'] = image_aug
        augmented['mask'] = mask_aug

    elif img.ndim == 3:
        # apply shif-scale and rotation on 3d data, apply the same transform to all slices
        images = []
        masks = []

        aug = ShiftScaleRotate(shift_limit=shift_limit, scale_limit=scale_limit, rotate_limit=rotate_limit,
                               border_mode=cv2.BORDER_REFLECT_101, p=probabillity)
        params = aug.get_params()
        for z in range(img.shape[0]):
            images.append(aug.apply(img[z, ...], interpolation=cv2.INTER_LINEAR, **params))
            masks.append(aug.apply(mask[z, ...], interpolation=cv2.INTER_NEAREST, **params))

        augmented['image'] = np.stack(images, axis=0)
        augmented['mask'] = np.stack(masks, axis=0)

    else:
        logging.error('Unsupported dim: {}, shape: {}'.format(img.ndim, img.shape))
        raise ('Wrong shape Exception in: {}'.format('show_2D_or_3D()'))

    return augmented['image'], augmented['mask']


def grid_dissortion_2D_or_3D(img, mask, probabillity=0.8, border_mode=cv2.BORDER_REFLECT_101, is_y_mask=True):
    """
    Apply grid dissortion
    :param img:
    :param mask:
    :return:
    """
    logging.debug('grid dissortion for: {}'.format(img.shape))
    augmented = {'image': None, 'mask': None}

    if is_y_mask:
        y_interpolation = cv2.INTER_NEAREST
    else:
        y_interpolation = cv2.INTER_LINEAR

    if isinstance(img, sitk.Image):
        img = sitk.GetArrayFromImage(img).astype(np.float32)

    if isinstance(mask, sitk.Image):
        mask = sitk.GetArrayFromImage(mask).astype(np.float32)

    # dont print anything if no images nor masks are given
    if img is None and mask is None:
        logging.error('No image data given')
        raise ('No image data given in grid dissortion')

    # replace mask with empty slice if none is given
    if mask is None:
        mask = np.zeros(img.shape)

    # replace image with empty slice if none is given
    if img is None:
        img = np.zeros(mask.shape)

    if img.ndim == 2:
        # apply grid augmentation on 2d data
        aug = GridDistortion(p=probabillity,border_mode=border_mode,mask_value=0, value=0)
        if is_y_mask:
            augmented = aug(image=img, mask=mask)
        else:
            steps = aug.get_params()
            augmented['image'] = aug.apply(img, steps['stepsx'], steps['stepsy'], interpolation=cv2.INTER_LINEAR)
            augmented['mask'] = aug.apply(mask, steps['stepsx'], steps['stepsy'], interpolation=cv2.INTER_LINEAR)
    elif img.ndim == 3:

        # apply grid augmentation on 3d data, apply the same transform to all slices
        images = []
        masks = []

        aug = GridDistortion(p=probabillity,border_mode=border_mode)
        steps = aug.get_params()
        for z in range(img.shape[0]):
            images.append(aug.apply(img[z,...], steps['stepsx'], steps['stepsy'], interpolation=y_interpolation))
            masks.append(aug.apply(mask[z,...], steps['stepsx'], steps['stepsy'], interpolation=y_interpolation))

        augmented['image'] = np.stack(images, axis=0)
        augmented['mask'] = np.stack(masks, axis=0)

    else:
        logging.error('Unsupported dim: {}, shape: {}'.format(img.ndim, img.shape))
        raise ('Wrong shape Exception in: {}'.format('show_2D_or_3D()'))

    return augmented['image'], augmented['mask']

def elastic_transoform_2D_or_3D(img, mask, probabillity=0.8):
    """
    Apply grid dissortion
    :param img:
    :param mask:
    :return:
    """
    logging.debug('grid dissortion for: {}'.format(img.shape))
    augmented = {'image': None, 'mask': None}

    if isinstance(img, sitk.Image):
        img = sitk.GetArrayFromImage(img).astype(np.float32)

    if isinstance(mask, sitk.Image):
        mask = sitk.GetArrayFromImage(mask).astype(np.float32)

    # dont print anything if no images nor masks are given
    if img is None and mask is None:
        logging.error('No image data given')
        raise ('No image data given in grid dissortion')

    # replace mask with empty slice if none is given
    if mask is None:
        mask = np.zeros(img.shape)

    # replace image with empty slice if none is given
    if img is None:
        img = np.zeros(mask.shape)

    if img.ndim == 2:

        # apply grid augmentation on 2d data
        aug = ElasticTransform(p=1, alpha=120, sigma=120 * 0.09, alpha_affine=120 * 0.08,border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0)
        augmented = aug(image=img, mask=mask)

    elif img.ndim == 3:

        # apply grid augmentation on 3d data, apply the same transform to all slices
        images = []
        masks = []

        aug = ElasticTransform(p=1, alpha=120, sigma=120 * 0.09, alpha_affine=120 * 0.08,border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0)
        steps = aug.get_params()
        for z in range(img.shape[0]):
            images.append(aug.apply(img[z,...], steps['stepsx'], steps['stepsy'], interpolation=cv2.INTER_LINEAR))
            masks.append(aug.apply(mask[z,...], steps['stepsx'], steps['stepsy'], interpolation=cv2.INTER_NEAREST))

        augmented['image'] = np.stack(images, axis=0)
        augmented['mask'] = np.stack(masks, axis=0)

    else:
        logging.error('Unsupported dim: {}, shape: {}'.format(img.ndim, img.shape))
        raise ('Wrong shape Exception in: {}'.format('show_2D_or_3D()'))

    return augmented['image'], augmented['mask']



def crop_to_square_2d_or_3d(img_nda, mask_nda, image_type='nda'):
    """
    Wrapper for 2d and 3d image/mask support
    :param img_nda:
    :param mask_nda:
    :return:
    """

    if isinstance(img_nda, sitk.Image):
        image_type = 'sitk'
        reference_img = img_nda
        img_nda = sitk.GetArrayFromImage(img_nda).astype(np.float32)

    if isinstance(mask_nda, sitk.Image):
        mask_nda = sitk.GetArrayFromImage(mask_nda).astype(np.float32)

    # dont print anything if no images nor masks are given
    if img_nda is None and mask_nda is None:
        logging.error('No image data given')
        raise ('No image data given in grid dissortion')

    # replace mask with empty slice if none is given
    if mask_nda is None:
        mask_nda = np.zeros(img_nda.shape)

    # replace image with empty slice if none is given
    if img_nda is None:
        img_nda = np.zeros(mask_nda.shape)

    if img_nda.ndim == 2:
        crop = crop_to_square_2d

    elif img_nda.ndim == 3:
        crop = crop_to_square_3d

    if image_type == 'sitk':
        # return a sitk.Image with all metadata as the uncroped image
        img, msk = crop(img_nda, mask_nda)
        return copy_meta(img,reference_img), copy_meta(msk, reference_img)

    return crop(img_nda, mask_nda)


def crop_to_square_3d(img_nda, mask_nda):
    """
    crop 3d numpy image/mask to square, croptthe longer side
    individual square cropping for image pairs such as used for ax2sax transformation
    :param img_nda:
    :param mask_nda:
    :return:
    """
    h, w = img_nda.shape[-2:]  # numpy shape has different order than sitk
    logging.debug('shape: {}'.format(img_nda.shape))
    if h != w:
        margin = (h - w) // 2

        # crop width if width > height
        if margin > 0:  # height is bigger than width, crop height
            logging.debug('margin: {}'.format(margin))
            img_nda = img_nda[:, margin:-margin, :]
            img_nda = img_nda[:, :w, :]  # make sure no ceiling errors

        elif margin < 0:  # width is bigger than height, crop width
            margin = -margin
            img_nda = img_nda[..., margin:-margin]
            img_nda = img_nda[..., :h]

    h, w = mask_nda.shape[-2:]  # numpy shape has different order than sitk
    logging.debug('shape: {}'.format(img_nda.shape))
    if h != w:
        margin = (h - w) // 2

        # crop width if width > height
        if margin > 0:  # height is bigger than width, crop height
            logging.debug('margin: {}'.format(margin))
            mask_nda = mask_nda[:, margin:-margin, :]
            mask_nda = mask_nda[:, :w, :]

        elif margin < 0:  # width is bigger than height, crop width
            margin = -margin
            mask_nda = mask_nda[..., margin:-margin]
            mask_nda = mask_nda[..., :h]

    return img_nda, mask_nda

def crop_to_square_3d_same_shape(img_nda, mask_nda):
    """
    crop 3d numpy image/mask to square, croptthe longer side
    Works only if img and mask have the same shape
    :param img_nda:
    :param mask_nda:
    :return:
    """
    h, w = img_nda.shape[-2:]  # numpy shape has different order than sitk
    logging.debug('shape: {}'.format(img_nda.shape))
    if h != w:
        margin = (h - w) // 2

        # crop width if width > height
        if margin > 0:  # height is bigger than width, crop height
            logging.debug('margin: {}'.format(margin))
            img_nda = img_nda[:, margin:-margin, :]
            img_nda = img_nda[:, :w, :]  # make sure no rounding errors
            mask_nda = mask_nda[:, margin:-margin, :]
            mask_nda = mask_nda[:, :w, :]

        elif margin < 0:  # width is bigger than height, crop width
            margin = -margin
            img_nda = img_nda[..., margin:-margin]
            mask_nda = mask_nda[..., margin:-margin]
            img_nda = img_nda[..., :h]
            mask_nda = mask_nda[..., :h]

    return img_nda, mask_nda


def crop_to_square_2d(img_nda, mask_nda):
    """
    center crop image and mask to square
    :param img_nda:
    :param mask_nda:
    :return:
    """

    w, h = mask_nda.shape[:2]
    # identify if width or height is bigger
    if h != w:
        margin = (h - w) // 2
        # crop width if width > height
        if margin > 0: # height > width, crop height
            img_nda = img_nda[margin:-margin, :]
            mask_nda = mask_nda[margin:-margin, :]
        elif margin < 0: # width > height, crop width
            margin = -margin
            img_nda = img_nda[:, margin:-margin]
            mask_nda = mask_nda[:, margin:-margin]

    return img_nda, mask_nda


def crop_center_2d(img, croph, cropw):
    """
    crop a 2D array inplane
    :param img:
    :param croph: target height as int
    :param cropw: target width as int
    :return: cropped image as np.dnarray with w,h
    """
    h, w = img.shape[:2]
    starth = h // 2 - (croph // 2)
    startw = w // 2 - (cropw // 2)
    starth = max(starth, 0)
    startw = max(startw, 0)

    return img[ starth:starth + croph,startw:startw + cropw, ...]


def crop_center_3d(img, cropz, cropx, cropy):
    """
    Crop z from zero to size
    Center crop x and y
    :param img:
    :param cropz:
    :param cropx:
    :param cropy:
    :return:
    """
    z, y, x = img.shape # get size of the last three axis
    logging.debug('image shape at the beginning of crop_center: {}\n cropx : {}'.format(img.shape, cropx))
    if cropx >= x and cropy >= y: # if x and y (square shape at this point) are smaller than the desired x,y dont crop
        logging.debug('Just crop z')
        return img[:cropz, ...] # crop only z
    else:
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        return img[:cropz, starty:starty + cropy, startx:startx + cropx]


def center_crop_or_resize_2d_or_3d(img_nda, mask_nda, dim):
    """

    :param img:
    :param mask:
    :param dim:
    :return:
    """

    if isinstance(img_nda, sitk.Image):
        img_nda = sitk.GetArrayFromImage(img_nda).astype(np.float32)

    if isinstance(mask_nda, sitk.Image):
        mask_nda = sitk.GetArrayFromImage(mask_nda).astype(np.float32)

    # dont print anything if no images nor masks are given
    if img_nda is None and mask_nda is None:
        logging.error('No image data given')
        raise ('No image data given in center_crop_or_resize')

    # replace mask with empty slice if none is given
    if mask_nda is None:
        mask_nda = np.zeros(img_nda.shape)

    # replace image with empty slice if none is given
    if img_nda is None:
        img_nda = np.zeros(mask_nda.shape)

    if img_nda.ndim == 2:
        # TODO: implement crop or pad for 2D
        crop = center_crop_or_pad_2d

    elif img_nda.ndim == 3:
        crop = center_crop_or_pad_3d
    else:
        raise NotImplementedError('Dim: {} not supported'.format(img_nda.ndim))

    return crop(img_nda, mask_nda, dim)


def center_crop_or_resize_3d(img_nda, mask_nda, dim):
    """
    center crop to given size, check if bigger or smaller
    requires square shape as input
    skimage resize takes the order parameter which defines the interpolation
        0: Nearest-neighbor
        1: Bi-linear (default)
        2: Bi-quadratic
        3: Bi-cubic
        4: Bi-quartic
        5: Bi-quintic
    img nda is a ndarray or tensor with z, y, x

    :param img_nda: numpy array
    :param mask_nda: numpy array
    :param dim: 2-3
    :return:
    """
    resized_by = 'no resize'
    # center crop in z
    # center crop slice by slice to given size, check if bigger or smaller
    # nda is already in inplane square resolution, need to check only x
    if (img_nda.shape[2] > dim[2] or img_nda.shape[0] > dim[0]):  # if nda bigger than wished output, crop, else resize
        resized_by = 'center crop'
        img_nda = crop_center_3d(img_nda, dim[0], dim[1], dim[2])
    if (mask_nda.shape[2] > dim[2] or mask_nda.shape[0] > dim[0]):  # if nda bigger than wished output, crop, else resize
        resized_by = 'center crop'
        mask_nda = crop_center_3d(mask_nda, dim[0], dim[1], dim[2])

    if img_nda.shape[2] < dim[2]: # sometimes we have a volume which should be cropped along z but resized along x and y
        resized_by = 'skimage resize'
        logging.debug('image too small, need to resize slice wise')
        logging.debug('image size: {}'.format(img_nda.shape))
        # resize
        imgs = []
        for img in img_nda:
            imgs.append(resize(img, dim[1:], mode='constant', preserve_range=True, order=3,anti_aliasing=True, clip=True))
        # pad along z
        img_nda = np.stack(imgs, axis=0)
    if mask_nda.shape[2] < dim[2]:  # sometimes we have a volume which should be cropped along z but resized along x and y
        resized_by = 'skimage resize'
        logging.debug('image too small, need to resize slice wise')
        logging.debug('image size: {}'.format(img_nda.shape))

        # resize
        masks = []
        for mask in mask_nda:
            masks.append(
                resize(mask, dim[1:], mode='constant', anti_aliasing=False, preserve_range=True, order=0, cval=0,
                       clip=True).astype(mask_nda.dtype))
        mask_nda = np.stack(masks, axis=0)

    return img_nda, mask_nda, resized_by


def center_crop_or_pad_3d(img_nda, mask_nda, dim):
    """
    center crop to given size, check if bigger or smaller
    requires square shape as input
    pad with zero if image is too small
    img nda is a ndarray or tensor with z, y, x

    :param img_nda: numpy array
    :param mask_nda: numpy array
    :param dim: 2-3
    :return:
    """
    resized_by = 'no pad'
    temp = np.zeros(dim)
    # first pad, than crop
    if img_nda.shape[2] < dim[2] or img_nda.shape[1] < dim[1] or img_nda.shape[0] < dim[0]:  # sometimes we have a volume which should be cropped along z but padded along x and y
        resized_by = 'zero pad'
        logging.debug('image size: {}'.format(img_nda.shape))
        # pad inplane
        imgs = []
        aug = PadIfNeeded(p=1.0,min_height=dim[1],min_width=dim[2],border_mode=cv2.BORDER_CONSTANT, value=0)
        for img in img_nda:
            data ={'image':img}
            res = aug(**data)
            imgs.append(res['image'])
        img_nda = np.stack(imgs, axis=0)
        # pad along z
        if img_nda.shape[0] < dim[0]:
            padding = int(np.ceil((dim[0] - img_nda.shape[0])/2))
            img_nda = np.pad(img_nda,[(padding,padding),(0,0), (0,0)], 'constant')

    if mask_nda.shape[2] < dim[2] or mask_nda.shape[1] < dim[1] or mask_nda.shape[0] < dim[0]:  # sometimes we have a volume which should be cropped along z but resized along x and y
        resized_by = 'zero pad'
        logging.debug('mask too small, need to resize slice wise')
        logging.debug('mask size: {}'.format(mask_nda.shape))

        # pad inplane
        masks = []
        aug = PadIfNeeded(p=1.0,min_height=dim[1],min_width=dim[2],border_mode=cv2.BORDER_CONSTANT, value=0)
        for mask in mask_nda:
            data = {'image': mask}
            res = aug(**data)
            masks.append(res['image'])
        mask_nda = np.stack(masks, axis=0)
        # pad along z
        if mask_nda.shape[0] < dim[0]:
            padding = int(np.ceil((dim[0] - mask_nda.shape[0])/2))
            mask_nda = np.pad(mask_nda,[(padding,padding),(0,0), (0,0)], 'constant')

    # center crop in z
    # center crop slice by slice to given size, check if bigger or smaller
    # nda is already in inplane square resolution, need to check only x
    if (img_nda.shape[2] > dim[2] or img_nda.shape[1] > dim[1] or img_nda.shape[0] > dim[0]):  # if first nda bigger than wished output, crop
        resized_by = 'center crop'
        img_nda = crop_center_3d(img_nda, dim[0], dim[1], dim[2])

    if (mask_nda.shape[2] > dim[2] or mask_nda.shape[1] > dim[1] or mask_nda.shape[0] > dim[0]):  # if second nda bigger than wished output, crop
        resized_by = 'center crop'
        mask_nda = crop_center_3d(mask_nda, dim[0], dim[1], dim[2])

    return img_nda, mask_nda, resized_by


def center_crop_or_resize_2d(img_nda, mask_nda, dim):
    """
    center crop to given size, check if bigger or smaller
    requires square shape as input
    skimage resize takes the order parameter which defines the interpolation
        0: Nearest-neighbor
        1: Bi-linear (default)
        2: Bi-quadratic
        3: Bi-cubic
        4: Bi-quartic
        5: Bi-quintic

    :param img_nda:
    :param mask_nda:
    :param dim:
    :return:
    """

    if img_nda.shape[0] > dim[0]:  # if nda is bigger than output shape, center crop, otherwise resize
        resized_by = 'center crop'
        img_nda = crop_center_2d(img_nda, dim[0], dim[1])
        mask_nda = crop_center_2d(mask_nda, dim[0], dim[1])
    else:
        resized_by = 'skimage resize'
        logging.debug('image too small, need to resize slice wise')
        # resize image
        img_nda = resize(img_nda, dim, mode='constant', preserve_range=True, anti_aliasing=True, order=3, clip=True)
        # resize mask
        mask_nda = resize(mask_nda, dim, mode='constant', anti_aliasing=False, preserve_range=True, order=0, cval=0,
                          clip=True).astype(mask_nda.dtype)

    return img_nda, mask_nda, resized_by

def center_crop_or_pad_2d(img_nda, mask_nda, dim):
    """
    pad and/or crop to given size

    :param img_nda:
    :param mask_nda:
    :param dim:
    :return:
    """
    resized_by = ''
    # first pad
    aug = PadIfNeeded(p=1.0, min_height=dim[0], min_width=dim[1], border_mode=cv2.BORDER_CONSTANT, value=0)

    if img_nda.shape[0] < dim[0] or img_nda.shape[1] < dim[1]:
        resized_by += 'zero pad'
        data = {'image': img_nda}
        res = aug(**data)
        img_nda = res['image']
    if mask_nda.shape[0] < dim[0] or mask_nda.shape[1] < dim[1]:
        resized_by += 'zero pad'
        data = {'image': mask_nda}
        res = aug(**data)
        mask_nda = res['image']

    # than crop
    if img_nda.shape[0] > dim[0] or img_nda.shape[1] > dim[1]:
        resized_by += 'center crop'
        img_nda = crop_center_2d(img_nda, dim[0], dim[1])
    if mask_nda.shape[0] > dim[0] or mask_nda.shape[1] > dim[1]:
        resized_by += 'center crop'
        mask_nda = crop_center_2d(mask_nda, dim[0], dim[1])

    return img_nda, mask_nda, resized_by


def transform_to_binary_mask(mask_nda, mask_values=[0, 1, 2, 3]):
    """
    Transform from a value-based representation to a binary channel based representation
    :param mask_nda:
    :param mask_values:
    :return:
    """
    # transform the labels to binary channel masks

    mask = np.zeros((*mask_nda.shape, len(mask_values)), dtype=np.bool)
    for ix, mask_value in enumerate(mask_values):
        mask[..., ix] = mask_nda == mask_value
    return mask


def from_channel_to_flat(binary_mask, start_c=0):

    """
    Transform a tensor or numpy nda from a channel-wise (one channel per label) representation
    to a value-based representation
    :param binary_mask:
    :return:
    """
    # convert to bool nda to allow later indexing
    binary_mask = binary_mask >= 0.5

    # reduce the shape by the channels
    temp = np.zeros(binary_mask.shape[:-1], dtype=np.uint8)

    for c in range(binary_mask.shape[-1]):
        temp[binary_mask[..., c]] = c + start_c
    return temp


def clip_quantile(img_nda, upper_quantile=.999, lower_boundary=0):
    """
    clip to values between 0 and .999 quantile
    :param img_nda:
    :param upper_quantile:
    :return:
    """


    ninenine_q = np.quantile(img_nda.flatten(), upper_quantile, overwrite_input=False)

    return np.clip(img_nda, lower_boundary, ninenine_q)


def normalise_image(img_nda, normaliser='minmax'):
    """
    Normalise Images to a given range,
    normaliser string repr for scaler, possible values: 'MinMax', 'Standard' and 'Robust'
    if no normalising method is defined use MinMax normalising
    :param img_nda:
    :param normaliser:
    :return:
    """
    # ignore case
    normaliser = normaliser.lower()

    if normaliser == 'standard':
        return (img_nda - np.mean(img_nda)) / (np.std(img_nda) + sys.float_info.epsilon)

        #return StandardScaler(copy=False, with_mean=True, with_std=True).fit_transform(img_nda)
    elif normaliser == 'robust':
        return RobustScaler(copy=False, quantile_range=(0.0, 95.0), with_centering=True,
                            with_scaling=True).fit_transform(img_nda)
    else:
        return (img_nda - img_nda.min()) / (img_nda.max() - img_nda.min() + sys.float_info.epsilon)


def pad_and_crop(ndarray, target_shape=(10, 10, 10)):
    """
    Center pad and crop a np.ndarray with any shape to a given target shape
    Parameters
    Pad and crop must be the complementary
    pad = floor(x),floor(x)+1
    crop = floor(x)+1, floor(x)
    ----------
    ndarray : numpy.ndarray of any shape
    target_shape : must have the same length as ndarray.ndim

    Returns np.ndarray with each axis either pad or crop
    -------

    """
    cropped = np.zeros(target_shape)
    target_shape = np.array(target_shape)
    logging.debug('input shape, crop_and_pad: {}'.format(ndarray.shape))
    logging.debug('target shape, crop_and_pad: {}'.format(target_shape))

    diff = ndarray.shape - target_shape

    # divide into summands to work with odd numbers
    # take the same numbers for left or right padding/cropping if the difference is dividable by 2
    # else take floor(x),floor(x)+1 for PAD (diff<0)
    # else take floor(x)+1, floor(x) for CROP (diff>0)
    d = list((int(x // 2), int(x // 2)) if x % 2 == 0 else (int(np.floor(x / 2)), int(np.floor(x / 2) + 1)) if x<0 else (int(np.floor(x / 2)+1), int(np.floor(x / 2))) for x in diff)
    # replace the second slice parameter if it is None, which slice until end of ndarray
    d = list((abs(x), abs(y)) if y != 0 else (abs(x), None) for x, y in d)
    # create a bool list, negative numbers --> pad, else --> crop
    pad_bool = diff < 0
    crop_bool = diff > 0

    # create one slice obj for cropping and one for padding
    pad = list(i if b else (None, None) for i, b in zip(d, pad_bool))
    crop = list(i if b else (None, None) for i, b in zip(d, crop_bool))

    # Create one tuple of slice calls per pad/crop
    # crop or pad from dif:-dif if second param not None, else replace by None to slice until the end
    # slice params: slice(start,end,steps)
    pad = tuple(slice(i[0], -i[1]) if i[1] != None else slice(i[0], i[1]) for i in pad)
    crop = tuple(slice(i[0], -i[1]) if i[1] != None else slice(i[0], i[1]) for i in crop)

    # crop and pad in one step
    cropped[pad] = ndarray[crop]
    return cropped

