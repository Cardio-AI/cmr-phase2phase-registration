import concurrent.futures
import logging
import os
import platform
import random
from concurrent.futures import as_completed
from random import choice
from time import time

import SimpleITK as sitk
# from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import tensorflow
from scipy.ndimage import gaussian_filter1d

from src.data.Dataset import describe_sitk, split_one_4d_sitk_in_list_of_3d_sitk, get_phases_as_onehot_gcn, \
    get_phases_as_onehot_acdc, get_n_windows_from_single4D, get_phases_as_idx_gcn, get_phases_as_idx_acdc, match_hist, \
    get_n_windows_between_phases_from_single4D, get_phases_as_idx_dmd, match_hist_any_dim
from src.data.Preprocess import resample_3D, clip_quantile, normalise_image, transform_to_binary_mask, load_masked_img, \
    augmentation_compose_2d_3d_4d, pad_and_crop, resample_t_of_4d, load_msk, calc_resampled_size, \
    align_inplane_with_ip
from src.visualization.Visualize import show_2D_or_3D


class BaseGenerator(tensorflow.keras.utils.Sequence):
    """
    Base generator class
    """

    def __init__(self, x=None, y=None, config={}):
        """
        Creates a datagenerator for a list of nrrd images and a list of nrrd masks
        :param x: list of nrrd image file names
        :param y: list of nrrd mask file names
        :param config:
        """

        logging.info('Create DataGenerator')

        if y is not None:  # return x, y
            assert (len(x) == len(y)), 'len(X) != len(Y)'

        def normalise_paths(elem):
            """
            recursive helper to clean filepaths, could handle list of lists and list of tuples
            """
            if type(elem) in [list, tuple]:
                return [normalise_paths(el) for el in elem]
            elif isinstance(elem, str):
                return os.path.normpath(elem)
            else:
                return elem

        # linux/windows cleaning
        if platform.system() == 'Linux':
            x = normalise_paths(x)
            y = normalise_paths(y)

        self.INDICES = list(range(len(x)))
        # override if necessary
        self.SINGLE_OUTPUT = config.get('SINGLE_OUTPUT', False)

        self.IMAGES = x
        self.LABELS = y

        # if streamhandler loglevel is set to debug, print each pre-processing step
        self.DEBUG_MODE = logging.getLogger().handlers[1].level == logging.DEBUG
        # self.DEBUG_MODE = False

        # read the config, set default values if param not given
        self.SCALER = config.get('SCALER', 'MinMax')
        self.AUGMENT = config.get('AUGMENT', False)
        self.AUGMENT_PROB = config.get('AUGMENT_PROB', 0.8)
        self.SHUFFLE = config.get('SHUFFLE', True)
        self.RESAMPLE = config.get('RESAMPLE', False)
        self.SPACING = config.get('SPACING', [1.25, 1.25])
        self.SEED = config.get('SEED', 42)
        self.DIM = config.get('DIM', [256, 256])
        self.BATCHSIZE = config.get('BATCHSIZE', 32)
        self.MASK_VALUES = config.get('MASK_VALUES', [0, 1, 2, 3])
        self.N_CLASSES = len(self.MASK_VALUES)
        # create one worker per image & mask (batchsize) for parallel pre-processing if nothing else is defined
        self.MAX_WORKERS = config.get('GENERATOR_WORKER', self.BATCHSIZE)
        self.MAX_WORKERS = min(32, self.MAX_WORKERS)

        if self.DEBUG_MODE:
            self.MAX_WORKERS = 1  # avoid parallelism when debugging, otherwise the plots are shuffled

        if not hasattr(self, 'X_SHAPE'):
            self.X_SHAPE = np.empty((self.BATCHSIZE, *self.DIM), dtype=np.float32)
            self.Y_SHAPE = np.empty((self.BATCHSIZE, *self.DIM, self.N_CLASSES), dtype=np.float32)

        logging.info(
            'Datagenerator created with: \n shape: {}\n spacing: {}\n batchsize: {}\n Scaler: {}\n Images: {} \n Augment: {} \n Thread workers: {}'.format(
                self.DIM,
                self.SPACING,
                self.BATCHSIZE,
                self.SCALER,
                len(
                    self.IMAGES),
                self.AUGMENT,
                self.MAX_WORKERS))

        self.on_epoch_end()

        if self.AUGMENT:
            logging.info('Data will be augmented (shift,scale and rotate) with albumentation')

        else:
            logging.info('No augmentation')

    def __plot_state_if_debug__(self, img, mask=None, start_time=None, step='raw'):

        if self.DEBUG_MODE:

            try:
                logging.debug('{}:'.format(step))
                logging.debug('{:0.3f} s'.format(time() - start_time))
                describe_sitk(img)
                describe_sitk(mask)
                if self.MASKS:
                    show_2D_or_3D(img, mask)
                    plt.show()
                else:
                    show_2D_or_3D(img)
                    plt.show()
                    # maybe this crashes sometimes, but will be caught
                    if mask:
                        show_2D_or_3D(mask)
                        plt.show()

            except Exception as e:
                logging.debug('plot image state failed: {}'.format(str(e)))

    def __len__(self):

        """
        Denotes the number of batches per epoch
        :return: number of batches
        """
        return int(np.floor(len(self.INDICES) / self.BATCHSIZE))

    def __getitem__(self, index):

        """
        Generate indexes for one batch of data
        :param index: int in the range of  {0: len(dataset)/Batchsize}
        :return: pre-processed batch
        """

        t0 = time()
        assert index < self.__len__(), 'invalid idx in batchgenerator: {} and len {}'.format(index, self.__len__())
        # collect n x indexes with n = Batchsize
        # starting from the given index parameter
        # which is in the range of  {0: len(dataset)/Batchsize}
        idxs = self.INDICES[index * self.BATCHSIZE: (index + 1) * self.BATCHSIZE]

        # Collects the value (a list of file names) for each index
        # list_IDs_temp = [self.LIST_IDS[k] for k in idxs]
        logging.debug('index generation: {}'.format(time() - t0))
        # Generate data
        return self.__data_generation__(idxs)

    def on_epoch_end(self):

        """
        Recreates and shuffle the indexes after each epoch
        :return: None
        """

        self.INDICES = np.arange(len(self.INDICES))
        if self.SHUFFLE:
            np.random.shuffle(self.INDICES)

    def __data_generation__(self, idxs):

        """
        Generates data containing batch_size samples

        :param list_IDs_temp:
        :return: X : (batchsize, *dim, n_channels), Y : (batchsize, *dim, number_of_classes)
        """

        # Initialization

        x = np.empty_like(self.X_SHAPE)
        y = np.empty_like(self.Y_SHAPE)

        futures = set()

        # spawn one thread per worker
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:

            t0 = time()
            # Generate data
            for i, ID in enumerate(idxs):

                try:
                    # keep ordering of the shuffled indexes
                    futures.add(executor.submit(self.__preprocess_one_image__, i, ID))

                except Exception as e:
                    logging.error(
                        'Exception {} in datagenerator with: image: {} or mask: {}'.format(str(e), self.IMAGES[ID],
                                                                                           self.LABELS[ID]))

        for i, future in enumerate(as_completed(futures)):
            # use the indexes i to place each processed example in the batch
            # otherwise slower images will always be at the end of the batch
            # Use the ID for exception handling as reference to the file name
            try:
                x_, y_, i, ID, needed_time = future.result()
                if self.SINGLE_OUTPUT:
                    x[i,], _ = x_, y_
                else:
                    x[i,], y[i,] = x_, y_
                logging.debug('img finished after {:0.3f} sec.'.format(needed_time))
            except Exception as e:
                logging.error(
                    'Exception {} in datagenerator with: image: {} or mask: {}'.format(str(e), self.IMAGES[ID],
                                                                                       self.LABELS[ID]))

        logging.debug('Batchsize: {} preprocessing took: {:0.3f} sec'.format(self.BATCHSIZE, time() - t0))
        if self.SINGLE_OUTPUT:
            return x.astype(np.float32), None
        else:
            return np.array(x.astype(np.float32)), np.array(y.astype(np.float32))

    def __preprocess_one_image__(self, i, ID):
        logging.error('not implemented error')


class DataGenerator(BaseGenerator):
    """
    Yields (X, Y) / image,mask for 2D and 3D U-net training
    could be used to yield (X, None)
    """

    def __init__(self, x=None, y=None, config=None):
        if config is None:
            config = {}
        self.MASKING_IMAGE = config.get('MASKING_IMAGE', False)
        self.SINGLE_OUTPUT = False
        self.MASKING_VALUES = config.get('MASKING_VALUES', [1, 2, 3])

        # how to get from image path to mask path
        # the wildcard is used to load a mask and cut the images by one or more labels
        self.REPLACE_DICT = {}
        GCN_REPLACE_WILDCARD = ('img', 'msk')
        ACDC_REPLACE_WILDCARD = ('.nii.gz', '_gt.nii.gz')

        if 'ACDC' in x[0]:
            self.REPLACE_WILDCARD = ACDC_REPLACE_WILDCARD
        else:
            self.REPLACE_WILDCARD = GCN_REPLACE_WILDCARD
        # if masks are given
        if y is not None:
            self.MASKS = True
        super().__init__(x=x, y=y, config=config)

    def __preprocess_one_image__(self, i, ID):

        t0 = time()
        if self.DEBUG_MODE:
            logging.debug(self.IMAGES[ID])
        # load image
        sitk_img = load_masked_img(sitk_img_f=self.IMAGES[ID], mask=self.MASKING_IMAGE,
                                   masking_values=self.MASKING_VALUES, replace=self.REPLACE_WILDCARD)
        # load mask
        sitk_msk = load_masked_img(sitk_img_f=self.LABELS[ID], mask=self.MASKING_IMAGE,
                                   masking_values=self.MASKING_VALUES, replace=self.REPLACE_WILDCARD,
                                   mask_labels=self.MASK_VALUES)

        self.__plot_state_if_debug__(sitk_img, sitk_msk, t0, 'raw')
        t1 = time()

        if self.RESAMPLE:

            # calc new size after resample image with given new spacing
            # sitk.spacing has the opposite order than np.shape and tf.shape
            # we use the numpy order z, y, x
            old_spacing_img = list(reversed(sitk_img.GetSpacing()))
            old_size_img = list(reversed(sitk_img.GetSize()))  # after reverse: z, y, x

            old_spacing_msk = list(reversed(sitk_msk.GetSpacing()))
            old_size_msk = list(reversed(sitk_msk.GetSize()))  # after reverse: z, y, x

            if sitk_img.GetDimension() == 2:
                y_s_img = (old_size_img[0] * old_spacing_img[0]) / self.SPACING[0]
                x_s_img = (old_size_img[1] * old_spacing_img[1]) / self.SPACING[1]
                new_size_img = (
                    int(np.round(x_s_img)), int(np.round(y_s_img)))  # this will be used for resampling, invert again

                y_s_msk = (old_size_msk[0] * old_spacing_msk[0]) / self.SPACING[0]
                x_s_msk = (old_size_msk[1] * old_spacing_msk[1]) / self.SPACING[1]
                new_size_msk = (
                    int(np.round(x_s_msk)), int(np.round(y_s_msk)))  # this will be used for resampling, invert again

            elif sitk_img.GetDimension() == 3:
                # round up
                z_s_img = np.round((old_size_img[0] * old_spacing_img[0])) / self.SPACING[0]
                # z_s_img = max(self.DIM[0],z_s_img)  # z must fit in the network input, resample with spacing or min network input
                y_s_img = np.round((old_size_img[1] * old_spacing_img[1])) / self.SPACING[1]
                x_s_img = np.round((old_size_img[2] * old_spacing_img[2])) / self.SPACING[2]
                new_size_img = (int(np.round(x_s_img)), int(np.round(y_s_img)), int(np.round(z_s_img)))

                z_s_msk = np.round((old_size_msk[0] * old_spacing_msk[0])) / self.SPACING[0]
                # z_s_msk = max(self.DIM[0],z_s_msk)  # z must fit in the network input, resample with spacing or min network input
                y_s_msk = np.round((old_size_msk[1] * old_spacing_msk[1])) / self.SPACING[1]
                x_s_msk = np.round((old_size_msk[2] * old_spacing_msk[2])) / self.SPACING[2]
                new_size_msk = (int(np.round(x_s_msk)), int(np.round(y_s_msk)), int(np.round(z_s_msk)))

                # we can also resize with the resamplefilter from sitk
                # this cuts the image on the bottom and right
                # new_size = self.DIM
            else:
                raise ('dimension not supported: {}'.format(sitk_img.GetDimension()))

            logging.debug('dimension: {}'.format(sitk_img.GetDimension()))
            logging.debug('Size before resample: {}'.format(sitk_img.GetSize()))

            # resample the image to given spacing and size
            sitk_img = resample_3D(sitk_img=sitk_img, size=new_size_img, spacing=list(reversed(self.SPACING)),
                                   interpolate=sitk.sitkLinear)
            if self.MASKS:  # if y is a mask, interpolate with nearest neighbor
                sitk_msk = resample_3D(sitk_img=sitk_msk, size=new_size_msk, spacing=list(reversed(self.SPACING)),
                                       interpolate=sitk.sitkNearestNeighbor)
            else:
                sitk_msk = resample_3D(sitk_img=sitk_msk, size=new_size_msk, spacing=list(reversed(self.SPACING)),
                                       interpolate=sitk.sitkLinear)

        elif sitk_img.GetDimension() == 3:  # 3d data needs to be resampled at least in z direction
            logging.debug(('resample in z direction'))
            logging.debug('Size before resample: {}'.format(sitk_img.GetSize()))

            size_img = sitk_img.GetSize()
            spacing_img = sitk_img.GetSpacing()

            size_msk = sitk_msk.GetSize()
            spacing_msk = sitk_msk.GetSpacing()
            logging.debug('spacing before resample: {}'.format(sitk_img.GetSpacing()))

            # keep x and y size/spacing, just extend the size in z, keep spacing of z --> pad with zero along
            new_size_img = (
                *size_img[:-1], self.DIM[0])  # take x and y from the current sitk, extend by z creates x,y,z
            new_spacing_img = (*spacing_img[:-1], self.SPACING[0])  # spacing is in opposite order

            new_size_msk = (
                *size_msk[:-1], self.DIM[0])  # take x and y from the current sitk, extend by z creates x,y,z
            new_spacing_msk = (*spacing_msk[:-1], self.SPACING[0])  # spacing is in opposite order

            sitk_img = resample_3D(sitk_img=sitk_img, size=(new_size_img), spacing=new_spacing_img,
                                   interpolate=sitk.sitkLinear)
            if self.MASKS:
                sitk_msk = resample_3D(sitk_img=sitk_msk, size=(new_size_msk), spacing=new_spacing_msk,
                                       interpolate=sitk.sitkNearestNeighbor)
            else:
                sitk_msk = resample_3D(sitk_img=sitk_msk, size=(new_size_msk), spacing=new_spacing_msk,
                                       interpolate=sitk.sitkLinear)

        logging.debug('Spacing after resample: {}'.format(sitk_img.GetSpacing()))
        logging.debug('Size after resample: {}'.format(sitk_img.GetSize()))

        # transform to nda for further processing
        img_nda = sitk.GetArrayFromImage(sitk_img)
        mask_nda = sitk.GetArrayFromImage(sitk_msk)

        self.__plot_state_if_debug__(img_nda, mask_nda, t1, 'resampled')
        t1 = time()

        # We need to normalise the image/before augmentation, albumentation expects them to be normalised
        img_nda = clip_quantile(img_nda, .999)
        img_nda = normalise_image(img_nda, normaliser=self.SCALER)
        # img_nda = normalise_image(img_nda, normaliser=self.SCALER)

        if not self.MASKS:  # yields the image two times for an autoencoder
            mask_nda = clip_quantile(mask_nda, .999)
            mask_nda = normalise_image(mask_nda, normaliser=self.SCALER)
            # mask_nda = normalise_image(mask_nda, normaliser=self.SCALER)

        self.__plot_state_if_debug__(img_nda, mask_nda, t1, '{} normalized image:'.format(self.SCALER))

        if self.AUGMENT:  # augment data with albumentation
            # use albumentation to apply random rotation scaling and shifts
            img_nda, mask_nda = augmentation_compose_2d_3d_4d(img_nda, mask_nda, probabillity=0.8)

            self.__plot_state_if_debug__(img_nda, mask_nda, t1, 'augmented')
            t1 = time()

        img_nda, mask_nda = map(lambda x: pad_and_crop(x, target_shape=self.DIM),
                                [img_nda, mask_nda])

        img_nda = normalise_image(img_nda, normaliser=self.SCALER)

        # transform the labels to binary channel masks
        # if masks are given, otherwise keep image as it is (for vae models, masks == False)
        if self.MASKS:
            mask_nda = transform_to_binary_mask(mask_nda, self.MASK_VALUES)
        else:  # yields two images
            mask_nda = normalise_image(mask_nda, normaliser=self.SCALER)
            mask_nda = mask_nda[..., np.newaxis]

        self.__plot_state_if_debug__(img_nda, mask_nda, t1, 'after crop')

        return img_nda[..., np.newaxis], mask_nda, i, ID, time() - t0

'''
class MotionDataGenerator(DataGenerator):
    """
    yields n input volumes and n output volumes
    """

    def __init__(self, x=None, y=None, config=None):

        if config is None:
            config = {}
        super().__init__(x=x, y=y, config=config)

        if type(x[0]) in [tuple, list]:
            # if this is the case we have a sequence of 3D volumes or a sequence of 2D images
            self.INPUT_VOLUMES = len(x[0])
            self.OUTPUT_VOLUMES = len(y[0])
            self.X_SHAPE = np.empty((self.BATCHSIZE, self.INPUT_VOLUMES, *self.DIM, 1), dtype=np.float32)
            self.Y_SHAPE = np.empty((self.BATCHSIZE, self.OUTPUT_VOLUMES, *self.DIM, 1), dtype=np.float32)

        self.MASKS = None  # need to check if this is still necessary!

        # define a random seed for albumentations
        random.seed(config.get('SEED', 42))

    def __data_generation__(self, idxs):

        """
        Loads and pre-process one entity of x and y


        :param idxs:
        :return: X : (batchsize, *dim, n_channels), Y : (batchsize, *dim, number_of_classes)
        """

        # Initialization
        x = np.empty_like(self.X_SHAPE)  # model input
        y = np.empty_like(self.Y_SHAPE)  # model output

        futures = set()

        # spawn one thread per worker
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:

            t0 = time()
            ID = ''
            # Generate data
            for i, ID in enumerate(idxs):

                try:
                    # remember the ordering of the shuffled indexes,
                    # otherwise files, that take longer are always at the batch end
                    futures.add(executor.submit(self.__preprocess_one_image__, i, ID))

                except Exception as e:
                    logging.error(
                        'Exception {} in datagenerator with: image: {} or mask: {}'.format(str(e), self.IMAGES[ID],
                                                                                           self.LABELS[ID]))

        for i, future in enumerate(as_completed(futures)):
            # use the indexes to order the batch
            # otherwise slower images will always be at the end of the batch
            try:
                x_, y_, i, ID, needed_time = future.result()
                x[i,], y[i,] = x_, y_
                logging.debug('img finished after {:0.3f} sec.'.format(needed_time))
            except Exception as e:
                # write these files into a dedicated error log
                PrintException()
                print(e)
                logging.error(
                    'Exception {} in datagenerator with:\n'
                    'image:\n'
                    '{}\n'
                    'mask:\n'
                    '{}'.format(str(e), self.IMAGES[ID], self.LABELS[ID]))

        logging.debug('Batchsize: {} preprocessing took: {:0.3f} sec'.format(self.BATCHSIZE, time() - t0))

        zeros = np.zeros((*x.shape[:-1], 3), dtype=np.float32)

        return tuple([[x, zeros], [y, zeros]])

    def __preprocess_one_image__(self, i, ID):

        t0 = time()

        x = self.IMAGES[ID]
        y = self.LABELS[ID]

        if not isinstance(x, list):
            x = [x]
        if not isinstance(y, list):
            y = [y]

        # use the load_masked_img wrapper to enable masking of the images, not necessary for the TMI paper
        # load image
        model_inputs = list(map(lambda x: load_masked_img(sitk_img_f=x, mask=self.MASKING_IMAGE,
                                                          masking_values=self.MASKING_VALUES,
                                                          replace=self.REPLACE_WILDCARD), x))

        model_outputs = list(map(lambda x: load_masked_img(sitk_img_f=x, mask=self.MASKING_IMAGE,
                                                           masking_values=self.MASKING_VALUES,
                                                           replace=self.REPLACE_WILDCARD), y))

        # test to train on ax,sax image pairs without ax2sax transformation

        self.__plot_state_if_debug__(model_inputs[0], model_outputs[0], t0, 'raw')
        t1 = time()

        if self.RESAMPLE:
            if model_inputs[0].GetDimension() in [2, 3]:

                # calc new size after resample image with given new spacing
                # sitk.spacing has the opposite order than np.shape and tf.shape
                # In the config we use the numpy order z, y, x which needs to be reversed for sitk
                def calc_resampled_size(sitk_img, target_spacing):
                    if type(target_spacing) in [list, tuple]:
                        target_spacing = np.array(target_spacing)
                    old_size = np.array(sitk_img.GetSize())
                    old_spacing = np.array(sitk_img.GetSpacing())
                    logging.debug('old size: {}, old spacing: {}, target spacing: {}'.format(old_size, old_spacing,
                                                                                             target_spacing))
                    new_size = (old_size * old_spacing) / target_spacing
                    return list(np.around(new_size).astype(np.int))

                # transform the spacing from numpy representation towards the sitk representation
                target_spacing = list(reversed(self.SPACING))
                new_size_inputs = list(map(lambda elem: calc_resampled_size(elem, target_spacing), model_inputs))
                new_size_outputs = list(map(lambda elem: calc_resampled_size(elem, target_spacing), model_outputs))

            else:
                raise NotImplementedError('dimension not supported: {}'.format(model_inputs[0].GetDimension()))

            logging.debug('dimension: {}'.format(model_inputs[0].GetDimension()))
            logging.debug('Size before resample: {}'.format(model_inputs[0].GetSize()))

            model_inputs = list(map(lambda x:
                                    resample_3D(sitk_img=x[0],
                                                size=x[1],
                                                spacing=target_spacing,
                                                interpolate=sitk.sitkLinear),
                                    zip(model_inputs, new_size_inputs)))

            model_outputs = list(map(lambda x:
                                     resample_3D(sitk_img=x[0],
                                                 size=x[1],
                                                 spacing=target_spacing,
                                                 interpolate=sitk.sitkLinear),
                                     zip(model_outputs, new_size_outputs)))

        logging.debug('Spacing after resample: {}'.format(model_inputs[0].GetSpacing()))
        logging.debug('Size after resample: {}'.format(model_inputs[0].GetSize()))

        # transform to nda for further processing
        model_inputs = list(map(lambda x: sitk.GetArrayFromImage(x), model_inputs))
        model_outputs = list(map(lambda x: sitk.GetArrayFromImage(x), model_outputs))

        self.__plot_state_if_debug__(model_inputs[0], model_outputs[0], t1, 'resampled')

        if self.AUGMENT:  # augment data with albumentation
            # use albumentation to apply random rotation scaling and shifts

            # we need to make sure to apply the same augmentation on the input and target data
            combined = np.stack(model_inputs + model_outputs, axis=0)
            combined = augmentation_compose_2d_3d_4d(img=combined, mask=None, probabillity=self.AUGMENT_PROB)
            model_inputs, model_outputs = np.split(combined, indices_or_sections=2, axis=0)

            self.__plot_state_if_debug__(img=model_inputs[0], mask=model_outputs[0], start_time=t1, step='augmented')
            t1 = time()

        # clip, pad/crop and normalise & extend last axis
        model_inputs = map(lambda x: clip_quantile(x, .9999), model_inputs)
        model_inputs = list(map(lambda x: pad_and_crop(x, target_shape=self.DIM), model_inputs))
        # model_inputs = list(map(lambda x: normalise_image(x, normaliser=self.SCALER), model_inputs)) # normalise per volume
        model_inputs = normalise_image(np.stack(model_inputs), normaliser=self.SCALER)  # normalise per 4D

        model_outputs = map(lambda x: clip_quantile(x, .9999), model_outputs)
        model_outputs = list(map(lambda x: pad_and_crop(x, target_shape=self.DIM), model_outputs))
        # model_outputs = list(map(lambda x: normalise_image(x, normaliser=self.SCALER), model_outputs)) # normalise per volume
        model_outputs = normalise_image(np.stack(model_outputs), normaliser=self.SCALER)  # normalise per 4D
        self.__plot_state_if_debug__(model_inputs[0], model_outputs[0], t1, 'clipped cropped and pad')

        return model_inputs[..., np.newaxis], model_outputs[..., np.newaxis], i, ID, time() - t0
'''
'''
class PhaseWindowGenerator(DataGenerator):
    """
    yields n input volumes and n output volumes
    """

    def __init__(self, x=None, y=None, config=None, yield_masks=False, in_memory=False):

        if config is None:
            config = {}
        super().__init__(x=x, y=y, config=config)

        self.config = config
        self.T_SPACING = config.get('T_SPACING', 10)
        self.PHASES = config.get('PHASES', 5)
        self.HIST_MATCHING = config.get('HIST_MATCHING', False)
        self.IMG_INTERPOLATION = config.get('IMG_INTERPOLATION', sitk.sitkLinear)
        self.MSK_INTERPOLATION = config.get('MSK_INTERPOLATION', sitk.sitkNearestNeighbor)
        self.AUGMENT_TEMP = config.get('AUGMENT_TEMP', False)
        self.AUGMENT_TEMP_RANGE = config.get('AUGMENT_TEMP_RANGE', (-2, 2))
        self.RESAMPLE_T = config.get('RESAMPLE_T', False)
        self.RESAMPLE_Z = config.get('RESAMPLE_Z', True)
        self.WINDOW_SIZE = config.get('WINDOW_SIZE', 1)
        self.IMG_CHANNELS = config.get('IMG_CHANNELS', 1)
        self.INPUT_T_ELEM = config.get('INPUT_T_ELEM', 0)
        self.REPLACE_WILDCARD = ('clean', 'mask')
        self.BETWEEN_PHASES = config.get('BETWEEN_PHASES', False)
        self.yield_masks = yield_masks
        self.TARGET_CHANNELS = config.get('TARGET_CHANNELS', 1)
        self.IN_MEMORY = in_memory
        self.REGISTER_BACKWARDS = config.get('REGISTER_BACKWARDS', False)

        """if self.yield_masks: # this is just for the case that we want to yield masks with the same pre-processing as applied to the images
            self.IMG_CHANNELS = 1
            self.MASKING_IMAGE = False # we cant mask the masks, turn it off, to make sure
            self.IMG_INTERPOLATION = sitk.sitkNearestNeighbor
            self.TARGET_CHANNELS = 1"""
        self.X_SHAPE = np.empty((self.BATCHSIZE, self.PHASES, *self.DIM, self.IMG_CHANNELS), dtype=np.float32)
        self.Y_SHAPE = np.empty((self.BATCHSIZE, self.PHASES, *self.DIM, self.TARGET_CHANNELS), dtype=np.float32)

        # this is a hack to figure out which dataset we use, without introducing a new config parameter
        self.ISACDC = False
        self.ISDMD = False
        if config.get('ISDMDDATA', False):
            self.ISDMD = True
        elif 'acdc' in self.IMAGES[0].lower():
            self.ISACDC = True

        # opens a dataframe with cleaned phases per patient
        if not self.ISACDC:
            self.METADATA_FILE = config.get('DF_META',
                                            '/mnt/ssd/data/gcn/02_imported_4D_unfiltered/SAx_3D_dicomTags_phase.csv')
            df = pd.read_csv(self.METADATA_FILE)
            self.DF_METADATA = df[['patient', 'ED#', 'MS#', 'ES#', 'PF#', 'MD#']]
        # TODO: need to check if this is still necessary!
        self.MASKS = None

        # in memory training for the cluster
        if self.IN_MEMORY:
            self.IMAGES_SITK = [load_masked_img(sitk_img_f=x, mask=self.MASKING_IMAGE,
                                                masking_values=self.MASKING_VALUES, replace=self.REPLACE_WILDCARD,
                                                maskAll=False) for x in self.IMAGES]

        # define a random seed for albumentations
        random.seed(config.get('SEED', 42))
        logging.info('params of generator:')  # print the parameters of this generator
        logging.info(list((k, v) for k, v in vars(self).items() if
                          type(v) in [int, str, list, bool] and str(k) not in ['IMAGES', 'LABELS']))

    def on_batch_end(self):
        """
        Use this callback for methods that should be executed after each new batch generation
        """
        pass

    def __data_generation__(self, list_IDs_temp):

        """
        Loads and pre-process one batch

        :param list_IDs_temp:
        :return: X : (batchsize, *dim, n_channels), Y : (batchsize, self.T_SHAPE, number_of_classes)
        """
        # use this for batch wise histogram-reference selection
        self.on_batch_end()

        # Initialization
        x = np.empty_like(self.X_SHAPE)  # model input
        y = np.empty_like(self.Y_SHAPE)  # model output

        futures = set()

        # spawn one thread per worker
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            t0 = time()
            ID = ''
            # Generate data
            for i, ID in enumerate(list_IDs_temp):

                try:
                    # remember the ordering of the shuffled indexes,
                    # otherwise files, that take longer are always at the batch end
                    futures.add(executor.submit(self.__preprocess_one_image__, i, ID))

                except Exception as e:
                    PrintException()
                    print(e)
                    logging.error(
                        'Exception {} in datagenerator with: image: {} or mask: {}'.format(str(e), self.IMAGES[ID],
                                                                                           self.LABELS[ID]))

        for i, future in enumerate(as_completed(futures)):
            # use the indexes to order the batch
            # otherwise slower images will always be at the end of the batch
            try:
                x_, y_, i, ID, needed_time = future.result()
                x[i,], y[i,] = x_, y_
                logging.debug('img finished after {:0.3f} sec.'.format(needed_time))
            except Exception as e:
                # write these files into a dedicated error log
                PrintException()
                print(e)
                logging.error(
                    'Exception {} in datagenerator with:\n'
                    'image:\n'
                    '{}\n'
                    'mask:\n'
                    '{}'.format(str(e), self.IMAGES[ID], self.LABELS[ID]))

        logging.debug('Batchsize: {} preprocessing took: {:0.3f} sec'.format(self.BATCHSIZE, time() - t0))
        zeros = np.zeros((*x.shape[:-1], 3), dtype=np.float32)
        return tuple([[x, zeros], [y, zeros]])

    def __preprocess_one_image__(self, i, ID):

        # --------------- HIST MATCHING REFERENCE VOL--------------
        ref = None
        apply_hist_matching = False
        if self.HIST_MATCHING and random.random() <= self.AUGMENT_PROB:
            apply_hist_matching = True
            ignore_z = 1
            # use a random image, given to this generator, as histogram template for histogram matching augmentation
            ref = sitk.GetArrayFromImage(sitk.ReadImage((choice(self.IMAGES))))
            ref = ref[choice(list(range(ref.shape[0] - 1))), choice(list(range(ref.shape[1] - 1))[ignore_z:-ignore_z])]
        t0 = time()
        t1 = time()

        x = self.IMAGES[ID]

        # --------------- LOAD THE MODEL INPUT--------------
        if self.IN_MEMORY:
            model_inputs = self.IMAGES_SITK[ID]
        else:
            # use the load_masked_img wrapper to enable masking of the images, currently not necessary, but nice to have
            model_inputs = load_masked_img(sitk_img_f=x, mask=self.MASKING_IMAGE,
                                           masking_values=self.MASKING_VALUES, replace=self.REPLACE_WILDCARD,
                                           maskAll=False)
        model_m_inputs = load_msk(f_name=x,
                                  valid_labels=[2])
        logging.debug('load and masking took: {:0.3f} s'.format(time() - t1))
        t1 = time()

        # --------------- TEMPORAL RESAMPLING AND Temp-AUGMENTATION--------------
        # resample the temporal resolution
        # if AUGMENT_TEMP --> add an temporal augmentation factor within the range given by: AUGMENT_TEMP_RANGE
        t_spacing = self.T_SPACING
        if self.AUGMENT_TEMP: t_spacing = t_spacing + random.randint(self.AUGMENT_TEMP_RANGE[0],
                                                                     self.AUGMENT_TEMP_RANGE[1])
        logging.debug('t-spacing: {}'.format(t_spacing))
        if self.RESAMPLE_T:
            temporal_sampling_factor = model_inputs.GetSpacing()[-1] / t_spacing
            model_inputs = resample_t_of_4d(model_inputs, t_spacing=t_spacing, interpolation=self.IMG_INTERPOLATION,
                                            ismask=False)
        else:
            temporal_sampling_factor = 1  # dont scale the indices if we dont resample T
        logging.debug('temp resampling took: {:0.3f} s'.format(time() - t1))
        t1 = time()

        # --------------- SPLIT IN 3D SITK IMAGES-------------
        # Create a list of 3D volumes for volume resampling
        model_inputs = split_one_4d_sitk_in_list_of_3d_sitk(model_inputs, axis=0, prob=self.AUGMENT_PROB)
        model_m_inputs = split_one_4d_sitk_in_list_of_3d_sitk(model_m_inputs, axis=0, prob=self.AUGMENT_PROB)
        logging.debug('split in t x 3D took: {:0.3f} s'.format(time() - t1))
        t1 = time()

        # --------------- LOAD INDICES FOR CARDIAC PHASES--------------
        # Returns the indices in the following order: 'ED#', 'MS#', 'ES#', 'PF#', 'MD#'
        if self.ISACDC:
            idx = get_phases_as_idx_acdc(x, temporal_sampling_factor, len(model_inputs))
        elif self.ISDMD:
            idx = get_phases_as_idx_dmd(x, self.DF_METADATA, temporal_sampling_factor, len(model_inputs))
        else:
            idx = get_phases_as_idx_gcn(x, self.DF_METADATA, temporal_sampling_factor, len(model_inputs))
        logging.debug('index loading took: {:0.3f} s'.format(time() - t1))
        # logging.debug('transposed: \n{}'.format(onehot))
        self.__plot_state_if_debug__(img=model_inputs[len(model_inputs) // 2], start_time=t0, step='raw')
        t1 = time()

        # --------------- SPATIAL RESAMPLING-------------
        if self.RESAMPLE:
            target_spacing = list(reversed(self.SPACING))
            input_z_spacing = model_inputs[0].GetSpacing()[-1]
            if not self.RESAMPLE_Z: # keep z
                target_spacing = target_spacing[:-1] + [input_z_spacing]
            if model_inputs[0].GetDimension() in [2, 3]:

                # calculate the new size of each 3D volume (after resample with the given spacing)
                # sitk.spacing has the opposite order than np.shape and tf.shape
                # In the config we use the numpy order z, y, x which needs to be reversed for sitk
                def calc_resampled_size(sitk_img, target_spacing):
                    if type(target_spacing) in [list, tuple]:
                        target_spacing = np.array(target_spacing)

                    old_size = np.array(sitk_img.GetSize())
                    old_spacing = np.array(sitk_img.GetSpacing())
                    logging.debug('old size: {}, old spacing: {}, target spacing: {}'.format(old_size, old_spacing,
                                                                                             target_spacing))
                    new_size = (old_size * old_spacing) / target_spacing
                    #if not resample_z: new_size =
                    return list(np.around(new_size).astype(np.int))

                # transform the spacing from numpy representation towards the sitk representation
                new_size_inputs = list(map(lambda elem: calc_resampled_size(elem, target_spacing), model_inputs))

            else:
                raise NotImplementedError('dimension not supported: {}'.format(model_inputs[0].GetDimension()))

            logging.debug('dimension: {}'.format(model_inputs[0].GetDimension()))
            logging.debug('Size before resample: {}'.format(model_inputs[0].GetSize()))

            model_inputs = list(map(lambda x:
                                    resample_3D(sitk_img=x[0],
                                                size=x[1],
                                                spacing=target_spacing,
                                                interpolate=self.IMG_INTERPOLATION),  # sitk.sitkLinear
                                    zip(model_inputs, new_size_inputs)))

            # CHANGED
            model_m_inputs = list(map(lambda x:
                                      resample_3D(sitk_img=x[0],
                                                  size=x[1],
                                                  spacing=target_spacing,
                                                  interpolate=self.MSK_INTERPOLATION),  # sitk.nearest
                                      zip(model_m_inputs, new_size_inputs)))

        logging.debug('Spacing after resample: {}'.format(model_inputs[0].GetSpacing()))
        logging.debug('Size after resample: {}'.format(model_inputs[0].GetSize()))
        logging.debug('spatial resampling took: {:0.3f} s'.format(time() - t1))
        t1 = time()

        # --------------- CONTINUE WITH ND-ARRAYS --------------
        # transform to nda for further processing
        model_inputs = np.stack(list(map(lambda x: sitk.GetArrayFromImage(x), model_inputs)), axis=0)
        model_m_inputs = np.stack(list(map(lambda x: sitk.GetArrayViewFromImage(x), model_m_inputs)), axis=0)

        logging.debug('transform to nda took: {:0.3f} s'.format(time() - t1))
        t1 = time()
        self.__plot_state_if_debug__(img=model_inputs[len(model_inputs) // 2], start_time=t1, step='resampled')

        # --------------- HIST MATCHING--------------
        if apply_hist_matching:
            model_inputs = match_hist(model_inputs, ref)
            logging.debug('hist matching took: {:0.3f} s'.format(time() - t1))
            t1 = time()

        # --------------- SLICE PAIRS OF INPUT AND TARGET VOLUMES ACCORDING TO CARDIAC PHASE IDX -------------
        # get the volumes of each phase window
        # combined --> t-w, t, t+w, We can use this window in different combinations as input and target

        model_inputs, model_m_inputs = align_inplane_with_ip(model_inputs=model_inputs,
                                                             msk_file_name=model_m_inputs,
                                                             roll2septum=False,
                                                             roll2lvbood=True,
                                                             rotate=False,
                                                             translate=True)

        if self.BETWEEN_PHASES:
            combined = get_n_windows_between_phases_from_single4D(model_inputs, idx,
                                                                  register_backwards=self.REGISTER_BACKWARDS)
        else:
            #combined = get_n_windows_from_single4D(model_inputs, idx, window_size=self.WINDOW_SIZE)
            # for the mask all volume
            combined = get_n_windows_between_phases_from_single4D(model_inputs, idx,
                                                                  register_backwards=self.REGISTER_BACKWARDS)

        logging.debug('windowing slicing took: {:0.3f} s'.format(time() - t1))
        t1 = time()
        combined = np.stack(combined, axis=-1)


        # --------------- Image Augmentation, this is done in 2D -------------
        if self.AUGMENT and random.random() <= self.AUGMENT_PROB:
            # use albumentation to apply random rotation scaling and shifts
            # we need to make sure to apply the same augmentation on the input and target data
            # Albumentation uses the Interpolation enum from opencv which is different to the SimpleITK enum
            combined = np.concatenate(combined, axis=0)
            logging.debug('shape combined: {}'.format(combined.shape))
            combined = augmentation_compose_2d_3d_4d(img=combined, mask=None, config=self.config)
            logging.debug('shape combined: {}'.format(combined.shape))
            # split into input and target
            combined = np.split(combined, indices_or_sections=3, axis=0)
            self.__plot_state_if_debug__(img=combined[self.INPUT_T_ELEM][0], start_time=t1, step='augmented')
            logging.debug('augmentation took: {:0.3f} s'.format(time() - t1))
            t1 = time()
        if self.IMG_CHANNELS == 1:
            model_inputs = combined[...,0:1]
            model_targets = combined[...,-1:]

        elif self.IMG_CHANNELS > 1:
            model_targets = combined[...,-1:]

            # model_inputs = transform_to_binary_mask(model_inputs, self.MASK_VALUES)
            # model_targets = transform_to_binary_mask(model_targets, self.MASK_VALUES)

        model_inputs = pad_and_crop(model_inputs, target_shape=(self.PHASES, *self.DIM, self.IMG_CHANNELS))
        model_targets = pad_and_crop(model_targets, target_shape=(self.PHASES, *self.DIM, self.TARGET_CHANNELS))
        logging.debug('pad/crop took: {:0.3f} s'.format(time() - t1))
        t1 = time()

        # clip, pad/crop and normalise & extend last axis
        # We repeat/tile the 3D volume at this time, to avoid resampling/augmenting the same slices multiple times
        # Ideally this saves computation time and memory

        if not self.yield_masks:
            model_inputs = clip_quantile(model_inputs, .9999)
            model_targets = clip_quantile(model_targets, .9999)
            logging.debug('quantile clipping took: {:0.3f} s'.format(time() - t1))
            t1 = time()

            model_inputs = normalise_image(model_inputs, normaliser=self.SCALER)  # normalise per 4D
            model_targets = normalise_image(model_targets, normaliser=self.SCALER)  # normalise per 4D
            logging.debug('normalisation took: {:0.3f} s'.format(time() - t1))
            t1 = time()

            self.__plot_state_if_debug__(img=model_inputs[len(model_inputs) // 2], start_time=t1,
                                         step='clipped cropped and pad')

        assert not np.any(np.isnan(model_inputs))
        assert not np.any(np.isnan(model_targets))

        return model_inputs, model_targets, i, ID, time() - t0
'''

class PhaseMaskWindowGenerator(DataGenerator):
    """
    yields n input volumes and n output volumes
    """

    def __init__(self, x=None, y=None, config=None, yield_masks=False, in_memory=False):

        if config is None:
            config = {}
        super().__init__(x=x, y=y, config=config)

        self.config = config
        self.T_SPACING = config.get('T_SPACING', 10)
        self.PHASES = config.get('PHASES', 5)
        self.HIST_MATCHING = config.get('HIST_MATCHING', False)
        self.REF_IMAGE = {}
        self.IMG_INTERPOLATION = config.get('IMG_INTERPOLATION', sitk.sitkLinear)
        self.MSK_INTERPOLATION = config.get('MSK_INTERPOLATION', sitk.sitkNearestNeighbor)
        self.AUGMENT_TEMP = config.get('AUGMENT_TEMP', False)
        self.AUGMENT_TEMP_RANGE = config.get('AUGMENT_TEMP_RANGE', (-2, 2))
        self.RESAMPLE_T = config.get('RESAMPLE_T', False)
        self.RESAMPLE_Z = config.get('RESAMPLE_Z', True)
        self.WINDOW_SIZE = config.get('WINDOW_SIZE', 1)
        self.IMG_CHANNELS = config.get('IMG_CHANNELS', 1)
        self.INPUT_T_ELEM = config.get('INPUT_T_ELEM', 0)
        self.REPLACE_WILDCARD = ('clean', 'mask')
        self.BETWEEN_PHASES = config.get('BETWEEN_PHASES', False)
        self.yield_masks = yield_masks
        self.TARGET_CHANNELS = 1
        self.IN_MEMORY = in_memory
        self.THREAD_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=12)
        self.ISTRAINING = config.get('ISTRAINING', True)  # true == sparse myo mask for displacement
        self.COMPOSE_CONSISTENCY = config.get('COMPOSE_CONSISTENCY', False)
        self.REGISTER_BACKWARDS = config.get('REGISTER_BACKWARDS', False)

        self.X_SHAPE = np.empty((self.BATCHSIZE, self.PHASES, *self.DIM, self.IMG_CHANNELS), dtype=np.float32)
        self.X2_SHAPE = np.empty((self.BATCHSIZE, self.PHASES, *self.DIM, self.IMG_CHANNELS), dtype=np.float32)
        self.Y_SHAPE = np.empty((self.BATCHSIZE, self.PHASES, *self.DIM, self.TARGET_CHANNELS), dtype=np.float32)

        # this is a hack to figure out which dataset we use, without introducing a new config parameter
        self.ISACDC = False
        self.ISDMD = False
        if config.get('ISDMDDATA', False):
            self.ISDMD = True
        elif 'acdc' in self.IMAGES[0].lower():
            self.ISACDC = True

        # opens a dataframe with cleaned phases per patient
        if not self.ISACDC:
            self.METADATA_FILE = config.get('DF_META',
                                            None)
            df = pd.read_csv(self.METADATA_FILE)
            df.columns = df.columns.str.lower()
            df['patient'] = df['patient'].str.lower()
            self.DF_METADATA = df[['patient', 'ed#', 'ms#', 'es#', 'pf#', 'md#']]
        # TODO: need to check if this is still necessary!
        self.MASKS = None

        # in memory training for the cluster
        if self.IN_MEMORY:
            zipped = [self.__pre_load_one_image__(i, i) for i in range(len(self.IMAGES))]
            self.IMAGES_SITK, self.MASKS_SITK, _, self.INDICIES = list(map(list, zip(*zipped)))

        # define a random seed for albumentations
        random.seed(config.get('SEED', 42))
        logging.info(
            'params of generator:')  # print the parameters of this generator, exclude in memory sitk files of file names
        logging.info(list((k, v) for k, v in vars(self).items() if
                          type(v) in [int, str, list, bool] and str(k) not in ['IMAGES', 'LABELS', 'IMAGES_SITK',
                                                                               'MASKS_SITK']))

    def on_batch_end(self):
        """
        Use this callback for methods that should be executed after each new batch generation
        """
        pass

    def __data_generation__(self, list_IDs_temp):

        """
        Loads and pre-process one batch

        :param list_IDs_temp:
        :return: X : (batchsize, *dim, n_channels), Y : (batchsize, self.T_SHAPE, number_of_classes)
        """
        # use this for batch wise histogram-reference selection
        # self.on_batch_end()

        # Initialization
        x = np.empty_like(self.X_SHAPE)  # CMR model input
        x2 = np.empty_like(self.X_SHAPE)  # CMR mask model input
        y = np.empty_like(self.Y_SHAPE)  # model output
        y2 = np.empty_like(self.Y_SHAPE)  # model output
        # model returns:
        # comp moved CMR moved, moved CMR, moved msk, flow, flow_comp

        futures = set()

        # spawn one thread per worker
        # with self.THREAD_POOL as executor:
        t0 = time()
        ID = ''
        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            try:
                # remember the ordering of the shuffled indexes,
                # otherwise files, that take longer are always at the batch end
                futures.add(self.THREAD_POOL.submit(self.__preprocess_one_image__, i, ID))

            except Exception as e:
                PrintException()
                print(e)
                logging.error(
                    'Exception {} in datagenerator with: image: {} or mask: {}'.format(str(e), self.IMAGES[ID],
                                                                                       self.LABELS[ID]))

        for i, future in enumerate(as_completed(futures)):
            # use the indexes to order the batch
            # otherwise slower images will always be at the end of the batch
            # for backwards, shifted is earlier in time (shift = k-1 or k-w)
            try:
                x_k, s_k, x_k_shifted, s_k_shifted, i, ID, needed_time = future.result()
                # print(x_.shape, x2_.shape, y_.shape,y2_.shape)
                x[i,], y[i,] = x_k, x_k_shifted # shifted to an earlier frame
                x2[i,], y2[i,] =s_k,  s_k_shifted # shifted to an earlier frame
                logging.debug('img finished after {:0.3f} sec.'.format(needed_time))
                #pass
            except Exception as e:
                # write these files into a dedicated error log
                PrintException()
                print(e)
                logging.error(
                    'Exception {} in datagenerator with:\n'
                    'image:\n'
                    '{}\n'
                    'mask:\n'
                    '{}'.format(str(e), self.IMAGES[ID], self.LABELS[ID]))
        # TODO: move the stacking part to self.__preprocess_one_image___()
        # data_generation should only contain the multi-threading logic
        # x = x_k
        # x2 = s_k
        # y = x_shifted
        # y2 = s_shifted
        # repeat the ED vol, compose transform will register each time step to this phase
        if self.REGISTER_BACKWARDS:
            # x = x_k; y = x_k-w
            y_p2ed = np.repeat(x[:, 0:1, ...], 5, axis=1) # here we move each phase to the ED phase, starting from ED
            y2_p2ed_m = np.repeat(x2[:, 0:1, ...], 5, axis=1) # repeat the ED mask
        else:
            raise NotImplementedError('need to validate if forward works as expected')
            y_p2ed = np.roll(x, shift=-1, axis=1) # here we move the ed phase to each phases, starting with MS - same target as p2p
            y2_p2ed_m = np.roll(x2, shift=-1, axis=1)
        logging.debug('Batchsize: {} preprocessing took: {:0.3f} sec'.format(self.BATCHSIZE, time() - t0))
        zeros = np.zeros((*x.shape[:-1], 3), dtype=np.float32)
        if self.COMPOSE_CONSISTENCY:
            y2 = np.concatenate([y2_p2ed_m, y2], axis=-1) # repeated ED mask and s_shifted
            x = np.concatenate([x,y, y_p2ed], axis=-1) # stack [x_k, x_shifted, ed_repeated]

            return tuple([[x, x2], [y_p2ed, y, y2, zeros, zeros]]) # [x_k, s_k], [ed_repeated, s_ed and s_shifted, x_shifted, phi_k2k, phi_k2ed]
        else:
            return tuple([[x, x2], [y, y2, zeros]])

    def __pre_load_one_image__(self, i, ID):

        # --------------- HIST MATCHING REFERENCE VOL--------------
        # moved this part to the live preprocessing steps to increase the combinations of hist matching
        t1 = time()

        x = self.IMAGES[ID]

        # use the load_masked_img wrapper to enable masking of the images, currently not necessary, but nice to have
        # Note replace mask = False with mask=sel.MASKING_IMAGE
        model_inputs = load_masked_img(sitk_img_f=x, mask=self.MASKING_IMAGE,
                                       masking_values=self.MASKING_VALUES, replace=self.REPLACE_WILDCARD, maskAll=False)
        model_m_inputs = load_msk(f_name=x.replace(self.REPLACE_WILDCARD[0], self.REPLACE_WILDCARD[1]),
                                  valid_labels=self.MASK_VALUES)
        logging.debug('load and masking took: {:0.3f} s'.format(time() - t1))
        t1 = time()

        # --------------- TEMPORAL RESAMPLING AND Temp-AUGMENTATION--------------
        # resample the temporal resolution
        # if AUGMENT_TEMP --> add an temporal augmentation factor within the range given by: AUGMENT_TEMP_RANGE
        t_spacing = self.T_SPACING
        old_size = model_inputs.GetSize()[-2]
        if self.AUGMENT_TEMP: t_spacing = t_spacing + random.randint(self.AUGMENT_TEMP_RANGE[0],
                                                                     self.AUGMENT_TEMP_RANGE[1])
        logging.debug('t-spacing: {}'.format(t_spacing))
        if self.RESAMPLE_T:
            temporal_sampling_factor = model_inputs.GetSpacing()[-1] / t_spacing
            model_inputs = resample_t_of_4d(model_inputs, t_spacing=t_spacing, interpolation=self.IMG_INTERPOLATION,
                                            ismask=False)
            model_m_inputs = resample_t_of_4d(model_m_inputs, t_spacing=t_spacing, interpolation=self.IMG_INTERPOLATION,
                                              ismask=False)
        else:
            temporal_sampling_factor = 1  # dont scale the indices if we dont resample T
        logging.debug('temp resampling took: {:0.3f} s'.format(time() - t1))
        t1 = time()

        # --------------- LOAD INDICES FOR CARDIAC PHASES--------------
        # Returns the indices in the following order: 'ED#', 'MS#', 'ES#', 'PF#', 'MD#'
        if self.ISACDC:
            raise NotImplementedError('need to validate if get_phases_as_idx_acdc works')
            idx = get_phases_as_idx_acdc(x, temporal_sampling_factor, len(model_inputs))
        elif self.ISDMD:
            idx = get_phases_as_idx_dmd(x, self.DF_METADATA, temporal_sampling_factor, len(model_inputs))
        else:
            raise NotImplementedError('need to validate if get_phases_as_idx_gcn works')
            idx = get_phases_as_idx_gcn(x, self.DF_METADATA, temporal_sampling_factor, len(model_inputs))
        logging.debug('index loading took: {:0.3f} s'.format(time() - t1))

        # --------------- SPLIT IN 3D SITK IMAGES-------------
        # Create a list of 3D volumes for volume resampling
        model_inputs = split_one_4d_sitk_in_list_of_3d_sitk(model_inputs, axis=0, prob=self.AUGMENT_PROB)
        model_m_inputs = split_one_4d_sitk_in_list_of_3d_sitk(model_m_inputs, axis=0, prob=self.AUGMENT_PROB)
        logging.debug('split in t x 3D took: {:0.3f} s'.format(time() - t1))
        t1 = time()


        # logging.debug('transposed: \n{}'.format(onehot))
        # self.__plot_state_if_debug__(img=model_inputs[len(model_inputs) // 2], start_time=t0, step='raw')
        t1 = time()

        # --------------- SPATIAL RESAMPLING-------------
        if self.RESAMPLE:
            target_spacing = list(reversed(self.SPACING))
            input_z_spacing = model_inputs[0].GetSpacing()[-1]
            if not self.RESAMPLE_Z: # keep z
                target_spacing = target_spacing[:-1] + [input_z_spacing]

            if model_inputs[0].GetDimension() in [2, 3]:

                # calculate the new size of each 3D volume (after resample with the given spacing)
                # sitk.spacing has the opposite order than np.shape and tf.shape
                # In the config we use the numpy order z, y, x which needs to be reversed for sitk
                def calc_resampled_size(sitk_img, target_spacing):
                    if type(target_spacing) in [list, tuple]:
                        target_spacing = np.array(target_spacing)
                    old_size = np.array(sitk_img.GetSize())
                    old_spacing = np.array(sitk_img.GetSpacing())
                    logging.debug('old size: {}, old spacing: {}, target spacing: {}'.format(old_size, old_spacing,
                                                                                             target_spacing))
                    new_size = (old_size * old_spacing) / target_spacing
                    return list(np.around(new_size).astype(np.int))

                # transform the spacing from numpy representation towards the sitk representation
                new_size_inputs = list(map(lambda elem: calc_resampled_size(elem, target_spacing), model_inputs))

            else:
                raise NotImplementedError('dimension not supported: {}'.format(model_inputs[0].GetDimension()))

            logging.debug('dimension: {}'.format(model_inputs[0].GetDimension()))
            logging.debug('Size before resample: {}'.format(model_inputs[0].GetSize()))

            model_inputs = list(map(lambda x:
                                    resample_3D(sitk_img=x[0],
                                                size=x[1],
                                                spacing=target_spacing,
                                                interpolate=self.IMG_INTERPOLATION),  # sitk.sitkLinear
                                    zip(model_inputs, new_size_inputs)))
            # CHANGED
            model_m_inputs = list(map(lambda x:
                                      resample_3D(sitk_img=x[0],
                                                  size=x[1],
                                                  spacing=target_spacing,
                                                  interpolate=self.MSK_INTERPOLATION),  # sitk.nearest
                                      zip(model_m_inputs, new_size_inputs)))

        logging.debug('Spacing after resample: {}'.format(model_inputs[0].GetSpacing()))
        logging.debug('Size after resample: {}'.format(model_inputs[0].GetSize()))
        logging.debug('spatial resampling took: {:0.3f} s'.format(time() - t1))
        t1 = time()

        # --------------- CONTINUE WITH ND-ARRAYS --------------
        # transform to nda for further processing
        model_inputs = np.stack(list(map(lambda x: sitk.GetArrayFromImage(x), model_inputs)), axis=0)
        model_m_inputs = np.stack(list(map(lambda x: sitk.GetArrayFromImage(x), model_m_inputs)), axis=0)
        spatial_sampling_factor = new_size_inputs[0][-1] / old_size
        # Create a sparse mask from the interpolated/resampled mask,
        # by this we drop the interpolated spatial slices and replace them with zero padded slices
        # during training we might want to use only valid masks,
        # during prediction we use the resampled one to avoid cropping the displacement field along z
        if self.ISTRAINING and self.RESAMPLE_Z:
            masks_given = np.array(np.arange(0, old_size))
            scaled = masks_given * spatial_sampling_factor
            scaled = np.around(scaled).astype(int)
            temp = np.zeros_like(model_m_inputs, dtype=np.float32)
            temp[:, scaled, ...] = model_m_inputs[:, scaled, ...]
            model_m_inputs = temp

        logging.debug('transform to nda took: {:0.3f} s'.format(time() - t1))
        t1 = time()


        # align and crop before smoothing, this improves the speed of the following steps
        # and reduces the memory footprint
        model_inputs, model_m_inputs = align_inplane_with_ip(model_inputs=model_inputs,
                                                             msk_file_name=model_m_inputs,
                                                             roll2septum=False,
                                                             roll2lvbood=True,
                                                             rotate=False,
                                                             translate=True)

        model_inputs = pad_and_crop(model_inputs, target_shape=(model_inputs.shape[0], *self.DIM))
        model_m_inputs = pad_and_crop(model_m_inputs, target_shape=(model_m_inputs.shape[0], *self.DIM))
        logging.debug('pad/crop took: {:0.3f} s'.format(time() - t1))
        t1 = time()

        # clip the upper and lower quantile of the cmr grey values
        q = 0.99
        q_lower = 1 - q
        lower_threshold = np.quantile(a=model_inputs, q=q_lower)
        model_inputs = clip_quantile(model_inputs, upper_quantile=q, lower_boundary=lower_threshold)

        #model_inputs = clip_quantile(model_inputs, .99)
        logging.debug('quantile clipping took: {:0.3f} s'.format(time() - t1))
        t1 = time()

        model_inputs = normalise_image(model_inputs, normaliser=self.SCALER)  # normalise per 4D
        logging.debug('normalisation took: {:0.3f} s'.format(time() - t1))
        t1 = time()

        # Added mask smoothness
        import scipy
        import scipy.ndimage
        if self.RESAMPLE_Z:  # smooth only if we have fake isotropy voxels, otherwise we use the original slices
            '''for t in range(model_m_inputs.shape[0]):
                if model_m_inputs[t].sum() > 0:  # we only need to smooth time steps with a mask
                        model_m_inputs[t]  = scipy.ndimage.binary_closing(model_m_inputs[t], iterations=5)
                        #model_m_inputs[t] = scipy.ndimage.gaussian_filter(model_m_inputs[t] , sigma=1, mode='nearest')
                        #model_m_inputs[t] = (model_m_inputs[t] >0.1).astype(np.float32)
                        #model_m_inputs[t] = scipy.ndimage.binary_dilation(model_m_inputs[t], iterations=1)
            '''
            # this 'smoothed' mask is a little greater than the version above, and covers more of the myocardium
            for t in range(model_m_inputs.shape[0]):
                if model_m_inputs[t].sum() > 0:  # we only need to smooth time steps with a mask
                    smooth = scipy.ndimage.binary_closing(model_m_inputs[t], iterations=5)
                    smooth = (smooth > 0.2).astype(np.float32)
                    for z in range(smooth.shape[0]):
                        smooth[z] = scipy.ndimage.gaussian_filter(smooth[z], sigma=0.8, mode='nearest')
                    model_m_inputs[t] = (smooth > 0.2).astype(np.float32)
        #show_2D_or_3D(model_inputs[5, ...], model_m_inputs[5, ...])'''

        # --------------- SLICE PAIRS OF INPUT AND TARGET VOLUMES ACCORDING TO CARDIAC PHASE IDX -------------
        # register backwards returns: [x_k-1, x_k]
        if self.BETWEEN_PHASES:
            combined = get_n_windows_between_phases_from_single4D(model_inputs, idx,
                                                                  register_backwards=self.REGISTER_BACKWARDS,
                                                                  intermediate=False)
            combined_m = get_n_windows_between_phases_from_single4D(model_m_inputs, idx,
                                                                    register_backwards=self.REGISTER_BACKWARDS,
                                                                    intermediate=False)
        else:  # Extract the motion before a keyframe, defined by a window size w
            # we expect the same temporal forward ordering as for k2k
            # [x_k-w, x_k]
            combined = get_n_windows_from_single4D(model_inputs, idx, window_size=self.WINDOW_SIZE,
                                                   register_backwards=self.REGISTER_BACKWARDS,
                                                   intermediate=False
                                                   )
            assert all(np.any(model_m_inputs, axis=(
            1, 2, 3))), 'window mode will only work with anatomical constraint if we have a mask for every timestep'
            if all(np.any(model_m_inputs, axis=(1, 2, 3))):
                # all time steps / frames in this 4D sequence have a mask (this is very likely a predicted mask)
                # use the masks defined by the window
                combined_m = get_n_windows_from_single4D(model_m_inputs, idx, window_size=self.WINDOW_SIZE,
                                                         register_backwards=self.REGISTER_BACKWARDS,
                                                         intermediate=False
                                                         )
            else:
                raise NotImplementedError('check this behaviour after rafactoring')
                # not all time steps have a mask, this is very likely a GT, with contours only at the 5 labelled phases,
                # use the labelled masks defined by the phases in the phase df
                # Use this for masks with only 5 time-steps labelled. moving and target mask will be the same
                combined_m = get_n_windows_between_phases_from_single4D(model_m_inputs, idx,
                                                                        register_backwards=self.REGISTER_BACKWARDS,
                                                                        intermediate=False)

            logging.debug('windowing slicing took: {:0.3f} s'.format(time() - t1))
            t1 = time()

            # results in: 5,z,x,y,c with c==3 or c==2
            # # [x_k-w, x_k]
        combined = np.stack(combined, axis=-1)
        combined_m = np.stack(combined_m, axis=-1)

        # check if the mask for one time step is zero
        if not all(np.any(combined_m, axis=(1, 2, 3, 4))):
            print('please check the masks!')

        logging.debug('stacking took: {:0.3f} s'.format(time() - t1))
        return combined, combined_m, i, idx

        #return model_inputs, model_m_inputs, i, idx

    def __preprocess_one_image__(self, i, ID):
        t0 = time()
        t1 = time()
        # --------------- LOAD THE MODEL INPUT--------------
        # combined: 5,z,x,y,c with c==2
        # temporal order of these channels: [nda[idx_shift_to_left], nda[idx_middle], nda[idxs]]
        if self.IN_MEMORY:
            combined, combined_m, idx = self.IMAGES_SITK[ID], self.MASKS_SITK[ID], self.INDICIES[ID]
        else:
            combined, combined_m, i, idx = self.__pre_load_one_image__(i, ID)

        # --------------- HIST MATCHING--------------
        if self.HIST_MATCHING and random.random() <= 0.5:
            # this image has the original inplane resolution

            if self.IN_MEMORY:
                ref_id = choice(range(len(self.IMAGES_SITK)))
                ref = self.IMAGES_SITK[ref_id]
                combined = match_hist_any_dim(combined, ref) # this histogram should be also clipped and normalised
            else:
                ref_id = choice(range(len(self.IMAGES)))
                ref = sitk.GetArrayFromImage(sitk.ReadImage(self.IMAGES[ref_id]))
                ref_m = sitk.GetArrayFromImage(sitk.ReadImage(self.IMAGES[ref_id].replace('clean', 'mask')))
                ref, ref_m = align_inplane_with_ip(model_inputs=ref,
                                                   msk_file_name=ref_m,
                                                   roll2septum=False,
                                                   roll2lvbood=True,
                                                   rotate=False,
                                                   translate=True)

                ref = pad_and_crop(ref, target_shape=(
                    ref.shape[0], *self.DIM))  # we do not resample here for computational reasons
                q = 0.99
                q_lower = 1 - q
                lower_threshold = np.quantile(a=ref, q=q_lower)
                ref = clip_quantile(ref, upper_quantile=q, lower_boundary=lower_threshold)
                # this is just a local hack,
                # Only inmemory processing matches the histogram in the right manner,
                # here we match a 4D CMR histogram on two temporally subsampled 4D volumes
                combined = match_hist_any_dim(combined, ref)  # match the 4D histogram
                combined = normalise_image(combined, normaliser=self.SCALER)  # normalise moving and fixed independent
                # combined[1] = normalise_image(combined[1], normaliser=self.SCALER)
                logging.debug('hist matching took: {:0.3f} s'.format(time() - t1))
                t1 = time()

        # continue with live data modifications, such as augmentation/normalisation and standardisation
        # --------------- Image Augmentation, this is done in 2D -------------
        if self.AUGMENT and random.random() <= self.AUGMENT_PROB:

            # use albumentation to apply random augmentations
            # we need to make sure to apply the same augmentation on the input and target data
            # Albumentation uses the Interpolation enum from opencv which is different to the SimpleITK enum
            shape_ = combined.shape
            shape_m = combined_m.shape
            combined = np.reshape(combined, newshape=(shape_[0] * shape_[-1], *shape_[1:-1]))
            combined_m = np.reshape(combined_m, newshape=(shape_m[0] * shape_m[-1], *shape_m[1:-1]))
            logging.debug('shape combined: {}'.format(combined.shape))

            combined, combined_m = augmentation_compose_2d_3d_4d(img=combined, mask=combined_m, config=self.config)
            logging.debug('shape combined: {}'.format(combined.shape))
            combined=np.reshape(combined, newshape=shape_)
            combined_m = np.reshape(combined_m, newshape=shape_m)

            # self.__plot_state_if_debug__(img=combined[self.INPUT_T_ELEM][0], start_time=t1, step='augmented')
            logging.debug('augmentation took: {:0.3f} s'.format(time() - t1))
            t1 = time()

        # combined = [x_shifted, x_k]
        # for k2k: [x_k-1, x_k]
        # for window: [x_k-w, x_k]
        # for composed []
        # register backwards: move x_k to fit x_shifted
        # x_k = moving, x_shifted = fixed
        # register backwards == pull vectors which sample from the target/fixed/x_shifted grid
        # this means vectors are from x_shifted/target/fixed (earlier in time) towards x_k/moving
        #combined = normalise_image(combined, normaliser=self.SCALER)  # normalise per 4D
        # model_inputs, model_m_inputs, model_targets, model_m_targets
        # shifted refers here to back in time (for backward registering)
        if self.IMG_CHANNELS==1:
            x_k = combined[..., -1:]
            x_k_shifted = combined[..., :1]
            s_k = combined_m[..., -1:]
            s_k_shifted = combined_m[...,:1]
        else:
            x_k = np.broadcast_to(combined[..., -1:], shape=self.X_SHAPE.shape[1:])
            x_k_shifted = combined[..., :1]
            s_k = transform_to_binary_mask(combined_m[..., -1], self.MASK_VALUES)#[..., -1:] # all labels mask
            s_k_shifted = combined_m[...,:1]


        assert not np.any(np.isnan(x_k))
        assert not np.any(np.isnan(x_k_shifted))

        return x_k, s_k, x_k_shifted, s_k_shifted, i, ID, time() - t0


import linecache
import sys


def PrintException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))
