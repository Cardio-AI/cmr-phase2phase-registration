from src.data.Dataset import save_all_3d_vols_new


def pred_fold(config, debug=True):
    # make sure all neccessary params in config are set
    # if not set them with default values
    from src.utils.Tensorflow_helper import choose_gpu_by_id
    # ------------------------------------------define GPU id/s to use
    GPU_IDS = config.get('GPU_IDS', '0,1')
    GPUS = choose_gpu_by_id(GPU_IDS)
    print(GPUS)
    # ------------------------------------------ import helpers
    # this should import glob, os, and many other standard libs
    from tensorflow.python.client import device_lib
    import tensorflow as tf
    tf.get_logger().setLevel('FATAL')
    import gc, logging, os, datetime, re
    from logging import info

    # local imports
    from src.utils.Utils_io import Console_and_file_logger, init_config, ensure_dir
    from src.utils.KerasCallbacks import get_callbacks
    from src.data.Dataset import get_trainings_files
    from src.data.Generators import PhaseWindowGenerator
    from src.models.Models import create_RegistrationModel
    import numpy as np

    from src.data.Dataset import save_gt_and_pred
    from src.data.Generators import PhaseMaskWindowGenerator
    from src.models.Models import create_affine_transformer_fixed, create_RegistrationModel_inkl_mask

    # import external libs
    import pandas as pd
    from time import time

    # make all config params known to the local namespace
    locals().update(config)

    # overwrite the experiment names and paths, so that each cv gets an own sub-folder
    EXPERIMENT = config.get('EXPERIMENT')
    FOLD = config.get('FOLD')

    EXPERIMENT = '{}_f{}'.format(EXPERIMENT, FOLD)
    timestemp = str(datetime.datetime.now().strftime(
        "%Y-%m-%d_%H_%M"))  # add a timestep to each project to make repeated experiments unique

    DATA_PATH_SAX = config.get('DATA_PATH_SAX')
    DF_FOLDS = config.get('DF_FOLDS')
    DF_META = config.get('DF_META', '/mnt/ssd/data/gcn/02_imported_4D_unfiltered/SAx_3D_dicomTags_phase')
    EPOCHS = config.get('EPOCHS', 100)

    Console_and_file_logger(path=EXPERIMENT, log_lvl=logging.INFO)

    # get kfolded data from DATA_ROOT and subdirectories
    # Load SAX volumes
    x_train_sax, y_train_sax, x_val_sax, y_val_sax = get_trainings_files(data_path=DATA_PATH_SAX,
                                                                         path_to_folds_df=DF_FOLDS,
                                                                         fold=FOLD)
    logging.info('SAX train CMR: {}, SAX train masks: {}'.format(len(x_train_sax), len(y_train_sax)))
    logging.info('SAX val CMR: {}, SAX val masks: {}'.format(len(x_val_sax), len(y_val_sax)))

    t0 = time()

    try:

        # load the model, to make sure we use the same as later for the evaluations
        model = create_RegistrationModel_inkl_mask(config)
        model.load_weights(os.path.join(config['MODEL_PATH'], 'model.h5'))
        logging.info('loaded model weights as h5 file')

        fixed_transformer = create_affine_transformer_fixed(config=config,networkname='fixed_transformer', fill_value=0, interp_method='linear')

        # create a generator with idemptent behaviour (no shuffle etc.)
        # make sure we save always the same patient
        pred_config = config.copy()
        pred_config['SHUFFLE'] = False
        pred_config['AUGMENT'] = False
        pred_config['AUGMENT_PHASES'] = False
        pred_config['AUGMENT_TEMP'] = False
        pred_config['BATCHSIZE'] = 1
        pred_config['HIST_MATCHING'] = False
        pred_config['ISTRAINING'] = False
        INPUT_T_ELEM = config.get('INPUT_T_ELEM', 0)
        pred_generator = PhaseMaskWindowGenerator(x_train_sax, x_train_sax, config=pred_config)
        full_pred_config = pred_config.copy()
        full_pred_config['MASKING_IMAGE'] = False
        full_image_generator = PhaseWindowGenerator(x_train_sax, x_train_sax, config=full_pred_config)
        x_train_sax_masks = [f.replace('clean', 'mask') for f in x_train_sax]
        '''pred_mask_generator = PhaseWindowGenerator(x_train_sax_masks, x_train_sax_masks, config=pred_config,
                                                   yield_masks=True)'''
        import SimpleITK as sitk
        pred_mask_config = pred_config.copy()
        pred_mask_config['MASKING_VALUES']=[2]
        pred_mask_config['MASKING_IMAGE']=True
        pred_mask_config['IMG_INTERPOLATION'] = sitk.sitkNearestNeighbor
        pred_mask_config['MSK_INTERPOLATION'] = sitk.sitkNearestNeighbor
        pred_myo_mask_generator = PhaseWindowGenerator(x_train_sax_masks, x_train_sax_masks, config=pred_mask_config,
                                                   yield_masks=True)
        '''pred_mask_config['MASKING_VALUES'] = [3]
        pred_lv_mask_generator = PhaseWindowGenerator(x_train_sax_masks, x_train_sax_masks, config=pred_mask_config,
                                                   yield_masks=True)'''
        pred_mask_config['MASKING_IMAGE'] = False
        pred_mask_config['IMG_CHANNELS'] = 1
        pred_mask_config['TARGET_CHANNELS'] = 1
        pred_lv_mask_generator = PhaseWindowGenerator(x_train_sax_masks, x_train_sax_masks, config=pred_mask_config,
                                                      yield_masks=True)

        # mask the vetorfield
        from scipy import ndimage
        # combined_mask = second_binary
        # combined_mask = first_binary + second_binary
        kernel = np.ones((1, 1, 5, 5, 5, 1))
        kernel_ = np.ones((1, 5, 5, 5, 1))
        kernel_small = np.ones((1, 1, 3, 3, 3, 1))
        prediction_tuple = x_train_sax, pred_generator, pred_myo_mask_generator, pred_lv_mask_generator, full_image_generator

        for filename, pred_batch, myo_mask_b, lv_mask_b, full_cmr in zip(*prediction_tuple):

            # first_vols shape:
            # Batch, Z, X, Y, Channels --> three timesteps - t_n-1, t_n, t_n+1

            first_vols_, second_vols_ = pred_batch
            first_mask, second_mask = myo_mask_b
            first_lvmask, second_lvmask = lv_mask_b
            first_vols_full, second_vols_full= full_cmr

            first_vols, second_vols = first_vols_[0], second_vols_[0]  # pick batch 0
            first_mask, second_mask = first_vols_[1], second_vols_[1]  # pick batch 0
            first_lvmask, second_lvmask = first_lvmask[0], second_lvmask[0]
            first_mask, second_mask = (first_mask>=0.5).astype(np.uint8), (second_mask>0.5).astype(np.uint8)
            # first_lvmask, second_lvmask = (first_lvmask >= 0.5).astype(np.uint8), (second_lvmask > 0.5).astype(np.uint8)
            first_lvmask, second_lvmask = (first_lvmask).astype(np.uint8), (second_lvmask).astype(np.uint8)
            first_vols_full, second_vols_full = first_vols_full[0], second_vols_full[0]  # pick batch 0

            first_vols = first_vols[..., INPUT_T_ELEM][..., np.newaxis]  # select the transformed source vol
            first_vols_full = first_vols_full[..., INPUT_T_ELEM][..., np.newaxis]  # select the transformed source vol
            first_mask = first_mask[..., INPUT_T_ELEM][..., np.newaxis]
            first_lvmask = first_lvmask[..., INPUT_T_ELEM][..., np.newaxis]

            preds = model.predict_on_batch(pred_batch)
            if len(preds) == 2:
                moved, vects = preds
                # if our model does not transform the mask, do it separately
                moved_m, _ = fixed_transformer.predict(x=[first_mask, vects])
            else:
                moved, moved_m, vects = preds
            moved = tf.cast(moved, tf.float32)
            # compose vectors

            from src.models.Models import create_dense_compose
            comp = create_dense_compose(config)
            vects_composed = comp.predict(vects)

            moved_m = tf.cast(moved_m > 0.5, tf.uint8)
            moved_m = ndimage.binary_closing(moved_m, structure=kernel, iterations=1)
            #moved_m = ndimage.binary_opening(moved_m, structure=kernel_small, iterations=1)

            moved_m = tf.cast(moved_m,tf.uint8)

            first_binary = first_mask > 0.5
            second_binary = second_mask > 0.5
            #first_binary = ndimage.binary_closing(first_binary, structure=kernel, iterations=1)


            #first_binary = ndimage.median_filter(first_mask, size=3, mode='nearest')
            #combined_mask = combined_mask.astype(np.float32)
            #second_binary = ndimage.convolve(second_binary, weights=kernel)
            #second_binary = second_binary>=0.2
            second_mask = tf.cast(second_binary, tf.uint8)
            first_mask = tf.cast(first_binary, tf.uint8)
            vects_full = np.copy(vects)
            second_binary = ndimage.binary_dilation(second_binary, structure=kernel_small, iterations=1)

            second_binary = second_binary[...,0]
            for dim in range(vects.shape[-1]):
                vects[...,dim][~second_binary] = 0

            # TODO: refactor?
            from src.data.Dataset import save_all_3d_vols
            pred_path = os.path.join(config.get('EXP_PATH'), 'pred')
            p = os.path.basename(filename).split('_volume')[0].lower()
            ensure_dir(pred_path)

            volumes = [first_vols[0],
                       second_vols[0],
                       moved[0],
                       moved_m[0],
                       vects[0],
                       vects_full[0],
                       vects_composed[0],
                       first_mask[0],
                       second_mask[0],
                       first_lvmask[0],
                       second_lvmask[0],
                       first_vols_full[0],
                       second_vols[0]]

            suffixes = ['_cmr.nii', '_targetcmr.nii', '_movedcmr.nii', '_movedmask.nii',
                        '_flow.nii', '_flow_full.nii', '_flow_comp.nii', '_mask.nii', '_targetmask.nii', '_lvmask.nii',
                         '_lvtargetmask.nii',
                         '_cmr_full.nii', '_targetcmr_full.nii'  ]

            if debug:
                save_all_3d_vols_new(volumes, vol_suffixes=suffixes,
                                 EXP_PATH=pred_path, exp=p)

            save_gt_and_pred(gt=second_mask[0], pred=moved_m[0], exp_path=config.get('EXP_PATH'), patient=p)

    except Exception as e:
        logging.error(e)
        logging.error(first_vols.shape)
        logging.error(second_vols.shape)
        logging.error(vects.shape)
        logging.error(moved_m.shape)

    # free as much memory as possible
    del pred_generator
    del model
    gc.collect()

    logging.info('pred on fold {} finished after {:0.3f} sec'.format(FOLD, time() - t0))
    return True


def main(args=None):
    # ------------------------------------------define logging and working directory
    # import the packages inside this function enables to train on different folds
    from ProjectRoot import change_wd_to_project_root
    change_wd_to_project_root()
    import sys, os, datetime
    sys.path.append(os.getcwd())
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # ------------------------------------------define GPU id/s to use, if given

    # local imports
    from src.utils.Utils_io import Console_and_file_logger, init_config
    # import external libs
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    import cv2
    import logging
    from logging import info

    EXPERIMENTS_ROOT = 'exp/'

    if args.exp:
        import json
        cfg = os.path.join(args.exp, 'config/config.json')
        print('config given: {}'.format(cfg))
        # load the experiment config
        with open(cfg, encoding='utf-8') as data_file:
            config = json.loads(data_file.read())

            EXPERIMENT = config.get('EXPERIMENT', 'UNDEFINED')
            Console_and_file_logger(EXPERIMENT, logging.INFO)
            info('Loaded config for experiment: {}'.format(EXPERIMENT))

            # make relative paths absolute
            config['MODEL_PATH'] = os.path.join(args.exp, 'model/')
            config['EXP_PATH'] = args.exp

            # Load SAX volumes
            # cluster to local data mapping
    if args.data:
        data_root = args.data
        config['DATA_PATH_SAX'] = os.path.join(data_root, 'sax')
        config['DF_FOLDS'] = os.path.join(data_root, 'df_kfold.csv')
        config['DF_META'] = os.path.join(data_root, 'SAx_3D_dicomTags_phase.csv')



    pred_fold(config)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='train a phase registration model')

    # usually these two parameters should encapsulate all experiment parameters
    parser.add_argument('-exp', action='store', default=None)
    parser.add_argument('-data', action='store', default=None)

    results = parser.parse_args()
    print('given parameters: {}'.format(results))

    try:
        main(results)
    except Exception as e:
        print(e)
    exit()