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
    import SimpleITK as sitk
    from scipy import ndimage
    from src.models.Models import create_dense_compose
    from src.data.Dataset import save_all_3d_vols
    import os

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

        pred_path = os.path.join(config.get('EXP_PATH'), 'pred')
        ensure_dir(pred_path)

        # create a generator with idempotent behaviour (no shuffle etc.)
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
        pred_generator = PhaseMaskWindowGenerator(x_val_sax, x_val_sax, config=pred_config)
        x_train_sax_masks = [f.replace('clean', 'mask') for f in x_val_sax]
        pred_mask_config = pred_config.copy()
        pred_mask_config['IMG_INTERPOLATION'] = sitk.sitkNearestNeighbor
        pred_mask_config['MSK_INTERPOLATION'] = sitk.sitkNearestNeighbor
        pred_mask_config['MASKING_IMAGE'] = False
        pred_mask_config['IMG_CHANNELS'] = 1
        pred_mask_config['TARGET_CHANNELS'] = 1
        masks_all_labels_generator = PhaseWindowGenerator(x_train_sax_masks, x_train_sax_masks, config=pred_mask_config,
                                                      yield_masks=True)
        # new version
        x, y = zip(*[masks_all_labels_generator.__getitem__(i) for i in range(len(pred_generator))])
        fullmsk_target, _  = zip(*y)
        fullmsk_target = np.concatenate(fullmsk_target, axis=0)
        x, y = zip(*[pred_generator.__getitem__(i) for i in range(len(pred_generator))])
        cmr_moving, msk_moving, _ = zip(*x)
        cmr_target, msk_target, _ = zip(*y)
        cmr_moving, msk_moving, cmr_target, msk_target = map(np.concatenate, [cmr_moving, msk_moving,cmr_target, msk_target])
        pred = model.predict(pred_generator)
        cmr_moved, msk_moved, flows = pred
        comp = create_dense_compose(config)
        vects_composed = comp.predict(flows)
        # mask the vector field

        # iterate over the patients and
        for i in range(len(x_val_sax)):
            filename = x_val_sax[i]
            cmr_mov = cmr_moving[i]
            cmr_t = cmr_target[i]
            cmr_m = cmr_moved[i]
            msk_mov = msk_moving[i]
            msk_t = msk_target[i]
            msk_m = msk_moved[i]
            flow = flows[i]
            flow_c = vects_composed[i]
            fullmsk_t = fullmsk_target[i]
            flow_masked = flow.copy()
            msk_myo = np.squeeze(msk_t.astype(np.bool))
            for dim in range(flow.shape[-1]):
                flow_masked[...,dim][~msk_myo] = 0

            # save all files of this patient
            p = os.path.basename(filename).split('_volume')[0].lower()
            volumes = [cmr_mov,cmr_t,cmr_m,msk_mov,msk_t,msk_m,flow,flow_c,flow_masked,fullmsk_t]

            suffixes = ['cmr_moving.nii', 'cmr_target.nii', 'cmr_moved.nii',
                        'myo_moving.nii', 'myo_target.nii', 'myo_moved.nii',
                        'flow.nii', 'flow_composed.nii', 'flow_masked.nii',
                        'fullmask_target.nii']
            if debug:
                save_all_3d_vols_new(volumes, vol_suffixes=suffixes,
                                     EXP_PATH=pred_path, exp=p)

            save_gt_and_pred(gt=msk_t, pred=msk_mov, exp_path=config.get('EXP_PATH'), patient=p)
        # end new version

    except Exception as e:
        logging.error(e)

    # free as much memory as possible
    del pred_generator
    del model
    del masks_all_labels_generator
    del fullmsk_target
    del cmr_moving
    del msk_moving
    del cmr_target
    del msk_target
    del pred
    del cmr_moved
    del msk_moved
    del flows
    del vects_composed
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