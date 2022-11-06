


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
    from src.models.Models import create_RegistrationModel_inkl_mask
    from src.data.Dataset import save_all_3d_vols_new

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

    # ------------------------------------------define GPU id/s to use
    GPU_IDS = config.get('GPU_IDS', '0,1')
    GPUS = choose_gpu_by_id(GPU_IDS)
    print(GPUS)
    print(tf.config.list_physical_devices('GPU'))

    EXPERIMENT = config.get('EXPERIMENT', 'UNDEFINED')
    Console_and_file_logger(EXPERIMENT, logging.INFO)
    info('Loaded config for experiment: {}'.format(EXPERIMENT))

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

    # load the model, to make sure we use the same as later for the evaluations
    model = create_RegistrationModel_inkl_mask(config)
    model.load_weights(os.path.join(config['MODEL_PATH'], 'model.h5'))
    logging.info('loaded model weights as h5 file')

    # quick fix
    pred_path = os.path.join(config.get('EXP_PATH'), 'pred')
    pred_path = pred_path.replace('f0/','').replace('f1/','').replace('f2/','').replace('f3/','')

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
    compose_given = config.get('COMPOSE_CONSISTENCY', False)
    masks_all_labels_generator = PhaseWindowGenerator(x_train_sax_masks, x_train_sax_masks, config=pred_mask_config,
                                                  yield_masks=True)
    # new version
    x, y = zip(*[masks_all_labels_generator.__getitem__(i) for i in range(len(masks_all_labels_generator))])
    fullmsk_target, _  = zip(*y)
    fullmsk_target = np.concatenate(fullmsk_target, axis=0)
    # here we get a list of batches, each with a batchsize of 1
    x, y = zip(*[pred_generator.__getitem__(i) for i in range(len(pred_generator))])
    cmr_moving, msk_moving = zip(*x)
    if len(y[0]) == 4:
        ed_repeated, cmr_target, msk_target, _ = zip(*y)
    elif len(y[0])==5:
        ed_repeated, cmr_target, msk_target, _, _ = zip(*y)
    else:
        cmr_target, msk_target, _ = zip(*y)
    cmr_moving, msk_moving, cmr_target, msk_target = map(np.concatenate, [cmr_moving, msk_moving,cmr_target, msk_target])
    pred = model.predict(pred_generator)
    if len(pred)==4:
        comp_moved2ed, cmr_moved, msk_moved, flows = pred
    elif len(pred)==5:
        comp_moved2ed, cmr_moved, msk_moved, flows, flows2ed = pred

    else:
        cmr_moved, msk_moved, flows = pred

    # mask the flow field with the target mask
    # also necessary for the compose
    flows_masked = flows.copy()
    msk_t = np.squeeze(msk_target>0.1)
    for dim in range(flows.shape[-1]):
        flows_masked[..., dim][~msk_t] = 0

    if compose_given:
        #msk_ed = np.repeat(msk_target[:,0:1],5,axis=1)# mask the compose flowfield with ED (fixed)
        flows_composed_masked = flows2ed.copy()
        #msk_ed = np.squeeze(msk_ed>0.1)
        #for dim in range(flows2ed.shape[-1]):
        #    flows_composed_masked[..., dim][~msk_ed] = 0
    else:
        comp = create_dense_compose(config)
        flows_composed_masked = comp.predict(flows_masked)

    # iterate over the patients and
    for i in range(len(x_val_sax)):
        filename = x_val_sax[i]
        cmr_mov = cmr_moving[i][...,0:1]
        cmr_t = cmr_target[i]
        cmr_m = cmr_moved[i]
        msk_mov = msk_moving[i][...,0:1]
        msk_t = msk_target[i] # target mask of each pair-wise p2p

        msk_m = msk_moved[i]
        flow = flows[i]
        flow_comp_m = flows_composed_masked[i]
        flow_masked = flows_masked[i]
        fullmsk_t = fullmsk_target[i]

        # save all files of this patient
        p = os.path.basename(filename).split('_volume')[0].lower()
        volumes = [cmr_mov,cmr_t,cmr_m,msk_mov,msk_t,msk_m,flow,flow_comp_m,flow_masked,fullmsk_t]

        suffixes = ['cmr_moving.nii', 'cmr_target.nii', 'cmr_moved.nii',
                    'myo_moving.nii', 'myo_target.nii', 'myo_moved.nii',
                    'flow.nii', 'flow_composed.nii', 'flow_masked.nii',
                    'fullmask_target.nii']
        if debug:
            save_all_3d_vols_new(volumes, vol_suffixes=suffixes,
                                 EXP_PATH=pred_path, exp=p, cfg=config)

        save_gt_and_pred(gt=msk_t, pred=msk_m, exp_path=pred_path.replace('pred',''), patient=p, cfg=config)
        # end new version
    return True

    # # free as much memory as possible
    # del pred_generator
    # del model
    # del masks_all_labels_generator
    # del fullmsk_target
    # del cmr_moving
    # del msk_moving
    # del cmr_target
    # del msk_target
    # del pred
    # del cmr_moved
    # del msk_moved
    # del flows
    # gc.collect()
    #
    # logging.info('pred on fold {} finished after {:0.3f} sec'.format(FOLD, time() - t0))
    # return True


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
    import glob
    import os
    import json
    if os.path.isdir(os.path.join(args.exp, 'model')):

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
        # if args.data:
        data_root = args.data
        config['DATA_PATH_SAX'] = os.path.join(data_root, 'sax')
        df_folds = os.path.join(data_root, 'df_kfold.csv')
        # this part is necessary if we try to predict on files that are not in the df folds file
        # meaning dta which we did not use in the cv (e.g. dmd control)
        if os.path.isfile(df_folds):
            config['DF_FOLDS'] = df_folds
        else:
            config['DF_FOLDS'] = None

        df_meta = os.path.join(data_root, 'SAx_3D_dicomTags_phase.csv')
        if os.path.isfile(df_meta):
            config['DF_META'] = df_meta
        else:
            config['DF_META'] = None
        pred_fold(config)
    else: # predict a CV
        for exp_fold in list(sorted(glob.glob(os.path.join(args.exp, 'f*/')))):

            cfg = os.path.join(exp_fold, 'config/config.json')
            print('config given: {}'.format(cfg))
            # load the experiment config
            with open(cfg, encoding='utf-8') as data_file:
                config = json.loads(data_file.read())

            EXPERIMENT = config.get('EXPERIMENT', 'UNDEFINED')
            Console_and_file_logger(EXPERIMENT, logging.INFO)
            info('Loaded config for experiment: {}'.format(EXPERIMENT))

            # make relative paths absolute
            config['MODEL_PATH'] = os.path.join(exp_fold, 'model/')
            config['EXP_PATH'] = exp_fold

                    # Load SAX volumes
                    # cluster to local data mapping
            #if args.data:
            data_root = args.data
            config['DATA_PATH_SAX'] = os.path.join(data_root, 'sax')
            df_folds = os.path.join(data_root, 'df_kfold.csv')
            # this part is necessary if we try to predict on files that are not in the df folds file
            # meaning dta which we did not use in the cv (e.g. dmd control)
            if os.path.isfile(df_folds) :
                config['DF_FOLDS'] = df_folds
            else :
                config['DF_FOLDS'] = None

            df_meta = os.path.join(data_root, 'SAx_3D_dicomTags_phase.csv')
            if os.path.isfile(df_meta):
                config['DF_META'] = df_meta
            else:
                config['DF_META'] = None
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