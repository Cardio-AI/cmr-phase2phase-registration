from src.data.Preprocess import from_channel_to_flat
from src.models.Evaluate_moved_myo import calc_dice


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
    #from tensorflow.python.client import device_lib
    import tensorflow as tf
    tf.get_logger().setLevel('FATAL')
    import gc, logging, os, datetime, re
    from logging import info

    # local imports
    from src.utils.Utils_io import Console_and_file_logger, init_config, ensure_dir
    from src.utils.KerasCallbacks import get_callbacks
    from src.data.Dataset import get_trainings_files
    #from src.data.Generators import PhaseWindowGenerator
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
    DF_META = config.get('DF_META', None)
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

    # quick fix to make sure we are in the exp root
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
    pred_generator = PhaseMaskWindowGenerator(x_val_sax, y_val_sax, config=pred_config)
    x_train_sax_masks = [f.replace('clean', 'mask') for f in x_val_sax]
    pred_mask_config = pred_config.copy()
    pred_mask_config['IMG_INTERPOLATION'] = sitk.sitkNearestNeighbor
    pred_mask_config['MSK_INTERPOLATION'] = sitk.sitkNearestNeighbor
    pred_mask_config['MASKING_IMAGE'] = False
    pred_mask_config['MASK_VALUES'] = [1,2,3]
    pred_mask_config['IMG_CHANNELS'] = 3
    pred_mask_config['TARGET_CHANNELS'] = 1
    compose_given = config.get('COMPOSE_CONSISTENCY', False)
    #masks_all_labels_generator = PhaseWindowGenerator(x_train_sax_masks, x_train_sax_masks, config=pred_mask_config,
    #                                              yield_masks=True)
    masks_all_labels_generator = PhaseMaskWindowGenerator(x_val_sax, y_val_sax, config=pred_mask_config)

    # iterate over the patients and
    for i in range(len(x_val_sax)):


        x, y = masks_all_labels_generator.__getitem__(i)
        _, fullmask_moving = x

        x, y = pred_generator.__getitem__(i)
        cmr_moving, msk_moving = x

        if len(y) == 4:
            ed_repeated, cmr_target, msk_target, _ = y
        elif len(y) == 5:
            ed_repeated, cmr_target, msk_target, _, _ = y
        else:
            cmr_target, msk_target, _ = y

        pred = model.predict_on_batch(pred_generator.__getitem__(i)[0])

        if len(pred) == 4:
            comp_moved2ed, cmr_moved, msk_moved, flows = pred
        elif len(pred) == 5:
            comp_moved2ed, cmr_moved, msk_moved, flows, flows2ed = pred
        else:
            cmr_moved, msk_moved, flows = pred

        flows_masked = flows
        target_msk_k2k = msk_target[..., 1:2]
        msk_t = np.squeeze(target_msk_k2k > 0.1)
        """if msk_t.shape[-1] > 1:
                msk_t = msk_t[..., None]"""
        # for dim in range(flows.shape[-1]):
        #   flows_masked[..., dim][~msk_t] = 0

        if compose_given:
            # msk_ed = np.repeat(msk_target[:,0:1],5,axis=1)# mask the compose flowfield with ED (fixed)
            flows_composed_masked = flows2ed.copy()
            target_msk_k2ed = msk_target[..., :1]
            msk_ed = np.squeeze(target_msk_k2ed > 0.1)
            # for dim in range(flows2ed.shape[-1]):
            #    flows_composed_masked[..., dim][~msk_ed] = 0
        else:
            comp = create_dense_compose(config)
            flows_composed_masked = comp.predict(flows_masked)

        filename = x_val_sax[i]
        cmr_mov = cmr_moving[0][..., 0:1]
        cmr_t = cmr_target[0]
        cmr_m = cmr_moved[0]
        msk_mov = msk_moving[0][..., 0:1]  # ED, MS, ..., MD
        msk_t = msk_target[0][..., 1:2]  # target mask of each pair-wise p2p MD,ED,MS,ES,PF
        if msk_t.shape[-1] == 2:
            msk_t_p2ed = msk_t[..., :1]
            msk_t = msk_t[..., -1:]

        msk_m = msk_moved[0][..., 1:2]
        flow = flows[0]
        flow_comp_m = flows_composed_masked[0]
        flow_masked = flows_masked[0]
        fullmsk_t = fullmask_moving[0]
        fullmsk_t = from_channel_to_flat(fullmsk_t, start_c=1)[..., None]

        if not all(np.any(msk_t, axis=(1,2,3))):
            print('please check the predicted masks, some timesteps of the target mask seem to be empty!')
            print('{}'.format(np.any(msk_t, axis=(1,2,3))))

        # save all files of this patient
        if 'volume' in os.path.basename(filename):
            p = os.path.basename(filename).split('_volume')[0].lower()
        else:
            p = os.path.basename(filename).split('__')[0].lower()
        volumes = [cmr_mov,cmr_t,cmr_m,msk_mov,msk_t,msk_m,flow,flow_comp_m,flow_masked,fullmsk_t]

        suffixes = ['cmr_moving.nii.gz', 'cmr_target.nii.gz', 'cmr_moved.nii.gz',
                    'myo_moving.nii.gz', 'myo_target.nii.gz', 'myo_moved.nii.gz',
                    'flow.nii.gz', 'flow_composed.nii.gz', 'flow_masked.nii.gz',
                    'fullmask_moving.nii.gz']
        if debug:
            save_all_3d_vols_new(volumes, vol_suffixes=suffixes,
                                 EXP_PATH=pred_path, exp=p, cfg=config)

        #save_gt_and_pred(gt=msk_t, pred=msk_m, exp_path=pred_path.replace('pred',''), patient=p, cfg=config)
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
        from pathlib import Path
        exp_path = Path(args.exp).parent
        gt_path = os.path.join(exp_path, 'gt_m')
        pred_path = os.path.join(exp_path, 'pred_m')
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

        gt_path = os.path.join(args.exp, 'gt_m')
        pred_path = os.path.join(args.exp, 'pred_m')
    try:
        logging.info('start dice calculation with: {}{}{}'.format(gt_path, pred_path, args.exp))
        calc_dice(gt_path, pred_path, args.exp)
    except Exception as e:
        print('Dice calculation failed with: {}'.format(e))

    try:
        is_dmd = True
        if args.iscontrol.lower() == 'true':
            is_dmd = False
        from src_julian.data.MyMoralesAndCompositionsAHA3 import calculate_strain
        from pathlib import Path
        metadata = Path(config.get('DATA_PATH_SAX')).parent.absolute()
        exp_path = Path(config.get('EXP_PATH')).parent.absolute()
        df_patients_p2p = calculate_strain(data_root=exp_path, metadata_path=metadata,
                                           debug=False, df_style='time', p2p_style=True, isDMD=is_dmd)
        df_patients_ed2p = calculate_strain(data_root=exp_path, metadata_path=metadata,
                                            debug=False, df_style='time', p2p_style=False, isDMD=is_dmd)

        x = 0
        df_patients_p2p.to_csv(os.path.join(exp_path, 'df_DMD_time_p2p.csv'), index=False)
        df_patients_ed2p.to_csv(os.path.join(exp_path, 'df_DMD_time_ed2p.csv'), index=False)
    except Exception as e:
        print(e)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='train a phase registration model')

    # usually these two parameters should encapsulate all experiment parameters
    parser.add_argument('-exp', action='store', default=None)
    parser.add_argument('-data', action='store', default=None)
    parser.add_argument('-iscontrol', choices=['True','False','true','false'],action='store', default='false')

    results = parser.parse_args()
    print('given parameters: {}'.format(results))


    try:
        main(results)
    except Exception as e:
        print(e)
    exit()