import logging

import numpy as np




def train_fold(config, in_memory=False):
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
    import numpy as np
    tf.get_logger().setLevel('FATAL')
    tf.random.set_seed(config.get('SEED', 42))
    np.random.seed(config.get('SEED', 42))
    import gc, logging, os, datetime, re
    from logging import info

    # local imports
    from src.utils.Utils_io import Console_and_file_logger, init_config, ensure_dir
    from src.utils.KerasCallbacks import get_callbacks
    from src.data.Dataset import get_trainings_files
    #from src.data.Generators import PhaseWindowGenerator
    from src.models.Models import create_RegistrationModel
    from src.data.Generators import PhaseMaskWindowGenerator
    from src.models.Models import create_RegistrationModel_inkl_mask

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

    EXPERIMENTS_ROOT = 'exp/'
    EXP_PATH = config.get('EXP_PATH')
    FOLD_PATH = os.path.join(EXP_PATH, 'f{}'.format(FOLD))
    MODEL_PATH = os.path.join(FOLD_PATH, 'model', )
    TENSORBOARD_PATH = os.path.join(FOLD_PATH, 'tensorboard_logs')
    CONFIG_PATH = os.path.join(FOLD_PATH, 'config')

    ensure_dir(MODEL_PATH)
    ensure_dir(TENSORBOARD_PATH)
    ensure_dir(CONFIG_PATH)

    DATA_PATH_SAX = config.get('DATA_PATH_SAX')
    DF_FOLDS = config.get('DF_FOLDS')
    DF_META = config.get('DF_META', '/mnt/ssd/data/gcn/02_imported_4D_unfiltered/SAx_3D_dicomTags_phase')
    EPOCHS = config.get('EPOCHS', 100)

    Console_and_file_logger(path=EXP_PATH, log_lvl=logging.INFO)
    config = init_config(config=locals(), save=True)
    """logging.info('Is built with tensorflow: {}'.format(tf.test.is_built_with_cuda()))
    logging.info('Visible devices:\n{}'.format(tf.config.list_physical_devices()))
    logging.info('Local devices: \n {}'.format(device_lib.list_local_devices()))"""

    # get kfolded data from DATA_ROOT and subdirectories
    # Load SAX volumes
    x_train_sax, y_train_sax, x_val_sax, y_val_sax = get_trainings_files(data_path=DATA_PATH_SAX,
                                                                         path_to_folds_df=DF_FOLDS,
                                                                         fold=FOLD)
    logging.info('SAX train CMR: {}, SAX train masks: {}'.format(len(x_train_sax), len(y_train_sax)))
    logging.info('SAX val CMR: {}, SAX val masks: {}'.format(len(x_val_sax), len(y_val_sax)))

    t0 = time()
    # check if we find each patient in the corresponding dataframe
    METADATA_FILE = DF_META
    df = pd.read_csv(METADATA_FILE)
    df.columns = df.columns.str.lower()
    DF_METADATA = df[['patient', 'ed#', 'ms#', 'es#', 'pf#', 'md#']].copy()
    DF_METADATA[['ed#', 'ms#', 'es#', 'pf#', 'md#']] = DF_METADATA[['ed#', 'ms#', 'es#', 'pf#', 'md#']].astype('int').copy()
    DF_METADATA.columns = DF_METADATA.columns.str.lower()
    DF_METADATA['patient'] = DF_METADATA.loc[:,'patient'].str.lower()

    files_ = x_train_sax + x_val_sax
    info('Check if we find the patient ID and phase mapping for all: {} files.'.format(len(files_)))

    check_if_patients_in_metadata_file(DF_METADATA, config, files_)

    # instantiate the batch generators
    """n = 10
    x_train_sax = x_train_sax[:n]
    x_val_sax = x_val_sax[:n]"""
    config['ISTRAINING'] = False
    batch_generator = PhaseMaskWindowGenerator(x_train_sax, y_train_sax, config=config, in_memory=in_memory)
    val_config = config.copy()
    val_config['AUGMENT'] = False
    val_config['HIST_MATCHING'] = False
    val_config['AUGMENT_TEMP'] = False
    val_config['ISTRAINING'] = False
    # val_config['RESAMPLE_T'] = False # this could yield phases which does not fit into the given dim
    validation_generator = PhaseMaskWindowGenerator(x_val_sax, y_val_sax, config=val_config, in_memory=in_memory)

    # get model
    model = create_RegistrationModel_inkl_mask(config)

    # write the model summary to a txt file
    with open(os.path.join(EXP_PATH, 'model_summary.txt'), 'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(line_length=150,print_fn=lambda x: fh.write(x + '\n'))

    tf.keras.utils.plot_model(
        model, show_shapes=True,
        to_file=os.path.join(EXP_PATH, 'model.png'),
        show_layer_names=True,
        rankdir='TB',
        expand_nested=False,
        dpi=96
    )

    # training
    initial_epoch = 0
    cb = get_callbacks(config, batch_generator, validation_generator)
    print('start training')
    #EPOCHS = 1
    model.fit(
        x=batch_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        callbacks=cb,
        initial_epoch=initial_epoch,
        max_queue_size=config.get('QUEUE_SIZE',6),
        verbose=2)


    try:
        # free as much memory as possible
        del batch_generator
        del validation_generator
        del model
        del cb
        gc.collect()

    except Exception as e:
        logging.error(e)

    logging.info('Fold {} finished after {:0.3f} sec'.format(FOLD, time() - t0))
    return config


def check_if_patients_in_metadata_file(DF_METADATA, config, files_):
    ISDMD = config.get('ISDMDDATA')
    info('ISDMD: {}'.format(ISDMD))
    for x in files_:
        try:
            if ISDMD:
                patient_str = os.path.basename(x).split('_volume')[0].lower()
                assert len(
                    patient_str) > 0, 'empty patient id found, please check the get_patient_id in fn train_fold()'
            else:
                patient_str = re.search('-(.{8})_', x).group(1).lower()
                assert (len(patient_str) == 8), 'matched patient ID from the phase sheet has a length of: {}'.format(
                    len(patient_str))
            # returns the indices in the following order: 'ED#', 'MS#', 'ES#', 'PF#', 'MD#'
            # reduce by one, as the indexes start at 0, the excel-sheet at 1
            ind = DF_METADATA[DF_METADATA.patient.str.contains(patient_str)][['ed#', 'ms#', 'es#', 'pf#', 'md#']]
            # for the original dmd ind we need to reduce the idx by one, no modulo necessary as there are no idx with 0
            indices = ind.values[0].astype(int)  # - 1

        except Exception as e:
            logging.info(patient_str)
            logging.info(ind)
            logging.info('indices: \n{}'.format(indices))
    info('Check Done!')


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

    EXPERIMENTS_ROOT = 'exp/'

    if args.cfg:
        import json
        cfg = args.cfg
        print('config given: {}'.format(cfg))
        # load the experiment config
        with open(cfg, encoding='utf-8') as data_file:
            config = json.loads(data_file.read())

        # if config given, define new paths, so that we make sure that:
        # 1. we dont overwrite a previous config
        # 2. we store the experiment in the current source directory (cluster/local)
        EXPERIMENT = config.get('EXPERIMENT', 'UNDEFINED')
        timestemp = str(datetime.datetime.now().strftime(
            "%Y-%m-%d_%H_%M"))  # ad a timestep to each project to make repeated experiments unique
        if args.jobid != None: timestemp += "_{}".format(args.jobid)
        config['EXP_PATH'] = os.path.join(EXPERIMENTS_ROOT, EXPERIMENT, timestemp)
        config['MODEL_PATH'] = os.path.join(config['EXP_PATH'], 'model', )
        config['TENSORBOARD_PATH'] = os.path.join(config['EXP_PATH'], 'tensorboard_logs')
        config['CONFIG_PATH'] = os.path.join(config['EXP_PATH'], 'config')
        config['HISTORY_PATH'] = os.path.join(config['EXP_PATH'], 'history')
        # Console_and_file_logger(path=config['EXP_PATH'])

        if args.data:  # if we specified a different data path (training from workspace or local node disk)
            config['DATA_PATH_SAX'] = os.path.join(args.data, "sax/")
            config['DF_FOLDS'] = os.path.join(args.data, "df_kfold.csv")
            config['DF_META'] = os.path.join(args.data, "SAx_3D_dicomTags_phase.csv")
        # we dont need to initialise this config, as it should already have the correct format,
        # The fold configs will be saved with each fold run
        # config = init_config(config=config, save=False)
        print(config)
    else:
        print('no config given, build a new one')
        raise NotImplementedError('Please specify a valid config!')

    import os
    from src.models.Evaluate_moved_myo import calc_dice
    from src.models.predict_motion_model import pred_fold

    for f in config.get('FOLDS', [0]):
        print('starting fold: {}'.format(f))
        config_ = config.copy()
        config_['FOLD'] = f
        cfg = train_fold(config_, in_memory=args.inmemory.lower()=='true')
        logging.info('start pred_fold with exp path: {}'.format(cfg.get('EXP_PATH', '')))
        pred_fold(cfg)
        exp_path = cfg.get('EXP_PATH')

    gt_path = os.path.join(exp_path, 'gt_m')
    pred_path = os.path.join(exp_path, 'pred_m')
    try:
        logging.info('start dice calculation with: {}{}{}'.format(gt_path,pred_path,exp_path))
        calc_dice(gt_path, pred_path, exp_path)
    except Exception as e:
        print('Dice calculation failed with: {}'.format(e))
    ### integration of the strain calculation
    try:
        from src_julian.data.MyMoralesAndCompositionsAHA3 import calculate_strain
        metadata = cfg.get('DATA_PATH_SAX').replace('sax','')
        logging.info('start P2P strain calculation with metadata: {}'.format(metadata))
        df_patients_p2p = calculate_strain(data_root=exp_path, metadata_path=metadata,
                                           debug=False, df_style='time', p2p_style=True, isDMD=True)
        logging.info('start P2ED strain calculation with metadata: {}'.format(metadata))
        df_patients_ed2p = calculate_strain(data_root=exp_path, metadata_path=metadata,
                                            debug=False, df_style='time', p2p_style=False, isDMD=True)

        x = 0
        logging.info('Writing Strain to: {}'.format(exp_path))
        df_patients_p2p.to_csv(os.path.join(exp_path, 'df_DMD_time_p2p.csv'), index=False)
        df_patients_ed2p.to_csv(os.path.join(exp_path, 'df_DMD_time_ed2p.csv'), index=False)
    except Exception as e:
        print('strain calculation failed: {}'.format(e))
    print('train fold: {}'.format(f))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='train a phase registration model')

    # usually these two parameters should encapsulate all experiment parameters
    parser.add_argument('-cfg', action='store', default=None)
    parser.add_argument('-data', action='store', default=None)
    parser.add_argument('-inmemory', action='store', default=None)
    parser.add_argument('-jobid', action='store', default=None)

    #
    parser.add_argument('-sax', action='store', default='/mnt/ssd/data/gcn/02_imported_4D_unfiltered/sax/')
    parser.add_argument('-folds', action='store', default='/mnt/ssd/data/gcn/02_imported_4D_unfiltered/df_kfold.csv')
    parser.add_argument('-meta', action='store',
                        default='/mnt/ssd/data/gcn/02_imported_4D_unfiltered/SAx_3D_dicomTags_phase')
    parser.add_argument('-exp', action='store', default='temp_exp')
    parser.add_argument('-add_lstm', action='store_true', default=False)
    parser.add_argument('-lstm_units', action='store', default=64, type=int)
    parser.add_argument('-depth', action='store', default=4, type=int)
    parser.add_argument('-filters', action='store', default=20, type=int)

    parser.add_argument('-aug', action='store', default=True)
    parser.add_argument('-paug', action='store', default=False)
    parser.add_argument('-prange', action='store', default=2, type=int)
    parser.add_argument('-taug', action='store', default=False)
    parser.add_argument('-trange', action='store', default=2, type=int)
    parser.add_argument('-resample', action='store', default=True)
    parser.add_argument('-tresample', action='store', default=False)
    parser.add_argument('-hmatch', action='store', default=False)
    parser.add_argument('-gausweight', action='store', default=20, type=int)
    parser.add_argument('-gaussigma', action='store', default=1, type=int)

    results = parser.parse_args()
    print('given parameters: {}'.format(results))

    try:
        main(results)
    except Exception as e:
        print(e)
    exit()