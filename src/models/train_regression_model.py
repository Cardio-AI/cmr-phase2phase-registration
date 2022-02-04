


def train_fold(config, in_memory=False):
    # make sure all necessary params in config are set
    # if not set them with default values
    import tensorflow
    import tensorflow as tf
    tf.get_logger().setLevel('FATAL')

    from src.utils.Tensorflow_helper import choose_gpu_by_id
    # ------------------------------------------define GPU id/s to use
    GPU_IDS = config.get('GPU_IDS', '0,1')
    GPUS = choose_gpu_by_id(GPU_IDS)
    print(GPUS)
    print(tf.config.list_physical_devices('GPU'))
    # ------------------------------------------ import helpers
    # this should import glob, os, and many other standard libs
    #from tensorflow.python.client import device_lib
    import gc, logging, os, datetime, re
    from logging import info

    # local imports
    from src.utils.Utils_io import Console_and_file_logger, init_config, ensure_dir
    from src.utils.KerasCallbacks import get_callbacks
    from src.data.Dataset import get_trainings_files
    from src.data.Generators import PhaseRegressionGenerator_v2
    from src.models.Models import create_PhaseRegressionModel_v2
    from src.models.Models import create_PhaseRegressionModel

    # import external libs
    import pandas as pd
    from time import time

    # make all config params known to the local namespace
    locals().update(config)

    # overwrite the experiment names and paths, so that each cv gets an own sub-folder
    EXPERIMENT = config.get('EXPERIMENT')
    FOLD = config.get('FOLD')

    EXPERIMENT = '{}f{}'.format(EXPERIMENT, FOLD)
    """timestemp = str(datetime.datetime.now().strftime(
        "%Y-%m-%d_%H_%M"))"""  # add a timestep to each project to make repeated experiments unique

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
    DF_META = config.get('DF_META', '/mnt/ssd/data/gcn/02_imported_4D_unfiltered/SAx_3D_dicomTags_phase.csv')
    EPOCHS = config.get('EPOCHS')

    Console_and_file_logger(path=FOLD_PATH, log_lvl=logging.INFO)
    config = init_config(config=locals(), save=True)
    logging.info('Is built with tensorflow: {}'.format(tf.test.is_built_with_cuda()))
    logging.info('Visible devices:\n{}'.format(tf.config.list_physical_devices()))
    #logging.info('Local devices: \n {}'.format(device_lib.list_local_devices()))

    # get kfolded data from DATA_ROOT and subdirectories
    # Load SAX volumes
    x_train_sax, y_train_sax, x_val_sax, y_val_sax = get_trainings_files(data_path=DATA_PATH_SAX,
                                                                         path_to_folds_df=DF_FOLDS,
                                                                        fold=FOLD)

    """examples = 12
    x_train_sax, y_train_sax, x_val_sax, y_val_sax = x_train_sax[:examples], y_train_sax[:examples], x_val_sax[:examples], y_val_sax[:examples]"""
    #x_train_sax = [x for x in x_train_sax if 'patient060' in x] * 4
    logging.info('SAX train CMR: {}, SAX train masks: {}'.format(len(x_train_sax), len(y_train_sax)))
    logging.info('SAX val CMR: {}, SAX val masks: {}'.format(len(x_val_sax), len(y_val_sax)))

    t0 = time()
    # check if we find each patient in the corresponding dataframe

    METADATA_FILE = DF_META
    df = pd.read_csv(METADATA_FILE, dtype={'patient':str, 'ED#':int, 'MS#':int, 'ES#':int, 'PF#':int, 'MD#':int})
    DF_METADATA = df[['patient', 'ED#', 'MS#', 'ES#', 'PF#', 'MD#']]

    files_ = x_train_sax + x_val_sax
    info('Check if we find the patient ID and phase mapping for all: {} files.'.format(len(files_)))
    for x in files_:
        try:
            patient_str, ind, indices = '','',''
            patient_str = re.search('-(.{8})_', x)
            if patient_str: # GCN data
                patient_str = patient_str.group(1).upper()
                assert (len(patient_str) == 8), 'matched patient ID from the phase sheet has a length of: {}, expected a length of 8 for GCN data'.format(
                    len(patient_str))
            else: # DMD data
                patient_str = os.path.basename(x).split('_volume')[0].lower()

            if 'nii.gz' in patient_str:  # ACDC files e.g.: patient001_4d.nii.gz
                patient_str = re.search('patient(.{3})_', x)
                patient_str = patient_str.group(1).upper()

            assert len(
                patient_str) > 0, 'empty patient id found, please check the get_patient_id in fn train_fold(), usually there are path problems'
            # returns the indices in the following order: 'ED#', 'MS#', 'ES#', 'PF#', 'MD#'
            # reduce by one, as the indexes start at 0, the excel-sheet at 1
            ind = DF_METADATA[DF_METADATA.patient.str.contains(patient_str)][['ED#', 'MS#', 'ES#', 'PF#', 'MD#']]
            indices = ind.values[0].astype(int) - 1

        except Exception as e:
            info(e)
            logging.info(patient_str)
            logging.info(ind)
            logging.info('indices: \n{}'.format(indices))
    info('Done!')

    # instantiate the batchgenerators
    batch_generator = PhaseRegressionGenerator_v2(x_train_sax, x_train_sax, config=config, in_memory=in_memory)
    val_config = config.copy()
    val_config['AUGMENT'] = False
    val_config['AUGMENT_PHASES'] = False
    val_config['HIST_MATCHING'] = False
    val_config['AUGMENT_TEMP'] = False
    validation_generator = PhaseRegressionGenerator_v2(x_val_sax, x_val_sax, config=val_config, in_memory=in_memory)

    # get model
    model = create_PhaseRegressionModel_v2(config)

    # write the model summary to a txt file
    with open(os.path.join(FOLD_PATH, 'model_summary.txt'), 'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(line_length=140, print_fn=lambda x: fh.write(x + '\n'))

    tf.keras.utils.plot_model(
        model, show_shapes=False,
        to_file=os.path.join(FOLD_PATH, 'model.png'),
        show_layer_names=True,
        rankdir='TB',
        expand_nested=False,
        dpi=96
    )

    # training
    initial_epoch = 0
    model.fit(
        x=batch_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        callbacks=get_callbacks(config, batch_generator, validation_generator),
        initial_epoch=initial_epoch,
        #max_queue_size=config.get('QUEUE_SIZE',2),
        # use_multiprocessing=False,
        # workers=12,
        verbose=1)

    # free as much memory as possible
    del batch_generator
    del validation_generator
    del model
    gc.collect()

    from src.models.predict_phase_reg_model import predict
    predict(config)

    logging.info('Fold {} finished after {:0.3f} sec'.format(FOLD, time() - t0))
    return True


def main(args=None, in_memory=False):
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    # ------------------------------------------define logging and working directory
    # import the packages inside this function enables to train on different folds
    from ProjectRoot import change_wd_to_project_root
    change_wd_to_project_root()
    import sys, os, datetime
    sys.path.append(os.getcwd())

    # local imports
    # import external libs
    import tensorflow as tf
    tf.get_logger().setLevel('FATAL')
    import cv2
    from src.utils.Utils_io import Console_and_file_logger, init_config

    EXPERIMENTS_ROOT = 'exp/'

    if args.cfg:
        import json
        cfg = args.cfg
        print('config given: {}'.format(cfg))
        # load the experiment config
        with open(cfg, encoding='utf-8') as data_file:
            config = json.loads(data_file.read())

        # Define new paths, so that we make sure that:
        # 1. we dont overwrite a previous config
        # 2. cluster based trainings are compatible with saving locally (cluster/local)
        # we dont need to initialise this config, as it should already have the correct formatings,
        # The fold configs will be saved within each fold run
        # add a timestep to each project to make repeated experiments unique
        EXPERIMENT = config.get('EXPERIMENT', 'UNDEFINED')
        timestemp = str(datetime.datetime.now().strftime("%Y-%m-%d_%H_%M"))

        config['EXP_PATH'] = os.path.join(EXPERIMENTS_ROOT, EXPERIMENT, timestemp)

        if args.data:  # if we specified a different data path (training from workspace or node temporal disk)
            config['DATA_PATH_SAX'] = os.path.join(args.data, "sax/")
            config['DF_FOLDS'] = os.path.join(args.data, "df_kfold.csv")
            config['DF_META'] = os.path.join(args.data, "SAx_3D_dicomTags_phase.csv")
        print(config)
    else:
        print('no config given, please select one from the  templates in exp/examples')



    for f in config.get('FOLDS', [0]):
        print('starting fold: {}'.format(f))
        config_ = config.copy()
        config_['FOLD'] = f
        train_fold(config_, in_memory=in_memory)
        print('train fold: {} finished'.format(f))

    from src.models.evaluate_phase_reg import evaluate

    evaluate(config.get('EXP_PATH'))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='train a phase registration model')

    # usually these two parameters should encapsulate all experiment parameters
    parser.add_argument('-cfg', action='store', default=None)
    parser.add_argument('-data', action='store', default=None)
    parser.add_argument('-inmemory', action='store', default=False) # enable in memory pre-processing on the cluster

    # anyway, there are cases were we want to define some specific parameters, a better choice would be to modify the config
    parser.add_argument('-sax', action='store', default='/mnt/ssd/data/gcn/02_imported_4D_unfiltered/sax/')
    parser.add_argument('-folds', action='store', default='/mnt/ssd/data/gcn/02_imported_4D_unfiltered/df_kfold.csv')
    parser.add_argument('-meta', action='store', default='/mnt/ssd/data/gcn/02_imported_4D_unfiltered/SAx_3D_dicomTags_phase.csv')
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
        import distutils.util
        in_memory = distutils.util.strtobool(results.inmemory)
        if in_memory:
            print('running in-memory={}, watch for memory overflow!'.format(in_memory))
        #main(results, in_memory=in_memory)
    except Exception as e:
        print(e)
    main(results, in_memory=in_memory)
    exit()
