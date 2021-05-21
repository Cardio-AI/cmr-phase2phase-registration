from src.data.Dataset import save_all_3d_vols


def train_fold(config):
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
    tf.get_logger().setLevel('ERROR')
    import gc, logging, os, datetime, re
    from logging import info

    # local imports
    from src.utils.Utils_io import Console_and_file_logger, init_config, ensure_dir
    from src.utils.KerasCallbacks import get_callbacks
    from src.data.Dataset import get_trainings_files
    from src.data.Generators import PhaseWindowGenerator
    from src.models.Models import create_RegistrationModel

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
    EXP_PATH = os.path.join(EXPERIMENTS_ROOT, EXPERIMENT, timestemp)
    MODEL_PATH = os.path.join(EXP_PATH, 'model', )
    TENSORBOARD_PATH = os.path.join(EXP_PATH, 'tensorboard_logs')
    CONFIG_PATH = os.path.join(EXP_PATH, 'config')
    HISTORY_PATH = os.path.join(EXP_PATH, 'history')
    ensure_dir(MODEL_PATH)
    ensure_dir(TENSORBOARD_PATH)
    ensure_dir(CONFIG_PATH)
    ensure_dir(HISTORY_PATH)

    DATA_PATH_SAX = config.get('DATA_PATH_SAX')
    DF_FOLDS = config.get('DF_FOLDS')
    DF_META = config.get('DF_META', '/mnt/ssd/data/gcn/02_imported_4D_unfiltered/SAx_3D_dicomTags_phase')
    EPOCHS = config.get('EPOCHS', 100)

    Console_and_file_logger(path=EXP_PATH, log_lvl=logging.INFO)
    config = init_config(config=locals(), save=True)
    logging.info('Is built with tensorflow: {}'.format(tf.test.is_built_with_cuda()))
    logging.info('Visible devices:\n{}'.format(tf.config.list_physical_devices()))
    logging.info('Local devices: \n {}'.format(device_lib.list_local_devices()))

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
    DF_METADATA = df[['patient', 'ED#', 'MS#', 'ES#', 'PF#', 'MD#']]

    files_ = x_train_sax + x_val_sax
    info('Check if we find the patient ID and phase mapping for all: {} files.'.format(len(files_)))
    for x in files_:
        try:
            patient_str = re.search('-(.{8})_', x).group(1).upper()

            assert (len(patient_str) == 8), 'matched patient ID from the phase sheet has a length of: {}'.format(
                len(patient_str))
            # returns the indices in the following order: 'ED#', 'MS#', 'ES#', 'PF#', 'MD#'
            # reduce by one, as the indexes start at 0, the excel-sheet at 1
            ind = DF_METADATA[DF_METADATA.patient.str.contains(patient_str)][['ED#', 'MS#', 'ES#', 'PF#', 'MD#']]
            indices = ind.values[0].astype(int) - 1

        except Exception as e:
            logging.info(patient_str)
            logging.info(ind)
            logging.info('indices: \n{}'.format(indices))
    info('Done!')

    # instantiate the batchgenerators

    batch_generator = PhaseWindowGenerator(x_train_sax, x_train_sax, config=config)
    val_config = config.copy()
    val_config['AUGMENT'] = False
    val_config['HIST_MATCHING'] = False
    val_config['AUGMENT_TEMP'] = False
    # val_config['RESAMPLE_T'] = False # this could yield phases which does not fit into the given dim
    validation_generator = PhaseWindowGenerator(x_val_sax, x_val_sax, config=val_config)

    # get model
    model = create_RegistrationModel(config)

    # write the model summary to a txt file
    with open(os.path.join(EXP_PATH, 'model_summary.txt'), 'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

    tf.keras.utils.plot_model(
        model, show_shapes=True,
        to_file=os.path.join(EXP_PATH, 'model.png'),
        show_layer_names=True,
        rankdir='TB',
        expand_nested=True,
        dpi=96
    )

    # training
    initial_epoch = 0
    cb = get_callbacks(config, batch_generator, validation_generator)
    print('start training')
    model.fit(
        x=batch_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        callbacks=cb,
        initial_epoch=initial_epoch,
        max_queue_size=config.get('QUEUE_SIZE',12),
        verbose=1)

    # predict on a some trainings-files
    example_batch = 0
    inputs, outputs = batch_generator.__getitem__(example_batch)
    pred = model.predict(x=inputs)

    transformed, flow = pred
    info('example predictions shape')
    info(transformed.shape)
    info(flow.shape)
    # TODO: refactor
    save_all_3d_vols(inputs[0], outputs[0], flow[0], config.get('EXP_PATH'), 'example_flow_0')
    save_all_3d_vols(inputs[1], outputs[1], flow[1], config.get('EXP_PATH'), 'example_flow_1')

    # free as much memory as possible
    del batch_generator
    del validation_generator
    del model
    gc.collect()

    logging.info('Fold {} finished after {:0.3f} sec'.format(FOLD, time() - t0))
    return True


def main(args=None):
    # ------------------------------------------define logging and working directory
    # import the packages inside this function enables to train on different folds
    from ProjectRoot import change_wd_to_project_root
    change_wd_to_project_root()
    import sys, os, datetime
    sys.path.append(os.getcwd())
    from src.utils.Tensorflow_helper import choose_gpu_by_id
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

        config['EXP_PATH'] = os.path.join(EXPERIMENTS_ROOT, EXPERIMENT, timestemp)
        config['MODEL_PATH'] = os.path.join(config['EXP_PATH'], 'model', )
        config['TENSORBOARD_PATH'] = os.path.join(config['EXP_PATH'], 'tensorboard_logs')
        config['CONFIG_PATH'] = os.path.join(config['EXP_PATH'], 'config')
        config['HISTORY_PATH'] = os.path.join(config['EXP_PATH'], 'history')
        # Console_and_file_logger(path=config['EXP_PATH'])

        if args.data:  # if we specified a different data path (training from workspace or node temporal disk)
            config['DATA_PATH_SAX'] = os.path.join(args.data, "sax/")
            config['DF_FOLDS'] = os.path.join(args.data, "df_kfold.csv")
            config['DF_META'] = os.path.join(args.data, "SAx_3D_dicomTags_phase")
        # we dont need to initialise this config, as it should already have the correct formatings,
        # The fold configs will be saved withn each fold run
        # config = init_config(config=config, save=False)
        print(config)
    else:
        print('no config given, build a new one')

        EXPERIMENT = args.exp
        timestemp = str(datetime.datetime.now().strftime(
            "%Y-%m-%d_%H_%M"))  # ad a timestep to each project to make repeated experiments unique

        EXP_PATH = os.path.join(EXPERIMENTS_ROOT, EXPERIMENT, timestemp)
        MODEL_PATH = os.path.join(EXP_PATH, 'model', )
        TENSORBOARD_PATH = os.path.join(EXP_PATH, 'tensorboard_logs')
        CONFIG_PATH = os.path.join(EXP_PATH, 'config')
        HISTORY_PATH = os.path.join(EXP_PATH, 'history')
        Console_and_file_logger(path=EXP_PATH)

        # define the input data paths and fold
        # first to the 4D Nrrd files,
        # second to a dataframe with a mapping of the Fold-number
        # Finally the path to the metadata
        DATA_PATH_SAX = args.sax
        DF_FOLDS = args.folds
        DF_META = args.meta
        FOLD = 0
        FOLDS = [0, 1, 2, 3]

        # General params
        SEED = 42  # define a seed for the generator shuffle
        BATCHSIZE = 8  # 32, 64, 24, 16, 1 for 3D use: 4
        GENERATOR_WORKER = BATCHSIZE  # if not set, use batchsize
        EPOCHS = 100

        DIM = [8, 64, 64]  # network input shape for spacing of 3, (z,y,x)
        T_SHAPE = 36
        T_SPACING = 55
        SPACING = [8, 3, 3]  # if resample, resample to this spacing, (z,y,x)

        # Model params
        ADD_BILSTM = args.add_lstm
        BILSTM_UNITS = args.lstm_units
        DEPTH = args.depth  # depth of the encoder
        FILTERS = args.filters  # initial number of filters, will be doubled after each downsampling block
        M_POOL = [1, 2, 2]  # size of max-pooling used for downsampling and upsampling
        F_SIZE = [3, 3, 3]  # conv filter size
        BN_FIRST = False  # decide if batch normalisation between conv and activation or afterwards
        BATCH_NORMALISATION = True  # apply BN or not
        PAD = 'same'  # padding strategy of the conv layers
        KERNEL_INIT = 'he_normal'  # conv weight initialisation
        OPTIMIZER = 'adam'  # Adam, Adagrad, RMSprop, Adadelta,  # https://keras.io/optimizers/
        ACTIVATION = 'relu'  # tf.keras.layers.LeakyReLU(), relu or any other non linear activation function
        LEARNING_RATE = 1e-4  # start with a huge lr to converge fast
        REDUCE_LR_ON_PLATEAU_PATIENCE = 5
        DECAY_FACTOR = 0.7  # Define a learning rate decay for the ReduceLROnPlateau callback
        POLY_LR_DECAY = False
        MIN_LR = 1e-12  # minimal lr, smaller lr does not improve the model
        DROPOUT_MIN = 0.4  # lower dropout at the shallow layers
        DROPOUT_MAX = 0.5  # higher dropout at the deep layers

        # Callback params
        MONITOR_FUNCTION = 'loss'
        MONITOR_MODE = 'min'
        SAVE_MODEL_FUNCTION = 'loss'
        SAVE_MODEL_MODE = 'min'
        MODEL_PATIENCE = 20
        SAVE_LEARNING_PROGRESS_AS_TF = True

        # Generator and Augmentation params
        BORDER_MODE = cv2.BORDER_REFLECT_101  # border mode for the data generation
        IMG_INTERPOLATION = cv2.INTER_LINEAR  # image interpolation in the genarator
        MSK_INTERPOLATION = cv2.INTER_NEAREST  # mask interpolation in the generator
        AUGMENT = args.aug  # a compose of 2D augmentation (grid distortion, 90degree rotation, brightness and shift)
        AUGMENT_PROB = 0.8
        AUGMENT_PHASES = args.paug
        AUGMENT_PHASES_RANGE = args.prange
        AUGMENT_TEMP = args.taug
        AUGMENT_TEMP_RANGE = args.trange
        REPEAT_ONEHOT = True
        SHUFFLE = True
        RESAMPLE = args.resample
        RESAMPLE_T = args.tresample
        HIST_MATCHING = args.hmatch
        SCALER = 'MinMax'  # MinMax, Standard or Robust
        # We define 5 target phases and a background phase for the pad/empty volumes
        PHASES = len(['ED#', 'MS#', 'ES#', 'PF#', 'MD#'])  # skipped 'pad backround manually added', due to repeating
        TARGET_SMOOTHING = True
        SMOOTHING_WEIGHT_CORRECT = args.gausweight
        GAUS_SIGMA = args.gaussigma

        print('init config')
        config = init_config(config=locals(), save=False)

    GPU_IDS = '0,1'
    GPUS = choose_gpu_by_id(GPU_IDS)
    print(GPUS)

    for f in config.get('FOLDS', [0]):
        print('starting fold: {}'.format(f))
        config_ = config.copy()
        config_['FOLD'] = f
        train_fold(config_)
        print('train fold: {}'.format(f))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='train a phase registration model')

    # usually these two parameters should encapsulate all experiment parameters
    parser.add_argument('-cfg', action='store', default=None)
    parser.add_argument('-data', action='store', default=None)

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
