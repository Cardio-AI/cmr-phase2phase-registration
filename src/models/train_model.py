# this should import glob, os, and many other standard libs



def train_fold(config):
    # make sure all neccessary params in config are set
    # if not set them with default values
    from src.utils.Tensorflow_helper import choose_gpu_by_id
    # ------------------------------------------define GPU id/s to use
    GPU_IDS = '0,1'
    GPUS = choose_gpu_by_id(GPU_IDS)
    print(GPUS)
    # ------------------------------------------ import helpers
    # this should import glob, os, and many other standard libs
    from tensorflow.python.client import device_lib
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')


    import gc
    import logging
    from logging import info
    import os
    import glob
    # local imports
    from src.utils.Utils_io import Console_and_file_logger, init_config
    from src.utils.KerasCallbacks import get_callbacks
    from src.data.Dataset import get_trainings_files

    # import external libs
    import pandas as pd
    from time import time



    config = init_config(config=config, save=True)
    globals().update(config)
    EXPERIMENT = config.get('EXPERIMENT')

    EXP_PATH = config.get('EXP_PATH')
    DATA_PATH_SAX = config.get('DATA_PATH_SAX')
    DF_FOLDS = config.get('DF_FOLDS')
    FOLD = config.get('FOLD')
    DF_META = config.get('DF_META', '/mnt/ssd/data/gcn/02_imported_4D_unfiltered/SAx_3D_dicomTags_phase')
    EPOCHS = config.get('EPOCHS')

    Console_and_file_logger(EXPERIMENT, logging.INFO)
    config = init_config(config=config, save=True)
    print(config)
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
    import re

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
    # logging.getLogger().setLevel(logging.INFO)
    from src.data.Generators import PhaseRegressionGenerator
    # config['SHUFFLE'] = False
    # config['AUGMENT'] = False
    # config['RESAMPLE'] = True
    # config['AUGMENT_PHASES'] = False
    batch_generator = PhaseRegressionGenerator(x_train_sax, x_train_sax, config=config)
    val_config = config.copy()
    val_config['AUGMENT'] = False
    val_config['AUGMENT_PHASES'] = False
    validation_generator = PhaseRegressionGenerator(x_val_sax, x_val_sax, config=val_config)

    # get model
    from src.models.Models import create_PhaseRegressionModel
    model = create_PhaseRegressionModel(config)

    # write the model summary to a txt file
    # Open the file

    with open(os.path.join(EXP_PATH, 'model_summary.txt'), 'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

    tf.keras.utils.plot_model(
        model, show_shapes=False,
        to_file=os.path.join(EXP_PATH, 'model.png'),
        show_layer_names=True,
        rankdir='TB',
        expand_nested=True, dpi=96
    )

    # training
    initial_epoch = 0

    model.fit(
        x=batch_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        callbacks=get_callbacks(config, batch_generator, validation_generator),
        initial_epoch=initial_epoch,
        max_queue_size=12,
        use_multiprocessing=False,
        verbose=1)

    # free as much memory as possible
    del batch_generator
    del validation_generator
    del model
    gc.collect()

    logging.info('Fold {} finished after {:0.3f} sec'.format(FOLD, time() - t0))
    return True
