# predict cardiac phases for a cv experiment
def predict(cfg_file, data_root='', c2l=False):
    """
    Predict on the held-out validation split
    Parameters
    ----------
    cfg_file :
    data_root :
    c2l :

    Returns
    -------

    """
    import json, logging, os
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    import tensorflow as tf
    tf.get_logger().setLevel('FATAL')
    from logging import info
    import numpy as np
    from src.data.Dataset import get_trainings_files
    from src.utils.Utils_io import Console_and_file_logger, ensure_dir
    from src.data.PhaseGenerators import PhaseRegressionGenerator_v2
    from src.models.PhaseRegModels import PhaseRegressionModel
    from ProjectRoot import change_wd_to_project_root
    change_wd_to_project_root()


    from src.utils.Tensorflow_helper import choose_gpu_by_id


    # load the experiment config
    if type(cfg_file) == type(''):
        with open(cfg_file, encoding='utf-8') as data_file:
            config = json.loads(data_file.read())
    else:
        config = cfg_file
    globals().update(config)

    # ------------------------------------------define GPU id/s to use
    GPU_IDS = config.get('GPU_IDS', '0,1')
    GPUS = choose_gpu_by_id(GPU_IDS)
    print(GPUS)
    print(tf.config.list_physical_devices('GPU'))

    EXPERIMENT = config.get('EXPERIMENT', 'UNDEFINED')
    Console_and_file_logger(EXPERIMENT, logging.INFO)
    info('Loaded config for experiment: {}'.format(EXPERIMENT))

    # Load SAX volumes
    # cluster to local data mapping
    if c2l:
        config['DATA_PATH_SAX'] = os.path.join(data_root, 'sax')
        config['DF_FOLDS'] = os.path.join(data_root, 'df_kfold.csv')
        config['DF_META'] = os.path.join(data_root, 'SAx_3D_dicomTags_phase.csv')

    x_train_sax, y_train_sax, x_val_sax, y_val_sax = get_trainings_files(data_path=config['DATA_PATH_SAX'],
                                                                         path_to_folds_df=config['DF_FOLDS'],
                                                                         fold=config['FOLD'])
    logging.info('SAX train CMR: {}, SAX train masks: {}'.format(len(x_train_sax), len(y_train_sax)))
    logging.info('SAX val CMR: {}, SAX val masks: {}'.format(len(x_val_sax), len(y_val_sax)))

    # turn off all augmentation operations while inference
    # create another config for the validation data
    # we want the prediction to run with batchsize of 1
    # otherwise we might inference only on the even number of val files
    # the mirrored strategy needs to get a single gpu instance named, otherwise batchsize=1 does not work
    val_config = config.copy()
    val_config['SHUFFLE'] = False
    val_config['AUGMENT'] = False
    val_config['AUGMENT_PHASES'] = False
    val_config['AUGMENT_TEMP'] = False
    val_config['BATCHSIZE'] = 1
    val_config['HIST_MATCHING'] = False
    val_config['GPUS'] = ['/gpu:0']
    validation_generator = PhaseRegressionGenerator_v2(x_val_sax, x_val_sax, config=val_config)

    model = PhaseRegressionModel(val_config).get_model()
    logging.info('Trying to load the model weights')
    logging.info('work dir: {}'.format(os.getcwd()))
    logging.info('model weights dir: {}'.format(os.path.join(val_config['MODEL_PATH'], 'model.h5')))
    model.load_weights(os.path.join(val_config['MODEL_PATH'], 'model.h5'))
    logging.info('loaded model weights as h5 file')

    # predict on the validation generator
    preds, moved, vects = model.predict(validation_generator)
    logging.info('Shape of the predictions: {}'.format(preds.shape))

    # get all ground truth vectors, each y is a list with [onehot,moved, zeros]
    gts = np.concatenate([y[0] for x, y in validation_generator],axis=0)
    logging.info('Shape of GT: {}'.format(gts.shape))

    pred_path = os.path.join(val_config['EXP_PATH'], 'pred')
    moved_path = os.path.join(val_config['EXP_PATH'], 'moved')
    ensure_dir(pred_path)
    ensure_dir(moved_path)
    pred_filename = os.path.join(pred_path, 'gtpred_fold{}.npy'.format(val_config['FOLD']))
    moved_filename = os.path.join(moved_path, 'moved_f{}.npy'.format(val_config['FOLD']))
    vects_filename = os.path.join(moved_path, 'vects_f{}.npy'.format(val_config['FOLD']))
    np.save(pred_filename, np.stack([gts, preds], axis=0))
    np.save(moved_filename, moved)
    np.save(vects_filename, vects)

    patients_filename = os.path.join(pred_path, 'patients.txt')
    with open(patients_filename, "a+") as f:
        _ = [f.write(str(val_config['FOLD']) + '_' + os.path.basename(elem) +'\n') for elem in x_val_sax]
    logging.info('saved as: \n{}\n{} \ndone!'.format(pred_filename, patients_filename))

if __name__ == "__main__":
    import argparse, os, sys, glob

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    parser = argparse.ArgumentParser(description='predict a phase registration model')

    # usually the exp root parameters should yield to a config, which encapsulate all experiment parameters
    parser.add_argument('-exp_root', action='store', default='/mnt/sds/sd20i001/sven/code/exp/miccai_baseline')
    parser.add_argument('-data', action='store', default='/mnt/ssd/data/gcn/02_imported_4D_unfiltered')
    parser.add_argument('-work_dir', action='store', default='/mnt/ssd/git/dynamic-cmr-models')
    parser.add_argument('-c2l', action='store_true', default=False)

    results = parser.parse_args()
    os.chdir(results.work_dir)
    sys.path.append(os.getcwd())
    print('given parameters: {}'.format(results))

    # get all cfgs - we expect to find 4 as we usually train a 4-fold cv
    # call the predict_fn for each cfg
    initial_search_pattern = 'config/config.json' # path to one experiment
    search_path = os.path.join(results.exp_root, initial_search_pattern)
    cfg_files = sorted(glob.glob(search_path))
    if len(cfg_files) == 0: # we called this script with the experiment root, search for fold sub-folders
        search_pattern = '**/config/config.json'
        search_path = os.path.join(results.exp_root, search_pattern)
        print(search_path)
        cfg_files = sorted(glob.glob(search_path))
        assert len(cfg_files) == 4, 'Expect 4 cfgs, but found {}'.format(len(cfg_files)) # avoid loading too many cfgs
    print(cfg_files)

    for cfg in cfg_files:
        try:
            predict(cfg_file=cfg, data_root=results.data, c2l=results.c2l)
        except Exception as e:
            print(e)
    exit()
