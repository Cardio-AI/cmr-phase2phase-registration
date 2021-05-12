


def main(cfg_file, data_root,c2l=False):

    import json, logging, os
    from logging import info
    import numpy as np
    from src.data.Dataset import get_trainings_files
    from src.utils.Utils_io import Console_and_file_logger, ensure_dir
    from src.data.Generators import PhaseRegressionGenerator
    from src.models.Models import create_PhaseRegressionModel

    # load the experiment config
    with open(cfg_file, encoding='utf-8') as data_file:
        config = json.loads(data_file.read())
    globals().update(config)

    EXPERIMENT = config.get('EXPERIMENT', 'UNDEFINED')
    Console_and_file_logger(EXPERIMENT, logging.INFO)
    info('Loaded config for experiment: {}'.format(EXPERIMENT))

    # Load SAX volumes
    # cluster to local data mapping
    if c2l:
        config['DATA_PATH_SAX'] = os.path.join(data_root,'/sax')
        config['DF_FOLDS'] = os.path.join(data_root,'df_kfold.csv')
        config['DF_META'] = os.path.join(data_root,'SAx_3D_dicomTags_phase')
    x_train_sax, y_train_sax, x_val_sax, y_val_sax = get_trainings_files(data_path=config['DATA_PATH_SAX'],
                                                                         path_to_folds_df=config['DF_FOLDS'],
                                                                         fold=config['FOLD'])
    logging.info('SAX train CMR: {}, SAX train masks: {}'.format(len(x_train_sax), len(y_train_sax)))
    logging.info('SAX val CMR: {}, SAX val masks: {}'.format(len(x_val_sax), len(y_val_sax)))

    config['SHUFFLE'] = False
    config['AUGMENT'] = False
    config['AUGMENT_PHASES'] = False
    config['AUGMENT_TEMP'] = False
    config['BATCHSIZE'] = 1
    batch_generator = PhaseRegressionGenerator(x_train_sax, x_train_sax, config=config)
    # create another config for the validation data, for the case of different evaluation
    val_config = config.copy()
    validation_generator = PhaseRegressionGenerator(x_val_sax, x_val_sax, config=val_config)

    model = create_PhaseRegressionModel(config)
    model.load_weights(os.path.join(config['MODEL_PATH'], 'model.h5'))
    logging.info('loaded model weights as h5 file')

    # predict on the validation generator
    preds = model.predict(validation_generator)
    logging.info(preds.shape)

    # get all ground truth vectors
    gts = np.stack([np.squeeze(y) for x, y in validation_generator])
    logging.info(gts.shape)

    pred_path = os.path.join(config['EXP_PATH'], 'pred')
    ensure_dir(pred_path)
    pred_filename = os.path.join(pred_path, 'gtpred_fold{}.npy'.format(config['FOLD']))
    np.save(pred_filename, np.stack([gts, preds], axis=0))
    logging.info('saved as: \n{} \ndone!'.format(pred_filename))



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='predict a phase registration model')

    # usually these two parameters should encapsulate all experiment parameters
    parser.add_argument('-exp_root', action='store', default='/mnt/ssd/git/dynamic-cmr-models/exp/local/miccai_baseline')
    parser.add_argument('-data', action='store', default='/mnt/ssd/data/gcn/02_imported_4D_unfiltered')
    parser.add_argument('-c2l', action='store_true', default=False)

    results = parser.parse_args()
    print('given parameters: {}'.format(results))

    # get all cfgs
    # call main for each cfg
    import glob, os
    search_pattern = '**/**/config/config.json'

    cfg_files = sorted(glob.glob(os.path.join(results.exp_root, search_pattern)))
    print(cfg_files)
    assert len(cfg_files) >= 1, 'No cfgs found'
    for cfg in cfg_files:
        try:
            main(cfg_file=cfg, data_root=results.data, c2l=results.c2l)
        except Exception as e:
            print(e)
    exit()