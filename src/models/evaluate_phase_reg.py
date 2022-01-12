import glob
import numpy as np
import os
import pandas as pd
def evaluate(exp_root, result_df='results.xlsx', pred_suffix = 'pred'):
    """
    Evaluate a cross-validation
    Expect to have predicted numpy files within each fold sub-dir
    Parameters
    ----------
    exp_root : (string) path to one experiment root, above the fold_n sub-folders
    result_df : (string) name of the results excel filename
    pred_suffix :

    Returns
    -------

    """
    print('eval: {}'.format(exp_root))
    from src.utils.Metrics import meandiff
    phases = ['ED', 'MS', 'ES', 'PF', 'MD']

    pred_wildcard = '{}/{}/gtpred*.npy'.format(exp_root, pred_suffix)
    print('path to predictions: {}'.format(pred_wildcard))
    all_pred_files = glob.glob(pred_wildcard)
    """if len(all_pred_files) == 0:
        all_pred_files = sorted(glob.glob('{}/**/**/{}/*.npy'.format(exp_root, pred_suffix), recursive=False))"""
    """if len(all_pred_files) == 0:
        all_pred_files = sorted(glob.glob('{}/{}/*.npy'.format(exp_root, pred_suffix), recursive=False))"""
    assert len(all_pred_files) >0, 'we expect any predicted files, but found: {} predictions'.format(len(all_pred_files))
    print('predictions found: {}'.format(len(all_pred_files)))

    # Load the numpy prediction files
    preds = list(map(lambda x : np.load(x), all_pred_files))
    # stack the numpy files
    preds = np.concatenate(preds, axis=1)

    # calculate the mean differences
    res = meandiff(preds[0], preds[1], apply_sum=False, apply_average=False)
    df = pd.DataFrame(res.numpy(), columns=phases)
    df.to_excel(os.path.join(exp_root, result_df))
    return df