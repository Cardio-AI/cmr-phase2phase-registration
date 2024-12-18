def calc_dice(gt_path, pred_path, export_path):
    import os
    import glob
    import os.path as p
    from medpy.metric.binary import hd, dc
    import SimpleITK as sitk
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    results_df = pd.DataFrame(columns=['patient', 'phase', 'dice', 'hd'])
    gts = sorted(glob.glob(p.join(gt_path, '*')))
    preds = sorted(glob.glob(p.join(pred_path, '*')))
    results_file = p.join(export_path, 'results.csv')
    assert len(gts) == len(preds), 'len(gt) {} != len(pred) {}'.format(len(gts), len(preds))

    results_df['patient'] = list(map(lambda x: ''.join(os.path.basename(x).split('_')[0:2]), gts))
    results_df['phase'] = list(map(lambda x: ''.join(os.path.basename(x).split('_')[-1][:2]), gts))
    gt_files = list(map(lambda x : sitk.GetArrayFromImage(sitk.ReadImage(x)), gts))
    pred_files = list(map(lambda x : sitk.GetArrayFromImage(sitk.ReadImage(x)), preds))
    zipped_files = zip(gt_files, pred_files)
    results_df['dice'] = list(
        map(lambda x: dc(x[0], x[1]),
            zipped_files))
    zipped_files = zip(gt_files, pred_files)
    results_df['hd'] = list(
        map(lambda x: hd(x[0], x[1]),
            zipped_files))
    results_df.to_csv(results_file, index=False)
    g = sns.violinplot(x='phase', y='dice', data=results_df[['patient', 'phase', 'dice']])
    g.set(ylim=(0,1))
    plt.savefig(os.path.join(export_path,'dice.png'))
    plt.clf()
    g = sns.violinplot(x='phase', y='hd', data=results_df[['patient', 'phase', 'hd']])
    g.set(ylim=(0, 25))
    plt.savefig(os.path.join(export_path, 'hd.png'))
    print('Evaluation done: \n{}'.format(export_path))


if __name__ == "__main__":
    import os
    exp_path = '/mnt/ssd/git/dynamic-cmr-models/exp/temp/phase2phase/v3/64_128_128/window1/reg0_001/dmd/NOmask_inkl_img_f0/2021-09-30_14_17'
    gt_path = os.path.join(exp_path, 'gt_m')
    pred_path = os.path.join(exp_path, 'pred_m')
    export_path = exp_path
    calc_dice(gt_path, pred_path, exp_path)