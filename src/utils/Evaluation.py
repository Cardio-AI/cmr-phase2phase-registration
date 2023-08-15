import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import scipy as scp
from scipy import stats
import pingouin as pg

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


import sklearn
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.datasets import make_classification
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, cross_validate, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import (precision_recall_curve,
                                 PrecisionRecallDisplay)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer
from scipy import stats
from xgboost import XGBClassifier


def get_pvals_uncorrected(df_DMD, target='lge', paired=False):
    """
    Derive uncorrected per patient, phase and segment T-Test p-values between two different groups in a pd.DataFrame
    Parameters
    ----------
    df_DMD : (pandas.DataFrame) -->either keyframe2keyframe or composed Strain results from calc_strain()
    alpha0 : (float) Significant threshold
    target : (str) group either by lge+/lge- or by dmd vs control
    paired : (bool)

    Returns (pandas.DataFrame) with a shape of 10 x 16 (5xRS + 5xCS = 10) (16 AHA segments)
    -------

    """
    # define results array
    n_tests = 10  # is the number of columns
    n_aha = 16
    pvalue_error = 999  # a value that is written when the pvalue is None
    dec_p = 5
    results_pvalues = np.ndarray((n_aha, n_tests))
    results_cintervals = np.ndarray((n_aha, n_tests), dtype=object)

    for i in range(n_tests):
        for aha in range(1, n_aha + 1):  # 1-16

            # our dmd splitted by LGE
            # our dmd splitted by lge: RS
            our_dmd_p2p0_rs_lgeneg = df_DMD[(df_DMD[target] == 0) & (df_DMD['aha'] == aha) & (df_DMD['phase'] == 0)][
                'our_rs']
            our_dmd_p2p1_rs_lgeneg = df_DMD[(df_DMD[target] == 0) & (df_DMD['aha'] == aha) & (df_DMD['phase'] == 1)][
                'our_rs']
            our_dmd_p2p2_rs_lgeneg = df_DMD[(df_DMD[target] == 0) & (df_DMD['aha'] == aha) & (df_DMD['phase'] == 2)][
                'our_rs']
            our_dmd_p2p3_rs_lgeneg = df_DMD[(df_DMD[target] == 0) & (df_DMD['aha'] == aha) & (df_DMD['phase'] == 3)][
                'our_rs']
            our_dmd_p2p4_rs_lgeneg = df_DMD[(df_DMD[target] == 0) & (df_DMD['aha'] == aha) & (df_DMD['phase'] == 4)][
                'our_rs']
            our_dmd_p2p0_rs_lgepos = df_DMD[(df_DMD[target] == 1) & (df_DMD['aha'] == aha) & (df_DMD['phase'] == 0)][
                'our_rs']
            our_dmd_p2p1_rs_lgepos = df_DMD[(df_DMD[target] == 1) & (df_DMD['aha'] == aha) & (df_DMD['phase'] == 1)][
                'our_rs']
            our_dmd_p2p2_rs_lgepos = df_DMD[(df_DMD[target] == 1) & (df_DMD['aha'] == aha) & (df_DMD['phase'] == 2)][
                'our_rs']
            our_dmd_p2p3_rs_lgepos = df_DMD[(df_DMD[target] == 1) & (df_DMD['aha'] == aha) & (df_DMD['phase'] == 3)][
                'our_rs']
            our_dmd_p2p4_rs_lgepos = df_DMD[(df_DMD[target] == 1) & (df_DMD['aha'] == aha) & (df_DMD['phase'] == 4)][
                'our_rs']

            # our dmd splitted by lge: CS
            our_dmd_p2p0_cs_lgeneg = df_DMD[(df_DMD[target] == 0) & (df_DMD['aha'] == aha) & (df_DMD['phase'] == 0)][
                'our_cs']
            our_dmd_p2p1_cs_lgeneg = df_DMD[(df_DMD[target] == 0) & (df_DMD['aha'] == aha) & (df_DMD['phase'] == 1)][
                'our_cs']
            our_dmd_p2p2_cs_lgeneg = df_DMD[(df_DMD[target] == 0) & (df_DMD['aha'] == aha) & (df_DMD['phase'] == 2)][
                'our_cs']
            our_dmd_p2p3_cs_lgeneg = df_DMD[(df_DMD[target] == 0) & (df_DMD['aha'] == aha) & (df_DMD['phase'] == 3)][
                'our_cs']
            our_dmd_p2p4_cs_lgeneg = df_DMD[(df_DMD[target] == 0) & (df_DMD['aha'] == aha) & (df_DMD['phase'] == 4)][
                'our_cs']
            our_dmd_p2p0_cs_lgepos = df_DMD[(df_DMD[target] == 1) & (df_DMD['aha'] == aha) & (df_DMD['phase'] == 0)][
                'our_cs']
            our_dmd_p2p1_cs_lgepos = df_DMD[(df_DMD[target] == 1) & (df_DMD['aha'] == aha) & (df_DMD['phase'] == 1)][
                'our_cs']
            our_dmd_p2p2_cs_lgepos = df_DMD[(df_DMD[target] == 1) & (df_DMD['aha'] == aha) & (df_DMD['phase'] == 2)][
                'our_cs']
            our_dmd_p2p3_cs_lgepos = df_DMD[(df_DMD[target] == 1) & (df_DMD['aha'] == aha) & (df_DMD['phase'] == 3)][
                'our_cs']
            our_dmd_p2p4_cs_lgepos = df_DMD[(df_DMD[target] == 1) & (df_DMD['aha'] == aha) & (df_DMD['phase'] == 4)][
                'our_cs']

            # define here, which p values shall be computed.
            # the first element of listing_a will be computed against the first element of listing_b etc.
            listing_a = [our_dmd_p2p0_rs_lgeneg, our_dmd_p2p1_rs_lgeneg, our_dmd_p2p2_rs_lgeneg, our_dmd_p2p3_rs_lgeneg,
                         our_dmd_p2p4_rs_lgeneg,
                         our_dmd_p2p0_cs_lgeneg, our_dmd_p2p1_cs_lgeneg, our_dmd_p2p2_cs_lgeneg, our_dmd_p2p3_cs_lgeneg,
                         our_dmd_p2p4_cs_lgeneg]
            listing_b = [our_dmd_p2p0_rs_lgepos, our_dmd_p2p1_rs_lgepos, our_dmd_p2p2_rs_lgepos, our_dmd_p2p3_rs_lgepos,
                         our_dmd_p2p4_rs_lgepos,
                         our_dmd_p2p0_cs_lgepos, our_dmd_p2p1_cs_lgepos, our_dmd_p2p2_cs_lgepos, our_dmd_p2p3_cs_lgepos,
                         our_dmd_p2p4_cs_lgepos]

            # define testing sets here
            if listing_a[i].size == 1 or listing_b[i].size == 1:
                results_pvalues[aha - 1, i] = pvalue_error
            else:
                res = pg.ttest(listing_a[i], listing_b[i], paired=paired)
                results_pvalues[aha - 1, i] = float(res['p-val'][0])

    # rounding
    results_pvalues = np.around(pd.DataFrame(results_pvalues), dec_p)

    return results_pvalues

def get_pvals_corrected(df_pvals_uncorrected, alpha0):
    """
    # correct the pvalues via Holm-Bonferroni method
    # Erläuterung der Heidelberg-Statistiker:
    #     hat dir die Holm-Bonferroni-Korrektur empfohlen. Die ist besser als die normale Bonferroni-Korrektur,
    #     weil die normale Bonferroni-Korrektur zu selten signifikant wird. Die Holm-Bonferroni-Korrektur funktioniert
    #     wie folgt. Du sortierst alle 6 mal 16 p-Werte der Größe nach, beginnend mit dem kleinsten. Dann vergleichst
    #     du den Allerkleinsten mit dem Signifikanzniveau α/(6*16). Wenn dieser p-Wert kleiner als das Signifikanzniveau
    #     ist, gehst du zum Zweitkleinsten und vergleichst ihn mit dem Signifikanzniveau α/(6*16 - 1). Wenn dieser wieder
    #     kleiner ist, gehst du zum Drittkleinsten und vergleichst ihn mit dem Signifikanzniveau α/(6*16 - 2). So machst
    #     du weiter mit langsamer größer werdenden Signifikanzniveaus, bis ein p-Wert nicht mehr kleiner als sein jeweiliges
    #     Niveau ist. Dann brichst du das Verfahren ab. Alle p-Werte, die bis dahin kleiner als ihr jeweiliges Niveau waren,
    #     sind dann signifikant. Alle anderen sind nicht signifikant.
    Parameters
    ----------
    df_pvals_uncorrected : (pd.DataFrame) dataframe with uncorrected ttest p-values
    alpha0 : (float) Expected threshold of significance

    Returns Tuple --> (np.ndarray, pd.DataFrame) df<alpha0 as mask, the dataframe with corrected p-values
    -------

    """

    msk_ss, pvals_corr = pg.multicomp(pvals=df_pvals_uncorrected.to_numpy(), alpha=alpha0, method='holm')
    pvals_corr = pd.DataFrame(pvals_corr)
    # df_pvals.style.apply(style_specific_cell, coords=np.where(msk_ss), axis=None)
    return msk_ss, pvals_corr

def my_spec(gt, pred):
    tn, fp, fn, tp = confusion_matrix(gt, pred).ravel()
    specificity = tn / (tn + fp)
    return specificity


def my_sens(gt, pred):
    tn, fp, fn, tp = confusion_matrix(gt, pred).ravel()
    sensitivity = tp / (tp + fn)
    return sensitivity


spec_m = make_scorer(my_spec, greater_is_better=True)
sens_m = make_scorer(my_sens, greater_is_better=True)
roc_m = make_scorer(sklearn.metrics.roc_auc_score, greater_is_better=True)
rec_m = make_scorer(sklearn.metrics.recall_score, greater_is_better=True)
acc_m = make_scorer(sklearn.metrics.accuracy_score, greater_is_better=True)
bacc_m = make_scorer(sklearn.metrics.balanced_accuracy_score, greater_is_better=True)
prec_m = make_scorer(sklearn.metrics.precision_score, greater_is_better=True)
f1_m = make_scorer(sklearn.metrics.f1_score, greater_is_better=True)


def get_peak_rs_cs(df, per_phase=True):
    if per_phase:
        # by phase
        pcs = df[df['phase'] == 1]['our_cs'].values
        prs = df[df['phase'] == 1]['our_rs'].values

    else:
        # get the peak radial and circumferential strains from all time steps
        # min/max
        pcs = np.stack(df.groupby(['phase'])['our_cs'].apply(list).values).astype(np.float32)
        pcs = pcs.min(axis=0)
        prs = np.stack(df.groupby(['phase'])['our_rs'].apply(list).values).astype(np.float32)
        prs = prs.max(axis=0)

    return prs, pcs


def create_df_peak(df_strain_comp, df_strain_p2p):
    # peak radial and circumferential strain

    df_strain_comp.sort_values(by=['pat', 'aha', 'phase'], inplace=True)
    df_strain_p2p.sort_values(by=['pat', 'aha', 'phase'], inplace=True)

    # composed peak by phase
    prs_com_p, pcs_com_p = get_peak_rs_cs(df_strain_comp, per_phase=True)
    # composed peak by max/min
    prs_com_arg, pcs_com_arg = get_peak_rs_cs(df_strain_comp, per_phase=False)
    # p2p peak by phase
    prs_p2p_p, pcs_p2p_p = get_peak_rs_cs(df_strain_p2p, per_phase=True)
    # p2p peak by max/min
    prs_p2p_arg, pcs_p2p_arg = get_peak_rs_cs(df_strain_p2p, per_phase=False)

    # 57 patients x 16 segments = 912 --> Peak strain per patient and segment
    df_peak = df_strain_comp[df_strain_comp['phase'] == 0].copy()

    # composed phase
    df_peak['prs_com_p'] = prs_com_p
    df_peak['pcs_com_p'] = pcs_com_p
    # composed arg-min/max
    df_peak['prs_com_arg'] = prs_com_arg
    df_peak['pcs_com_arg'] = pcs_com_arg
    # p2p phase
    df_peak['prs_p2p_p'] = prs_p2p_p
    df_peak['pcs_p2p_p'] = pcs_p2p_p
    # p2p arg-min/max
    df_peak['prs_p2p_arg'] = prs_p2p_arg
    df_peak['pcs_p2p_arg'] = pcs_p2p_arg
    # minor cleaning and dtype casting
    df_peak.drop(labels='phase', axis=1, inplace=True)
    df_peak.drop(labels='our_rs', axis=1, inplace=True)
    df_peak.drop(labels='our_cs', axis=1, inplace=True)
    df_peak.sort_values(by=['pat', 'aha'], inplace=True)
    df_peak = df_peak.apply(lambda x: x.astype(np.float32, errors='ignore'), axis=0)
    print(df_peak.shape)
    return df_peak


def cross_validate_f1(x, y):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from sklearn.neural_network import MLPClassifier
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.rcParams.update({'font.size': 16})
    from sklearn.model_selection import StratifiedKFold, KFold

    cv = 5
    skf = StratifiedKFold(n_splits=cv)
    #skf = KFold(n_splits=cv)

    clfs = {}
    clfs['MLP'] = make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(100,50,10), random_state=1,
              solver='adam', max_iter=10000))
    clfs['Logistic Regression'] = LogisticRegression(random_state=1, class_weight='balanced', max_iter=1000)
    clfs['Random Forest'] = make_pipeline(MinMaxScaler(), RandomForestClassifier(n_estimators=500, random_state=1,
                                                                                   class_weight='balanced'))  # RandomForestClassifier(n_estimators=100, random_state=1, class_weight='balanced')
    clfs['Naive Bayes'] = GaussianNB()
    clfs['Scaled DecissionTree'] = make_pipeline(MinMaxScaler(), tree.DecisionTreeClassifier(class_weight='balanced'))
    clfs['KNN'] = make_pipeline(MinMaxScaler(), KNeighborsClassifier(n_neighbors=2))
    clfs['Scaled SVC(poly)'] = make_pipeline(MinMaxScaler(),
                                             SVC(kernel='poly', gamma='auto', class_weight='balanced', C=100))
    clfs['SVC(poly)'] = SVC(kernel='poly', gamma='auto', class_weight='balanced', C=100)

    params = {'booster': 'dart',
         'max_depth': 5, 'learning_rate': 0.1,
         'objective': 'binary:logistic',
         'sample_type': 'uniform',
         'normalize_type': 'tree',
         'rate_drop': 0.1,
         'skip_drop': 0.5}
    clfs['xgboost'] = XGBClassifier(**params)
    # ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
    # NeighborhoodComponentsAnalysis(n_components=10, random_state=random_state),

    clfs['Ensemble'] = VotingClassifier(
        estimators=[

            ('lr', clfs['Logistic Regression']),
            ('rf', clfs['Random Forest']),
            ('mlp', clfs['MLP']),
            ('svc', clfs['Scaled SVC(poly)']),
            ('dt', clfs['Scaled DecissionTree'])
        ],
        voting='hard')

    fig, ax = plt.subplots(1, len(clfs.keys()), figsize=(25, 5))
    i = 0
    for label, clf in clfs.items():
        y_pred = cross_val_predict(clf, x, y, cv=skf)
        # scoring=['f1','recall', 'balanced_accuracy', my_sens, my_spec]
        scoring = {'f1': f1_m, 'recall': rec_m, 'balanced_accuracy': bacc_m, 'sens': sens_m, 'spec': spec_m}
        scores = cross_validate(clf, x, y, scoring=scoring, cv=skf)
        print('*' * 10, label, '*' * 10)
        print(scores['test_recall'])
        # print(scores)
        print("Balanced accuracy: %0.2f (+/- %0.2f) [%s]" % (
        scores['test_balanced_accuracy'].mean(), scores['test_balanced_accuracy'].std(), label))
        print("Sensitivity: %0.2f (+/- %0.2f) [%s]" % (scores['test_sens'].mean(), scores['test_sens'].std(), label))
        print("Specifity: %0.2f (+/- %0.2f) [%s]" % (scores['test_spec'].mean(), scores['test_spec'].std(), label))
        # print("F1: %0.2f (+/- %0.2f) [%s]" % (scores['test_f1'].mean(), scores['test_f1'].std(), label))
        # print("Recall: %0.2f (+/- %0.2f) [%s]" % (scores['test_recall'].mean(), scores['test_recall'].std(), label))

        ConfusionMatrixDisplay.from_predictions(y, y_pred, labels=[1, 0], ax=ax[i], colorbar=False)
        ax[i].set_title(label)
        ax[i].xaxis.label.set_visible(False)
        ax[i].yaxis.label.set_visible(False)
        i = i + 1
    plt.show()


def create_grid_search(refit='balanced_accuracy', cv=5):
    gammas = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 'scale', 'auto']
    Cs = [0.1, 1, 5, 10, 20, 100, 1e3]
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    weights = ['balanced', None]
    degree = [2, 3, 4, 5]
    n_estimators = [10, 100, 500, 1000]
    criterions = ['gini', 'entropy', 'log_loss']
    penalties = ['l2']
    solvers = ['liblinear']
    solver_mlp = ['adam', 'sdg', 'lbfgs']
    hidden_layer_sizes = [(100,), (100,50,10), (50,20,10), (5,10,5)]
    depths = [2,3,5,10]

    scaler = [StandardScaler(), MinMaxScaler(), None]

    from sklearn.model_selection import StratifiedKFold, KFold
    skf = StratifiedKFold(n_splits=cv)
    #skf = KFold(n_splits=cv)

    """[
        'standardscaler__copy', 'standardscaler__with_mean', 'standardscaler__with_std',
        'svc__C', 'svc__break_ties', 'svc__cache_size', 'svc__class_weight', 'svc__coef0',
        'svc__decision_function_shape', 'svc__degree', 'svc__gamma', 'svc__kernel', 'svc__max_iter',
        'svc__probability', 'svc__random_state', 'svc__shrinking', 'svc__tol', 'svc__verbose']"""

    svc_params = {'clf': (SVC(),),
                  'clf__gamma': gammas,
                  'clf__C': Cs,
                  'clf__kernel': kernels,
                  'clf__class_weight': weights,
                  'clf__degree': degree,
                  'scaler': scaler}

    ################ ensemble #############
    clfs = {}
    clfs['MLP'] = make_pipeline(MinMaxScaler(), MLPClassifier(hidden_layer_sizes=(100, 50, 10), random_state=1,
                                                              solver='adam', max_iter=10000))
    clfs['Logistic Regression'] = LogisticRegression(random_state=1, class_weight='balanced', max_iter=1000)
    clfs['Random Forest'] = make_pipeline(MinMaxScaler(), RandomForestClassifier(n_estimators=500, random_state=1,
                                                                                 class_weight='balanced'))  # RandomForestClassifier(n_estimators=100, random_state=1, class_weight='balanced')
    clfs['Scaled DecisionTree'] = make_pipeline(MinMaxScaler(), tree.DecisionTreeClassifier(class_weight='balanced'))
    clfs['Scaled SVC(poly)'] = make_pipeline(MinMaxScaler(),
                                             SVC(kernel='poly', gamma='auto', class_weight='balanced', C=100))

    eclf = VotingClassifier(
        estimators=[

            ('lr', clfs['Logistic Regression']),
            ('rf', clfs['Random Forest']),
            ('mlp', clfs['MLP']),
            ('svc', clfs['Scaled SVC(poly)']),
            ('dt', clfs['Scaled DecisionTree'])
        ],
        voting='hard')

    """clf1 = LogisticRegression(random_state=1, class_weight='balanced', penalty='l2', )
    clf2 = RandomForestClassifier(n_estimators=100, random_state=1, class_weight='balanced')
    clf4 = tree.DecisionTreeClassifier(class_weight='balanced')
    clf6 = SVC(kernel='poly', gamma='scale', class_weight='balanced', C=1, degree=3)
    clf7 = MLPClassifier(hidden_layer_sizes=(100, 50, 10), random_state=1,
                                                              solver='adam',max_iter=1000)

    eclf = VotingClassifier(
        estimators=[

            ('lr', clf1),
            ('rf', clf2),
            ('svc', clf6),
            ('dt', clf4),
            ('mlp', clf7)
        ],
        voting='hard')"""

    ens_params = {'clf': (eclf,),
                  'scaler': scaler}

    rf_params = {'clf': (RandomForestClassifier(random_state=1, class_weight='balanced'),),
                 'clf__n_estimators': n_estimators,
                 'clf__class_weight': weights,
                 'scaler': scaler}
    lr_params = {'clf': (LogisticRegression(random_state=1, max_iter=10000, class_weight='balanced'),),
                 'clf__class_weight': weights,
                 'clf__penalty': penalties,
                 'clf__C': Cs,
                 'clf__solver': solvers,
                 'scaler': scaler}
    et_params = {'clf': (ExtraTreesClassifier(n_estimators=250, max_depth=5, random_state=1),),
                 'clf__n_estimators': n_estimators,
                 'scaler': scaler}
    dt_params = {'clf': (tree.DecisionTreeClassifier(class_weight='balanced'),),
                 'clf__criterion': criterions,
                 'scaler': scaler}
    mlp_params = {'clf':(MLPClassifier(hidden_layer_sizes=(100, 50, 10), random_state=1,
                                                              solver='adam', max_iter=10000),),
                  'clf__solver': solver_mlp,
                  'clf__hidden_layer_sizes':hidden_layer_sizes,
                  'scaler': scaler}
    nv_params = {'clf': (GaussianNB(),),
                 'scaler': scaler}

    params_ = {'booster': 'dart',
              'max_depth': 5, 'learning_rate': 0.1,
              'objective': 'binary:logistic',
              'sample_type': 'uniform',
              'normalize_type': 'tree',
              'rate_drop': 0.1,
              'skip_drop': 0.5}
    xgb_params = {'clf' : (XGBClassifier(**params_),),
                  'clf__max_depth': depths,
                  'scaler':scaler}

    params = [rf_params, svc_params, lr_params, et_params, dt_params, ens_params, mlp_params, xgb_params, nv_params]

    pipeline = Pipeline(steps=[
        ('scaler', None),
        #('features', SelectKBest(score_func=chi2, k=60)),
        ('clf', None)
    ])
    scoring = {'f1': f1_m, 'recall': rec_m, 'balanced_accuracy': bacc_m, 'sens': sens_m, 'spec': spec_m,
               'roc_auc': roc_m,
               #'precision': prec_m,
               'accuracy': acc_m}
    #scoring ={'balanced_accuracy': bacc_m}
    # scoring = ['recall', 'accuracy', 'balanced_accuracy', 'average_precision','precision', 'f1', 'roc_auc']
    return GridSearchCV(estimator=pipeline,
                        param_grid=params,
                        scoring=scoring,
                        refit=refit,  # f1, balanced_accuracy
                        cv=skf,
                        n_jobs=16)


def ttest_per_keyframe(df, hue='lge'):

    ps = {}
    temp_y = np.stack(df.groupby(['pat'])[hue].apply(list).values).astype(np.float32)
    temp_y_patients = temp_y.sum(axis=1) > 0
    pat_pos = df.pat.unique()[temp_y_patients == True]
    pat_neg = df.pat.unique()[temp_y_patients == False]

    df_pos = df[df['pat'].isin(pat_pos)]
    df_neg = df[df['pat'].isin(pat_neg)]
    for p in df.phase.unique():
        df_pos_ = df_pos[df_pos.phase.isin([p])]
        df_neg_ = df_neg[df_neg.phase.isin([p])]
        ps['{}_{}'.format('rs', p)] = stats.ttest_ind(df_neg_['our_rs'], df_pos_['our_rs'])[1]
        ps['{}_{}'.format('cs', p)] = stats.ttest_ind(df_neg_['our_cs'], df_pos_['our_cs'])[1]
    return ps


def plot_strain_per_time(df, title=None, method=None, hue='lge', sig_niv = 0.05):
    """
    Plot a split violinplot per time (keyframe)
    Parameters
    ----------
    df :
    title :
    method :
    hue :
    sig_niv :

    Returns
    -------

    """
    # method: (str): enum, one of p2p, comp, window
    import seaborn
    from scipy import stats
    sb.set_context('paper')
    sb.set(font_scale=1.5)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    # scale to % values
    df = df.copy()
    df['our_rs'] = df['our_rs'] * 100
    df['our_cs'] = df['our_cs'] * 100



    if method == 'p2p':
        phases = ['MD-ED', 'ED-MS', 'MS-ES', 'ES-PF', 'PF-MD']
    elif method == 'comp':
        phases = ['ED-ED', 'ED-MS', 'ED-ES', 'ED-PF', 'ED-MD']
    elif method == 'window':
        phases = ['ED-w-->ED', 'MS-w-->MS', 'ES-w-->ES', 'PF-w-->PF', 'MD-w-->MD']
    ax1, ax2 = ax
    ax1 = sb.lineplot(x="phase", y="our_cs",
                      hue=hue,
                      err_style='band', legend=False, ax=ax1,
                      data=df)
    ax1 = sb.violinplot(x="phase", y="our_cs", inner='quart',
                        ax=ax1,
                        data=df,
                        split=True, hue=hue
                        )
    ax1 = sb.stripplot(x="phase", y="our_cs",
                       ax=ax1,
                       data=df,
                       hue=hue
                       )
    # _ = ax1.set_ylim((-25., 25))
    _ = ax1.set_xticks([0, 1, 2, 3, 4], minor=False)
    _ = ax1.set_xticklabels(phases)
    _ = ax1.legend(['neg', 'pos'])
    _ = ax1.set_xlabel('')
    _ = ax1.set_ylabel('Circumferential Strain (%)')

    ax2 = sb.lineplot(x="phase", y="our_rs",
                      hue=hue,
                      err_style='band', legend=False, ax=ax2,
                      data=df)
    ax2 = sb.violinplot(x="phase", y="our_rs", inner='quart',
                        ax=ax2,
                        data=df,
                        split=True, hue=hue
                        )
    ax2 = sb.stripplot(x="phase", y="our_rs",
                       ax=ax2,
                       data=df,
                       hue=hue
                       )

    # _ = ax2.set_ylim((-25, 150))
    _ = ax2.set_xticks([0, 1, 2, 3, 4], minor=False)
    _ = ax2.set_xticklabels(phases)
    _ = ax2.legend(['neg', 'pos'], framealpha=0.5)
    _ = ax2.set_xlabel('')
    _ = ax2.set_ylabel('Radial Strain (%)')
    if title is not None:
        fig.suptitle(title)

    try:
        # phase p-values
        # annotate the xticks with ttest p-values
        y_rs = 0.02 + df['our_rs'].max()
        y_cs = 0.02 + df['our_cs'].max()

        # added
        ps = ttest_per_keyframe(df, hue=hue)

        for xtick in ax1.get_xticks():
            ax1.text(xtick, y_cs, '{}'.format('**' if ps['{}_{}'.format('cs', xtick)] < sig_niv else ''),
                     horizontalalignment='center', size='small', color='black')

        for xtick in ax2.get_xticks():
            ax2.text(xtick, y_rs, '{}'.format('**' if ps['{}_{}'.format('rs', xtick)] < sig_niv else ''),
                     horizontalalignment='center', size='small', color='black')

        # segmental p-values
        alpha0 = float(sig_niv)
        strain = 'RS'
        strain_col = 'our_rs'
        phase_idx = 0
        df_pvals_uncorrected = get_pvals_uncorrected(df, target=hue)
        print(df_pvals_uncorrected.shape)
        msk_ss, df_pvals_corrected = get_pvals_corrected(df_pvals_uncorrected, alpha0=alpha0)
        plt.show()
        for c in df_pvals_corrected.columns:
            if c >= 5: # first 5 columns are the radial strain for the five phases, col 6 - 10 are the CS strain values
                strain = 'CS'
                strain_col = 'our_cs'
                phase_idx = phase_idx % 5
            sig_segments = df_pvals_corrected.index[df_pvals_corrected[c] < sig_niv].tolist()
            pvalues = df_pvals_corrected.iloc[sig_segments, c].tolist()
            if sig_segments:
                print('****'*10)
                print(
                '{} strain in phase: {} - significant AHA segments: {}, p-values: {}'.format(strain, phases[phase_idx],
                                                                                             sig_segments, pvalues))
                '''print(df[(df.phase == phase_idx) & (df.aha.isin(sig_segments)) & (
                            df[hue] == 1)].groupby('aha')[strain_col].mean())
                print(df[(df.phase == phase_idx) & (df.aha.isin(sig_segments)) & (
                        df[hue] == 0)].groupby('aha')[strain_col].mean())'''
                print(df[(df.phase == phase_idx) & (df.aha.isin(sig_segments))].groupby(['aha',hue])[strain_col].mean())
                fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                sb.violinplot(ax=ax,
                              x='aha',
                              y=strain_col,
                              hue=hue,
                              split=True,
                            data=df[(df.phase == phase_idx) & (df.aha.isin(sig_segments))][['aha', strain_col, hue]]).set_title(phases[phase_idx])
                plt.show()
            phase_idx += 1


    except Exception as e:
        print(e)
        pass
    plt.tight_layout()
    return fig




def plot_report(clf, x, y, label=None, cv=5):

    #pats = np.stack(df_strain_comp.groupby(['pat', 'aha'])['pat'].apply(list).values)[0]
    if label: clf = clf.set_params(**label)
    skf = StratifiedKFold(n_splits=cv)
    y_pred = cross_val_predict(clf, x, y, cv=skf)
    hits = y_pred == y
    scores2 = cross_validate(clf, x, y,
                             scoring={'specificity': spec_m, 'sensitivity': sens_m, 'roc': roc_m, 'recall': rec_m,
                                      'accuracy': acc_m, 'balanced_accuracy': bacc_m, 'precision': prec_m, 'f1': f1_m},
                             cv=skf)
    print("Sensitivity: {:0.2f} (+/- {:0.2f})".format(scores2['test_sensitivity'].mean(),
                                                      scores2['test_sensitivity'].std(), ""))
    print(
        "Specifity: {:0.2f} (+/- {:0.2f})".format(scores2['test_specificity'].mean(), scores2['test_specificity'].std(),
                                                  ""))

    print('params: {}'.format(label))
    print("Recall: {:0.2f} (+/- {:0.2f})".format(scores2['test_recall'].mean(), scores2['test_recall'].std()))
    print("Accuracy: {:0.2f} (+/- {:0.2f})".format(scores2['test_accuracy'].mean(), scores2['test_accuracy'].std()))
    print("Precision: {:0.2f} (+/- {:0.2f})".format(scores2['test_precision'].mean(), scores2['test_precision'].std()))
    print("Balanced Accuracy: {:0.2f} (+/- {:0.2f})".format(scores2['test_balanced_accuracy'].mean(),
                                                            scores2['test_balanced_accuracy'].std()))
    print("F1: {:0.2f} (+/- {:0.2f})".format(scores2['test_f1'].mean(), scores2['test_f1'].std()))
    print("AUC: {:0.2f} (+/- {:0.2f})".format(scores2['test_roc'].mean(), scores2['test_roc'].std()))
    disp = ConfusionMatrixDisplay.from_predictions(y, y_pred, labels=[1, 0], display_labels=['positive', 'negative'],
                                                   colorbar=True)
    disp = PrecisionRecallDisplay.from_predictions(y, y_pred)

    return hits, y_pred