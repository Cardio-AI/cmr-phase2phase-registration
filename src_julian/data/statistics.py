# define logging and working directory
from ProjectRoot import change_wd_to_project_root
change_wd_to_project_root()

import numpy as np
import pandas as pd
import os, logging
from logging import info as INFO
from src_julian.utils.skhelperfunctions import Console_and_file_logger
from src_julian.utils.skhelperfunctions import extract_segments
from sklearn.metrics import confusion_matrix

# set up logging
Console_and_file_logger('mvfviz/dmd_temp', logging.INFO)



# patients = [ item for item in sorted(os.listdir(path_to_segmentation_folder)) if os.path.isdir(os.path.join(path_to_segmentation_folder, item)) ]

# import xls
path_to_metadata_xls = '/mnt/ssd/julian/data/metadata/DMDTarique_1.9.xlsx'
df = pd.read_excel(io=path_to_metadata_xls, sheet_name='clean DMD', engine='openpyxl')

# read patients names list
patients = df['pat']

# inits
df_patients = []

for patient in patients:

    df_patient = pd.DataFrame()
    columns = ['pat', 'aha', 'soa', 'lgepos']
    df_patient['pat'] = np.repeat(patient,repeats=16,axis=0)
    df_patient['aha'] = np.arange(1, 17)
    df_patient['soa'] = extract_segments(df[df['pat'] == patient]['soa'].values[0])
    df_patient['lgepos'] = extract_segments(df[df['pat'] == patient]['lgepos'].values[0])

    # append the patient df to the whole list of patients df
    df_patients.append(df_patient)

    # iteration info
    INFO(patient)

df_patients = pd.concat(df_patients, axis=0)

C = confusion_matrix(y_true=df_patients['lgepos'], y_pred=df_patients['soa'])
tn, fn, tp, fp = (C[0,0], C[1,0], C[1,1], C[0,1])

N = tn+fn+tp+fp

accuracy = int(np.round((tp+tn)/N*100))
specifity = int(np.round(tn/(tn+fp)*100))
sensitivity = int(np.round(tp/(tp+fn)*100))
precision = int(np.round(tp/(tp+fp)*100))
missrate = int(np.round(fn/(fn+tp)*100))

x=0


# lge = np.repeat(extract_segments(df_cleandmd[df_cleandmd['ID'] == patient_name]['LGE positive segments'].values[0]), repeats=5, axis=0)
#
#
#
# df_patient = pd.DataFrame()
# # columns = ['patient id', 'AHA segment no', 'cvi_prs', 'cvi_pcs', 'our_prs', 'our_pcs', 'AHA segment lge positive']
# columns = ['patient id', 'AHA segment no', 'phase', 'our_rs', 'our_cs', 'AHA segment lge positive']
# df_patient['patient id'] = patientid
# df_patient['AHA segment no'] = ahano
# df_patient['phase'] = phaseno
# df_patient['our_rs'] = rs_AHA_overtime
# df_patient['our_cs'] = cs_AHA_overtime
# df_patient['AHA segment lge positive'] = lge

##############NOTES ON QQPLOT, PEARSON AND SPEARMAN##############
#     itk_MCPRS = get_mean_strain_values_from_Morales(array=Radial_itk, masks=masks_rot_itk)
    #     itk_MCPCS = get_mean_strain_values_from_Morales(array=Circumferential_itk, masks=masks_rot_itk)
    #     add_MCPRS = get_mean_strain_values_from_Morales(array=Radial_add, masks=masks_rot_add)
    #     add_MCPCS = get_mean_strain_values_from_Morales(array=Circumferential_add, masks=masks_rot_add)
    #
    #     ###############WRITE RESULTS###############
    #     # gather ground truth parameter for comparison
    #     # create df which contains patients metadata
    #     df_groundtruth = pd.read_excel(io=path_to_metadata_xls, sheet_name='clean DMD', dtype='object', index_col=0)
    #     groundtruth_MCPRS = get_parameter_from_xls(dataframe=df_groundtruth, patientname=patient_name,
    #                                                parametername='mid-cavity peak radial strain')
    #     groundtruth_MCPCS = get_parameter_from_xls(dataframe=df_groundtruth, patientname=patient_name,
    #                                                parametername='mid-cavity peak circumferential strain')
    #
    #     # current results
    #     tobewritten = [patient_name,
    #                    groundtruth_MCPRS, itk_MCPRS.max(), add_MCPRS.max(),
    #                    groundtruth_MCPCS, itk_MCPCS.min(), add_MCPCS.min()]
    #
    #     # append to big list
    #     results.append(tobewritten)
    #
    #     INFO('patient ' + str(i+1) + ' / ' + str(len(patients)) + ' finished')
    #
    # # write the results list to a file
    # strain_folder = '/mnt/ssd/julian/data/outputs/strainresults/'
    # filename = '2021.08.19_OLDVERSION_' + patients[0] + '.txt'
    # with open(strain_folder + filename, 'w') as f:
    #     for item in results:
    #         f.write("%s\n" % item)
    #
    # # create DATAFRAME
    # # plot the results list as boxplot/violinplot
    # df = pd.DataFrame(results)
    # df.columns = ['ID',
    #               'Circle_MCPRS', 'itk_MCPRS', 'add_MCPRS',
    #               'Circle_MCPCS', 'itk_MCPCS', 'add_MCPCS']
    #
    # import seaborn as sns
    # sns.set_theme()
    # plt.figure()
    # ax = sns.boxplot(data=df)
    # ax = sns.violinplot(data=df)
    #
    # # inspecting the results df
    # import pandas as pd
    # path='/mnt/ssd/julian/data/outputs/strainresults/2021.08.18_flowfullgtMCPRSvsITKvsJUSTADDED.txt'
    # df = pd.read_csv(path)
    # df.columns = ['ID', 'Circle_MCPRS', 'itk_MCPRS', 'add_MCPRS']
    # # cut the square brackets from the data
    # df['ID'] = df['ID'].map(lambda x: x.lstrip('['))
    # df['add_MCPRS'] = df['add_MCPRS'].map(lambda x: x.rstrip(']'))
    # df['add_MCPRS'] = pd.to_numeric(df['add_MCPRS']) # convert to float for consecutive rounding
    # df.round(decimals=1)
    #
    # # plot correlation
    # import seaborn as sns
    # sns.set_theme()
    # ax=sns.scatterplot(data=df, x='Circle_MCPRS', y='itk_MCPRS', label='itk_MCPRS')
    # ax=sns.scatterplot(data=df, x='Circle_MCPRS', y='add_MCPRS', label='add_MCPRS')
    # ax.set_xlabel('Circle_MCPRS')
    # ax.set_ylabel('Second Method')
    # ax.set_title('Scatter Plot')
    #
    # import seaborn as sns
    # sns.set_theme()
    # ax=sns.scatterplot(data=df, x='Circle_MCPCS', y='itk_MCPCS', label='itk_MCPCS')
    # ax=sns.scatterplot(data=df, x='Circle_MCPCS', y='add_MCPCS', label='add_MCPCS')
    # ax.set_xlabel('Circle_MCPCS')
    # ax.set_ylabel('Second Method')
    # ax.set_title('Scatter Plot')
    #
    # sns.lmplot(x='Circle_MCPRS', y='itk_MCPRS', data=df)
    # plt.xlim(0, 50)
    # plt.ylim(0, 120)
    # plt.title('Circle vs itk')
    #
    # sns.lmplot(x='Circle_MCPRS', y='add_MCPRS', data=df)
    # plt.xlim(0, 50)
    # plt.ylim(0, 120)
    # plt.title('Circle vs add')
    #
    # sns.lmplot(x='Circle_MCPCS', y='itk_MCPCS', data=df)
    # plt.xlim(-5, -25)
    # plt.ylim(-15, 0)
    # plt.title('Circle vs itk')
    #
    # sns.lmplot(x='Circle_MCPCS', y='add_MCPCS', data=df)
    # plt.xlim(-5, -25)
    # plt.ylim(-15, 0)
    # plt.title('Circle vs add')
    #
    # # spearman R
    # from scipy import stats
    # stats.spearmanr(df['Circle_MCPRS'], df['itk_MCPRS'])
    # stats.spearmanr(df['Circle_MCPRS'], df['add_MCPRS'])
    #
    # from scipy import stats
    # stats.spearmanr(df['Circle_MCPCS'], df['itk_MCPCS'])
    # stats.spearmanr(df['Circle_MCPCS'], df['add_MCPCS'])
    #
    # # histograms
    # df['Circle_MCPRS'].hist()
    # df['itk_MCPRS'].hist()
    # df['add_MCPRS'].hist()
    #
    # df['Circle_MCPCS'].hist()
    # df['itk_MCPCS'].hist()
    # df['add_MCPCS'].hist()
    #
    # # seaborn qq plot to check if data is normally distributed
    # import statsmodels.api as sm
    # from scipy.stats import norm
    # import pylab
    # sm.qqplot(df['Circle_MCPRS'], line='s')
    # sm.qqplot(df['itk_MCPRS'], line='s')
    # sm.qqplot(df['add_MCPRS'], line='s')
    #
    # # seaborn qq plot to check if data is normally distributed
    # import statsmodels.api as sm
    # from scipy.stats import norm
    # import pylab
    # sm.qqplot(df['Circle_MCPCS'], line='s')
    # sm.qqplot(df['itk_MCPCS'], line='s')
    # sm.qqplot(df['add_MCPCS'], line='s')
    #
    # # pearson R
    # from scipy import stats
    # stats.pearsonr(df['Circle_MCPRS'], df['itk_MCPRS'])
    # stats.pearsonr(df['Circle_MCPRS'], df['add_MCPRS'])
    #
    # from scipy import stats
    # stats.pearsonr(df['Circle_MCPCS'], df['itk_MCPCS'])
    # stats.pearsonr(df['Circle_MCPCS'], df['add_MCPCS'])
