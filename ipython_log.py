# IPython log file

# define logging and working directory
from ProjectRoot import change_wd_to_project_root
change_wd_to_project_root()
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import random
from collections import Counter
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 30})
import pandas as pd
import numpy as np
import scipy.interpolate
from scipy.interpolate import interp1d
from src.utils.Notebook_imports import *

from ipywidgets import interact
import ipywidgets as widgets
from IPython.core.display import display, HTML
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
from IPython import display as disp
disp.Image("https://media-de.amboss.com/media/thumbs/big_5f22e89d3edf1.jpg")
# mapping between the 5 defined cardiac phases and the lv volume curve
disp.Image("notebooks/Dataset/Wiggers_Diagram_21.png")
disp.Image("https://media-de.amboss.com/media/thumbs/big_5e147f74d36fe.jpg")
from ipyfilechooser import FileChooser
vects_chooser = FileChooser(os.path.join(os.getcwd(),'/mnt/ssd/git/dynamic-cmr-models/exp/phasereg_v3/acdc/'), '')
display(vects_chooser)
# load the vectors
pathstovectnpy = sorted(glob.glob(os.path.join(vects_chooser.selected, 'moved', '*vects*.npy')))
print(pathstovectnpy)
nda_vects = np.concatenate([np.load(path_) for path_ in pathstovectnpy],axis=0)
print(nda_vects.shape)
# load the phase gt and pred
pred_path = os.path.join(vects_chooser.selected, 'pred')
pathtsophasenpy = sorted(glob.glob(os.path.join(pred_path, '*gtpred*.npy')))
print(pathtsophasenpy)
nda_phase = np.concatenate([np.load(path_) for path_ in pathtsophasenpy],axis=1)
print(nda_phase.shape)
gt_, pred_ = np.split(nda_phase, axis=0,indices_or_sections=2)
print(gt_.shape)
gt= gt_[0,:,0]
pred = pred_[0,:,0]
print(gt.shape)
gt_len = gt_[0,:,1]
# load some moved examples for easier understanding of the dimensions
pathtomoved = sorted(glob.glob(os.path.join(vects_chooser.selected, 'moved','*moved*.npy')))
print(len(pathtomoved))
mov = np.concatenate([np.load(path_) for path_ in pathtomoved],axis=0)
print(mov.shape) # patient,time,spatial-z,y,x,channel, e.g.: 69,40,16,64,64,1
# plot one moved timestep of one patient = one 3D volume 
# should go from apex to base
from src.visualization.Visualize import show_2D_or_3D
temp = mov[10,0,...,0]
_ = show_2D_or_3D(temp, cmap='gray', interpolation='none')
import tensorflow as tf
from tensorflow.image import ssim
ssim(mov[10,0], mov[10,1],max_val=1.0, filter_size=11,
                          filter_sigma=1.5, k1=0.01, k2=0.03).numpy().mean()
from skimage.metrics import structural_similarity as ssim
ssim(mov[10,0], mov[10,1],multichannel=True)
plt.imshow(gt[0].T)
plt.imshow(pred[0].T)
import tensorflow as tf
import sys
# returns a matrix with the indicies as values, similar to np.indicies
def get_idxs_tf(x):
    return tf.cast(
        tf.reshape(tf.where(tf.ones((x[0], x[1], x[2]))), (x[0], x[1], x[2], 3)),
        tf.float32)


# returns a matrix with vectors pointing to the center
def get_centers_tf(x):
    return tf.cast(
        tf.tile(tf.convert_to_tensor([x[0] // 2, x[1] // 2, x[2] // 2])[tf.newaxis, tf.newaxis, tf.newaxis, ...],
                (x[0], x[1], x[2], 1)), tf.float32)

def get_angle_tf(a, b):
    # this should work for batches of n-dimensional vectors
    # α = arccos[(a · b) / (|a| * |b|)]
    # |v| = √(x² + y² + z²)
    """
    in 3D space
    If vectors a = [xa, ya, za], b = [xb, yb, zb], then:
    α = arccos[(xa * xb + ya * yb + za * zb) / (√(xa2 + ya2 + za2) * √(xb2 + yb2 + zb2))]
    """
    import math as m
    pi = tf.constant(m.pi)
    b = tf.cast(b, dtype=a.dtype)
    inner = tf.einsum('...i,...i->...', a, b)
    norms = tf.norm(a, ord='euclidean', axis=-1) * tf.norm(b, ord='euclidean', axis=-1)  # [...,None]
    cos = inner / (norms + sys.float_info.epsilon)
    #rad = tf.math.acos(tf.clip_by_value(cos, -1.0, 1.0))
    # rad2deg conversion
    #deg = rad * (180.0/pi)
    return cos[..., tf.newaxis]

dim = [16,64,64]
idx = get_idxs_tf(dim)
c = get_centers_tf(dim)
centers = c - idx
centers_tensor = centers[tf.newaxis, ...]
flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x, centers_tensor), name='flow2direction')
minmax_lambda = lambda x : x[1] + (((x[0] - np.min(x[0])) * (x[2]-x[1]))/(np.max(x[0]) - np.min(x[0])))
# define some central params
lower, mid, upper = -1,0, 1
# align norm an direction by the cardiac phase ED
# resample all 1D feature vectors to the same length
# this should help to validate the pre-defined rules, and if both together explains the cardiac phases

norms = []
dirs= []
cycle_len=[]
gt_ind = []
xval = np.linspace(0,1,40)

for i,n in enumerate(nda_vects[:]):
    cardiac_cycle_length = int(gt_len[i,:,0].sum())
    cycle_len.append(cardiac_cycle_length)
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
    gt_ind.append(ind)
    dir_axis=0
    gt_ed = ind[0]
    #assert cardiac_cycle_length>=gt_ed
    temp = n[:cardiac_cycle_length]
    
    
    dim = temp.shape[1:-1]
    #print(dim)
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    #print('centers: ',c.dtype)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    # direction
    directions = flow2direction_lambda(temp)[...,0]
    # direction mean
    directions = np.mean(directions,axis=(1,2,3))
    # direction ed aligned
    directions = np.roll(directions, -1*gt_ed)
    # direction interpolate to unique length
    f = interp1d(np.linspace(0,1,directions.shape[0]), directions, kind='linear')
    directions = f(xval)
    # direction min/max normalised between -1,1

    directions = minmax_lambda([directions,lower,upper])
    
    # magnitude/norm as mean
    norm_nda = np.linalg.norm(temp, axis=-1).mean(axis=(1,2,3))
    # norm ed aligned
    norm_nda = np.roll(norm_nda, -1*gt_ed)
    # interpolate to unique length
    f = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')
    norm_nda = f(xval)
    # norm min/max aligned to 0,1
    norm_nda = minmax_lambda([norm_nda,mid,upper])
    
    norms.append(norm_nda)
    dirs.append(directions)
# get the mean values
gt_ind = np.stack(gt_ind,axis=0)
cycle_len = np.stack(cycle_len, axis=0)
norms = np.stack(norms, axis=0)
dirs = np.stack(dirs, axis=0)
norms_m = [norms.mean(axis=0)]
dirs_m=[dirs.mean(axis=0)]
# seaborn
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax.margins(0,0)
for n in dirs:
    _ = ax.plot(n, alpha=0.5, zorder=0)

df = pd.DataFrame(dirs).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='blue', label='d_t',ci='sd',err_style='bars',zorder=2,legend=False,ax=ax)
#_ = ax.set_title('Mean direction +/- SD - aligned at ED')
_ = ax.set_xlabel('Time - linear interpolated to 40 frames')
_ = ax.set_ylabel('direction d_t')
_ = ax.legend(loc='upper left')
ax2 = ax.twinx()
ax2.margins(0,0)
df = pd.DataFrame(norms).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='black', label='|v_t|',ci='sd', linestyle='dashed',err_style='bars',zorder=2,legend=False,ax=ax2)
_ = ax2.set_ylabel('norm |v_t|')

_ = ax2.legend(loc='upper right')
folds_chooser = FileChooser(os.path.join(os.getcwd(),'/mnt/ssd/git/dynamic-cmr-models/exp/phasereg_v3/acdc/'), '')
display(folds_chooser)
# define df and helper for pathology extractio
# this should have the same order as our inference data
#df_meta = pd.read_csv(folds_chooser.selected)
df_meta = pd.read_csv('/mnt/sds/sd20i001/sven/data/acdc/02_imported_4D_unfiltered/df_kfold.csv')
df_meta = df_meta.loc[:, ~df_meta.columns.str.contains('^Unnamed')]
df_meta = df_meta[df_meta.patient != 'patient090']  #we excluded this patient

def get_msk_for_pathology(df_, pathology='minf'):
    msk = []
    for f in [0,1,2,3,4]:
        patients = df_[df_.fold.isin([f])]
        patients = patients[patients['phase']=='ED']
        pat = patients[patients['modality'] == 'test']['patient'].str.lower().unique()
        sub_df = patients[patients.patient.isin(pat)].drop_duplicates(ignore_index=True, subset='patient')
        sub_df = sub_df.drop('x_path',axis=1).drop('y_path',axis=1)
        msk.append(sub_df['pathology'].str.lower()==pathology.lower())
    return np.concatenate(msk)
df_meta
# interactive plot per pathology
@interact
def plot_per_pathology(p=df_meta.pathology.unique()):
    print(p)
    msk = get_msk_for_pathology(df_meta, p)

    import seaborn as sb
    sb.set_context('paper')
    sb.set(font_scale = 2)
    fig, ax = plt.subplots(figsize=(25,5))
    ax.margins(0,0)
    dirs_minf = dirs[msk]
    norms_minf = norms[msk]
    for n in dirs_minf:
        _ = ax.plot(n, alpha=0.5, zorder=0)

    df = pd.DataFrame(dirs_minf).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, color='blue', label='d_t',ci='sd',err_style='bars',zorder=2,legend=False,ax=ax)
    #_ = ax.set_title('Mean direction +/- SD - aligned at ED')
    _ = ax.set_xlabel('Time - linear interpolated to 40 frames')
    _ = ax.set_ylabel('direction d_t')
    _ = ax.legend(loc='upper left')
    ax2 = ax.twinx()
    ax2.margins(0,0)
    df = pd.DataFrame(norms_minf).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, color='black', label='|v_t|',ci='sd', linestyle='dashed',err_style='bars',zorder=2,legend=False,ax=ax2)
    _ = ax2.set_ylabel('norm |v_t|')

    _ = ax2.legend(loc='upper right')
    plt.show()
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax2 = ax.twinx()
ax2.margins(0,0)
ax.margins(0,0)
for p in df_meta.pathology.unique():
    msk = get_msk_for_pathology(df_meta, p)
    dirs_minf = dirs[msk]
    norms_minf = norms[msk]
    #for n in dirs_minf:
    #    _ = ax.plot(n, alpha=0.5, zorder=0)

    df = pd.DataFrame(dirs_minf).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, label='d_t_{}'.format(p),ci='sd',err_style=None,zorder=2,legend=False,ax=ax)
    #_ = ax.set_title('Mean direction +/- SD - aligned at ED')
    _ = ax.set_xlabel('Time - linear interpolated to 40 frames')
    _ = ax.set_ylabel('direction d_t')
    _ = ax.legend(loc='upper left')
    
    df = pd.DataFrame(norms_minf).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, label='|v_t|_{}'.format(p),ci='sd', linestyle='dashed',err_style=None,zorder=2,legend=False,ax=ax2)
    _ = ax2.set_ylabel('norm |v_t|')

    _ = ax2.legend(loc='upper right')
plt.show()
# plot one line per patient - norm
fig, ax = plt.subplots(figsize=(25,5))
for n in norms:
    _ = plt.plot(n)
_ = ax.set_title('Magnitudes aligned at ED, resampling shape 40,')
# plot one line per direction
fig, ax = plt.subplots(figsize=(25,5))
for n in dirs:
    _ = plt.plot(n)
_ = ax.set_title('Directions aligned at ED, resampling shape 40,')
from src.visualization.Visualize import show_2D_or_3D
from src.data.Preprocess import clip_quantile
minmax_lambda = lambda x : x[1] + (((x[0] - np.min(x[0])) * (x[2]-x[1]))/(np.max(x[0]) - np.min(x[0])))
nda_temp = nda_vects[0]
# norm_1 = nda_temp[:,:,:32,:32]# top left
norm_1 = nda_temp[:,:,:32,32:]# top right
norm_1 = np.linalg.norm(norm_1, axis=-1)
nda_temp = np.linalg.norm(nda_temp, axis=-1)
nda_temp = clip_quantile(nda_temp, 0.99)
nda_temp = minmax_lambda([nda_temp,0,1])
plt.hist(nda_temp.flatten())
plt.show()
#nda_temp[nda_temp<=0.2] = 0
nda_temp = nda_temp<=0.2
#plt.hist(nda_temp.flatten())
print(nda_temp.shape)
_ = show_2D_or_3D(nda_temp[:,0], allow_slicing=True)
_ = show_2D_or_3D(nda_temp[:,-1], allow_slicing=True)

# top right
_ = show_2D_or_3D(norm_1[:,0], allow_slicing=True,cmap='gray', interpolation='none')
fig, ax = plt.subplots(figsize=(25,3))
plt.plot(np.argmax(gt[0], axis=1))
ax.set_yticks([0, 1, 2, 3, 4], minor=False)
phases = ['ED', 'MS', 'ES', 'PF', 'MD']
_ = ax.set_yticklabels(phases)
# plot the mean/max norm for one patient oveer time
nda_1d_max = np.max(nda_temp,axis=(1,2,3))
nda_1d_mean = np.mean(nda_temp,axis=(1,2,3))
nda_1d_sum = np.sum(nda_temp,axis=(1,2,3))

fig, ax = plt.subplots(figsize=(25,3))
ax.set_title('max norm')
_ = plt.plot(nda_1d_max); plt.show()

fig, ax = plt.subplots(figsize=(25,3))
ax.set_title('mean norm')
_ = plt.plot(nda_1d_mean)

fig, ax = plt.subplots(figsize=(25,3))
ax.set_title('mean sum')
_ = plt.plot(nda_1d_sum)
#ax.set_ylim(0.0,0.15)
temp = np.arange(10)
temp[:0]
def detect_phases(dir_1d_mean, length):
    
    import scipy.signal as sig
    
        # MS
    # Global min of f(x)
    ms = np.argmin(dir_1d_mean)
    ms = ms -1 # take the bucket before the first min peak
    
    
    # ES
    # First time f(x)>0 after MS
    cycle = np.concatenate([dir_1d_mean[ms:], dir_1d_mean[:ms]])
    temp_ = 0
    es_found=False
    negative_slope = False
    for idx,elem in enumerate(cycle):
        if elem<0:
            negative_slope=True
            temp_ = idx
        elif elem>=0 and negative_slope:
            es_found = True
            #temp_ = idx
            break # stop after first zero-transition
    if es_found:
        es = ms + temp_
        #es = es-1
    else:
        es = ms + 1 # the frame after ms, fallback
    if es>=length:
        es = np.mod(es,length)
        print('ES overflow: {}, ms:{}'.format(es,ms))
     
    
    # PF
    # First peak after ES, min height 0.6
    seq = dir_1d_mean[es:]
    peaks, prop = sig.find_peaks(seq, height=0.6)#height=0.6 we normalise between -1 and 1, PF should be close to argmax

    if len(peaks>0):
        pf = es + peaks[0] # take the peak after es
        pf = pf -1
    else:
        print('pf not clear, set to ES {} + 1'.format(es))
        pf = es + 1
    pf = np.mod(pf, length)
      

    # ED
    # Between pf and ms: last time f(x) cross zero from positive to negative
    # a priori knowledge ED needs a minimal distance of 2 frames towards MS
    # CHANGED the minimal distance between ED and MS
    cycle = np.concatenate([dir_1d_mean[pf:], dir_1d_mean[:ms]])
    #print(cycle)
    ed_found = False
    last_idx_positive = True # we start at the pf, which is the peak(dir)
    for idx,elem in enumerate(cycle):
        
        if elem>=0:
            last_idx_positive = True
        # remember the last idx before the direction gets negative the last time before ms
        elif elem<0 and last_idx_positive: # first time direction negative
            ed_found = True # for fallbacks
            temp_ = idx # idx before negative direction
            #print('found transition at: {}'.format(idx))
            last_idx_positive = False # remember only the first idx after transition
        
    if ed_found:
        ed = pf + temp_
        #print('ed:{}, pf:{}, temp_:{}, lenght: {}'.format(ed,pf,temp_,length))
    else: 
        # if we dont find a transition from positive to negative, take the idx which is the closest to zero
        temp_ = np.argmin(np.abs(cycle)) # make sure we have a minimal distance
        ed = pf + temp_
        print('ED: no transition found between {}-{} , closest id to 0: {}, ed = {}'.format(pf,ms,temp_,ed))
    
    if ed>=length:
        #print('gt ed: {}, ed: {}, length: {}'.format(gted,ed,length))
        ed = np.mod(ed,length)
        #print('mod ed: {}'.format(ed))
    #ed = ed-1 # take the bucket before negative
        
    # MD
    # Middle between PF and ED
    ed_slice_idx = ed
    if ed_slice_idx<=pf: # ed overflow --> beginning of cmr stack
        ed_slice_idx = length + ed
    md = (pf+ed_slice_idx)//2 # the bucket after the middle
    md = md + 1
    md = np.mod(md,length)
    
    return np.array([ed,ms,es,pf,md])
# try to predict the phases unsupervised just from the norm/direction curve

def get_phases_from_vects(vects_nda, length=-1, plot=True, dir_axis=0, gtind=[], figsize=(25,3)):
    import scipy.signal
    import scipy.signal as sig
    import scipy.ndimage
    from scipy.ndimage import gaussian_filter1d
    
    gted=gtind[0]
    
    # signal smoothing
    kernel_size = 4
    kernel = np.ones(kernel_size) / kernel_size
    z = 0
    
    
    dim_z = vects_nda.shape[1]
    #vects_nda = vects_nda[:int(length)]
    
    
    # define some helper lambdas
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    dim = vects_nda.shape[1:-1]
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]

    
    # norm of the vector
    norm_nda = np.linalg.norm(vects_nda[...,dir_axis:], axis=-1)
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    norm_nda_thres = clip_quantile(norm_nda, 0.9)
    norm_nda = minmax_lambda([norm_nda,0,1])
    norm_msk = norm_nda>=0.05
    norm_nda = norm_nda * norm_msk
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    nda_1d_mean = np.mean(norm_nda,axis=(1,2,3))
    nda_1d_max = np.max(norm_nda,axis=(1,2,3))
    nda_1d_median = np.median(norm_nda,axis=(1,2,3))
    nda_1d_mean_smooth = np.convolve(nda_1d_mean, kernel, mode='same')
    #print(nda_1d_mean_smooth.shape, nda_1d_mean.shape)
    
    
    # direction relative to the center
    
    directions = flow2direction_lambda(vects_nda)[...,0].numpy()
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    directions = minmax_lambda([directions,lower,upper])
    #plt.hist(directions.flatten());plt.show()
    directions = directions * norm_msk
    #plt.hist(directions.flatten());plt.show()
    
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    dir_1d_mean = np.mean(directions,axis=(1,2,3))
    dir_1d_median = np.median(directions,axis=(1,2,3))

    
    # Find phases
    # 1st smooth the direction with a rolling averge (kernelsize=4)
    # or via gaussian filter
    # 2nd min/max normalise each direction vector in a range [-1:1]
    
    
    dir_1d_mean = gaussian_filter1d(dir_1d_mean, sigma=2, mode='wrap') # original: dir_1d_mean
    
    ################# CHANGED !!!!!! ######################## RECENTLY NO CHANGES
    # min/max scaling
    dir_1d_mean = minmax_lambda([dir_1d_mean,lower,upper])
    
    
#     # test with corner specific vector fields
#     y,x = directions.shape[-2:]
    
#     vects_mean = minmax_lambda([np.mean(vects_nda,axis=(1,2,3,4)),lower,upper])
#     vects_1 = vects_nda[:,:,:y//2,:x//2,:]
#     vects_1_mean = minmax_lambda([np.mean(vects_1,axis=(1,2,3,4)),lower,upper])
#     vects_2 = vects_nda[:,:,y//2:,:x//2,:]
#     vects_2_mean = minmax_lambda([np.mean(vects_2,axis=(1,2,3,4)),lower,upper])
#     vects_3 = vects_nda[:,:,y//2:,x//2:,:]
#     vects_3_mean = minmax_lambda([np.mean(vects_3,axis=(1,2,3,4)),lower,upper])
#     vects_4 = vects_nda[:,:,:y//2,x//2:,:]
#     vects_4_mean = minmax_lambda([np.mean(vects_4,axis=(1,2,3,4)),lower,upper])
    
    
#     # direction per corner
#     dir_1 = directions[:,:,:y//2,:x//2] # top left, plot
#     dir_1_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_2 = directions[:,:,y//2:,:x//2] # top right, plot
#     dir_2_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_3 = directions[:,:,y//2:,x//2:] # bottom left
#     dir_3_mean = minmax_lambda([np.mean(dir_3,axis=(1,2,3)),lower,upper])
#     dir_4 = directions[:,:,:y//2,x//2:] # bottom right
#     dir_4_mean = minmax_lambda([np.mean(dir_4,axis=(1,2,3)),lower,upper])
    

#     # norm per corner
#     norm_mean = minmax_lambda([nda_1d_mean, mid, upper])
#     norm_1 = norm_nda[:,:,:y//2,:x//2]
#     norm_1_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_2 = norm_nda[:,:,y//2:,:x//2]
#     norm_2_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_3 = norm_nda[:,:,y//2:,x//2:]
#     norm_3_mean = minmax_lambda([np.mean(norm_3,axis=(1,2,3)),mid,upper])
#     norm_4 = norm_nda[:,:,:y//2,x//2:]
#     norm_4_mean = minmax_lambda([np.mean(norm_4,axis=(1,2,3)),mid,upper])
    

    ind = detect_phases(dir_1d_mean=dir_1d_mean[:length], length=length)
    
    
    #print('ed: {}, ms: {}, es: {}, pf: {}, md: {}'.format(*ind))
    
        # plot the mean/max norm for one patient oveer time
    if plot:
        
        # DIR 2D+t
        
        dir_2d_t = directions[:,z]
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        #dir_2d_t = minmax_lambda([dir_2d_t,lower,upper])
        div_cmap = 'bwr'
        fig = show_2D_or_3D(dir_2d_t,allow_slicing=False, f_size=(25,2),cmap=div_cmap)
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('dir 2d+t\nmid-cavity')
        #ax.margins(0,0)
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[len(ax.get_images())//2], cax=cax, orientation='horizontal')

        

        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # DIR 2D T x Z
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        directions_tz = directions.mean(axis=(2,3)) # original: norm()
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        _ = ax.imshow(directions_tz.T, aspect='auto', label='aasd', cmap=div_cmap, origin='lower')
        _ = ax.set_xticks(gtind, minor=False)
        ax2 = ax.twinx()
        _ = ax2.plot(dir_1d_mean, c='black',label='dir 1d+t')
        #_ = ax.set_xticks(gtind, minor=False)
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('dir z+t\napex:base')
        #ax.legend(loc='upper left')
        
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1
        
        #ax=fig.add_subplot(rows,1,pos)
        #_ = ax.plot(dir_1d_mean, c='black',label='dir 1d+t')
        #_ = ax.plot(dir_1_mean, label='top left')
        #_ = ax.plot(dir_2_mean, label='top right')
        #_ = ax.plot(dir_3_mean, label='bottom left')
        #_ = ax.plot(dir_4_mean, label='bottom right')
        #ax.legend(loc='upper left')
        #ax.set_xticks(gtind, minor=False)
        #ax.margins(0.022,0)
        #ax.label_outer()
        #pos=pos+1
        #plt.subplots_adjust(wspace=0, hspace=0.1)
        #plt.show()
        norm_cmap = 'hot'
        
        # NORM 2D + t
        norm_2d_t = norm_nda[:,z]
        norm_2d_t = minmax_lambda([norm_2d_t,mid,upper])
        fig = show_2D_or_3D(norm_2d_t,allow_slicing=False, f_size=(25,2),cmap=norm_cmap, interpolation='none')
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('norm 2d+t\nmid-cavity')
        #ax.margins(0,0)
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[0], cax=cax, orientation='horizontal')
        
        
        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # NORM 2D TxZ
        norm2d = minmax_lambda([norm_nda.mean(axis=(2,3)),mid,upper])
        _ = ax.imshow(norm2d.T, aspect='auto', origin='lower',cmap=norm_cmap, interpolation='none')
        _ = ax.set_xticks(gtind, minor=False)
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('norm z+t\napex:base')
        ax2 = ax.twinx()
        _ = ax2.plot(nda_1d_mean, c='black',label='norm 1d+t')
        #ax.spines['top'].set_visible(False)
        #_ = ax.legend(loc='upper left')
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1

        """ax=fig.add_subplot(rows,1,pos)
        _ = ax.plot(norm_mean, label='norm 1d+t')
        #_ = ax.plot(norm_1_mean, label='top left')
        #_ = ax.plot(norm_2_mean, label='top right')
        #_ = ax.plot(norm_3_mean, label='bottom left')
        #_ = ax.plot(norm_4_mean, label='bottom right')
        ax.legend(loc='upper left')
        ax.set_xticks(gtind, minor=False)
        ax.margins(0.022,0)
        ax.label_outer()
        pos = pos+1
        plt.subplots_adjust(wspace=0, hspace=0.1)"""
        #plt.tight_layout()
        plt.show()

        
    
    return ind
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
p = 215#215
gt_length = int(gt_len[p,:,0].sum())
ind = np.argmax(gt[p][:gt_length],axis=0)
get_phases_from_vects(nda_vects[p][:gt_length], length=gt_length,plot=True,dir_axis=0, gtind=ind)
# try to predict the phases unsupervised just from the norm/direction curve

def get_phases_from_vects(vects_nda, length=-1, plot=True, dir_axis=0, gtind=[], figsize=(25,3)):
    import scipy.signal
    import scipy.signal as sig
    import scipy.ndimage
    from scipy.ndimage import gaussian_filter1d
    
    gted=gtind[0]
    
    # signal smoothing
    kernel_size = 4
    kernel = np.ones(kernel_size) / kernel_size
    z = 0
    
    
    dim_z = vects_nda.shape[1]
    #vects_nda = vects_nda[:int(length)]
    
    
    # define some helper lambdas
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    dim = vects_nda.shape[1:-1]
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]

    
    # norm of the vector
    norm_nda = np.linalg.norm(vects_nda[...,dir_axis:], axis=-1)
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    norm_nda_thres = clip_quantile(norm_nda, 0.9)
    norm_nda = minmax_lambda([norm_nda,0,1])
    norm_msk = norm_nda>=0.05
    norm_nda = norm_nda * norm_msk
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    nda_1d_mean = np.mean(norm_nda,axis=(1,2,3))
    nda_1d_max = np.max(norm_nda,axis=(1,2,3))
    nda_1d_median = np.median(norm_nda,axis=(1,2,3))
    nda_1d_mean_smooth = np.convolve(nda_1d_mean, kernel, mode='same')
    #print(nda_1d_mean_smooth.shape, nda_1d_mean.shape)
    
    
    # direction relative to the center
    
    directions = flow2direction_lambda(vects_nda)[...,0].numpy()
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    directions = minmax_lambda([directions,lower,upper])
    #plt.hist(directions.flatten());plt.show()
    directions = directions * norm_msk
    #plt.hist(directions.flatten());plt.show()
    
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    dir_1d_mean = np.mean(directions,axis=(1,2,3))
    dir_1d_median = np.median(directions,axis=(1,2,3))

    
    # Find phases
    # 1st smooth the direction with a rolling averge (kernelsize=4)
    # or via gaussian filter
    # 2nd min/max normalise each direction vector in a range [-1:1]
    
    
    dir_1d_mean = gaussian_filter1d(dir_1d_mean, sigma=2, mode='wrap') # original: dir_1d_mean
    
    ################# CHANGED !!!!!! ######################## RECENTLY NO CHANGES
    # min/max scaling
    dir_1d_mean = minmax_lambda([dir_1d_mean,lower,upper])
    
    
#     # test with corner specific vector fields
#     y,x = directions.shape[-2:]
    
#     vects_mean = minmax_lambda([np.mean(vects_nda,axis=(1,2,3,4)),lower,upper])
#     vects_1 = vects_nda[:,:,:y//2,:x//2,:]
#     vects_1_mean = minmax_lambda([np.mean(vects_1,axis=(1,2,3,4)),lower,upper])
#     vects_2 = vects_nda[:,:,y//2:,:x//2,:]
#     vects_2_mean = minmax_lambda([np.mean(vects_2,axis=(1,2,3,4)),lower,upper])
#     vects_3 = vects_nda[:,:,y//2:,x//2:,:]
#     vects_3_mean = minmax_lambda([np.mean(vects_3,axis=(1,2,3,4)),lower,upper])
#     vects_4 = vects_nda[:,:,:y//2,x//2:,:]
#     vects_4_mean = minmax_lambda([np.mean(vects_4,axis=(1,2,3,4)),lower,upper])
    
    
#     # direction per corner
#     dir_1 = directions[:,:,:y//2,:x//2] # top left, plot
#     dir_1_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_2 = directions[:,:,y//2:,:x//2] # top right, plot
#     dir_2_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_3 = directions[:,:,y//2:,x//2:] # bottom left
#     dir_3_mean = minmax_lambda([np.mean(dir_3,axis=(1,2,3)),lower,upper])
#     dir_4 = directions[:,:,:y//2,x//2:] # bottom right
#     dir_4_mean = minmax_lambda([np.mean(dir_4,axis=(1,2,3)),lower,upper])
    

#     # norm per corner
#     norm_mean = minmax_lambda([nda_1d_mean, mid, upper])
#     norm_1 = norm_nda[:,:,:y//2,:x//2]
#     norm_1_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_2 = norm_nda[:,:,y//2:,:x//2]
#     norm_2_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_3 = norm_nda[:,:,y//2:,x//2:]
#     norm_3_mean = minmax_lambda([np.mean(norm_3,axis=(1,2,3)),mid,upper])
#     norm_4 = norm_nda[:,:,:y//2,x//2:]
#     norm_4_mean = minmax_lambda([np.mean(norm_4,axis=(1,2,3)),mid,upper])
    

    ind = detect_phases(dir_1d_mean=dir_1d_mean[:length], length=length)
    
    
    #print('ed: {}, ms: {}, es: {}, pf: {}, md: {}'.format(*ind))
    
        # plot the mean/max norm for one patient oveer time
    if plot:
        
        # DIR 2D+t
        
        dir_2d_t = directions[:,z]
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        #dir_2d_t = minmax_lambda([dir_2d_t,lower,upper])
        div_cmap = 'bwr'
        fig = show_2D_or_3D(dir_2d_t,allow_slicing=False, f_size=(25,2),cmap=div_cmap)
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('dir 2d+t\nmid-cavity')
        #ax.margins(0,0)
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[len(ax.get_images())//2], cax=cax, orientation='horizontal')

        

        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # DIR 2D T x Z
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        directions_tz = directions.mean(axis=(2,3)) # original: norm()
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        _ = ax.imshow(directions_tz.T, aspect='auto', label='aasd', cmap=div_cmap, origin='lower')
        _ = ax.set_xticks(gtind, minor=False)
        ax2 = ax.twinx()
        _ = ax2.plot(dir_1d_mean, c='black',label='dir 1d+t')
        #_ = ax.set_xticks(gtind, minor=False)
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('dir z+t\napex:base')
        #ax.legend(loc='upper left')
        
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1
        
        #ax=fig.add_subplot(rows,1,pos)
        #_ = ax.plot(dir_1d_mean, c='black',label='dir 1d+t')
        #_ = ax.plot(dir_1_mean, label='top left')
        #_ = ax.plot(dir_2_mean, label='top right')
        #_ = ax.plot(dir_3_mean, label='bottom left')
        #_ = ax.plot(dir_4_mean, label='bottom right')
        #ax.legend(loc='upper left')
        #ax.set_xticks(gtind, minor=False)
        #ax.margins(0.022,0)
        #ax.label_outer()
        #pos=pos+1
        #plt.subplots_adjust(wspace=0, hspace=0.1)
        #plt.show()
        norm_cmap = 'hot'
        
        # NORM 2D + t
        norm_2d_t = norm_nda[:,z]
        norm_2d_t = minmax_lambda([norm_2d_t,mid,upper])
        fig = show_2D_or_3D(norm_2d_t,allow_slicing=False, f_size=(25,2),cmap=norm_cmap, interpolation='none')
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('norm 2d+t\nmid-cavity')
        #ax.margins(0,0)
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[0], cax=cax, orientation='horizontal')
        
        
        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # NORM 2D TxZ
        norm2d = minmax_lambda([norm_nda.mean(axis=(2,3)),mid,upper])
        _ = ax.imshow(norm2d.T, aspect='auto', origin='lower',cmap=norm_cmap, interpolation='none')
        _ = ax.set_xticks(gtind, minor=False)
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('norm z+t\napex:base')
        ax2 = ax.twinx()
        _ = ax2.plot(nda_1d_mean, c='black',label='norm 1d+t')
        #ax.spines['top'].set_visible(False)
        #_ = ax.legend(loc='upper left')
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1

        """ax=fig.add_subplot(rows,1,pos)
        _ = ax.plot(norm_mean, label='norm 1d+t')
        #_ = ax.plot(norm_1_mean, label='top left')
        #_ = ax.plot(norm_2_mean, label='top right')
        #_ = ax.plot(norm_3_mean, label='bottom left')
        #_ = ax.plot(norm_4_mean, label='bottom right')
        ax.legend(loc='upper left')
        ax.set_xticks(gtind, minor=False)
        ax.margins(0.022,0)
        ax.label_outer()
        pos = pos+1
        plt.subplots_adjust(wspace=0, hspace=0.1)"""
        #plt.tight_layout()
        plt.show()

        
    
    return ind
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
p = 2#215
gt_length = int(gt_len[p,:,0].sum())
ind = np.argmax(gt[p][:gt_length],axis=0)
get_phases_from_vects(nda_vects[p][:gt_length], length=gt_length,plot=True,dir_axis=0, gtind=ind)
@interact
def compare_phases(i=(0,nda_vects.shape[0]-1), plot=True):
    
    figsize = (25,1)
    weight = 1
    z = 0
    
    phases = ['ED', 'MS', 'ES', 'PF', 'MD']
    cardiac_cycle_length = int(gt_len[i,:,-1].sum())
    # crop or not
    cardiac_cycle_length_ = gt[i].shape[0]-3 # plot the full length minus border for visualisation
    
    vects = nda_vects[i][:cardiac_cycle_length]
    
    
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0) # 
    ind_pred = np.argmax(pred[i][:],axis=0)
    
    temp = mov[i,:cardiac_cycle_length,z] # cardiac_cycle_length_
    fig = show_2D_or_3D(temp,allow_slicing=False)
    ax = fig.get_axes()[0]
    _ = ax.set_ylabel('CMR 2d+t)\nmid-cavity')
    plt.show()
    
    fig= plt.figure(figsize=figsize)
    ax=fig.add_subplot(111)
    ax.set_xticks(ind, minor=False)
    ax.set_yticks([0, 1, 2, 3, 4], minor=False)
    ax.set_yticklabels(phases, rotation=0)
    ind = np.array(ind)
    onehot_gt = np.zeros((ind.size, cardiac_cycle_length)) # cardiac_cycle_length_
    onehot_gt[np.arange(ind.size), ind] = weight
    #ax.imshow(gt[i][:cardiac_cycle_length].T,aspect='auto') # with uncertainity
    ax.imshow(onehot_gt,aspect='auto', cmap='bone') # gt as binary phase2time mapping
    _ = ax.set_yticklabels(phases)
    ax.set_title('gt')
    ax.margins(0,0)
    
    indices = get_phases_from_vects(vects, length=cardiac_cycle_length, plot=plot,dir_axis=0, gtind=ind, figsize=figsize)

    indices = np.array(indices)
    onehot = np.zeros((indices.size, cardiac_cycle_length)) # cardiac_cycle_length_
    onehot[np.arange(indices.size), indices] = weight
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xticks(indices, minor=False)
    ax.set_yticks([0, 1, 2, 3, 4], minor=False)
    ax.set_yticklabels(phases, rotation=0)
    plt.imshow(onehot,aspect='auto', cmap='bone')
    _ = ax.set_yticklabels(phases)
    ax.margins(0,0)
    #ax.set_title('prediction')
    plt.show()
    
    print(phases)
    print('gt:', ind)
    print('u: ', indices)
    
    
    # this would plot the supervised prediction
    #print('p: ', ind_pred)
    """fig, ax = plt.subplots(figsize=figsize)
    ax.set_xticks(ind_pred, minor=False)
    ax.set_yticks([0, 1, 2, 3, 4], minor=False)
    ax.set_yticklabels(phases, rotation=45)
    plt.imshow(pred[i][:cardiac_cycle_length].T,aspect='auto')
    _ = ax.set_yticklabels(phases);plt.show()"""
    # patient 107 gcn --> good curve for plotting
    # 200 no cut
    # 125 no cut
    # 194 moderate cut 
    # 134 strong cut-off
pred_u = np.zeros_like(gt)
upred_ind = []
cycle_len=[]
print(pred_u.shape)
for i in range(pred_u.shape[0]):
    weight = 1
    
    cardiac_cycle_length = int(gt_len[i,:,0].sum())
    cycle_len.append(cardiac_cycle_length)
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
    #print(gt[i][:cardiac_cycle_length].T.shape)
    
    indices = get_phases_from_vects(nda_vects[i][:cardiac_cycle_length], length=cardiac_cycle_length, gtind=ind,plot=False,dir_axis=0)
    upred_ind.append(indices)
    indices = np.array(indices)
    onehot = np.zeros((indices.size, cardiac_cycle_length))
    onehot[np.arange(indices.size), indices] = weight
    pred_u[i][0:cardiac_cycle_length] = onehot.T
upred_ind=np.stack(upred_ind, axis=0)
cycle_len = np.stack(cycle_len, axis=0)
# re-create a compatible shape for the metric fn
gt_ = np.stack([gt,gt_len], axis=1)
pred_ = np.stack([pred_u,np.zeros_like(pred_u)], axis=1)

# create a dataframe for further plots
from src.utils.Metrics import meandiff
phases = ['ED', 'MS', 'ES', 'PF', 'MD']
res = meandiff(gt_, pred_, apply_sum=False, apply_average=False)
df = pd.DataFrame(res.numpy(), columns=phases)
# get the patient idx, per label where this method fails the most
label = 0
error_thres = 6
if type(res) != np.ndarray: res.numpy()
print('label:{} \naFD>{}\nPatient idx: {}'.format(phases[label],error_thres,np.where(res[:,label]>error_thres)[0]))
print('gt length: {}'.format(cycle_len[res[:,label]>error_thres]))
print('gt indices: {}'.format(gt_ind[:,label][(res[:,label]>error_thres)]))
print('pred indices: {}'.format(upred_ind[:,label][(res[:,label]>error_thres)]))
print('aFD: {}'.format(res[:,label][res[:,label]>error_thres]))
# TOF unsupervised center 14_41 gaus 2
sb.set_context('paper')
sb.set(font_scale = 2)
_ =df.plot(kind='box')
sb.violinplot(data=df)
pd.options.display.float_format = "{:,.2f}".format
df_summarized = pd.concat([df.mean(axis=0),df.std(axis=0), df.median(axis=0)], axis=1)
df_summarized.columns = ['mean', 'SD', 'meadian']
print(df_summarized.mean())
print(df_summarized)

sb.set_context('paper')
sb.set(font_scale = 1.8)
fig, axes = plt.subplots(1,5,figsize=(25,4))
axes = axes.flatten()
for i,ax in enumerate(axes):
    _ = ax.scatter(upred_ind[:,i],gt_ind[:,i])
    _ = ax.axline((1, 1), slope=1)
    _ = ax.set_title(phases[i])
    _ = ax.set_xlabel('Index - pred')
    _ = ax.set_ylabel('Index - gt')
    _ = ax.set_xlim([0,35])
    _ = ax.set_ylim([0,35])
plt.tight_layout()
# try to predict the phases unsupervised just from the norm/direction curve

def get_phases_from_vects(vects_nda, length=-1, plot=True, dir_axis=0, gtind=[], figsize=(25,3)):
    import scipy.signal
    import scipy.signal as sig
    import scipy.ndimage
    from scipy.ndimage import gaussian_filter1d
    
    gted=gtind[0]
    
    # signal smoothing
    kernel_size = 4
    kernel = np.ones(kernel_size) / kernel_size
    z = 0
    
    
    dim_z = vects_nda.shape[1]
    #vects_nda = vects_nda[:int(length)]
    
    
    # define some helper lambdas
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    dim = vects_nda.shape[1:-1]
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]

    
    # norm of the vector
    norm_nda = np.linalg.norm(vects_nda[...,dir_axis:], axis=-1)
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    norm_nda_thres = clip_quantile(norm_nda, 0.9)
    norm_nda = minmax_lambda([norm_nda,0,1])
    norm_msk = norm_nda>=0.0
    norm_nda = norm_nda * norm_msk
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    nda_1d_mean = np.mean(norm_nda,axis=(1,2,3))
    nda_1d_max = np.max(norm_nda,axis=(1,2,3))
    nda_1d_median = np.median(norm_nda,axis=(1,2,3))
    nda_1d_mean_smooth = np.convolve(nda_1d_mean, kernel, mode='same')
    #print(nda_1d_mean_smooth.shape, nda_1d_mean.shape)
    
    
    # direction relative to the center
    
    directions = flow2direction_lambda(vects_nda)[...,0].numpy()
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    directions = minmax_lambda([directions,lower,upper])
    #plt.hist(directions.flatten());plt.show()
    directions = directions * norm_msk
    #plt.hist(directions.flatten());plt.show()
    
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    dir_1d_mean = np.mean(directions,axis=(1,2,3))
    dir_1d_median = np.median(directions,axis=(1,2,3))

    
    # Find phases
    # 1st smooth the direction with a rolling averge (kernelsize=4)
    # or via gaussian filter
    # 2nd min/max normalise each direction vector in a range [-1:1]
    
    
    dir_1d_mean = gaussian_filter1d(dir_1d_mean, sigma=2, mode='wrap') # original: dir_1d_mean
    
    ################# CHANGED !!!!!! ######################## RECENTLY NO CHANGES
    # min/max scaling
    dir_1d_mean = minmax_lambda([dir_1d_mean,lower,upper])
    
    
#     # test with corner specific vector fields
#     y,x = directions.shape[-2:]
    
#     vects_mean = minmax_lambda([np.mean(vects_nda,axis=(1,2,3,4)),lower,upper])
#     vects_1 = vects_nda[:,:,:y//2,:x//2,:]
#     vects_1_mean = minmax_lambda([np.mean(vects_1,axis=(1,2,3,4)),lower,upper])
#     vects_2 = vects_nda[:,:,y//2:,:x//2,:]
#     vects_2_mean = minmax_lambda([np.mean(vects_2,axis=(1,2,3,4)),lower,upper])
#     vects_3 = vects_nda[:,:,y//2:,x//2:,:]
#     vects_3_mean = minmax_lambda([np.mean(vects_3,axis=(1,2,3,4)),lower,upper])
#     vects_4 = vects_nda[:,:,:y//2,x//2:,:]
#     vects_4_mean = minmax_lambda([np.mean(vects_4,axis=(1,2,3,4)),lower,upper])
    
    
#     # direction per corner
#     dir_1 = directions[:,:,:y//2,:x//2] # top left, plot
#     dir_1_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_2 = directions[:,:,y//2:,:x//2] # top right, plot
#     dir_2_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_3 = directions[:,:,y//2:,x//2:] # bottom left
#     dir_3_mean = minmax_lambda([np.mean(dir_3,axis=(1,2,3)),lower,upper])
#     dir_4 = directions[:,:,:y//2,x//2:] # bottom right
#     dir_4_mean = minmax_lambda([np.mean(dir_4,axis=(1,2,3)),lower,upper])
    

#     # norm per corner
#     norm_mean = minmax_lambda([nda_1d_mean, mid, upper])
#     norm_1 = norm_nda[:,:,:y//2,:x//2]
#     norm_1_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_2 = norm_nda[:,:,y//2:,:x//2]
#     norm_2_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_3 = norm_nda[:,:,y//2:,x//2:]
#     norm_3_mean = minmax_lambda([np.mean(norm_3,axis=(1,2,3)),mid,upper])
#     norm_4 = norm_nda[:,:,:y//2,x//2:]
#     norm_4_mean = minmax_lambda([np.mean(norm_4,axis=(1,2,3)),mid,upper])
    

    ind = detect_phases(dir_1d_mean=dir_1d_mean[:length], length=length)
    
    
    #print('ed: {}, ms: {}, es: {}, pf: {}, md: {}'.format(*ind))
    
        # plot the mean/max norm for one patient oveer time
    if plot:
        
        # DIR 2D+t
        
        dir_2d_t = directions[:,z]
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        #dir_2d_t = minmax_lambda([dir_2d_t,lower,upper])
        div_cmap = 'bwr'
        fig = show_2D_or_3D(dir_2d_t,allow_slicing=False, f_size=(25,2),cmap=div_cmap)
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('dir 2d+t\nmid-cavity')
        #ax.margins(0,0)
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[len(ax.get_images())//2], cax=cax, orientation='horizontal')

        

        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # DIR 2D T x Z
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        directions_tz = directions.mean(axis=(2,3)) # original: norm()
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        _ = ax.imshow(directions_tz.T, aspect='auto', label='aasd', cmap=div_cmap, origin='lower')
        _ = ax.set_xticks(gtind, minor=False)
        ax2 = ax.twinx()
        _ = ax2.plot(dir_1d_mean, c='black',label='dir 1d+t')
        #_ = ax.set_xticks(gtind, minor=False)
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('dir z+t\napex:base')
        #ax.legend(loc='upper left')
        
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1
        
        #ax=fig.add_subplot(rows,1,pos)
        #_ = ax.plot(dir_1d_mean, c='black',label='dir 1d+t')
        #_ = ax.plot(dir_1_mean, label='top left')
        #_ = ax.plot(dir_2_mean, label='top right')
        #_ = ax.plot(dir_3_mean, label='bottom left')
        #_ = ax.plot(dir_4_mean, label='bottom right')
        #ax.legend(loc='upper left')
        #ax.set_xticks(gtind, minor=False)
        #ax.margins(0.022,0)
        #ax.label_outer()
        #pos=pos+1
        #plt.subplots_adjust(wspace=0, hspace=0.1)
        #plt.show()
        norm_cmap = 'hot'
        
        # NORM 2D + t
        norm_2d_t = norm_nda[:,z]
        norm_2d_t = minmax_lambda([norm_2d_t,mid,upper])
        fig = show_2D_or_3D(norm_2d_t,allow_slicing=False, f_size=(25,2),cmap=norm_cmap, interpolation='none')
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('norm 2d+t\nmid-cavity')
        #ax.margins(0,0)
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[0], cax=cax, orientation='horizontal')
        
        
        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # NORM 2D TxZ
        norm2d = minmax_lambda([norm_nda.mean(axis=(2,3)),mid,upper])
        _ = ax.imshow(norm2d.T, aspect='auto', origin='lower',cmap=norm_cmap, interpolation='none')
        _ = ax.set_xticks(gtind, minor=False)
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('norm z+t\napex:base')
        ax2 = ax.twinx()
        _ = ax2.plot(nda_1d_mean, c='black',label='norm 1d+t')
        #ax.spines['top'].set_visible(False)
        #_ = ax.legend(loc='upper left')
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1

        """ax=fig.add_subplot(rows,1,pos)
        _ = ax.plot(norm_mean, label='norm 1d+t')
        #_ = ax.plot(norm_1_mean, label='top left')
        #_ = ax.plot(norm_2_mean, label='top right')
        #_ = ax.plot(norm_3_mean, label='bottom left')
        #_ = ax.plot(norm_4_mean, label='bottom right')
        ax.legend(loc='upper left')
        ax.set_xticks(gtind, minor=False)
        ax.margins(0.022,0)
        ax.label_outer()
        pos = pos+1
        plt.subplots_adjust(wspace=0, hspace=0.1)"""
        #plt.tight_layout()
        plt.show()

        
    
    return ind
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
p = 2#215
gt_length = int(gt_len[p,:,0].sum())
ind = np.argmax(gt[p][:gt_length],axis=0)
get_phases_from_vects(nda_vects[p][:gt_length], length=gt_length,plot=True,dir_axis=0, gtind=ind)
@interact
def compare_phases(i=(0,nda_vects.shape[0]-1), plot=True):
    
    figsize = (25,1)
    weight = 1
    z = 0
    
    phases = ['ED', 'MS', 'ES', 'PF', 'MD']
    cardiac_cycle_length = int(gt_len[i,:,-1].sum())
    # crop or not
    cardiac_cycle_length_ = gt[i].shape[0]-3 # plot the full length minus border for visualisation
    
    vects = nda_vects[i][:cardiac_cycle_length]
    
    
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0) # 
    ind_pred = np.argmax(pred[i][:],axis=0)
    
    temp = mov[i,:cardiac_cycle_length,z] # cardiac_cycle_length_
    fig = show_2D_or_3D(temp,allow_slicing=False)
    ax = fig.get_axes()[0]
    _ = ax.set_ylabel('CMR 2d+t)\nmid-cavity')
    plt.show()
    
    fig= plt.figure(figsize=figsize)
    ax=fig.add_subplot(111)
    ax.set_xticks(ind, minor=False)
    ax.set_yticks([0, 1, 2, 3, 4], minor=False)
    ax.set_yticklabels(phases, rotation=0)
    ind = np.array(ind)
    onehot_gt = np.zeros((ind.size, cardiac_cycle_length)) # cardiac_cycle_length_
    onehot_gt[np.arange(ind.size), ind] = weight
    #ax.imshow(gt[i][:cardiac_cycle_length].T,aspect='auto') # with uncertainity
    ax.imshow(onehot_gt,aspect='auto', cmap='bone') # gt as binary phase2time mapping
    _ = ax.set_yticklabels(phases)
    ax.set_title('gt')
    ax.margins(0,0)
    
    indices = get_phases_from_vects(vects, length=cardiac_cycle_length, plot=plot,dir_axis=0, gtind=ind, figsize=figsize)

    indices = np.array(indices)
    onehot = np.zeros((indices.size, cardiac_cycle_length)) # cardiac_cycle_length_
    onehot[np.arange(indices.size), indices] = weight
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xticks(indices, minor=False)
    ax.set_yticks([0, 1, 2, 3, 4], minor=False)
    ax.set_yticklabels(phases, rotation=0)
    plt.imshow(onehot,aspect='auto', cmap='bone')
    _ = ax.set_yticklabels(phases)
    ax.margins(0,0)
    #ax.set_title('prediction')
    plt.show()
    
    print(phases)
    print('gt:', ind)
    print('u: ', indices)
    
    
    # this would plot the supervised prediction
    #print('p: ', ind_pred)
    """fig, ax = plt.subplots(figsize=figsize)
    ax.set_xticks(ind_pred, minor=False)
    ax.set_yticks([0, 1, 2, 3, 4], minor=False)
    ax.set_yticklabels(phases, rotation=45)
    plt.imshow(pred[i][:cardiac_cycle_length].T,aspect='auto')
    _ = ax.set_yticklabels(phases);plt.show()"""
    # patient 107 gcn --> good curve for plotting
    # 200 no cut
    # 125 no cut
    # 194 moderate cut 
    # 134 strong cut-off
pred_u = np.zeros_like(gt)
upred_ind = []
cycle_len=[]
print(pred_u.shape)
for i in range(pred_u.shape[0]):
    weight = 1
    
    cardiac_cycle_length = int(gt_len[i,:,0].sum())
    cycle_len.append(cardiac_cycle_length)
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
    #print(gt[i][:cardiac_cycle_length].T.shape)
    
    indices = get_phases_from_vects(nda_vects[i][:cardiac_cycle_length], length=cardiac_cycle_length, gtind=ind,plot=False,dir_axis=0)
    upred_ind.append(indices)
    indices = np.array(indices)
    onehot = np.zeros((indices.size, cardiac_cycle_length))
    onehot[np.arange(indices.size), indices] = weight
    pred_u[i][0:cardiac_cycle_length] = onehot.T
upred_ind=np.stack(upred_ind, axis=0)
cycle_len = np.stack(cycle_len, axis=0)
# re-create a compatible shape for the metric fn
gt_ = np.stack([gt,gt_len], axis=1)
pred_ = np.stack([pred_u,np.zeros_like(pred_u)], axis=1)

# create a dataframe for further plots
from src.utils.Metrics import meandiff
phases = ['ED', 'MS', 'ES', 'PF', 'MD']
res = meandiff(gt_, pred_, apply_sum=False, apply_average=False)
df = pd.DataFrame(res.numpy(), columns=phases)
# get the patient idx, per label where this method fails the most
label = 0
error_thres = 6
if type(res) != np.ndarray: res.numpy()
print('label:{} \naFD>{}\nPatient idx: {}'.format(phases[label],error_thres,np.where(res[:,label]>error_thres)[0]))
print('gt length: {}'.format(cycle_len[res[:,label]>error_thres]))
print('gt indices: {}'.format(gt_ind[:,label][(res[:,label]>error_thres)]))
print('pred indices: {}'.format(upred_ind[:,label][(res[:,label]>error_thres)]))
print('aFD: {}'.format(res[:,label][res[:,label]>error_thres]))
# TOF unsupervised center 14_41 gaus 2
sb.set_context('paper')
sb.set(font_scale = 2)
_ =df.plot(kind='box')
sb.violinplot(data=df)
pd.options.display.float_format = "{:,.2f}".format
df_summarized = pd.concat([df.mean(axis=0),df.std(axis=0), df.median(axis=0)], axis=1)
df_summarized.columns = ['mean', 'SD', 'meadian']
print(df_summarized.mean())
print(df_summarized)

sb.set_context('paper')
sb.set(font_scale = 1.8)
fig, axes = plt.subplots(1,5,figsize=(25,4))
axes = axes.flatten()
for i,ax in enumerate(axes):
    _ = ax.scatter(upred_ind[:,i],gt_ind[:,i])
    _ = ax.axline((1, 1), slope=1)
    _ = ax.set_title(phases[i])
    _ = ax.set_xlabel('Index - pred')
    _ = ax.set_ylabel('Index - gt')
    _ = ax.set_xlim([0,35])
    _ = ax.set_ylim([0,35])
plt.tight_layout()
@interact
def compare_phases(i=(0,nda_vects.shape[0]-1), plot=True):
    
    figsize = (25,1)
    weight = 1
    z = 0
    
    phases = ['ED', 'MS', 'ES', 'PF', 'MD']
    cardiac_cycle_length = int(gt_len[i,:,-1].sum())
    # crop or not
    cardiac_cycle_length_ = gt[i].shape[0]-3 # plot the full length minus border for visualisation
    
    vects = nda_vects[i][:cardiac_cycle_length]
    
    
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0) # 
    ind_pred = np.argmax(pred[i][:],axis=0)
    
    temp = mov[i,:cardiac_cycle_length,z] # cardiac_cycle_length_
    fig = show_2D_or_3D(temp,allow_slicing=False)
    ax = fig.get_axes()[0]
    _ = ax.set_ylabel('CMR 2d+t)\nmid-cavity')
    plt.show()
    
    fig= plt.figure(figsize=figsize)
    ax=fig.add_subplot(111)
    ax.set_xticks(ind, minor=False)
    ax.set_yticks([0, 1, 2, 3, 4], minor=False)
    ax.set_yticklabels(phases, rotation=0)
    ind = np.array(ind)
    onehot_gt = np.zeros((ind.size, cardiac_cycle_length)) # cardiac_cycle_length_
    onehot_gt[np.arange(ind.size), ind] = weight
    #ax.imshow(gt[i][:cardiac_cycle_length].T,aspect='auto') # with uncertainity
    ax.imshow(onehot_gt,aspect='auto', cmap='bone') # gt as binary phase2time mapping
    _ = ax.set_yticklabels(phases)
    ax.set_title('gt')
    ax.margins(0,0)
    
    indices = get_phases_from_vects(vects, length=cardiac_cycle_length, plot=plot,dir_axis=0, gtind=ind, figsize=figsize)

    indices = np.array(indices)
    onehot = np.zeros((indices.size, cardiac_cycle_length)) # cardiac_cycle_length_
    onehot[np.arange(indices.size), indices] = weight
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xticks(indices, minor=False)
    ax.set_yticks([0, 1, 2, 3, 4], minor=False)
    ax.set_yticklabels(phases, rotation=0)
    plt.imshow(onehot,aspect='auto', cmap='bone')
    _ = ax.set_yticklabels(phases)
    ax.margins(0,0)
    #ax.set_title('prediction')
    plt.show()
    
    print(phases)
    print('gt:', ind)
    print('u: ', indices)
    
    
    # this would plot the supervised prediction
    #print('p: ', ind_pred)
    """fig, ax = plt.subplots(figsize=figsize)
    ax.set_xticks(ind_pred, minor=False)
    ax.set_yticks([0, 1, 2, 3, 4], minor=False)
    ax.set_yticklabels(phases, rotation=45)
    plt.imshow(pred[i][:cardiac_cycle_length].T,aspect='auto')
    _ = ax.set_yticklabels(phases);plt.show()"""
    # patient 107 gcn --> good curve for plotting
    # 200 no cut
    # 125 no cut
    # 194 moderate cut 
    # 134 strong cut-off
@interact
def compare_phases(i=(0,nda_vects.shape[0]-1), plot=True):
    
    figsize = (25,1)
    weight = 1
    z = 0
    
    phases = ['ED', 'MS', 'ES', 'PF', 'MD']
    cardiac_cycle_length = int(gt_len[i,:,-1].sum())
    # crop or not
    cardiac_cycle_length_ = gt[i].shape[0]-3 # plot the full length minus border for visualisation
    
    vects = nda_vects[i][:cardiac_cycle_length]
    
    
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0) # 
    ind_pred = np.argmax(pred[i][:],axis=0)
    
    temp = mov[i,:cardiac_cycle_length,z] # cardiac_cycle_length_
    fig = show_2D_or_3D(temp,allow_slicing=False)
    ax = fig.get_axes()[0]
    _ = ax.set_ylabel('CMR 2d+t)\nmid-cavity')
    plt.show()
    
    fig= plt.figure(figsize=figsize)
    ax=fig.add_subplot(111)
    ax.set_xticks(ind, minor=False)
    ax.set_yticks([0, 1, 2, 3, 4], minor=False)
    ax.set_yticklabels(phases, rotation=0)
    ind = np.array(ind)
    onehot_gt = np.zeros((ind.size, cardiac_cycle_length)) # cardiac_cycle_length_
    onehot_gt[np.arange(ind.size), ind] = weight
    #ax.imshow(gt[i][:cardiac_cycle_length].T,aspect='auto') # with uncertainity
    ax.imshow(onehot_gt,aspect='auto', cmap='bone') # gt as binary phase2time mapping
    _ = ax.set_yticklabels(phases)
    ax.set_title('gt')
    ax.margins(0,0)
    
    indices = get_phases_from_vects(vects, length=cardiac_cycle_length, plot=plot,dir_axis=0, gtind=ind, figsize=figsize)

    indices = np.array(indices)
    onehot = np.zeros((indices.size, cardiac_cycle_length)) # cardiac_cycle_length_
    onehot[np.arange(indices.size), indices] = weight
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xticks(indices, minor=False)
    ax.set_yticks([0, 1, 2, 3, 4], minor=False)
    ax.set_yticklabels(phases, rotation=0)
    plt.imshow(onehot,aspect='auto', cmap='bone')
    _ = ax.set_yticklabels(phases)
    ax.margins(0,0)
    #ax.set_title('prediction')
    plt.show()
    
    print(phases)
    print('gt:', ind)
    print('u: ', indices)
    
    
    # this would plot the supervised prediction
    #print('p: ', ind_pred)
    """fig, ax = plt.subplots(figsize=figsize)
    ax.set_xticks(ind_pred, minor=False)
    ax.set_yticks([0, 1, 2, 3, 4], minor=False)
    ax.set_yticklabels(phases, rotation=45)
    plt.imshow(pred[i][:cardiac_cycle_length].T,aspect='auto')
    _ = ax.set_yticklabels(phases);plt.show()"""
    # patient 107 gcn --> good curve for plotting
    # 200 no cut
    # 125 no cut
    # 194 moderate cut 
    # 134 strong cut-off
@interact
def compare_phases(i=(0,nda_vects.shape[0]-1), plot=True):
    import matplotlib as mpl
    mpl.rcParams.update(mpl.rcParamsDefault)
    figsize = (25,1)
    weight = 1
    z = 0
    
    phases = ['ED', 'MS', 'ES', 'PF', 'MD']
    cardiac_cycle_length = int(gt_len[i,:,-1].sum())
    # crop or not
    cardiac_cycle_length_ = gt[i].shape[0]-3 # plot the full length minus border for visualisation
    
    vects = nda_vects[i][:cardiac_cycle_length]
    
    
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0) # 
    ind_pred = np.argmax(pred[i][:],axis=0)
    
    temp = mov[i,:cardiac_cycle_length,z] # cardiac_cycle_length_
    fig = show_2D_or_3D(temp,allow_slicing=False)
    ax = fig.get_axes()[0]
    _ = ax.set_ylabel('CMR 2d+t)\nmid-cavity')
    plt.show()
    
    fig= plt.figure(figsize=figsize)
    ax=fig.add_subplot(111)
    ax.set_xticks(ind, minor=False)
    ax.set_yticks([0, 1, 2, 3, 4], minor=False)
    ax.set_yticklabels(phases, rotation=0)
    ind = np.array(ind)
    onehot_gt = np.zeros((ind.size, cardiac_cycle_length)) # cardiac_cycle_length_
    onehot_gt[np.arange(ind.size), ind] = weight
    #ax.imshow(gt[i][:cardiac_cycle_length].T,aspect='auto') # with uncertainity
    ax.imshow(onehot_gt,aspect='auto', cmap='bone') # gt as binary phase2time mapping
    _ = ax.set_yticklabels(phases)
    ax.set_title('gt')
    ax.margins(0,0)
    
    indices = get_phases_from_vects(vects, length=cardiac_cycle_length, plot=plot,dir_axis=0, gtind=ind, figsize=figsize)

    indices = np.array(indices)
    onehot = np.zeros((indices.size, cardiac_cycle_length)) # cardiac_cycle_length_
    onehot[np.arange(indices.size), indices] = weight
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xticks(indices, minor=False)
    ax.set_yticks([0, 1, 2, 3, 4], minor=False)
    ax.set_yticklabels(phases, rotation=0)
    plt.imshow(onehot,aspect='auto', cmap='bone')
    _ = ax.set_yticklabels(phases)
    ax.margins(0,0)
    #ax.set_title('prediction')
    plt.show()
    
    print(phases)
    print('gt:', ind)
    print('u: ', indices)
    
    
    # this would plot the supervised prediction
    #print('p: ', ind_pred)
    """fig, ax = plt.subplots(figsize=figsize)
    ax.set_xticks(ind_pred, minor=False)
    ax.set_yticks([0, 1, 2, 3, 4], minor=False)
    ax.set_yticklabels(phases, rotation=45)
    plt.imshow(pred[i][:cardiac_cycle_length].T,aspect='auto')
    _ = ax.set_yticklabels(phases);plt.show()"""
    # patient 107 gcn --> good curve for plotting
    # 200 no cut
    # 125 no cut
    # 194 moderate cut 
    # 134 strong cut-off
pred_u = np.zeros_like(gt)
upred_ind = []
cycle_len=[]
print(pred_u.shape)
for i in range(pred_u.shape[0]):
    weight = 1
    
    cardiac_cycle_length = int(gt_len[i,:,0].sum())
    cycle_len.append(cardiac_cycle_length)
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
    #print(gt[i][:cardiac_cycle_length].T.shape)
    
    indices = get_phases_from_vects(nda_vects[i][:cardiac_cycle_length], length=cardiac_cycle_length, gtind=ind,plot=False,dir_axis=0)
    upred_ind.append(indices)
    indices = np.array(indices)
    onehot = np.zeros((indices.size, cardiac_cycle_length))
    onehot[np.arange(indices.size), indices] = weight
    pred_u[i][0:cardiac_cycle_length] = onehot.T
upred_ind=np.stack(upred_ind, axis=0)
cycle_len = np.stack(cycle_len, axis=0)
# re-create a compatible shape for the metric fn
gt_ = np.stack([gt,gt_len], axis=1)
pred_ = np.stack([pred_u,np.zeros_like(pred_u)], axis=1)

# create a dataframe for further plots
from src.utils.Metrics import meandiff
phases = ['ED', 'MS', 'ES', 'PF', 'MD']
res = meandiff(gt_, pred_, apply_sum=False, apply_average=False)
df = pd.DataFrame(res.numpy(), columns=phases)
# get the patient idx, per label where this method fails the most
label = 0
error_thres = 6
if type(res) != np.ndarray: res.numpy()
print('label:{} \naFD>{}\nPatient idx: {}'.format(phases[label],error_thres,np.where(res[:,label]>error_thres)[0]))
print('gt length: {}'.format(cycle_len[res[:,label]>error_thres]))
print('gt indices: {}'.format(gt_ind[:,label][(res[:,label]>error_thres)]))
print('pred indices: {}'.format(upred_ind[:,label][(res[:,label]>error_thres)]))
print('aFD: {}'.format(res[:,label][res[:,label]>error_thres]))
# TOF unsupervised center 14_41 gaus 2
sb.set_context('paper')
sb.set(font_scale = 2)
_ =df.plot(kind='box')
sb.violinplot(data=df)
pd.options.display.float_format = "{:,.2f}".format
df_summarized = pd.concat([df.mean(axis=0),df.std(axis=0), df.median(axis=0)], axis=1)
df_summarized.columns = ['mean', 'SD', 'meadian']
print(df_summarized.mean())
print(df_summarized)

sb.set_context('paper')
sb.set(font_scale = 1.8)
fig, axes = plt.subplots(1,5,figsize=(25,4))
axes = axes.flatten()
for i,ax in enumerate(axes):
    _ = ax.scatter(upred_ind[:,i],gt_ind[:,i])
    _ = ax.axline((1, 1), slope=1)
    _ = ax.set_title(phases[i])
    _ = ax.set_xlabel('Index - pred')
    _ = ax.set_ylabel('Index - gt')
    _ = ax.set_xlim([0,35])
    _ = ax.set_ylim([0,35])
plt.tight_layout()
# load the vectors
pathstovectnpy = sorted(glob.glob(os.path.join(vects_chooser.selected, 'moved', '*vects*.npy')))
print(pathstovectnpy)
nda_vects = np.concatenate([np.load(path_) for path_ in pathstovectnpy],axis=0)
print(nda_vects.shape)
# load the phase gt and pred
pred_path = os.path.join(vects_chooser.selected, 'pred')
pathtsophasenpy = sorted(glob.glob(os.path.join(pred_path, '*gtpred*.npy')))
print(pathtsophasenpy)
nda_phase = np.concatenate([np.load(path_) for path_ in pathtsophasenpy],axis=1)
print(nda_phase.shape)
gt_, pred_ = np.split(nda_phase, axis=0,indices_or_sections=2)
print(gt_.shape)
gt= gt_[0,:,0]
pred = pred_[0,:,0]
print(gt.shape)
gt_len = gt_[0,:,1]
# load some moved examples for easier understanding of the dimensions
pathtomoved = sorted(glob.glob(os.path.join(vects_chooser.selected, 'moved','*moved*.npy')))
print(len(pathtomoved))
mov = np.concatenate([np.load(path_) for path_ in pathtomoved],axis=0)
print(mov.shape) # patient,time,spatial-z,y,x,channel, e.g.: 69,40,16,64,64,1
# plot one moved timestep of one patient = one 3D volume 
# should go from apex to base
from src.visualization.Visualize import show_2D_or_3D
temp = mov[10,0,...,0]
_ = show_2D_or_3D(temp, cmap='gray', interpolation='none')
import tensorflow as tf
from tensorflow.image import ssim
ssim(mov[10,0], mov[10,1],max_val=1.0, filter_size=11,
                          filter_sigma=1.5, k1=0.01, k2=0.03).numpy().mean()
from skimage.metrics import structural_similarity as ssim
ssim(mov[10,0], mov[10,1],multichannel=True)
plt.imshow(gt[0].T)
plt.imshow(pred[0].T)
import tensorflow as tf
import sys
# returns a matrix with the indicies as values, similar to np.indicies
def get_idxs_tf(x):
    return tf.cast(
        tf.reshape(tf.where(tf.ones((x[0], x[1], x[2]))), (x[0], x[1], x[2], 3)),
        tf.float32)


# returns a matrix with vectors pointing to the center
def get_centers_tf(x):
    return tf.cast(
        tf.tile(tf.convert_to_tensor([x[0] // 2, x[1] // 2, x[2] // 2])[tf.newaxis, tf.newaxis, tf.newaxis, ...],
                (x[0], x[1], x[2], 1)), tf.float32)

def get_angle_tf(a, b):
    # this should work for batches of n-dimensional vectors
    # α = arccos[(a · b) / (|a| * |b|)]
    # |v| = √(x² + y² + z²)
    """
    in 3D space
    If vectors a = [xa, ya, za], b = [xb, yb, zb], then:
    α = arccos[(xa * xb + ya * yb + za * zb) / (√(xa2 + ya2 + za2) * √(xb2 + yb2 + zb2))]
    """
    import math as m
    pi = tf.constant(m.pi)
    b = tf.cast(b, dtype=a.dtype)
    inner = tf.einsum('...i,...i->...', a, b)
    norms = tf.norm(a, ord='euclidean', axis=-1) * tf.norm(b, ord='euclidean', axis=-1)  # [...,None]
    cos = inner / (norms + sys.float_info.epsilon)
    #rad = tf.math.acos(tf.clip_by_value(cos, -1.0, 1.0))
    # rad2deg conversion
    #deg = rad * (180.0/pi)
    return cos[..., tf.newaxis]

dim = [16,64,64]
idx = get_idxs_tf(dim)
c = get_centers_tf(dim)
centers = c - idx
centers_tensor = centers[tf.newaxis, ...]
flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x, centers_tensor), name='flow2direction')
minmax_lambda = lambda x : x[1] + (((x[0] - np.min(x[0])) * (x[2]-x[1]))/(np.max(x[0]) - np.min(x[0])))
# define some central params
lower, mid, upper = -1,0, 1
# align norm an direction by the cardiac phase ED
# resample all 1D feature vectors to the same length
# this should help to validate the pre-defined rules, and if both together explains the cardiac phases

norms = []
dirs= []
cycle_len=[]
gt_ind = []
xval = np.linspace(0,1,40)

for i,n in enumerate(nda_vects[:]):
    cardiac_cycle_length = int(gt_len[i,:,0].sum())
    cycle_len.append(cardiac_cycle_length)
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
    gt_ind.append(ind)
    dir_axis=0
    gt_ed = ind[0]
    #assert cardiac_cycle_length>=gt_ed
    temp = n[:cardiac_cycle_length]
    
    
    dim = temp.shape[1:-1]
    #print(dim)
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    #print('centers: ',c.dtype)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    # direction
    directions = flow2direction_lambda(temp)[...,0]
    # direction mean
    directions = np.mean(directions,axis=(1,2,3))
    # direction ed aligned
    directions = np.roll(directions, -1*gt_ed)
    # direction interpolate to unique length
    f = interp1d(np.linspace(0,1,directions.shape[0]), directions, kind='linear')
    directions = f(xval)
    # direction min/max normalised between -1,1

    directions = minmax_lambda([directions,lower,upper])
    
    # magnitude/norm as mean
    norm_nda = np.linalg.norm(temp, axis=-1).mean(axis=(1,2,3))
    # norm ed aligned
    norm_nda = np.roll(norm_nda, -1*gt_ed)
    # interpolate to unique length
    f = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')
    norm_nda = f(xval)
    # norm min/max aligned to 0,1
    norm_nda = minmax_lambda([norm_nda,mid,upper])
    
    norms.append(norm_nda)
    dirs.append(directions)
# get the mean values
gt_ind = np.stack(gt_ind,axis=0)
cycle_len = np.stack(cycle_len, axis=0)
norms = np.stack(norms, axis=0)
dirs = np.stack(dirs, axis=0)
norms_m = [norms.mean(axis=0)]
dirs_m=[dirs.mean(axis=0)]
# seaborn
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax.margins(0,0)
for n in dirs:
    _ = ax.plot(n, alpha=0.5, zorder=0)

df = pd.DataFrame(dirs).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='blue', label='d_t',ci='sd',err_style='bars',zorder=2,legend=False,ax=ax)
#_ = ax.set_title('Mean direction +/- SD - aligned at ED')
_ = ax.set_xlabel('Time - linear interpolated to 40 frames')
_ = ax.set_ylabel('direction d_t')
_ = ax.legend(loc='upper left')
ax2 = ax.twinx()
ax2.margins(0,0)
df = pd.DataFrame(norms).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='black', label='|v_t|',ci='sd', linestyle='dashed',err_style='bars',zorder=2,legend=False,ax=ax2)
_ = ax2.set_ylabel('norm |v_t|')

_ = ax2.legend(loc='upper right')
from src.visualization.Visualize import show_2D_or_3D
from src.data.Preprocess import clip_quantile
minmax_lambda = lambda x : x[1] + (((x[0] - np.min(x[0])) * (x[2]-x[1]))/(np.max(x[0]) - np.min(x[0])))
nda_temp = nda_vects[0]
# norm_1 = nda_temp[:,:,:32,:32]# top left
norm_1 = nda_temp[:,:,:32,32:]# top right
norm_1 = np.linalg.norm(norm_1, axis=-1)
nda_temp = np.linalg.norm(nda_temp, axis=-1)
nda_temp = clip_quantile(nda_temp, 0.99)
nda_temp = minmax_lambda([nda_temp,0,1])
plt.hist(nda_temp.flatten())
plt.show()
#nda_temp[nda_temp<=0.2] = 0
nda_temp = nda_temp<=0.2
#plt.hist(nda_temp.flatten())
print(nda_temp.shape)
_ = show_2D_or_3D(nda_temp[:,0], allow_slicing=True)
_ = show_2D_or_3D(nda_temp[:,-1], allow_slicing=True)

# top right
_ = show_2D_or_3D(norm_1[:,0], allow_slicing=True,cmap='gray', interpolation='none')
from src.visualization.Visualize import show_2D_or_3D
from src.data.Preprocess import clip_quantile
nda_temp = nda_vects[0]
# norm_1 = nda_temp[:,:,:32,:32]# top left
norm_1 = nda_temp[:,:,:32,32:]# top right
norm_1 = np.linalg.norm(norm_1, axis=-1)
nda_temp = np.linalg.norm(nda_temp, axis=-1)
plt.hist(nda_temp.flatten())
nda_temp = clip_quantile(nda_temp, 0.99)
nda_temp = minmax_lambda([nda_temp,0,1])
plt.hist(nda_temp.flatten())
plt.show()
#nda_temp[nda_temp<=0.2] = 0
nda_temp = nda_temp<=0.2
#plt.hist(nda_temp.flatten())
print(nda_temp.shape)
_ = show_2D_or_3D(nda_temp[:,0], allow_slicing=True)
_ = show_2D_or_3D(nda_temp[:,-1], allow_slicing=True)

# top right
_ = show_2D_or_3D(norm_1[:,0], allow_slicing=True,cmap='gray', interpolation='none')
from src.visualization.Visualize import show_2D_or_3D
from src.data.Preprocess import clip_quantile
nda_temp = nda_vects[0]
# norm_1 = nda_temp[:,:,:32,:32]# top left
norm_1 = nda_temp[:,:,:32,32:]# top right
norm_1 = np.linalg.norm(norm_1, axis=-1)
nda_temp = np.linalg.norm(nda_temp, axis=-1)
plt.hist(nda_temp.flatten());plt.show()
nda_temp = clip_quantile(nda_temp, 0.99)
nda_temp = minmax_lambda([nda_temp,0,1])
plt.hist(nda_temp.flatten())
plt.show()
#nda_temp[nda_temp<=0.2] = 0
nda_temp = nda_temp<=0.2
#plt.hist(nda_temp.flatten())
print(nda_temp.shape)
_ = show_2D_or_3D(nda_temp[:,0], allow_slicing=True)
_ = show_2D_or_3D(nda_temp[:,-1], allow_slicing=True)

# top right
_ = show_2D_or_3D(norm_1[:,0], allow_slicing=True,cmap='gray', interpolation='none')
from src.visualization.Visualize import show_2D_or_3D
from src.data.Preprocess import clip_quantile
nda_temp = nda_vects[0]
# norm_1 = nda_temp[:,:,:32,:32]# top left
norm_1 = nda_temp[:,:,:32,32:]# top right
norm_1 = np.linalg.norm(norm_1, axis=-1)
nda_temp = np.linalg.norm(nda_temp, axis=-1)
plt.hist(nda_temp.flatten());plt.show()
nda_temp = clip_quantile(nda_temp, 0.99)
nda_temp = minmax_lambda([nda_temp,0,1])
plt.hist(nda_temp.flatten())
plt.show()
nda_msk = nda_temp<=0.2
nda_temp = nda_temp * nda_msk
#plt.hist(nda_temp.flatten())
print(nda_temp.shape)
_ = show_2D_or_3D(nda_temp[:,0], allow_slicing=True)
_ = show_2D_or_3D(nda_temp[:,-1], allow_slicing=True)

# top right
_ = show_2D_or_3D(norm_1[:,0], allow_slicing=True,cmap='gray', interpolation='none')
fig, ax = plt.subplots(figsize=(25,3))
plt.plot(np.argmax(gt[0], axis=1))
ax.set_yticks([0, 1, 2, 3, 4], minor=False)
phases = ['ED', 'MS', 'ES', 'PF', 'MD']
_ = ax.set_yticklabels(phases)
# plot the mean/max norm for one patient oveer time
nda_1d_max = np.max(nda_temp,axis=(1,2,3))
nda_1d_mean = np.mean(nda_temp,axis=(1,2,3))
nda_1d_sum = np.sum(nda_temp,axis=(1,2,3))

fig, ax = plt.subplots(figsize=(25,3))
ax.set_title('max norm')
_ = plt.plot(nda_1d_max); plt.show()

fig, ax = plt.subplots(figsize=(25,3))
ax.set_title('mean norm')
_ = plt.plot(nda_1d_mean)

fig, ax = plt.subplots(figsize=(25,3))
ax.set_title('mean sum')
_ = plt.plot(nda_1d_sum)
#ax.set_ylim(0.0,0.15)
from src.visualization.Visualize import show_2D_or_3D
from src.data.Preprocess import clip_quantile
nda_temp = nda_vects[0]
# norm_1 = nda_temp[:,:,:32,:32]# top left
norm_1 = nda_temp[:,:,:32,32:]# top right
norm_1 = np.linalg.norm(norm_1, axis=-1)
nda_temp = np.linalg.norm(nda_temp, axis=-1)
plt.hist(nda_temp.flatten());plt.show()
nda_temp = clip_quantile(nda_temp, 0.99)
nda_temp = minmax_lambda([nda_temp,0,1])
plt.hist(nda_temp.flatten())
plt.show()
nda_msk = (nda_temp<=0.2).astype(np.float32)
nda_temp = nda_temp * nda_msk
#plt.hist(nda_temp.flatten())
print(nda_temp.shape)
_ = show_2D_or_3D(nda_temp[:,0], allow_slicing=True)
_ = show_2D_or_3D(nda_temp[:,-1], allow_slicing=True)

# top right
_ = show_2D_or_3D(norm_1[:,0], allow_slicing=True,cmap='gray', interpolation='none')
from src.visualization.Visualize import show_2D_or_3D
from src.data.Preprocess import clip_quantile
nda_temp = nda_vects[0]
# norm_1 = nda_temp[:,:,:32,:32]# top left
norm_1 = nda_temp[:,:,:32,32:]# top right
norm_1 = np.linalg.norm(norm_1, axis=-1)
nda_temp = np.linalg.norm(nda_temp, axis=-1)
plt.hist(nda_temp.flatten());plt.show()
nda_temp = clip_quantile(nda_temp, 0.99)
nda_temp = minmax_lambda([nda_temp,0,1])
plt.hist(nda_temp.flatten())
plt.show()
nda_msk = (nda_temp>=0.2).astype(np.float32)
nda_temp = nda_temp * nda_msk
#plt.hist(nda_temp.flatten())
print(nda_temp.shape)
_ = show_2D_or_3D(nda_temp[:,0], allow_slicing=True)
_ = show_2D_or_3D(nda_temp[:,-1], allow_slicing=True)

# top right
_ = show_2D_or_3D(norm_1[:,0], allow_slicing=True,cmap='gray', interpolation='none')
fig, ax = plt.subplots(figsize=(25,3))
plt.plot(np.argmax(gt[0], axis=1))
ax.set_yticks([0, 1, 2, 3, 4], minor=False)
phases = ['ED', 'MS', 'ES', 'PF', 'MD']
_ = ax.set_yticklabels(phases)
# plot the mean/max norm for one patient oveer time
nda_1d_max = np.max(nda_temp,axis=(1,2,3))
nda_1d_mean = np.mean(nda_temp,axis=(1,2,3))
nda_1d_sum = np.sum(nda_temp,axis=(1,2,3))

fig, ax = plt.subplots(figsize=(25,3))
ax.set_title('max norm')
_ = plt.plot(nda_1d_max); plt.show()

fig, ax = plt.subplots(figsize=(25,3))
ax.set_title('mean norm')
_ = plt.plot(nda_1d_mean)

fig, ax = plt.subplots(figsize=(25,3))
ax.set_title('mean sum')
_ = plt.plot(nda_1d_sum)
#ax.set_ylim(0.0,0.15)
def detect_phases(dir_1d_mean, length):
    
    import scipy.signal as sig
    
        # MS
    # Global min of f(x)
    ms = np.argmin(dir_1d_mean)
    ms = ms -1 # take the bucket before the first min peak
    
    
    # ES
    # First time f(x)>0 after MS
    cycle = np.concatenate([dir_1d_mean[ms:], dir_1d_mean[:ms]])
    temp_ = 0
    es_found=False
    negative_slope = False
    for idx,elem in enumerate(cycle):
        if elem<0:
            negative_slope=True
            temp_ = idx
        elif elem>=0 and negative_slope:
            es_found = True
            #temp_ = idx
            break # stop after first zero-transition
    if es_found:
        es = ms + temp_
        #es = es-1
    else:
        es = ms + 1 # the frame after ms, fallback
    if es>=length:
        es = np.mod(es,length)
        print('ES overflow: {}, ms:{}'.format(es,ms))
     
    
    # PF
    # First peak after ES, min height 0.6
    seq = dir_1d_mean[es:]
    peaks, prop = sig.find_peaks(seq, height=0.6)#height=0.6 we normalise between -1 and 1, PF should be close to argmax

    if len(peaks>0):
        pf = es + peaks[0] # take the peak after es
        pf = pf -1
    else:
        print('pf not clear, set to ES {} + 1'.format(es))
        pf = es + 1
    pf = np.mod(pf, length)
      

    # ED
    # Between pf and ms: last time f(x) cross zero from positive to negative
    # a priori knowledge ED needs a minimal distance of 2 frames towards MS
    # CHANGED the minimal distance between ED and MS
    cycle = np.concatenate([dir_1d_mean[pf:], dir_1d_mean[:ms]])
    #print(cycle)
    ed_found = False
    last_idx_positive = True # we start at the pf, which is the peak(dir)
    for idx,elem in enumerate(cycle):
        
        if elem>=0:
            last_idx_positive = True
        # remember the last idx before the direction gets negative the last time before ms
        elif elem<0 and last_idx_positive: # first time direction negative
            ed_found = True # for fallbacks
            temp_ = idx # idx before negative direction
            #print('found transition at: {}'.format(idx))
            last_idx_positive = False # remember only the first idx after transition
        
    if ed_found:
        ed = pf + temp_
        #print('ed:{}, pf:{}, temp_:{}, lenght: {}'.format(ed,pf,temp_,length))
    else: 
        # if we dont find a transition from positive to negative, take the idx which is the closest to zero
        temp_ = np.argmin(np.abs(cycle)) # make sure we have a minimal distance
        ed = pf + temp_
        print('ED: no transition found between {}-{} , closest id to 0: {}, ed = {}'.format(pf,ms,temp_,ed))
    
    if ed>=length:
        #print('gt ed: {}, ed: {}, length: {}'.format(gted,ed,length))
        ed = np.mod(ed,length)
        #print('mod ed: {}'.format(ed))
    #ed = ed-1 # take the bucket before negative
        
    # MD
    # Middle between PF and ED
    ed_slice_idx = ed
    if ed_slice_idx<=pf: # ed overflow --> beginning of cmr stack
        ed_slice_idx = length + ed
    md = (pf+ed_slice_idx)//2 # the bucket after the middle
    md = md + 1
    md = np.mod(md,length)
    
    return np.array([ed,ms,es,pf,md])
# try to predict the phases unsupervised just from the norm/direction curve

def get_phases_from_vects(vects_nda, length=-1, plot=True, dir_axis=0, gtind=[], figsize=(25,3)):
    import scipy.signal
    import scipy.signal as sig
    import scipy.ndimage
    from scipy.ndimage import gaussian_filter1d
    
    gted=gtind[0]
    
    # signal smoothing
    kernel_size = 4
    kernel = np.ones(kernel_size) / kernel_size
    z = 0
    
    
    dim_z = vects_nda.shape[1]
    #vects_nda = vects_nda[:int(length)]
    
    
    # define some helper lambdas
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    dim = vects_nda.shape[1:-1]
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]

    
    # norm of the vector
    norm_nda = np.linalg.norm(vects_nda[...,dir_axis:], axis=-1)
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    norm_nda_thres = clip_quantile(norm_nda, 0.9)
    norm_nda = minmax_lambda([norm_nda,0,1])
    norm_msk = norm_nda>=0.0
    norm_nda = norm_nda * norm_msk
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    nda_1d_mean = np.mean(norm_nda,axis=(1,2,3))
    nda_1d_max = np.max(norm_nda,axis=(1,2,3))
    nda_1d_median = np.median(norm_nda,axis=(1,2,3))
    nda_1d_mean_smooth = np.convolve(nda_1d_mean, kernel, mode='same')
    #print(nda_1d_mean_smooth.shape, nda_1d_mean.shape)
    
    
    # direction relative to the center
    
    directions = flow2direction_lambda(vects_nda)[...,0].numpy()
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    directions = minmax_lambda([directions,lower,upper])
    #plt.hist(directions.flatten());plt.show()
    directions = directions * norm_msk
    #plt.hist(directions.flatten());plt.show()
    
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    dir_1d_mean = np.mean(directions,axis=(1,2,3))
    dir_1d_median = np.median(directions,axis=(1,2,3))

    
    # Find phases
    # 1st smooth the direction with a rolling averge (kernelsize=4)
    # or via gaussian filter
    # 2nd min/max normalise each direction vector in a range [-1:1]
    
    
    dir_1d_mean = gaussian_filter1d(dir_1d_mean, sigma=2, mode='wrap') # original: dir_1d_mean
    
    ################# CHANGED !!!!!! ######################## RECENTLY NO CHANGES
    # min/max scaling
    dir_1d_mean = minmax_lambda([dir_1d_mean,lower,upper])
    
    
#     # test with corner specific vector fields
#     y,x = directions.shape[-2:]
    
#     vects_mean = minmax_lambda([np.mean(vects_nda,axis=(1,2,3,4)),lower,upper])
#     vects_1 = vects_nda[:,:,:y//2,:x//2,:]
#     vects_1_mean = minmax_lambda([np.mean(vects_1,axis=(1,2,3,4)),lower,upper])
#     vects_2 = vects_nda[:,:,y//2:,:x//2,:]
#     vects_2_mean = minmax_lambda([np.mean(vects_2,axis=(1,2,3,4)),lower,upper])
#     vects_3 = vects_nda[:,:,y//2:,x//2:,:]
#     vects_3_mean = minmax_lambda([np.mean(vects_3,axis=(1,2,3,4)),lower,upper])
#     vects_4 = vects_nda[:,:,:y//2,x//2:,:]
#     vects_4_mean = minmax_lambda([np.mean(vects_4,axis=(1,2,3,4)),lower,upper])
    
    
#     # direction per corner
#     dir_1 = directions[:,:,:y//2,:x//2] # top left, plot
#     dir_1_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_2 = directions[:,:,y//2:,:x//2] # top right, plot
#     dir_2_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_3 = directions[:,:,y//2:,x//2:] # bottom left
#     dir_3_mean = minmax_lambda([np.mean(dir_3,axis=(1,2,3)),lower,upper])
#     dir_4 = directions[:,:,:y//2,x//2:] # bottom right
#     dir_4_mean = minmax_lambda([np.mean(dir_4,axis=(1,2,3)),lower,upper])
    

#     # norm per corner
#     norm_mean = minmax_lambda([nda_1d_mean, mid, upper])
#     norm_1 = norm_nda[:,:,:y//2,:x//2]
#     norm_1_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_2 = norm_nda[:,:,y//2:,:x//2]
#     norm_2_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_3 = norm_nda[:,:,y//2:,x//2:]
#     norm_3_mean = minmax_lambda([np.mean(norm_3,axis=(1,2,3)),mid,upper])
#     norm_4 = norm_nda[:,:,:y//2,x//2:]
#     norm_4_mean = minmax_lambda([np.mean(norm_4,axis=(1,2,3)),mid,upper])
    

    ind = detect_phases(dir_1d_mean=dir_1d_mean[:length], length=length)
    
    
    #print('ed: {}, ms: {}, es: {}, pf: {}, md: {}'.format(*ind))
    
        # plot the mean/max norm for one patient oveer time
    if plot:
        
        # DIR 2D+t
        
        dir_2d_t = directions[:,z]
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        #dir_2d_t = minmax_lambda([dir_2d_t,lower,upper])
        div_cmap = 'bwr'
        fig = show_2D_or_3D(dir_2d_t,allow_slicing=False, f_size=(25,2),cmap=div_cmap)
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('dir 2d+t\nmid-cavity')
        #ax.margins(0,0)
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[len(ax.get_images())//2], cax=cax, orientation='horizontal')

        

        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # DIR 2D T x Z
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        directions_tz = directions.mean(axis=(2,3)) # original: norm()
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        _ = ax.imshow(directions_tz.T, aspect='auto', label='aasd', cmap=div_cmap, origin='lower')
        _ = ax.set_xticks(gtind, minor=False)
        ax2 = ax.twinx()
        _ = ax2.plot(dir_1d_mean, c='black',label='dir 1d+t')
        #_ = ax.set_xticks(gtind, minor=False)
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('dir z+t\napex:base')
        #ax.legend(loc='upper left')
        
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1
        
        #ax=fig.add_subplot(rows,1,pos)
        #_ = ax.plot(dir_1d_mean, c='black',label='dir 1d+t')
        #_ = ax.plot(dir_1_mean, label='top left')
        #_ = ax.plot(dir_2_mean, label='top right')
        #_ = ax.plot(dir_3_mean, label='bottom left')
        #_ = ax.plot(dir_4_mean, label='bottom right')
        #ax.legend(loc='upper left')
        #ax.set_xticks(gtind, minor=False)
        #ax.margins(0.022,0)
        #ax.label_outer()
        #pos=pos+1
        #plt.subplots_adjust(wspace=0, hspace=0.1)
        #plt.show()
        norm_cmap = 'hot'
        
        # NORM 2D + t
        norm_2d_t = norm_nda[:,z]
        norm_2d_t = minmax_lambda([norm_2d_t,mid,upper])
        fig = show_2D_or_3D(norm_2d_t,allow_slicing=False, f_size=(25,2),cmap=norm_cmap, interpolation='none')
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('norm 2d+t\nmid-cavity')
        #ax.margins(0,0)
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[0], cax=cax, orientation='horizontal')
        
        
        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # NORM 2D TxZ
        norm2d = minmax_lambda([norm_nda.mean(axis=(2,3)),mid,upper])
        _ = ax.imshow(norm2d.T, aspect='auto', origin='lower',cmap=norm_cmap, interpolation='none')
        _ = ax.set_xticks(gtind, minor=False)
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('norm z+t\napex:base')
        ax2 = ax.twinx()
        _ = ax2.plot(nda_1d_mean, c='black',label='norm 1d+t')
        #ax.spines['top'].set_visible(False)
        #_ = ax.legend(loc='upper left')
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1

        """ax=fig.add_subplot(rows,1,pos)
        _ = ax.plot(norm_mean, label='norm 1d+t')
        #_ = ax.plot(norm_1_mean, label='top left')
        #_ = ax.plot(norm_2_mean, label='top right')
        #_ = ax.plot(norm_3_mean, label='bottom left')
        #_ = ax.plot(norm_4_mean, label='bottom right')
        ax.legend(loc='upper left')
        ax.set_xticks(gtind, minor=False)
        ax.margins(0.022,0)
        ax.label_outer()
        pos = pos+1
        plt.subplots_adjust(wspace=0, hspace=0.1)"""
        #plt.tight_layout()
        plt.show()

        
    
    return ind
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
p = 2#215
gt_length = int(gt_len[p,:,0].sum())
ind = np.argmax(gt[p][:gt_length],axis=0)
get_phases_from_vects(nda_vects[p][:gt_length], length=gt_length,plot=True,dir_axis=0, gtind=ind)
@interact
def compare_phases(i=(0,nda_vects.shape[0]-1), plot=True):
    import matplotlib as mpl
    mpl.rcParams.update(mpl.rcParamsDefault)
    figsize = (25,1)
    weight = 1
    z = 0
    
    phases = ['ED', 'MS', 'ES', 'PF', 'MD']
    cardiac_cycle_length = int(gt_len[i,:,-1].sum())
    # crop or not
    cardiac_cycle_length_ = gt[i].shape[0]-3 # plot the full length minus border for visualisation
    
    vects = nda_vects[i][:cardiac_cycle_length]
    
    
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0) # 
    ind_pred = np.argmax(pred[i][:],axis=0)
    
    temp = mov[i,:cardiac_cycle_length,z] # cardiac_cycle_length_
    fig = show_2D_or_3D(temp,allow_slicing=False)
    ax = fig.get_axes()[0]
    _ = ax.set_ylabel('CMR 2d+t)\nmid-cavity')
    plt.show()
    
    fig= plt.figure(figsize=figsize)
    ax=fig.add_subplot(111)
    ax.set_xticks(ind, minor=False)
    ax.set_yticks([0, 1, 2, 3, 4], minor=False)
    ax.set_yticklabels(phases, rotation=0)
    ind = np.array(ind)
    onehot_gt = np.zeros((ind.size, cardiac_cycle_length)) # cardiac_cycle_length_
    onehot_gt[np.arange(ind.size), ind] = weight
    #ax.imshow(gt[i][:cardiac_cycle_length].T,aspect='auto') # with uncertainity
    ax.imshow(onehot_gt,aspect='auto', cmap='bone') # gt as binary phase2time mapping
    _ = ax.set_yticklabels(phases)
    ax.set_title('gt')
    ax.margins(0,0)
    
    indices = get_phases_from_vects(vects, length=cardiac_cycle_length, plot=plot,dir_axis=0, gtind=ind, figsize=figsize)

    indices = np.array(indices)
    onehot = np.zeros((indices.size, cardiac_cycle_length)) # cardiac_cycle_length_
    onehot[np.arange(indices.size), indices] = weight
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xticks(indices, minor=False)
    ax.set_yticks([0, 1, 2, 3, 4], minor=False)
    ax.set_yticklabels(phases, rotation=0)
    plt.imshow(onehot,aspect='auto', cmap='bone')
    _ = ax.set_yticklabels(phases)
    ax.margins(0,0)
    #ax.set_title('prediction')
    plt.show()
    
    print(phases)
    print('gt:', ind)
    print('u: ', indices)
    
    
    # this would plot the supervised prediction
    #print('p: ', ind_pred)
    """fig, ax = plt.subplots(figsize=figsize)
    ax.set_xticks(ind_pred, minor=False)
    ax.set_yticks([0, 1, 2, 3, 4], minor=False)
    ax.set_yticklabels(phases, rotation=45)
    plt.imshow(pred[i][:cardiac_cycle_length].T,aspect='auto')
    _ = ax.set_yticklabels(phases);plt.show()"""
    # patient 107 gcn --> good curve for plotting
    # 200 no cut
    # 125 no cut
    # 194 moderate cut 
    # 134 strong cut-off
pred_u = np.zeros_like(gt)
upred_ind = []
cycle_len=[]
print(pred_u.shape)
for i in range(pred_u.shape[0]):
    weight = 1
    
    cardiac_cycle_length = int(gt_len[i,:,0].sum())
    cycle_len.append(cardiac_cycle_length)
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
    #print(gt[i][:cardiac_cycle_length].T.shape)
    
    indices = get_phases_from_vects(nda_vects[i][:cardiac_cycle_length], length=cardiac_cycle_length, gtind=ind,plot=False,dir_axis=0)
    upred_ind.append(indices)
    indices = np.array(indices)
    onehot = np.zeros((indices.size, cardiac_cycle_length))
    onehot[np.arange(indices.size), indices] = weight
    pred_u[i][0:cardiac_cycle_length] = onehot.T
upred_ind=np.stack(upred_ind, axis=0)
cycle_len = np.stack(cycle_len, axis=0)
# re-create a compatible shape for the metric fn
gt_ = np.stack([gt,gt_len], axis=1)
pred_ = np.stack([pred_u,np.zeros_like(pred_u)], axis=1)

# create a dataframe for further plots
from src.utils.Metrics import meandiff
phases = ['ED', 'MS', 'ES', 'PF', 'MD']
res = meandiff(gt_, pred_, apply_sum=False, apply_average=False)
df = pd.DataFrame(res.numpy(), columns=phases)
# get the patient idx, per label where this method fails the most
label = 0
error_thres = 6
if type(res) != np.ndarray: res.numpy()
print('label:{} \naFD>{}\nPatient idx: {}'.format(phases[label],error_thres,np.where(res[:,label]>error_thres)[0]))
print('gt length: {}'.format(cycle_len[res[:,label]>error_thres]))
print('gt indices: {}'.format(gt_ind[:,label][(res[:,label]>error_thres)]))
print('pred indices: {}'.format(upred_ind[:,label][(res[:,label]>error_thres)]))
print('aFD: {}'.format(res[:,label][res[:,label]>error_thres]))
# TOF unsupervised center 14_41 gaus 2
sb.set_context('paper')
sb.set(font_scale = 2)
_ =df.plot(kind='box')
sb.violinplot(data=df)
pd.options.display.float_format = "{:,.2f}".format
df_summarized = pd.concat([df.mean(axis=0),df.std(axis=0), df.median(axis=0)], axis=1)
df_summarized.columns = ['mean', 'SD', 'meadian']
print(df_summarized.mean())
print(df_summarized)

sb.set_context('paper')
sb.set(font_scale = 1.8)
fig, axes = plt.subplots(1,5,figsize=(25,4))
axes = axes.flatten()
for i,ax in enumerate(axes):
    _ = ax.scatter(upred_ind[:,i],gt_ind[:,i])
    _ = ax.axline((1, 1), slope=1)
    _ = ax.set_title(phases[i])
    _ = ax.set_xlabel('Index - pred')
    _ = ax.set_ylabel('Index - gt')
    _ = ax.set_xlim([0,35])
    _ = ax.set_ylim([0,35])
plt.tight_layout()
@interact
def compare_phases(i=(0,nda_vects.shape[0]-1), plot=True):
    import matplotlib as mpl
    mpl.rcParams.update(mpl.rcParamsDefault)
    figsize = (25,1)
    weight = 1
    z = 0
    
    phases = ['ED', 'MS', 'ES', 'PF', 'MD']
    cardiac_cycle_length = int(gt_len[i,:,-1].sum())
    # crop or not
    cardiac_cycle_length_ = gt[i].shape[0]-3 # plot the full length minus border for visualisation
    
    vects = nda_vects[i][:cardiac_cycle_length]
    
    
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0) # 
    ind_pred = np.argmax(pred[i][:],axis=0)
    
    temp = mov[i,:cardiac_cycle_length,z] # cardiac_cycle_length_
    fig = show_2D_or_3D(temp,allow_slicing=False)
    ax = fig.get_axes()[0]
    _ = ax.set_ylabel('CMR 2d+t)\nmid-cavity')
    plt.show()
    
    fig= plt.figure(figsize=figsize)
    ax=fig.add_subplot(111)
    ax.set_xticks(ind, minor=False)
    ax.set_yticks([0, 1, 2, 3, 4], minor=False)
    ax.set_yticklabels(phases, rotation=0)
    ind = np.array(ind)
    onehot_gt = np.zeros((ind.size, cardiac_cycle_length)) # cardiac_cycle_length_
    onehot_gt[np.arange(ind.size), ind] = weight
    #ax.imshow(gt[i][:cardiac_cycle_length].T,aspect='auto') # with uncertainity
    ax.imshow(onehot_gt,aspect='auto', cmap='bone') # gt as binary phase2time mapping
    _ = ax.set_yticklabels(phases)
    ax.set_title('gt')
    ax.margins(0,0)
    
    indices = get_phases_from_vects(vects, length=cardiac_cycle_length, plot=plot,dir_axis=0, gtind=ind, figsize=figsize)

    indices = np.array(indices)
    onehot = np.zeros((indices.size, cardiac_cycle_length)) # cardiac_cycle_length_
    onehot[np.arange(indices.size), indices] = weight
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xticks(indices, minor=False)
    ax.set_yticks([0, 1, 2, 3, 4], minor=False)
    ax.set_yticklabels(phases, rotation=0)
    plt.imshow(onehot,aspect='auto', cmap='bone')
    _ = ax.set_yticklabels(phases)
    ax.margins(0,0)
    #ax.set_title('prediction')
    plt.show()
    
    print(phases)
    print('gt:', ind)
    print('u: ', indices)
    
    
    # this would plot the supervised prediction
    #print('p: ', ind_pred)
    """fig, ax = plt.subplots(figsize=figsize)
    ax.set_xticks(ind_pred, minor=False)
    ax.set_yticks([0, 1, 2, 3, 4], minor=False)
    ax.set_yticklabels(phases, rotation=45)
    plt.imshow(pred[i][:cardiac_cycle_length].T,aspect='auto')
    _ = ax.set_yticklabels(phases);plt.show()"""
    # patient 107 gcn --> good curve for plotting
    # 200 no cut
    # 125 no cut
    # 194 moderate cut 
    # 134 strong cut-off
@interact
def compare_phases(i=(0,nda_vects.shape[0]-1), plot=True):
    import matplotlib as mpl
    mpl.rcParams.update(mpl.rcParamsDefault)
    figsize = (25,1)
    weight = 1
    z = 0
    
    phases = ['ED', 'MS', 'ES', 'PF', 'MD']
    cardiac_cycle_length = int(gt_len[i,:,-1].sum())
    # crop or not
    cardiac_cycle_length_ = gt[i].shape[0]-3 # plot the full length minus border for visualisation
    
    vects = nda_vects[i][:cardiac_cycle_length]
    
    
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0) # 
    ind_pred = np.argmax(pred[i][:],axis=0)
    
    temp = mov[i,:cardiac_cycle_length,z] # cardiac_cycle_length_
    fig = show_2D_or_3D(temp,allow_slicing=False)
    ax = fig.get_axes()[0]
    _ = ax.set_ylabel('CMR 2d+t)\nmid-cavity')
    plt.show()
    
    fig= plt.figure(figsize=figsize)
    ax=fig.add_subplot(111)
    ax.set_xticks(ind, minor=False)
    ax.set_yticks([0, 1, 2, 3, 4], minor=False)
    ax.set_yticklabels(phases, rotation=0)
    ind = np.array(ind)
    onehot_gt = np.zeros((ind.size, cardiac_cycle_length)) # cardiac_cycle_length_
    onehot_gt[np.arange(ind.size), ind] = weight
    #ax.imshow(gt[i][:cardiac_cycle_length].T,aspect='auto') # with uncertainity
    ax.imshow(onehot_gt,aspect='auto', cmap='bone') # gt as binary phase2time mapping
    _ = ax.set_yticklabels(phases)
    ax.set_title('gt')
    ax.margins(0,0)
    
    indices = get_phases_from_vects(vects, length=cardiac_cycle_length, plot=plot,dir_axis=0, gtind=ind, figsize=figsize)

    indices = np.array(indices)
    onehot = np.zeros((indices.size, cardiac_cycle_length)) # cardiac_cycle_length_
    onehot[np.arange(indices.size), indices] = weight
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xticks(indices, minor=False)
    ax.set_yticks([0, 1, 2, 3, 4], minor=False)
    ax.set_yticklabels(phases, rotation=0)
    plt.imshow(onehot,aspect='auto', cmap='bone')
    _ = ax.set_yticklabels(phases)
    ax.margins(0,0)
    #ax.set_title('prediction')
    plt.show()
    
    print(phases)
    print('gt:', ind)
    print('u: ', indices)
    
    
    # this would plot the supervised prediction
    #print('p: ', ind_pred)
    """fig, ax = plt.subplots(figsize=figsize)
    ax.set_xticks(ind_pred, minor=False)
    ax.set_yticks([0, 1, 2, 3, 4], minor=False)
    ax.set_yticklabels(phases, rotation=45)
    plt.imshow(pred[i][:cardiac_cycle_length].T,aspect='auto')
    _ = ax.set_yticklabels(phases);plt.show()"""
    # patient 107 gcn --> good curve for plotting
    # 200 no cut
    # 125 no cut
    # 194 moderate cut 
    # 134 strong cut-off
@interact
def compare_phases(i=(0,nda_vects.shape[0]-1), plot=True):
    import matplotlib as mpl
    mpl.rcParams.update(mpl.rcParamsDefault)
    figsize = (25,1)
    weight = 1
    z = 0
    
    phases = ['ED', 'MS', 'ES', 'PF', 'MD']
    cardiac_cycle_length = int(gt_len[i,:,-1].sum())
    # crop or not
    cardiac_cycle_length_ = gt[i].shape[0]-3 # plot the full length minus border for visualisation
    
    vects = nda_vects[i][:cardiac_cycle_length]
    
    
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0) # 
    ind_pred = np.argmax(pred[i][:],axis=0)
    
    temp = mov[i,:cardiac_cycle_length,z] # cardiac_cycle_length_
    fig = show_2D_or_3D(temp,allow_slicing=False)
    ax = fig.get_axes()[0]
    _ = ax.set_ylabel('CMR 2d+t)\nmid-cavity')
    plt.show()
    
    fig= plt.figure(figsize=figsize)
    ax=fig.add_subplot(111)
    ax.set_xticks(ind, minor=False)
    ax.set_yticks([0, 1, 2, 3, 4], minor=False)
    ax.set_yticklabels(phases, rotation=0)
    ind = np.array(ind)
    onehot_gt = np.zeros((ind.size, cardiac_cycle_length)) # cardiac_cycle_length_
    onehot_gt[np.arange(ind.size), ind] = weight
    #ax.imshow(gt[i][:cardiac_cycle_length].T,aspect='auto') # with uncertainity
    ax.imshow(onehot_gt,aspect='auto', cmap='bone') # gt as binary phase2time mapping
    _ = ax.set_yticklabels(phases)
    ax.set_title('gt')
    ax.margins(0,0)
    
    indices = get_phases_from_vects(vects, length=cardiac_cycle_length, plot=plot,dir_axis=0, gtind=ind, figsize=figsize)

    indices = np.array(indices)
    onehot = np.zeros((indices.size, cardiac_cycle_length)) # cardiac_cycle_length_
    onehot[np.arange(indices.size), indices] = weight
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xticks(indices, minor=False)
    ax.set_yticks([0, 1, 2, 3, 4], minor=False)
    ax.set_yticklabels(phases, rotation=0)
    plt.imshow(onehot,aspect='auto', cmap='bone')
    _ = ax.set_yticklabels(phases)
    ax.margins(0,0)
    #ax.set_title('prediction')
    plt.show()
    
    print(phases)
    print('gt:', ind)
    print('u: ', indices)
    
    
    # this would plot the supervised prediction
    #print('p: ', ind_pred)
    """fig, ax = plt.subplots(figsize=figsize)
    ax.set_xticks(ind_pred, minor=False)
    ax.set_yticks([0, 1, 2, 3, 4], minor=False)
    ax.set_yticklabels(phases, rotation=45)
    plt.imshow(pred[i][:cardiac_cycle_length].T,aspect='auto')
    _ = ax.set_yticklabels(phases);plt.show()"""
    # patient 107 gcn --> good curve for plotting
    # 200 no cut
    # 125 no cut
    # 194 moderate cut 
    # 134 strong cut-off
# try to predict the phases unsupervised just from the norm/direction curve

def get_phases_from_vects(vects_nda, length=-1, plot=True, dir_axis=0, gtind=[], figsize=(25,3)):
    import scipy.signal
    import scipy.signal as sig
    import scipy.ndimage
    from scipy.ndimage import gaussian_filter1d
    
    gted=gtind[0]
    
    # signal smoothing
    kernel_size = 4
    kernel = np.ones(kernel_size) / kernel_size
    z = 0
    
    
    dim_z = vects_nda.shape[1]
    #vects_nda = vects_nda[:int(length)]
    
    
    # define some helper lambdas
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    dim = vects_nda.shape[1:-1]
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]

    
    # norm of the vector
    norm_nda = np.linalg.norm(vects_nda[...,dir_axis:], axis=-1)
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    norm_nda_thres = clip_quantile(norm_nda, 0.9)
    norm_nda = minmax_lambda([norm_nda,0,1])
    norm_msk = norm_nda>=0.0
    norm_nda = norm_nda * norm_msk
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    nda_1d_mean = np.mean(norm_nda,axis=(1,2,3))
    nda_1d_max = np.max(norm_nda,axis=(1,2,3))
    nda_1d_median = np.median(norm_nda,axis=(1,2,3))
    nda_1d_mean_smooth = np.convolve(nda_1d_mean, kernel, mode='same')
    #print(nda_1d_mean_smooth.shape, nda_1d_mean.shape)
    
    
    # direction relative to the center
    
    directions = flow2direction_lambda(vects_nda)[...,0].numpy()
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    directions = minmax_lambda([directions,lower,upper])
    #plt.hist(directions.flatten());plt.show()
    directions = directions * norm_msk
    #plt.hist(directions.flatten());plt.show()
    
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    dir_1d_mean = np.mean(directions,axis=(1,2,3))
    dir_1d_median = np.median(directions,axis=(1,2,3))

    
    # Find phases
    # 1st smooth the direction with a rolling averge (kernelsize=4)
    # or via gaussian filter
    # 2nd min/max normalise each direction vector in a range [-1:1]
    
    
    dir_1d_mean = gaussian_filter1d(dir_1d_mean, sigma=2, mode='wrap') # original: dir_1d_mean
    
    ################# CHANGED !!!!!! ######################## RECENTLY NO CHANGES
    # min/max scaling
    dir_1d_mean = minmax_lambda([dir_1d_mean,lower,upper])
    
    
#     # test with corner specific vector fields
#     y,x = directions.shape[-2:]
    
#     vects_mean = minmax_lambda([np.mean(vects_nda,axis=(1,2,3,4)),lower,upper])
#     vects_1 = vects_nda[:,:,:y//2,:x//2,:]
#     vects_1_mean = minmax_lambda([np.mean(vects_1,axis=(1,2,3,4)),lower,upper])
#     vects_2 = vects_nda[:,:,y//2:,:x//2,:]
#     vects_2_mean = minmax_lambda([np.mean(vects_2,axis=(1,2,3,4)),lower,upper])
#     vects_3 = vects_nda[:,:,y//2:,x//2:,:]
#     vects_3_mean = minmax_lambda([np.mean(vects_3,axis=(1,2,3,4)),lower,upper])
#     vects_4 = vects_nda[:,:,:y//2,x//2:,:]
#     vects_4_mean = minmax_lambda([np.mean(vects_4,axis=(1,2,3,4)),lower,upper])
    
    
#     # direction per corner
#     dir_1 = directions[:,:,:y//2,:x//2] # top left, plot
#     dir_1_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_2 = directions[:,:,y//2:,:x//2] # top right, plot
#     dir_2_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_3 = directions[:,:,y//2:,x//2:] # bottom left
#     dir_3_mean = minmax_lambda([np.mean(dir_3,axis=(1,2,3)),lower,upper])
#     dir_4 = directions[:,:,:y//2,x//2:] # bottom right
#     dir_4_mean = minmax_lambda([np.mean(dir_4,axis=(1,2,3)),lower,upper])
    

#     # norm per corner
#     norm_mean = minmax_lambda([nda_1d_mean, mid, upper])
#     norm_1 = norm_nda[:,:,:y//2,:x//2]
#     norm_1_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_2 = norm_nda[:,:,y//2:,:x//2]
#     norm_2_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_3 = norm_nda[:,:,y//2:,x//2:]
#     norm_3_mean = minmax_lambda([np.mean(norm_3,axis=(1,2,3)),mid,upper])
#     norm_4 = norm_nda[:,:,:y//2,x//2:]
#     norm_4_mean = minmax_lambda([np.mean(norm_4,axis=(1,2,3)),mid,upper])
    

    ind = detect_phases(dir_1d_mean=dir_1d_mean[:length], length=length)
    
    
    #print('ed: {}, ms: {}, es: {}, pf: {}, md: {}'.format(*ind))
    
        # plot the mean/max norm for one patient oveer time
    if plot:
        
        # DIR 2D+t
        
        vect_2d_t = vects_nda[:,z]
        fig = show_2D_or_3D(vect_2d_t,allow_slicing=False, f_size=(25,2),cmap=div_cmap)
        
        dir_2d_t = directions[:,z]
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        #dir_2d_t = minmax_lambda([dir_2d_t,lower,upper])
        div_cmap = 'bwr'
        fig = show_2D_or_3D(dir_2d_t,allow_slicing=False, f_size=(25,2),cmap=div_cmap)
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('dir 2d+t\nmid-cavity')
        #ax.margins(0,0)
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[len(ax.get_images())//2], cax=cax, orientation='horizontal')

        

        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # DIR 2D T x Z
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        directions_tz = directions.mean(axis=(2,3)) # original: norm()
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        _ = ax.imshow(directions_tz.T, aspect='auto', label='aasd', cmap=div_cmap, origin='lower')
        _ = ax.set_xticks(gtind, minor=False)
        ax2 = ax.twinx()
        _ = ax2.plot(dir_1d_mean, c='black',label='dir 1d+t')
        #_ = ax.set_xticks(gtind, minor=False)
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('dir z+t\napex:base')
        #ax.legend(loc='upper left')
        
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1
        
        #ax=fig.add_subplot(rows,1,pos)
        #_ = ax.plot(dir_1d_mean, c='black',label='dir 1d+t')
        #_ = ax.plot(dir_1_mean, label='top left')
        #_ = ax.plot(dir_2_mean, label='top right')
        #_ = ax.plot(dir_3_mean, label='bottom left')
        #_ = ax.plot(dir_4_mean, label='bottom right')
        #ax.legend(loc='upper left')
        #ax.set_xticks(gtind, minor=False)
        #ax.margins(0.022,0)
        #ax.label_outer()
        #pos=pos+1
        #plt.subplots_adjust(wspace=0, hspace=0.1)
        #plt.show()
        norm_cmap = 'hot'
        
        # NORM 2D + t
        norm_2d_t = norm_nda[:,z]
        norm_2d_t = minmax_lambda([norm_2d_t,mid,upper])
        fig = show_2D_or_3D(norm_2d_t,allow_slicing=False, f_size=(25,2),cmap=norm_cmap, interpolation='none')
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('norm 2d+t\nmid-cavity')
        #ax.margins(0,0)
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[0], cax=cax, orientation='horizontal')
        
        
        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # NORM 2D TxZ
        norm2d = minmax_lambda([norm_nda.mean(axis=(2,3)),mid,upper])
        _ = ax.imshow(norm2d.T, aspect='auto', origin='lower',cmap=norm_cmap, interpolation='none')
        _ = ax.set_xticks(gtind, minor=False)
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('norm z+t\napex:base')
        ax2 = ax.twinx()
        _ = ax2.plot(nda_1d_mean, c='black',label='norm 1d+t')
        #ax.spines['top'].set_visible(False)
        #_ = ax.legend(loc='upper left')
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1

        """ax=fig.add_subplot(rows,1,pos)
        _ = ax.plot(norm_mean, label='norm 1d+t')
        #_ = ax.plot(norm_1_mean, label='top left')
        #_ = ax.plot(norm_2_mean, label='top right')
        #_ = ax.plot(norm_3_mean, label='bottom left')
        #_ = ax.plot(norm_4_mean, label='bottom right')
        ax.legend(loc='upper left')
        ax.set_xticks(gtind, minor=False)
        ax.margins(0.022,0)
        ax.label_outer()
        pos = pos+1
        plt.subplots_adjust(wspace=0, hspace=0.1)"""
        #plt.tight_layout()
        plt.show()

        
    
    return ind
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
p = 2#215
gt_length = int(gt_len[p,:,0].sum())
ind = np.argmax(gt[p][:gt_length],axis=0)
get_phases_from_vects(nda_vects[p][:gt_length], length=gt_length,plot=True,dir_axis=0, gtind=ind)
# try to predict the phases unsupervised just from the norm/direction curve

def get_phases_from_vects(vects_nda, length=-1, plot=True, dir_axis=0, gtind=[], figsize=(25,3)):
    import scipy.signal
    import scipy.signal as sig
    import scipy.ndimage
    from scipy.ndimage import gaussian_filter1d
    
    gted=gtind[0]
    
    # signal smoothing
    kernel_size = 4
    kernel = np.ones(kernel_size) / kernel_size
    z = 0
    
    
    dim_z = vects_nda.shape[1]
    #vects_nda = vects_nda[:int(length)]
    
    
    # define some helper lambdas
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    dim = vects_nda.shape[1:-1]
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]

    
    # norm of the vector
    norm_nda = np.linalg.norm(vects_nda[...,dir_axis:], axis=-1)
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    norm_nda_thres = clip_quantile(norm_nda, 0.9)
    norm_nda = minmax_lambda([norm_nda,0,1])
    norm_msk = norm_nda>=0.0
    norm_nda = norm_nda * norm_msk
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    nda_1d_mean = np.mean(norm_nda,axis=(1,2,3))
    nda_1d_max = np.max(norm_nda,axis=(1,2,3))
    nda_1d_median = np.median(norm_nda,axis=(1,2,3))
    nda_1d_mean_smooth = np.convolve(nda_1d_mean, kernel, mode='same')
    #print(nda_1d_mean_smooth.shape, nda_1d_mean.shape)
    
    
    # direction relative to the center
    
    directions = flow2direction_lambda(vects_nda)[...,0].numpy()
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    directions = minmax_lambda([directions,lower,upper])
    #plt.hist(directions.flatten());plt.show()
    directions = directions * norm_msk
    #plt.hist(directions.flatten());plt.show()
    
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    dir_1d_mean = np.mean(directions,axis=(1,2,3))
    dir_1d_median = np.median(directions,axis=(1,2,3))

    
    # Find phases
    # 1st smooth the direction with a rolling averge (kernelsize=4)
    # or via gaussian filter
    # 2nd min/max normalise each direction vector in a range [-1:1]
    
    
    dir_1d_mean = gaussian_filter1d(dir_1d_mean, sigma=2, mode='wrap') # original: dir_1d_mean
    
    ################# CHANGED !!!!!! ######################## RECENTLY NO CHANGES
    # min/max scaling
    dir_1d_mean = minmax_lambda([dir_1d_mean,lower,upper])
    
    
#     # test with corner specific vector fields
#     y,x = directions.shape[-2:]
    
#     vects_mean = minmax_lambda([np.mean(vects_nda,axis=(1,2,3,4)),lower,upper])
#     vects_1 = vects_nda[:,:,:y//2,:x//2,:]
#     vects_1_mean = minmax_lambda([np.mean(vects_1,axis=(1,2,3,4)),lower,upper])
#     vects_2 = vects_nda[:,:,y//2:,:x//2,:]
#     vects_2_mean = minmax_lambda([np.mean(vects_2,axis=(1,2,3,4)),lower,upper])
#     vects_3 = vects_nda[:,:,y//2:,x//2:,:]
#     vects_3_mean = minmax_lambda([np.mean(vects_3,axis=(1,2,3,4)),lower,upper])
#     vects_4 = vects_nda[:,:,:y//2,x//2:,:]
#     vects_4_mean = minmax_lambda([np.mean(vects_4,axis=(1,2,3,4)),lower,upper])
    
    
#     # direction per corner
#     dir_1 = directions[:,:,:y//2,:x//2] # top left, plot
#     dir_1_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_2 = directions[:,:,y//2:,:x//2] # top right, plot
#     dir_2_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_3 = directions[:,:,y//2:,x//2:] # bottom left
#     dir_3_mean = minmax_lambda([np.mean(dir_3,axis=(1,2,3)),lower,upper])
#     dir_4 = directions[:,:,:y//2,x//2:] # bottom right
#     dir_4_mean = minmax_lambda([np.mean(dir_4,axis=(1,2,3)),lower,upper])
    

#     # norm per corner
#     norm_mean = minmax_lambda([nda_1d_mean, mid, upper])
#     norm_1 = norm_nda[:,:,:y//2,:x//2]
#     norm_1_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_2 = norm_nda[:,:,y//2:,:x//2]
#     norm_2_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_3 = norm_nda[:,:,y//2:,x//2:]
#     norm_3_mean = minmax_lambda([np.mean(norm_3,axis=(1,2,3)),mid,upper])
#     norm_4 = norm_nda[:,:,:y//2,x//2:]
#     norm_4_mean = minmax_lambda([np.mean(norm_4,axis=(1,2,3)),mid,upper])
    

    ind = detect_phases(dir_1d_mean=dir_1d_mean[:length], length=length)
    
    
    #print('ed: {}, ms: {}, es: {}, pf: {}, md: {}'.format(*ind))
    
        # plot the mean/max norm for one patient oveer time
    if plot:
        
        # DIR 2D+t
        
        vect_2d_t = vects_nda[:,z]
        fig = show_2D_or_3D(vect_2d_t,allow_slicing=False, f_size=(25,2))
        
        dir_2d_t = directions[:,z]
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        #dir_2d_t = minmax_lambda([dir_2d_t,lower,upper])
        div_cmap = 'bwr'
        fig = show_2D_or_3D(dir_2d_t,allow_slicing=False, f_size=(25,2),cmap=div_cmap)
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('dir 2d+t\nmid-cavity')
        #ax.margins(0,0)
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[len(ax.get_images())//2], cax=cax, orientation='horizontal')

        

        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # DIR 2D T x Z
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        directions_tz = directions.mean(axis=(2,3)) # original: norm()
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        _ = ax.imshow(directions_tz.T, aspect='auto', label='aasd', cmap=div_cmap, origin='lower')
        _ = ax.set_xticks(gtind, minor=False)
        ax2 = ax.twinx()
        _ = ax2.plot(dir_1d_mean, c='black',label='dir 1d+t')
        #_ = ax.set_xticks(gtind, minor=False)
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('dir z+t\napex:base')
        #ax.legend(loc='upper left')
        
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1
        
        #ax=fig.add_subplot(rows,1,pos)
        #_ = ax.plot(dir_1d_mean, c='black',label='dir 1d+t')
        #_ = ax.plot(dir_1_mean, label='top left')
        #_ = ax.plot(dir_2_mean, label='top right')
        #_ = ax.plot(dir_3_mean, label='bottom left')
        #_ = ax.plot(dir_4_mean, label='bottom right')
        #ax.legend(loc='upper left')
        #ax.set_xticks(gtind, minor=False)
        #ax.margins(0.022,0)
        #ax.label_outer()
        #pos=pos+1
        #plt.subplots_adjust(wspace=0, hspace=0.1)
        #plt.show()
        norm_cmap = 'hot'
        
        # NORM 2D + t
        norm_2d_t = norm_nda[:,z]
        norm_2d_t = minmax_lambda([norm_2d_t,mid,upper])
        fig = show_2D_or_3D(norm_2d_t,allow_slicing=False, f_size=(25,2),cmap=norm_cmap, interpolation='none')
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('norm 2d+t\nmid-cavity')
        #ax.margins(0,0)
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[0], cax=cax, orientation='horizontal')
        
        
        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # NORM 2D TxZ
        norm2d = minmax_lambda([norm_nda.mean(axis=(2,3)),mid,upper])
        _ = ax.imshow(norm2d.T, aspect='auto', origin='lower',cmap=norm_cmap, interpolation='none')
        _ = ax.set_xticks(gtind, minor=False)
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('norm z+t\napex:base')
        ax2 = ax.twinx()
        _ = ax2.plot(nda_1d_mean, c='black',label='norm 1d+t')
        #ax.spines['top'].set_visible(False)
        #_ = ax.legend(loc='upper left')
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1

        """ax=fig.add_subplot(rows,1,pos)
        _ = ax.plot(norm_mean, label='norm 1d+t')
        #_ = ax.plot(norm_1_mean, label='top left')
        #_ = ax.plot(norm_2_mean, label='top right')
        #_ = ax.plot(norm_3_mean, label='bottom left')
        #_ = ax.plot(norm_4_mean, label='bottom right')
        ax.legend(loc='upper left')
        ax.set_xticks(gtind, minor=False)
        ax.margins(0.022,0)
        ax.label_outer()
        pos = pos+1
        plt.subplots_adjust(wspace=0, hspace=0.1)"""
        #plt.tight_layout()
        plt.show()

        
    
    return ind
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
p = 2#215
gt_length = int(gt_len[p,:,0].sum())
ind = np.argmax(gt[p][:gt_length],axis=0)
get_phases_from_vects(nda_vects[p][:gt_length], length=gt_length,plot=True,dir_axis=0, gtind=ind)
# try to predict the phases unsupervised just from the norm/direction curve

def get_phases_from_vects(vects_nda, length=-1, plot=True, dir_axis=0, gtind=[], figsize=(25,3)):
    import scipy.signal
    import scipy.signal as sig
    import scipy.ndimage
    from scipy.ndimage import gaussian_filter1d
    
    gted=gtind[0]
    
    # signal smoothing
    kernel_size = 4
    kernel = np.ones(kernel_size) / kernel_size
    z = 0
    
    
    dim_z = vects_nda.shape[1]
    #vects_nda = vects_nda[:int(length)]
    
    
    # define some helper lambdas
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    dim = vects_nda.shape[1:-1]
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]

    
    # norm of the vector
    norm_nda = np.linalg.norm(vects_nda[...,dir_axis:], axis=-1)
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    norm_nda_thres = clip_quantile(norm_nda, 0.9)
    norm_nda = minmax_lambda([norm_nda,0,1])
    norm_msk = norm_nda>=0.0
    norm_nda = norm_nda * norm_msk
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    nda_1d_mean = np.mean(norm_nda,axis=(1,2,3))
    nda_1d_max = np.max(norm_nda,axis=(1,2,3))
    nda_1d_median = np.median(norm_nda,axis=(1,2,3))
    nda_1d_mean_smooth = np.convolve(nda_1d_mean, kernel, mode='same')
    #print(nda_1d_mean_smooth.shape, nda_1d_mean.shape)
    
    
    # direction relative to the center
    
    directions = flow2direction_lambda(vects_nda)[...,0].numpy()
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    directions = minmax_lambda([directions,lower,upper])
    #plt.hist(directions.flatten());plt.show()
    directions = directions * norm_msk
    #plt.hist(directions.flatten());plt.show()
    
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    dir_1d_mean = np.mean(directions,axis=(1,2,3))
    dir_1d_median = np.median(directions,axis=(1,2,3))

    
    # Find phases
    # 1st smooth the direction with a rolling averge (kernelsize=4)
    # or via gaussian filter
    # 2nd min/max normalise each direction vector in a range [-1:1]
    
    
    dir_1d_mean = gaussian_filter1d(dir_1d_mean, sigma=2, mode='wrap') # original: dir_1d_mean
    
    ################# CHANGED !!!!!! ######################## RECENTLY NO CHANGES
    # min/max scaling
    dir_1d_mean = minmax_lambda([dir_1d_mean,lower,upper])
    
    
#     # test with corner specific vector fields
#     y,x = directions.shape[-2:]
    
#     vects_mean = minmax_lambda([np.mean(vects_nda,axis=(1,2,3,4)),lower,upper])
#     vects_1 = vects_nda[:,:,:y//2,:x//2,:]
#     vects_1_mean = minmax_lambda([np.mean(vects_1,axis=(1,2,3,4)),lower,upper])
#     vects_2 = vects_nda[:,:,y//2:,:x//2,:]
#     vects_2_mean = minmax_lambda([np.mean(vects_2,axis=(1,2,3,4)),lower,upper])
#     vects_3 = vects_nda[:,:,y//2:,x//2:,:]
#     vects_3_mean = minmax_lambda([np.mean(vects_3,axis=(1,2,3,4)),lower,upper])
#     vects_4 = vects_nda[:,:,:y//2,x//2:,:]
#     vects_4_mean = minmax_lambda([np.mean(vects_4,axis=(1,2,3,4)),lower,upper])
    
    
#     # direction per corner
#     dir_1 = directions[:,:,:y//2,:x//2] # top left, plot
#     dir_1_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_2 = directions[:,:,y//2:,:x//2] # top right, plot
#     dir_2_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_3 = directions[:,:,y//2:,x//2:] # bottom left
#     dir_3_mean = minmax_lambda([np.mean(dir_3,axis=(1,2,3)),lower,upper])
#     dir_4 = directions[:,:,:y//2,x//2:] # bottom right
#     dir_4_mean = minmax_lambda([np.mean(dir_4,axis=(1,2,3)),lower,upper])
    

#     # norm per corner
#     norm_mean = minmax_lambda([nda_1d_mean, mid, upper])
#     norm_1 = norm_nda[:,:,:y//2,:x//2]
#     norm_1_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_2 = norm_nda[:,:,y//2:,:x//2]
#     norm_2_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_3 = norm_nda[:,:,y//2:,x//2:]
#     norm_3_mean = minmax_lambda([np.mean(norm_3,axis=(1,2,3)),mid,upper])
#     norm_4 = norm_nda[:,:,:y//2,x//2:]
#     norm_4_mean = minmax_lambda([np.mean(norm_4,axis=(1,2,3)),mid,upper])
    

    ind = detect_phases(dir_1d_mean=dir_1d_mean[:length], length=length)
    
    
    #print('ed: {}, ms: {}, es: {}, pf: {}, md: {}'.format(*ind))
    
        # plot the mean/max norm for one patient oveer time
    if plot:
        
        # DIR 2D+t
        
        vect_2d_t = vects_nda[:,z]
        fig = show_2D_or_3D(vect_2d_t,allow_slicing=False, f_size=(25,2))
        
        dir_2d_t = directions[:,z]
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        #dir_2d_t = minmax_lambda([dir_2d_t,lower,upper])
        div_cmap = 'bwr'
        fig = show_2D_or_3D(dir_2d_t,allow_slicing=False, f_size=(25,2),cmap=div_cmap)
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('dir 2d+t\nmid-cavity')
        #ax.margins(0,0)
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[len(ax.get_images())//2], cax=cax, orientation='horizontal')

        

        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # DIR 2D T x Z
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        directions_tz = directions.mean(axis=(2,3)) # original: norm()
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        _ = ax.imshow(directions_tz.T, aspect='auto', label='aasd', cmap=div_cmap, origin='lower')
        _ = ax.set_xticks(gtind, minor=False)
        ax2 = ax.twinx()
        _ = ax2.plot(dir_1d_mean, c='black',label='dir 1d+t')
        #_ = ax.set_xticks(gtind, minor=False)
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('dir z+t\napex:base')
        #ax.legend(loc='upper left')
        
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1
        
        #ax=fig.add_subplot(rows,1,pos)
        #_ = ax.plot(dir_1d_mean, c='black',label='dir 1d+t')
        #_ = ax.plot(dir_1_mean, label='top left')
        #_ = ax.plot(dir_2_mean, label='top right')
        #_ = ax.plot(dir_3_mean, label='bottom left')
        #_ = ax.plot(dir_4_mean, label='bottom right')
        #ax.legend(loc='upper left')
        #ax.set_xticks(gtind, minor=False)
        #ax.margins(0.022,0)
        #ax.label_outer()
        #pos=pos+1
        #plt.subplots_adjust(wspace=0, hspace=0.1)
        #plt.show()
        norm_cmap = 'hot'
        
        # NORM 2D + t
        norm_2d_t = norm_nda[:,z]
        norm_2d_t = minmax_lambda([norm_2d_t,mid,upper])
        fig = show_2D_or_3D(norm_2d_t,allow_slicing=False, f_size=(25,2),cmap=norm_cmap, interpolation='none')
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('norm 2d+t\nmid-cavity')
        #ax.margins(0,0)
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[0], cax=cax, orientation='horizontal')
        
        
        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # NORM 2D TxZ
        norm2d = minmax_lambda([norm_nda.mean(axis=(2,3)),mid,upper])
        _ = ax.imshow(norm2d.T, aspect='auto', origin='lower',cmap=norm_cmap, interpolation='none')
        _ = ax.set_xticks(gtind, minor=False)
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('norm z+t\napex:base')
        ax2 = ax.twinx()
        _ = ax2.plot(nda_1d_mean, c='black',label='norm 1d+t')
        #ax.spines['top'].set_visible(False)
        #_ = ax.legend(loc='upper left')
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1

        """ax=fig.add_subplot(rows,1,pos)
        _ = ax.plot(norm_mean, label='norm 1d+t')
        #_ = ax.plot(norm_1_mean, label='top left')
        #_ = ax.plot(norm_2_mean, label='top right')
        #_ = ax.plot(norm_3_mean, label='bottom left')
        #_ = ax.plot(norm_4_mean, label='bottom right')
        ax.legend(loc='upper left')
        ax.set_xticks(gtind, minor=False)
        ax.margins(0.022,0)
        ax.label_outer()
        pos = pos+1
        plt.subplots_adjust(wspace=0, hspace=0.1)"""
        #plt.tight_layout()
        plt.show()

        
    
    return ind
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
p = 2#215
gt_length = int(gt_len[p,:,0].sum())
ind = np.argmax(gt[p][:gt_length],axis=0)
get_phases_from_vects(nda_vects[p][:gt_length], length=gt_length,plot=True,dir_axis=0, gtind=ind)
# try to predict the phases unsupervised just from the norm/direction curve

def get_phases_from_vects(vects_nda, length=-1, plot=True, dir_axis=0, gtind=[], figsize=(25,3)):
    import scipy.signal
    import scipy.signal as sig
    import scipy.ndimage
    from scipy.ndimage import gaussian_filter1d
    
    gted=gtind[0]
    
    # signal smoothing
    kernel_size = 4
    kernel = np.ones(kernel_size) / kernel_size
    z = 0
    
    
    dim_z = vects_nda.shape[1]
    #vects_nda = vects_nda[:int(length)]
    
    
    # define some helper lambdas
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    dim = vects_nda.shape[1:-1]
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]

    
    # norm of the vector
    norm_nda = np.linalg.norm(vects_nda[...,dir_axis:], axis=-1)
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    norm_nda_thres = clip_quantile(norm_nda, 0.9)
    norm_nda = minmax_lambda([norm_nda,0,1])
    norm_msk = norm_nda>=0.0
    norm_nda = norm_nda * norm_msk
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    nda_1d_mean = np.mean(norm_nda,axis=(1,2,3))
    nda_1d_max = np.max(norm_nda,axis=(1,2,3))
    nda_1d_median = np.median(norm_nda,axis=(1,2,3))
    nda_1d_mean_smooth = np.convolve(nda_1d_mean, kernel, mode='same')
    #print(nda_1d_mean_smooth.shape, nda_1d_mean.shape)
    
    
    # direction relative to the center
    
    directions = flow2direction_lambda(vects_nda)[...,0].numpy()
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    directions = minmax_lambda([directions,lower,upper])
    #plt.hist(directions.flatten());plt.show()
    directions = directions * norm_msk
    #plt.hist(directions.flatten());plt.show()
    
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    dir_1d_mean = np.mean(directions,axis=(1,2,3))
    dir_1d_median = np.median(directions,axis=(1,2,3))

    
    # Find phases
    # 1st smooth the direction with a rolling averge (kernelsize=4)
    # or via gaussian filter
    # 2nd min/max normalise each direction vector in a range [-1:1]
    
    
    dir_1d_mean = gaussian_filter1d(dir_1d_mean, sigma=2, mode='wrap') # original: dir_1d_mean
    
    ################# CHANGED !!!!!! ######################## RECENTLY NO CHANGES
    # min/max scaling
    dir_1d_mean = minmax_lambda([dir_1d_mean,lower,upper])
    
    
#     # test with corner specific vector fields
#     y,x = directions.shape[-2:]
    
#     vects_mean = minmax_lambda([np.mean(vects_nda,axis=(1,2,3,4)),lower,upper])
#     vects_1 = vects_nda[:,:,:y//2,:x//2,:]
#     vects_1_mean = minmax_lambda([np.mean(vects_1,axis=(1,2,3,4)),lower,upper])
#     vects_2 = vects_nda[:,:,y//2:,:x//2,:]
#     vects_2_mean = minmax_lambda([np.mean(vects_2,axis=(1,2,3,4)),lower,upper])
#     vects_3 = vects_nda[:,:,y//2:,x//2:,:]
#     vects_3_mean = minmax_lambda([np.mean(vects_3,axis=(1,2,3,4)),lower,upper])
#     vects_4 = vects_nda[:,:,:y//2,x//2:,:]
#     vects_4_mean = minmax_lambda([np.mean(vects_4,axis=(1,2,3,4)),lower,upper])
    
    
#     # direction per corner
#     dir_1 = directions[:,:,:y//2,:x//2] # top left, plot
#     dir_1_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_2 = directions[:,:,y//2:,:x//2] # top right, plot
#     dir_2_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_3 = directions[:,:,y//2:,x//2:] # bottom left
#     dir_3_mean = minmax_lambda([np.mean(dir_3,axis=(1,2,3)),lower,upper])
#     dir_4 = directions[:,:,:y//2,x//2:] # bottom right
#     dir_4_mean = minmax_lambda([np.mean(dir_4,axis=(1,2,3)),lower,upper])
    

#     # norm per corner
#     norm_mean = minmax_lambda([nda_1d_mean, mid, upper])
#     norm_1 = norm_nda[:,:,:y//2,:x//2]
#     norm_1_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_2 = norm_nda[:,:,y//2:,:x//2]
#     norm_2_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_3 = norm_nda[:,:,y//2:,x//2:]
#     norm_3_mean = minmax_lambda([np.mean(norm_3,axis=(1,2,3)),mid,upper])
#     norm_4 = norm_nda[:,:,:y//2,x//2:]
#     norm_4_mean = minmax_lambda([np.mean(norm_4,axis=(1,2,3)),mid,upper])
    

    ind = detect_phases(dir_1d_mean=dir_1d_mean[:length], length=length)
    
    
    #print('ed: {}, ms: {}, es: {}, pf: {}, md: {}'.format(*ind))
    
        # plot the mean/max norm for one patient oveer time
    if plot:
        
        # DIR 2D+t
        
        vect_2d_t = vects_nda[:,z]
        fig = show_2D_or_3D(vect_2d_t,allow_slicing=False, f_size=(25,2))
        
        dir_2d_t = directions[:,z]
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        #dir_2d_t = minmax_lambda([dir_2d_t,lower,upper])
        div_cmap = 'bwr'
        fig = show_2D_or_3D(dir_2d_t,allow_slicing=False, f_size=(25,2),cmap=div_cmap)
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('dir 2d+t\nmid-cavity')
        #ax.margins(0,0)
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[len(ax.get_images())//2], cax=cax, orientation='horizontal')

        

        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # DIR 2D T x Z
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        directions_tz = directions.mean(axis=(2,3)) # original: norm()
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        _ = ax.imshow(directions_tz.T, aspect='auto', label='aasd', cmap=div_cmap, origin='lower')
        _ = ax.set_xticks(gtind, minor=False)
        ax2 = ax.twinx()
        _ = ax2.plot(dir_1d_mean, c='black',label='dir 1d+t')
        #_ = ax.set_xticks(gtind, minor=False)
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('dir z+t\napex:base')
        #ax.legend(loc='upper left')
        
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1
        
        #ax=fig.add_subplot(rows,1,pos)
        #_ = ax.plot(dir_1d_mean, c='black',label='dir 1d+t')
        #_ = ax.plot(dir_1_mean, label='top left')
        #_ = ax.plot(dir_2_mean, label='top right')
        #_ = ax.plot(dir_3_mean, label='bottom left')
        #_ = ax.plot(dir_4_mean, label='bottom right')
        #ax.legend(loc='upper left')
        #ax.set_xticks(gtind, minor=False)
        #ax.margins(0.022,0)
        #ax.label_outer()
        #pos=pos+1
        #plt.subplots_adjust(wspace=0, hspace=0.1)
        #plt.show()
        norm_cmap = 'hot'
        
        # NORM 2D + t
        norm_2d_t = norm_nda[:,z]
        norm_2d_t = minmax_lambda([norm_2d_t,mid,upper])
        fig = show_2D_or_3D(norm_2d_t,allow_slicing=False, f_size=(25,2),cmap=norm_cmap, interpolation='none')
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('norm 2d+t\nmid-cavity')
        #ax.margins(0,0)
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[0], cax=cax, orientation='horizontal')
        
        
        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # NORM 2D TxZ
        norm2d = minmax_lambda([norm_nda.mean(axis=(2,3)),mid,upper])
        _ = ax.imshow(norm2d.T, aspect='auto', origin='lower',cmap=norm_cmap, interpolation='none')
        _ = ax.set_xticks(gtind, minor=False)
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('norm z+t\napex:base')
        ax2 = ax.twinx()
        _ = ax2.plot(nda_1d_mean, c='black',label='norm 1d+t')
        #ax.spines['top'].set_visible(False)
        #_ = ax.legend(loc='upper left')
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1

        """ax=fig.add_subplot(rows,1,pos)
        _ = ax.plot(norm_mean, label='norm 1d+t')
        #_ = ax.plot(norm_1_mean, label='top left')
        #_ = ax.plot(norm_2_mean, label='top right')
        #_ = ax.plot(norm_3_mean, label='bottom left')
        #_ = ax.plot(norm_4_mean, label='bottom right')
        ax.legend(loc='upper left')
        ax.set_xticks(gtind, minor=False)
        ax.margins(0.022,0)
        ax.label_outer()
        pos = pos+1
        plt.subplots_adjust(wspace=0, hspace=0.1)"""
        #plt.tight_layout()
        plt.show()

        
    
    return ind
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
p = 2#215
gt_length = int(gt_len[p,:,0].sum())
ind = np.argmax(gt[p][:gt_length],axis=0)
get_phases_from_vects(nda_vects[p][:gt_length], length=gt_length,plot=True,dir_axis=0, gtind=ind)
# try to predict the phases unsupervised just from the norm/direction curve

def get_phases_from_vects(vects_nda, length=-1, plot=True, dir_axis=0, gtind=[], figsize=(25,3)):
    import scipy.signal
    import scipy.signal as sig
    import scipy.ndimage
    from scipy.ndimage import gaussian_filter1d
    
    gted=gtind[0]
    
    # signal smoothing
    kernel_size = 4
    kernel = np.ones(kernel_size) / kernel_size
    z = 0
    
    
    dim_z = vects_nda.shape[1]
    #vects_nda = vects_nda[:int(length)]
    
    
    # define some helper lambdas
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    dim = vects_nda.shape[1:-1]
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]

    
    # norm of the vector
    norm_nda = np.linalg.norm(vects_nda[...,dir_axis:], axis=-1)
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    norm_nda_thres = clip_quantile(norm_nda, 0.9)
    norm_nda = minmax_lambda([norm_nda,0,1])
    norm_msk = norm_nda>=0.0
    norm_nda = norm_nda * norm_msk
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    nda_1d_mean = np.mean(norm_nda,axis=(1,2,3))
    nda_1d_max = np.max(norm_nda,axis=(1,2,3))
    nda_1d_median = np.median(norm_nda,axis=(1,2,3))
    nda_1d_mean_smooth = np.convolve(nda_1d_mean, kernel, mode='same')
    #print(nda_1d_mean_smooth.shape, nda_1d_mean.shape)
    
    
    # direction relative to the center
    
    directions = flow2direction_lambda(vects_nda)[...,0].numpy()
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    directions = minmax_lambda([directions,lower,upper])
    #plt.hist(directions.flatten());plt.show()
    directions = directions * norm_msk
    #plt.hist(directions.flatten());plt.show()
    
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    dir_1d_mean = np.mean(directions,axis=(1,2,3))
    dir_1d_median = np.median(directions,axis=(1,2,3))

    
    # Find phases
    # 1st smooth the direction with a rolling averge (kernelsize=4)
    # or via gaussian filter
    # 2nd min/max normalise each direction vector in a range [-1:1]
    
    
    dir_1d_mean = gaussian_filter1d(dir_1d_mean, sigma=2, mode='wrap') # original: dir_1d_mean
    
    ################# CHANGED !!!!!! ######################## RECENTLY NO CHANGES
    # min/max scaling
    dir_1d_mean = minmax_lambda([dir_1d_mean,lower,upper])
    
    
#     # test with corner specific vector fields
#     y,x = directions.shape[-2:]
    
#     vects_mean = minmax_lambda([np.mean(vects_nda,axis=(1,2,3,4)),lower,upper])
#     vects_1 = vects_nda[:,:,:y//2,:x//2,:]
#     vects_1_mean = minmax_lambda([np.mean(vects_1,axis=(1,2,3,4)),lower,upper])
#     vects_2 = vects_nda[:,:,y//2:,:x//2,:]
#     vects_2_mean = minmax_lambda([np.mean(vects_2,axis=(1,2,3,4)),lower,upper])
#     vects_3 = vects_nda[:,:,y//2:,x//2:,:]
#     vects_3_mean = minmax_lambda([np.mean(vects_3,axis=(1,2,3,4)),lower,upper])
#     vects_4 = vects_nda[:,:,:y//2,x//2:,:]
#     vects_4_mean = minmax_lambda([np.mean(vects_4,axis=(1,2,3,4)),lower,upper])
    
    
#     # direction per corner
#     dir_1 = directions[:,:,:y//2,:x//2] # top left, plot
#     dir_1_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_2 = directions[:,:,y//2:,:x//2] # top right, plot
#     dir_2_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_3 = directions[:,:,y//2:,x//2:] # bottom left
#     dir_3_mean = minmax_lambda([np.mean(dir_3,axis=(1,2,3)),lower,upper])
#     dir_4 = directions[:,:,:y//2,x//2:] # bottom right
#     dir_4_mean = minmax_lambda([np.mean(dir_4,axis=(1,2,3)),lower,upper])
    

#     # norm per corner
#     norm_mean = minmax_lambda([nda_1d_mean, mid, upper])
#     norm_1 = norm_nda[:,:,:y//2,:x//2]
#     norm_1_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_2 = norm_nda[:,:,y//2:,:x//2]
#     norm_2_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_3 = norm_nda[:,:,y//2:,x//2:]
#     norm_3_mean = minmax_lambda([np.mean(norm_3,axis=(1,2,3)),mid,upper])
#     norm_4 = norm_nda[:,:,:y//2,x//2:]
#     norm_4_mean = minmax_lambda([np.mean(norm_4,axis=(1,2,3)),mid,upper])
    

    ind = detect_phases(dir_1d_mean=dir_1d_mean[:length], length=length)
    
    
    #print('ed: {}, ms: {}, es: {}, pf: {}, md: {}'.format(*ind))
    
        # plot the mean/max norm for one patient oveer time
    if plot:
        
        # DIR 2D+t
        
        vect_2d_t = vects_nda[:,z]
        fig = show_2D_or_3D(vect_2d_t,allow_slicing=False, f_size=(25,2))
        
        dir_2d_t = directions[:,z]
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        #dir_2d_t = minmax_lambda([dir_2d_t,lower,upper])
        div_cmap = 'bwr'
        fig = show_2D_or_3D(dir_2d_t,allow_slicing=False, f_size=(25,2),cmap=div_cmap)
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('dir 2d+t\nmid-cavity')
        #ax.margins(0,0)
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[len(ax.get_images())//2], cax=cax, orientation='horizontal')

        

        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # DIR 2D T x Z
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        directions_tz = directions.mean(axis=(2,3)) # original: norm()
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        _ = ax.imshow(directions_tz.T, aspect='auto', label='aasd', cmap=div_cmap, origin='lower')
        _ = ax.set_xticks(gtind, minor=False)
        ax2 = ax.twinx()
        _ = ax2.plot(dir_1d_mean, c='black',label='dir 1d+t')
        #_ = ax.set_xticks(gtind, minor=False)
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('dir z+t\napex:base')
        #ax.legend(loc='upper left')
        
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1
        
        #ax=fig.add_subplot(rows,1,pos)
        #_ = ax.plot(dir_1d_mean, c='black',label='dir 1d+t')
        #_ = ax.plot(dir_1_mean, label='top left')
        #_ = ax.plot(dir_2_mean, label='top right')
        #_ = ax.plot(dir_3_mean, label='bottom left')
        #_ = ax.plot(dir_4_mean, label='bottom right')
        #ax.legend(loc='upper left')
        #ax.set_xticks(gtind, minor=False)
        #ax.margins(0.022,0)
        #ax.label_outer()
        #pos=pos+1
        #plt.subplots_adjust(wspace=0, hspace=0.1)
        #plt.show()
        norm_cmap = 'hot'
        
        # NORM 2D + t
        norm_2d_t = norm_nda[:,z]
        norm_2d_t = minmax_lambda([norm_2d_t,mid,upper])
        fig = show_2D_or_3D(norm_2d_t,allow_slicing=False, f_size=(25,2),cmap=norm_cmap, interpolation='none')
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('norm 2d+t\nmid-cavity')
        #ax.margins(0,0)
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[0], cax=cax, orientation='horizontal')
        
        
        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # NORM 2D TxZ
        norm2d = minmax_lambda([norm_nda.mean(axis=(2,3)),mid,upper])
        _ = ax.imshow(norm2d.T, aspect='auto', origin='lower',cmap=norm_cmap, interpolation='none')
        _ = ax.set_xticks(gtind, minor=False)
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('norm z+t\napex:base')
        ax2 = ax.twinx()
        _ = ax2.plot(nda_1d_mean, c='black',label='norm 1d+t')
        #ax.spines['top'].set_visible(False)
        #_ = ax.legend(loc='upper left')
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1

        """ax=fig.add_subplot(rows,1,pos)
        _ = ax.plot(norm_mean, label='norm 1d+t')
        #_ = ax.plot(norm_1_mean, label='top left')
        #_ = ax.plot(norm_2_mean, label='top right')
        #_ = ax.plot(norm_3_mean, label='bottom left')
        #_ = ax.plot(norm_4_mean, label='bottom right')
        ax.legend(loc='upper left')
        ax.set_xticks(gtind, minor=False)
        ax.margins(0.022,0)
        ax.label_outer()
        pos = pos+1
        plt.subplots_adjust(wspace=0, hspace=0.1)"""
        #plt.tight_layout()
        plt.show()

        
    
    return ind
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
p = 2#215
gt_length = int(gt_len[p,:,0].sum())
ind = np.argmax(gt[p][:gt_length],axis=0)
get_phases_from_vects(nda_vects[p][:gt_length], length=gt_length,plot=True,dir_axis=0, gtind=ind)
@interact
def compare_phases(i=(0,nda_vects.shape[0]-1), plot=True):
    import matplotlib as mpl
    mpl.rcParams.update(mpl.rcParamsDefault)
    figsize = (25,1)
    weight = 1
    z = 0
    
    phases = ['ED', 'MS', 'ES', 'PF', 'MD']
    cardiac_cycle_length = int(gt_len[i,:,-1].sum())
    # crop or not
    cardiac_cycle_length_ = gt[i].shape[0]-3 # plot the full length minus border for visualisation
    
    vects = nda_vects[i][:cardiac_cycle_length]
    
    
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0) # 
    ind_pred = np.argmax(pred[i][:],axis=0)
    
    temp = mov[i,:cardiac_cycle_length,z] # cardiac_cycle_length_
    fig = show_2D_or_3D(temp,allow_slicing=False)
    ax = fig.get_axes()[0]
    _ = ax.set_ylabel('CMR 2d+t)\nmid-cavity')
    plt.show()
    
    fig= plt.figure(figsize=figsize)
    ax=fig.add_subplot(111)
    ax.set_xticks(ind, minor=False)
    ax.set_yticks([0, 1, 2, 3, 4], minor=False)
    ax.set_yticklabels(phases, rotation=0)
    ind = np.array(ind)
    onehot_gt = np.zeros((ind.size, cardiac_cycle_length)) # cardiac_cycle_length_
    onehot_gt[np.arange(ind.size), ind] = weight
    #ax.imshow(gt[i][:cardiac_cycle_length].T,aspect='auto') # with uncertainity
    ax.imshow(onehot_gt,aspect='auto', cmap='bone') # gt as binary phase2time mapping
    _ = ax.set_yticklabels(phases)
    ax.set_title('gt')
    ax.margins(0,0)
    
    indices = get_phases_from_vects(vects, length=cardiac_cycle_length, plot=plot,dir_axis=0, gtind=ind, figsize=figsize)

    indices = np.array(indices)
    onehot = np.zeros((indices.size, cardiac_cycle_length)) # cardiac_cycle_length_
    onehot[np.arange(indices.size), indices] = weight
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xticks(indices, minor=False)
    ax.set_yticks([0, 1, 2, 3, 4], minor=False)
    ax.set_yticklabels(phases, rotation=0)
    plt.imshow(onehot,aspect='auto', cmap='bone')
    _ = ax.set_yticklabels(phases)
    ax.margins(0,0)
    #ax.set_title('prediction')
    plt.show()
    
    print(phases)
    print('gt:', ind)
    print('u: ', indices)
    
    
    # this would plot the supervised prediction
    #print('p: ', ind_pred)
    """fig, ax = plt.subplots(figsize=figsize)
    ax.set_xticks(ind_pred, minor=False)
    ax.set_yticks([0, 1, 2, 3, 4], minor=False)
    ax.set_yticklabels(phases, rotation=45)
    plt.imshow(pred[i][:cardiac_cycle_length].T,aspect='auto')
    _ = ax.set_yticklabels(phases);plt.show()"""
    # patient 107 gcn --> good curve for plotting
    # 200 no cut
    # 125 no cut
    # 194 moderate cut 
    # 134 strong cut-off
# try to predict the phases unsupervised just from the norm/direction curve

def get_phases_from_vects(vects_nda, length=-1, plot=True, dir_axis=0, gtind=[], figsize=(25,3)):
    import scipy.signal
    import scipy.signal as sig
    import scipy.ndimage
    from scipy.ndimage import gaussian_filter1d
    
    gted=gtind[0]
    
    # signal smoothing
    kernel_size = 4
    kernel = np.ones(kernel_size) / kernel_size
    z = 0
    
    
    dim_z = vects_nda.shape[1]
    #vects_nda = vects_nda[:int(length)]
    
    
    # define some helper lambdas
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    dim = vects_nda.shape[1:-1]
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]

    
    # norm of the vector
    norm_nda = np.linalg.norm(vects_nda[...,dir_axis:], axis=-1)
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    norm_nda_thres = clip_quantile(norm_nda, 0.9)
    norm_nda = minmax_lambda([norm_nda,0,1])
    norm_msk = norm_nda>=0.0
    norm_nda = norm_nda * norm_msk
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    nda_1d_mean = np.mean(norm_nda,axis=(1,2,3))
    nda_1d_max = np.max(norm_nda,axis=(1,2,3))
    nda_1d_median = np.median(norm_nda,axis=(1,2,3))
    nda_1d_mean_smooth = np.convolve(nda_1d_mean, kernel, mode='same')
    #print(nda_1d_mean_smooth.shape, nda_1d_mean.shape)
    
    
    # direction relative to the center
    
    directions = flow2direction_lambda(vects_nda)[...,0].numpy()
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    directions = minmax_lambda([directions,lower,upper])
    #plt.hist(directions.flatten());plt.show()
    directions = directions * norm_msk
    #plt.hist(directions.flatten());plt.show()
    
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    dir_1d_mean = np.mean(directions,axis=(1,2,3))
    dir_1d_median = np.median(directions,axis=(1,2,3))

    
    # Find phases
    # 1st smooth the direction with a rolling averge (kernelsize=4)
    # or via gaussian filter
    # 2nd min/max normalise each direction vector in a range [-1:1]
    
    
    dir_1d_mean = gaussian_filter1d(dir_1d_mean, sigma=2, mode='wrap') # original: dir_1d_mean
    
    ################# CHANGED !!!!!! ######################## RECENTLY NO CHANGES
    # min/max scaling
    dir_1d_mean = minmax_lambda([dir_1d_mean,lower,upper])
    
    
#     # test with corner specific vector fields
#     y,x = directions.shape[-2:]
    
#     vects_mean = minmax_lambda([np.mean(vects_nda,axis=(1,2,3,4)),lower,upper])
#     vects_1 = vects_nda[:,:,:y//2,:x//2,:]
#     vects_1_mean = minmax_lambda([np.mean(vects_1,axis=(1,2,3,4)),lower,upper])
#     vects_2 = vects_nda[:,:,y//2:,:x//2,:]
#     vects_2_mean = minmax_lambda([np.mean(vects_2,axis=(1,2,3,4)),lower,upper])
#     vects_3 = vects_nda[:,:,y//2:,x//2:,:]
#     vects_3_mean = minmax_lambda([np.mean(vects_3,axis=(1,2,3,4)),lower,upper])
#     vects_4 = vects_nda[:,:,:y//2,x//2:,:]
#     vects_4_mean = minmax_lambda([np.mean(vects_4,axis=(1,2,3,4)),lower,upper])
    
    
#     # direction per corner
#     dir_1 = directions[:,:,:y//2,:x//2] # top left, plot
#     dir_1_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_2 = directions[:,:,y//2:,:x//2] # top right, plot
#     dir_2_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_3 = directions[:,:,y//2:,x//2:] # bottom left
#     dir_3_mean = minmax_lambda([np.mean(dir_3,axis=(1,2,3)),lower,upper])
#     dir_4 = directions[:,:,:y//2,x//2:] # bottom right
#     dir_4_mean = minmax_lambda([np.mean(dir_4,axis=(1,2,3)),lower,upper])
    

#     # norm per corner
#     norm_mean = minmax_lambda([nda_1d_mean, mid, upper])
#     norm_1 = norm_nda[:,:,:y//2,:x//2]
#     norm_1_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_2 = norm_nda[:,:,y//2:,:x//2]
#     norm_2_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_3 = norm_nda[:,:,y//2:,x//2:]
#     norm_3_mean = minmax_lambda([np.mean(norm_3,axis=(1,2,3)),mid,upper])
#     norm_4 = norm_nda[:,:,:y//2,x//2:]
#     norm_4_mean = minmax_lambda([np.mean(norm_4,axis=(1,2,3)),mid,upper])
    

    ind = detect_phases(dir_1d_mean=dir_1d_mean[:length], length=length)
    
    
    #print('ed: {}, ms: {}, es: {}, pf: {}, md: {}'.format(*ind))
    
        # plot the mean/max norm for one patient oveer time
    if plot:
        
        
        # VECT 2D+t
        vect_2d_t = vects_nda[:,z]
        fig = show_2D_or_3D(vect_2d_t,allow_slicing=False, f_size=(25,2))
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('v_t \nmid-cavity')
        #ax.margins(0,0)
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[len(ax.get_images())//2], cax=cax, orientation='horizontal')
        
        # DIR 2D+t
        dir_2d_t = directions[:,z]
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        #dir_2d_t = minmax_lambda([dir_2d_t,lower,upper])
        div_cmap = 'bwr'
        fig = show_2D_or_3D(dir_2d_t,allow_slicing=False, f_size=(25,2),cmap=div_cmap)
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel(r'\alpha_t \nmid-cavity')
        #ax.margins(0,0)
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[len(ax.get_images())//2], cax=cax, orientation='horizontal')

        

        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # DIR 2D T x Z
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        directions_tz = directions.mean(axis=(2,3)) # original: norm()
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        _ = ax.imshow(directions_tz.T, aspect='auto', label='aasd', cmap=div_cmap, origin='lower')
        _ = ax.set_xticks(gtind, minor=False)
        ax2 = ax.twinx()
        _ = ax2.plot(dir_1d_mean, c='black',label='dir 1d+t')
        #_ = ax.set_xticks(gtind, minor=False)
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('dir z+t\napex:base')
        #ax.legend(loc='upper left')
        
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1
        
        #ax=fig.add_subplot(rows,1,pos)
        #_ = ax.plot(dir_1d_mean, c='black',label='dir 1d+t')
        #_ = ax.plot(dir_1_mean, label='top left')
        #_ = ax.plot(dir_2_mean, label='top right')
        #_ = ax.plot(dir_3_mean, label='bottom left')
        #_ = ax.plot(dir_4_mean, label='bottom right')
        #ax.legend(loc='upper left')
        #ax.set_xticks(gtind, minor=False)
        #ax.margins(0.022,0)
        #ax.label_outer()
        #pos=pos+1
        #plt.subplots_adjust(wspace=0, hspace=0.1)
        #plt.show()
        norm_cmap = 'hot'
        
        # NORM 2D + t
        norm_2d_t = norm_nda[:,z]
        norm_2d_t = minmax_lambda([norm_2d_t,mid,upper])
        fig = show_2D_or_3D(norm_2d_t,allow_slicing=False, f_size=(25,2),cmap=norm_cmap, interpolation='none')
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('norm 2d+t\nmid-cavity')
        #ax.margins(0,0)
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[0], cax=cax, orientation='horizontal')
        
        
        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # NORM 2D TxZ
        norm2d = minmax_lambda([norm_nda.mean(axis=(2,3)),mid,upper])
        _ = ax.imshow(norm2d.T, aspect='auto', origin='lower',cmap=norm_cmap, interpolation='none')
        _ = ax.set_xticks(gtind, minor=False)
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('norm z+t\napex:base')
        ax2 = ax.twinx()
        _ = ax2.plot(nda_1d_mean, c='black',label='norm 1d+t')
        #ax.spines['top'].set_visible(False)
        #_ = ax.legend(loc='upper left')
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1

        """ax=fig.add_subplot(rows,1,pos)
        _ = ax.plot(norm_mean, label='norm 1d+t')
        #_ = ax.plot(norm_1_mean, label='top left')
        #_ = ax.plot(norm_2_mean, label='top right')
        #_ = ax.plot(norm_3_mean, label='bottom left')
        #_ = ax.plot(norm_4_mean, label='bottom right')
        ax.legend(loc='upper left')
        ax.set_xticks(gtind, minor=False)
        ax.margins(0.022,0)
        ax.label_outer()
        pos = pos+1
        plt.subplots_adjust(wspace=0, hspace=0.1)"""
        #plt.tight_layout()
        plt.show()

        
    
    return ind
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
p = 2#215
gt_length = int(gt_len[p,:,0].sum())
ind = np.argmax(gt[p][:gt_length],axis=0)
get_phases_from_vects(nda_vects[p][:gt_length], length=gt_length,plot=True,dir_axis=0, gtind=ind)
# try to predict the phases unsupervised just from the norm/direction curve

def get_phases_from_vects(vects_nda, length=-1, plot=True, dir_axis=0, gtind=[], figsize=(25,3)):
    import scipy.signal
    import scipy.signal as sig
    import scipy.ndimage
    from scipy.ndimage import gaussian_filter1d
    
    gted=gtind[0]
    
    # signal smoothing
    kernel_size = 4
    kernel = np.ones(kernel_size) / kernel_size
    z = 0
    
    
    dim_z = vects_nda.shape[1]
    #vects_nda = vects_nda[:int(length)]
    
    
    # define some helper lambdas
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    dim = vects_nda.shape[1:-1]
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]

    
    # norm of the vector
    norm_nda = np.linalg.norm(vects_nda[...,dir_axis:], axis=-1)
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    norm_nda_thres = clip_quantile(norm_nda, 0.9)
    norm_nda = minmax_lambda([norm_nda,0,1])
    norm_msk = norm_nda>=0.0
    norm_nda = norm_nda * norm_msk
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    nda_1d_mean = np.mean(norm_nda,axis=(1,2,3))
    nda_1d_max = np.max(norm_nda,axis=(1,2,3))
    nda_1d_median = np.median(norm_nda,axis=(1,2,3))
    nda_1d_mean_smooth = np.convolve(nda_1d_mean, kernel, mode='same')
    #print(nda_1d_mean_smooth.shape, nda_1d_mean.shape)
    
    
    # direction relative to the center
    
    directions = flow2direction_lambda(vects_nda)[...,0].numpy()
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    directions = minmax_lambda([directions,lower,upper])
    #plt.hist(directions.flatten());plt.show()
    directions = directions * norm_msk
    #plt.hist(directions.flatten());plt.show()
    
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    dir_1d_mean = np.mean(directions,axis=(1,2,3))
    dir_1d_median = np.median(directions,axis=(1,2,3))

    
    # Find phases
    # 1st smooth the direction with a rolling averge (kernelsize=4)
    # or via gaussian filter
    # 2nd min/max normalise each direction vector in a range [-1:1]
    
    
    dir_1d_mean = gaussian_filter1d(dir_1d_mean, sigma=2, mode='wrap') # original: dir_1d_mean
    
    ################# CHANGED !!!!!! ######################## RECENTLY NO CHANGES
    # min/max scaling
    dir_1d_mean = minmax_lambda([dir_1d_mean,lower,upper])
    
    
#     # test with corner specific vector fields
#     y,x = directions.shape[-2:]
    
#     vects_mean = minmax_lambda([np.mean(vects_nda,axis=(1,2,3,4)),lower,upper])
#     vects_1 = vects_nda[:,:,:y//2,:x//2,:]
#     vects_1_mean = minmax_lambda([np.mean(vects_1,axis=(1,2,3,4)),lower,upper])
#     vects_2 = vects_nda[:,:,y//2:,:x//2,:]
#     vects_2_mean = minmax_lambda([np.mean(vects_2,axis=(1,2,3,4)),lower,upper])
#     vects_3 = vects_nda[:,:,y//2:,x//2:,:]
#     vects_3_mean = minmax_lambda([np.mean(vects_3,axis=(1,2,3,4)),lower,upper])
#     vects_4 = vects_nda[:,:,:y//2,x//2:,:]
#     vects_4_mean = minmax_lambda([np.mean(vects_4,axis=(1,2,3,4)),lower,upper])
    
    
#     # direction per corner
#     dir_1 = directions[:,:,:y//2,:x//2] # top left, plot
#     dir_1_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_2 = directions[:,:,y//2:,:x//2] # top right, plot
#     dir_2_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_3 = directions[:,:,y//2:,x//2:] # bottom left
#     dir_3_mean = minmax_lambda([np.mean(dir_3,axis=(1,2,3)),lower,upper])
#     dir_4 = directions[:,:,:y//2,x//2:] # bottom right
#     dir_4_mean = minmax_lambda([np.mean(dir_4,axis=(1,2,3)),lower,upper])
    

#     # norm per corner
#     norm_mean = minmax_lambda([nda_1d_mean, mid, upper])
#     norm_1 = norm_nda[:,:,:y//2,:x//2]
#     norm_1_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_2 = norm_nda[:,:,y//2:,:x//2]
#     norm_2_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_3 = norm_nda[:,:,y//2:,x//2:]
#     norm_3_mean = minmax_lambda([np.mean(norm_3,axis=(1,2,3)),mid,upper])
#     norm_4 = norm_nda[:,:,:y//2,x//2:]
#     norm_4_mean = minmax_lambda([np.mean(norm_4,axis=(1,2,3)),mid,upper])
    

    ind = detect_phases(dir_1d_mean=dir_1d_mean[:length], length=length)
    
    
    #print('ed: {}, ms: {}, es: {}, pf: {}, md: {}'.format(*ind))
    
        # plot the mean/max norm for one patient oveer time
    if plot:
        
        
        # VECT 2D+t
        vect_2d_t = vects_nda[:,z]
        fig = show_2D_or_3D(vect_2d_t,allow_slicing=False, f_size=(25,2))
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('v_t \nmid-cavity')

        
        # DIR 2D+t
        dir_2d_t = directions[:,z]
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        #dir_2d_t = minmax_lambda([dir_2d_t,lower,upper])
        div_cmap = 'bwr'
        fig = show_2D_or_3D(dir_2d_t,allow_slicing=False, f_size=(25,2),cmap=div_cmap)
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('\alpha_t \nmid-cavity')
        #ax.margins(0,0)
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[len(ax.get_images())//2], cax=cax, orientation='horizontal')

        

        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # DIR 2D T x Z
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        directions_tz = directions.mean(axis=(2,3)) # original: norm()
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        _ = ax.imshow(directions_tz.T, aspect='auto', label='aasd', cmap=div_cmap, origin='lower')
        _ = ax.set_xticks(gtind, minor=False)
        ax2 = ax.twinx()
        _ = ax2.plot(dir_1d_mean, c='black',label='dir 1d+t')
        #_ = ax.set_xticks(gtind, minor=False)
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('dir z+t\napex:base')
        #ax.legend(loc='upper left')
        
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1
        
        #ax=fig.add_subplot(rows,1,pos)
        #_ = ax.plot(dir_1d_mean, c='black',label='dir 1d+t')
        #_ = ax.plot(dir_1_mean, label='top left')
        #_ = ax.plot(dir_2_mean, label='top right')
        #_ = ax.plot(dir_3_mean, label='bottom left')
        #_ = ax.plot(dir_4_mean, label='bottom right')
        #ax.legend(loc='upper left')
        #ax.set_xticks(gtind, minor=False)
        #ax.margins(0.022,0)
        #ax.label_outer()
        #pos=pos+1
        #plt.subplots_adjust(wspace=0, hspace=0.1)
        #plt.show()
        norm_cmap = 'hot'
        
        # NORM 2D + t
        norm_2d_t = norm_nda[:,z]
        norm_2d_t = minmax_lambda([norm_2d_t,mid,upper])
        fig = show_2D_or_3D(norm_2d_t,allow_slicing=False, f_size=(25,2),cmap=norm_cmap, interpolation='none')
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('norm 2d+t\nmid-cavity')
        #ax.margins(0,0)
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[0], cax=cax, orientation='horizontal')
        
        
        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # NORM 2D TxZ
        norm2d = minmax_lambda([norm_nda.mean(axis=(2,3)),mid,upper])
        _ = ax.imshow(norm2d.T, aspect='auto', origin='lower',cmap=norm_cmap, interpolation='none')
        _ = ax.set_xticks(gtind, minor=False)
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('norm z+t\napex:base')
        ax2 = ax.twinx()
        _ = ax2.plot(nda_1d_mean, c='black',label='norm 1d+t')
        #ax.spines['top'].set_visible(False)
        #_ = ax.legend(loc='upper left')
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1

        """ax=fig.add_subplot(rows,1,pos)
        _ = ax.plot(norm_mean, label='norm 1d+t')
        #_ = ax.plot(norm_1_mean, label='top left')
        #_ = ax.plot(norm_2_mean, label='top right')
        #_ = ax.plot(norm_3_mean, label='bottom left')
        #_ = ax.plot(norm_4_mean, label='bottom right')
        ax.legend(loc='upper left')
        ax.set_xticks(gtind, minor=False)
        ax.margins(0.022,0)
        ax.label_outer()
        pos = pos+1
        plt.subplots_adjust(wspace=0, hspace=0.1)"""
        #plt.tight_layout()
        plt.show()

        
    
    return ind
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
p = 2#215
gt_length = int(gt_len[p,:,0].sum())
ind = np.argmax(gt[p][:gt_length],axis=0)
get_phases_from_vects(nda_vects[p][:gt_length], length=gt_length,plot=True,dir_axis=0, gtind=ind)
# try to predict the phases unsupervised just from the norm/direction curve

def get_phases_from_vects(vects_nda, length=-1, plot=True, dir_axis=0, gtind=[], figsize=(25,3)):
    import scipy.signal
    import scipy.signal as sig
    import scipy.ndimage
    from scipy.ndimage import gaussian_filter1d
    
    gted=gtind[0]
    
    # signal smoothing
    kernel_size = 4
    kernel = np.ones(kernel_size) / kernel_size
    z = 0
    
    
    dim_z = vects_nda.shape[1]
    #vects_nda = vects_nda[:int(length)]
    
    
    # define some helper lambdas
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    dim = vects_nda.shape[1:-1]
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]

    
    # norm of the vector
    norm_nda = np.linalg.norm(vects_nda[...,dir_axis:], axis=-1)
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    norm_nda_thres = clip_quantile(norm_nda, 0.9)
    norm_nda = minmax_lambda([norm_nda,0,1])
    norm_msk = norm_nda>=0.0
    norm_nda = norm_nda * norm_msk
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    nda_1d_mean = np.mean(norm_nda,axis=(1,2,3))
    nda_1d_max = np.max(norm_nda,axis=(1,2,3))
    nda_1d_median = np.median(norm_nda,axis=(1,2,3))
    nda_1d_mean_smooth = np.convolve(nda_1d_mean, kernel, mode='same')
    #print(nda_1d_mean_smooth.shape, nda_1d_mean.shape)
    
    
    # direction relative to the center
    
    directions = flow2direction_lambda(vects_nda)[...,0].numpy()
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    directions = minmax_lambda([directions,lower,upper])
    #plt.hist(directions.flatten());plt.show()
    directions = directions * norm_msk
    #plt.hist(directions.flatten());plt.show()
    
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    dir_1d_mean = np.mean(directions,axis=(1,2,3))
    dir_1d_median = np.median(directions,axis=(1,2,3))

    
    # Find phases
    # 1st smooth the direction with a rolling averge (kernelsize=4)
    # or via gaussian filter
    # 2nd min/max normalise each direction vector in a range [-1:1]
    
    
    dir_1d_mean = gaussian_filter1d(dir_1d_mean, sigma=2, mode='wrap') # original: dir_1d_mean
    
    ################# CHANGED !!!!!! ######################## RECENTLY NO CHANGES
    # min/max scaling
    dir_1d_mean = minmax_lambda([dir_1d_mean,lower,upper])
    
    
#     # test with corner specific vector fields
#     y,x = directions.shape[-2:]
    
#     vects_mean = minmax_lambda([np.mean(vects_nda,axis=(1,2,3,4)),lower,upper])
#     vects_1 = vects_nda[:,:,:y//2,:x//2,:]
#     vects_1_mean = minmax_lambda([np.mean(vects_1,axis=(1,2,3,4)),lower,upper])
#     vects_2 = vects_nda[:,:,y//2:,:x//2,:]
#     vects_2_mean = minmax_lambda([np.mean(vects_2,axis=(1,2,3,4)),lower,upper])
#     vects_3 = vects_nda[:,:,y//2:,x//2:,:]
#     vects_3_mean = minmax_lambda([np.mean(vects_3,axis=(1,2,3,4)),lower,upper])
#     vects_4 = vects_nda[:,:,:y//2,x//2:,:]
#     vects_4_mean = minmax_lambda([np.mean(vects_4,axis=(1,2,3,4)),lower,upper])
    
    
#     # direction per corner
#     dir_1 = directions[:,:,:y//2,:x//2] # top left, plot
#     dir_1_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_2 = directions[:,:,y//2:,:x//2] # top right, plot
#     dir_2_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_3 = directions[:,:,y//2:,x//2:] # bottom left
#     dir_3_mean = minmax_lambda([np.mean(dir_3,axis=(1,2,3)),lower,upper])
#     dir_4 = directions[:,:,:y//2,x//2:] # bottom right
#     dir_4_mean = minmax_lambda([np.mean(dir_4,axis=(1,2,3)),lower,upper])
    

#     # norm per corner
#     norm_mean = minmax_lambda([nda_1d_mean, mid, upper])
#     norm_1 = norm_nda[:,:,:y//2,:x//2]
#     norm_1_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_2 = norm_nda[:,:,y//2:,:x//2]
#     norm_2_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_3 = norm_nda[:,:,y//2:,x//2:]
#     norm_3_mean = minmax_lambda([np.mean(norm_3,axis=(1,2,3)),mid,upper])
#     norm_4 = norm_nda[:,:,:y//2,x//2:]
#     norm_4_mean = minmax_lambda([np.mean(norm_4,axis=(1,2,3)),mid,upper])
    

    ind = detect_phases(dir_1d_mean=dir_1d_mean[:length], length=length)
    
    
    #print('ed: {}, ms: {}, es: {}, pf: {}, md: {}'.format(*ind))
    
        # plot the mean/max norm for one patient oveer time
    if plot:
        
        
        # VECT 2D+t
        vect_2d_t = vects_nda[:,z]
        fig = show_2D_or_3D(vect_2d_t,allow_slicing=False, f_size=(25,2))
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('v_t \nmid-cavity')

        
        # DIR 2D+t
        dir_2d_t = directions[:,z]
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        #dir_2d_t = minmax_lambda([dir_2d_t,lower,upper])
        div_cmap = 'bwr'
        fig = show_2D_or_3D(dir_2d_t,allow_slicing=False, f_size=(25,2),cmap=div_cmap)
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('alpha_t \nmid-cavity')
        #ax.margins(0,0)
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[len(ax.get_images())//2], cax=cax, orientation='horizontal')

        

        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # DIR 2D T x Z
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        directions_tz = directions.mean(axis=(2,3)) # original: norm()
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        _ = ax.imshow(directions_tz.T, aspect='auto', label='aasd', cmap=div_cmap, origin='lower')
        _ = ax.set_xticks(gtind, minor=False)
        ax2 = ax.twinx()
        _ = ax2.plot(dir_1d_mean, c='black',label='dir 1d+t')
        #_ = ax.set_xticks(gtind, minor=False)
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('dir z+t\napex:base')
        #ax.legend(loc='upper left')
        
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1
        
        #ax=fig.add_subplot(rows,1,pos)
        #_ = ax.plot(dir_1d_mean, c='black',label='dir 1d+t')
        #_ = ax.plot(dir_1_mean, label='top left')
        #_ = ax.plot(dir_2_mean, label='top right')
        #_ = ax.plot(dir_3_mean, label='bottom left')
        #_ = ax.plot(dir_4_mean, label='bottom right')
        #ax.legend(loc='upper left')
        #ax.set_xticks(gtind, minor=False)
        #ax.margins(0.022,0)
        #ax.label_outer()
        #pos=pos+1
        #plt.subplots_adjust(wspace=0, hspace=0.1)
        #plt.show()
        norm_cmap = 'hot'
        
        # NORM 2D + t
        norm_2d_t = norm_nda[:,z]
        norm_2d_t = minmax_lambda([norm_2d_t,mid,upper])
        fig = show_2D_or_3D(norm_2d_t,allow_slicing=False, f_size=(25,2),cmap=norm_cmap, interpolation='none')
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('norm 2d+t\nmid-cavity')
        #ax.margins(0,0)
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[0], cax=cax, orientation='horizontal')
        
        
        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # NORM 2D TxZ
        norm2d = minmax_lambda([norm_nda.mean(axis=(2,3)),mid,upper])
        _ = ax.imshow(norm2d.T, aspect='auto', origin='lower',cmap=norm_cmap, interpolation='none')
        _ = ax.set_xticks(gtind, minor=False)
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('norm z+t\napex:base')
        ax2 = ax.twinx()
        _ = ax2.plot(nda_1d_mean, c='black',label='norm 1d+t')
        #ax.spines['top'].set_visible(False)
        #_ = ax.legend(loc='upper left')
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1

        """ax=fig.add_subplot(rows,1,pos)
        _ = ax.plot(norm_mean, label='norm 1d+t')
        #_ = ax.plot(norm_1_mean, label='top left')
        #_ = ax.plot(norm_2_mean, label='top right')
        #_ = ax.plot(norm_3_mean, label='bottom left')
        #_ = ax.plot(norm_4_mean, label='bottom right')
        ax.legend(loc='upper left')
        ax.set_xticks(gtind, minor=False)
        ax.margins(0.022,0)
        ax.label_outer()
        pos = pos+1
        plt.subplots_adjust(wspace=0, hspace=0.1)"""
        #plt.tight_layout()
        plt.show()

        
    
    return ind
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
p = 2#215
gt_length = int(gt_len[p,:,0].sum())
ind = np.argmax(gt[p][:gt_length],axis=0)
get_phases_from_vects(nda_vects[p][:gt_length], length=gt_length,plot=True,dir_axis=0, gtind=ind)
# try to predict the phases unsupervised just from the norm/direction curve

def get_phases_from_vects(vects_nda, length=-1, plot=True, dir_axis=0, gtind=[], figsize=(25,3)):
    import scipy.signal
    import scipy.signal as sig
    import scipy.ndimage
    from scipy.ndimage import gaussian_filter1d
    
    gted=gtind[0]
    
    # signal smoothing
    kernel_size = 4
    kernel = np.ones(kernel_size) / kernel_size
    z = 0
    
    
    dim_z = vects_nda.shape[1]
    #vects_nda = vects_nda[:int(length)]
    
    
    # define some helper lambdas
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    dim = vects_nda.shape[1:-1]
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]

    
    # norm of the vector
    norm_nda = np.linalg.norm(vects_nda[...,dir_axis:], axis=-1)
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    norm_nda_thres = clip_quantile(norm_nda, 0.9)
    norm_nda = minmax_lambda([norm_nda,0,1])
    norm_msk = norm_nda>=0.0
    norm_nda = norm_nda * norm_msk
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    nda_1d_mean = np.mean(norm_nda,axis=(1,2,3))
    nda_1d_max = np.max(norm_nda,axis=(1,2,3))
    nda_1d_median = np.median(norm_nda,axis=(1,2,3))
    nda_1d_mean_smooth = np.convolve(nda_1d_mean, kernel, mode='same')
    #print(nda_1d_mean_smooth.shape, nda_1d_mean.shape)
    
    
    # direction relative to the center
    
    directions = flow2direction_lambda(vects_nda)[...,0].numpy()
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    directions = minmax_lambda([directions,lower,upper])
    #plt.hist(directions.flatten());plt.show()
    directions = directions * norm_msk
    #plt.hist(directions.flatten());plt.show()
    
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    dir_1d_mean = np.mean(directions,axis=(1,2,3))
    dir_1d_median = np.median(directions,axis=(1,2,3))

    
    # Find phases
    # 1st smooth the direction with a rolling averge (kernelsize=4)
    # or via gaussian filter
    # 2nd min/max normalise each direction vector in a range [-1:1]
    
    
    dir_1d_mean = gaussian_filter1d(dir_1d_mean, sigma=2, mode='wrap') # original: dir_1d_mean
    
    ################# CHANGED !!!!!! ######################## RECENTLY NO CHANGES
    # min/max scaling
    dir_1d_mean = minmax_lambda([dir_1d_mean,lower,upper])
    
    
#     # test with corner specific vector fields
#     y,x = directions.shape[-2:]
    
#     vects_mean = minmax_lambda([np.mean(vects_nda,axis=(1,2,3,4)),lower,upper])
#     vects_1 = vects_nda[:,:,:y//2,:x//2,:]
#     vects_1_mean = minmax_lambda([np.mean(vects_1,axis=(1,2,3,4)),lower,upper])
#     vects_2 = vects_nda[:,:,y//2:,:x//2,:]
#     vects_2_mean = minmax_lambda([np.mean(vects_2,axis=(1,2,3,4)),lower,upper])
#     vects_3 = vects_nda[:,:,y//2:,x//2:,:]
#     vects_3_mean = minmax_lambda([np.mean(vects_3,axis=(1,2,3,4)),lower,upper])
#     vects_4 = vects_nda[:,:,:y//2,x//2:,:]
#     vects_4_mean = minmax_lambda([np.mean(vects_4,axis=(1,2,3,4)),lower,upper])
    
    
#     # direction per corner
#     dir_1 = directions[:,:,:y//2,:x//2] # top left, plot
#     dir_1_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_2 = directions[:,:,y//2:,:x//2] # top right, plot
#     dir_2_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_3 = directions[:,:,y//2:,x//2:] # bottom left
#     dir_3_mean = minmax_lambda([np.mean(dir_3,axis=(1,2,3)),lower,upper])
#     dir_4 = directions[:,:,:y//2,x//2:] # bottom right
#     dir_4_mean = minmax_lambda([np.mean(dir_4,axis=(1,2,3)),lower,upper])
    

#     # norm per corner
#     norm_mean = minmax_lambda([nda_1d_mean, mid, upper])
#     norm_1 = norm_nda[:,:,:y//2,:x//2]
#     norm_1_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_2 = norm_nda[:,:,y//2:,:x//2]
#     norm_2_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_3 = norm_nda[:,:,y//2:,x//2:]
#     norm_3_mean = minmax_lambda([np.mean(norm_3,axis=(1,2,3)),mid,upper])
#     norm_4 = norm_nda[:,:,:y//2,x//2:]
#     norm_4_mean = minmax_lambda([np.mean(norm_4,axis=(1,2,3)),mid,upper])
    

    ind = detect_phases(dir_1d_mean=dir_1d_mean[:length], length=length)
    
    
    #print('ed: {}, ms: {}, es: {}, pf: {}, md: {}'.format(*ind))
    
        # plot the mean/max norm for one patient oveer time
    if plot:
        
        
        # VECT 2D+t
        vect_2d_t = vects_nda[:,z]
        fig = show_2D_or_3D(vect_2d_t,allow_slicing=False, f_size=(25,2))
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('v_t \nmid-cavity')

        
        # DIR 2D+t
        dir_2d_t = directions[:,z]
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        #dir_2d_t = minmax_lambda([dir_2d_t,lower,upper])
        div_cmap = 'bwr'
        fig = show_2D_or_3D(dir_2d_t,allow_slicing=False, f_size=(25,2),cmap=div_cmap)
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('alpha_t \nmid-cavity')
        #ax.margins(0,0)
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[len(ax.get_images())//2], cax=cax, orientation='horizontal')

        

        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # DIR 2D T x Z
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        directions_tz = directions.mean(axis=(2,3)) # original: norm()
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        _ = ax.imshow(directions_tz.T, aspect='auto', label='aasd', cmap=div_cmap, origin='lower')
        _ = ax.set_xticks(gtind, minor=False)
        ax2 = ax.twinx()
        _ = ax2.plot(dir_1d_mean, c='black',label='dir 1d+t')
        #_ = ax.set_xticks(gtind, minor=False)
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('dir z+t\napex:base')
        #ax.legend(loc='upper left')
        
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1
        
        #ax=fig.add_subplot(rows,1,pos)
        #_ = ax.plot(dir_1d_mean, c='black',label='dir 1d+t')
        #_ = ax.plot(dir_1_mean, label='top left')
        #_ = ax.plot(dir_2_mean, label='top right')
        #_ = ax.plot(dir_3_mean, label='bottom left')
        #_ = ax.plot(dir_4_mean, label='bottom right')
        #ax.legend(loc='upper left')
        #ax.set_xticks(gtind, minor=False)
        #ax.margins(0.022,0)
        #ax.label_outer()
        #pos=pos+1
        #plt.subplots_adjust(wspace=0, hspace=0.1)
        #plt.show()
        norm_cmap = 'hot'
        
        # NORM 2D + t
        norm_2d_t = norm_nda[:,z]
        norm_2d_t = minmax_lambda([norm_2d_t,mid,upper])
        fig = show_2D_or_3D(norm_2d_t,allow_slicing=False, f_size=(25,2),cmap=norm_cmap, interpolation='none')
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('norm 2d+t\nmid-cavity')
        #ax.margins(0,0)
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[0], cax=cax, orientation='horizontal')
        
        
        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # NORM 2D TxZ
        norm2d = minmax_lambda([norm_nda.mean(axis=(2,3)),mid,upper])
        _ = ax.imshow(norm2d.T, aspect='auto', origin='lower',cmap=norm_cmap, interpolation='none')
        _ = ax.set_xticks(gtind, minor=False)
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('norm z+t\napex:base')
        ax2 = ax.twinx()
        _ = ax2.plot(nda_1d_mean, c='black',label='norm 1d+t')
        #ax.spines['top'].set_visible(False)
        #_ = ax.legend(loc='upper left')
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1

        """ax=fig.add_subplot(rows,1,pos)
        _ = ax.plot(norm_mean, label='norm 1d+t')
        #_ = ax.plot(norm_1_mean, label='top left')
        #_ = ax.plot(norm_2_mean, label='top right')
        #_ = ax.plot(norm_3_mean, label='bottom left')
        #_ = ax.plot(norm_4_mean, label='bottom right')
        ax.legend(loc='upper left')
        ax.set_xticks(gtind, minor=False)
        ax.margins(0.022,0)
        ax.label_outer()
        pos = pos+1
        plt.subplots_adjust(wspace=0, hspace=0.1)"""
        #plt.tight_layout()
        plt.show()

        
    
    return ind
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
p = 2#215
gt_length = int(gt_len[p,:,0].sum())
ind = np.argmax(gt[p][:gt_length],axis=0)
get_phases_from_vects(nda_vects[p][:gt_length], length=gt_length,plot=True,dir_axis=0, gtind=ind)
# try to predict the phases unsupervised just from the norm/direction curve

def get_phases_from_vects(vects_nda, length=-1, plot=True, dir_axis=0, gtind=[], figsize=(25,3)):
    import scipy.signal
    import scipy.signal as sig
    import scipy.ndimage
    from scipy.ndimage import gaussian_filter1d
    
    gted=gtind[0]
    
    # signal smoothing
    kernel_size = 4
    kernel = np.ones(kernel_size) / kernel_size
    z = 0
    
    
    dim_z = vects_nda.shape[1]
    #vects_nda = vects_nda[:int(length)]
    
    
    # define some helper lambdas
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    dim = vects_nda.shape[1:-1]
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]

    
    # norm of the vector
    norm_nda = np.linalg.norm(vects_nda[...,dir_axis:], axis=-1)
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    norm_nda_thres = clip_quantile(norm_nda, 0.9)
    norm_nda = minmax_lambda([norm_nda,0,1])
    norm_msk = norm_nda>=0.0
    norm_nda = norm_nda * norm_msk
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    nda_1d_mean = np.mean(norm_nda,axis=(1,2,3))
    nda_1d_max = np.max(norm_nda,axis=(1,2,3))
    nda_1d_median = np.median(norm_nda,axis=(1,2,3))
    nda_1d_mean_smooth = np.convolve(nda_1d_mean, kernel, mode='same')
    #print(nda_1d_mean_smooth.shape, nda_1d_mean.shape)
    
    
    # direction relative to the center
    
    directions = flow2direction_lambda(vects_nda)[...,0].numpy()
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    directions = minmax_lambda([directions,lower,upper])
    #plt.hist(directions.flatten());plt.show()
    directions = directions * norm_msk
    #plt.hist(directions.flatten());plt.show()
    
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    dir_1d_mean = np.mean(directions,axis=(1,2,3))
    dir_1d_median = np.median(directions,axis=(1,2,3))

    
    # Find phases
    # 1st smooth the direction with a rolling averge (kernelsize=4)
    # or via gaussian filter
    # 2nd min/max normalise each direction vector in a range [-1:1]
    
    
    dir_1d_mean = gaussian_filter1d(dir_1d_mean, sigma=2, mode='wrap') # original: dir_1d_mean
    
    ################# CHANGED !!!!!! ######################## RECENTLY NO CHANGES
    # min/max scaling
    dir_1d_mean = minmax_lambda([dir_1d_mean,lower,upper])
    
    
#     # test with corner specific vector fields
#     y,x = directions.shape[-2:]
    
#     vects_mean = minmax_lambda([np.mean(vects_nda,axis=(1,2,3,4)),lower,upper])
#     vects_1 = vects_nda[:,:,:y//2,:x//2,:]
#     vects_1_mean = minmax_lambda([np.mean(vects_1,axis=(1,2,3,4)),lower,upper])
#     vects_2 = vects_nda[:,:,y//2:,:x//2,:]
#     vects_2_mean = minmax_lambda([np.mean(vects_2,axis=(1,2,3,4)),lower,upper])
#     vects_3 = vects_nda[:,:,y//2:,x//2:,:]
#     vects_3_mean = minmax_lambda([np.mean(vects_3,axis=(1,2,3,4)),lower,upper])
#     vects_4 = vects_nda[:,:,:y//2,x//2:,:]
#     vects_4_mean = minmax_lambda([np.mean(vects_4,axis=(1,2,3,4)),lower,upper])
    
    
#     # direction per corner
#     dir_1 = directions[:,:,:y//2,:x//2] # top left, plot
#     dir_1_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_2 = directions[:,:,y//2:,:x//2] # top right, plot
#     dir_2_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_3 = directions[:,:,y//2:,x//2:] # bottom left
#     dir_3_mean = minmax_lambda([np.mean(dir_3,axis=(1,2,3)),lower,upper])
#     dir_4 = directions[:,:,:y//2,x//2:] # bottom right
#     dir_4_mean = minmax_lambda([np.mean(dir_4,axis=(1,2,3)),lower,upper])
    

#     # norm per corner
#     norm_mean = minmax_lambda([nda_1d_mean, mid, upper])
#     norm_1 = norm_nda[:,:,:y//2,:x//2]
#     norm_1_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_2 = norm_nda[:,:,y//2:,:x//2]
#     norm_2_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_3 = norm_nda[:,:,y//2:,x//2:]
#     norm_3_mean = minmax_lambda([np.mean(norm_3,axis=(1,2,3)),mid,upper])
#     norm_4 = norm_nda[:,:,:y//2,x//2:]
#     norm_4_mean = minmax_lambda([np.mean(norm_4,axis=(1,2,3)),mid,upper])
    

    ind = detect_phases(dir_1d_mean=dir_1d_mean[:length], length=length)
    
    
    #print('ed: {}, ms: {}, es: {}, pf: {}, md: {}'.format(*ind))
    
        # plot the mean/max norm for one patient oveer time
    if plot:
        
        
        # VECT 2D+t
        vect_2d_t = vects_nda[:,z]
        fig = show_2D_or_3D(vect_2d_t,allow_slicing=False, f_size=(25,2))
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('v_t \nmid-cavity')

        
        # DIR 2D+t
        dir_2d_t = directions[:,z]
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        #dir_2d_t = minmax_lambda([dir_2d_t,lower,upper])
        div_cmap = 'bwr'
        fig = show_2D_or_3D(dir_2d_t,allow_slicing=False, f_size=(25,2),cmap=div_cmap)
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('alpha_t \nmid-cavity')
        #ax.margins(0,0)
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[len(ax.get_images())//2], cax=cax, orientation='horizontal')

        

        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # DIR 2D T x Z
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        directions_tz = directions.mean(axis=(2,3)) # original: norm()
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        _ = ax.imshow(directions_tz.T, aspect='auto', label='aasd', cmap=div_cmap, origin='lower')
        _ = ax.set_xticks(gtind, minor=False)
        ax2 = ax.twinx()
        _ = ax2.plot(dir_1d_mean, c='black',label='dir 1d+t')
        #_ = ax.set_xticks(gtind, minor=False)
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('dir z+t\napex:base')
        #ax.legend(loc='upper left')
        
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1
        
        #ax=fig.add_subplot(rows,1,pos)
        #_ = ax.plot(dir_1d_mean, c='black',label='dir 1d+t')
        #_ = ax.plot(dir_1_mean, label='top left')
        #_ = ax.plot(dir_2_mean, label='top right')
        #_ = ax.plot(dir_3_mean, label='bottom left')
        #_ = ax.plot(dir_4_mean, label='bottom right')
        #ax.legend(loc='upper left')
        #ax.set_xticks(gtind, minor=False)
        #ax.margins(0.022,0)
        #ax.label_outer()
        #pos=pos+1
        #plt.subplots_adjust(wspace=0, hspace=0.1)
        #plt.show()
        norm_cmap = 'hot'
        
        # NORM 2D + t
        norm_2d_t = norm_nda[:,z]
        norm_2d_t = minmax_lambda([norm_2d_t,mid,upper])
        fig = show_2D_or_3D(norm_2d_t,allow_slicing=False, f_size=(25,2),cmap=norm_cmap, interpolation='none')
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('norm 2d+t\nmid-cavity')
        #ax.margins(0,0)
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[0], cax=cax, orientation='horizontal')
        
        
        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # NORM 2D TxZ
        norm2d = minmax_lambda([norm_nda.mean(axis=(2,3)),mid,upper])
        _ = ax.imshow(norm2d.T, aspect='auto', origin='lower',cmap=norm_cmap, interpolation='none')
        _ = ax.set_xticks(gtind, minor=False)
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('norm z+t\napex:base')
        ax2 = ax.twinx()
        _ = ax2.plot(nda_1d_mean, c='black',label='norm 1d+t')
        #ax.spines['top'].set_visible(False)
        #_ = ax.legend(loc='upper left')
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1

        """ax=fig.add_subplot(rows,1,pos)
        _ = ax.plot(norm_mean, label='norm 1d+t')
        #_ = ax.plot(norm_1_mean, label='top left')
        #_ = ax.plot(norm_2_mean, label='top right')
        #_ = ax.plot(norm_3_mean, label='bottom left')
        #_ = ax.plot(norm_4_mean, label='bottom right')
        ax.legend(loc='upper left')
        ax.set_xticks(gtind, minor=False)
        ax.margins(0.022,0)
        ax.label_outer()
        pos = pos+1
        plt.subplots_adjust(wspace=0, hspace=0.1)"""
        #plt.tight_layout()
        plt.show()

        
    
    return ind
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
p = 2#215
gt_length = int(gt_len[p,:,0].sum())
ind = np.argmax(gt[p][:gt_length],axis=0)
get_phases_from_vects(nda_vects[p][:gt_length], length=gt_length,plot=True,dir_axis=0, gtind=ind)
# try to predict the phases unsupervised just from the norm/direction curve

def get_phases_from_vects(vects_nda, length=-1, plot=True, dir_axis=0, gtind=[], figsize=(25,3)):
    import scipy.signal
    import scipy.signal as sig
    import scipy.ndimage
    from scipy.ndimage import gaussian_filter1d
    
    gted=gtind[0]
    
    # signal smoothing
    kernel_size = 4
    kernel = np.ones(kernel_size) / kernel_size
    z = 0
    
    
    dim_z = vects_nda.shape[1]
    #vects_nda = vects_nda[:int(length)]
    
    
    # define some helper lambdas
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    dim = vects_nda.shape[1:-1]
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]

    
    # norm of the vector
    norm_nda = np.linalg.norm(vects_nda[...,dir_axis:], axis=-1)
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    norm_nda_thres = clip_quantile(norm_nda, 0.9)
    norm_nda = minmax_lambda([norm_nda,0,1])
    norm_msk = norm_nda>=0.0
    norm_nda = norm_nda * norm_msk
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    nda_1d_mean = np.mean(norm_nda,axis=(1,2,3))
    nda_1d_max = np.max(norm_nda,axis=(1,2,3))
    nda_1d_median = np.median(norm_nda,axis=(1,2,3))
    nda_1d_mean_smooth = np.convolve(nda_1d_mean, kernel, mode='same')
    #print(nda_1d_mean_smooth.shape, nda_1d_mean.shape)
    
    
    # direction relative to the center
    
    directions = flow2direction_lambda(vects_nda)[...,0].numpy()
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    directions = minmax_lambda([directions,lower,upper])
    #plt.hist(directions.flatten());plt.show()
    directions = directions * norm_msk
    #plt.hist(directions.flatten());plt.show()
    
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    dir_1d_mean = np.mean(directions,axis=(1,2,3))
    dir_1d_median = np.median(directions,axis=(1,2,3))

    
    # Find phases
    # 1st smooth the direction with a rolling averge (kernelsize=4)
    # or via gaussian filter
    # 2nd min/max normalise each direction vector in a range [-1:1]
    
    
    dir_1d_mean = gaussian_filter1d(dir_1d_mean, sigma=2, mode='wrap') # original: dir_1d_mean
    
    ################# CHANGED !!!!!! ######################## RECENTLY NO CHANGES
    # min/max scaling
    dir_1d_mean = minmax_lambda([dir_1d_mean,lower,upper])
    
    
#     # test with corner specific vector fields
#     y,x = directions.shape[-2:]
    
#     vects_mean = minmax_lambda([np.mean(vects_nda,axis=(1,2,3,4)),lower,upper])
#     vects_1 = vects_nda[:,:,:y//2,:x//2,:]
#     vects_1_mean = minmax_lambda([np.mean(vects_1,axis=(1,2,3,4)),lower,upper])
#     vects_2 = vects_nda[:,:,y//2:,:x//2,:]
#     vects_2_mean = minmax_lambda([np.mean(vects_2,axis=(1,2,3,4)),lower,upper])
#     vects_3 = vects_nda[:,:,y//2:,x//2:,:]
#     vects_3_mean = minmax_lambda([np.mean(vects_3,axis=(1,2,3,4)),lower,upper])
#     vects_4 = vects_nda[:,:,:y//2,x//2:,:]
#     vects_4_mean = minmax_lambda([np.mean(vects_4,axis=(1,2,3,4)),lower,upper])
    
    
#     # direction per corner
#     dir_1 = directions[:,:,:y//2,:x//2] # top left, plot
#     dir_1_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_2 = directions[:,:,y//2:,:x//2] # top right, plot
#     dir_2_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_3 = directions[:,:,y//2:,x//2:] # bottom left
#     dir_3_mean = minmax_lambda([np.mean(dir_3,axis=(1,2,3)),lower,upper])
#     dir_4 = directions[:,:,:y//2,x//2:] # bottom right
#     dir_4_mean = minmax_lambda([np.mean(dir_4,axis=(1,2,3)),lower,upper])
    

#     # norm per corner
#     norm_mean = minmax_lambda([nda_1d_mean, mid, upper])
#     norm_1 = norm_nda[:,:,:y//2,:x//2]
#     norm_1_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_2 = norm_nda[:,:,y//2:,:x//2]
#     norm_2_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_3 = norm_nda[:,:,y//2:,x//2:]
#     norm_3_mean = minmax_lambda([np.mean(norm_3,axis=(1,2,3)),mid,upper])
#     norm_4 = norm_nda[:,:,:y//2,x//2:]
#     norm_4_mean = minmax_lambda([np.mean(norm_4,axis=(1,2,3)),mid,upper])
    

    ind = detect_phases(dir_1d_mean=dir_1d_mean[:length], length=length)
    
    
    #print('ed: {}, ms: {}, es: {}, pf: {}, md: {}'.format(*ind))
    
        # plot the mean/max norm for one patient oveer time
    if plot:
        
        
        # VECT 2D+t
        vect_2d_t = vects_nda[:,z]
        fig = show_2D_or_3D(vect_2d_t,allow_slicing=False, f_size=(25,2))
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('v_t \nmid-cavity')

        
        # DIR 2D+t
        dir_2d_t = directions[:,z]
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        #dir_2d_t = minmax_lambda([dir_2d_t,lower,upper])
        div_cmap = 'bwr'
        fig = show_2D_or_3D(dir_2d_t,allow_slicing=False, f_size=(25,2),cmap=div_cmap)
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('alpha_t \nmid-cavity')
        #ax.margins(0,0)
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[len(ax.get_images())//2], cax=cax, orientation='horizontal')

        

        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # DIR 2D T x Z
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        directions_tz = directions.mean(axis=(2,3)) # original: norm()
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        _ = ax.imshow(directions_tz.T, aspect='auto', label='aasd', cmap=div_cmap, origin='lower')
        _ = ax.set_xticks(gtind, minor=False)
        ax2 = ax.twinx()
        _ = ax2.plot(dir_1d_mean, c='black',label='dir 1d+t')
        #_ = ax.set_xticks(gtind, minor=False)
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('dir z+t\napex:base')
        #ax.legend(loc='upper left')
        
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1
        
        #ax=fig.add_subplot(rows,1,pos)
        #_ = ax.plot(dir_1d_mean, c='black',label='dir 1d+t')
        #_ = ax.plot(dir_1_mean, label='top left')
        #_ = ax.plot(dir_2_mean, label='top right')
        #_ = ax.plot(dir_3_mean, label='bottom left')
        #_ = ax.plot(dir_4_mean, label='bottom right')
        #ax.legend(loc='upper left')
        #ax.set_xticks(gtind, minor=False)
        #ax.margins(0.022,0)
        #ax.label_outer()
        #pos=pos+1
        #plt.subplots_adjust(wspace=0, hspace=0.1)
        #plt.show()
        norm_cmap = 'hot'
        
        # NORM 2D + t
        norm_2d_t = norm_nda[:,z]
        norm_2d_t = minmax_lambda([norm_2d_t,mid,upper])
        fig = show_2D_or_3D(norm_2d_t,allow_slicing=False, f_size=(25,2),cmap=norm_cmap, interpolation='none')
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('norm 2d+t\nmid-cavity')
        #ax.margins(0,0)
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[0], cax=cax, orientation='horizontal')
        
        
        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # NORM 2D TxZ
        norm2d = minmax_lambda([norm_nda.mean(axis=(2,3)),mid,upper])
        _ = ax.imshow(norm2d.T, aspect='auto', origin='lower',cmap=norm_cmap, interpolation='none')
        _ = ax.set_xticks(gtind, minor=False)
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('norm z+t\napex:base')
        ax2 = ax.twinx()
        _ = ax2.plot(nda_1d_mean, c='black',label='norm 1d+t')
        #ax.spines['top'].set_visible(False)
        #_ = ax.legend(loc='upper left')
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1

        """ax=fig.add_subplot(rows,1,pos)
        _ = ax.plot(norm_mean, label='norm 1d+t')
        #_ = ax.plot(norm_1_mean, label='top left')
        #_ = ax.plot(norm_2_mean, label='top right')
        #_ = ax.plot(norm_3_mean, label='bottom left')
        #_ = ax.plot(norm_4_mean, label='bottom right')
        ax.legend(loc='upper left')
        ax.set_xticks(gtind, minor=False)
        ax.margins(0.022,0)
        ax.label_outer()
        pos = pos+1
        plt.subplots_adjust(wspace=0, hspace=0.1)"""
        #plt.tight_layout()
        plt.show()

        
    
    return ind
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
p = 2#215
gt_length = int(gt_len[p,:,0].sum())
ind = np.argmax(gt[p][:gt_length],axis=0)
get_phases_from_vects(nda_vects[p][:gt_length], length=gt_length,plot=True,dir_axis=0, gtind=ind)
# try to predict the phases unsupervised just from the norm/direction curve

def get_phases_from_vects(vects_nda, length=-1, plot=True, dir_axis=0, gtind=[], figsize=(25,3)):
    import scipy.signal
    import scipy.signal as sig
    import scipy.ndimage
    from scipy.ndimage import gaussian_filter1d
    
    gted=gtind[0]
    
    # signal smoothing
    kernel_size = 4
    kernel = np.ones(kernel_size) / kernel_size
    z = 0
    
    
    dim_z = vects_nda.shape[1]
    #vects_nda = vects_nda[:int(length)]
    
    
    # define some helper lambdas
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    dim = vects_nda.shape[1:-1]
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]

    
    # norm of the vector
    norm_nda = np.linalg.norm(vects_nda[...,dir_axis:], axis=-1)
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    norm_nda_thres = clip_quantile(norm_nda, 0.9)
    norm_nda = minmax_lambda([norm_nda,0,1])
    norm_msk = norm_nda>=0.0
    norm_nda = norm_nda * norm_msk
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    nda_1d_mean = np.mean(norm_nda,axis=(1,2,3))
    nda_1d_max = np.max(norm_nda,axis=(1,2,3))
    nda_1d_median = np.median(norm_nda,axis=(1,2,3))
    nda_1d_mean_smooth = np.convolve(nda_1d_mean, kernel, mode='same')
    #print(nda_1d_mean_smooth.shape, nda_1d_mean.shape)
    
    
    # direction relative to the center
    
    directions = flow2direction_lambda(vects_nda)[...,0].numpy()
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    directions = minmax_lambda([directions,lower,upper])
    #plt.hist(directions.flatten());plt.show()
    directions = directions * norm_msk
    #plt.hist(directions.flatten());plt.show()
    
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    dir_1d_mean = np.mean(directions,axis=(1,2,3))
    dir_1d_median = np.median(directions,axis=(1,2,3))

    
    # Find phases
    # 1st smooth the direction with a rolling averge (kernelsize=4)
    # or via gaussian filter
    # 2nd min/max normalise each direction vector in a range [-1:1]
    
    
    dir_1d_mean = gaussian_filter1d(dir_1d_mean, sigma=2, mode='wrap') # original: dir_1d_mean
    
    ################# CHANGED !!!!!! ######################## RECENTLY NO CHANGES
    # min/max scaling
    dir_1d_mean = minmax_lambda([dir_1d_mean,lower,upper])
    
    
#     # test with corner specific vector fields
#     y,x = directions.shape[-2:]
    
#     vects_mean = minmax_lambda([np.mean(vects_nda,axis=(1,2,3,4)),lower,upper])
#     vects_1 = vects_nda[:,:,:y//2,:x//2,:]
#     vects_1_mean = minmax_lambda([np.mean(vects_1,axis=(1,2,3,4)),lower,upper])
#     vects_2 = vects_nda[:,:,y//2:,:x//2,:]
#     vects_2_mean = minmax_lambda([np.mean(vects_2,axis=(1,2,3,4)),lower,upper])
#     vects_3 = vects_nda[:,:,y//2:,x//2:,:]
#     vects_3_mean = minmax_lambda([np.mean(vects_3,axis=(1,2,3,4)),lower,upper])
#     vects_4 = vects_nda[:,:,:y//2,x//2:,:]
#     vects_4_mean = minmax_lambda([np.mean(vects_4,axis=(1,2,3,4)),lower,upper])
    
    
#     # direction per corner
#     dir_1 = directions[:,:,:y//2,:x//2] # top left, plot
#     dir_1_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_2 = directions[:,:,y//2:,:x//2] # top right, plot
#     dir_2_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_3 = directions[:,:,y//2:,x//2:] # bottom left
#     dir_3_mean = minmax_lambda([np.mean(dir_3,axis=(1,2,3)),lower,upper])
#     dir_4 = directions[:,:,:y//2,x//2:] # bottom right
#     dir_4_mean = minmax_lambda([np.mean(dir_4,axis=(1,2,3)),lower,upper])
    

#     # norm per corner
#     norm_mean = minmax_lambda([nda_1d_mean, mid, upper])
#     norm_1 = norm_nda[:,:,:y//2,:x//2]
#     norm_1_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_2 = norm_nda[:,:,y//2:,:x//2]
#     norm_2_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_3 = norm_nda[:,:,y//2:,x//2:]
#     norm_3_mean = minmax_lambda([np.mean(norm_3,axis=(1,2,3)),mid,upper])
#     norm_4 = norm_nda[:,:,:y//2,x//2:]
#     norm_4_mean = minmax_lambda([np.mean(norm_4,axis=(1,2,3)),mid,upper])
    

    ind = detect_phases(dir_1d_mean=dir_1d_mean[:length], length=length)
    
    
    #print('ed: {}, ms: {}, es: {}, pf: {}, md: {}'.format(*ind))
    
        # plot the mean/max norm for one patient oveer time
    if plot:
        
        
        # VECT 2D+t
        vect_2d_t = vects_nda[:,z]
        fig = show_2D_or_3D(vect_2d_t,allow_slicing=False, f_size=(25,2))
        print(type(fig))
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('v_t \nmid-cavity')

        
        # DIR 2D+t
        dir_2d_t = directions[:,z]
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        #dir_2d_t = minmax_lambda([dir_2d_t,lower,upper])
        div_cmap = 'bwr'
        fig = show_2D_or_3D(dir_2d_t,allow_slicing=False, f_size=(25,2),cmap=div_cmap)
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('alpha_t \nmid-cavity')
        #ax.margins(0,0)
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[len(ax.get_images())//2], cax=cax, orientation='horizontal')

        

        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # DIR 2D T x Z
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        directions_tz = directions.mean(axis=(2,3)) # original: norm()
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        _ = ax.imshow(directions_tz.T, aspect='auto', label='aasd', cmap=div_cmap, origin='lower')
        _ = ax.set_xticks(gtind, minor=False)
        ax2 = ax.twinx()
        _ = ax2.plot(dir_1d_mean, c='black',label='dir 1d+t')
        #_ = ax.set_xticks(gtind, minor=False)
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('dir z+t\napex:base')
        #ax.legend(loc='upper left')
        
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1
        
        #ax=fig.add_subplot(rows,1,pos)
        #_ = ax.plot(dir_1d_mean, c='black',label='dir 1d+t')
        #_ = ax.plot(dir_1_mean, label='top left')
        #_ = ax.plot(dir_2_mean, label='top right')
        #_ = ax.plot(dir_3_mean, label='bottom left')
        #_ = ax.plot(dir_4_mean, label='bottom right')
        #ax.legend(loc='upper left')
        #ax.set_xticks(gtind, minor=False)
        #ax.margins(0.022,0)
        #ax.label_outer()
        #pos=pos+1
        #plt.subplots_adjust(wspace=0, hspace=0.1)
        #plt.show()
        norm_cmap = 'hot'
        
        # NORM 2D + t
        norm_2d_t = norm_nda[:,z]
        norm_2d_t = minmax_lambda([norm_2d_t,mid,upper])
        fig = show_2D_or_3D(norm_2d_t,allow_slicing=False, f_size=(25,2),cmap=norm_cmap, interpolation='none')
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('norm 2d+t\nmid-cavity')
        #ax.margins(0,0)
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[0], cax=cax, orientation='horizontal')
        
        
        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # NORM 2D TxZ
        norm2d = minmax_lambda([norm_nda.mean(axis=(2,3)),mid,upper])
        _ = ax.imshow(norm2d.T, aspect='auto', origin='lower',cmap=norm_cmap, interpolation='none')
        _ = ax.set_xticks(gtind, minor=False)
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('norm z+t\napex:base')
        ax2 = ax.twinx()
        _ = ax2.plot(nda_1d_mean, c='black',label='norm 1d+t')
        #ax.spines['top'].set_visible(False)
        #_ = ax.legend(loc='upper left')
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1

        """ax=fig.add_subplot(rows,1,pos)
        _ = ax.plot(norm_mean, label='norm 1d+t')
        #_ = ax.plot(norm_1_mean, label='top left')
        #_ = ax.plot(norm_2_mean, label='top right')
        #_ = ax.plot(norm_3_mean, label='bottom left')
        #_ = ax.plot(norm_4_mean, label='bottom right')
        ax.legend(loc='upper left')
        ax.set_xticks(gtind, minor=False)
        ax.margins(0.022,0)
        ax.label_outer()
        pos = pos+1
        plt.subplots_adjust(wspace=0, hspace=0.1)"""
        #plt.tight_layout()
        plt.show()

        
    
    return ind
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
p = 2#215
gt_length = int(gt_len[p,:,0].sum())
ind = np.argmax(gt[p][:gt_length],axis=0)
get_phases_from_vects(nda_vects[p][:gt_length], length=gt_length,plot=True,dir_axis=0, gtind=ind)
# try to predict the phases unsupervised just from the norm/direction curve

def get_phases_from_vects(vects_nda, length=-1, plot=True, dir_axis=0, gtind=[], figsize=(25,3)):
    import scipy.signal
    import scipy.signal as sig
    import scipy.ndimage
    from scipy.ndimage import gaussian_filter1d
    
    gted=gtind[0]
    
    # signal smoothing
    kernel_size = 4
    kernel = np.ones(kernel_size) / kernel_size
    z = 0
    
    
    dim_z = vects_nda.shape[1]
    #vects_nda = vects_nda[:int(length)]
    
    
    # define some helper lambdas
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    dim = vects_nda.shape[1:-1]
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]

    
    # norm of the vector
    norm_nda = np.linalg.norm(vects_nda[...,dir_axis:], axis=-1)
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    norm_nda_thres = clip_quantile(norm_nda, 0.9)
    norm_nda = minmax_lambda([norm_nda,0,1])
    norm_msk = norm_nda>=0.0
    norm_nda = norm_nda * norm_msk
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    nda_1d_mean = np.mean(norm_nda,axis=(1,2,3))
    nda_1d_max = np.max(norm_nda,axis=(1,2,3))
    nda_1d_median = np.median(norm_nda,axis=(1,2,3))
    nda_1d_mean_smooth = np.convolve(nda_1d_mean, kernel, mode='same')
    #print(nda_1d_mean_smooth.shape, nda_1d_mean.shape)
    
    
    # direction relative to the center
    
    directions = flow2direction_lambda(vects_nda)[...,0].numpy()
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    directions = minmax_lambda([directions,lower,upper])
    #plt.hist(directions.flatten());plt.show()
    directions = directions * norm_msk
    #plt.hist(directions.flatten());plt.show()
    
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    dir_1d_mean = np.mean(directions,axis=(1,2,3))
    dir_1d_median = np.median(directions,axis=(1,2,3))

    
    # Find phases
    # 1st smooth the direction with a rolling averge (kernelsize=4)
    # or via gaussian filter
    # 2nd min/max normalise each direction vector in a range [-1:1]
    
    
    dir_1d_mean = gaussian_filter1d(dir_1d_mean, sigma=2, mode='wrap') # original: dir_1d_mean
    
    ################# CHANGED !!!!!! ######################## RECENTLY NO CHANGES
    # min/max scaling
    dir_1d_mean = minmax_lambda([dir_1d_mean,lower,upper])
    
    
#     # test with corner specific vector fields
#     y,x = directions.shape[-2:]
    
#     vects_mean = minmax_lambda([np.mean(vects_nda,axis=(1,2,3,4)),lower,upper])
#     vects_1 = vects_nda[:,:,:y//2,:x//2,:]
#     vects_1_mean = minmax_lambda([np.mean(vects_1,axis=(1,2,3,4)),lower,upper])
#     vects_2 = vects_nda[:,:,y//2:,:x//2,:]
#     vects_2_mean = minmax_lambda([np.mean(vects_2,axis=(1,2,3,4)),lower,upper])
#     vects_3 = vects_nda[:,:,y//2:,x//2:,:]
#     vects_3_mean = minmax_lambda([np.mean(vects_3,axis=(1,2,3,4)),lower,upper])
#     vects_4 = vects_nda[:,:,:y//2,x//2:,:]
#     vects_4_mean = minmax_lambda([np.mean(vects_4,axis=(1,2,3,4)),lower,upper])
    
    
#     # direction per corner
#     dir_1 = directions[:,:,:y//2,:x//2] # top left, plot
#     dir_1_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_2 = directions[:,:,y//2:,:x//2] # top right, plot
#     dir_2_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_3 = directions[:,:,y//2:,x//2:] # bottom left
#     dir_3_mean = minmax_lambda([np.mean(dir_3,axis=(1,2,3)),lower,upper])
#     dir_4 = directions[:,:,:y//2,x//2:] # bottom right
#     dir_4_mean = minmax_lambda([np.mean(dir_4,axis=(1,2,3)),lower,upper])
    

#     # norm per corner
#     norm_mean = minmax_lambda([nda_1d_mean, mid, upper])
#     norm_1 = norm_nda[:,:,:y//2,:x//2]
#     norm_1_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_2 = norm_nda[:,:,y//2:,:x//2]
#     norm_2_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_3 = norm_nda[:,:,y//2:,x//2:]
#     norm_3_mean = minmax_lambda([np.mean(norm_3,axis=(1,2,3)),mid,upper])
#     norm_4 = norm_nda[:,:,:y//2,x//2:]
#     norm_4_mean = minmax_lambda([np.mean(norm_4,axis=(1,2,3)),mid,upper])
    

    ind = detect_phases(dir_1d_mean=dir_1d_mean[:length], length=length)
    
    
    #print('ed: {}, ms: {}, es: {}, pf: {}, md: {}'.format(*ind))
    
        # plot the mean/max norm for one patient oveer time
    if plot:
        
        
        # VECT 2D+t
        vect_2d_t = vects_nda[:,z]
        fig = show_2D_or_3D(vect_2d_t,allow_slicing=False, f_size=(25,2))
        print(type(fig))
        ax = fig.get_axes()[0]
        print(type(ax))
        _ = ax.set_ylabel('v_t \nmid-cavity')

        
        # DIR 2D+t
        dir_2d_t = directions[:,z]
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        #dir_2d_t = minmax_lambda([dir_2d_t,lower,upper])
        div_cmap = 'bwr'
        fig = show_2D_or_3D(dir_2d_t,allow_slicing=False, f_size=(25,2),cmap=div_cmap)
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('alpha_t \nmid-cavity')
        #ax.margins(0,0)
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[len(ax.get_images())//2], cax=cax, orientation='horizontal')

        

        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # DIR 2D T x Z
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        directions_tz = directions.mean(axis=(2,3)) # original: norm()
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        _ = ax.imshow(directions_tz.T, aspect='auto', label='aasd', cmap=div_cmap, origin='lower')
        _ = ax.set_xticks(gtind, minor=False)
        ax2 = ax.twinx()
        _ = ax2.plot(dir_1d_mean, c='black',label='dir 1d+t')
        #_ = ax.set_xticks(gtind, minor=False)
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('dir z+t\napex:base')
        #ax.legend(loc='upper left')
        
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1
        
        #ax=fig.add_subplot(rows,1,pos)
        #_ = ax.plot(dir_1d_mean, c='black',label='dir 1d+t')
        #_ = ax.plot(dir_1_mean, label='top left')
        #_ = ax.plot(dir_2_mean, label='top right')
        #_ = ax.plot(dir_3_mean, label='bottom left')
        #_ = ax.plot(dir_4_mean, label='bottom right')
        #ax.legend(loc='upper left')
        #ax.set_xticks(gtind, minor=False)
        #ax.margins(0.022,0)
        #ax.label_outer()
        #pos=pos+1
        #plt.subplots_adjust(wspace=0, hspace=0.1)
        #plt.show()
        norm_cmap = 'hot'
        
        # NORM 2D + t
        norm_2d_t = norm_nda[:,z]
        norm_2d_t = minmax_lambda([norm_2d_t,mid,upper])
        fig = show_2D_or_3D(norm_2d_t,allow_slicing=False, f_size=(25,2),cmap=norm_cmap, interpolation='none')
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('norm 2d+t\nmid-cavity')
        #ax.margins(0,0)
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[0], cax=cax, orientation='horizontal')
        
        
        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # NORM 2D TxZ
        norm2d = minmax_lambda([norm_nda.mean(axis=(2,3)),mid,upper])
        _ = ax.imshow(norm2d.T, aspect='auto', origin='lower',cmap=norm_cmap, interpolation='none')
        _ = ax.set_xticks(gtind, minor=False)
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('norm z+t\napex:base')
        ax2 = ax.twinx()
        _ = ax2.plot(nda_1d_mean, c='black',label='norm 1d+t')
        #ax.spines['top'].set_visible(False)
        #_ = ax.legend(loc='upper left')
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1

        """ax=fig.add_subplot(rows,1,pos)
        _ = ax.plot(norm_mean, label='norm 1d+t')
        #_ = ax.plot(norm_1_mean, label='top left')
        #_ = ax.plot(norm_2_mean, label='top right')
        #_ = ax.plot(norm_3_mean, label='bottom left')
        #_ = ax.plot(norm_4_mean, label='bottom right')
        ax.legend(loc='upper left')
        ax.set_xticks(gtind, minor=False)
        ax.margins(0.022,0)
        ax.label_outer()
        pos = pos+1
        plt.subplots_adjust(wspace=0, hspace=0.1)"""
        #plt.tight_layout()
        plt.show()

        
    
    return ind
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
p = 2#215
gt_length = int(gt_len[p,:,0].sum())
ind = np.argmax(gt[p][:gt_length],axis=0)
get_phases_from_vects(nda_vects[p][:gt_length], length=gt_length,plot=True,dir_axis=0, gtind=ind)
# try to predict the phases unsupervised just from the norm/direction curve

def get_phases_from_vects(vects_nda, length=-1, plot=True, dir_axis=0, gtind=[], figsize=(25,3)):
    import scipy.signal
    import scipy.signal as sig
    import scipy.ndimage
    from scipy.ndimage import gaussian_filter1d
    
    gted=gtind[0]
    
    # signal smoothing
    kernel_size = 4
    kernel = np.ones(kernel_size) / kernel_size
    z = 0
    
    
    dim_z = vects_nda.shape[1]
    #vects_nda = vects_nda[:int(length)]
    
    
    # define some helper lambdas
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    dim = vects_nda.shape[1:-1]
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]

    
    # norm of the vector
    norm_nda = np.linalg.norm(vects_nda[...,dir_axis:], axis=-1)
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    norm_nda_thres = clip_quantile(norm_nda, 0.9)
    norm_nda = minmax_lambda([norm_nda,0,1])
    norm_msk = norm_nda>=0.0
    norm_nda = norm_nda * norm_msk
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    nda_1d_mean = np.mean(norm_nda,axis=(1,2,3))
    nda_1d_max = np.max(norm_nda,axis=(1,2,3))
    nda_1d_median = np.median(norm_nda,axis=(1,2,3))
    nda_1d_mean_smooth = np.convolve(nda_1d_mean, kernel, mode='same')
    #print(nda_1d_mean_smooth.shape, nda_1d_mean.shape)
    
    
    # direction relative to the center
    
    directions = flow2direction_lambda(vects_nda)[...,0].numpy()
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    directions = minmax_lambda([directions,lower,upper])
    #plt.hist(directions.flatten());plt.show()
    directions = directions * norm_msk
    #plt.hist(directions.flatten());plt.show()
    
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    dir_1d_mean = np.mean(directions,axis=(1,2,3))
    dir_1d_median = np.median(directions,axis=(1,2,3))

    
    # Find phases
    # 1st smooth the direction with a rolling averge (kernelsize=4)
    # or via gaussian filter
    # 2nd min/max normalise each direction vector in a range [-1:1]
    
    
    dir_1d_mean = gaussian_filter1d(dir_1d_mean, sigma=2, mode='wrap') # original: dir_1d_mean
    
    ################# CHANGED !!!!!! ######################## RECENTLY NO CHANGES
    # min/max scaling
    dir_1d_mean = minmax_lambda([dir_1d_mean,lower,upper])
    
    
#     # test with corner specific vector fields
#     y,x = directions.shape[-2:]
    
#     vects_mean = minmax_lambda([np.mean(vects_nda,axis=(1,2,3,4)),lower,upper])
#     vects_1 = vects_nda[:,:,:y//2,:x//2,:]
#     vects_1_mean = minmax_lambda([np.mean(vects_1,axis=(1,2,3,4)),lower,upper])
#     vects_2 = vects_nda[:,:,y//2:,:x//2,:]
#     vects_2_mean = minmax_lambda([np.mean(vects_2,axis=(1,2,3,4)),lower,upper])
#     vects_3 = vects_nda[:,:,y//2:,x//2:,:]
#     vects_3_mean = minmax_lambda([np.mean(vects_3,axis=(1,2,3,4)),lower,upper])
#     vects_4 = vects_nda[:,:,:y//2,x//2:,:]
#     vects_4_mean = minmax_lambda([np.mean(vects_4,axis=(1,2,3,4)),lower,upper])
    
    
#     # direction per corner
#     dir_1 = directions[:,:,:y//2,:x//2] # top left, plot
#     dir_1_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_2 = directions[:,:,y//2:,:x//2] # top right, plot
#     dir_2_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_3 = directions[:,:,y//2:,x//2:] # bottom left
#     dir_3_mean = minmax_lambda([np.mean(dir_3,axis=(1,2,3)),lower,upper])
#     dir_4 = directions[:,:,:y//2,x//2:] # bottom right
#     dir_4_mean = minmax_lambda([np.mean(dir_4,axis=(1,2,3)),lower,upper])
    

#     # norm per corner
#     norm_mean = minmax_lambda([nda_1d_mean, mid, upper])
#     norm_1 = norm_nda[:,:,:y//2,:x//2]
#     norm_1_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_2 = norm_nda[:,:,y//2:,:x//2]
#     norm_2_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_3 = norm_nda[:,:,y//2:,x//2:]
#     norm_3_mean = minmax_lambda([np.mean(norm_3,axis=(1,2,3)),mid,upper])
#     norm_4 = norm_nda[:,:,:y//2,x//2:]
#     norm_4_mean = minmax_lambda([np.mean(norm_4,axis=(1,2,3)),mid,upper])
    

    ind = detect_phases(dir_1d_mean=dir_1d_mean[:length], length=length)
    
    
    #print('ed: {}, ms: {}, es: {}, pf: {}, md: {}'.format(*ind))
    
        # plot the mean/max norm for one patient oveer time
    if plot:
        
        
        # VECT 2D+t
        vect_2d_t = vects_nda[:,z]
        fig = show_2D_or_3D(vect_2d_t,allow_slicing=False, f_size=(25,2))
        fig = plt.gcf()
        print(type(fig))
        ax = fig.get_axes()[0]
        print(type(ax))
        _ = ax.set_ylabel('v_t \nmid-cavity')

        
        # DIR 2D+t
        dir_2d_t = directions[:,z]
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        #dir_2d_t = minmax_lambda([dir_2d_t,lower,upper])
        div_cmap = 'bwr'
        fig = show_2D_or_3D(dir_2d_t,allow_slicing=False, f_size=(25,2),cmap=div_cmap)
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('alpha_t \nmid-cavity')
        #ax.margins(0,0)
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[len(ax.get_images())//2], cax=cax, orientation='horizontal')

        

        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # DIR 2D T x Z
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        directions_tz = directions.mean(axis=(2,3)) # original: norm()
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        _ = ax.imshow(directions_tz.T, aspect='auto', label='aasd', cmap=div_cmap, origin='lower')
        _ = ax.set_xticks(gtind, minor=False)
        ax2 = ax.twinx()
        _ = ax2.plot(dir_1d_mean, c='black',label='dir 1d+t')
        #_ = ax.set_xticks(gtind, minor=False)
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('dir z+t\napex:base')
        #ax.legend(loc='upper left')
        
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1
        
        #ax=fig.add_subplot(rows,1,pos)
        #_ = ax.plot(dir_1d_mean, c='black',label='dir 1d+t')
        #_ = ax.plot(dir_1_mean, label='top left')
        #_ = ax.plot(dir_2_mean, label='top right')
        #_ = ax.plot(dir_3_mean, label='bottom left')
        #_ = ax.plot(dir_4_mean, label='bottom right')
        #ax.legend(loc='upper left')
        #ax.set_xticks(gtind, minor=False)
        #ax.margins(0.022,0)
        #ax.label_outer()
        #pos=pos+1
        #plt.subplots_adjust(wspace=0, hspace=0.1)
        #plt.show()
        norm_cmap = 'hot'
        
        # NORM 2D + t
        norm_2d_t = norm_nda[:,z]
        norm_2d_t = minmax_lambda([norm_2d_t,mid,upper])
        fig = show_2D_or_3D(norm_2d_t,allow_slicing=False, f_size=(25,2),cmap=norm_cmap, interpolation='none')
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('norm 2d+t\nmid-cavity')
        #ax.margins(0,0)
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[0], cax=cax, orientation='horizontal')
        
        
        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # NORM 2D TxZ
        norm2d = minmax_lambda([norm_nda.mean(axis=(2,3)),mid,upper])
        _ = ax.imshow(norm2d.T, aspect='auto', origin='lower',cmap=norm_cmap, interpolation='none')
        _ = ax.set_xticks(gtind, minor=False)
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('norm z+t\napex:base')
        ax2 = ax.twinx()
        _ = ax2.plot(nda_1d_mean, c='black',label='norm 1d+t')
        #ax.spines['top'].set_visible(False)
        #_ = ax.legend(loc='upper left')
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1

        """ax=fig.add_subplot(rows,1,pos)
        _ = ax.plot(norm_mean, label='norm 1d+t')
        #_ = ax.plot(norm_1_mean, label='top left')
        #_ = ax.plot(norm_2_mean, label='top right')
        #_ = ax.plot(norm_3_mean, label='bottom left')
        #_ = ax.plot(norm_4_mean, label='bottom right')
        ax.legend(loc='upper left')
        ax.set_xticks(gtind, minor=False)
        ax.margins(0.022,0)
        ax.label_outer()
        pos = pos+1
        plt.subplots_adjust(wspace=0, hspace=0.1)"""
        #plt.tight_layout()
        plt.show()

        
    
    return ind
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
p = 2#215
gt_length = int(gt_len[p,:,0].sum())
ind = np.argmax(gt[p][:gt_length],axis=0)
get_phases_from_vects(nda_vects[p][:gt_length], length=gt_length,plot=True,dir_axis=0, gtind=ind)
# try to predict the phases unsupervised just from the norm/direction curve

def get_phases_from_vects(vects_nda, length=-1, plot=True, dir_axis=0, gtind=[], figsize=(25,3)):
    import scipy.signal
    import scipy.signal as sig
    import scipy.ndimage
    from scipy.ndimage import gaussian_filter1d
    
    gted=gtind[0]
    
    # signal smoothing
    kernel_size = 4
    kernel = np.ones(kernel_size) / kernel_size
    z = 0
    
    
    dim_z = vects_nda.shape[1]
    #vects_nda = vects_nda[:int(length)]
    
    
    # define some helper lambdas
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    dim = vects_nda.shape[1:-1]
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]

    
    # norm of the vector
    norm_nda = np.linalg.norm(vects_nda[...,dir_axis:], axis=-1)
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    norm_nda_thres = clip_quantile(norm_nda, 0.9)
    norm_nda = minmax_lambda([norm_nda,0,1])
    norm_msk = norm_nda>=0.0
    norm_nda = norm_nda * norm_msk
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    nda_1d_mean = np.mean(norm_nda,axis=(1,2,3))
    nda_1d_max = np.max(norm_nda,axis=(1,2,3))
    nda_1d_median = np.median(norm_nda,axis=(1,2,3))
    nda_1d_mean_smooth = np.convolve(nda_1d_mean, kernel, mode='same')
    #print(nda_1d_mean_smooth.shape, nda_1d_mean.shape)
    
    
    # direction relative to the center
    
    directions = flow2direction_lambda(vects_nda)[...,0].numpy()
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    directions = minmax_lambda([directions,lower,upper])
    #plt.hist(directions.flatten());plt.show()
    directions = directions * norm_msk
    #plt.hist(directions.flatten());plt.show()
    
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    dir_1d_mean = np.mean(directions,axis=(1,2,3))
    dir_1d_median = np.median(directions,axis=(1,2,3))

    
    # Find phases
    # 1st smooth the direction with a rolling averge (kernelsize=4)
    # or via gaussian filter
    # 2nd min/max normalise each direction vector in a range [-1:1]
    
    
    dir_1d_mean = gaussian_filter1d(dir_1d_mean, sigma=2, mode='wrap') # original: dir_1d_mean
    
    ################# CHANGED !!!!!! ######################## RECENTLY NO CHANGES
    # min/max scaling
    dir_1d_mean = minmax_lambda([dir_1d_mean,lower,upper])
    
    
#     # test with corner specific vector fields
#     y,x = directions.shape[-2:]
    
#     vects_mean = minmax_lambda([np.mean(vects_nda,axis=(1,2,3,4)),lower,upper])
#     vects_1 = vects_nda[:,:,:y//2,:x//2,:]
#     vects_1_mean = minmax_lambda([np.mean(vects_1,axis=(1,2,3,4)),lower,upper])
#     vects_2 = vects_nda[:,:,y//2:,:x//2,:]
#     vects_2_mean = minmax_lambda([np.mean(vects_2,axis=(1,2,3,4)),lower,upper])
#     vects_3 = vects_nda[:,:,y//2:,x//2:,:]
#     vects_3_mean = minmax_lambda([np.mean(vects_3,axis=(1,2,3,4)),lower,upper])
#     vects_4 = vects_nda[:,:,:y//2,x//2:,:]
#     vects_4_mean = minmax_lambda([np.mean(vects_4,axis=(1,2,3,4)),lower,upper])
    
    
#     # direction per corner
#     dir_1 = directions[:,:,:y//2,:x//2] # top left, plot
#     dir_1_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_2 = directions[:,:,y//2:,:x//2] # top right, plot
#     dir_2_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_3 = directions[:,:,y//2:,x//2:] # bottom left
#     dir_3_mean = minmax_lambda([np.mean(dir_3,axis=(1,2,3)),lower,upper])
#     dir_4 = directions[:,:,:y//2,x//2:] # bottom right
#     dir_4_mean = minmax_lambda([np.mean(dir_4,axis=(1,2,3)),lower,upper])
    

#     # norm per corner
#     norm_mean = minmax_lambda([nda_1d_mean, mid, upper])
#     norm_1 = norm_nda[:,:,:y//2,:x//2]
#     norm_1_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_2 = norm_nda[:,:,y//2:,:x//2]
#     norm_2_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_3 = norm_nda[:,:,y//2:,x//2:]
#     norm_3_mean = minmax_lambda([np.mean(norm_3,axis=(1,2,3)),mid,upper])
#     norm_4 = norm_nda[:,:,:y//2,x//2:]
#     norm_4_mean = minmax_lambda([np.mean(norm_4,axis=(1,2,3)),mid,upper])
    

    ind = detect_phases(dir_1d_mean=dir_1d_mean[:length], length=length)
    
    
    #print('ed: {}, ms: {}, es: {}, pf: {}, md: {}'.format(*ind))
    
        # plot the mean/max norm for one patient oveer time
    if plot:
        
        
        # VECT 2D+t
        vect_2d_t = vects_nda[:,z]
        fig = show_2D_or_3D(vect_2d_t,allow_slicing=False, f_size=(25,2))
        fig = plt.gcf()
        print(type(fig))
        ax = fig.get_axes()[0]
        print(type(ax))
        _ = ax.set_ylabel('v_t \nmid-cavity')

        
        # DIR 2D+t
        dir_2d_t = directions[:,z]
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        #dir_2d_t = minmax_lambda([dir_2d_t,lower,upper])
        div_cmap = 'bwr'
        fig = show_2D_or_3D(dir_2d_t,allow_slicing=False, f_size=(25,2),cmap=div_cmap)
        print(type(fig))
        ax = fig.get_axes()[0]
        print(type(ax))
        _ = ax.set_ylabel('alpha_t \nmid-cavity')
        #ax.margins(0,0)
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[len(ax.get_images())//2], cax=cax, orientation='horizontal')

        

        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # DIR 2D T x Z
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        directions_tz = directions.mean(axis=(2,3)) # original: norm()
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        _ = ax.imshow(directions_tz.T, aspect='auto', label='aasd', cmap=div_cmap, origin='lower')
        _ = ax.set_xticks(gtind, minor=False)
        ax2 = ax.twinx()
        _ = ax2.plot(dir_1d_mean, c='black',label='dir 1d+t')
        #_ = ax.set_xticks(gtind, minor=False)
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('dir z+t\napex:base')
        #ax.legend(loc='upper left')
        
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1
        
        #ax=fig.add_subplot(rows,1,pos)
        #_ = ax.plot(dir_1d_mean, c='black',label='dir 1d+t')
        #_ = ax.plot(dir_1_mean, label='top left')
        #_ = ax.plot(dir_2_mean, label='top right')
        #_ = ax.plot(dir_3_mean, label='bottom left')
        #_ = ax.plot(dir_4_mean, label='bottom right')
        #ax.legend(loc='upper left')
        #ax.set_xticks(gtind, minor=False)
        #ax.margins(0.022,0)
        #ax.label_outer()
        #pos=pos+1
        #plt.subplots_adjust(wspace=0, hspace=0.1)
        #plt.show()
        norm_cmap = 'hot'
        
        # NORM 2D + t
        norm_2d_t = norm_nda[:,z]
        norm_2d_t = minmax_lambda([norm_2d_t,mid,upper])
        fig = show_2D_or_3D(norm_2d_t,allow_slicing=False, f_size=(25,2),cmap=norm_cmap, interpolation='none')
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('norm 2d+t\nmid-cavity')
        #ax.margins(0,0)
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[0], cax=cax, orientation='horizontal')
        
        
        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # NORM 2D TxZ
        norm2d = minmax_lambda([norm_nda.mean(axis=(2,3)),mid,upper])
        _ = ax.imshow(norm2d.T, aspect='auto', origin='lower',cmap=norm_cmap, interpolation='none')
        _ = ax.set_xticks(gtind, minor=False)
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('norm z+t\napex:base')
        ax2 = ax.twinx()
        _ = ax2.plot(nda_1d_mean, c='black',label='norm 1d+t')
        #ax.spines['top'].set_visible(False)
        #_ = ax.legend(loc='upper left')
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1

        """ax=fig.add_subplot(rows,1,pos)
        _ = ax.plot(norm_mean, label='norm 1d+t')
        #_ = ax.plot(norm_1_mean, label='top left')
        #_ = ax.plot(norm_2_mean, label='top right')
        #_ = ax.plot(norm_3_mean, label='bottom left')
        #_ = ax.plot(norm_4_mean, label='bottom right')
        ax.legend(loc='upper left')
        ax.set_xticks(gtind, minor=False)
        ax.margins(0.022,0)
        ax.label_outer()
        pos = pos+1
        plt.subplots_adjust(wspace=0, hspace=0.1)"""
        #plt.tight_layout()
        plt.show()

        
    
    return ind
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
p = 2#215
gt_length = int(gt_len[p,:,0].sum())
ind = np.argmax(gt[p][:gt_length],axis=0)
get_phases_from_vects(nda_vects[p][:gt_length], length=gt_length,plot=True,dir_axis=0, gtind=ind)
# try to predict the phases unsupervised just from the norm/direction curve

def get_phases_from_vects(vects_nda, length=-1, plot=True, dir_axis=0, gtind=[], figsize=(25,3)):
    import scipy.signal
    import scipy.signal as sig
    import scipy.ndimage
    from scipy.ndimage import gaussian_filter1d
    
    gted=gtind[0]
    
    # signal smoothing
    kernel_size = 4
    kernel = np.ones(kernel_size) / kernel_size
    z = 0
    
    
    dim_z = vects_nda.shape[1]
    #vects_nda = vects_nda[:int(length)]
    
    
    # define some helper lambdas
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    dim = vects_nda.shape[1:-1]
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]

    
    # norm of the vector
    norm_nda = np.linalg.norm(vects_nda[...,dir_axis:], axis=-1)
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    norm_nda_thres = clip_quantile(norm_nda, 0.9)
    norm_nda = minmax_lambda([norm_nda,0,1])
    norm_msk = norm_nda>=0.0
    norm_nda = norm_nda * norm_msk
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    nda_1d_mean = np.mean(norm_nda,axis=(1,2,3))
    nda_1d_max = np.max(norm_nda,axis=(1,2,3))
    nda_1d_median = np.median(norm_nda,axis=(1,2,3))
    nda_1d_mean_smooth = np.convolve(nda_1d_mean, kernel, mode='same')
    #print(nda_1d_mean_smooth.shape, nda_1d_mean.shape)
    
    
    # direction relative to the center
    
    directions = flow2direction_lambda(vects_nda)[...,0].numpy()
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    directions = minmax_lambda([directions,lower,upper])
    #plt.hist(directions.flatten());plt.show()
    directions = directions * norm_msk
    #plt.hist(directions.flatten());plt.show()
    
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    dir_1d_mean = np.mean(directions,axis=(1,2,3))
    dir_1d_median = np.median(directions,axis=(1,2,3))

    
    # Find phases
    # 1st smooth the direction with a rolling averge (kernelsize=4)
    # or via gaussian filter
    # 2nd min/max normalise each direction vector in a range [-1:1]
    
    
    dir_1d_mean = gaussian_filter1d(dir_1d_mean, sigma=2, mode='wrap') # original: dir_1d_mean
    
    ################# CHANGED !!!!!! ######################## RECENTLY NO CHANGES
    # min/max scaling
    dir_1d_mean = minmax_lambda([dir_1d_mean,lower,upper])
    
    
#     # test with corner specific vector fields
#     y,x = directions.shape[-2:]
    
#     vects_mean = minmax_lambda([np.mean(vects_nda,axis=(1,2,3,4)),lower,upper])
#     vects_1 = vects_nda[:,:,:y//2,:x//2,:]
#     vects_1_mean = minmax_lambda([np.mean(vects_1,axis=(1,2,3,4)),lower,upper])
#     vects_2 = vects_nda[:,:,y//2:,:x//2,:]
#     vects_2_mean = minmax_lambda([np.mean(vects_2,axis=(1,2,3,4)),lower,upper])
#     vects_3 = vects_nda[:,:,y//2:,x//2:,:]
#     vects_3_mean = minmax_lambda([np.mean(vects_3,axis=(1,2,3,4)),lower,upper])
#     vects_4 = vects_nda[:,:,:y//2,x//2:,:]
#     vects_4_mean = minmax_lambda([np.mean(vects_4,axis=(1,2,3,4)),lower,upper])
    
    
#     # direction per corner
#     dir_1 = directions[:,:,:y//2,:x//2] # top left, plot
#     dir_1_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_2 = directions[:,:,y//2:,:x//2] # top right, plot
#     dir_2_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_3 = directions[:,:,y//2:,x//2:] # bottom left
#     dir_3_mean = minmax_lambda([np.mean(dir_3,axis=(1,2,3)),lower,upper])
#     dir_4 = directions[:,:,:y//2,x//2:] # bottom right
#     dir_4_mean = minmax_lambda([np.mean(dir_4,axis=(1,2,3)),lower,upper])
    

#     # norm per corner
#     norm_mean = minmax_lambda([nda_1d_mean, mid, upper])
#     norm_1 = norm_nda[:,:,:y//2,:x//2]
#     norm_1_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_2 = norm_nda[:,:,y//2:,:x//2]
#     norm_2_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_3 = norm_nda[:,:,y//2:,x//2:]
#     norm_3_mean = minmax_lambda([np.mean(norm_3,axis=(1,2,3)),mid,upper])
#     norm_4 = norm_nda[:,:,:y//2,x//2:]
#     norm_4_mean = minmax_lambda([np.mean(norm_4,axis=(1,2,3)),mid,upper])
    

    ind = detect_phases(dir_1d_mean=dir_1d_mean[:length], length=length)
    
    
    #print('ed: {}, ms: {}, es: {}, pf: {}, md: {}'.format(*ind))
    
        # plot the mean/max norm for one patient oveer time
    if plot:
        
        
        # VECT 2D+t
        vect_2d_t = vects_nda[:,z]
        fig1 = show_2D_or_3D(vect_2d_t,allow_slicing=False, f_size=(25,2))
        ax1 = fig1.get_axes()[0]
        _ = ax1.set_ylabel('v_t \nmid-cavity')

        
        # DIR 2D+t
        dir_2d_t = directions[:,z]
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        #dir_2d_t = minmax_lambda([dir_2d_t,lower,upper])
        div_cmap = 'bwr'
        fig = show_2D_or_3D(dir_2d_t,allow_slicing=False, f_size=(25,2),cmap=div_cmap)
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('alpha_t \nmid-cavity')
        #ax.margins(0,0)
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[len(ax.get_images())//2], cax=cax, orientation='horizontal')

        

        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # DIR 2D T x Z
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        directions_tz = directions.mean(axis=(2,3)) # original: norm()
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        _ = ax.imshow(directions_tz.T, aspect='auto', label='aasd', cmap=div_cmap, origin='lower')
        _ = ax.set_xticks(gtind, minor=False)
        ax2 = ax.twinx()
        _ = ax2.plot(dir_1d_mean, c='black',label='dir 1d+t')
        #_ = ax.set_xticks(gtind, minor=False)
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('dir z+t\napex:base')
        #ax.legend(loc='upper left')
        
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1
        
        #ax=fig.add_subplot(rows,1,pos)
        #_ = ax.plot(dir_1d_mean, c='black',label='dir 1d+t')
        #_ = ax.plot(dir_1_mean, label='top left')
        #_ = ax.plot(dir_2_mean, label='top right')
        #_ = ax.plot(dir_3_mean, label='bottom left')
        #_ = ax.plot(dir_4_mean, label='bottom right')
        #ax.legend(loc='upper left')
        #ax.set_xticks(gtind, minor=False)
        #ax.margins(0.022,0)
        #ax.label_outer()
        #pos=pos+1
        #plt.subplots_adjust(wspace=0, hspace=0.1)
        #plt.show()
        norm_cmap = 'hot'
        
        # NORM 2D + t
        norm_2d_t = norm_nda[:,z]
        norm_2d_t = minmax_lambda([norm_2d_t,mid,upper])
        fig = show_2D_or_3D(norm_2d_t,allow_slicing=False, f_size=(25,2),cmap=norm_cmap, interpolation='none')
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('norm 2d+t\nmid-cavity')
        #ax.margins(0,0)
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[0], cax=cax, orientation='horizontal')
        
        
        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # NORM 2D TxZ
        norm2d = minmax_lambda([norm_nda.mean(axis=(2,3)),mid,upper])
        _ = ax.imshow(norm2d.T, aspect='auto', origin='lower',cmap=norm_cmap, interpolation='none')
        _ = ax.set_xticks(gtind, minor=False)
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('norm z+t\napex:base')
        ax2 = ax.twinx()
        _ = ax2.plot(nda_1d_mean, c='black',label='norm 1d+t')
        #ax.spines['top'].set_visible(False)
        #_ = ax.legend(loc='upper left')
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1

        """ax=fig.add_subplot(rows,1,pos)
        _ = ax.plot(norm_mean, label='norm 1d+t')
        #_ = ax.plot(norm_1_mean, label='top left')
        #_ = ax.plot(norm_2_mean, label='top right')
        #_ = ax.plot(norm_3_mean, label='bottom left')
        #_ = ax.plot(norm_4_mean, label='bottom right')
        ax.legend(loc='upper left')
        ax.set_xticks(gtind, minor=False)
        ax.margins(0.022,0)
        ax.label_outer()
        pos = pos+1
        plt.subplots_adjust(wspace=0, hspace=0.1)"""
        #plt.tight_layout()
        plt.show()

        
    
    return ind
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
p = 2#215
gt_length = int(gt_len[p,:,0].sum())
ind = np.argmax(gt[p][:gt_length],axis=0)
get_phases_from_vects(nda_vects[p][:gt_length], length=gt_length,plot=True,dir_axis=0, gtind=ind)
@interact
def compare_phases(i=(0,nda_vects.shape[0]-1), plot=True):
    import matplotlib as mpl
    mpl.rcParams.update(mpl.rcParamsDefault)
    figsize = (25,1)
    weight = 1
    z = 0
    
    phases = ['ED', 'MS', 'ES', 'PF', 'MD']
    cardiac_cycle_length = int(gt_len[i,:,-1].sum())
    # crop or not
    cardiac_cycle_length_ = gt[i].shape[0]-3 # plot the full length minus border for visualisation
    
    vects = nda_vects[i][:cardiac_cycle_length]
    
    
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0) # 
    ind_pred = np.argmax(pred[i][:],axis=0)
    
    temp = mov[i,:cardiac_cycle_length,z] # cardiac_cycle_length_
    fig = show_2D_or_3D(temp,allow_slicing=False)
    ax = fig.get_axes()[0]
    _ = ax.set_ylabel('CMR 2d+t)\nmid-cavity')
    plt.show()
    
    fig= plt.figure(figsize=figsize)
    ax=fig.add_subplot(111)
    ax.set_xticks(ind, minor=False)
    ax.set_yticks([0, 1, 2, 3, 4], minor=False)
    ax.set_yticklabels(phases, rotation=0)
    ind = np.array(ind)
    onehot_gt = np.zeros((ind.size, cardiac_cycle_length)) # cardiac_cycle_length_
    onehot_gt[np.arange(ind.size), ind] = weight
    #ax.imshow(gt[i][:cardiac_cycle_length].T,aspect='auto') # with uncertainity
    ax.imshow(onehot_gt,aspect='auto', cmap='bone') # gt as binary phase2time mapping
    _ = ax.set_yticklabels(phases)
    ax.set_title('gt')
    ax.margins(0,0)
    
    indices = get_phases_from_vects(vects, length=cardiac_cycle_length, plot=plot,dir_axis=0, gtind=ind, figsize=figsize)

    indices = np.array(indices)
    onehot = np.zeros((indices.size, cardiac_cycle_length)) # cardiac_cycle_length_
    onehot[np.arange(indices.size), indices] = weight
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xticks(indices, minor=False)
    ax.set_yticks([0, 1, 2, 3, 4], minor=False)
    ax.set_yticklabels(phases, rotation=0)
    plt.imshow(onehot,aspect='auto', cmap='bone')
    _ = ax.set_yticklabels(phases)
    ax.margins(0,0)
    #ax.set_title('prediction')
    plt.show()
    
    print(phases)
    print('gt:', ind)
    print('u: ', indices)
    
    
    # this would plot the supervised prediction
    #print('p: ', ind_pred)
    """fig, ax = plt.subplots(figsize=figsize)
    ax.set_xticks(ind_pred, minor=False)
    ax.set_yticks([0, 1, 2, 3, 4], minor=False)
    ax.set_yticklabels(phases, rotation=45)
    plt.imshow(pred[i][:cardiac_cycle_length].T,aspect='auto')
    _ = ax.set_yticklabels(phases);plt.show()"""
    # patient 107 gcn --> good curve for plotting
    # 200 no cut
    # 125 no cut
    # 194 moderate cut 
    # 134 strong cut-off
# seaborn
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax.margins(0,0)
for n in dirs:
    _ = ax.plot(n, alpha=0.5, zorder=0)

df = pd.DataFrame(dirs).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='blue', label='d_t',ci='sd',err_style='bars',zorder=2,legend=False,ax=ax)
#_ = ax.set_title('Mean direction +/- SD - aligned at ED')
_ = ax.set_xlabel('Time - linear interpolated to 40 frames')
_ = ax.set_ylabel(r'$\alpha_t')
_ = ax.legend(loc='upper left')
ax2 = ax.twinx()
ax2.margins(0,0)
df = pd.DataFrame(norms).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='black', label='|v_t|',ci='sd', linestyle='dashed',err_style='bars',zorder=2,legend=False,ax=ax2)
_ = ax2.set_ylabel('norm |v_t|')

_ = ax2.legend(loc='upper right')
# seaborn
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax.margins(0,0)
for n in dirs:
    _ = ax.plot(n, alpha=0.5, zorder=0)

df = pd.DataFrame(dirs).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='blue', label='d_t',ci='sd',err_style='bars',zorder=2,legend=False,ax=ax)
#_ = ax.set_title('Mean direction +/- SD - aligned at ED')
_ = ax.set_xlabel('Time - linear interpolated to 40 frames')
_ = ax.set_ylabel(r'$\alpha$_t')
_ = ax.legend(loc='upper left')
ax2 = ax.twinx()
ax2.margins(0,0)
df = pd.DataFrame(norms).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='black', label='|v_t|',ci='sd', linestyle='dashed',err_style='bars',zorder=2,legend=False,ax=ax2)
_ = ax2.set_ylabel('norm |v_t|')

_ = ax2.legend(loc='upper right')
# seaborn
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax.margins(0,0)
for n in dirs:
    _ = ax.plot(n, alpha=0.5, zorder=0)

df = pd.DataFrame(dirs).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='blue', label=r'$\alpha$_t',ci='sd',err_style='bars',zorder=2,legend=False,ax=ax)
#_ = ax.set_title('Mean direction +/- SD - aligned at ED')
_ = ax.set_xlabel('Time - linear interpolated to 40 frames')
_ = ax.set_ylabel(r'$\alpha$_t')
_ = ax.legend(loc='upper left')
ax2 = ax.twinx()
ax2.margins(0,0)
df = pd.DataFrame(norms).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='black', label=r'|$\vec v_t$|',ci='sd', linestyle='dashed',err_style='bars',zorder=2,legend=False,ax=ax2)
_ = ax2.set_ylabel('norm |v_t|')

_ = ax2.legend(loc='upper right')
# seaborn
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax.margins(0,0)
for n in dirs:
    _ = ax.plot(n, alpha=0.5, zorder=0)

df = pd.DataFrame(dirs).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='blue', label=r'$\alpha$_t',ci='sd',err_style='bars',zorder=2,legend=False,ax=ax)
#_ = ax.set_title('Mean direction +/- SD - aligned at ED')
_ = ax.set_xlabel('Time - linear interpolated to 40 frames')
_ = ax.set_ylabel(r'$\alpha$_t')
_ = ax.legend(loc='upper left')
ax2 = ax.twinx()
ax2.margins(0,0)
df = pd.DataFrame(norms).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='black', label=r'|$\vec{v_t}$|',ci='sd', linestyle='dashed',err_style='bars',zorder=2,legend=False,ax=ax2)
_ = ax2.set_ylabel('norm |v_t|')

_ = ax2.legend(loc='upper right')
# seaborn
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax.margins(0,0)
for n in dirs:
    _ = ax.plot(n, alpha=0.5, zorder=0)

df = pd.DataFrame(dirs).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='blue', label=r'$\alpha_t$',ci='sd',err_style='bars',zorder=2,legend=False,ax=ax)
#_ = ax.set_title('Mean direction +/- SD - aligned at ED')
_ = ax.set_xlabel('Time - linear interpolated to 40 frames')
_ = ax.set_ylabel(r'$\alpha$_t')
_ = ax.legend(loc='upper left')
ax2 = ax.twinx()
ax2.margins(0,0)
df = pd.DataFrame(norms).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='black', label=r'|$\vec{v_t}$|',ci='sd', linestyle='dashed',err_style='bars',zorder=2,legend=False,ax=ax2)
_ = ax2.set_ylabel('norm |v_t|')

_ = ax2.legend(loc='upper right')
# seaborn
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax.margins(0,0)
for n in dirs:
    _ = ax.plot(n, alpha=0.5, zorder=0)

df = pd.DataFrame(dirs).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='blue', label=r'$\alpha_t$',ci='sd',err_style='bars',zorder=2,legend=False,ax=ax)
#_ = ax.set_title('Mean direction +/- SD - aligned at ED')
_ = ax.set_xlabel('Time - linear interpolated to 40 frames')
_ = ax.set_ylabel(r'$\alpha$_t')
_ = ax.legend(loc='upper left')
ax2 = ax.twinx()
ax2.margins(0,0)
df = pd.DataFrame(norms).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='black', label=r'|$\vec{v_t}$|',ci='sd', linestyle='dashed',err_style='bars',zorder=2,legend=False,ax=ax2)
_ = ax2.set_ylabel(r'|$\vec{v_t}$|')

_ = ax2.legend(loc='upper right')
# seaborn
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax.margins(0,0)
for n in dirs:
    _ = ax.plot(n, alpha=0.5, zorder=0)

df = pd.DataFrame(dirs).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='blue', label=r'$\alpha_t$',ci='sd',err_style='bars',zorder=2,legend=False,ax=ax)
#_ = ax.set_title('Mean direction +/- SD - aligned at ED')
_ = ax.set_xlabel('Time (t) - linear interpolated to 40 frames')
_ = ax.set_ylabel(r'$\alpha$_t')
_ = ax.legend(loc='upper left')
ax2 = ax.twinx()
ax2.margins(0,0)
df = pd.DataFrame(norms).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='black', label=r'|$\vec{v_t}$|',ci='sd', linestyle='dashed',err_style='bars',zorder=2,legend=False,ax=ax2)
_ = ax2.set_ylabel(r'|$\vec{v_t}$|')

_ = ax2.legend(loc='upper right')
# seaborn
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax.margins(0,0)
for n in dirs:
    _ = ax.plot(n, alpha=0.5, zorder=0)

df = pd.DataFrame(dirs).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='blue', label=r'$\alpha_t$',ci='sd',err_style='bars',zorder=2,legend=False,ax=ax)
#_ = ax.set_title('Mean direction +/- SD - aligned at ED')
_ = ax.set_xlabel('Time (t) - linear interpolated to 40 frames')
_ = ax.set_ylabel(r'$\alpha$_t')
_ = ax.legend(loc='upper left')
ax2 = ax.twinx()
ax2.margins(0,0)
df = pd.DataFrame(norms).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='black', label=r'|$\vec{v_t}$|',ci='sd', linestyle='dashed',err_style='bars',zorder=2,legend=False,ax=ax2)
_ = ax2.set_ylabel(r'|$\vec{v_t}$|')

_ = ax2.legend(loc='upper right')
# seaborn
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax.margins(0,0)
for n in dirs:
    _ = ax.plot(n, alpha=0.5, zorder=0)

df = pd.DataFrame(dirs).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='blue', label=r'$\alpha_t$',ci='sd',err_style='bars',zorder=2,legend=False,ax=ax)
#_ = ax.set_title('Mean direction +/- SD - aligned at ED')
_ = ax.set_xlabel('Time (t) - linear interpolated to 40 frames')
_ = ax.set_ylabel(r'$\alpha_t$')
_ = ax.legend(loc='upper left')
ax2 = ax.twinx()
ax2.margins(0,0)
df = pd.DataFrame(norms).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='black', label=r'|$\vec{v_t}$|',ci='sd', linestyle='dashed',err_style='bars',zorder=2,legend=False,ax=ax2)
_ = ax2.set_ylabel(r'|$\vec{v_t}$|')

_ = ax2.legend(loc='upper right')
# seaborn
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax.margins(0,0)
for n in dirs:
    _ = ax.plot(n, alpha=0.5, zorder=0)

df = pd.DataFrame(dirs).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='blue', label=r'$\alpha_t$',ci='sd',err_style='bars',zorder=2,legend=False,ax=ax)
#_ = ax.set_title('Mean direction +/- SD - aligned at ED')
_ = ax.set_xlabel('Time (t) - linear interpolated to 40 frames')
_ = ax.set_ylabel(r'direction angle $\alpha_t$')
_ = ax.legend(loc='upper left')
ax2 = ax.twinx()
ax2.margins(0,0)
df = pd.DataFrame(norms).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='black', label=r'|$\vec{v_t}$|',ci='sd', linestyle='dashed',err_style='bars',zorder=2,legend=False,ax=ax2)
_ = ax2.set_ylabel(r'|$\vec{v_t}$|')

_ = ax2.legend(loc='upper right')
# seaborn
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax.margins(0,0)
for n in dirs:
    _ = ax.plot(n, alpha=0.5, zorder=0)

df = pd.DataFrame(dirs).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='blue', label=r'$\alpha_t$',ci='sd',err_style='bars',zorder=2,legend=False,ax=ax)
#_ = ax.set_title('Mean direction +/- SD - aligned at ED')
_ = ax.set_xlabel('Time (t) - linear interpolated to 40 frames')
_ = ax.set_ylabel(r'deformation angle $\alpha_t$')
_ = ax.legend(loc='upper left')
ax2 = ax.twinx()
ax2.margins(0,0)
df = pd.DataFrame(norms).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='black', label=r'|$\vec{v_t}$|',ci='sd', linestyle='dashed',err_style='bars',zorder=2,legend=False,ax=ax2)
_ = ax2.set_ylabel(r'|$\vec{v_t}$|')

_ = ax2.legend(loc='upper right')
# seaborn
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax.margins(0,0)
for n in dirs:
    _ = ax.plot(n, alpha=0.5, zorder=0)

df = pd.DataFrame(dirs).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='blue', label=r'$\alpha_t$',ci='sd',err_style='bars',zorder=2,legend=False,ax=ax)
#_ = ax.set_title('Mean direction +/- SD - aligned at ED')
_ = ax.set_xlabel('Time (t) - linear interpolated to 40 frames')
_ = ax.set_ylabel(r'deformation angle $\alpha_t$')
_ = ax.legend(loc='upper left')
ax2 = ax.twinx()
ax2.margins(0,0)
df = pd.DataFrame(norms).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='black', label=r'|$\phi\vec{v_t}$|',ci='sd', linestyle='dashed',err_style='bars',zorder=2,legend=False,ax=ax2)
_ = ax2.set_ylabel(r'|$\vec{v_t}$|')

_ = ax2.legend(loc='upper right')
# seaborn
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax.margins(0,0)
for n in dirs:
    _ = ax.plot(n, alpha=0.5, zorder=0)

df = pd.DataFrame(dirs).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='blue', label=r'$\alpha_t$',ci='sd',err_style='bars',zorder=2,legend=False,ax=ax)
#_ = ax.set_title('Mean direction +/- SD - aligned at ED')
_ = ax.set_xlabel('Time (t) - linear interpolated to 40 frames')
_ = ax.set_ylabel(r'deformation angle $\alpha_t$')
_ = ax.legend(loc='upper left')
ax2 = ax.twinx()
ax2.margins(0,0)
df = pd.DataFrame(norms).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='black', label=r'|$\phi_t$|',ci='sd', linestyle='dashed',err_style='bars',zorder=2,legend=False,ax=ax2)
_ = ax2.set_ylabel(r'|$\phi_t$|')

_ = ax2.legend(loc='upper right')
# seaborn
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax.margins(0,0)
for n in dirs:
    _ = ax.plot(n, alpha=0.5, zorder=0)

df = pd.DataFrame(dirs).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='blue', label=r'$\alpha_t$',ci='sd',err_style='bars',zorder=2,legend=False,ax=ax)
#_ = ax.set_title('Mean direction +/- SD - aligned at ED')
_ = ax.set_xlabel('Time (t) - linear interpolated to 40 frames')
_ = ax.set_ylabel(r'deformation angle ($\alpha_t$)')
_ = ax.legend(loc='upper left')
ax2 = ax.twinx()
ax2.margins(0,0)
df = pd.DataFrame(norms).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='black', label=r'|$\phi_t$|',ci='sd', linestyle='dashed',err_style='bars',zorder=2,legend=False,ax=ax2)
_ = ax2.set_ylabel(r'deformation norm |$\phi_t$|')

_ = ax2.legend(loc='upper right')
# seaborn
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax.margins(0,0)
for n in dirs:
    _ = ax.plot(n, alpha=0.5, zorder=0)

df = pd.DataFrame(dirs).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='blue', label=r'$\alpha_t$',ci='sd',err_style='bars',zorder=2,legend=False,ax=ax)
#_ = ax.set_title('Mean direction +/- SD - aligned at ED')
_ = ax.set_xlabel('Time (t) - linear interpolated to 40 frames')
_ = ax.set_ylabel(r'deformation \angle ($\alpha_t$)')
_ = ax.legend(loc='upper left')
ax2 = ax.twinx()
ax2.margins(0,0)
df = pd.DataFrame(norms).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='black', label=r'|$\phi_t$|',ci='sd', linestyle='dashed',err_style='bars',zorder=2,legend=False,ax=ax2)
_ = ax2.set_ylabel(r'deformation norm |$\phi_t$|')

_ = ax2.legend(loc='upper right')
# seaborn
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax.margins(0,0)
for n in dirs:
    _ = ax.plot(n, alpha=0.5, zorder=0)

df = pd.DataFrame(dirs).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='blue', label=r'$\alpha_t$',ci='sd',err_style='bars',zorder=2,legend=False,ax=ax)
#_ = ax.set_title('Mean direction +/- SD - aligned at ED')
_ = ax.set_xlabel('Time (t) - linear interpolated to 40 frames')
_ = ax.set_ylabel(r'deformation $\angle$ ($\alpha_t$)')
_ = ax.legend(loc='upper left')
ax2 = ax.twinx()
ax2.margins(0,0)
df = pd.DataFrame(norms).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='black', label=r'|$\phi_t$|',ci='sd', linestyle='dashed',err_style='bars',zorder=2,legend=False,ax=ax2)
_ = ax2.set_ylabel(r'deformation norm |$\phi_t$|')

_ = ax2.legend(loc='upper right')
# seaborn
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax.margins(0,0)
for n in dirs:
    _ = ax.plot(n, alpha=0.5, zorder=0)

df = pd.DataFrame(dirs).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='blue', label=r'$\alpha_t$',ci='sd',err_style='bars',zorder=2,legend=False,ax=ax)
#_ = ax.set_title('Mean direction +/- SD - aligned at ED')
_ = ax.set_xlabel('Time (t) - linear interpolated to 40 frames')
_ = ax.set_ylabel(r'deformation $\1angle$ ($\alpha_t$)')
_ = ax.legend(loc='upper left')
ax2 = ax.twinx()
ax2.margins(0,0)
df = pd.DataFrame(norms).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='black', label=r'|$\phi_t$|',ci='sd', linestyle='dashed',err_style='bars',zorder=2,legend=False,ax=ax2)
_ = ax2.set_ylabel(r'deformation norm |$\phi_t$|')

_ = ax2.legend(loc='upper right')
# seaborn
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax.margins(0,0)
for n in dirs:
    _ = ax.plot(n, alpha=0.5, zorder=0)

df = pd.DataFrame(dirs).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='blue', label=r'$\alpha_t$',ci='sd',err_style='bars',zorder=2,legend=False,ax=ax)
#_ = ax.set_title('Mean direction +/- SD - aligned at ED')
_ = ax.set_xlabel('Time (t) - linear interpolated to 40 frames')
_ = ax.set_ylabel(r'deformation $\angle$ ($\alpha_t$)')
_ = ax.legend(loc='upper left')
ax2 = ax.twinx()
ax2.margins(0,0)
df = pd.DataFrame(norms).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='black', label=r'|$\phi_t$|',ci='sd', linestyle='dashed',err_style='bars',zorder=2,legend=False,ax=ax2)
_ = ax2.set_ylabel(r'deformation norm |$\phi_t$|')

_ = ax2.legend(loc='upper right')
# seaborn
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax.margins(0,0)
for n in dirs:
    _ = ax.plot(n, alpha=0.5, zorder=0)

df = pd.DataFrame(dirs).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='blue', label=r'$\alpha_t$',ci='sd',err_style='bars',zorder=2,legend=False,ax=ax)
#_ = ax.set_title('Mean direction +/- SD - aligned at ED')
_ = ax.set_xlabel('Time (t) - linear interpolated to 40 frames')
_ = ax.set_ylabel(r'deformation $\measuredangle$ ($\alpha_t$)')
_ = ax.legend(loc='upper left')
ax2 = ax.twinx()
ax2.margins(0,0)
df = pd.DataFrame(norms).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='black', label=r'|$\phi_t$|',ci='sd', linestyle='dashed',err_style='bars',zorder=2,legend=False,ax=ax2)
_ = ax2.set_ylabel(r'deformation norm |$\phi_t$|')

_ = ax2.legend(loc='upper right')
# seaborn
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax.margins(0,0)
for n in dirs:
    _ = ax.plot(n, alpha=0.5, zorder=0)

df = pd.DataFrame(dirs).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='blue', label=r'$\alpha_t$',ci='sd',err_style='bars',zorder=2,legend=False,ax=ax)
#_ = ax.set_title('Mean direction +/- SD - aligned at ED')
_ = ax.set_xlabel('Time (t) - linear interpolated to 40 frames')
_ = ax.set_ylabel(r'direction angle ($\alpha_t$) of \phi_t')
_ = ax.legend(loc='upper left')
ax2 = ax.twinx()
ax2.margins(0,0)
df = pd.DataFrame(norms).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='black', label=r'|$\phi_t$|',ci='sd', linestyle='dashed',err_style='bars',zorder=2,legend=False,ax=ax2)
_ = ax2.set_ylabel(r'deformation norm |$\phi_t$|')

_ = ax2.legend(loc='upper right')
# seaborn
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax.margins(0,0)
for n in dirs:
    _ = ax.plot(n, alpha=0.5, zorder=0)

df = pd.DataFrame(dirs).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='blue', label=r'$\alpha_t$',ci='sd',err_style='bars',zorder=2,legend=False,ax=ax)
#_ = ax.set_title('Mean direction +/- SD - aligned at ED')
_ = ax.set_xlabel('Time (t) - linear interpolated to 40 frames')
_ = ax.set_ylabel(r'Angle ($\alpha_t$) of \phi_t')
_ = ax.legend(loc='upper left')
ax2 = ax.twinx()
ax2.margins(0,0)
df = pd.DataFrame(norms).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='black', label=r'|$\phi_t$|',ci='sd', linestyle='dashed',err_style='bars',zorder=2,legend=False,ax=ax2)
_ = ax2.set_ylabel(r'deformation norm |$\phi_t$|')

_ = ax2.legend(loc='upper right')
# seaborn
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax.margins(0,0)
for n in dirs:
    _ = ax.plot(n, alpha=0.5, zorder=0)

df = pd.DataFrame(dirs).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='blue', label=r'$\alpha_t$',ci='sd',err_style='bars',zorder=2,legend=False,ax=ax)
#_ = ax.set_title('Mean direction +/- SD - aligned at ED')
_ = ax.set_xlabel('Time (t) - linear interpolated to 40 frames')
_ = ax.set_ylabel(r'Angle ($\alpha_t$) of $\phi_t$')
_ = ax.legend(loc='upper left')
ax2 = ax.twinx()
ax2.margins(0,0)
df = pd.DataFrame(norms).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='black', label=r'|$\phi_t$|',ci='sd', linestyle='dashed',err_style='bars',zorder=2,legend=False,ax=ax2)
_ = ax2.set_ylabel(r'deformation norm |$\phi_t$|')

_ = ax2.legend(loc='upper right')
# seaborn
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax.margins(0,0)
for n in dirs:
    _ = ax.plot(n, alpha=0.5, zorder=0)

df = pd.DataFrame(dirs).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='blue', label=r'$\alpha_t$',ci='sd',err_style='bars',zorder=2,legend=False,ax=ax)
#_ = ax.set_title('Mean direction +/- SD - aligned at ED')
_ = ax.set_xlabel('Time (t) - linear interpolated to 40 frames')
_ = ax.set_ylabel(r'Angle ($\alpha_t$) of $\phi_t$')
_ = ax.legend(loc='upper left')
ax2 = ax.twinx()
ax2.margins(0,0)
df = pd.DataFrame(norms).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='black', label=r'|$\phi_t$|',ci='sd', linestyle='dashed',err_style='bars',zorder=2,legend=False,ax=ax2)
_ = ax2.set_ylabel(r'Norm |$\phi_t$|')

_ = ax2.legend(loc='upper right')
# seaborn
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax.margins(0,0)
for n in dirs:
    _ = ax.plot(n, alpha=0.5, zorder=0)

df = pd.DataFrame(dirs).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='blue', label=r'$\alpha_t$',ci='sd',err_style='bars',zorder=2,legend=False,ax=ax)
#_ = ax.set_title('Mean direction +/- SD - aligned at ED')
_ = ax.set_xlabel('Time (t) - linear interpolated to 40 frames')
_ = ax.set_ylabel(r'Angle $\alpha_t$ of $\phi_t$')
_ = ax.legend(loc='upper left')
ax2 = ax.twinx()
ax2.margins(0,0)
df = pd.DataFrame(norms).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='black', label=r'|$\phi_t$|',ci='sd', linestyle='dashed',err_style='bars',zorder=2,legend=False,ax=ax2)
_ = ax2.set_ylabel(r'Norm |$\phi_t$|')

_ = ax2.legend(loc='upper right')
# load the vectors
pathstovectnpy = sorted(glob.glob(os.path.join(vects_chooser.selected, 'moved', '*vects*.npy')))
print(pathstovectnpy)
nda_vects = np.concatenate([np.load(path_) for path_ in pathstovectnpy],axis=0)
print(nda_vects.shape)
# load the phase gt and pred
pred_path = os.path.join(vects_chooser.selected, 'pred')
pathtsophasenpy = sorted(glob.glob(os.path.join(pred_path, '*gtpred*.npy')))
print(pathtsophasenpy)
nda_phase = np.concatenate([np.load(path_) for path_ in pathtsophasenpy],axis=1)
print(nda_phase.shape)
gt_, pred_ = np.split(nda_phase, axis=0,indices_or_sections=2)
print(gt_.shape)
gt= gt_[0,:,0]
pred = pred_[0,:,0]
print(gt.shape)
gt_len = gt_[0,:,1]
# load some moved examples for easier understanding of the dimensions
pathtomoved = sorted(glob.glob(os.path.join(vects_chooser.selected, 'moved','*moved*.npy')))
print(len(pathtomoved))
mov = np.concatenate([np.load(path_) for path_ in pathtomoved],axis=0)
print(mov.shape) # patient,time,spatial-z,y,x,channel, e.g.: 69,40,16,64,64,1
# plot one moved timestep of one patient = one 3D volume 
# should go from apex to base
from src.visualization.Visualize import show_2D_or_3D
temp = mov[10,0,...,0]
_ = show_2D_or_3D(temp, cmap='gray', interpolation='none')
import tensorflow as tf
from tensorflow.image import ssim
ssim(mov[10,0], mov[10,1],max_val=1.0, filter_size=11,
                          filter_sigma=1.5, k1=0.01, k2=0.03).numpy().mean()
from skimage.metrics import structural_similarity as ssim
ssim(mov[10,0], mov[10,1],multichannel=True)
plt.imshow(gt[0].T)
plt.imshow(pred[0].T)
import tensorflow as tf
import sys
# returns a matrix with the indicies as values, similar to np.indicies
def get_idxs_tf(x):
    return tf.cast(
        tf.reshape(tf.where(tf.ones((x[0], x[1], x[2]))), (x[0], x[1], x[2], 3)),
        tf.float32)


# returns a matrix with vectors pointing to the center
def get_centers_tf(x):
    return tf.cast(
        tf.tile(tf.convert_to_tensor([x[0] // 2, x[1] // 2, x[2] // 2])[tf.newaxis, tf.newaxis, tf.newaxis, ...],
                (x[0], x[1], x[2], 1)), tf.float32)

def get_angle_tf(a, b):
    # this should work for batches of n-dimensional vectors
    # α = arccos[(a · b) / (|a| * |b|)]
    # |v| = √(x² + y² + z²)
    """
    in 3D space
    If vectors a = [xa, ya, za], b = [xb, yb, zb], then:
    α = arccos[(xa * xb + ya * yb + za * zb) / (√(xa2 + ya2 + za2) * √(xb2 + yb2 + zb2))]
    """
    import math as m
    pi = tf.constant(m.pi)
    b = tf.cast(b, dtype=a.dtype)
    inner = tf.einsum('...i,...i->...', a, b)
    norms = tf.norm(a, ord='euclidean', axis=-1) * tf.norm(b, ord='euclidean', axis=-1)  # [...,None]
    cos = inner / (norms + sys.float_info.epsilon)
    #rad = tf.math.acos(tf.clip_by_value(cos, -1.0, 1.0))
    # rad2deg conversion
    #deg = rad * (180.0/pi)
    return cos[..., tf.newaxis]

dim = [16,64,64]
idx = get_idxs_tf(dim)
c = get_centers_tf(dim)
centers = c - idx
centers_tensor = centers[tf.newaxis, ...]
flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x, centers_tensor), name='flow2direction')
minmax_lambda = lambda x : x[1] + (((x[0] - np.min(x[0])) * (x[2]-x[1]))/(np.max(x[0]) - np.min(x[0])))
# define some central params
lower, mid, upper = -1,0, 1
# align norm an direction by the cardiac phase ED
# resample all 1D feature vectors to the same length
# this should help to validate the pre-defined rules, and if both together explains the cardiac phases

norms = []
dirs= []
cycle_len=[]
gt_ind = []
xval = np.linspace(0,1,40)

for i,n in enumerate(nda_vects[:]):
    cardiac_cycle_length = int(gt_len[i,:,0].sum())
    cycle_len.append(cardiac_cycle_length)
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
    gt_ind.append(ind)
    dir_axis=0
    gt_ed = ind[0]
    #assert cardiac_cycle_length>=gt_ed
    temp = n[:cardiac_cycle_length]
    
    
    dim = temp.shape[1:-1]
    #print(dim)
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    #print('centers: ',c.dtype)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    # direction
    directions = flow2direction_lambda(temp)[...,0]
    # direction mean
    directions = np.mean(directions,axis=(1,2,3))
    # direction ed aligned
    directions = np.roll(directions, -1*gt_ed)
    # direction interpolate to unique length
    f = interp1d(np.linspace(0,1,directions.shape[0]), directions, kind='linear')
    directions = f(xval)
    # direction min/max normalised between -1,1

    directions = minmax_lambda([directions,lower,upper])
    
    # magnitude/norm as mean
    norm_nda = np.linalg.norm(temp, axis=-1).mean(axis=(1,2,3))
    # norm ed aligned
    norm_nda = np.roll(norm_nda, -1*gt_ed)
    # interpolate to unique length
    f = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')
    norm_nda = f(xval)
    # norm min/max aligned to 0,1
    norm_nda = minmax_lambda([norm_nda,mid,upper])
    
    norms.append(norm_nda)
    dirs.append(directions)
# get the mean values
gt_ind = np.stack(gt_ind,axis=0)
cycle_len = np.stack(cycle_len, axis=0)
norms = np.stack(norms, axis=0)
dirs = np.stack(dirs, axis=0)
norms_m = [norms.mean(axis=0)]
dirs_m=[dirs.mean(axis=0)]
# seaborn
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax.margins(0,0)
for n in dirs:
    _ = ax.plot(n, alpha=0.5, zorder=0)

df = pd.DataFrame(dirs).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='blue', label=r'$\alpha_t$',ci='sd',err_style='bars',zorder=2,legend=False,ax=ax)
#_ = ax.set_title('Mean direction +/- SD - aligned at ED')
_ = ax.set_xlabel('Time (t) - linear interpolated to 40 frames')
_ = ax.set_ylabel(r'Angle $\alpha_t$ of $\phi_t$')
_ = ax.legend(loc='upper left')
ax2 = ax.twinx()
ax2.margins(0,0)
df = pd.DataFrame(norms).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='black', label=r'|$\phi_t$|',ci='sd', linestyle='dashed',err_style='bars',zorder=2,legend=False,ax=ax2)
_ = ax2.set_ylabel(r'Norm |$\phi_t$|')

_ = ax2.legend(loc='upper right')
folds_chooser = FileChooser(os.path.join(os.getcwd(),'/mnt/ssd/git/dynamic-cmr-models/exp/phasereg_v3/acdc/'), '')
display(folds_chooser)
# define df and helper for pathology extractio
# this should have the same order as our inference data
#df_meta = pd.read_csv(folds_chooser.selected)
df_meta = pd.read_csv('/mnt/sds/sd20i001/sven/data/acdc/02_imported_4D_unfiltered/df_kfold.csv')
df_meta = df_meta.loc[:, ~df_meta.columns.str.contains('^Unnamed')]
df_meta = df_meta[df_meta.patient != 'patient090']  #we excluded this patient

def get_msk_for_pathology(df_, pathology='minf'):
    msk = []
    for f in [0,1,2,3,4]:
        patients = df_[df_.fold.isin([f])]
        patients = patients[patients['phase']=='ED']
        pat = patients[patients['modality'] == 'test']['patient'].str.lower().unique()
        sub_df = patients[patients.patient.isin(pat)].drop_duplicates(ignore_index=True, subset='patient')
        sub_df = sub_df.drop('x_path',axis=1).drop('y_path',axis=1)
        msk.append(sub_df['pathology'].str.lower()==pathology.lower())
    return np.concatenate(msk)
df_meta
# interactive plot per pathology
@interact
def plot_per_pathology(p=df_meta.pathology.unique()):
    print(p)
    msk = get_msk_for_pathology(df_meta, p)

    import seaborn as sb
    sb.set_context('paper')
    sb.set(font_scale = 2)
    fig, ax = plt.subplots(figsize=(25,5))
    ax.margins(0,0)
    dirs_minf = dirs[msk]
    norms_minf = norms[msk]
    for n in dirs_minf:
        _ = ax.plot(n, alpha=0.5, zorder=0)

    df = pd.DataFrame(dirs_minf).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, color='blue', label='d_t',ci='sd',err_style='bars',zorder=2,legend=False,ax=ax)
    #_ = ax.set_title('Mean direction +/- SD - aligned at ED')
    _ = ax.set_xlabel('Time - linear interpolated to 40 frames')
    _ = ax.set_ylabel('direction d_t')
    _ = ax.legend(loc='upper left')
    ax2 = ax.twinx()
    ax2.margins(0,0)
    df = pd.DataFrame(norms_minf).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, color='black', label='|v_t|',ci='sd', linestyle='dashed',err_style='bars',zorder=2,legend=False,ax=ax2)
    _ = ax2.set_ylabel('norm |v_t|')

    _ = ax2.legend(loc='upper right')
    plt.show()
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax2 = ax.twinx()
ax2.margins(0,0)
ax.margins(0,0)
for p in df_meta.pathology.unique():
    msk = get_msk_for_pathology(df_meta, p)
    dirs_minf = dirs[msk]
    norms_minf = norms[msk]
    #for n in dirs_minf:
    #    _ = ax.plot(n, alpha=0.5, zorder=0)

    df = pd.DataFrame(dirs_minf).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, label='d_t_{}'.format(p),ci='sd',err_style=None,zorder=2,legend=False,ax=ax)
    #_ = ax.set_title('Mean direction +/- SD - aligned at ED')
    _ = ax.set_xlabel('Time - linear interpolated to 40 frames')
    _ = ax.set_ylabel('direction d_t')
    _ = ax.legend(loc='upper left')
    
    df = pd.DataFrame(norms_minf).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, label='|v_t|_{}'.format(p),ci='sd', linestyle='dashed',err_style=None,zorder=2,legend=False,ax=ax2)
    _ = ax2.set_ylabel('norm |v_t|')

    _ = ax2.legend(loc='upper right')
plt.show()
# plot one line per patient - norm
fig, ax = plt.subplots(figsize=(25,5))
for n in norms:
    _ = plt.plot(n)
_ = ax.set_title('Magnitudes aligned at ED, resampling shape 40,')
from src.visualization.Visualize import show_2D_or_3D
from src.data.Preprocess import clip_quantile
nda_temp = nda_vects[0]
# norm_1 = nda_temp[:,:,:32,:32]# top left
norm_1 = nda_temp[:,:,:32,32:]# top right
norm_1 = np.linalg.norm(norm_1, axis=-1)
nda_temp = np.linalg.norm(nda_temp, axis=-1)
plt.hist(nda_temp.flatten());plt.show()
nda_temp = clip_quantile(nda_temp, 0.99)
nda_temp = minmax_lambda([nda_temp,0,1])
plt.hist(nda_temp.flatten())
plt.show()
nda_msk = (nda_temp>=0.2).astype(np.float32)
nda_temp = nda_temp * nda_msk
#plt.hist(nda_temp.flatten())
print(nda_temp.shape)
_ = show_2D_or_3D(nda_temp[:,0], allow_slicing=True)
_ = show_2D_or_3D(nda_temp[:,-1], allow_slicing=True)

# top right
_ = show_2D_or_3D(norm_1[:,0], allow_slicing=True,cmap='gray', interpolation='none')
from src.visualization.Visualize import show_2D_or_3D
from src.data.Preprocess import clip_quantile
nda_temp = nda_vects[0]
# norm_1 = nda_temp[:,:,:32,:32]# top left
norm_1 = nda_temp[:,:,:32,32:]# top right
norm_1 = np.linalg.norm(norm_1, axis=-1)
nda_norm = np.linalg.norm(nda_temp, axis=-1)
plt.hist(nda_norm.flatten());plt.show()
nda_temp = clip_quantile(nda_norm, 0.99)
nda_norm = minmax_lambda([nda_norm,0,1])
plt.hist(nda_norm.flatten())
plt.show()
# mask phi with a threshold norm matrix 
nda_msk = (nda_norm>=0.2).astype(np.float32)
nda_temp = nda_temp * nda_msk
#plt.hist(nda_temp.flatten())
print(nda_temp.shape)
_ = show_2D_or_3D(nda_temp[:,0], allow_slicing=True)
_ = show_2D_or_3D(nda_temp[:,-1], allow_slicing=True)

# top right
_ = show_2D_or_3D(norm_1[:,0], allow_slicing=True,cmap='gray', interpolation='none')
from src.visualization.Visualize import show_2D_or_3D
from src.data.Preprocess import clip_quantile
nda_temp = nda_vects[0]
# norm_1 = nda_temp[:,:,:32,:32]# top left
norm_1 = nda_temp[:,:,:32,32:]# top right
norm_1 = np.linalg.norm(norm_1, axis=-1)
nda_norm = np.linalg.norm(nda_temp, axis=-1)
plt.hist(nda_norm.flatten());plt.show()
nda_norm = clip_quantile(nda_norm, 0.99)
nda_norm = minmax_lambda([nda_norm,0,1])
plt.hist(nda_norm.flatten())
plt.show()
# mask phi with a threshold norm matrix 
nda_msk = (nda_norm>=0.2).astype(np.float32)
nda_temp = nda_temp * nda_msk
#plt.hist(nda_temp.flatten())
print(nda_temp.shape)
_ = show_2D_or_3D(nda_temp[:,0], allow_slicing=True)
_ = show_2D_or_3D(nda_temp[:,-1], allow_slicing=True)

# top right
_ = show_2D_or_3D(norm_1[:,0], allow_slicing=True,cmap='gray', interpolation='none')
from src.visualization.Visualize import show_2D_or_3D
from src.data.Preprocess import clip_quantile
nda_temp = nda_vects[0]
# norm_1 = nda_temp[:,:,:32,:32]# top left
norm_1 = nda_temp[:,:,:32,32:]# top right
norm_1 = np.linalg.norm(norm_1, axis=-1)
nda_norm = np.linalg.norm(nda_temp, axis=-1)
plt.hist(nda_norm.flatten());plt.show()
nda_norm = clip_quantile(nda_norm, 0.99)
nda_norm = minmax_lambda([nda_norm,0,1])
plt.hist(nda_norm.flatten())
plt.show()
# mask phi with a threshold norm matrix 
nda_msk = (nda_norm>=0.2).astype(np.float32)
nda_temp = nda_norm * nda_msk
#plt.hist(nda_temp.flatten())
print(nda_temp.shape)
_ = show_2D_or_3D(nda_temp[:,0], allow_slicing=True)
_ = show_2D_or_3D(nda_temp[:,-1], allow_slicing=True)

# top right
_ = show_2D_or_3D(norm_1[:,0], allow_slicing=True,cmap='gray', interpolation='none')
from src.visualization.Visualize import show_2D_or_3D
from src.data.Preprocess import clip_quantile
nda_temp = nda_vects[0]
# norm_1 = nda_temp[:,:,:32,:32]# top left
norm_1 = nda_temp[:,:,:32,32:]# top right
norm_1 = np.linalg.norm(norm_1, axis=-1)
nda_norm_ = np.linalg.norm(nda_temp, axis=-1)
plt.hist(nda_norm_.flatten());plt.show()
nda_norm = clip_quantile(nda_norm_, 0.99)
nda_norm = minmax_lambda([nda_norm,0,1])
plt.hist(nda_norm.flatten())
plt.show()
# mask phi with a threshold norm matrix 
nda_msk = (nda_norm>=0.2).astype(np.float32)
nda_temp = nda_norm * nda_msk
#plt.hist(nda_temp.flatten())
print(nda_temp.shape)
_ = show_2D_or_3D(nda_norm_[:,0], allow_slicing=True)
_ = show_2D_or_3D(nda_temp[:,0], allow_slicing=True)
_ = show_2D_or_3D(nda_temp[:,-1], allow_slicing=True)

# top right
_ = show_2D_or_3D(norm_1[:,0], allow_slicing=True,cmap='gray', interpolation='none')
norms.shape
import sklearn
from sklearn.decomposition import PCA
low_dim = PCA(n_components=2).fit_transform(norms)
plt.scatter(low_dim)
import sklearn
from sklearn.decomposition import PCA
low_dim = PCA(n_components=2).fit_transform(norms)
plt.scatter(low_dim[:0],low_dim[:1])
import sklearn
from sklearn.decomposition import PCA
low_dim = PCA(n_components=2).fit_transform(norms)
plt.scatter(low_dim[:,0],low_dim[:,1])
pathologies = np.zeros(40)
for p in df_meta.pathology.unique():
    msk = get_msk_for_pathology(df_meta, p)
    print(msk.shape)
    dirs_minf = dirs[msk]
    norms_minf = norms[msk]
pathologies = np.zeros(40)
for p,i in enumerate(df_meta.pathology.unique()):
    msk = get_msk_for_pathology(df_meta, p)
    print(msk.shape)
    pathologies[msk] = i
pathologies = np.zeros(40)
for i,p in enumerate(df_meta.pathology.unique()):
    msk = get_msk_for_pathology(df_meta, p)
    print(msk.shape)
    pathologies[msk] = i
pathologies = np.zeros(99)
for i,p in enumerate(df_meta.pathology.unique()):
    msk = get_msk_for_pathology(df_meta, p)
    print(msk.shape)
    pathologies[msk] = i
import sklearn
from sklearn.decomposition import PCA
y = np.zeros(99)
for i,p in enumerate(df_meta.pathology.unique()):
    msk = get_msk_for_pathology(df_meta, p)
    print(msk.shape)
    y[msk] = i
low_dim = PCA(n_components=2).fit_transform(norms)
plt.scatter(low_dim[:,0],low_dim[:,1], c=y)
import sklearn
from sklearn.decomposition import PCA
y = np.zeros(99)
for i,p in enumerate(df_meta.pathology.unique()):
    msk = get_msk_for_pathology(df_meta, p)
    y[msk] = i
low_dim = PCA(n_components=2).fit_transform(norms)
plt.scatter(low_dim[:,0],low_dim[:,1], c=y)
import sklearn
from sklearn.decomposition import PCA
y = np.zeros(99)
for i,p in enumerate(df_meta.pathology.unique()):
    msk = get_msk_for_pathology(df_meta, p)
    y[msk] = i
low_dim = PCA(n_components=2).fit_transform(norms)
_ =plt.scatter(low_dim[:,0],low_dim[:,1], c=y)
import sklearn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
y = np.zeros(99)
for i,p in enumerate(df_meta.pathology.unique()):
    msk = get_msk_for_pathology(df_meta, p)
    y[msk] = i
low_dim = TSNE(n_components=2).fit_transform(norms)
#low_dim = PCA(n_components=2).fit_transform(norms)
_ =plt.scatter(low_dim[:,0],low_dim[:,1], c=y)
# align norm an direction by the cardiac phase ED
# resample all 1D feature vectors to the same length
# this should help to validate the pre-defined rules, and if both together explains the cardiac phases

norms = []
dirs= []
cycle_len=[]
gt_ind = []
xval = np.linspace(0,1,40)

for i,n in enumerate(nda_vects[:]):
    cardiac_cycle_length = int(gt_len[i,:,0].sum())
    cycle_len.append(cardiac_cycle_length)
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
    gt_ind.append(ind)
    dir_axis=0
    gt_ed = ind[0]
    #assert cardiac_cycle_length>=gt_ed
    temp = n[:cardiac_cycle_length]
    
    
    dim = temp.shape[1:-1]
    #print(dim)
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    #print('centers: ',c.dtype)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    # direction
    directions = flow2direction_lambda(temp)[...,0]
    # direction mean
    directions = np.mean(directions,axis=(1,2,3))
    # direction ed aligned
    directions = np.roll(directions, -1*gt_ed)
    # direction interpolate to unique length
    f = interp1d(np.linspace(0,1,directions.shape[0]), directions, kind='linear')
    directions = f(xval)
    # direction min/max normalised between -1,1

    directions = minmax_lambda([directions,lower,upper])
    
    # magnitude/norm as mean
    norm_full = np.linalg.norm(temp, axis=-1)
    norm_nda = norm_full.mean(axis=(1,2,3))
    # norm ed aligned
    norm_nda = np.roll(norm_nda, -1*gt_ed)
    norm_full = np.roll(norm_full, -1*gt_ed)
    # interpolate to unique length
    f = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')
    norm_nda = f(xval)
    norm_full = f(norm_full)
    # norm min/max aligned to 0,1
    norm_nda = minmax_lambda([norm_nda,mid,upper])
    
    norms.append(norm_nda)
    dirs.append(directions)
# align norm an direction by the cardiac phase ED
# resample all 1D feature vectors to the same length
# this should help to validate the pre-defined rules, and if both together explains the cardiac phases

norms = []
dirs= []
cycle_len=[]
gt_ind = []
xval = np.linspace(0,1,40)

for i,n in enumerate(nda_vects[:]):
    cardiac_cycle_length = int(gt_len[i,:,0].sum())
    cycle_len.append(cardiac_cycle_length)
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
    gt_ind.append(ind)
    dir_axis=0
    gt_ed = ind[0]
    #assert cardiac_cycle_length>=gt_ed
    temp = n[:cardiac_cycle_length]
    
    
    dim = temp.shape[1:-1]
    #print(dim)
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    #print('centers: ',c.dtype)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    # direction
    directions = flow2direction_lambda(temp)[...,0]
    # direction mean
    directions = np.mean(directions,axis=(1,2,3))
    # direction ed aligned
    directions = np.roll(directions, -1*gt_ed)
    # direction interpolate to unique length
    f = interp1d(np.linspace(0,1,directions.shape[0]), directions, kind='linear')
    directions = f(xval)
    # direction min/max normalised between -1,1

    directions = minmax_lambda([directions,lower,upper])
    
    # magnitude/norm as mean
    norm_full = np.linalg.norm(temp, axis=-1)
    norm_nda = norm_full.mean(axis=(1,2,3))
    # norm ed aligned
    norm_nda = np.roll(norm_nda, -1*gt_ed)
    norm_full = np.roll(norm_full, -1*gt_ed)
    # interpolate to unique length
    f = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')
    norm_nda = f(xval)
    #norm_full = f(norm_full)
    # norm min/max aligned to 0,1
    norm_nda = minmax_lambda([norm_nda,mid,upper])
    
    norms.append(norm_nda)
    dirs.append(directions)
# align norm an direction by the cardiac phase ED
# resample all 1D feature vectors to the same length
# this should help to validate the pre-defined rules, and if both together explains the cardiac phases

norms = []
dirs= []
cycle_len=[]
gt_ind = []
xval = np.linspace(0,1,40)

for i,n in enumerate(nda_vects[:]):
    cardiac_cycle_length = int(gt_len[i,:,0].sum())
    cycle_len.append(cardiac_cycle_length)
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
    gt_ind.append(ind)
    dir_axis=0
    gt_ed = ind[0]
    #assert cardiac_cycle_length>=gt_ed
    temp = n[:cardiac_cycle_length]
    
    
    dim = temp.shape[1:-1]
    #print(dim)
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    #print('centers: ',c.dtype)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    # direction
    directions = flow2direction_lambda(temp)[...,0]
    # direction mean
    directions = np.mean(directions,axis=(1,2,3))
    # direction ed aligned
    directions = np.roll(directions, -1*gt_ed)
    # direction interpolate to unique length
    f = interp1d(np.linspace(0,1,directions.shape[0]), directions, kind='linear')
    directions = f(xval)
    # direction min/max normalised between -1,1

    directions = minmax_lambda([directions,lower,upper])
    
    # magnitude/norm as mean
    norm_full = np.linalg.norm(temp, axis=-1)
    norm_nda = norm_full.mean(axis=(1,2,3))
    # norm ed aligned
    norm_nda = np.roll(norm_nda, -1*gt_ed)
    norm_full = np.roll(norm_full, -1*gt_ed)
    # interpolate to unique length
    f = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')
    #norm_nda = f(xval)
    norm_nda = np.interp(xvals, np.linspace(0,1,norm_nda.shape[0]), norm_nda)
    #norm_full = f(norm_full)
    # norm min/max aligned to 0,1
    norm_nda = minmax_lambda([norm_nda,mid,upper])
    
    norms.append(norm_nda)
    dirs.append(directions)
# align norm an direction by the cardiac phase ED
# resample all 1D feature vectors to the same length
# this should help to validate the pre-defined rules, and if both together explains the cardiac phases

norms = []
dirs= []
cycle_len=[]
gt_ind = []
xval = np.linspace(0,1,40)

for i,n in enumerate(nda_vects[:]):
    cardiac_cycle_length = int(gt_len[i,:,0].sum())
    cycle_len.append(cardiac_cycle_length)
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
    gt_ind.append(ind)
    dir_axis=0
    gt_ed = ind[0]
    #assert cardiac_cycle_length>=gt_ed
    temp = n[:cardiac_cycle_length]
    
    
    dim = temp.shape[1:-1]
    #print(dim)
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    #print('centers: ',c.dtype)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    # direction
    directions = flow2direction_lambda(temp)[...,0]
    # direction mean
    directions = np.mean(directions,axis=(1,2,3))
    # direction ed aligned
    directions = np.roll(directions, -1*gt_ed)
    # direction interpolate to unique length
    f = interp1d(np.linspace(0,1,directions.shape[0]), directions, kind='linear')
    directions = f(xval)
    # direction min/max normalised between -1,1

    directions = minmax_lambda([directions,lower,upper])
    
    # magnitude/norm as mean
    norm_full = np.linalg.norm(temp, axis=-1)
    norm_nda = norm_full.mean(axis=(1,2,3))
    # norm ed aligned
    norm_nda = np.roll(norm_nda, -1*gt_ed)
    norm_full = np.roll(norm_full, -1*gt_ed)
    # interpolate to unique length
    f = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')
    #norm_nda = f(xval)
    norm_nda = np.interp(xval, np.linspace(0,1,norm_nda.shape[0]), norm_nda)
    #norm_full = f(norm_full)
    # norm min/max aligned to 0,1
    norm_nda = minmax_lambda([norm_nda,mid,upper])
    
    norms.append(norm_nda)
    dirs.append(directions)
# get the mean values
gt_ind = np.stack(gt_ind,axis=0)
cycle_len = np.stack(cycle_len, axis=0)
norms = np.stack(norms, axis=0)
dirs = np.stack(dirs, axis=0)
norms_m = [norms.mean(axis=0)]
dirs_m=[dirs.mean(axis=0)]
# seaborn
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax.margins(0,0)
for n in dirs:
    _ = ax.plot(n, alpha=0.5, zorder=0)

df = pd.DataFrame(dirs).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='blue', label=r'$\alpha_t$',ci='sd',err_style='bars',zorder=2,legend=False,ax=ax)
#_ = ax.set_title('Mean direction +/- SD - aligned at ED')
_ = ax.set_xlabel('Time (t) - linear interpolated to 40 frames')
_ = ax.set_ylabel(r'Angle $\alpha_t$ of $\phi_t$')
_ = ax.legend(loc='upper left')
ax2 = ax.twinx()
ax2.margins(0,0)
df = pd.DataFrame(norms).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='black', label=r'|$\phi_t$|',ci='sd', linestyle='dashed',err_style='bars',zorder=2,legend=False,ax=ax2)
_ = ax2.set_ylabel(r'Norm |$\phi_t$|')

_ = ax2.legend(loc='upper right')
# align norm an direction by the cardiac phase ED
# resample all 1D feature vectors to the same length
# this should help to validate the pre-defined rules, and if both together explains the cardiac phases

norms = []
dirs= []
cycle_len=[]
gt_ind = []
xval = np.linspace(0,1,40)

for i,n in enumerate(nda_vects[:]):
    cardiac_cycle_length = int(gt_len[i,:,0].sum())
    cycle_len.append(cardiac_cycle_length)
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
    gt_ind.append(ind)
    dir_axis=0
    gt_ed = ind[0]
    #assert cardiac_cycle_length>=gt_ed
    temp = n[:cardiac_cycle_length]
    
    
    dim = temp.shape[1:-1]
    #print(dim)
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    #print('centers: ',c.dtype)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    # direction
    directions = flow2direction_lambda(temp)[...,0]
    # direction mean
    directions = np.mean(directions,axis=(1,2,3))
    # direction ed aligned
    directions = np.roll(directions, -1*gt_ed)
    # direction interpolate to unique length
    f = interp1d(np.linspace(0,1,directions.shape[0]), directions, kind='linear')
    directions = f(xval)
    # direction min/max normalised between -1,1

    directions = minmax_lambda([directions,lower,upper])
    
    # magnitude/norm as mean
    norm_full = np.linalg.norm(temp, axis=-1)
    norm_nda = norm_full.mean(axis=(1,2,3))
    # norm ed aligned
    norm_nda = np.roll(norm_nda, -1*gt_ed)
    norm_full = np.roll(norm_full, -1*gt_ed)
    # interpolate to unique length
    f = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')
    #norm_nda = f(xval)
    norm_nda = np.interp(xval, np.linspace(0,1,norm_nda.shape[0]), norm_nda)
    norm_full = np.interp(xval, np.linspace(0,1,norm_nda.shape[0]), norm_full)
    # norm min/max aligned to 0,1
    norm_nda = minmax_lambda([norm_nda,mid,upper])
    
    norms.append(norm_nda)
    dirs.append(directions)
# align norm an direction by the cardiac phase ED
# resample all 1D feature vectors to the same length
# this should help to validate the pre-defined rules, and if both together explains the cardiac phases

norms = []
dirs= []
cycle_len=[]
gt_ind = []
xval = np.linspace(0,1,40)

for i,n in enumerate(nda_vects[:]):
    cardiac_cycle_length = int(gt_len[i,:,0].sum())
    cycle_len.append(cardiac_cycle_length)
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
    gt_ind.append(ind)
    dir_axis=0
    gt_ed = ind[0]
    #assert cardiac_cycle_length>=gt_ed
    temp = n[:cardiac_cycle_length]
    
    
    dim = temp.shape[1:-1]
    #print(dim)
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    #print('centers: ',c.dtype)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    # direction
    directions = flow2direction_lambda(temp)[...,0]
    # direction mean
    directions = np.mean(directions,axis=(1,2,3))
    # direction ed aligned
    directions = np.roll(directions, -1*gt_ed)
    # direction interpolate to unique length
    f = interp1d(np.linspace(0,1,directions.shape[0]), directions, kind='linear')
    directions = f(xval)
    # direction min/max normalised between -1,1

    directions = minmax_lambda([directions,lower,upper])
    
    # magnitude/norm as mean
    norm_full = np.linalg.norm(temp, axis=-1)
    norm_nda = norm_full.mean(axis=(1,2,3))
    # norm ed aligned
    norm_nda = np.roll(norm_nda, -1*gt_ed)
    # interpolate to unique length
    #f = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')
    #norm_nda = f(xval)
    norm_nda = np.interp(xval, np.linspace(0,1,norm_nda.shape[0]), norm_nda)
    # norm min/max aligned to 0,1
    norm_nda = minmax_lambda([norm_nda,mid,upper])
    
    norms.append(norm_nda)
    dirs.append(directions)
get_ipython().run_cell_magic('timeit', '', "f = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')\nnorm_nda = f(xval)\n")
i = 0
cardiac_cycle_length = int(gt_len[i,:,0].sum())
cycle_len.append(cardiac_cycle_length)
ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
gt_ind.append(ind)
dir_axis=0
gt_ed = ind[0]
#assert cardiac_cycle_length>=gt_ed
temp = n[:cardiac_cycle_length]
norm_full = np.linalg.norm(temp, axis=-1)
norm_nda = norm_full.mean(axis=(1,2,3))
get_ipython().run_cell_magic('timeit', '', "f = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')\nnorm_nda = f(xval)\n")
f = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')
norm_nda = f(xval)
i = 0
cardiac_cycle_length = int(gt_len[i,:,0].sum())
cycle_len.append(cardiac_cycle_length)
ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
gt_ind.append(ind)
dir_axis=0
gt_ed = ind[0]
#assert cardiac_cycle_length>=gt_ed
temp = n[:cardiac_cycle_length]
norm_full = np.linalg.norm(temp, axis=-1)
norm_nda = norm_full.mean(axis=(1,2,3))
get_ipython().run_cell_magic('time', '', "f = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')\nnorm_nda = f(xval)\n")
get_ipython().run_cell_magic('logstart', '-t', "f = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')\nnorm_nda = f(xval)\n")
get_ipython().run_line_magic('logstart', '-t')
f = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')
norm_nda = f(xval)
# Mon, 28 Feb 2022 15:25:27
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 15:25:36
get_ipython().run_line_magic('time', '')
f = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')
norm_nda = f(xval)
# Mon, 28 Feb 2022 15:25:36
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 15:25:43
i = 0
cardiac_cycle_length = int(gt_len[i,:,0].sum())
cycle_len.append(cardiac_cycle_length)
ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
gt_ind.append(ind)
dir_axis=0
gt_ed = ind[0]
#assert cardiac_cycle_length>=gt_ed
temp = n[:cardiac_cycle_length]
norm_full = np.linalg.norm(temp, axis=-1)
norm_nda = norm_full.mean(axis=(1,2,3))
# Mon, 28 Feb 2022 15:25:43
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 15:25:44
get_ipython().run_line_magic('time', '')
f = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')
norm_nda = f(xval)
# Mon, 28 Feb 2022 15:25:44
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 15:25:46
i = 0
cardiac_cycle_length = int(gt_len[i,:,0].sum())
cycle_len.append(cardiac_cycle_length)
ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
gt_ind.append(ind)
dir_axis=0
gt_ed = ind[0]
#assert cardiac_cycle_length>=gt_ed
temp = n[:cardiac_cycle_length]
norm_full = np.linalg.norm(temp, axis=-1)
norm_nda = norm_full.mean(axis=(1,2,3))
# Mon, 28 Feb 2022 15:25:46
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 15:25:52
get_ipython().run_line_magic('time', '')
norm_nda = np.interp(xval, np.linspace(0,1,norm_nda.shape[0]), norm_nda)
# Mon, 28 Feb 2022 15:25:52
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 15:26:09
get_ipython().run_line_magic('timeit', '')
f = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')
norm_nda = f(xval)
# Mon, 28 Feb 2022 15:26:09
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 15:26:17
get_ipython().run_cell_magic('timeit', '', "f = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')\nnorm_nda = f(xval)\n")
# Mon, 28 Feb 2022 15:26:20
get_ipython().run_line_magic('timeit', '')
f = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')
norm_nda = f(xval)
# Mon, 28 Feb 2022 15:26:20
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 15:26:24
get_ipython().run_line_magic('time', '')
f = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')
norm_nda = f(xval)
# Mon, 28 Feb 2022 15:26:24
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 15:26:45
get_ipython().run_cell_magic('time', '', "f = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')\nnorm_nda = f(xval)\n")
# Mon, 28 Feb 2022 15:26:45
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 15:26:48
get_ipython().run_cell_magic('time', '', 'norm_nda = np.interp(xval, np.linspace(0,1,norm_nda.shape[0]), norm_nda)\n')
# Mon, 28 Feb 2022 15:26:48
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 15:26:59
get_ipython().run_cell_magic('time', '', "norm_nda = norm_full.mean(axis=(1,2,3))\nf = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')\nnorm_nda = f(xval)\n")
# Mon, 28 Feb 2022 15:26:59
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 15:27:01
get_ipython().run_cell_magic('time', '', 'norm_nda = norm_full.mean(axis=(1,2,3))\nnorm_nda = np.interp(xval, np.linspace(0,1,norm_nda.shape[0]), norm_nda)\n')
# Mon, 28 Feb 2022 15:27:01
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:15:15
patient_paths = sorted(glob.glob('/mnt/ssd/data/acdc/orig_save/all/*/Info.cfg'))
print(len(patient_paths))
print(patient_paths[0:5])
# Mon, 28 Feb 2022 18:15:15
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:15:16
# this patient is excluded due to cut diastolic sequence
patient_paths = [p for p in patient_paths if '090' not in p]
print(len(patient_paths))
print(patient_paths[0:5])
# Mon, 28 Feb 2022 18:15:16
patients = [os.path.basename(os.path.dirname(p)) for p in patient_paths]
print(len(patients))
print(patients[0:5])
# Mon, 28 Feb 2022 18:15:16
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:15:17
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:15:18
file_4d = glob.glob(os.path.join(os.path.dirname(patient_paths[94]), '*4d.nii.gz'))
print(file_4d)
frames_total = sitk.GetArrayFromImage(sitk.ReadImage(file_4d[0])).shape[0]
print(frames_total)
# Mon, 28 Feb 2022 18:15:19
import yaml
def read_cfg_file(f):
    """Helper to open cfg files"""
    with open(f, 'r') as yml_file:
        cfg = yaml.load(yml_file, Loader=yaml.FullLoader)
    return cfg

def get_idx(cfg_f):
    patient = os.path.basename(os.path.dirname(cfg_f))
    patient_short = patient.split('patient')[1]
    cfg = read_cfg_file(cfg_f)
    file_4d = glob.glob(os.path.join(os.path.dirname(cfg_f), '*4d.nii.gz'))
    frames_total = sitk.GetArrayFromImage(sitk.ReadImage(file_4d[0])).shape[0]
    frames_total_cfg = int('{:02}'.format(cfg.get('NbFrame', 'NOPHASE')))
    if frames_total != frames_total_cfg:
        print('patient: {} total: {}, total cfg: {}'.format(patient_short, frames_total, frames_total_cfg))
    frame_ed = '{:02}'.format(cfg.get('ED', 'NOPHASE'))
    frame_es = '{:02}'.format(cfg.get('ES', 'NOPHASE'))
    return patient_short, int(frame_ed), int(frame_es), int(frames_total)
# Mon, 28 Feb 2022 18:15:19
idxs = [get_idx(c) for c in patient_paths]
# Mon, 28 Feb 2022 18:15:37
df = pd.DataFrame(idxs, columns=['patient', 'ED#', 'ES#', 'total'])
df
# Mon, 28 Feb 2022 18:15:37
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:15:39
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:15:40
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:15:41
df_phase_file = '/mnt/ssd/data/acdc/02_imported_4D_unfiltered/SAx_3D_dicomTags_phase.csv'
df_phase = pd.read_csv(df_phase_file,dtype={'patient':str, 'ED#':int, 'MS#':int, 'ES#':int, 'PF#':int, 'MD#':int})
df_phase = df_phase[['patient', 'ED#','ES#']]
df_phase
# Mon, 28 Feb 2022 18:15:41
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:15:42
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:16:41
df_phase_file = '/mnt/ssd/data/acdc/02_imported_4D_unfiltered/SAx_3D_dicomTags_phase.csv'
df_phase = pd.read_csv(df_phase_file,dtype={'patient':str, 'ED#':int, 'MS#':int, 'ES#':int, 'PF#':int, 'MD#':int})
# Mon, 28 Feb 2022 18:16:41
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:16:42
df_phase
# Mon, 28 Feb 2022 18:16:42
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:16:49
df_phase = df_phase[['ED#','MS#','ES#','PF#','MD#',]]
df_phase.idxmin(1).value_counts()
# Mon, 28 Feb 2022 18:16:49
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:16:50
print(df_phase.mean())
print(df_phase.std())
ax = df_phase.plot.box()
df_phase.plot.hist()
# Mon, 28 Feb 2022 18:16:50
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:16:59
df_phase_file = '/mnt/ssd/data/gcn/02_imported_4D_unfiltered/SAx_3D_dicomTags_phase.csv'
df_phase = pd.read_csv(df_phase_file,dtype={'patient':str, 'ED#':int, 'MS#':int, 'ES#':int, 'PF#':int, 'MD#':int})
# Mon, 28 Feb 2022 18:16:59
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:17:01
df_phase = df_phase[['ED#','MS#','ES#','PF#','MD#',]]
df_phase.idxmin(1).value_counts()/5
# Mon, 28 Feb 2022 18:17:01
print(df_phase.mean())
print(df_phase.std())
ax = df_phase.plot.box()
df_phase.plot.hist()
# Mon, 28 Feb 2022 18:17:01
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:17:02
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:33:41
df_phase.min()
# Mon, 28 Feb 2022 18:33:41
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:35:47
df_phase.max()
# Mon, 28 Feb 2022 18:35:47
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:36:22
df_phase_file = '/mnt/ssd/data/dcn/02_imported_4D_unfiltered/SAx_3D_dicomTags_phase.csv'
df_phase = pd.read_csv(df_phase_file,dtype={'patient':str, 'ED#':int, 'MS#':int, 'ES#':int, 'PF#':int, 'MD#':int})
# Mon, 28 Feb 2022 18:36:28
df_phase_file = '/mnt/ssd/data/gcn/02_imported_4D_unfiltered/SAx_3D_dicomTags_phase.csv'
df_phase = pd.read_csv(df_phase_file,dtype={'patient':str, 'ED#':int, 'MS#':int, 'ES#':int, 'PF#':int, 'MD#':int})
# Mon, 28 Feb 2022 18:36:28
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:36:29
df_phase.max()
# Mon, 28 Feb 2022 18:36:29
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:36:37
df_phase = df_phase[['ED#','MS#','ES#','PF#','MD#',]]
df_phase.idxmin(1).value_counts()
# Mon, 28 Feb 2022 18:36:37
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:36:41
print(df_phase.mean())
print(df_phase.std())
ax = df_phase.plot.box()
df_phase.plot.hist()
# Mon, 28 Feb 2022 18:36:41
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:36:57
df_phase_file = '/mnt/ssd/data/gcn/02_imported_4D_unfiltered/SAx_3D_dicomTags_phase.csv'
df_phase = pd.read_csv(df_phase_file,dtype={'patient':str, 'ED#':int, 'MS#':int, 'ES#':int, 'PF#':int, 'MD#':int})
df_phase = df_phase[['ED#','MS#','ES#','PF#','MD#',]]
# Mon, 28 Feb 2022 18:36:57
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:36:58
df_phase.max()
# Mon, 28 Feb 2022 18:36:58
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:37:03
df_phase.min()
# Mon, 28 Feb 2022 18:37:03
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:37:19
df_phase.max()
# Mon, 28 Feb 2022 18:37:19
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:37:42
df_phase_file = '/mnt/ssd/data/acdc/02_imported_4D_unfiltered/SAx_3D_dicomTags_phase.csv'
df_phase = pd.read_csv(df_phase_file,dtype={'patient':str, 'ED#':int, 'MS#':int, 'ES#':int, 'PF#':int, 'MD#':int})
df_phase = df_phase[['ED#','MS#','ES#','PF#','MD#',]]
# Mon, 28 Feb 2022 18:37:42
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:37:43
df_phase.max()
# Mon, 28 Feb 2022 18:37:43
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:37:59
df_phase.min()
# Mon, 28 Feb 2022 18:37:59
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:38:26
df_phase.max()
# Mon, 28 Feb 2022 18:38:26
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:39:02
df_phase_file = '/mnt/ssd/data/gcn/02_imported_4D_unfiltered/SAx_3D_dicomTags_phase.csv'
df_phase = pd.read_csv(df_phase_file,dtype={'patient':str, 'ED#':int, 'MS#':int, 'ES#':int, 'PF#':int, 'MD#':int})
df_phase = df_phase[['ED#','MS#','ES#','PF#','MD#',]]
# Mon, 28 Feb 2022 18:39:02
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:39:04
df_phase.max()
# Mon, 28 Feb 2022 18:39:04
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:39:07
df_phase.min()
# Mon, 28 Feb 2022 18:39:07
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:39:16
df_phase.max()
# Mon, 28 Feb 2022 18:39:16
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:39:53
df_phase_file = '/mnt/ssd/data/gcn/02_imported_4D_unfiltered/SAx_3D_dicomTags_phase.csv'
df_phase = pd.read_csv(df_phase_file,dtype={'patient':str, 'ED#':int, 'MS#':int, 'ES#':int, 'PF#':int, 'MD#':int})
df_phase = df_phase[['ED#','MS#','ES#','PF#','MD#']]
# Mon, 28 Feb 2022 18:39:53
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:39:54
df_phase.max()
# Mon, 28 Feb 2022 18:39:54
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:39:55
df_phase.idxmin(1).value_counts()
# Mon, 28 Feb 2022 18:39:55
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:40:00
print(df_phase.mean())
print(df_phase.std())
ax = df_phase.plot.box()
df_phase.plot.hist()
# Mon, 28 Feb 2022 18:40:00
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:48:48
# TOF unsupervised center 14_41 gaus 2
sb.set_context('paper')
sb.set(font_scale = 2)
_ =df.plot(kind='box')
sb.violinplot(data=df)
pd.options.display.float_format = "{:,.2f}".format
df_summarized = pd.concat([df.mean(axis=0),df.std(axis=0), df.median(axis=0)], axis=1)
df_summarized.columns = ['mean', 'SD', 'meadian']
print(df_summarized.mean())
print(df_summarized)

sb.set_context('paper')
sb.set(font_scale = 1.8)
fig, axes = plt.subplots(1,5,figsize=(25,4))
axes = axes.flatten()
for i,ax in enumerate(axes):
    _ = ax.scatter(upred_ind[:,i],gt_ind[:,i])
    _ = ax.axline((1, 1), slope=1)
    _ = ax.set_title(phases[i])
    _ = ax.set_xlabel('Index - pred')
    _ = ax.set_ylabel('Index - gt')
    _ = ax.set_xlim([0,35])
    _ = ax.set_ylim([0,35])
plt.tight_layout()
# Mon, 28 Feb 2022 18:49:16
# load the vectors
pathstovectnpy = sorted(glob.glob(os.path.join(vects_chooser.selected, 'moved', '*vects*.npy')))
print(pathstovectnpy)
nda_vects = np.concatenate([np.load(path_) for path_ in pathstovectnpy],axis=0)
print(nda_vects.shape)
# Mon, 28 Feb 2022 18:49:26
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:49:27
# load the phase gt and pred
pred_path = os.path.join(vects_chooser.selected, 'pred')
pathtsophasenpy = sorted(glob.glob(os.path.join(pred_path, '*gtpred*.npy')))
print(pathtsophasenpy)
nda_phase = np.concatenate([np.load(path_) for path_ in pathtsophasenpy],axis=1)
print(nda_phase.shape)
gt_, pred_ = np.split(nda_phase, axis=0,indices_or_sections=2)
print(gt_.shape)
gt= gt_[0,:,0]
pred = pred_[0,:,0]
print(gt.shape)
gt_len = gt_[0,:,1]
# Mon, 28 Feb 2022 18:49:27
# load some moved examples for easier understanding of the dimensions
pathtomoved = sorted(glob.glob(os.path.join(vects_chooser.selected, 'moved','*moved*.npy')))
print(len(pathtomoved))
mov = np.concatenate([np.load(path_) for path_ in pathtomoved],axis=0)
print(mov.shape) # patient,time,spatial-z,y,x,channel, e.g.: 69,40,16,64,64,1
# Mon, 28 Feb 2022 18:49:27
# plot one moved timestep of one patient = one 3D volume 
# should go from apex to base
from src.visualization.Visualize import show_2D_or_3D
temp = mov[10,0,...,0]
_ = show_2D_or_3D(temp, cmap='gray', interpolation='none')
# Mon, 28 Feb 2022 18:49:28
import tensorflow as tf
from tensorflow.image import ssim
ssim(mov[10,0], mov[10,1],max_val=1.0, filter_size=11,
                          filter_sigma=1.5, k1=0.01, k2=0.03).numpy().mean()
# Mon, 28 Feb 2022 18:49:28
from skimage.metrics import structural_similarity as ssim
ssim(mov[10,0], mov[10,1],multichannel=True)
# Mon, 28 Feb 2022 18:49:28
plt.imshow(gt[0].T)
# Mon, 28 Feb 2022 18:49:28
plt.imshow(pred[0].T)
# Mon, 28 Feb 2022 18:49:28
import tensorflow as tf
import sys
# returns a matrix with the indicies as values, similar to np.indicies
def get_idxs_tf(x):
    return tf.cast(
        tf.reshape(tf.where(tf.ones((x[0], x[1], x[2]))), (x[0], x[1], x[2], 3)),
        tf.float32)


# returns a matrix with vectors pointing to the center
def get_centers_tf(x):
    return tf.cast(
        tf.tile(tf.convert_to_tensor([x[0] // 2, x[1] // 2, x[2] // 2])[tf.newaxis, tf.newaxis, tf.newaxis, ...],
                (x[0], x[1], x[2], 1)), tf.float32)

def get_angle_tf(a, b):
    # this should work for batches of n-dimensional vectors
    # α = arccos[(a · b) / (|a| * |b|)]
    # |v| = √(x² + y² + z²)
    """
    in 3D space
    If vectors a = [xa, ya, za], b = [xb, yb, zb], then:
    α = arccos[(xa * xb + ya * yb + za * zb) / (√(xa2 + ya2 + za2) * √(xb2 + yb2 + zb2))]
    """
    import math as m
    pi = tf.constant(m.pi)
    b = tf.cast(b, dtype=a.dtype)
    inner = tf.einsum('...i,...i->...', a, b)
    norms = tf.norm(a, ord='euclidean', axis=-1) * tf.norm(b, ord='euclidean', axis=-1)  # [...,None]
    cos = inner / (norms + sys.float_info.epsilon)
    #rad = tf.math.acos(tf.clip_by_value(cos, -1.0, 1.0))
    # rad2deg conversion
    #deg = rad * (180.0/pi)
    return cos[..., tf.newaxis]

dim = [16,64,64]
idx = get_idxs_tf(dim)
c = get_centers_tf(dim)
centers = c - idx
centers_tensor = centers[tf.newaxis, ...]
flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x, centers_tensor), name='flow2direction')
minmax_lambda = lambda x : x[1] + (((x[0] - np.min(x[0])) * (x[2]-x[1]))/(np.max(x[0]) - np.min(x[0])))
# define some central params
lower, mid, upper = -1,0, 1
# Mon, 28 Feb 2022 18:49:28
# align norm an direction by the cardiac phase ED
# resample all 1D feature vectors to the same length
# this should help to validate the pre-defined rules, and if both together explains the cardiac phases

norms = []
dirs= []
cycle_len=[]
gt_ind = []
xval = np.linspace(0,1,40)

for i,n in enumerate(nda_vects[:]):
    cardiac_cycle_length = int(gt_len[i,:,0].sum())
    cycle_len.append(cardiac_cycle_length)
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
    gt_ind.append(ind)
    dir_axis=0
    gt_ed = ind[0]
    #assert cardiac_cycle_length>=gt_ed
    temp = n[:cardiac_cycle_length]
    
    
    dim = temp.shape[1:-1]
    #print(dim)
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    #print('centers: ',c.dtype)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    # direction
    directions = flow2direction_lambda(temp)[...,0]
    # direction mean
    directions = np.mean(directions,axis=(1,2,3))
    # direction ed aligned
    directions = np.roll(directions, -1*gt_ed)
    # direction interpolate to unique length
    f = interp1d(np.linspace(0,1,directions.shape[0]), directions, kind='linear')
    directions = f(xval)
    # direction min/max normalised between -1,1

    directions = minmax_lambda([directions,lower,upper])
    
    # magnitude/norm as mean
    norm_full = np.linalg.norm(temp, axis=-1)
    norm_nda = norm_full.mean(axis=(1,2,3))
    # norm ed aligned
    norm_nda = np.roll(norm_nda, -1*gt_ed)
    # interpolate to unique length
    #f = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')
    #norm_nda = f(xval)
    norm_nda = np.interp(xval, np.linspace(0,1,norm_nda.shape[0]), norm_nda)
    # norm min/max aligned to 0,1
    norm_nda = minmax_lambda([norm_nda,mid,upper])
    
    norms.append(norm_nda)
    dirs.append(directions)
# Mon, 28 Feb 2022 18:49:40
# get the mean values
gt_ind = np.stack(gt_ind,axis=0)
cycle_len = np.stack(cycle_len, axis=0)
norms = np.stack(norms, axis=0)
dirs = np.stack(dirs, axis=0)
norms_m = [norms.mean(axis=0)]
dirs_m=[dirs.mean(axis=0)]
# Mon, 28 Feb 2022 18:49:40
i = 0
cardiac_cycle_length = int(gt_len[i,:,0].sum())
cycle_len.append(cardiac_cycle_length)
ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
gt_ind.append(ind)
dir_axis=0
gt_ed = ind[0]
#assert cardiac_cycle_length>=gt_ed
temp = n[:cardiac_cycle_length]
norm_full = np.linalg.norm(temp, axis=-1)
norm_nda = norm_full.mean(axis=(1,2,3))
# Mon, 28 Feb 2022 18:49:40
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:54:45
# ACDC unsupervised center
sb.set_context('paper')
fig, axes = plt.subplots(1,5,figsize=(25,4))
axes = axes.flatten()
for i,ax in enumerate(axes):
    _ = ax.scatter(upred_ind[:,i],gt_ind[:,i])
    _ = ax.axline((1, 1), slope=1)
    _ = ax.set_title(phases[i])
    _ = ax.set_xlabel('unsupervised')
    _ = ax.set_ylabel('gt')
    _ = ax.set_xlim([0,35])
    _ = ax.set_ylim([0,35])
    
# Mon, 28 Feb 2022 18:54:49
# TOF unsupervised center 14_41 gaus 2
sb.set_context('paper')
sb.set(font_scale = 2)
_ =df.plot(kind='box')
sb.violinplot(data=df)
pd.options.display.float_format = "{:,.2f}".format
df_summarized = pd.concat([df.mean(axis=0),df.std(axis=0), df.median(axis=0)], axis=1)
df_summarized.columns = ['mean', 'SD', 'meadian']
print(df_summarized.mean())
print(df_summarized)

sb.set_context('paper')
sb.set(font_scale = 1.8)
fig, axes = plt.subplots(1,5,figsize=(25,4))
axes = axes.flatten()
for i,ax in enumerate(axes):
    _ = ax.scatter(upred_ind[:,i],gt_ind[:,i])
    _ = ax.axline((1, 1), slope=1)
    _ = ax.set_title(phases[i])
    _ = ax.set_xlabel('Index - pred')
    _ = ax.set_ylabel('Index - gt')
    _ = ax.set_xlim([0,35])
    _ = ax.set_ylim([0,35])
plt.tight_layout()
# Mon, 28 Feb 2022 18:55:56
"""i = 0
cardiac_cycle_length = int(gt_len[i,:,0].sum())
cycle_len.append(cardiac_cycle_length)
ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
gt_ind.append(ind)
dir_axis=0
gt_ed = ind[0]
#assert cardiac_cycle_length>=gt_ed
temp = n[:cardiac_cycle_length]
norm_full = np.linalg.norm(temp, axis=-1)
norm_nda = norm_full.mean(axis=(1,2,3))"""
# Mon, 28 Feb 2022 18:55:56
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:55:57
get_ipython().run_cell_magic('time', '', "norm_nda = norm_full.mean(axis=(1,2,3))\nf = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')\nnorm_nda = f(xval)\n")
# Mon, 28 Feb 2022 18:55:57
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:55:58
get_ipython().run_cell_magic('time', '', 'norm_nda = norm_full.mean(axis=(1,2,3))\nnorm_nda = np.interp(xval, np.linspace(0,1,norm_nda.shape[0]), norm_nda)\n')
# Mon, 28 Feb 2022 18:55:58
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:56:00
# seaborn
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax.margins(0,0)
for n in dirs:
    _ = ax.plot(n, alpha=0.5, zorder=0)

df = pd.DataFrame(dirs).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='blue', label=r'$\alpha_t$',ci='sd',err_style='bars',zorder=2,legend=False,ax=ax)
#_ = ax.set_title('Mean direction +/- SD - aligned at ED')
_ = ax.set_xlabel('Time (t) - linear interpolated to 40 frames')
_ = ax.set_ylabel(r'Angle $\alpha_t$ of $\phi_t$')
_ = ax.legend(loc='upper left')
ax2 = ax.twinx()
ax2.margins(0,0)
df = pd.DataFrame(norms).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='black', label=r'|$\phi_t$|',ci='sd', linestyle='dashed',err_style='bars',zorder=2,legend=False,ax=ax2)
_ = ax2.set_ylabel(r'Norm |$\phi_t$|')

_ = ax2.legend(loc='upper right')
# Mon, 28 Feb 2022 18:56:00
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:56:01
folds_chooser = FileChooser(os.path.join(os.getcwd(),'/mnt/ssd/git/dynamic-cmr-models/exp/phasereg_v3/acdc/'), '')
display(folds_chooser)
# Mon, 28 Feb 2022 18:56:01
# define df and helper for pathology extractio
# this should have the same order as our inference data
#df_meta = pd.read_csv(folds_chooser.selected)
df_meta = pd.read_csv('/mnt/sds/sd20i001/sven/data/acdc/02_imported_4D_unfiltered/df_kfold.csv')
df_meta = df_meta.loc[:, ~df_meta.columns.str.contains('^Unnamed')]
df_meta = df_meta[df_meta.patient != 'patient090']  #we excluded this patient

def get_msk_for_pathology(df_, pathology='minf'):
    msk = []
    for f in [0,1,2,3,4]:
        patients = df_[df_.fold.isin([f])]
        patients = patients[patients['phase']=='ED']
        pat = patients[patients['modality'] == 'test']['patient'].str.lower().unique()
        sub_df = patients[patients.patient.isin(pat)].drop_duplicates(ignore_index=True, subset='patient')
        sub_df = sub_df.drop('x_path',axis=1).drop('y_path',axis=1)
        msk.append(sub_df['pathology'].str.lower()==pathology.lower())
    return np.concatenate(msk)
df_meta
# Mon, 28 Feb 2022 18:56:01
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:56:02
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:56:03
# interactive plot per pathology
@interact
def plot_per_pathology(p=df_meta.pathology.unique()):
    print(p)
    msk = get_msk_for_pathology(df_meta, p)

    import seaborn as sb
    sb.set_context('paper')
    sb.set(font_scale = 2)
    fig, ax = plt.subplots(figsize=(25,5))
    ax.margins(0,0)
    dirs_minf = dirs[msk]
    norms_minf = norms[msk]
    for n in dirs_minf:
        _ = ax.plot(n, alpha=0.5, zorder=0)

    df = pd.DataFrame(dirs_minf).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, color='blue', label='d_t',ci='sd',err_style='bars',zorder=2,legend=False,ax=ax)
    #_ = ax.set_title('Mean direction +/- SD - aligned at ED')
    _ = ax.set_xlabel('Time - linear interpolated to 40 frames')
    _ = ax.set_ylabel('direction d_t')
    _ = ax.legend(loc='upper left')
    ax2 = ax.twinx()
    ax2.margins(0,0)
    df = pd.DataFrame(norms_minf).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, color='black', label='|v_t|',ci='sd', linestyle='dashed',err_style='bars',zorder=2,legend=False,ax=ax2)
    _ = ax2.set_ylabel('norm |v_t|')

    _ = ax2.legend(loc='upper right')
    plt.show()
# Mon, 28 Feb 2022 18:56:03
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax2 = ax.twinx()
ax2.margins(0,0)
ax.margins(0,0)
for p in df_meta.pathology.unique():
    msk = get_msk_for_pathology(df_meta, p)
    dirs_minf = dirs[msk]
    norms_minf = norms[msk]
    #for n in dirs_minf:
    #    _ = ax.plot(n, alpha=0.5, zorder=0)

    df = pd.DataFrame(dirs_minf).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, label='d_t_{}'.format(p),ci='sd',err_style=None,zorder=2,legend=False,ax=ax)
    #_ = ax.set_title('Mean direction +/- SD - aligned at ED')
    _ = ax.set_xlabel('Time - linear interpolated to 40 frames')
    _ = ax.set_ylabel('direction d_t')
    _ = ax.legend(loc='upper left')
    
    df = pd.DataFrame(norms_minf).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, label='|v_t|_{}'.format(p),ci='sd', linestyle='dashed',err_style=None,zorder=2,legend=False,ax=ax2)
    _ = ax2.set_ylabel('norm |v_t|')

    _ = ax2.legend(loc='upper right')
plt.show()
# Mon, 28 Feb 2022 18:56:04
import sklearn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
y = np.zeros(99)
for i,p in enumerate(df_meta.pathology.unique()):
    msk = get_msk_for_pathology(df_meta, p)
    y[msk] = i
low_dim = TSNE(n_components=2).fit_transform(norms)
#low_dim = PCA(n_components=2).fit_transform(norms)
_ =plt.scatter(low_dim[:,0],low_dim[:,1], c=y)
# Mon, 28 Feb 2022 18:56:04
# plot one line per patient - norm
fig, ax = plt.subplots(figsize=(25,5))
for n in norms:
    _ = plt.plot(n)
_ = ax.set_title('Magnitudes aligned at ED, resampling shape 40,')
# Mon, 28 Feb 2022 18:56:04
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:56:05
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:56:06
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:56:07
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:56:08
# plot one line per direction
fig, ax = plt.subplots(figsize=(25,5))
for n in dirs:
    _ = plt.plot(n)
_ = ax.set_title('Directions aligned at ED, resampling shape 40,')
# Mon, 28 Feb 2022 18:56:08
from src.visualization.Visualize import show_2D_or_3D
from src.data.Preprocess import clip_quantile
nda_temp = nda_vects[0]
# norm_1 = nda_temp[:,:,:32,:32]# top left
norm_1 = nda_temp[:,:,:32,32:]# top right
norm_1 = np.linalg.norm(norm_1, axis=-1)
nda_norm_ = np.linalg.norm(nda_temp, axis=-1)
plt.hist(nda_norm_.flatten());plt.show()
nda_norm = clip_quantile(nda_norm_, 0.99)
nda_norm = minmax_lambda([nda_norm,0,1])
plt.hist(nda_norm.flatten())
plt.show()
# mask phi with a threshold norm matrix 
nda_msk = (nda_norm>=0.2).astype(np.float32)
nda_temp = nda_norm * nda_msk
#plt.hist(nda_temp.flatten())
print(nda_temp.shape)
_ = show_2D_or_3D(nda_norm_[:,0], allow_slicing=True)
_ = show_2D_or_3D(nda_temp[:,0], allow_slicing=True)
_ = show_2D_or_3D(nda_temp[:,-1], allow_slicing=True)

# top right
_ = show_2D_or_3D(norm_1[:,0], allow_slicing=True,cmap='gray', interpolation='none')
# Mon, 28 Feb 2022 18:56:10
fig, ax = plt.subplots(figsize=(25,3))
plt.plot(np.argmax(gt[0], axis=1))
ax.set_yticks([0, 1, 2, 3, 4], minor=False)
phases = ['ED', 'MS', 'ES', 'PF', 'MD']
_ = ax.set_yticklabels(phases)
# Mon, 28 Feb 2022 18:56:11
# plot the mean/max norm for one patient oveer time
nda_1d_max = np.max(nda_temp,axis=(1,2,3))
nda_1d_mean = np.mean(nda_temp,axis=(1,2,3))
nda_1d_sum = np.sum(nda_temp,axis=(1,2,3))

fig, ax = plt.subplots(figsize=(25,3))
ax.set_title('max norm')
_ = plt.plot(nda_1d_max); plt.show()

fig, ax = plt.subplots(figsize=(25,3))
ax.set_title('mean norm')
_ = plt.plot(nda_1d_mean)

fig, ax = plt.subplots(figsize=(25,3))
ax.set_title('mean sum')
_ = plt.plot(nda_1d_sum)
#ax.set_ylim(0.0,0.15)
# Mon, 28 Feb 2022 18:56:11
temp = np.arange(10)
temp[:0]
# Mon, 28 Feb 2022 18:56:11
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:56:12
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:56:13
def detect_phases(dir_1d_mean, length):
    
    import scipy.signal as sig
    
        # MS
    # Global min of f(x)
    ms = np.argmin(dir_1d_mean)
    ms = ms -1 # take the bucket before the first min peak
    
    
    # ES
    # First time f(x)>0 after MS
    cycle = np.concatenate([dir_1d_mean[ms:], dir_1d_mean[:ms]])
    temp_ = 0
    es_found=False
    negative_slope = False
    for idx,elem in enumerate(cycle):
        if elem<0:
            negative_slope=True
            temp_ = idx
        elif elem>=0 and negative_slope:
            es_found = True
            #temp_ = idx
            break # stop after first zero-transition
    if es_found:
        es = ms + temp_
        #es = es-1
    else:
        es = ms + 1 # the frame after ms, fallback
    if es>=length:
        es = np.mod(es,length)
        print('ES overflow: {}, ms:{}'.format(es,ms))
     
    
    # PF
    # First peak after ES, min height 0.6
    seq = dir_1d_mean[es:]
    peaks, prop = sig.find_peaks(seq, height=0.6)#height=0.6 we normalise between -1 and 1, PF should be close to argmax

    if len(peaks>0):
        pf = es + peaks[0] # take the peak after es
        pf = pf -1
    else:
        print('pf not clear, set to ES {} + 1'.format(es))
        pf = es + 1
    pf = np.mod(pf, length)
      

    # ED
    # Between pf and ms: last time f(x) cross zero from positive to negative
    # a priori knowledge ED needs a minimal distance of 2 frames towards MS
    # CHANGED the minimal distance between ED and MS
    cycle = np.concatenate([dir_1d_mean[pf:], dir_1d_mean[:ms]])
    #print(cycle)
    ed_found = False
    last_idx_positive = True # we start at the pf, which is the peak(dir)
    for idx,elem in enumerate(cycle):
        
        if elem>=0:
            last_idx_positive = True
        # remember the last idx before the direction gets negative the last time before ms
        elif elem<0 and last_idx_positive: # first time direction negative
            ed_found = True # for fallbacks
            temp_ = idx # idx before negative direction
            #print('found transition at: {}'.format(idx))
            last_idx_positive = False # remember only the first idx after transition
        
    if ed_found:
        ed = pf + temp_
        #print('ed:{}, pf:{}, temp_:{}, lenght: {}'.format(ed,pf,temp_,length))
    else: 
        # if we dont find a transition from positive to negative, take the idx which is the closest to zero
        temp_ = np.argmin(np.abs(cycle)) # make sure we have a minimal distance
        ed = pf + temp_
        print('ED: no transition found between {}-{} , closest id to 0: {}, ed = {}'.format(pf,ms,temp_,ed))
    
    if ed>=length:
        #print('gt ed: {}, ed: {}, length: {}'.format(gted,ed,length))
        ed = np.mod(ed,length)
        #print('mod ed: {}'.format(ed))
    #ed = ed-1 # take the bucket before negative
        
    # MD
    # Middle between PF and ED
    ed_slice_idx = ed
    if ed_slice_idx<=pf: # ed overflow --> beginning of cmr stack
        ed_slice_idx = length + ed
    md = (pf+ed_slice_idx)//2 # the bucket after the middle
    md = md + 1
    md = np.mod(md,length)
    
    return np.array([ed,ms,es,pf,md])
# Mon, 28 Feb 2022 18:56:13
# try to predict the phases unsupervised just from the norm/direction curve

def get_phases_from_vects(vects_nda, length=-1, plot=True, dir_axis=0, gtind=[], figsize=(25,3)):
    import scipy.signal
    import scipy.signal as sig
    import scipy.ndimage
    from scipy.ndimage import gaussian_filter1d
    
    gted=gtind[0]
    
    # signal smoothing
    kernel_size = 4
    kernel = np.ones(kernel_size) / kernel_size
    z = 0
    
    
    dim_z = vects_nda.shape[1]
    #vects_nda = vects_nda[:int(length)]
    
    
    # define some helper lambdas
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    dim = vects_nda.shape[1:-1]
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]

    
    # norm of the vector
    norm_nda = np.linalg.norm(vects_nda[...,dir_axis:], axis=-1)
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    norm_nda_thres = clip_quantile(norm_nda, 0.9)
    norm_nda = minmax_lambda([norm_nda,0,1])
    norm_msk = norm_nda>=0.0
    norm_nda = norm_nda * norm_msk
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    nda_1d_mean = np.mean(norm_nda,axis=(1,2,3))
    nda_1d_max = np.max(norm_nda,axis=(1,2,3))
    nda_1d_median = np.median(norm_nda,axis=(1,2,3))
    nda_1d_mean_smooth = np.convolve(nda_1d_mean, kernel, mode='same')
    #print(nda_1d_mean_smooth.shape, nda_1d_mean.shape)
    
    
    # direction relative to the center
    
    directions = flow2direction_lambda(vects_nda)[...,0].numpy()
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    directions = minmax_lambda([directions,lower,upper])
    #plt.hist(directions.flatten());plt.show()
    directions = directions * norm_msk
    #plt.hist(directions.flatten());plt.show()
    
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    dir_1d_mean = np.mean(directions,axis=(1,2,3))
    dir_1d_median = np.median(directions,axis=(1,2,3))

    
    # Find phases
    # 1st smooth the direction with a rolling averge (kernelsize=4)
    # or via gaussian filter
    # 2nd min/max normalise each direction vector in a range [-1:1]
    
    
    dir_1d_mean = gaussian_filter1d(dir_1d_mean, sigma=2, mode='wrap') # original: dir_1d_mean
    
    ################# CHANGED !!!!!! ######################## RECENTLY NO CHANGES
    # min/max scaling
    dir_1d_mean = minmax_lambda([dir_1d_mean,lower,upper])
    
    
#     # test with corner specific vector fields
#     y,x = directions.shape[-2:]
    
#     vects_mean = minmax_lambda([np.mean(vects_nda,axis=(1,2,3,4)),lower,upper])
#     vects_1 = vects_nda[:,:,:y//2,:x//2,:]
#     vects_1_mean = minmax_lambda([np.mean(vects_1,axis=(1,2,3,4)),lower,upper])
#     vects_2 = vects_nda[:,:,y//2:,:x//2,:]
#     vects_2_mean = minmax_lambda([np.mean(vects_2,axis=(1,2,3,4)),lower,upper])
#     vects_3 = vects_nda[:,:,y//2:,x//2:,:]
#     vects_3_mean = minmax_lambda([np.mean(vects_3,axis=(1,2,3,4)),lower,upper])
#     vects_4 = vects_nda[:,:,:y//2,x//2:,:]
#     vects_4_mean = minmax_lambda([np.mean(vects_4,axis=(1,2,3,4)),lower,upper])
    
    
#     # direction per corner
#     dir_1 = directions[:,:,:y//2,:x//2] # top left, plot
#     dir_1_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_2 = directions[:,:,y//2:,:x//2] # top right, plot
#     dir_2_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_3 = directions[:,:,y//2:,x//2:] # bottom left
#     dir_3_mean = minmax_lambda([np.mean(dir_3,axis=(1,2,3)),lower,upper])
#     dir_4 = directions[:,:,:y//2,x//2:] # bottom right
#     dir_4_mean = minmax_lambda([np.mean(dir_4,axis=(1,2,3)),lower,upper])
    

#     # norm per corner
#     norm_mean = minmax_lambda([nda_1d_mean, mid, upper])
#     norm_1 = norm_nda[:,:,:y//2,:x//2]
#     norm_1_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_2 = norm_nda[:,:,y//2:,:x//2]
#     norm_2_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_3 = norm_nda[:,:,y//2:,x//2:]
#     norm_3_mean = minmax_lambda([np.mean(norm_3,axis=(1,2,3)),mid,upper])
#     norm_4 = norm_nda[:,:,:y//2,x//2:]
#     norm_4_mean = minmax_lambda([np.mean(norm_4,axis=(1,2,3)),mid,upper])
    

    ind = detect_phases(dir_1d_mean=dir_1d_mean[:length], length=length)
    
    
    #print('ed: {}, ms: {}, es: {}, pf: {}, md: {}'.format(*ind))
    
        # plot the mean/max norm for one patient oveer time
    if plot:
        
        
        # VECT 2D+t
        vect_2d_t = vects_nda[:,z]
        fig1 = show_2D_or_3D(vect_2d_t,allow_slicing=False, f_size=(25,2))
        ax1 = fig1.get_axes()[0]
        _ = ax1.set_ylabel('phi_t \nmid-cavity')

        
        # DIR 2D+t
        dir_2d_t = directions[:,z]
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        #dir_2d_t = minmax_lambda([dir_2d_t,lower,upper])
        div_cmap = 'bwr'
        fig = show_2D_or_3D(dir_2d_t,allow_slicing=False, f_size=(25,2),cmap=div_cmap)
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('alpha_t \nmid-cavity')
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[len(ax.get_images())//2], cax=cax, orientation='horizontal')

        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # DIR 2D T x Z
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        directions_tz = directions.mean(axis=(2,3)) # original: norm()
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        _ = ax.imshow(directions_tz.T, aspect='auto', label='aasd', cmap=div_cmap, origin='lower')
        _ = ax.set_xticks(gtind, minor=False)
        ax2 = ax.twinx()
        _ = ax2.plot(dir_1d_mean, c='black',label='dir 1d+t')
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('dir z+t\napex:base')
        
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1
        
        #ax=fig.add_subplot(rows,1,pos)
        #_ = ax.plot(dir_1d_mean, c='black',label='dir 1d+t')
        #_ = ax.plot(dir_1_mean, label='top left')
        #_ = ax.plot(dir_2_mean, label='top right')
        #_ = ax.plot(dir_3_mean, label='bottom left')
        #_ = ax.plot(dir_4_mean, label='bottom right')
        #ax.legend(loc='upper left')
        #ax.set_xticks(gtind, minor=False)
        #ax.margins(0.022,0)
        #ax.label_outer()
        #pos=pos+1
        #plt.subplots_adjust(wspace=0, hspace=0.1)
        #plt.show()
        norm_cmap = 'hot'
        
        # NORM 2D + t
        norm_2d_t = norm_nda[:,z]
        norm_2d_t = minmax_lambda([norm_2d_t,mid,upper])
        fig = show_2D_or_3D(norm_2d_t,allow_slicing=False, f_size=(25,2),cmap=norm_cmap, interpolation='none')
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('norm 2d+t\nmid-cavity')
        #ax.margins(0,0)
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[0], cax=cax, orientation='horizontal')
        
        
        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # NORM 2D TxZ
        norm2d = minmax_lambda([norm_nda.mean(axis=(2,3)),mid,upper])
        _ = ax.imshow(norm2d.T, aspect='auto', origin='lower',cmap=norm_cmap, interpolation='none')
        _ = ax.set_xticks(gtind, minor=False)
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('norm z+t\napex:base')
        ax2 = ax.twinx()
        _ = ax2.plot(nda_1d_mean, c='black',label='norm 1d+t')
        #ax.spines['top'].set_visible(False)
        #_ = ax.legend(loc='upper left')
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1

        """ax=fig.add_subplot(rows,1,pos)
        _ = ax.plot(norm_mean, label='norm 1d+t')
        #_ = ax.plot(norm_1_mean, label='top left')
        #_ = ax.plot(norm_2_mean, label='top right')
        #_ = ax.plot(norm_3_mean, label='bottom left')
        #_ = ax.plot(norm_4_mean, label='bottom right')
        ax.legend(loc='upper left')
        ax.set_xticks(gtind, minor=False)
        ax.margins(0.022,0)
        ax.label_outer()
        pos = pos+1
        plt.subplots_adjust(wspace=0, hspace=0.1)"""
        #plt.tight_layout()
        plt.show()

        
    
    return ind
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
p = 2#215
gt_length = int(gt_len[p,:,0].sum())
ind = np.argmax(gt[p][:gt_length],axis=0)
get_phases_from_vects(nda_vects[p][:gt_length], length=gt_length,plot=True,dir_axis=0, gtind=ind)
# Mon, 28 Feb 2022 18:56:16
@interact
def compare_phases(i=(0,nda_vects.shape[0]-1), plot=True):
    import matplotlib as mpl
    mpl.rcParams.update(mpl.rcParamsDefault)
    figsize = (25,1)
    weight = 1
    z = 0
    
    phases = ['ED', 'MS', 'ES', 'PF', 'MD']
    cardiac_cycle_length = int(gt_len[i,:,-1].sum())
    # crop or not
    cardiac_cycle_length_ = gt[i].shape[0]-3 # plot the full length minus border for visualisation
    
    vects = nda_vects[i][:cardiac_cycle_length]
    
    
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0) # 
    ind_pred = np.argmax(pred[i][:],axis=0)
    
    temp = mov[i,:cardiac_cycle_length,z] # cardiac_cycle_length_
    fig = show_2D_or_3D(temp,allow_slicing=False)
    ax = fig.get_axes()[0]
    _ = ax.set_ylabel('CMR 2d+t)\nmid-cavity')
    plt.show()
    
    fig= plt.figure(figsize=figsize)
    ax=fig.add_subplot(111)
    ax.set_xticks(ind, minor=False)
    ax.set_yticks([0, 1, 2, 3, 4], minor=False)
    ax.set_yticklabels(phases, rotation=0)
    ind = np.array(ind)
    onehot_gt = np.zeros((ind.size, cardiac_cycle_length)) # cardiac_cycle_length_
    onehot_gt[np.arange(ind.size), ind] = weight
    #ax.imshow(gt[i][:cardiac_cycle_length].T,aspect='auto') # with uncertainity
    ax.imshow(onehot_gt,aspect='auto', cmap='bone') # gt as binary phase2time mapping
    _ = ax.set_yticklabels(phases)
    ax.set_title('gt')
    ax.margins(0,0)
    
    indices = get_phases_from_vects(vects, length=cardiac_cycle_length, plot=plot,dir_axis=0, gtind=ind, figsize=figsize)

    indices = np.array(indices)
    onehot = np.zeros((indices.size, cardiac_cycle_length)) # cardiac_cycle_length_
    onehot[np.arange(indices.size), indices] = weight
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xticks(indices, minor=False)
    ax.set_yticks([0, 1, 2, 3, 4], minor=False)
    ax.set_yticklabels(phases, rotation=0)
    plt.imshow(onehot,aspect='auto', cmap='bone')
    _ = ax.set_yticklabels(phases)
    ax.margins(0,0)
    #ax.set_title('prediction')
    plt.show()
    
    print(phases)
    print('gt:', ind)
    print('u: ', indices)
    
    
    # this would plot the supervised prediction
    #print('p: ', ind_pred)
    """fig, ax = plt.subplots(figsize=figsize)
    ax.set_xticks(ind_pred, minor=False)
    ax.set_yticks([0, 1, 2, 3, 4], minor=False)
    ax.set_yticklabels(phases, rotation=45)
    plt.imshow(pred[i][:cardiac_cycle_length].T,aspect='auto')
    _ = ax.set_yticklabels(phases);plt.show()"""
    # patient 107 gcn --> good curve for plotting
    # 200 no cut
    # 125 no cut
    # 194 moderate cut 
    # 134 strong cut-off
# Mon, 28 Feb 2022 18:56:19
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:56:20
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:56:20
pred_u = np.zeros_like(gt)
upred_ind = []
cycle_len=[]
print(pred_u.shape)
for i in range(pred_u.shape[0]):
    weight = 1
    
    cardiac_cycle_length = int(gt_len[i,:,0].sum())
    cycle_len.append(cardiac_cycle_length)
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
    #print(gt[i][:cardiac_cycle_length].T.shape)
    
    indices = get_phases_from_vects(nda_vects[i][:cardiac_cycle_length], length=cardiac_cycle_length, gtind=ind,plot=False,dir_axis=0)
    upred_ind.append(indices)
    indices = np.array(indices)
    onehot = np.zeros((indices.size, cardiac_cycle_length))
    onehot[np.arange(indices.size), indices] = weight
    pred_u[i][0:cardiac_cycle_length] = onehot.T
upred_ind=np.stack(upred_ind, axis=0)
cycle_len = np.stack(cycle_len, axis=0)
# re-create a compatible shape for the metric fn
gt_ = np.stack([gt,gt_len], axis=1)
pred_ = np.stack([pred_u,np.zeros_like(pred_u)], axis=1)

# create a dataframe for further plots
from src.utils.Metrics import meandiff
phases = ['ED', 'MS', 'ES', 'PF', 'MD']
res = meandiff(gt_, pred_, apply_sum=False, apply_average=False)
df = pd.DataFrame(res.numpy(), columns=phases)
# Mon, 28 Feb 2022 18:56:37
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:56:38
# get the patient idx, per label where this method fails the most
label = 0
error_thres = 6
if type(res) != np.ndarray: res.numpy()
print('label:{} \naFD>{}\nPatient idx: {}'.format(phases[label],error_thres,np.where(res[:,label]>error_thres)[0]))
print('gt length: {}'.format(cycle_len[res[:,label]>error_thres]))
print('gt indices: {}'.format(gt_ind[:,label][(res[:,label]>error_thres)]))
print('pred indices: {}'.format(upred_ind[:,label][(res[:,label]>error_thres)]))
print('aFD: {}'.format(res[:,label][res[:,label]>error_thres]))
# Mon, 28 Feb 2022 18:56:38
# TOF unsupervised center 14_41 gaus 2
sb.set_context('paper')
sb.set(font_scale = 2)
_ =df.plot(kind='box')
sb.violinplot(data=df)
pd.options.display.float_format = "{:,.2f}".format
df_summarized = pd.concat([df.mean(axis=0),df.std(axis=0), df.median(axis=0)], axis=1)
df_summarized.columns = ['mean', 'SD', 'meadian']
print(df_summarized.mean())
print(df_summarized)

sb.set_context('paper')
sb.set(font_scale = 1.8)
fig, axes = plt.subplots(1,5,figsize=(25,4))
axes = axes.flatten()
for i,ax in enumerate(axes):
    _ = ax.scatter(upred_ind[:,i],gt_ind[:,i])
    _ = ax.axline((1, 1), slope=1)
    _ = ax.set_title(phases[i])
    _ = ax.set_xlabel('Index - pred')
    _ = ax.set_ylabel('Index - gt')
    _ = ax.set_xlim([0,35])
    _ = ax.set_ylim([0,35])
plt.tight_layout()
# Mon, 28 Feb 2022 18:56:38
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:56:39
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:56:40
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:56:41
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:56:42
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:56:43
_jupyterlab_variableinspector_dict_list()
# Mon, 28 Feb 2022 18:59:12
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 08:08:57
# predict per pathology, on aligned curves
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax2 = ax.twinx()
ax2.margins(0,0)
ax.margins(0,0)
ind_all = []

for p in df_meta.pathology.unique():
    msk = get_msk_for_pathology(df_meta, p)
    dirs_minf = dirs[msk]
    norms_minf = norms[msk]
    #for n in dirs_minf:
    #    _ = ax.plot(n, alpha=0.5, zorder=0)
    ind_all.extend(np.stack([detect_phases(elem, len(elem), for elem in dirs_minf)], axis=0))
    print(ind_all[0].shape)

    df = pd.DataFrame(dirs_minf).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, label='d_t_{}'.format(p),ci='sd',err_style=None,zorder=2,legend=False,ax=ax)
    #_ = ax.set_title('Mean direction +/- SD - aligned at ED')
    _ = ax.set_xlabel('Time - linear interpolated to 40 frames')
    _ = ax.set_ylabel('direction d_t')
    _ = ax.legend(loc='upper left')
    
    df = pd.DataFrame(norms_minf).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, label='|v_t|_{}'.format(p),ci='sd', linestyle='dashed',err_style=None,zorder=2,legend=False,ax=ax2)
    _ = ax2.set_ylabel('norm |v_t|')

    _ = ax2.legend(loc='upper right')
plt.show()
# Tue, 01 Mar 2022 08:08:58
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 08:09:06
# predict per pathology, on aligned curves
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax2 = ax.twinx()
ax2.margins(0,0)
ax.margins(0,0)
ind_all = []

for p in df_meta.pathology.unique():
    msk = get_msk_for_pathology(df_meta, p)
    dirs_minf = dirs[msk]
    norms_minf = norms[msk]
    #for n in dirs_minf:
    #    _ = ax.plot(n, alpha=0.5, zorder=0)
    ind_all.extend(np.stack([detect_phases(elem, len(elem) for elem in dirs_minf)], axis=0))
    print(ind_all[0].shape)

    df = pd.DataFrame(dirs_minf).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, label='d_t_{}'.format(p),ci='sd',err_style=None,zorder=2,legend=False,ax=ax)
    #_ = ax.set_title('Mean direction +/- SD - aligned at ED')
    _ = ax.set_xlabel('Time - linear interpolated to 40 frames')
    _ = ax.set_ylabel('direction d_t')
    _ = ax.legend(loc='upper left')
    
    df = pd.DataFrame(norms_minf).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, label='|v_t|_{}'.format(p),ci='sd', linestyle='dashed',err_style=None,zorder=2,legend=False,ax=ax2)
    _ = ax2.set_ylabel('norm |v_t|')

    _ = ax2.legend(loc='upper right')
plt.show()
# Tue, 01 Mar 2022 08:09:15
# predict per pathology, on aligned curves
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax2 = ax.twinx()
ax2.margins(0,0)
ax.margins(0,0)
ind_all = []

for p in df_meta.pathology.unique():
    msk = get_msk_for_pathology(df_meta, p)
    dirs_minf = dirs[msk]
    norms_minf = norms[msk]
    #for n in dirs_minf:
    #    _ = ax.plot(n, alpha=0.5, zorder=0)
    ind_all.extend(np.stack([detect_phases(elem, len(elem)) for elem in dirs_minf)], axis=0))
    print(ind_all[0].shape)

    df = pd.DataFrame(dirs_minf).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, label='d_t_{}'.format(p),ci='sd',err_style=None,zorder=2,legend=False,ax=ax)
    #_ = ax.set_title('Mean direction +/- SD - aligned at ED')
    _ = ax.set_xlabel('Time - linear interpolated to 40 frames')
    _ = ax.set_ylabel('direction d_t')
    _ = ax.legend(loc='upper left')
    
    df = pd.DataFrame(norms_minf).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, label='|v_t|_{}'.format(p),ci='sd', linestyle='dashed',err_style=None,zorder=2,legend=False,ax=ax2)
    _ = ax2.set_ylabel('norm |v_t|')

    _ = ax2.legend(loc='upper right')
plt.show()
# Tue, 01 Mar 2022 08:09:15
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 08:09:37
# predict per pathology, on aligned curves
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax2 = ax.twinx()
ax2.margins(0,0)
ax.margins(0,0)
ind_all = []

for p in df_meta.pathology.unique():
    msk = get_msk_for_pathology(df_meta, p)
    dirs_minf = dirs[msk]
    norms_minf = norms[msk]
    #for n in dirs_minf:
    #    _ = ax.plot(n, alpha=0.5, zorder=0)
    ind_all.extend(np.stack([detect_phases(elem, len(elem)) for elem in dirs_minf], axis=0)))
    print(ind_all[0].shape)

    df = pd.DataFrame(dirs_minf).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, label='d_t_{}'.format(p),ci='sd',err_style=None,zorder=2,legend=False,ax=ax)
    #_ = ax.set_title('Mean direction +/- SD - aligned at ED')
    _ = ax.set_xlabel('Time - linear interpolated to 40 frames')
    _ = ax.set_ylabel('direction d_t')
    _ = ax.legend(loc='upper left')
    
    df = pd.DataFrame(norms_minf).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, label='|v_t|_{}'.format(p),ci='sd', linestyle='dashed',err_style=None,zorder=2,legend=False,ax=ax2)
    _ = ax2.set_ylabel('norm |v_t|')

    _ = ax2.legend(loc='upper right')
plt.show()
# Tue, 01 Mar 2022 08:09:58
# predict per pathology, on aligned curves
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax2 = ax.twinx()
ax2.margins(0,0)
ax.margins(0,0)
ind_all = []

for p in df_meta.pathology.unique():
    msk = get_msk_for_pathology(df_meta, p)
    dirs_minf = dirs[msk]
    norms_minf = norms[msk]
    #for n in dirs_minf:
    #    _ = ax.plot(n, alpha=0.5, zorder=0)
    ind_all.extend(np.stack([detect_phases(elem, len(elem)) for elem in dirs_minf], axis=0))
    print(ind_all[0].shape)

    df = pd.DataFrame(dirs_minf).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, label='d_t_{}'.format(p),ci='sd',err_style=None,zorder=2,legend=False,ax=ax)
    #_ = ax.set_title('Mean direction +/- SD - aligned at ED')
    _ = ax.set_xlabel('Time - linear interpolated to 40 frames')
    _ = ax.set_ylabel('direction d_t')
    _ = ax.legend(loc='upper left')
    
    df = pd.DataFrame(norms_minf).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, label='|v_t|_{}'.format(p),ci='sd', linestyle='dashed',err_style=None,zorder=2,legend=False,ax=ax2)
    _ = ax2.set_ylabel('norm |v_t|')

    _ = ax2.legend(loc='upper right')
plt.show()
# Tue, 01 Mar 2022 08:09:59
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 08:10:29
ind_all
# Tue, 01 Mar 2022 08:10:29
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 08:10:55
len(ind_all)
# Tue, 01 Mar 2022 08:10:55
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 08:12:50
pd.DataFrame(ind_all).hist()
# Tue, 01 Mar 2022 08:12:51
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 08:20:36
# align norm an direction by the cardiac phase ED
# resample all 1D feature vectors to the same length
# this should help to validate the pre-defined rules, and if both together explains the cardiac phases

norms = []
dirs= []
cycle_len=[]
gt_ind = []
xval = np.linspace(0,1,40)

for i,n in enumerate(nda_vects[:]):
    cardiac_cycle_length = int(gt_len[i,:,0].sum())
    cycle_len.append(cardiac_cycle_length)
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
    gt_ind.append(ind)
    dir_axis=0
    gt_ed = ind[0]
    #assert cardiac_cycle_length>=gt_ed
    temp = n[:cardiac_cycle_length]
    
    
    dim = temp.shape[1:-1]
    #print(dim)
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    #print('centers: ',c.dtype)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    # direction
    directions = flow2direction_lambda(temp)[...,0]
    # direction mean
    directions = np.mean(directions,axis=(1,2,3))
    # direction ed aligned
    directions = np.roll(directions, -1*gt_ed)
    # direction interpolate to unique length
    f = interp1d(np.linspace(0,1,directions.shape[0]), directions, kind='linear')
    directions = f(xval)
    # direction min/max normalised between -1,1

    #directions = minmax_lambda([directions,lower,upper])
    
    # magnitude/norm as mean
    norm_full = np.linalg.norm(temp, axis=-1)
    norm_nda = norm_full.mean(axis=(1,2,3))
    # norm ed aligned
    norm_nda = np.roll(norm_nda, -1*gt_ed)
    # interpolate to unique length
    #f = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')
    #norm_nda = f(xval)
    norm_nda = np.interp(xval, np.linspace(0,1,norm_nda.shape[0]), norm_nda)
    # norm min/max aligned to 0,1
    #norm_nda = minmax_lambda([norm_nda,mid,upper])
    
    norms.append(norm_nda)
    dirs.append(directions)
# Tue, 01 Mar 2022 08:20:48
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 08:20:52
# get the mean values
gt_ind = np.stack(gt_ind,axis=0)
cycle_len = np.stack(cycle_len, axis=0)
norms = np.stack(norms, axis=0)
dirs = np.stack(dirs, axis=0)
norms_m = [norms.mean(axis=0)]
dirs_m=[dirs.mean(axis=0)]
# Tue, 01 Mar 2022 08:20:52
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 08:20:55
"""i = 0
cardiac_cycle_length = int(gt_len[i,:,0].sum())
cycle_len.append(cardiac_cycle_length)
ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
gt_ind.append(ind)
dir_axis=0
gt_ed = ind[0]
#assert cardiac_cycle_length>=gt_ed
temp = n[:cardiac_cycle_length]
norm_full = np.linalg.norm(temp, axis=-1)
norm_nda = norm_full.mean(axis=(1,2,3))"""
# Tue, 01 Mar 2022 08:20:55
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 08:21:00
# seaborn
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax.margins(0,0)
for n in dirs:
    _ = ax.plot(n, alpha=0.5, zorder=0)

df = pd.DataFrame(dirs).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='blue', label=r'$\alpha_t$',ci='sd',err_style='bars',zorder=2,legend=False,ax=ax)
#_ = ax.set_title('Mean direction +/- SD - aligned at ED')
_ = ax.set_xlabel('Time (t) - linear interpolated to 40 frames')
_ = ax.set_ylabel(r'Angle $\alpha_t$ of $\phi_t$')
_ = ax.legend(loc='upper left')
ax2 = ax.twinx()
ax2.margins(0,0)
df = pd.DataFrame(norms).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='black', label=r'|$\phi_t$|',ci='sd', linestyle='dashed',err_style='bars',zorder=2,legend=False,ax=ax2)
_ = ax2.set_ylabel(r'Norm |$\phi_t$|')

_ = ax2.legend(loc='upper right')
# Tue, 01 Mar 2022 08:21:00
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 08:21:11
# define df and helper for pathology extractio
# this should have the same order as our inference data
#df_meta = pd.read_csv(folds_chooser.selected)
df_meta = pd.read_csv('/mnt/sds/sd20i001/sven/data/acdc/02_imported_4D_unfiltered/df_kfold.csv')
df_meta = df_meta.loc[:, ~df_meta.columns.str.contains('^Unnamed')]
df_meta = df_meta[df_meta.patient != 'patient090']  #we excluded this patient

def get_msk_for_pathology(df_, pathology='minf'):
    msk = []
    for f in [0,1,2,3,4]:
        patients = df_[df_.fold.isin([f])]
        patients = patients[patients['phase']=='ED']
        pat = patients[patients['modality'] == 'test']['patient'].str.lower().unique()
        sub_df = patients[patients.patient.isin(pat)].drop_duplicates(ignore_index=True, subset='patient')
        sub_df = sub_df.drop('x_path',axis=1).drop('y_path',axis=1)
        msk.append(sub_df['pathology'].str.lower()==pathology.lower())
    return np.concatenate(msk)
df_meta
# Tue, 01 Mar 2022 08:21:11
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 08:21:15
# interactive plot per pathology
@interact
def plot_per_pathology(p=df_meta.pathology.unique()):
    print(p)
    msk = get_msk_for_pathology(df_meta, p)

    import seaborn as sb
    sb.set_context('paper')
    sb.set(font_scale = 2)
    fig, ax = plt.subplots(figsize=(25,5))
    ax.margins(0,0)
    dirs_minf = dirs[msk]
    norms_minf = norms[msk]
    for n in dirs_minf:
        _ = ax.plot(n, alpha=0.5, zorder=0)

    df = pd.DataFrame(dirs_minf).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, color='blue', label='d_t',ci='sd',err_style='bars',zorder=2,legend=False,ax=ax)
    #_ = ax.set_title('Mean direction +/- SD - aligned at ED')
    _ = ax.set_xlabel('Time - linear interpolated to 40 frames')
    _ = ax.set_ylabel('direction d_t')
    _ = ax.legend(loc='upper left')
    ax2 = ax.twinx()
    ax2.margins(0,0)
    df = pd.DataFrame(norms_minf).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, color='black', label='|v_t|',ci='sd', linestyle='dashed',err_style='bars',zorder=2,legend=False,ax=ax2)
    _ = ax2.set_ylabel('norm |v_t|')

    _ = ax2.legend(loc='upper right')
    plt.show()
# Tue, 01 Mar 2022 08:21:15
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 08:21:19
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax2 = ax.twinx()
ax2.margins(0,0)
ax.margins(0,0)
for p in df_meta.pathology.unique():
    msk = get_msk_for_pathology(df_meta, p)
    dirs_minf = dirs[msk]
    norms_minf = norms[msk]
    #for n in dirs_minf:
    #    _ = ax.plot(n, alpha=0.5, zorder=0)

    df = pd.DataFrame(dirs_minf).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, label='d_t_{}'.format(p),ci='sd',err_style=None,zorder=2,legend=False,ax=ax)
    #_ = ax.set_title('Mean direction +/- SD - aligned at ED')
    _ = ax.set_xlabel('Time - linear interpolated to 40 frames')
    _ = ax.set_ylabel('direction d_t')
    _ = ax.legend(loc='upper left')
    
    df = pd.DataFrame(norms_minf).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, label='|v_t|_{}'.format(p),ci='sd', linestyle='dashed',err_style=None,zorder=2,legend=False,ax=ax2)
    _ = ax2.set_ylabel('norm |v_t|')

    _ = ax2.legend(loc='upper right')
plt.show()
# Tue, 01 Mar 2022 08:21:20
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 08:21:55
# predict per pathology, on aligned curves
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax2 = ax.twinx()
ax2.margins(0,0)
ax.margins(0,0)
ind_all = []

for p in df_meta.pathology.unique():
    msk = get_msk_for_pathology(df_meta, p)
    dirs_minf = dirs[msk]
    norms_minf = norms[msk]
    #for n in dirs_minf:
    #    _ = ax.plot(n, alpha=0.5, zorder=0)
    ind_all.extend(np.stack([detect_phases(elem, len(elem)) for elem in dirs_minf], axis=0))

    df = pd.DataFrame(dirs_minf).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, label='d_t_{}'.format(p),ci='sd',err_style=None,zorder=2,legend=False,ax=ax)
    #_ = ax.set_title('Mean direction +/- SD - aligned at ED')
    _ = ax.set_xlabel('Time - linear interpolated to 40 frames')
    _ = ax.set_ylabel('direction d_t')
    _ = ax.legend(loc='upper left')
    
    df = pd.DataFrame(norms_minf).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, label='|v_t|_{}'.format(p),ci='sd', linestyle='dashed',err_style=None,zorder=2,legend=False,ax=ax2)
    _ = ax2.set_ylabel('norm |v_t|')

    _ = ax2.legend(loc='upper right')
plt.show()
# Tue, 01 Mar 2022 08:21:56
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 08:22:08
pd.DataFrame(ind_all).hist()
# Tue, 01 Mar 2022 08:22:08
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 08:24:00
# interactive plot per pathology
@interact
def plot_per_pathology(p=df_meta.pathology.unique()):
    print(p)
    msk = get_msk_for_pathology(df_meta, p)

    import seaborn as sb
    sb.set_context('paper')
    sb.set(font_scale = 2)
    fig, ax = plt.subplots(figsize=(25,5))
    ax.margins(0,0)
    dirs_minf = dirs[msk]
    norms_minf = norms[msk]
    for n in dirs_minf:
        _ = ax.plot(n, alpha=0.5, zorder=0)

    df = pd.DataFrame(dirs).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, color='blue', label=r'$\alpha_t$',ci='sd',err_style='bars',zorder=2,legend=False,ax=ax)
    #_ = ax.set_title('Mean direction +/- SD - aligned at ED')
    _ = ax.set_xlabel('Time (t) - linear interpolated to 40 frames')
    _ = ax.set_ylabel(r'Angle $\alpha_t$ of $\phi_t$')
    _ = ax.legend(loc='upper left')
    ax2 = ax.twinx()
    ax2.margins(0,0)
    df = pd.DataFrame(norms).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, color='black', label=r'|$\phi_t$|',ci='sd', linestyle='dashed',err_style='bars',zorder=2,legend=False,ax=ax2)
    _ = ax2.set_ylabel(r'Norm |$\phi_t$|')

    _ = ax2.legend(loc='upper right')
# Tue, 01 Mar 2022 08:24:00
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 08:24:59
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax2 = ax.twinx()
ax2.margins(0,0)
ax.margins(0,0)
for p in df_meta.pathology.unique():
    msk = get_msk_for_pathology(df_meta, p)
    dirs_minf = dirs[msk]
    norms_minf = norms[msk]
    #for n in dirs_minf:
    #    _ = ax.plot(n, alpha=0.5, zorder=0)
    
    df = pd.DataFrame(dirs).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, color='blue', label=r'$\alpha_t$',ci='sd',err_style='bars',zorder=2,legend=False,ax=ax)
    #_ = ax.set_title('Mean direction +/- SD - aligned at ED')
    _ = ax.set_xlabel('Time (t) - linear interpolated to 40 frames')
    _ = ax.set_ylabel(r'Angle $\alpha_t$ of $\phi_t$')
    _ = ax.legend(loc='upper left')
    ax2 = ax.twinx()
    ax2.margins(0,0)
    df = pd.DataFrame(norms).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, color='black', label=r'|$\phi_t$|',ci='sd', linestyle='dashed',err_style='bars',zorder=2,legend=False,ax=ax2)
    _ = ax2.set_ylabel(r'Norm |$\phi_t$|')

    _ = ax2.legend(loc='upper right')

#     df = pd.DataFrame(dirs_minf).melt()
#     _ = sb.lineplot(x='variable', y='value',data=df, label='d_t_{}'.format(p),ci='sd',err_style=None,zorder=2,legend=False,ax=ax)
#     #_ = ax.set_title('Mean direction +/- SD - aligned at ED')
#     _ = ax.set_xlabel('Time - linear interpolated to 40 frames')
#     _ = ax.set_ylabel('direction d_t')
#     _ = ax.legend(loc='upper left')
    
#     df = pd.DataFrame(norms_minf).melt()
#     _ = sb.lineplot(x='variable', y='value',data=df, label='|v_t|_{}'.format(p),ci='sd', linestyle='dashed',err_style=None,zorder=2,legend=False,ax=ax2)
#     _ = ax2.set_ylabel('norm |v_t|')

#     _ = ax2.legend(loc='upper right')
plt.show()
# Tue, 01 Mar 2022 08:24:59
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 08:25:19
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax2 = ax.twinx()
ax2.margins(0,0)
ax.margins(0,0)
for p in df_meta.pathology.unique():
    msk = get_msk_for_pathology(df_meta, p)
    dirs_minf = dirs[msk]
    norms_minf = norms[msk]
    #for n in dirs_minf:
    #    _ = ax.plot(n, alpha=0.5, zorder=0)
    
    df = pd.DataFrame(dirs).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, color='blue', label=r'$\alpha_t$',ci='sd',err_style='bars',zorder=2,legend=False,ax=ax)
    #_ = ax.set_title('Mean direction +/- SD - aligned at ED')
    _ = ax.set_xlabel('Time (t) - linear interpolated to 40 frames')
    _ = ax.set_ylabel(r'Angle $\alpha_t$ of $\phi_t$')
    _ = ax.legend(loc='upper left')
    ax2 = ax.twinx()
    ax2.margins(0,0)
    df = pd.DataFrame(norms).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, color='black', label=r'|$\phi_t$|',ci='sd', linestyle='dashed',err_style='bars',zorder=2,legend=False,ax=ax2)
    _ = ax2.set_ylabel(r'Norm |$\phi_t$|')

    _ = ax2.legend(loc='upper right')
plt.show()
# Tue, 01 Mar 2022 08:25:20
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 08:25:32
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax2 = ax.twinx()
ax2.margins(0,0)
ax.margins(0,0)
for p in df_meta.pathology.unique():
    msk = get_msk_for_pathology(df_meta, p)
    dirs_minf = dirs[msk]
    norms_minf = norms[msk]
    #for n in dirs_minf:
    #    _ = ax.plot(n, alpha=0.5, zorder=0)
    
    df = pd.DataFrame(dirs).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, color='blue', label=r'$\alpha_t$',ci='sd',err_style='bars',zorder=2,legend=False,ax=ax)
    #_ = ax.set_title('Mean direction +/- SD - aligned at ED')
    _ = ax.set_xlabel('Time (t) - linear interpolated to 40 frames')
    _ = ax.set_ylabel(r'Angle $\alpha_t$ of $\phi_t$')
    _ = ax.legend(loc='upper left')
    ax2 = ax.twinx()
    ax2.margins(0,0)
    df = pd.DataFrame(norms).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, color='black', label=r'|$\phi_t$|',ci='sd', linestyle='dashed',err_style='bars',zorder=2,legend=False,ax=ax2)
    _ = ax2.set_ylabel(r'Norm |$\phi_t$|')
    _ = ax2.legend(loc='upper right')
plt.show()
# Tue, 01 Mar 2022 08:25:33
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 08:26:39
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax2 = ax.twinx()
ax2.margins(0,0)
ax.margins(0,0)
for p in df_meta.pathology.unique():
    msk = get_msk_for_pathology(df_meta, p)
    dirs_minf = dirs[msk]
    norms_minf = norms[msk]
    #for n in dirs_minf:
    #    _ = ax.plot(n, alpha=0.5, zorder=0)
    
    df = pd.DataFrame(dirs).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, color='blue', label=r'$\alpha_t$ {}'.format(p),ci='sd',err_style='bars',zorder=2,legend=False,ax=ax)
    #_ = ax.set_title('Mean direction +/- SD - aligned at ED')
    _ = ax.set_xlabel('Time (t) - linear interpolated to 40 frames')
    _ = ax.set_ylabel(r'Angle $\alpha_t$ of $\phi_t$')
    _ = ax.legend(loc='upper left')
    ax2 = ax.twinx()
    ax2.margins(0,0)
    df = pd.DataFrame(norms).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, color='black', label=r'|$\phi_t$| {}'.format(p),ci='sd', linestyle='dashed',err_style='bars',zorder=2,legend=False,ax=ax2)
    _ = ax2.set_ylabel(r'Norm |$\phi_t$|')
    _ = ax2.legend(loc='upper right')
    
    
#     df = pd.DataFrame(dirs_minf).melt()
#     _ = sb.lineplot(x='variable', y='value',data=df, label='d_t_{}'.format(p),ci='sd',err_style=None,zorder=2,legend=False,ax=ax)
#     #_ = ax.set_title('Mean direction +/- SD - aligned at ED')
#     _ = ax.set_xlabel('Time - linear interpolated to 40 frames')
#     _ = ax.set_ylabel('direction d_t')
#     _ = ax.legend(loc='upper left')
    
#     df = pd.DataFrame(norms_minf).melt()
#     _ = sb.lineplot(x='variable', y='value',data=df, label='|v_t|_{}'.format(p),ci='sd', linestyle='dashed',err_style=None,zorder=2,legend=False,ax=ax2)
#     _ = ax2.set_ylabel('norm |v_t|')

#     _ = ax2.legend(loc='upper right')
plt.show()
# Tue, 01 Mar 2022 08:26:40
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 08:27:29
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax2 = ax.twinx()
ax2.margins(0,0)
ax.margins(0,0)
for p in df_meta.pathology.unique():
    msk = get_msk_for_pathology(df_meta, p)
    dirs_p = dirs[msk]
    norms_p = norms[msk]
    #for n in dirs_minf:
    #    _ = ax.plot(n, alpha=0.5, zorder=0)
    
    df = pd.DataFrame(dirs_p).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, color='blue', label=r'$\alpha_t$ {}'.format(p),ci='sd',err_style='bars',zorder=2,legend=False,ax=ax)
    #_ = ax.set_title('Mean direction +/- SD - aligned at ED')
    _ = ax.set_xlabel('Time (t) - linear interpolated to 40 frames')
    _ = ax.set_ylabel(r'Angle $\alpha_t$ of $\phi_t$')
    _ = ax.legend(loc='upper left')
    ax2 = ax.twinx()
    ax2.margins(0,0)
    df = pd.DataFrame(norms_p).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, color='black', label=r'|$\phi_t$| {}'.format(p),ci='sd', linestyle='dashed',err_style='bars',zorder=2,legend=False,ax=ax2)
    _ = ax2.set_ylabel(r'Norm |$\phi_t$|')
    _ = ax2.legend(loc='upper right')
    
    
#     df = pd.DataFrame(dirs_minf).melt()
#     _ = sb.lineplot(x='variable', y='value',data=df, label='d_t_{}'.format(p),ci='sd',err_style=None,zorder=2,legend=False,ax=ax)
#     #_ = ax.set_title('Mean direction +/- SD - aligned at ED')
#     _ = ax.set_xlabel('Time - linear interpolated to 40 frames')
#     _ = ax.set_ylabel('direction d_t')
#     _ = ax.legend(loc='upper left')
    
#     df = pd.DataFrame(norms_minf).melt()
#     _ = sb.lineplot(x='variable', y='value',data=df, label='|v_t|_{}'.format(p),ci='sd', linestyle='dashed',err_style=None,zorder=2,legend=False,ax=ax2)
#     _ = ax2.set_ylabel('norm |v_t|')

#     _ = ax2.legend(loc='upper right')
plt.show()
# Tue, 01 Mar 2022 08:27:30
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 08:27:52
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax2 = ax.twinx()
ax2.margins(0,0)
ax.margins(0,0)
for p in df_meta.pathology.unique():
    msk = get_msk_for_pathology(df_meta, p)
    dirs_p = dirs[msk]
    norms_p = norms[msk]
    #for n in dirs_minf:
    #    _ = ax.plot(n, alpha=0.5, zorder=0)
    
    df = pd.DataFrame(dirs_p).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, label=r'$\alpha_t$ {}'.format(p),ci='sd',err_style='bars',zorder=2,legend=False,ax=ax)
    #_ = ax.set_title('Mean direction +/- SD - aligned at ED')
    _ = ax.set_xlabel('Time (t) - linear interpolated to 40 frames')
    _ = ax.set_ylabel(r'Angle $\alpha_t$ of $\phi_t$')
    _ = ax.legend(loc='upper left')
    ax2 = ax.twinx()
    ax2.margins(0,0)
    df = pd.DataFrame(norms_p).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, label=r'|$\phi_t$| {}'.format(p),ci='sd', linestyle='dashed',err_style='bars',zorder=2,legend=False,ax=ax2)
    _ = ax2.set_ylabel(r'Norm |$\phi_t$|')
    _ = ax2.legend(loc='upper right')
    
    
#     df = pd.DataFrame(dirs_minf).melt()
#     _ = sb.lineplot(x='variable', y='value',data=df, label='d_t_{}'.format(p),ci='sd',err_style=None,zorder=2,legend=False,ax=ax)
#     #_ = ax.set_title('Mean direction +/- SD - aligned at ED')
#     _ = ax.set_xlabel('Time - linear interpolated to 40 frames')
#     _ = ax.set_ylabel('direction d_t')
#     _ = ax.legend(loc='upper left')
    
#     df = pd.DataFrame(norms_minf).melt()
#     _ = sb.lineplot(x='variable', y='value',data=df, label='|v_t|_{}'.format(p),ci='sd', linestyle='dashed',err_style=None,zorder=2,legend=False,ax=ax2)
#     _ = ax2.set_ylabel('norm |v_t|')

#     _ = ax2.legend(loc='upper right')
plt.show()
# Tue, 01 Mar 2022 08:27:53
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 08:28:21
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax2 = ax.twinx()
ax2.margins(0,0)
ax.margins(0,0)
for p in df_meta.pathology.unique():
    msk = get_msk_for_pathology(df_meta, p)
    dirs_p = dirs[msk]
    norms_p = norms[msk]
    #for n in dirs_minf:
    #    _ = ax.plot(n, alpha=0.5, zorder=0)
    
    df = pd.DataFrame(dirs_p).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, label=r'$\alpha_t$ {}'.format(p),ci='sd',err_style='None,zorder=2,legend=False,ax=ax)
    #_ = ax.set_title('Mean direction +/- SD - aligned at ED')
    _ = ax.set_xlabel('Time (t) - linear interpolated to 40 frames')
    _ = ax.set_ylabel(r'Angle $\alpha_t$ of $\phi_t$')
    _ = ax.legend(loc='upper left')
    ax2 = ax.twinx()
    ax2.margins(0,0)
    df = pd.DataFrame(norms_p).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, label=r'|$\phi_t$| {}'.format(p),ci='sd', linestyle='dashed',err_style=None,zorder=2,legend=False,ax=ax2)
    _ = ax2.set_ylabel(r'Norm |$\phi_t$|')
    _ = ax2.legend(loc='upper right')
    
    
#     df = pd.DataFrame(dirs_minf).melt()
#     _ = sb.lineplot(x='variable', y='value',data=df, label='d_t_{}'.format(p),ci='sd',err_style=None,zorder=2,legend=False,ax=ax)
#     #_ = ax.set_title('Mean direction +/- SD - aligned at ED')
#     _ = ax.set_xlabel('Time - linear interpolated to 40 frames')
#     _ = ax.set_ylabel('direction d_t')
#     _ = ax.legend(loc='upper left')
    
#     df = pd.DataFrame(norms_minf).melt()
#     _ = sb.lineplot(x='variable', y='value',data=df, label='|v_t|_{}'.format(p),ci='sd', linestyle='dashed',err_style=None,zorder=2,legend=False,ax=ax2)
#     _ = ax2.set_ylabel('norm |v_t|')

#     _ = ax2.legend(loc='upper right')
plt.show()
# Tue, 01 Mar 2022 08:28:21
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 08:28:26
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax2 = ax.twinx()
ax2.margins(0,0)
ax.margins(0,0)
for p in df_meta.pathology.unique():
    msk = get_msk_for_pathology(df_meta, p)
    dirs_p = dirs[msk]
    norms_p = norms[msk]
    #for n in dirs_minf:
    #    _ = ax.plot(n, alpha=0.5, zorder=0)
    
    df = pd.DataFrame(dirs_p).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, label=r'$\alpha_t$ {}'.format(p),ci='sd',err_style=None,zorder=2,legend=False,ax=ax)
    #_ = ax.set_title('Mean direction +/- SD - aligned at ED')
    _ = ax.set_xlabel('Time (t) - linear interpolated to 40 frames')
    _ = ax.set_ylabel(r'Angle $\alpha_t$ of $\phi_t$')
    _ = ax.legend(loc='upper left')
    ax2 = ax.twinx()
    ax2.margins(0,0)
    df = pd.DataFrame(norms_p).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, label=r'|$\phi_t$| {}'.format(p),ci='sd', linestyle='dashed',err_style=None,zorder=2,legend=False,ax=ax2)
    _ = ax2.set_ylabel(r'Norm |$\phi_t$|')
    _ = ax2.legend(loc='upper right')
    
    
#     df = pd.DataFrame(dirs_minf).melt()
#     _ = sb.lineplot(x='variable', y='value',data=df, label='d_t_{}'.format(p),ci='sd',err_style=None,zorder=2,legend=False,ax=ax)
#     #_ = ax.set_title('Mean direction +/- SD - aligned at ED')
#     _ = ax.set_xlabel('Time - linear interpolated to 40 frames')
#     _ = ax.set_ylabel('direction d_t')
#     _ = ax.legend(loc='upper left')
    
#     df = pd.DataFrame(norms_minf).melt()
#     _ = sb.lineplot(x='variable', y='value',data=df, label='|v_t|_{}'.format(p),ci='sd', linestyle='dashed',err_style=None,zorder=2,legend=False,ax=ax2)
#     _ = ax2.set_ylabel('norm |v_t|')

#     _ = ax2.legend(loc='upper right')
plt.show()
# Tue, 01 Mar 2022 08:28:27
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 08:29:09
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax2 = ax.twinx()
ax2.margins(0,0)
ax.margins(0,0)
for p in df_meta.pathology.unique():
    msk = get_msk_for_pathology(df_meta, p)
    dirs_p = dirs[msk]
    norms_p = norms[msk]
    #for n in dirs_minf:
    #    _ = ax.plot(n, alpha=0.5, zorder=0)
    
    df = pd.DataFrame(dirs_p).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, label=r'$\alpha_t$ {}'.format(p),ci='sd',err_style=None,zorder=2,legend=False,ax=ax)
    #_ = ax.set_title('Mean direction +/- SD - aligned at ED')
    _ = ax.set_xlabel('Time (t) - linear interpolated to 40 frames')
    _ = ax.set_ylabel(r'Angle $\alpha_t$ of $\phi_t$')
    _ = ax.legend(loc='upper left')
    #ax2 = ax.twinx()
    #ax2.margins(0,0)
    df = pd.DataFrame(norms_p).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, label=r'|$\phi_t$| {}'.format(p),ci='sd', linestyle='dashed',err_style=None,zorder=2,legend=False,ax=ax2)
    _ = ax2.set_ylabel(r'Norm |$\phi_t$|')
    _ = ax2.legend(loc='upper right')
    
    
#     df = pd.DataFrame(dirs_minf).melt()
#     _ = sb.lineplot(x='variable', y='value',data=df, label='d_t_{}'.format(p),ci='sd',err_style=None,zorder=2,legend=False,ax=ax)
#     #_ = ax.set_title('Mean direction +/- SD - aligned at ED')
#     _ = ax.set_xlabel('Time - linear interpolated to 40 frames')
#     _ = ax.set_ylabel('direction d_t')
#     _ = ax.legend(loc='upper left')
    
#     df = pd.DataFrame(norms_minf).melt()
#     _ = sb.lineplot(x='variable', y='value',data=df, label='|v_t|_{}'.format(p),ci='sd', linestyle='dashed',err_style=None,zorder=2,legend=False,ax=ax2)
#     _ = ax2.set_ylabel('norm |v_t|')

#     _ = ax2.legend(loc='upper right')
plt.show()
# Tue, 01 Mar 2022 08:29:09
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 08:29:24
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax2 = ax.twinx()
ax2.margins(0,0)
ax.margins(0,0)
for p in df_meta.pathology.unique():
    msk = get_msk_for_pathology(df_meta, p)
    dirs_p = dirs[msk]
    norms_p = norms[msk]
    #for n in dirs_minf:
    #    _ = ax.plot(n, alpha=0.5, zorder=0)
    
    df = pd.DataFrame(dirs_p).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, label=r'$\alpha_t$ {}'.format(p),ci='sd',err_style=None,zorder=2,legend=False,ax=ax)
    #_ = ax.set_title('Mean direction +/- SD - aligned at ED')
    _ = ax.set_xlabel('Time (t) - linear interpolated to 40 frames')
    _ = ax.set_ylabel(r'Angle $\alpha_t$ of $\phi_t$')
    _ = ax.legend(loc='upper left')
    df = pd.DataFrame(norms_p).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, label=r'|$\phi_t$| {}'.format(p),ci='sd', linestyle='dashed',err_style=None,zorder=2,legend=False,ax=ax2)
    _ = ax2.set_ylabel(r'Norm |$\phi_t$|')
    _ = ax2.legend(loc='upper right')
    
plt.show()
# Tue, 01 Mar 2022 08:29:25
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 08:30:01
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax2 = ax.twinx()
ax2.margins(0,0)
ax.margins(0,0)
for p in df_meta.pathology.unique():
    msk = get_msk_for_pathology(df_meta, p)
    dirs_p = dirs[msk]
    norms_p = norms[msk]
    #for n in dirs_minf:
    #    _ = ax.plot(n, alpha=0.5, zorder=0)
    
    df = pd.DataFrame(dirs_p).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, label=r'$\alpha_t$ {}'.format(p),ci='sd',err_style='bar',zorder=2,legend=False,ax=ax)
    #_ = ax.set_title('Mean direction +/- SD - aligned at ED')
    _ = ax.set_xlabel('Time (t) - linear interpolated to 40 frames')
    _ = ax.set_ylabel(r'Angle $\alpha_t$ of $\phi_t$')
    _ = ax.legend(loc='upper left')
    df = pd.DataFrame(norms_p).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, label=r'|$\phi_t$| {}'.format(p),ci='sd', linestyle='dashed',err_style=None,zorder=2,legend=False,ax=ax2)
    _ = ax2.set_ylabel(r'Norm |$\phi_t$|')
    _ = ax2.legend(loc='upper right')
    
plt.show()
# Tue, 01 Mar 2022 08:30:10
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax2 = ax.twinx()
ax2.margins(0,0)
ax.margins(0,0)
for p in df_meta.pathology.unique():
    msk = get_msk_for_pathology(df_meta, p)
    dirs_p = dirs[msk]
    norms_p = norms[msk]
    #for n in dirs_minf:
    #    _ = ax.plot(n, alpha=0.5, zorder=0)
    
    df = pd.DataFrame(dirs_p).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, label=r'$\alpha_t$ {}'.format(p),ci='sd',err_style='bars',zorder=2,legend=False,ax=ax)
    #_ = ax.set_title('Mean direction +/- SD - aligned at ED')
    _ = ax.set_xlabel('Time (t) - linear interpolated to 40 frames')
    _ = ax.set_ylabel(r'Angle $\alpha_t$ of $\phi_t$')
    _ = ax.legend(loc='upper left')
    df = pd.DataFrame(norms_p).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, label=r'|$\phi_t$| {}'.format(p),ci='sd', linestyle='dashed',err_style=None,zorder=2,legend=False,ax=ax2)
    _ = ax2.set_ylabel(r'Norm |$\phi_t$|')
    _ = ax2.legend(loc='upper right')
    
plt.show()
# Tue, 01 Mar 2022 08:30:10
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 08:30:17
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax2 = ax.twinx()
ax2.margins(0,0)
ax.margins(0,0)
for p in df_meta.pathology.unique():
    msk = get_msk_for_pathology(df_meta, p)
    dirs_p = dirs[msk]
    norms_p = norms[msk]
    #for n in dirs_minf:
    #    _ = ax.plot(n, alpha=0.5, zorder=0)
    
    df = pd.DataFrame(dirs_p).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, label=r'$\alpha_t$ {}'.format(p),ci='sd',err_style='band',zorder=2,legend=False,ax=ax)
    #_ = ax.set_title('Mean direction +/- SD - aligned at ED')
    _ = ax.set_xlabel('Time (t) - linear interpolated to 40 frames')
    _ = ax.set_ylabel(r'Angle $\alpha_t$ of $\phi_t$')
    _ = ax.legend(loc='upper left')
    df = pd.DataFrame(norms_p).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, label=r'|$\phi_t$| {}'.format(p),ci='sd', linestyle='dashed',err_style=None,zorder=2,legend=False,ax=ax2)
    _ = ax2.set_ylabel(r'Norm |$\phi_t$|')
    _ = ax2.legend(loc='upper right')
    
plt.show()
# Tue, 01 Mar 2022 08:30:17
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 08:30:27
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax2 = ax.twinx()
ax2.margins(0,0)
ax.margins(0,0)
for p in df_meta.pathology.unique():
    msk = get_msk_for_pathology(df_meta, p)
    dirs_p = dirs[msk]
    norms_p = norms[msk]
    #for n in dirs_minf:
    #    _ = ax.plot(n, alpha=0.5, zorder=0)
    
    df = pd.DataFrame(dirs_p).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, label=r'$\alpha_t$ {}'.format(p),ci='sd',err_style=None,zorder=2,legend=False,ax=ax)
    #_ = ax.set_title('Mean direction +/- SD - aligned at ED')
    _ = ax.set_xlabel('Time (t) - linear interpolated to 40 frames')
    _ = ax.set_ylabel(r'Angle $\alpha_t$ of $\phi_t$')
    _ = ax.legend(loc='upper left')
    df = pd.DataFrame(norms_p).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, label=r'|$\phi_t$| {}'.format(p),ci='sd', linestyle='dashed',err_style=None,zorder=2,legend=False,ax=ax2)
    _ = ax2.set_ylabel(r'Norm |$\phi_t$|')
    _ = ax2.legend(loc='upper right')
    
plt.show()
# Tue, 01 Mar 2022 08:30:27
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 08:43:21
# predict per pathology, on aligned curves
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax2 = ax.twinx()
ax2.margins(0,0)
ax.margins(0,0)
ind_all = []

for p in df_meta.pathology.unique():
    msk = get_msk_for_pathology(df_meta, p)
    dirs_p = dirs[msk]
    norms_p = norms[msk]
    ind_ = np.stack([detect_phases(elem, len(elem)) for elem in dirs_p], axis=0)
    pd.DataFrame(ind_).hist()
    ind_all.extend(ind_)
plt.show()
# Tue, 01 Mar 2022 08:43:23
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 08:44:06
# predict per pathology, on aligned curves
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
ind_all = []

for p in df_meta.pathology.unique():
    msk = get_msk_for_pathology(df_meta, p)
    dirs_p = dirs[msk]
    norms_p = norms[msk]
    ind_ = np.stack([detect_phases(elem, len(elem)) for elem in dirs_p], axis=0)
    pd.DataFrame(ind_).hist()
    ind_all.extend(ind_)
# Tue, 01 Mar 2022 08:44:08
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 08:44:25
# align norm an direction by the cardiac phase ED
# resample all 1D feature vectors to the same length
# this should help to validate the pre-defined rules, and if both together explains the cardiac phases

norms = []
dirs= []
cycle_len=[]
gt_ind = []
xval = np.linspace(0,1,40)

for i,n in enumerate(nda_vects[:]):
    cardiac_cycle_length = int(gt_len[i,:,0].sum())
    cycle_len.append(cardiac_cycle_length)
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
    gt_ind.append(ind)
    dir_axis=0
    gt_ed = ind[0]
    #assert cardiac_cycle_length>=gt_ed
    temp = n[:cardiac_cycle_length]
    
    
    dim = temp.shape[1:-1]
    #print(dim)
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    #print('centers: ',c.dtype)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    # direction
    directions = flow2direction_lambda(temp)[...,0]
    # direction mean
    directions = np.mean(directions,axis=(1,2,3))
    # direction ed aligned
    directions = np.roll(directions, -1*gt_ed)
    # direction interpolate to unique length
    f = interp1d(np.linspace(0,1,directions.shape[0]), directions, kind='linear')
    directions = f(xval)
    # direction min/max normalised between -1,1

    directions = minmax_lambda([directions,lower,upper])
    
    # magnitude/norm as mean
    norm_full = np.linalg.norm(temp, axis=-1)
    norm_nda = norm_full.mean(axis=(1,2,3))
    # norm ed aligned
    norm_nda = np.roll(norm_nda, -1*gt_ed)
    # interpolate to unique length
    #f = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')
    #norm_nda = f(xval)
    norm_nda = np.interp(xval, np.linspace(0,1,norm_nda.shape[0]), norm_nda)
    # norm min/max aligned to 0,1
    norm_nda = minmax_lambda([norm_nda,mid,upper])
    
    norms.append(norm_nda)
    dirs.append(directions)
# Tue, 01 Mar 2022 08:44:37
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 08:44:38
# get the mean values
gt_ind = np.stack(gt_ind,axis=0)
cycle_len = np.stack(cycle_len, axis=0)
norms = np.stack(norms, axis=0)
dirs = np.stack(dirs, axis=0)
norms_m = [norms.mean(axis=0)]
dirs_m=[dirs.mean(axis=0)]
# Tue, 01 Mar 2022 08:44:38
# seaborn
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax.margins(0,0)
for n in dirs:
    _ = ax.plot(n, alpha=0.5, zorder=0)

df = pd.DataFrame(dirs).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='blue', label=r'$\alpha_t$',ci='sd',err_style='bars',zorder=2,legend=False,ax=ax)
#_ = ax.set_title('Mean direction +/- SD - aligned at ED')
_ = ax.set_xlabel('Time (t) - linear interpolated to 40 frames')
_ = ax.set_ylabel(r'Angle $\alpha_t$ of $\phi_t$')
_ = ax.legend(loc='upper left')
ax2 = ax.twinx()
ax2.margins(0,0)
df = pd.DataFrame(norms).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='black', label=r'|$\phi_t$|',ci='sd', linestyle='dashed',err_style='bars',zorder=2,legend=False,ax=ax2)
_ = ax2.set_ylabel(r'Norm |$\phi_t$|')

_ = ax2.legend(loc='upper right')
# Tue, 01 Mar 2022 08:44:39
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 08:44:40
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 08:44:51
# predict per pathology, on aligned curves
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
ind_all = []

for p in df_meta.pathology.unique():
    msk = get_msk_for_pathology(df_meta, p)
    dirs_p = dirs[msk]
    norms_p = norms[msk]
    ind_ = np.stack([detect_phases(elem, len(elem)) for elem in dirs_p], axis=0)
    pd.DataFrame(ind_).hist()
    ind_all.extend(ind_)
# Tue, 01 Mar 2022 08:44:52
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 08:44:58
# predict per pathology, on aligned curves
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
ind_all = []

for p in df_meta.pathology.unique():
    msk = get_msk_for_pathology(df_meta, p)
    dirs_p = dirs[msk]
    norms_p = norms[msk]
    ind_ = np.stack([detect_phases(elem, len(elem)) for elem in dirs_p], axis=0)
    _ = pd.DataFrame(ind_).hist()
    ind_all.extend(ind_)
# Tue, 01 Mar 2022 08:45:00
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 08:45:59
# predict per pathology, on aligned curves
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
ind_all = []

for p in df_meta.pathology.unique():
    msk = get_msk_for_pathology(df_meta, p)
    dirs_p = dirs[msk]
    norms_p = norms[msk]
    ind_ = np.stack([detect_phases(elem, len(elem)) for elem in dirs_p], axis=0)
    _ = pd.DataFrame(ind_).plot(kind='hist')
    ind_all.extend(ind_)
# Tue, 01 Mar 2022 08:46:00
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 08:46:19
# predict per pathology, on aligned curves
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
ind_all = []

for p in df_meta.pathology.unique():
    msk = get_msk_for_pathology(df_meta, p)
    dirs_p = dirs[msk]
    norms_p = norms[msk]
    ind_ = np.stack([detect_phases(elem, len(elem)) for elem in dirs_p], axis=0)
    _ = pd.DataFrame(ind_).plot(kind='violin')
    ind_all.extend(ind_)
# Tue, 01 Mar 2022 08:46:25
# predict per pathology, on aligned curves
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
ind_all = []

for p in df_meta.pathology.unique():
    msk = get_msk_for_pathology(df_meta, p)
    dirs_p = dirs[msk]
    norms_p = norms[msk]
    ind_ = np.stack([detect_phases(elem, len(elem)) for elem in dirs_p], axis=0)
    _ = pd.DataFrame(ind_).plot(kind='box')
    ind_all.extend(ind_)
# Tue, 01 Mar 2022 08:46:26
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 08:58:46
# align norm an direction by the cardiac phase ED
# resample all 1D feature vectors to the same length
# this should help to validate the pre-defined rules, and if both together explains the cardiac phases
target_t = 40
norms = []
dirs= []
cycle_len=[]
gt_ind = []
gt_ind_rolled = []
xval = np.linspace(0,1,target_t)

for i,n in enumerate(nda_vects[:]):
    cardiac_cycle_length = int(gt_len[i,:,0].sum())
    cycle_len.append(cardiac_cycle_length)
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
    gt_onehot = gt[i][:cardiac_cycle_length]
    gt_ind.append(ind)
    
    dir_axis=0
    gt_ed = ind[0]
    #assert cardiac_cycle_length>=gt_ed
    temp = n[:cardiac_cycle_length]
    
    gt_onehot_rolled = np.roll(gt_onehot, -1*ed)
    resize_factor = target_t/cardiac_cycle_length
    gt_onehot_rolled = np.rint(gt_ind_rolled * resize_factor)
    gt_ind_rolled.append(gt_ind_rolled)
    
    dim = temp.shape[1:-1]
    #print(dim)
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    #print('centers: ',c.dtype)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    # direction
    directions = flow2direction_lambda(temp)[...,0]
    # direction mean
    directions = np.mean(directions,axis=(1,2,3))
    # direction ed aligned
    directions = np.roll(directions, -1*gt_ed)
    # direction interpolate to unique length
    f = interp1d(np.linspace(0,1,directions.shape[0]), directions, kind='linear')
    directions = f(xval)
    # direction min/max normalised between -1,1

    directions = minmax_lambda([directions,lower,upper])
    
    # magnitude/norm as mean
    norm_full = np.linalg.norm(temp, axis=-1)
    norm_nda = norm_full.mean(axis=(1,2,3))
    # norm ed aligned
    norm_nda = np.roll(norm_nda, -1*gt_ed)
    # interpolate to unique length
    #f = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')
    #norm_nda = f(xval)
    norm_nda = np.interp(xval, np.linspace(0,1,norm_nda.shape[0]), norm_nda)
    # norm min/max aligned to 0,1
    norm_nda = minmax_lambda([norm_nda,mid,upper])
    
    norms.append(norm_nda)
    dirs.append(directions)
# Tue, 01 Mar 2022 08:59:13
# align norm an direction by the cardiac phase ED
# resample all 1D feature vectors to the same length
# this should help to validate the pre-defined rules, and if both together explains the cardiac phases
target_t = 40
norms = []
dirs= []
cycle_len=[]
gt_ind = []
gt_ind_rolled = []
xval = np.linspace(0,1,target_t)

for i,n in enumerate(nda_vects[:]):
    cardiac_cycle_length = int(gt_len[i,:,0].sum())
    cycle_len.append(cardiac_cycle_length)
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
    gt_onehot = gt[i][:cardiac_cycle_length]
    gt_ind.append(ind)
    
    dir_axis=0
    gt_ed = ind[0]
    #assert cardiac_cycle_length>=gt_ed
    temp = n[:cardiac_cycle_length]
    
    dim = temp.shape[1:-1]
    #print(dim)
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    #print('centers: ',c.dtype)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    # direction
    directions = flow2direction_lambda(temp)[...,0]
    # direction mean
    directions = np.mean(directions,axis=(1,2,3))
    # direction ed aligned
    directions = np.roll(directions, -1*gt_ed)
    # direction interpolate to unique length
    f = interp1d(np.linspace(0,1,directions.shape[0]), directions, kind='linear')
    directions = f(xval)
    # direction min/max normalised between -1,1

    directions = minmax_lambda([directions,lower,upper])
    
    # magnitude/norm as mean
    norm_full = np.linalg.norm(temp, axis=-1)
    norm_nda = norm_full.mean(axis=(1,2,3))
    # norm ed aligned
    norm_nda = np.roll(norm_nda, -1*gt_ed)
    # interpolate to unique length
    #f = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')
    #norm_nda = f(xval)
    norm_nda = np.interp(xval, np.linspace(0,1,norm_nda.shape[0]), norm_nda)
    # norm min/max aligned to 0,1
    norm_nda = minmax_lambda([norm_nda,mid,upper])
    
    gt_onehot_rolled = np.roll(gt_onehot, -1*ed)
    resize_factor = target_t/cardiac_cycle_length
    gt_onehot_rolled = np.rint(gt_ind_rolled * resize_factor)
    gt_ind_rolled.append(gt_ind_rolled)
    
    norms.append(norm_nda)
    dirs.append(directions)
# Tue, 01 Mar 2022 08:59:31
# align norm an direction by the cardiac phase ED
# resample all 1D feature vectors to the same length
# this should help to validate the pre-defined rules, and if both together explains the cardiac phases
target_t = 40
norms = []
dirs= []
cycle_len=[]
gt_ind = []
gt_ind_rolled = []
xval = np.linspace(0,1,target_t)

for i,n in enumerate(nda_vects[:]):
    cardiac_cycle_length = int(gt_len[i,:,0].sum())
    cycle_len.append(cardiac_cycle_length)
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
    gt_onehot = gt[i][:cardiac_cycle_length]
    gt_ind.append(ind)
    
    dir_axis=0
    gt_ed = ind[0]
    #assert cardiac_cycle_length>=gt_ed
    temp = n[:cardiac_cycle_length]
    
    dim = temp.shape[1:-1]
    #print(dim)
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    #print('centers: ',c.dtype)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    # direction
    directions = flow2direction_lambda(temp)[...,0]
    # direction mean
    directions = np.mean(directions,axis=(1,2,3))
    # direction ed aligned
    directions = np.roll(directions, -1*gt_ed)
    # direction interpolate to unique length
    f = interp1d(np.linspace(0,1,directions.shape[0]), directions, kind='linear')
    directions = f(xval)
    # direction min/max normalised between -1,1

    directions = minmax_lambda([directions,lower,upper])
    
    # magnitude/norm as mean
    norm_full = np.linalg.norm(temp, axis=-1)
    norm_nda = norm_full.mean(axis=(1,2,3))
    # norm ed aligned
    norm_nda = np.roll(norm_nda, -1*gt_ed)
    # interpolate to unique length
    #f = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')
    #norm_nda = f(xval)
    norm_nda = np.interp(xval, np.linspace(0,1,norm_nda.shape[0]), norm_nda)
    # norm min/max aligned to 0,1
    norm_nda = minmax_lambda([norm_nda,mid,upper])
    
    gt_onehot_rolled = np.roll(gt_onehot, -1*gt_ed)
    resize_factor = target_t/cardiac_cycle_length
    gt_onehot_rolled = np.rint(gt_ind_rolled * resize_factor)
    gt_ind_rolled.append(gt_ind_rolled)
    
    norms.append(norm_nda)
    dirs.append(directions)
# Tue, 01 Mar 2022 09:00:10
# align norm an direction by the cardiac phase ED
# resample all 1D feature vectors to the same length
# this should help to validate the pre-defined rules, and if both together explains the cardiac phases
target_t = 40
norms = []
dirs= []
cycle_len=[]
gt_ind = []
gt_ind_rolled = []
xval = np.linspace(0,1,target_t)

for i,n in enumerate(nda_vects[:]):
    cardiac_cycle_length = int(gt_len[i,:,0].sum())
    cycle_len.append(cardiac_cycle_length)
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
    gt_onehot = gt[i][:cardiac_cycle_length]
    gt_ind.append(ind)
    
    dir_axis=0
    gt_ed = ind[0]
    #assert cardiac_cycle_length>=gt_ed
    temp = n[:cardiac_cycle_length]
    
    dim = temp.shape[1:-1]
    #print(dim)
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    #print('centers: ',c.dtype)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    # direction
    directions = flow2direction_lambda(temp)[...,0]
    # direction mean
    directions = np.mean(directions,axis=(1,2,3))
    # direction ed aligned
    directions = np.roll(directions, -1*gt_ed)
    # direction interpolate to unique length
    f = interp1d(np.linspace(0,1,directions.shape[0]), directions, kind='linear')
    directions = f(xval)
    # direction min/max normalised between -1,1

    directions = minmax_lambda([directions,lower,upper])
    
    # magnitude/norm as mean
    norm_full = np.linalg.norm(temp, axis=-1)
    norm_nda = norm_full.mean(axis=(1,2,3))
    # norm ed aligned
    norm_nda = np.roll(norm_nda, -1*gt_ed)
    # interpolate to unique length
    #f = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')
    #norm_nda = f(xval)
    norm_nda = np.interp(xval, np.linspace(0,1,norm_nda.shape[0]), norm_nda)
    # norm min/max aligned to 0,1
    norm_nda = minmax_lambda([norm_nda,mid,upper])
    
    gt_onehot_rolled = np.roll(gt_onehot, -1*gt_ed)
    resize_factor = target_t/cardiac_cycle_length
    print(resize_factor.shape)
    gt_onehot_rolled = np.rint(gt_ind_rolled * resize_factor)
    gt_ind_rolled.append(gt_ind_rolled)
    
    norms.append(norm_nda)
    dirs.append(directions)
# Tue, 01 Mar 2022 09:00:18
# align norm an direction by the cardiac phase ED
# resample all 1D feature vectors to the same length
# this should help to validate the pre-defined rules, and if both together explains the cardiac phases
target_t = 40
norms = []
dirs= []
cycle_len=[]
gt_ind = []
gt_ind_rolled = []
xval = np.linspace(0,1,target_t)

for i,n in enumerate(nda_vects[:]):
    cardiac_cycle_length = int(gt_len[i,:,0].sum())
    cycle_len.append(cardiac_cycle_length)
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
    gt_onehot = gt[i][:cardiac_cycle_length]
    gt_ind.append(ind)
    
    dir_axis=0
    gt_ed = ind[0]
    #assert cardiac_cycle_length>=gt_ed
    temp = n[:cardiac_cycle_length]
    
    dim = temp.shape[1:-1]
    #print(dim)
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    #print('centers: ',c.dtype)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    # direction
    directions = flow2direction_lambda(temp)[...,0]
    # direction mean
    directions = np.mean(directions,axis=(1,2,3))
    # direction ed aligned
    directions = np.roll(directions, -1*gt_ed)
    # direction interpolate to unique length
    f = interp1d(np.linspace(0,1,directions.shape[0]), directions, kind='linear')
    directions = f(xval)
    # direction min/max normalised between -1,1

    directions = minmax_lambda([directions,lower,upper])
    
    # magnitude/norm as mean
    norm_full = np.linalg.norm(temp, axis=-1)
    norm_nda = norm_full.mean(axis=(1,2,3))
    # norm ed aligned
    norm_nda = np.roll(norm_nda, -1*gt_ed)
    # interpolate to unique length
    #f = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')
    #norm_nda = f(xval)
    norm_nda = np.interp(xval, np.linspace(0,1,norm_nda.shape[0]), norm_nda)
    # norm min/max aligned to 0,1
    norm_nda = minmax_lambda([norm_nda,mid,upper])
    
    gt_onehot_rolled = np.roll(gt_onehot, -1*gt_ed)
    resize_factor = target_t/cardiac_cycle_length
    print(resize_factor)
    gt_onehot_rolled = np.rint(gt_ind_rolled * resize_factor)
    gt_ind_rolled.append(gt_ind_rolled)
    
    norms.append(norm_nda)
    dirs.append(directions)
# Tue, 01 Mar 2022 09:00:34
# align norm an direction by the cardiac phase ED
# resample all 1D feature vectors to the same length
# this should help to validate the pre-defined rules, and if both together explains the cardiac phases
target_t = 40
norms = []
dirs= []
cycle_len=[]
gt_ind = []
gt_ind_rolled = []
xval = np.linspace(0,1,target_t)

for i,n in enumerate(nda_vects[:]):
    cardiac_cycle_length = int(gt_len[i,:,0].sum())
    cycle_len.append(cardiac_cycle_length)
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
    gt_onehot = gt[i][:cardiac_cycle_length]
    gt_ind.append(ind)
    
    dir_axis=0
    gt_ed = ind[0]
    #assert cardiac_cycle_length>=gt_ed
    temp = n[:cardiac_cycle_length]
    
    dim = temp.shape[1:-1]
    #print(dim)
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    #print('centers: ',c.dtype)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    # direction
    directions = flow2direction_lambda(temp)[...,0]
    # direction mean
    directions = np.mean(directions,axis=(1,2,3))
    # direction ed aligned
    directions = np.roll(directions, -1*gt_ed)
    # direction interpolate to unique length
    f = interp1d(np.linspace(0,1,directions.shape[0]), directions, kind='linear')
    directions = f(xval)
    # direction min/max normalised between -1,1

    directions = minmax_lambda([directions,lower,upper])
    
    # magnitude/norm as mean
    norm_full = np.linalg.norm(temp, axis=-1)
    norm_nda = norm_full.mean(axis=(1,2,3))
    # norm ed aligned
    norm_nda = np.roll(norm_nda, -1*gt_ed)
    # interpolate to unique length
    #f = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')
    #norm_nda = f(xval)
    norm_nda = np.interp(xval, np.linspace(0,1,norm_nda.shape[0]), norm_nda)
    # norm min/max aligned to 0,1
    norm_nda = minmax_lambda([norm_nda,mid,upper])
    
    gt_onehot_rolled = np.roll(gt_onehot, -1*gt_ed)
    resize_factor = target_t/cardiac_cycle_length
    print(resize_factor)
    gt_onehot_rolled = np.rint(np.array(gt_ind_rolled) * resize_factor)
    gt_ind_rolled.append(gt_ind_rolled)
    
    norms.append(norm_nda)
    dirs.append(directions)
# Tue, 01 Mar 2022 09:02:31
# align norm an direction by the cardiac phase ED
# resample all 1D feature vectors to the same length
# this should help to validate the pre-defined rules, and if both together explains the cardiac phases
target_t = 40
norms = []
dirs= []
cycle_len=[]
gt_ind = []
gt_ind_rolled = []
xval = np.linspace(0,1,target_t)

for i,n in enumerate(nda_vects[:]):
    cardiac_cycle_length = int(gt_len[i,:,0].sum())
    cycle_len.append(cardiac_cycle_length)
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
    gt_onehot = gt[i][:cardiac_cycle_length]
    gt_ind.append(ind)
    
    dir_axis=0
    gt_ed = ind[0]
    #assert cardiac_cycle_length>=gt_ed
    temp = n[:cardiac_cycle_length]
    
    dim = temp.shape[1:-1]
    #print(dim)
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    #print('centers: ',c.dtype)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    # direction
    directions = flow2direction_lambda(temp)[...,0]
    # direction mean
    directions = np.mean(directions,axis=(1,2,3))
    # direction ed aligned
    directions = np.roll(directions, -1*gt_ed)
    # direction interpolate to unique length
    f = interp1d(np.linspace(0,1,directions.shape[0]), directions, kind='linear')
    directions = f(xval)
    # direction min/max normalised between -1,1

    directions = minmax_lambda([directions,lower,upper])
    
    # magnitude/norm as mean
    norm_full = np.linalg.norm(temp, axis=-1)
    norm_nda = norm_full.mean(axis=(1,2,3))
    # norm ed aligned
    norm_nda = np.roll(norm_nda, -1*gt_ed)
    # interpolate to unique length
    #f = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')
    #norm_nda = f(xval)
    norm_nda = np.interp(xval, np.linspace(0,1,norm_nda.shape[0]), norm_nda)
    # norm min/max aligned to 0,1
    norm_nda = minmax_lambda([norm_nda,mid,upper])
    
    gt_onehot_rolled = np.roll(gt_onehot, -1*gt_ed)
    resize_factor = target_t/cardiac_cycle_length
    gt_onehot_rolled = np.argmax(gt_onehot_rolled,axis=0)
    gt_onehot_rolled = np.rint(gt_ind_rolled * resize_factor)
    gt_ind_rolled.append(gt_ind_rolled)
    
    norms.append(norm_nda)
    dirs.append(directions)
# Tue, 01 Mar 2022 09:02:54
# align norm an direction by the cardiac phase ED
# resample all 1D feature vectors to the same length
# this should help to validate the pre-defined rules, and if both together explains the cardiac phases
target_t = 40
norms = []
dirs= []
cycle_len=[]
gt_ind = []
gt_ind_rolled = []
xval = np.linspace(0,1,target_t)

for i,n in enumerate(nda_vects[:]):
    cardiac_cycle_length = int(gt_len[i,:,0].sum())
    cycle_len.append(cardiac_cycle_length)
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
    gt_onehot = gt[i][:cardiac_cycle_length]
    gt_ind.append(ind)
    
    dir_axis=0
    gt_ed = ind[0]
    #assert cardiac_cycle_length>=gt_ed
    temp = n[:cardiac_cycle_length]
    
    dim = temp.shape[1:-1]
    #print(dim)
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    #print('centers: ',c.dtype)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    # direction
    directions = flow2direction_lambda(temp)[...,0]
    # direction mean
    directions = np.mean(directions,axis=(1,2,3))
    # direction ed aligned
    directions = np.roll(directions, -1*gt_ed)
    # direction interpolate to unique length
    f = interp1d(np.linspace(0,1,directions.shape[0]), directions, kind='linear')
    directions = f(xval)
    # direction min/max normalised between -1,1

    directions = minmax_lambda([directions,lower,upper])
    
    # magnitude/norm as mean
    norm_full = np.linalg.norm(temp, axis=-1)
    norm_nda = norm_full.mean(axis=(1,2,3))
    # norm ed aligned
    norm_nda = np.roll(norm_nda, -1*gt_ed)
    # interpolate to unique length
    #f = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')
    #norm_nda = f(xval)
    norm_nda = np.interp(xval, np.linspace(0,1,norm_nda.shape[0]), norm_nda)
    # norm min/max aligned to 0,1
    norm_nda = minmax_lambda([norm_nda,mid,upper])
    
    gt_onehot_rolled = np.roll(gt_onehot, -1*gt_ed)
    resize_factor = target_t/cardiac_cycle_length
    gt_onehot_rolled = np.argmax(gt_onehot_rolled,axis=0)
    gt_onehot_rolled = np.rint(gt_ind_rolled * np.array(resize_factor))
    gt_ind_rolled.append(gt_ind_rolled)
    
    norms.append(norm_nda)
    dirs.append(directions)
# Tue, 01 Mar 2022 09:03:15
gt_ind[0]
# Tue, 01 Mar 2022 09:03:15
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 09:03:22
gt_ind[0] * 1.3
# Tue, 01 Mar 2022 09:03:22
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 09:04:14
# align norm an direction by the cardiac phase ED
# resample all 1D feature vectors to the same length
# this should help to validate the pre-defined rules, and if both together explains the cardiac phases
target_t = 40
norms = []
dirs= []
cycle_len=[]
gt_ind = []
gt_ind_rolled = []
xval = np.linspace(0,1,target_t)

for i,n in enumerate(nda_vects[:]):
    cardiac_cycle_length = int(gt_len[i,:,0].sum())
    cycle_len.append(cardiac_cycle_length)
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
    gt_onehot = gt[i][:cardiac_cycle_length]
    gt_ind.append(ind)
    
    dir_axis=0
    gt_ed = ind[0]
    #assert cardiac_cycle_length>=gt_ed
    temp = n[:cardiac_cycle_length]
    
    dim = temp.shape[1:-1]
    #print(dim)
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    #print('centers: ',c.dtype)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    # direction
    directions = flow2direction_lambda(temp)[...,0]
    # direction mean
    directions = np.mean(directions,axis=(1,2,3))
    # direction ed aligned
    directions = np.roll(directions, -1*gt_ed)
    # direction interpolate to unique length
    f = interp1d(np.linspace(0,1,directions.shape[0]), directions, kind='linear')
    directions = f(xval)
    # direction min/max normalised between -1,1

    directions = minmax_lambda([directions,lower,upper])
    
    # magnitude/norm as mean
    norm_full = np.linalg.norm(temp, axis=-1)
    norm_nda = norm_full.mean(axis=(1,2,3))
    # norm ed aligned
    norm_nda = np.roll(norm_nda, -1*gt_ed)
    # interpolate to unique length
    #f = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')
    #norm_nda = f(xval)
    norm_nda = np.interp(xval, np.linspace(0,1,norm_nda.shape[0]), norm_nda)
    # norm min/max aligned to 0,1
    norm_nda = minmax_lambda([norm_nda,mid,upper])
    
    gt_onehot_rolled = np.roll(gt_onehot, -1*gt_ed)
    resize_factor = target_t/cardiac_cycle_length
    gt_onehot_rolled = np.argmax(gt_onehot_rolled,axis=0)
    print(gt_ind_rolled)
    gt_onehot_rolled = np.rint(gt_ind_rolled * resize_factor)
    gt_ind_rolled.append(gt_ind_rolled)
    
    norms.append(norm_nda)
    dirs.append(directions)
# Tue, 01 Mar 2022 09:04:33
# align norm an direction by the cardiac phase ED
# resample all 1D feature vectors to the same length
# this should help to validate the pre-defined rules, and if both together explains the cardiac phases
target_t = 40
norms = []
dirs= []
cycle_len=[]
gt_ind = []
gt_ind_rolled = []
xval = np.linspace(0,1,target_t)

for i,n in enumerate(nda_vects[:]):
    cardiac_cycle_length = int(gt_len[i,:,0].sum())
    cycle_len.append(cardiac_cycle_length)
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
    gt_onehot = gt[i][:cardiac_cycle_length]
    gt_ind.append(ind)
    
    dir_axis=0
    gt_ed = ind[0]
    #assert cardiac_cycle_length>=gt_ed
    temp = n[:cardiac_cycle_length]
    
    dim = temp.shape[1:-1]
    #print(dim)
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    #print('centers: ',c.dtype)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    # direction
    directions = flow2direction_lambda(temp)[...,0]
    # direction mean
    directions = np.mean(directions,axis=(1,2,3))
    # direction ed aligned
    directions = np.roll(directions, -1*gt_ed)
    # direction interpolate to unique length
    f = interp1d(np.linspace(0,1,directions.shape[0]), directions, kind='linear')
    directions = f(xval)
    # direction min/max normalised between -1,1

    directions = minmax_lambda([directions,lower,upper])
    
    # magnitude/norm as mean
    norm_full = np.linalg.norm(temp, axis=-1)
    norm_nda = norm_full.mean(axis=(1,2,3))
    # norm ed aligned
    norm_nda = np.roll(norm_nda, -1*gt_ed)
    # interpolate to unique length
    #f = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')
    #norm_nda = f(xval)
    norm_nda = np.interp(xval, np.linspace(0,1,norm_nda.shape[0]), norm_nda)
    # norm min/max aligned to 0,1
    norm_nda = minmax_lambda([norm_nda,mid,upper])
    
    gt_onehot_rolled = np.roll(gt_onehot, -1*gt_ed)
    print(gt_ind_rolled)
    resize_factor = target_t/cardiac_cycle_length
    gt_onehot_rolled = np.argmax(gt_onehot_rolled,axis=0)
    print(gt_ind_rolled)
    gt_onehot_rolled = np.rint(gt_ind_rolled * resize_factor)
    gt_ind_rolled.append(gt_ind_rolled)
    
    norms.append(norm_nda)
    dirs.append(directions)
# Tue, 01 Mar 2022 09:04:49
# align norm an direction by the cardiac phase ED
# resample all 1D feature vectors to the same length
# this should help to validate the pre-defined rules, and if both together explains the cardiac phases
target_t = 40
norms = []
dirs= []
cycle_len=[]
gt_ind = []
gt_ind_rolled = []
xval = np.linspace(0,1,target_t)

for i,n in enumerate(nda_vects[:]):
    cardiac_cycle_length = int(gt_len[i,:,0].sum())
    cycle_len.append(cardiac_cycle_length)
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
    gt_onehot = gt[i][:cardiac_cycle_length]
    gt_ind.append(ind)
    
    dir_axis=0
    gt_ed = ind[0]
    #assert cardiac_cycle_length>=gt_ed
    temp = n[:cardiac_cycle_length]
    
    dim = temp.shape[1:-1]
    #print(dim)
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    #print('centers: ',c.dtype)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    # direction
    directions = flow2direction_lambda(temp)[...,0]
    # direction mean
    directions = np.mean(directions,axis=(1,2,3))
    # direction ed aligned
    directions = np.roll(directions, -1*gt_ed)
    # direction interpolate to unique length
    f = interp1d(np.linspace(0,1,directions.shape[0]), directions, kind='linear')
    directions = f(xval)
    # direction min/max normalised between -1,1

    directions = minmax_lambda([directions,lower,upper])
    
    # magnitude/norm as mean
    norm_full = np.linalg.norm(temp, axis=-1)
    norm_nda = norm_full.mean(axis=(1,2,3))
    # norm ed aligned
    norm_nda = np.roll(norm_nda, -1*gt_ed)
    # interpolate to unique length
    #f = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')
    #norm_nda = f(xval)
    norm_nda = np.interp(xval, np.linspace(0,1,norm_nda.shape[0]), norm_nda)
    # norm min/max aligned to 0,1
    norm_nda = minmax_lambda([norm_nda,mid,upper])
    
    print(gt_onehot)
    gt_onehot_rolled = np.roll(gt_onehot, -1*gt_ed)
    print(gt_ind_rolled)
    resize_factor = target_t/cardiac_cycle_length
    gt_onehot_rolled = np.argmax(gt_onehot_rolled,axis=0)
    print(gt_ind_rolled)
    gt_onehot_rolled = np.rint(gt_ind_rolled * resize_factor)
    gt_ind_rolled.append(gt_ind_rolled)
    
    norms.append(norm_nda)
    dirs.append(directions)
# Tue, 01 Mar 2022 09:05:54
# align norm an direction by the cardiac phase ED
# resample all 1D feature vectors to the same length
# this should help to validate the pre-defined rules, and if both together explains the cardiac phases
target_t = 40
norms = []
dirs= []
cycle_len=[]
gt_ind = []
gt_ind_rolled = []
xval = np.linspace(0,1,target_t)

for i,n in enumerate(nda_vects[:]):
    cardiac_cycle_length = int(gt_len[i,:,0].sum())
    cycle_len.append(cardiac_cycle_length)
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
    gt_onehot = gt[i][:cardiac_cycle_length]
    gt_ind.append(ind)
    
    dir_axis=0
    gt_ed = ind[0]
    #assert cardiac_cycle_length>=gt_ed
    temp = n[:cardiac_cycle_length]
    
    dim = temp.shape[1:-1]
    #print(dim)
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    #print('centers: ',c.dtype)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    # direction
    directions = flow2direction_lambda(temp)[...,0]
    # direction mean
    directions = np.mean(directions,axis=(1,2,3))
    # direction ed aligned
    directions = np.roll(directions, -1*gt_ed)
    # direction interpolate to unique length
    f = interp1d(np.linspace(0,1,directions.shape[0]), directions, kind='linear')
    directions = f(xval)
    # direction min/max normalised between -1,1

    directions = minmax_lambda([directions,lower,upper])
    
    # magnitude/norm as mean
    norm_full = np.linalg.norm(temp, axis=-1)
    norm_nda = norm_full.mean(axis=(1,2,3))
    # norm ed aligned
    norm_nda = np.roll(norm_nda, -1*gt_ed)
    # interpolate to unique length
    #f = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')
    #norm_nda = f(xval)
    norm_nda = np.interp(xval, np.linspace(0,1,norm_nda.shape[0]), norm_nda)
    # norm min/max aligned to 0,1
    norm_nda = minmax_lambda([norm_nda,mid,upper])
    
    print(gt_onehot.shape)
    gt_onehot_rolled = np.roll(gt_onehot, -1*gt_ed)
    print(gt_ind_rolled)
    resize_factor = target_t/cardiac_cycle_length
    gt_onehot_rolled = np.argmax(gt_onehot_rolled,axis=0)
    print(gt_ind_rolled)
    gt_onehot_rolled = np.rint(gt_ind_rolled * resize_factor)
    gt_ind_rolled.append(gt_ind_rolled)
    
    norms.append(norm_nda)
    dirs.append(directions)
# Tue, 01 Mar 2022 09:06:40
# align norm an direction by the cardiac phase ED
# resample all 1D feature vectors to the same length
# this should help to validate the pre-defined rules, and if both together explains the cardiac phases
target_t = 40
norms = []
dirs= []
cycle_len=[]
gt_ind = []
gt_ind_rolled = []
xval = np.linspace(0,1,target_t)

for i,n in enumerate(nda_vects[:]):
    cardiac_cycle_length = int(gt_len[i,:,0].sum())
    cycle_len.append(cardiac_cycle_length)
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
    gt_onehot = gt[i][:cardiac_cycle_length]
    gt_ind.append(ind)
    
    dir_axis=0
    gt_ed = ind[0]
    #assert cardiac_cycle_length>=gt_ed
    temp = n[:cardiac_cycle_length]
    
    dim = temp.shape[1:-1]
    #print(dim)
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    #print('centers: ',c.dtype)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    # direction
    directions = flow2direction_lambda(temp)[...,0]
    # direction mean
    directions = np.mean(directions,axis=(1,2,3))
    # direction ed aligned
    directions = np.roll(directions, -1*gt_ed)
    # direction interpolate to unique length
    f = interp1d(np.linspace(0,1,directions.shape[0]), directions, kind='linear')
    directions = f(xval)
    # direction min/max normalised between -1,1

    directions = minmax_lambda([directions,lower,upper])
    
    # magnitude/norm as mean
    norm_full = np.linalg.norm(temp, axis=-1)
    norm_nda = norm_full.mean(axis=(1,2,3))
    # norm ed aligned
    norm_nda = np.roll(norm_nda, -1*gt_ed)
    # interpolate to unique length
    #f = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')
    #norm_nda = f(xval)
    norm_nda = np.interp(xval, np.linspace(0,1,norm_nda.shape[0]), norm_nda)
    # norm min/max aligned to 0,1
    norm_nda = minmax_lambda([norm_nda,mid,upper])
    
    print(gt_onehot.shape)
    gt_onehot_rolled = np.roll(gt_onehot, -1*gt_ed)
    print(gt_onehot_rolled)
    resize_factor = target_t/cardiac_cycle_length
    gt_onehot_rolled = np.argmax(gt_onehot_rolled,axis=0)
    print(gt_ind_rolled)
    gt_onehot_rolled = np.rint(gt_ind_rolled * resize_factor)
    gt_ind_rolled.append(gt_ind_rolled)
    
    norms.append(norm_nda)
    dirs.append(directions)
# Tue, 01 Mar 2022 09:06:51
# align norm an direction by the cardiac phase ED
# resample all 1D feature vectors to the same length
# this should help to validate the pre-defined rules, and if both together explains the cardiac phases
target_t = 40
norms = []
dirs= []
cycle_len=[]
gt_ind = []
gt_ind_rolled = []
xval = np.linspace(0,1,target_t)

for i,n in enumerate(nda_vects[:]):
    cardiac_cycle_length = int(gt_len[i,:,0].sum())
    cycle_len.append(cardiac_cycle_length)
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
    gt_onehot = gt[i][:cardiac_cycle_length]
    gt_ind.append(ind)
    
    dir_axis=0
    gt_ed = ind[0]
    #assert cardiac_cycle_length>=gt_ed
    temp = n[:cardiac_cycle_length]
    
    dim = temp.shape[1:-1]
    #print(dim)
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    #print('centers: ',c.dtype)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    # direction
    directions = flow2direction_lambda(temp)[...,0]
    # direction mean
    directions = np.mean(directions,axis=(1,2,3))
    # direction ed aligned
    directions = np.roll(directions, -1*gt_ed)
    # direction interpolate to unique length
    f = interp1d(np.linspace(0,1,directions.shape[0]), directions, kind='linear')
    directions = f(xval)
    # direction min/max normalised between -1,1

    directions = minmax_lambda([directions,lower,upper])
    
    # magnitude/norm as mean
    norm_full = np.linalg.norm(temp, axis=-1)
    norm_nda = norm_full.mean(axis=(1,2,3))
    # norm ed aligned
    norm_nda = np.roll(norm_nda, -1*gt_ed)
    # interpolate to unique length
    #f = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')
    #norm_nda = f(xval)
    norm_nda = np.interp(xval, np.linspace(0,1,norm_nda.shape[0]), norm_nda)
    # norm min/max aligned to 0,1
    norm_nda = minmax_lambda([norm_nda,mid,upper])
    
    print(gt_onehot.shape)
    gt_onehot_rolled = np.roll(gt_onehot, -1*gt_ed)
    print(gt_onehot_rolled.shape)
    resize_factor = target_t/cardiac_cycle_length
    gt_onehot_rolled = np.argmax(gt_onehot_rolled,axis=0)
    print(gt_ind_rolled)
    gt_onehot_rolled = np.rint(gt_ind_rolled * resize_factor)
    gt_ind_rolled.append(gt_ind_rolled)
    
    norms.append(norm_nda)
    dirs.append(directions)
# Tue, 01 Mar 2022 09:07:28
# align norm an direction by the cardiac phase ED
# resample all 1D feature vectors to the same length
# this should help to validate the pre-defined rules, and if both together explains the cardiac phases
target_t = 40
norms = []
dirs= []
cycle_len=[]
gt_ind = []
gt_ind_rolled = []
xval = np.linspace(0,1,target_t)

for i,n in enumerate(nda_vects[:]):
    cardiac_cycle_length = int(gt_len[i,:,0].sum())
    cycle_len.append(cardiac_cycle_length)
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
    gt_onehot = gt[i][:cardiac_cycle_length]
    gt_ind.append(ind)
    
    dir_axis=0
    gt_ed = ind[0]
    #assert cardiac_cycle_length>=gt_ed
    temp = n[:cardiac_cycle_length]
    
    dim = temp.shape[1:-1]
    #print(dim)
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    #print('centers: ',c.dtype)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    # direction
    directions = flow2direction_lambda(temp)[...,0]
    # direction mean
    directions = np.mean(directions,axis=(1,2,3))
    # direction ed aligned
    directions = np.roll(directions, -1*gt_ed)
    # direction interpolate to unique length
    f = interp1d(np.linspace(0,1,directions.shape[0]), directions, kind='linear')
    directions = f(xval)
    # direction min/max normalised between -1,1

    directions = minmax_lambda([directions,lower,upper])
    
    # magnitude/norm as mean
    norm_full = np.linalg.norm(temp, axis=-1)
    norm_nda = norm_full.mean(axis=(1,2,3))
    # norm ed aligned
    norm_nda = np.roll(norm_nda, -1*gt_ed)
    # interpolate to unique length
    #f = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')
    #norm_nda = f(xval)
    norm_nda = np.interp(xval, np.linspace(0,1,norm_nda.shape[0]), norm_nda)
    # norm min/max aligned to 0,1
    norm_nda = minmax_lambda([norm_nda,mid,upper])
    
    print(gt_onehot.shape)
    gt_onehot_rolled = np.roll(gt_onehot, -1*gt_ed)
    print(gt_onehot_rolled.shape)
    resize_factor = target_t/cardiac_cycle_length
    gt_onehot_rolled = np.argmax(gt_onehot_rolled,axis=0)
    print(gt_onehot_rolled)
    gt_onehot_rolled = np.rint(gt_onehot_rolled * resize_factor)
    gt_ind_rolled.append(gt_onehot_rolled)
    
    norms.append(norm_nda)
    dirs.append(directions)
# Tue, 01 Mar 2022 09:09:35
# align norm an direction by the cardiac phase ED
# resample all 1D feature vectors to the same length
# this should help to validate the pre-defined rules, and if both together explains the cardiac phases
target_t = 40
norms = []
dirs= []
cycle_len=[]
gt_ind = []
gt_ind_rolled = []
xval = np.linspace(0,1,target_t)

for i,n in enumerate(nda_vects[:]):
    cardiac_cycle_length = int(gt_len[i,:,0].sum())
    cycle_len.append(cardiac_cycle_length)
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
    gt_onehot = gt[i][:cardiac_cycle_length]
    gt_ind.append(ind)
    
    dir_axis=0
    gt_ed = ind[0]
    #assert cardiac_cycle_length>=gt_ed
    temp = n[:cardiac_cycle_length]
    
    dim = temp.shape[1:-1]
    #print(dim)
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    #print('centers: ',c.dtype)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    # direction
    directions = flow2direction_lambda(temp)[...,0]
    # direction mean
    directions = np.mean(directions,axis=(1,2,3))
    # direction ed aligned
    directions = np.roll(directions, -1*gt_ed)
    # direction interpolate to unique length
    f = interp1d(np.linspace(0,1,directions.shape[0]), directions, kind='linear')
    directions = f(xval)
    # direction min/max normalised between -1,1

    directions = minmax_lambda([directions,lower,upper])
    
    # magnitude/norm as mean
    norm_full = np.linalg.norm(temp, axis=-1)
    norm_nda = norm_full.mean(axis=(1,2,3))
    # norm ed aligned
    norm_nda = np.roll(norm_nda, -1*gt_ed)
    # interpolate to unique length
    #f = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')
    #norm_nda = f(xval)
    norm_nda = np.interp(xval, np.linspace(0,1,norm_nda.shape[0]), norm_nda)
    # norm min/max aligned to 0,1
    norm_nda = minmax_lambda([norm_nda,mid,upper])
    
    gt_onehot_rolled = np.roll(gt_onehot, -1*gt_ed)
    print(cardiac_cycle_length, gt_onehot_rolled.shape)
    resize_factor = target_t/cardiac_cycle_length
    gt_onehot_rolled = np.argmax(gt_onehot_rolled,axis=0)
    
    gt_onehot_rolled = np.clip(np.rint(gt_onehot_rolled * resize_factor),a_min=0,a_max=cardiac_cycle_length-1)
    gt_ind_rolled.append(gt_onehot_rolled)
    
    norms.append(norm_nda)
    dirs.append(directions)
# Tue, 01 Mar 2022 09:10:29
# align norm an direction by the cardiac phase ED
# resample all 1D feature vectors to the same length
# this should help to validate the pre-defined rules, and if both together explains the cardiac phases
target_t = 40
norms = []
dirs= []
cycle_len=[]
gt_ind = []
gt_ind_rolled = []
xval = np.linspace(0,1,target_t)

for i,n in enumerate(nda_vects[:]):
    cardiac_cycle_length = int(gt_len[i,:,0].sum())
    cycle_len.append(cardiac_cycle_length)
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
    gt_onehot = gt[i][:cardiac_cycle_length]
    gt_ind.append(ind)
    
    dir_axis=0
    gt_ed = ind[0]
    #assert cardiac_cycle_length>=gt_ed
    temp = n[:cardiac_cycle_length]
    
    dim = temp.shape[1:-1]
    #print(dim)
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    #print('centers: ',c.dtype)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    # direction
    directions = flow2direction_lambda(temp)[...,0]
    # direction mean
    directions = np.mean(directions,axis=(1,2,3))
    # direction ed aligned
    directions = np.roll(directions, -1*gt_ed)
    # direction interpolate to unique length
    f = interp1d(np.linspace(0,1,directions.shape[0]), directions, kind='linear')
    directions = f(xval)
    # direction min/max normalised between -1,1

    directions = minmax_lambda([directions,lower,upper])
    
    # magnitude/norm as mean
    norm_full = np.linalg.norm(temp, axis=-1)
    norm_nda = norm_full.mean(axis=(1,2,3))
    # norm ed aligned
    norm_nda = np.roll(norm_nda, -1*gt_ed)
    # interpolate to unique length
    #f = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')
    #norm_nda = f(xval)
    norm_nda = np.interp(xval, np.linspace(0,1,norm_nda.shape[0]), norm_nda)
    # norm min/max aligned to 0,1
    norm_nda = minmax_lambda([norm_nda,mid,upper])
    
    gt_onehot_rolled = np.roll(gt_onehot, -1*gt_ed)
    resize_factor = target_t/cardiac_cycle_length
    gt_onehot_rolled = np.argmax(gt_onehot_rolled,axis=0)
    
    gt_onehot_rolled = np.clip(np.rint(gt_onehot_rolled * resize_factor),a_min=0,a_max=cardiac_cycle_length-1)
    gt_ind_rolled.append(gt_onehot_rolled)
    
    norms.append(norm_nda)
    dirs.append(directions)
# Tue, 01 Mar 2022 09:10:41
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 09:11:06
# get the mean values
gt_ind_scaled = np.stack(gt_ind_rolled, axis=0)
gt_ind = np.stack(gt_ind,axis=0)
cycle_len = np.stack(cycle_len, axis=0)
norms = np.stack(norms, axis=0)
dirs = np.stack(dirs, axis=0)
norms_m = [norms.mean(axis=0)]
dirs_m=[dirs.mean(axis=0)]
# Tue, 01 Mar 2022 09:11:06
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 09:13:00
# plot the distribution of the aligned and scaled gt indicies
_ = pd.DataFrame(gt_ind_scaled).plot(kind='box')
# Tue, 01 Mar 2022 09:13:00
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 09:15:43
# align norm an direction by the cardiac phase ED
# resample all 1D feature vectors to the same length
# this should help to validate the pre-defined rules, and if both together explains the cardiac phases
target_t = 40
norms = []
dirs= []
cycle_len=[]
gt_ind = []
gt_ind_rolled = []
xval = np.linspace(0,1,target_t)

for i,n in enumerate(nda_vects[:]):
    cardiac_cycle_length = int(gt_len[i,:,0].sum())
    cycle_len.append(cardiac_cycle_length)
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
    gt_onehot = gt[i][:cardiac_cycle_length]
    gt_ind.append(ind)
    
    dir_axis=0
    gt_ed = ind[0]
    #assert cardiac_cycle_length>=gt_ed
    temp = n[:cardiac_cycle_length]
    
    dim = temp.shape[1:-1]
    #print(dim)
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    #print('centers: ',c.dtype)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    # direction
    directions = flow2direction_lambda(temp)[...,0]
    # direction mean
    directions = np.mean(directions,axis=(1,2,3))
    # direction ed aligned
    directions = np.roll(directions, -1*gt_ed)
    # direction interpolate to unique length
    f = interp1d(np.linspace(0,1,directions.shape[0]), directions, kind='linear')
    directions = f(xval)
    # direction min/max normalised between -1,1

    directions = minmax_lambda([directions,lower,upper])
    
    # magnitude/norm as mean
    norm_full = np.linalg.norm(temp, axis=-1)
    norm_nda = norm_full.mean(axis=(1,2,3))
    # norm ed aligned
    norm_nda = np.roll(norm_nda, -1*gt_ed)
    # interpolate to unique length
    #f = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')
    #norm_nda = f(xval)
    norm_nda = np.interp(xval, np.linspace(0,1,norm_nda.shape[0]), norm_nda)
    # norm min/max aligned to 0,1
    norm_nda = minmax_lambda([norm_nda,mid,upper])
    
    gt_onehot_rolled = np.roll(gt_onehot, -1*gt_ed)
    resize_factor = target_t/cardiac_cycle_length
    gt_onehot_rolled = np.argmax(gt_onehot_rolled,axis=0)
    print(gt_onehot_rolled)
    gt_onehot_rolled = np.clip(np.rint(gt_onehot_rolled * resize_factor),a_min=0,a_max=cardiac_cycle_length-1)
    print(gt_onehot_rolled)
    gt_ind_rolled.append(gt_onehot_rolled)
    
    norms.append(norm_nda)
    dirs.append(directions)
# Tue, 01 Mar 2022 09:19:22
# align norm an direction by the cardiac phase ED
# resample all 1D feature vectors to the same length
# this should help to validate the pre-defined rules, and if both together explains the cardiac phases
target_t = 40
norms = []
dirs= []
cycle_len=[]
gt_ind = []
gt_ind_rolled = []
xval = np.linspace(0,1,target_t)

for i,n in enumerate(nda_vects[:]):
    cardiac_cycle_length = int(gt_len[i,:,0].sum())
    cycle_len.append(cardiac_cycle_length)
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
    gt_onehot = gt[i][:cardiac_cycle_length]
    gt_ind.append(ind)
    
    dir_axis=0
    gt_ed = ind[0]
    #assert cardiac_cycle_length>=gt_ed
    temp = n[:cardiac_cycle_length]
    
    dim = temp.shape[1:-1]
    #print(dim)
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    #print('centers: ',c.dtype)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    # direction
    directions = flow2direction_lambda(temp)[...,0]
    # direction mean
    directions = np.mean(directions,axis=(1,2,3))
    # direction ed aligned
    directions = np.roll(directions, -1*gt_ed)
    # direction interpolate to unique length
    f = interp1d(np.linspace(0,1,directions.shape[0]), directions, kind='linear')
    directions = f(xval)
    # direction min/max normalised between -1,1

    directions = minmax_lambda([directions,lower,upper])
    
    # magnitude/norm as mean
    norm_full = np.linalg.norm(temp, axis=-1)
    norm_nda = norm_full.mean(axis=(1,2,3))
    # norm ed aligned
    norm_nda = np.roll(norm_nda, -1*gt_ed)
    # interpolate to unique length
    #f = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')
    #norm_nda = f(xval)
    norm_nda = np.interp(xval, np.linspace(0,1,norm_nda.shape[0]), norm_nda)
    # norm min/max aligned to 0,1
    norm_nda = minmax_lambda([norm_nda,mid,upper])
    
    gt_onehot_rolled = np.roll(gt_onehot.T, -1*gt_ed)
    resize_factor = target_t/cardiac_cycle_length
    gt_onehot_rolled = np.argmax(gt_onehot_rolled,axis=0)
    print(gt_onehot_rolled)
    gt_onehot_rolled = np.clip(np.rint(gt_onehot_rolled * resize_factor),a_min=0,a_max=cardiac_cycle_length-1)
    print(gt_onehot_rolled)
    gt_ind_rolled.append(gt_onehot_rolled)
    
    norms.append(norm_nda)
    dirs.append(directions)
# Tue, 01 Mar 2022 09:20:28
# align norm an direction by the cardiac phase ED
# resample all 1D feature vectors to the same length
# this should help to validate the pre-defined rules, and if both together explains the cardiac phases
target_t = 40
norms = []
dirs= []
cycle_len=[]
gt_ind = []
gt_ind_rolled = []
xval = np.linspace(0,1,target_t)

for i,n in enumerate(nda_vects[:]):
    cardiac_cycle_length = int(gt_len[i,:,0].sum())
    cycle_len.append(cardiac_cycle_length)
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
    gt_onehot = gt[i][:cardiac_cycle_length]
    gt_ind.append(ind)
    
    dir_axis=0
    gt_ed = ind[0]
    #assert cardiac_cycle_length>=gt_ed
    temp = n[:cardiac_cycle_length]
    
    dim = temp.shape[1:-1]
    #print(dim)
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    #print('centers: ',c.dtype)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    # direction
    directions = flow2direction_lambda(temp)[...,0]
    # direction mean
    directions = np.mean(directions,axis=(1,2,3))
    # direction ed aligned
    directions = np.roll(directions, -1*gt_ed)
    # direction interpolate to unique length
    f = interp1d(np.linspace(0,1,directions.shape[0]), directions, kind='linear')
    directions = f(xval)
    # direction min/max normalised between -1,1

    directions = minmax_lambda([directions,lower,upper])
    
    # magnitude/norm as mean
    norm_full = np.linalg.norm(temp, axis=-1)
    norm_nda = norm_full.mean(axis=(1,2,3))
    # norm ed aligned
    norm_nda = np.roll(norm_nda, -1*gt_ed)
    # interpolate to unique length
    #f = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')
    #norm_nda = f(xval)
    norm_nda = np.interp(xval, np.linspace(0,1,norm_nda.shape[0]), norm_nda)
    # norm min/max aligned to 0,1
    norm_nda = minmax_lambda([norm_nda,mid,upper])
    
    plt.imshow(gt_onehot);plt.show()
    gt_onehot_rolled = np.roll(gt_onehot, -1*gt_ed)
    plt.imshow(gt_onehot_rolled);plt.show()
    resize_factor = target_t/cardiac_cycle_length
    gt_onehot_rolled = np.argmax(gt_onehot_rolled,axis=0)
    print(gt_onehot_rolled)
    gt_onehot_rolled = np.clip(np.rint(gt_onehot_rolled * resize_factor),a_min=0,a_max=cardiac_cycle_length-1)
    print(gt_onehot_rolled)
    gt_ind_rolled.append(gt_onehot_rolled)
    
    norms.append(norm_nda)
    dirs.append(directions)
# Tue, 01 Mar 2022 09:20:45
# align norm an direction by the cardiac phase ED
# resample all 1D feature vectors to the same length
# this should help to validate the pre-defined rules, and if both together explains the cardiac phases
target_t = 40
norms = []
dirs= []
cycle_len=[]
gt_ind = []
gt_ind_rolled = []
xval = np.linspace(0,1,target_t)

for i,n in enumerate(nda_vects[:]):
    cardiac_cycle_length = int(gt_len[i,:,0].sum())
    cycle_len.append(cardiac_cycle_length)
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
    gt_onehot = gt[i][:cardiac_cycle_length]
    gt_ind.append(ind)
    
    dir_axis=0
    gt_ed = ind[0]
    #assert cardiac_cycle_length>=gt_ed
    temp = n[:cardiac_cycle_length]
    
    dim = temp.shape[1:-1]
    #print(dim)
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    #print('centers: ',c.dtype)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    # direction
    directions = flow2direction_lambda(temp)[...,0]
    # direction mean
    directions = np.mean(directions,axis=(1,2,3))
    # direction ed aligned
    directions = np.roll(directions, -1*gt_ed)
    # direction interpolate to unique length
    f = interp1d(np.linspace(0,1,directions.shape[0]), directions, kind='linear')
    directions = f(xval)
    # direction min/max normalised between -1,1

    directions = minmax_lambda([directions,lower,upper])
    
    # magnitude/norm as mean
    norm_full = np.linalg.norm(temp, axis=-1)
    norm_nda = norm_full.mean(axis=(1,2,3))
    # norm ed aligned
    norm_nda = np.roll(norm_nda, -1*gt_ed)
    # interpolate to unique length
    #f = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')
    #norm_nda = f(xval)
    norm_nda = np.interp(xval, np.linspace(0,1,norm_nda.shape[0]), norm_nda)
    # norm min/max aligned to 0,1
    norm_nda = minmax_lambda([norm_nda,mid,upper])
    
    plt.imshow(gt_onehot.T);plt.show()
    gt_onehot_rolled = np.roll(gt_onehot, -1*gt_ed)
    plt.imshow(gt_onehot_rolled.T);plt.show()
    resize_factor = target_t/cardiac_cycle_length
    gt_onehot_rolled = np.argmax(gt_onehot_rolled,axis=0)
    print(gt_onehot_rolled)
    gt_onehot_rolled = np.clip(np.rint(gt_onehot_rolled * resize_factor),a_min=0,a_max=cardiac_cycle_length-1)
    print(gt_onehot_rolled)
    gt_ind_rolled.append(gt_onehot_rolled)
    
    norms.append(norm_nda)
    dirs.append(directions)
# Tue, 01 Mar 2022 09:21:16
# align norm an direction by the cardiac phase ED
# resample all 1D feature vectors to the same length
# this should help to validate the pre-defined rules, and if both together explains the cardiac phases
target_t = 40
norms = []
dirs= []
cycle_len=[]
gt_ind = []
gt_ind_rolled = []
xval = np.linspace(0,1,target_t)

for i,n in enumerate(nda_vects[:]):
    cardiac_cycle_length = int(gt_len[i,:,0].sum())
    cycle_len.append(cardiac_cycle_length)
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
    gt_onehot = gt[i][:cardiac_cycle_length]
    gt_ind.append(ind)
    
    dir_axis=0
    gt_ed = ind[0]
    #assert cardiac_cycle_length>=gt_ed
    temp = n[:cardiac_cycle_length]
    
    dim = temp.shape[1:-1]
    #print(dim)
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    #print('centers: ',c.dtype)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    # direction
    directions = flow2direction_lambda(temp)[...,0]
    # direction mean
    directions = np.mean(directions,axis=(1,2,3))
    # direction ed aligned
    directions = np.roll(directions, -1*gt_ed)
    # direction interpolate to unique length
    f = interp1d(np.linspace(0,1,directions.shape[0]), directions, kind='linear')
    directions = f(xval)
    # direction min/max normalised between -1,1

    directions = minmax_lambda([directions,lower,upper])
    
    # magnitude/norm as mean
    norm_full = np.linalg.norm(temp, axis=-1)
    norm_nda = norm_full.mean(axis=(1,2,3))
    # norm ed aligned
    norm_nda = np.roll(norm_nda, -1*gt_ed)
    # interpolate to unique length
    #f = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')
    #norm_nda = f(xval)
    norm_nda = np.interp(xval, np.linspace(0,1,norm_nda.shape[0]), norm_nda)
    # norm min/max aligned to 0,1
    norm_nda = minmax_lambda([norm_nda,mid,upper])
    
    plt.imshow(gt_onehot.T);plt.show()
    print(gt_ed)
    gt_onehot_rolled = np.roll(gt_onehot, -1*gt_ed)
    plt.imshow(gt_onehot_rolled.T);plt.show()
    resize_factor = target_t/cardiac_cycle_length
    gt_onehot_rolled = np.argmax(gt_onehot_rolled,axis=0)
    print(gt_onehot_rolled)
    gt_onehot_rolled = np.clip(np.rint(gt_onehot_rolled * resize_factor),a_min=0,a_max=cardiac_cycle_length-1)
    print(gt_onehot_rolled)
    gt_ind_rolled.append(gt_onehot_rolled)
    
    norms.append(norm_nda)
    dirs.append(directions)
# Tue, 01 Mar 2022 09:22:44
# align norm an direction by the cardiac phase ED
# resample all 1D feature vectors to the same length
# this should help to validate the pre-defined rules, and if both together explains the cardiac phases
target_t = 40
norms = []
dirs= []
cycle_len=[]
gt_ind = []
gt_ind_rolled = []
xval = np.linspace(0,1,target_t)

for i,n in enumerate(nda_vects[:]):
    cardiac_cycle_length = int(gt_len[i,:,0].sum())
    cycle_len.append(cardiac_cycle_length)
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
    gt_onehot = gt[i][:cardiac_cycle_length]
    gt_ind.append(ind)
    
    dir_axis=0
    gt_ed = ind[0]
    #assert cardiac_cycle_length>=gt_ed
    temp = n[:cardiac_cycle_length]
    
    dim = temp.shape[1:-1]
    #print(dim)
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    #print('centers: ',c.dtype)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    # direction
    directions = flow2direction_lambda(temp)[...,0]
    # direction mean
    directions = np.mean(directions,axis=(1,2,3))
    # direction ed aligned
    directions = np.roll(directions, -1*gt_ed)
    # direction interpolate to unique length
    f = interp1d(np.linspace(0,1,directions.shape[0]), directions, kind='linear')
    directions = f(xval)
    # direction min/max normalised between -1,1

    directions = minmax_lambda([directions,lower,upper])
    
    # magnitude/norm as mean
    norm_full = np.linalg.norm(temp, axis=-1)
    norm_nda = norm_full.mean(axis=(1,2,3))
    # norm ed aligned
    norm_nda = np.roll(norm_nda, -1*gt_ed)
    # interpolate to unique length
    #f = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')
    #norm_nda = f(xval)
    norm_nda = np.interp(xval, np.linspace(0,1,norm_nda.shape[0]), norm_nda)
    # norm min/max aligned to 0,1
    norm_nda = minmax_lambda([norm_nda,mid,upper])
    
    plt.imshow(gt_onehot.T);plt.show()
    print(gt_ed)
    gt_onehot_rolled = np.roll(gt_onehot, -1*gt_ed, axis=0)
    plt.imshow(gt_onehot_rolled.T);plt.show()
    resize_factor = target_t/cardiac_cycle_length
    gt_onehot_rolled = np.argmax(gt_onehot_rolled,axis=0)
    print(gt_onehot_rolled)
    gt_onehot_rolled = np.clip(np.rint(gt_onehot_rolled * resize_factor),a_min=0,a_max=cardiac_cycle_length-1)
    print(gt_onehot_rolled)
    gt_ind_rolled.append(gt_onehot_rolled)
    
    norms.append(norm_nda)
    dirs.append(directions)
# Tue, 01 Mar 2022 09:23:12
# align norm an direction by the cardiac phase ED
# resample all 1D feature vectors to the same length
# this should help to validate the pre-defined rules, and if both together explains the cardiac phases
target_t = 40
norms = []
dirs= []
cycle_len=[]
gt_ind = []
gt_ind_rolled = []
xval = np.linspace(0,1,target_t)

for i,n in enumerate(nda_vects[:]):
    cardiac_cycle_length = int(gt_len[i,:,0].sum())
    cycle_len.append(cardiac_cycle_length)
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
    gt_onehot = gt[i][:cardiac_cycle_length]
    gt_ind.append(ind)
    
    dir_axis=0
    gt_ed = ind[0]
    #assert cardiac_cycle_length>=gt_ed
    temp = n[:cardiac_cycle_length]
    
    dim = temp.shape[1:-1]
    #print(dim)
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    #print('centers: ',c.dtype)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    # direction
    directions = flow2direction_lambda(temp)[...,0]
    # direction mean
    directions = np.mean(directions,axis=(1,2,3))
    # direction ed aligned
    directions = np.roll(directions, -1*gt_ed)
    # direction interpolate to unique length
    f = interp1d(np.linspace(0,1,directions.shape[0]), directions, kind='linear')
    directions = f(xval)
    # direction min/max normalised between -1,1

    directions = minmax_lambda([directions,lower,upper])
    
    # magnitude/norm as mean
    norm_full = np.linalg.norm(temp, axis=-1)
    norm_nda = norm_full.mean(axis=(1,2,3))
    # norm ed aligned
    norm_nda = np.roll(norm_nda, -1*gt_ed)
    # interpolate to unique length
    #f = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')
    #norm_nda = f(xval)
    norm_nda = np.interp(xval, np.linspace(0,1,norm_nda.shape[0]), norm_nda)
    # norm min/max aligned to 0,1
    norm_nda = minmax_lambda([norm_nda,mid,upper])
    
    #plt.imshow(gt_onehot.T);plt.show()
    #print(gt_ed)
    gt_onehot_rolled = np.roll(gt_onehot, -1*gt_ed, axis=0)
    #plt.imshow(gt_onehot_rolled.T);plt.show()
    resize_factor = target_t/cardiac_cycle_length
    gt_onehot_rolled = np.argmax(gt_onehot_rolled,axis=0)
    #print(gt_onehot_rolled)
    gt_onehot_rolled = np.clip(np.rint(gt_onehot_rolled * resize_factor),a_min=0,a_max=cardiac_cycle_length-1)
    #print(gt_onehot_rolled)
    gt_ind_rolled.append(gt_onehot_rolled)
    
    norms.append(norm_nda)
    dirs.append(directions)
# Tue, 01 Mar 2022 09:23:23
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 09:23:25
# get the mean values
gt_ind_scaled = np.stack(gt_ind_rolled, axis=0)
gt_ind = np.stack(gt_ind,axis=0)
cycle_len = np.stack(cycle_len, axis=0)
norms = np.stack(norms, axis=0)
dirs = np.stack(dirs, axis=0)
norms_m = [norms.mean(axis=0)]
dirs_m=[dirs.mean(axis=0)]
# Tue, 01 Mar 2022 09:23:25
# plot the distribution of the aligned and scaled gt indicies
_ = pd.DataFrame(gt_ind_scaled).plot(kind='box')
# Tue, 01 Mar 2022 09:23:25
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 09:23:26
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 09:26:05
# seaborn
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax.margins(0,0)
for n in dirs:
    _ = ax.plot(n, alpha=0.5, zorder=0)

df = pd.DataFrame(dirs).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='blue', label=r'$\alpha_t$',ci='sd',err_style='bars',zorder=2,legend=False,ax=ax)
#_ = ax.set_title('Mean direction +/- SD - aligned at ED')
_ = ax.set_xticks(gt_ind_scaled.mean())
_ = ax.set_xlabel('Time (t) - linear interpolated to 40 frames')
_ = ax.set_ylabel(r'Angle $\alpha_t$ of $\phi_t$')
_ = ax.legend(loc='upper left')
ax2 = ax.twinx()
ax2.margins(0,0)
df = pd.DataFrame(norms).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='black', label=r'|$\phi_t$|',ci='sd', linestyle='dashed',err_style='bars',zorder=2,legend=False,ax=ax2)
_ = ax2.set_ylabel(r'Norm |$\phi_t$|')

_ = ax2.legend(loc='upper right')
# Tue, 01 Mar 2022 09:26:36
# seaborn
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax.margins(0,0)
for n in dirs:
    _ = ax.plot(n, alpha=0.5, zorder=0)

df = pd.DataFrame(dirs).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='blue', label=r'$\alpha_t$',ci='sd',err_style='bars',zorder=2,legend=False,ax=ax)
#_ = ax.set_title('Mean direction +/- SD - aligned at ED')
_ = ax.set_xticks(np.rint(gt_ind_scaled.mean()))
_ = ax.set_xlabel('Time (t) - linear interpolated to 40 frames')
_ = ax.set_ylabel(r'Angle $\alpha_t$ of $\phi_t$')
_ = ax.legend(loc='upper left')
ax2 = ax.twinx()
ax2.margins(0,0)
df = pd.DataFrame(norms).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='black', label=r'|$\phi_t$|',ci='sd', linestyle='dashed',err_style='bars',zorder=2,legend=False,ax=ax2)
_ = ax2.set_ylabel(r'Norm |$\phi_t$|')

_ = ax2.legend(loc='upper right')
# Tue, 01 Mar 2022 09:28:08
# seaborn
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax.margins(0,0)
for n in dirs:
    _ = ax.plot(n, alpha=0.5, zorder=0)

df = pd.DataFrame(dirs).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='blue', label=r'$\alpha_t$',ci='sd',err_style='bars',zorder=2,legend=False,ax=ax)
#_ = ax.set_title('Mean direction +/- SD - aligned at ED')
 _ = ax.set_xticks(np.rint(gt_ind_scaled.mean().values), minor=False)
_ = ax.set_xlabel('Time (t) - linear interpolated to 40 frames')
_ = ax.set_ylabel(r'Angle $\alpha_t$ of $\phi_t$')
_ = ax.legend(loc='upper left')
ax2 = ax.twinx()
ax2.margins(0,0)
df = pd.DataFrame(norms).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='black', label=r'|$\phi_t$|',ci='sd', linestyle='dashed',err_style='bars',zorder=2,legend=False,ax=ax2)
_ = ax2.set_ylabel(r'Norm |$\phi_t$|')

_ = ax2.legend(loc='upper right')
# Tue, 01 Mar 2022 09:28:08
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 09:28:32
# seaborn
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax.margins(0,0)
for n in dirs:
    _ = ax.plot(n, alpha=0.5, zorder=0)

df = pd.DataFrame(dirs).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='blue', label=r'$\alpha_t$',ci='sd',err_style='bars',zorder=2,legend=False,ax=ax)
#_ = ax.set_title('Mean direction +/- SD - aligned at ED')
_ = ax.set_xticks(np.rint(gt_ind_scaled.mean().values), minor=False)
_ = ax.set_xlabel('Time (t) - linear interpolated to 40 frames')
_ = ax.set_ylabel(r'Angle $\alpha_t$ of $\phi_t$')
_ = ax.legend(loc='upper left')
ax2 = ax.twinx()
ax2.margins(0,0)
df = pd.DataFrame(norms).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='black', label=r'|$\phi_t$|',ci='sd', linestyle='dashed',err_style='bars',zorder=2,legend=False,ax=ax2)
_ = ax2.set_ylabel(r'Norm |$\phi_t$|')

_ = ax2.legend(loc='upper right')
# Tue, 01 Mar 2022 09:28:58
gt_ind_scaled.mean()
# Tue, 01 Mar 2022 09:28:58
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 09:29:08
gt_ind_scaled
# Tue, 01 Mar 2022 09:29:08
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 09:29:17
gt_ind_scaled.mean(axis=0)
# Tue, 01 Mar 2022 09:29:17
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 09:29:30
# seaborn
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax.margins(0,0)
for n in dirs:
    _ = ax.plot(n, alpha=0.5, zorder=0)

df = pd.DataFrame(dirs).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='blue', label=r'$\alpha_t$',ci='sd',err_style='bars',zorder=2,legend=False,ax=ax)
#_ = ax.set_title('Mean direction +/- SD - aligned at ED')
_ = ax.set_xticks(np.rint(gt_ind_scaled.mean(axis=0).values), minor=False)
_ = ax.set_xlabel('Time (t) - linear interpolated to 40 frames')
_ = ax.set_ylabel(r'Angle $\alpha_t$ of $\phi_t$')
_ = ax.legend(loc='upper left')
ax2 = ax.twinx()
ax2.margins(0,0)
df = pd.DataFrame(norms).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='black', label=r'|$\phi_t$|',ci='sd', linestyle='dashed',err_style='bars',zorder=2,legend=False,ax=ax2)
_ = ax2.set_ylabel(r'Norm |$\phi_t$|')

_ = ax2.legend(loc='upper right')
# Tue, 01 Mar 2022 09:29:48
# seaborn
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax.margins(0,0)
for n in dirs:
    _ = ax.plot(n, alpha=0.5, zorder=0)

df = pd.DataFrame(dirs).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='blue', label=r'$\alpha_t$',ci='sd',err_style='bars',zorder=2,legend=False,ax=ax)
#_ = ax.set_title('Mean direction +/- SD - aligned at ED')
_ = ax.set_xticks(np.rint(gt_ind_scaled.mean(axis=0)), minor=False)
_ = ax.set_xlabel('Time (t) - linear interpolated to 40 frames')
_ = ax.set_ylabel(r'Angle $\alpha_t$ of $\phi_t$')
_ = ax.legend(loc='upper left')
ax2 = ax.twinx()
ax2.margins(0,0)
df = pd.DataFrame(norms).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='black', label=r'|$\phi_t$|',ci='sd', linestyle='dashed',err_style='bars',zorder=2,legend=False,ax=ax2)
_ = ax2.set_ylabel(r'Norm |$\phi_t$|')

_ = ax2.legend(loc='upper right')
# Tue, 01 Mar 2022 09:29:48
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 09:36:00
# align norm an direction by the cardiac phase ED
# resample all 1D feature vectors to the same length
# this should help to validate the pre-defined rules, and if both together explains the cardiac phases
target_t = 40
norms = []
dirs= []
cycle_len=[]
gt_ind = []
gt_ind_rolled = []
xval = np.linspace(0,1,target_t)

for i,n in enumerate(nda_vects[:]):
    cardiac_cycle_length = int(gt_len[i,:,0].sum())
    cycle_len.append(cardiac_cycle_length)
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
    gt_onehot = gt[i][:cardiac_cycle_length]
    gt_ind.append(ind)
    
    dir_axis=0
    gt_ed = ind[0]
    #assert cardiac_cycle_length>=gt_ed
    temp = n[:cardiac_cycle_length]
    
    dim = temp.shape[1:-1]
    #print(dim)
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    #print('centers: ',c.dtype)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    # direction
    directions = flow2direction_lambda(temp)[...,0]
    # direction mean
    directions = np.mean(directions,axis=(1,2,3))
    # direction ed aligned
    directions = np.roll(directions, -1*gt_ed)
    # direction interpolate to unique length
    f = interp1d(np.linspace(0,1,directions.shape[0]), directions, kind='linear')
    directions = f(xval)
    # direction min/max normalised between -1,1

    directions = minmax_lambda([directions,lower,upper])
    
    # magnitude/norm as mean
    norm_full = np.linalg.norm(temp, axis=-1)
    norm_nda = norm_full.mean(axis=(1,2,3))
    # norm ed aligned
    norm_nda = np.roll(norm_nda, -1*gt_ed)
    # interpolate to unique length
    #f = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')
    #norm_nda = f(xval)
    norm_nda = np.interp(xval, np.linspace(0,1,norm_nda.shape[0]), norm_nda)
    # norm min/max aligned to 0,1
    norm_nda = minmax_lambda([norm_nda,mid,upper])
    
    #plt.imshow(gt_onehot.T);plt.show()
    #print(gt_ed)
    gt_onehot_rolled = np.roll(gt_onehot, -1*gt_ed, axis=0)
    #plt.imshow(gt_onehot_rolled.T);plt.show()
    resize_factor = target_t/cardiac_cycle_length
    gt_onehot_rolled = np.argmax(gt_onehot_rolled,axis=0)
    #print(gt_onehot_rolled)
    gt_onehot_rolled = np.clip(np.rint(gt_onehot_rolled * resize_factor),a_min=0,a_max=target_t-1)
    #print(gt_onehot_rolled)
    gt_ind_rolled.append(gt_onehot_rolled)
    
    norms.append(norm_nda)
    dirs.append(directions)
# Tue, 01 Mar 2022 09:36:12
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 09:36:13
# get the mean values
gt_ind_scaled = np.stack(gt_ind_rolled, axis=0)
gt_ind = np.stack(gt_ind,axis=0)
cycle_len = np.stack(cycle_len, axis=0)
norms = np.stack(norms, axis=0)
dirs = np.stack(dirs, axis=0)
norms_m = [norms.mean(axis=0)]
dirs_m=[dirs.mean(axis=0)]
# Tue, 01 Mar 2022 09:36:13
# plot the distribution of the aligned and scaled gt indicies
_ = pd.DataFrame(gt_ind_scaled).plot(kind='box')
# Tue, 01 Mar 2022 09:36:13
# seaborn
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax.margins(0,0)
for n in dirs:
    _ = ax.plot(n, alpha=0.5, zorder=0)

df = pd.DataFrame(dirs).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='blue', label=r'$\alpha_t$',ci='sd',err_style='bars',zorder=2,legend=False,ax=ax)
#_ = ax.set_title('Mean direction +/- SD - aligned at ED')
_ = ax.set_xticks(np.rint(gt_ind_scaled.mean(axis=0)), minor=False)
_ = ax.set_xlabel('Time (t) - linear interpolated to 40 frames')
_ = ax.set_ylabel(r'Angle $\alpha_t$ of $\phi_t$')
_ = ax.legend(loc='upper left')
ax2 = ax.twinx()
ax2.margins(0,0)
df = pd.DataFrame(norms).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='black', label=r'|$\phi_t$|',ci='sd', linestyle='dashed',err_style='bars',zorder=2,legend=False,ax=ax2)
_ = ax2.set_ylabel(r'Norm |$\phi_t$|')

_ = ax2.legend(loc='upper right')
# Tue, 01 Mar 2022 09:36:14
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 09:36:15
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 09:36:16
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 09:39:33
# predict per pathology, on aligned curves
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
ind_all = []

for p in df_meta.pathology.unique():
    msk = get_msk_for_pathology(df_meta, p)
    dirs_p = dirs[msk]
    norms_p = norms[msk]
    ind_ = np.stack([detect_phases(elem, len(elem)) for elem in dirs_p], axis=0)
    _ = pd.DataFrame(ind_).plot(kind='box')
    ind_all.extend(ind_)
# Tue, 01 Mar 2022 09:39:34
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 09:39:58
pd.DataFrame(gt_ind_scaled).hist()
# Tue, 01 Mar 2022 09:39:58
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 09:41:05
fig, ax = plt.subplots(5,1)
pd.DataFrame(gt_ind_scaled).hist(ax=ax)
# Tue, 01 Mar 2022 09:41:06
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 09:41:12
fig, ax = plt.subplots(1,5)
pd.DataFrame(gt_ind_scaled).hist(ax=ax)
# Tue, 01 Mar 2022 09:41:13
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 09:41:28
fig, ax = plt.subplots(1,5, figsize=(20,5))
pd.DataFrame(gt_ind_scaled).hist(ax=ax)
# Tue, 01 Mar 2022 09:41:28
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 09:41:35
fig, ax = plt.subplots(1,5, figsize=(20,2))
pd.DataFrame(gt_ind_scaled).hist(ax=ax)
# Tue, 01 Mar 2022 09:41:35
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 09:41:42
fig, ax = plt.subplots(1,5, figsize=(20,2))
_ pd.DataFrame(gt_ind_scaled).hist(ax=ax)
# Tue, 01 Mar 2022 09:41:45
fig, ax = plt.subplots(1,5, figsize=(20,2))
_ = pd.DataFrame(gt_ind_scaled).hist(ax=ax)
# Tue, 01 Mar 2022 09:41:46
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 09:42:37
fig, ax = plt.subplots(1,5, figsize=(20,2))
_ = pd.DataFrame(gt_ind_scaled).hist(ax=ax)
_ = [a.set_xlim(0,40) for a in ax.flatten()]
# Tue, 01 Mar 2022 09:42:37
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 09:43:55
fig, ax = plt.subplots(1,5, figsize=(20,2))
_ = pd.DataFrame(gt_ind_scaled, columns=phases).hist(ax=ax)
_ = [a.set_xlim(0,40) for a in ax.flatten()]
# Tue, 01 Mar 2022 09:43:56
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 09:44:12
# predict per pathology, on aligned curves
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
ind_all = []

for p in df_meta.pathology.unique():
    msk = get_msk_for_pathology(df_meta, p)
    dirs_p = dirs[msk]
    norms_p = norms[msk]
    ind_ = np.stack([detect_phases(elem, len(elem)) for elem in dirs_p], axis=0)
    _ = pd.DataFrame(ind_, columns=phases).plot(kind='box')
    ind_all.extend(ind_)
# Tue, 01 Mar 2022 09:44:13
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 09:44:29
# plot the distribution of the aligned and scaled gt indicies
_ = pd.DataFrame(gt_ind_scaled, columns=phases).plot(kind='box')
# Tue, 01 Mar 2022 09:44:29
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 09:48:34
# load the vectors
pathstovectnpy = sorted(glob.glob(os.path.join(vects_chooser.selected, 'moved', '*vects*.npy')))
print(pathstovectnpy)
nda_vects = np.concatenate([np.load(path_) for path_ in pathstovectnpy],axis=0)
print(nda_vects.shape)
# Tue, 01 Mar 2022 09:50:02
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 09:50:03
# load the phase gt and pred
pred_path = os.path.join(vects_chooser.selected, 'pred')
pathtsophasenpy = sorted(glob.glob(os.path.join(pred_path, '*gtpred*.npy')))
print(pathtsophasenpy)
nda_phase = np.concatenate([np.load(path_) for path_ in pathtsophasenpy],axis=1)
print(nda_phase.shape)
gt_, pred_ = np.split(nda_phase, axis=0,indices_or_sections=2)
print(gt_.shape)
gt= gt_[0,:,0]
pred = pred_[0,:,0]
print(gt.shape)
gt_len = gt_[0,:,1]
# Tue, 01 Mar 2022 09:50:03
# load some moved examples for easier understanding of the dimensions
pathtomoved = sorted(glob.glob(os.path.join(vects_chooser.selected, 'moved','*moved*.npy')))
print(len(pathtomoved))
mov = np.concatenate([np.load(path_) for path_ in pathtomoved],axis=0)
print(mov.shape) # patient,time,spatial-z,y,x,channel, e.g.: 69,40,16,64,64,1
# Tue, 01 Mar 2022 09:50:31
# plot one moved timestep of one patient = one 3D volume 
# should go from apex to base
from src.visualization.Visualize import show_2D_or_3D
temp = mov[10,0,...,0]
_ = show_2D_or_3D(temp, cmap='gray', interpolation='none')
# Tue, 01 Mar 2022 09:50:31
import tensorflow as tf
from tensorflow.image import ssim
ssim(mov[10,0], mov[10,1],max_val=1.0, filter_size=11,
                          filter_sigma=1.5, k1=0.01, k2=0.03).numpy().mean()
# Tue, 01 Mar 2022 09:50:31
from skimage.metrics import structural_similarity as ssim
ssim(mov[10,0], mov[10,1],multichannel=True)
# Tue, 01 Mar 2022 09:50:31
plt.imshow(gt[0].T)
# Tue, 01 Mar 2022 09:50:31
plt.imshow(pred[0].T)
# Tue, 01 Mar 2022 09:50:32
import tensorflow as tf
import sys
# returns a matrix with the indicies as values, similar to np.indicies
def get_idxs_tf(x):
    return tf.cast(
        tf.reshape(tf.where(tf.ones((x[0], x[1], x[2]))), (x[0], x[1], x[2], 3)),
        tf.float32)


# returns a matrix with vectors pointing to the center
def get_centers_tf(x):
    return tf.cast(
        tf.tile(tf.convert_to_tensor([x[0] // 2, x[1] // 2, x[2] // 2])[tf.newaxis, tf.newaxis, tf.newaxis, ...],
                (x[0], x[1], x[2], 1)), tf.float32)

def get_angle_tf(a, b):
    # this should work for batches of n-dimensional vectors
    # α = arccos[(a · b) / (|a| * |b|)]
    # |v| = √(x² + y² + z²)
    """
    in 3D space
    If vectors a = [xa, ya, za], b = [xb, yb, zb], then:
    α = arccos[(xa * xb + ya * yb + za * zb) / (√(xa2 + ya2 + za2) * √(xb2 + yb2 + zb2))]
    """
    import math as m
    pi = tf.constant(m.pi)
    b = tf.cast(b, dtype=a.dtype)
    inner = tf.einsum('...i,...i->...', a, b)
    norms = tf.norm(a, ord='euclidean', axis=-1) * tf.norm(b, ord='euclidean', axis=-1)  # [...,None]
    cos = inner / (norms + sys.float_info.epsilon)
    #rad = tf.math.acos(tf.clip_by_value(cos, -1.0, 1.0))
    # rad2deg conversion
    #deg = rad * (180.0/pi)
    return cos[..., tf.newaxis]

dim = [16,64,64]
idx = get_idxs_tf(dim)
c = get_centers_tf(dim)
centers = c - idx
centers_tensor = centers[tf.newaxis, ...]
flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x, centers_tensor), name='flow2direction')
minmax_lambda = lambda x : x[1] + (((x[0] - np.min(x[0])) * (x[2]-x[1]))/(np.max(x[0]) - np.min(x[0])))
# define some central params
lower, mid, upper = -1,0, 1
# Tue, 01 Mar 2022 09:50:32
# align norm an direction by the cardiac phase ED
# resample all 1D feature vectors to the same length
# this should help to validate the pre-defined rules, and if both together explains the cardiac phases
target_t = 40
norms = []
dirs= []
cycle_len=[]
gt_ind = []
gt_ind_rolled = []
xval = np.linspace(0,1,target_t)

for i,n in enumerate(nda_vects[:]):
    cardiac_cycle_length = int(gt_len[i,:,0].sum())
    cycle_len.append(cardiac_cycle_length)
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
    gt_onehot = gt[i][:cardiac_cycle_length]
    gt_ind.append(ind)
    
    dir_axis=0
    gt_ed = ind[0]
    #assert cardiac_cycle_length>=gt_ed
    temp = n[:cardiac_cycle_length]
    
    dim = temp.shape[1:-1]
    #print(dim)
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    #print('centers: ',c.dtype)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    # direction
    directions = flow2direction_lambda(temp)[...,0]
    # direction mean
    directions = np.mean(directions,axis=(1,2,3))
    # direction ed aligned
    directions = np.roll(directions, -1*gt_ed)
    # direction interpolate to unique length
    f = interp1d(np.linspace(0,1,directions.shape[0]), directions, kind='linear')
    directions = f(xval)
    # direction min/max normalised between -1,1

    directions = minmax_lambda([directions,lower,upper])
    
    # magnitude/norm as mean
    norm_full = np.linalg.norm(temp, axis=-1)
    norm_nda = norm_full.mean(axis=(1,2,3))
    # norm ed aligned
    norm_nda = np.roll(norm_nda, -1*gt_ed)
    # interpolate to unique length
    #f = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')
    #norm_nda = f(xval)
    norm_nda = np.interp(xval, np.linspace(0,1,norm_nda.shape[0]), norm_nda)
    # norm min/max aligned to 0,1
    norm_nda = minmax_lambda([norm_nda,mid,upper])
    
    #plt.imshow(gt_onehot.T);plt.show()
    #print(gt_ed)
    gt_onehot_rolled = np.roll(gt_onehot, -1*gt_ed, axis=0)
    #plt.imshow(gt_onehot_rolled.T);plt.show()
    resize_factor = target_t/cardiac_cycle_length
    gt_onehot_rolled = np.argmax(gt_onehot_rolled,axis=0)
    #print(gt_onehot_rolled)
    gt_onehot_rolled = np.clip(np.rint(gt_onehot_rolled * resize_factor),a_min=0,a_max=target_t-1)
    #print(gt_onehot_rolled)
    gt_ind_rolled.append(gt_onehot_rolled)
    
    norms.append(norm_nda)
    dirs.append(directions)
# Tue, 01 Mar 2022 09:50:59
# get the mean values
gt_ind_scaled = np.stack(gt_ind_rolled, axis=0)
gt_ind = np.stack(gt_ind,axis=0)
cycle_len = np.stack(cycle_len, axis=0)
norms = np.stack(norms, axis=0)
dirs = np.stack(dirs, axis=0)
norms_m = [norms.mean(axis=0)]
dirs_m=[dirs.mean(axis=0)]
# Tue, 01 Mar 2022 09:50:59
# plot the distribution of the aligned and scaled gt indicies
_ = pd.DataFrame(gt_ind_scaled, columns=phases).plot(kind='box')
# Tue, 01 Mar 2022 09:50:59
# seaborn
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax.margins(0,0)
for n in dirs:
    _ = ax.plot(n, alpha=0.5, zorder=0)

df = pd.DataFrame(dirs).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='blue', label=r'$\alpha_t$',ci='sd',err_style='bars',zorder=2,legend=False,ax=ax)
#_ = ax.set_title('Mean direction +/- SD - aligned at ED')
_ = ax.set_xticks(np.rint(gt_ind_scaled.mean(axis=0)), minor=False)
_ = ax.set_xlabel('Time (t) - linear interpolated to 40 frames')
_ = ax.set_ylabel(r'Angle $\alpha_t$ of $\phi_t$')
_ = ax.legend(loc='upper left')
ax2 = ax.twinx()
ax2.margins(0,0)
df = pd.DataFrame(norms).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='black', label=r'|$\phi_t$|',ci='sd', linestyle='dashed',err_style='bars',zorder=2,legend=False,ax=ax2)
_ = ax2.set_ylabel(r'Norm |$\phi_t$|')

_ = ax2.legend(loc='upper right')
# Tue, 01 Mar 2022 09:50:59
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 09:51:01
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 09:51:02
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 09:51:03
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 09:51:04
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 09:51:04
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 09:51:05
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 09:51:06
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 09:51:08
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 09:51:09
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 09:51:10
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 09:51:11
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 09:57:15
# align norm an direction by the cardiac phase ED
# resample all 1D feature vectors to the same length
# this should help to validate the pre-defined rules, and if both together explains the cardiac phases
target_t = 30
norms = []
dirs= []
cycle_len=[]
gt_ind = []
gt_ind_rolled = []
xval = np.linspace(0,1,target_t)

for i,n in enumerate(nda_vects[:]):
    cardiac_cycle_length = int(gt_len[i,:,0].sum())
    cycle_len.append(cardiac_cycle_length)
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
    gt_onehot = gt[i][:cardiac_cycle_length]
    gt_ind.append(ind)
    
    dir_axis=0
    gt_ed = ind[0]
    #assert cardiac_cycle_length>=gt_ed
    temp = n[:cardiac_cycle_length]
    
    dim = temp.shape[1:-1]
    #print(dim)
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    #print('centers: ',c.dtype)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    # direction
    directions = flow2direction_lambda(temp)[...,0]
    # direction mean
    directions = np.mean(directions,axis=(1,2,3))
    # direction ed aligned
    directions = np.roll(directions, -1*gt_ed)
    # direction interpolate to unique length
    f = interp1d(np.linspace(0,1,directions.shape[0]), directions, kind='linear')
    directions = f(xval)
    # direction min/max normalised between -1,1

    directions = minmax_lambda([directions,lower,upper])
    
    # magnitude/norm as mean
    norm_full = np.linalg.norm(temp, axis=-1)
    norm_nda = norm_full.mean(axis=(1,2,3))
    # norm ed aligned
    norm_nda = np.roll(norm_nda, -1*gt_ed)
    # interpolate to unique length
    #f = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')
    #norm_nda = f(xval)
    norm_nda = np.interp(xval, np.linspace(0,1,norm_nda.shape[0]), norm_nda)
    # norm min/max aligned to 0,1
    norm_nda = minmax_lambda([norm_nda,mid,upper])
    
    # roll, scale, round and clip the gt indicies, to get an aligned distribution of the labels
    gt_onehot_rolled = np.roll(gt_onehot, -1*gt_ed, axis=0)
    resize_factor = target_t/cardiac_cycle_length
    gt_onehot_rolled = np.argmax(gt_onehot_rolled,axis=0)
    gt_onehot_rolled = np.clip(np.rint(gt_onehot_rolled * resize_factor),a_min=0,a_max=target_t-1)
    gt_ind_rolled.append(gt_onehot_rolled)
    
    norms.append(norm_nda)
    dirs.append(directions)
# Tue, 01 Mar 2022 09:57:42
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 09:57:43
# get the mean values
gt_ind_scaled = np.stack(gt_ind_rolled, axis=0)
gt_ind = np.stack(gt_ind,axis=0)
cycle_len = np.stack(cycle_len, axis=0)
norms = np.stack(norms, axis=0)
dirs = np.stack(dirs, axis=0)
norms_m = [norms.mean(axis=0)]
dirs_m=[dirs.mean(axis=0)]
# Tue, 01 Mar 2022 09:57:43
# plot the distribution of the aligned and scaled gt indicies
_ = pd.DataFrame(gt_ind_scaled, columns=phases).plot(kind='box')
# Tue, 01 Mar 2022 09:57:44
"""i = 0
cardiac_cycle_length = int(gt_len[i,:,0].sum())
cycle_len.append(cardiac_cycle_length)
ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
gt_ind.append(ind)
dir_axis=0
gt_ed = ind[0]
#assert cardiac_cycle_length>=gt_ed
temp = n[:cardiac_cycle_length]
norm_full = np.linalg.norm(temp, axis=-1)
norm_nda = norm_full.mean(axis=(1,2,3))"""
# Tue, 01 Mar 2022 09:57:44
# seaborn
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax.margins(0,0)
for n in dirs:
    _ = ax.plot(n, alpha=0.5, zorder=0)

df = pd.DataFrame(dirs).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='blue', label=r'$\alpha_t$',ci='sd',err_style='bars',zorder=2,legend=False,ax=ax)
#_ = ax.set_title('Mean direction +/- SD - aligned at ED')
_ = ax.set_xticks(np.rint(gt_ind_scaled.mean(axis=0)), minor=False)
_ = ax.set_xlabel('Time (t) - linear interpolated to 40 frames')
_ = ax.set_ylabel(r'Angle $\alpha_t$ of $\phi_t$')
_ = ax.legend(loc='upper left')
ax2 = ax.twinx()
ax2.margins(0,0)
df = pd.DataFrame(norms).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='black', label=r'|$\phi_t$|',ci='sd', linestyle='dashed',err_style='bars',zorder=2,legend=False,ax=ax2)
_ = ax2.set_ylabel(r'Norm |$\phi_t$|')

_ = ax2.legend(loc='upper right')
# Tue, 01 Mar 2022 09:57:44
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 09:57:45
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 09:57:46
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 09:57:47
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 09:59:52
# plot the distribution of the aligned and scaled gt indicies
temp = pd.DataFrame(gt_ind_scaled, columns=phases)
print(temp.mean(), temp.std())
_ = temp.plot(kind='box')
# Tue, 01 Mar 2022 09:59:52
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:01:22
# plot the distribution of the aligned and scaled gt indicies
fig, ax1, ax2 = plt.subplots(1,2)
temp = pd.DataFrame(gt_ind_scaled, columns=phases)
print(temp.mean(), temp.std())
_ = temp.plot(kind='box', ax=ax1)
# Tue, 01 Mar 2022 10:01:55
# plot the distribution of the aligned and scaled gt indicies
fig, (ax1,ax2) = plt.subplots(1,2)
temp = pd.DataFrame(gt_ind_scaled, columns=phases)
print(temp.mean(), temp.std())
_ = temp.plot(kind='box', ax=ax1)
# Tue, 01 Mar 2022 10:01:55
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:02:08
# plot the distribution of the aligned and scaled gt indicies
fig, (ax1,ax2) = plt.subplots(1,2, figsize=10,3)
temp = pd.DataFrame(gt_ind_scaled, columns=phases)
print(temp.mean(), temp.std())
_ = temp.plot(kind='box', ax=ax1)
# Tue, 01 Mar 2022 10:02:08
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:02:15
# plot the distribution of the aligned and scaled gt indicies
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10,3))
temp = pd.DataFrame(gt_ind_scaled, columns=phases)
print(temp.mean(), temp.std())
_ = temp.plot(kind='box', ax=ax1)
# Tue, 01 Mar 2022 10:02:15
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:02:28
# plot the distribution of the aligned and scaled gt indicies
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10,3))
temp = pd.DataFrame(gt_ind_scaled, columns=phases)
print(temp.mean(), temp.std())
_ = temp.plot(kind='box', ax=ax1)
_ = temp.plot(kind='hist', ax=ax2)
# Tue, 01 Mar 2022 10:02:29
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:02:55
# plot the distribution of the aligned and scaled gt indicies
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10,3))
temp = pd.DataFrame(gt_ind_scaled, columns=phases)
print(temp.mean(), temp.std())
_ = temp.plot(kind='box', ax=ax1)
_ = temp.plot(kind='kde', ax=ax2)
# Tue, 01 Mar 2022 10:03:07
# plot the distribution of the aligned and scaled gt indicies
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10,3))
temp = pd.DataFrame(gt_ind_scaled, columns=phases)
print(temp.mean(), temp.std())
_ = temp.plot(kind='box', ax=ax1)
_ = temp.plot(kind='area', ax=ax2)
# Tue, 01 Mar 2022 10:03:08
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:03:21
# plot the distribution of the aligned and scaled gt indicies
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10,3))
temp = pd.DataFrame(gt_ind_scaled, columns=phases)
print(temp.mean(), temp.std())
_ = temp.plot(kind='box', ax=ax1)
_ = temp.plot(kind='pie', ax=ax2)
# Tue, 01 Mar 2022 10:03:35
# plot the distribution of the aligned and scaled gt indicies
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10,3))
temp = pd.DataFrame(gt_ind_scaled, columns=phases)
print(temp.mean(), temp.std())
_ = temp.plot(kind='box', ax=ax1)
_ = temp.plot(kind='line', ax=ax2)
# Tue, 01 Mar 2022 10:03:35
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:04:13
# plot the distribution of the aligned and scaled gt indicies
fig, (ax1,ax2, ax3) = plt.subplots(1,3, figsize=(10,3))
temp = pd.DataFrame(gt_ind_scaled, columns=phases)
print(temp.mean(), temp.std())
_ = temp.plot(kind='box', ax=ax1)
_ = temp.plot(kind='line', ax=ax2, label=None)
# Tue, 01 Mar 2022 10:04:13
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:04:45
# plot the distribution of the aligned and scaled gt indicies
fig, (ax1,ax2, ax3) = plt.subplots(1,3, figsize=(15,5))
temp = pd.DataFrame(gt_ind_scaled, columns=phases)
print(temp.mean(), temp.std())
_ = temp.plot(kind='box', ax=ax1)
_ = temp.plot(kind='line', ax=ax2, legend=None)
# Tue, 01 Mar 2022 10:04:45
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:05:10
# plot the distribution of the aligned and scaled gt indicies
fig, (ax1,ax2, ax3) = plt.subplots(1,3, figsize=(15,5))
temp = pd.DataFrame(gt_ind_scaled, columns=phases)
print(temp.mean(), temp.std())
_ = temp.plot(kind='box', ax=ax1)
_ = temp.plot(kind='line', ax=ax2, legend=None)
_ = temp.plot(kind='hist', ax=ax3, legend=None)
# Tue, 01 Mar 2022 10:05:11
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:05:44
# plot the distribution of the aligned and scaled gt indicies
fig, (ax1,ax2, ax3) = plt.subplots(1,3, figsize=(15,5))
temp = pd.DataFrame(gt_ind_scaled, columns=phases)
print(temp.mean(), temp.std())
_ = temp.plot(kind='box', ax=ax1)
_ = temp.plot(kind='scatter', ax=ax2, legend=None)
_ = temp.plot(kind='hist', ax=ax3, legend=None)
# Tue, 01 Mar 2022 10:05:47
# plot the distribution of the aligned and scaled gt indicies
fig, (ax1,ax2, ax3) = plt.subplots(1,3, figsize=(15,5))
temp = pd.DataFrame(gt_ind_scaled, columns=phases)
print(temp.mean(), temp.std())
_ = temp.plot(kind='box', ax=ax1)
_ = temp.plot(kind='line', ax=ax2, legend=None)
_ = temp.plot(kind='hist', ax=ax3, legend=None)
# Tue, 01 Mar 2022 10:05:48
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:06:42
# plot the distribution of the aligned and scaled gt indicies
fig, (ax1,ax2, ax3) = plt.subplots(1,3, figsize=(15,5))
temp = pd.DataFrame(gt_ind_scaled, columns=phases)
print(temp.mean(), temp.std())
_ = temp.plot(kind='box', ax=ax1)
_ = temp.plot(kind='line', ax=ax2, legend=None)
_ = temp.plot(kind='hist', ax=ax3, legend=None,ylabel='')
# Tue, 01 Mar 2022 10:06:42
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:07:22
# plot the distribution of the aligned and scaled gt indicies
fig, (ax1,ax2, ax3) = plt.subplots(1,3, figsize=(15,5))
temp = pd.DataFrame(gt_ind_scaled, columns=phases)
print(temp.mean(), temp.std())
_ = temp.plot(kind='box', ax=ax1)
_ = temp.plot(kind='line', ax=ax2, legend=None)
_ = temp.plot(kind='hist', ax=ax3, legend=None)
ax3.set_ylabel('')
# Tue, 01 Mar 2022 10:07:22
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:36:48
# plot the distribution of the aligned and scaled gt indicies
fig, (ax1,ax2, ax3) = plt.subplots(1,3, figsize=(15,5))
temp = pd.DataFrame(gt_ind_scaled, columns=phases)
print(temp.mean(), temp.std())
sb.violinplot(data=temp)
#_ = temp.plot(kind='box', ax=ax1)
_ = temp.plot(kind='line', ax=ax2, legend=None)
_ = temp.plot(kind='hist', ax=ax3, legend=None)
ax3.set_ylabel('')
# Tue, 01 Mar 2022 10:36:49
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:37:02
# plot the distribution of the aligned and scaled gt indicies
fig, (ax1,ax2, ax3) = plt.subplots(1,3, figsize=(15,5))
temp = pd.DataFrame(gt_ind_scaled, columns=phases)
print(temp.mean(), temp.std())
sb.violinplot(data=temp.melt())
#_ = temp.plot(kind='box', ax=ax1)
_ = temp.plot(kind='line', ax=ax2, legend=None)
_ = temp.plot(kind='hist', ax=ax3, legend=None)
ax3.set_ylabel('')
# Tue, 01 Mar 2022 10:37:02
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:37:10
# plot the distribution of the aligned and scaled gt indicies
fig, (ax1,ax2, ax3) = plt.subplots(1,3, figsize=(15,5))
temp = pd.DataFrame(gt_ind_scaled, columns=phases)
print(temp.mean(), temp.std())
sb.violinplot(data=temp.melt(), ax=ax1)
#_ = temp.plot(kind='box', ax=ax1)
_ = temp.plot(kind='line', ax=ax2, legend=None)
_ = temp.plot(kind='hist', ax=ax3, legend=None)
ax3.set_ylabel('')
# Tue, 01 Mar 2022 10:37:11
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:37:18
# plot the distribution of the aligned and scaled gt indicies
fig, (ax1,ax2, ax3) = plt.subplots(1,3, figsize=(15,5))
temp = pd.DataFrame(gt_ind_scaled, columns=phases)
print(temp.mean(), temp.std())
sb.violinplot(data=temp, ax=ax1)
#_ = temp.plot(kind='box', ax=ax1)
_ = temp.plot(kind='line', ax=ax2, legend=None)
_ = temp.plot(kind='hist', ax=ax3, legend=None)
ax3.set_ylabel('')
# Tue, 01 Mar 2022 10:37:18
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:38:16
# plot the distribution of the aligned and scaled gt indicies
fig, (ax1,ax2, ax3) = plt.subplots(1,3, figsize=(15,5))
temp = pd.DataFrame(gt_ind_scaled, columns=phases)
print(temp.mean(), temp.std())
ax1 = sb.violinplot(data=temp, ax=ax1)
ax1 = sb.catplot(data=temp, ax=ax1)
#_ = temp.plot(kind='box', ax=ax1)
_ = temp.plot(kind='line', ax=ax2, legend=None)
_ = temp.plot(kind='hist', ax=ax3, legend=None)
ax3.set_ylabel('')
# Tue, 01 Mar 2022 10:38:17
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:38:24
# plot the distribution of the aligned and scaled gt indicies
fig, (ax1,ax2, ax3) = plt.subplots(1,3, figsize=(15,5))
temp = pd.DataFrame(gt_ind_scaled, columns=phases)
print(temp.mean(), temp.std())
ax1 = sb.violinplot(data=temp, ax=ax1)
ax1 = sb.stripplot(data=temp, ax=ax1)
#_ = temp.plot(kind='box', ax=ax1)
_ = temp.plot(kind='line', ax=ax2, legend=None)
_ = temp.plot(kind='hist', ax=ax3, legend=None)
ax3.set_ylabel('')
# Tue, 01 Mar 2022 10:38:24
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:39:49
temp[temp['MD']<10]
# Tue, 01 Mar 2022 10:39:49
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:41:34
temp[temp['MD']>10]
# Tue, 01 Mar 2022 10:41:34
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:42:13
# plot the distribution of the aligned and scaled gt indicies
fig, (ax1,ax2, ax3) = plt.subplots(1,3, figsize=(15,5))
temp = pd.DataFrame(gt_ind_scaled, columns=phases)
# TODO check this outlier with MD ==2
temp = temp[temp['MD']>10]
print(temp.mean(), temp.std())
ax1 = sb.violinplot(data=temp, ax=ax1)
ax1 = sb.stripplot(data=temp, ax=ax1)
#_ = temp.plot(kind='box', ax=ax1)
_ = temp.plot(kind='line', ax=ax2, legend=None)
_ = temp.plot(kind='hist', ax=ax3, legend=None)
ax3.set_ylabel('')
# Tue, 01 Mar 2022 10:42:13
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:42:56
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:44:07
# TOF unsupervised center 14_41 gaus 2
sb.set_context('paper')
sb.set(font_scale = 2)
_ =df.plot(kind='box')
ax = sb.violinplot(data=df)
ax = sb.stripplot(data=df)
pd.options.display.float_format = "{:,.2f}".format
df_summarized = pd.concat([df.mean(axis=0),df.std(axis=0), df.median(axis=0)], axis=1)
df_summarized.columns = ['mean', 'SD', 'meadian']
print(df_summarized.mean())
print(df_summarized)

sb.set_context('paper')
sb.set(font_scale = 1.8)
fig, axes = plt.subplots(1,5,figsize=(25,4))
axes = axes.flatten()
for i,ax in enumerate(axes):
    _ = ax.scatter(upred_ind[:,i],gt_ind[:,i])
    _ = ax.axline((1, 1), slope=1)
    _ = ax.set_title(phases[i])
    _ = ax.set_xlabel('Index - pred')
    _ = ax.set_ylabel('Index - gt')
    _ = ax.set_xlim([0,35])
    _ = ax.set_ylim([0,35])
plt.tight_layout()
# Tue, 01 Mar 2022 10:44:13
pred_u = np.zeros_like(gt)
upred_ind = []
cycle_len=[]
print(pred_u.shape)
for i in range(pred_u.shape[0]):
    weight = 1
    
    cardiac_cycle_length = int(gt_len[i,:,0].sum())
    cycle_len.append(cardiac_cycle_length)
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
    #print(gt[i][:cardiac_cycle_length].T.shape)
    
    indices = get_phases_from_vects(nda_vects[i][:cardiac_cycle_length], length=cardiac_cycle_length, gtind=ind,plot=False,dir_axis=0)
    upred_ind.append(indices)
    indices = np.array(indices)
    onehot = np.zeros((indices.size, cardiac_cycle_length))
    onehot[np.arange(indices.size), indices] = weight
    pred_u[i][0:cardiac_cycle_length] = onehot.T
upred_ind=np.stack(upred_ind, axis=0)
cycle_len = np.stack(cycle_len, axis=0)
# re-create a compatible shape for the metric fn
gt_ = np.stack([gt,gt_len], axis=1)
pred_ = np.stack([pred_u,np.zeros_like(pred_u)], axis=1)

# create a dataframe for further plots
from src.utils.Metrics import meandiff
phases = ['ED', 'MS', 'ES', 'PF', 'MD']
res = meandiff(gt_, pred_, apply_sum=False, apply_average=False)
df = pd.DataFrame(res.numpy(), columns=phases)
# Tue, 01 Mar 2022 10:44:53
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:44:54
# get the patient idx, per label where this method fails the most
label = 0
error_thres = 6
if type(res) != np.ndarray: res.numpy()
print('label:{} \naFD>{}\nPatient idx: {}'.format(phases[label],error_thres,np.where(res[:,label]>error_thres)[0]))
print('gt length: {}'.format(cycle_len[res[:,label]>error_thres]))
print('gt indices: {}'.format(gt_ind[:,label][(res[:,label]>error_thres)]))
print('pred indices: {}'.format(upred_ind[:,label][(res[:,label]>error_thres)]))
print('aFD: {}'.format(res[:,label][res[:,label]>error_thres]))
# Tue, 01 Mar 2022 10:44:54
# TOF unsupervised center 14_41 gaus 2
sb.set_context('paper')
sb.set(font_scale = 2)
_ =df.plot(kind='box')
ax = sb.violinplot(data=df)
ax = sb.stripplot(data=df)
pd.options.display.float_format = "{:,.2f}".format
df_summarized = pd.concat([df.mean(axis=0),df.std(axis=0), df.median(axis=0)], axis=1)
df_summarized.columns = ['mean', 'SD', 'meadian']
print(df_summarized.mean())
print(df_summarized)

sb.set_context('paper')
sb.set(font_scale = 1.8)
fig, axes = plt.subplots(1,5,figsize=(25,4))
axes = axes.flatten()
for i,ax in enumerate(axes):
    _ = ax.scatter(upred_ind[:,i],gt_ind[:,i])
    _ = ax.axline((1, 1), slope=1)
    _ = ax.set_title(phases[i])
    _ = ax.set_xlabel('Index - pred')
    _ = ax.set_ylabel('Index - gt')
    _ = ax.set_xlim([0,35])
    _ = ax.set_ylim([0,35])
plt.tight_layout()
# Tue, 01 Mar 2022 10:44:54
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:44:55
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:45:38
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:45:55
# load the vectors
pathstovectnpy = sorted(glob.glob(os.path.join(vects_chooser.selected, 'moved', '*vects*.npy')))
print(pathstovectnpy)
nda_vects = np.concatenate([np.load(path_) for path_ in pathstovectnpy],axis=0)
print(nda_vects.shape)
# Tue, 01 Mar 2022 10:46:04
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:46:04
# load the phase gt and pred
pred_path = os.path.join(vects_chooser.selected, 'pred')
pathtsophasenpy = sorted(glob.glob(os.path.join(pred_path, '*gtpred*.npy')))
print(pathtsophasenpy)
nda_phase = np.concatenate([np.load(path_) for path_ in pathtsophasenpy],axis=1)
print(nda_phase.shape)
gt_, pred_ = np.split(nda_phase, axis=0,indices_or_sections=2)
print(gt_.shape)
gt= gt_[0,:,0]
pred = pred_[0,:,0]
print(gt.shape)
gt_len = gt_[0,:,1]
# Tue, 01 Mar 2022 10:46:05
# load some moved examples for easier understanding of the dimensions
pathtomoved = sorted(glob.glob(os.path.join(vects_chooser.selected, 'moved','*moved*.npy')))
print(len(pathtomoved))
mov = np.concatenate([np.load(path_) for path_ in pathtomoved],axis=0)
print(mov.shape) # patient,time,spatial-z,y,x,channel, e.g.: 69,40,16,64,64,1
# Tue, 01 Mar 2022 10:46:08
# plot one moved timestep of one patient = one 3D volume 
# should go from apex to base
from src.visualization.Visualize import show_2D_or_3D
temp = mov[10,0,...,0]
_ = show_2D_or_3D(temp, cmap='gray', interpolation='none')
# Tue, 01 Mar 2022 10:46:08
import tensorflow as tf
from tensorflow.image import ssim
ssim(mov[10,0], mov[10,1],max_val=1.0, filter_size=11,
                          filter_sigma=1.5, k1=0.01, k2=0.03).numpy().mean()
# Tue, 01 Mar 2022 10:46:08
from skimage.metrics import structural_similarity as ssim
ssim(mov[10,0], mov[10,1],multichannel=True)
# Tue, 01 Mar 2022 10:46:08
plt.imshow(gt[0].T)
# Tue, 01 Mar 2022 10:46:08
plt.imshow(pred[0].T)
# Tue, 01 Mar 2022 10:46:08
import tensorflow as tf
import sys
# returns a matrix with the indicies as values, similar to np.indicies
def get_idxs_tf(x):
    return tf.cast(
        tf.reshape(tf.where(tf.ones((x[0], x[1], x[2]))), (x[0], x[1], x[2], 3)),
        tf.float32)


# returns a matrix with vectors pointing to the center
def get_centers_tf(x):
    return tf.cast(
        tf.tile(tf.convert_to_tensor([x[0] // 2, x[1] // 2, x[2] // 2])[tf.newaxis, tf.newaxis, tf.newaxis, ...],
                (x[0], x[1], x[2], 1)), tf.float32)

def get_angle_tf(a, b):
    # this should work for batches of n-dimensional vectors
    # α = arccos[(a · b) / (|a| * |b|)]
    # |v| = √(x² + y² + z²)
    """
    in 3D space
    If vectors a = [xa, ya, za], b = [xb, yb, zb], then:
    α = arccos[(xa * xb + ya * yb + za * zb) / (√(xa2 + ya2 + za2) * √(xb2 + yb2 + zb2))]
    """
    import math as m
    pi = tf.constant(m.pi)
    b = tf.cast(b, dtype=a.dtype)
    inner = tf.einsum('...i,...i->...', a, b)
    norms = tf.norm(a, ord='euclidean', axis=-1) * tf.norm(b, ord='euclidean', axis=-1)  # [...,None]
    cos = inner / (norms + sys.float_info.epsilon)
    #rad = tf.math.acos(tf.clip_by_value(cos, -1.0, 1.0))
    # rad2deg conversion
    #deg = rad * (180.0/pi)
    return cos[..., tf.newaxis]

dim = [16,64,64]
idx = get_idxs_tf(dim)
c = get_centers_tf(dim)
centers = c - idx
centers_tensor = centers[tf.newaxis, ...]
flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x, centers_tensor), name='flow2direction')
minmax_lambda = lambda x : x[1] + (((x[0] - np.min(x[0])) * (x[2]-x[1]))/(np.max(x[0]) - np.min(x[0])))
# define some central params
lower, mid, upper = -1,0, 1
# Tue, 01 Mar 2022 10:46:08
# align norm an direction by the cardiac phase ED
# resample all 1D feature vectors to the same length
# this should help to validate the pre-defined rules, and if both together explains the cardiac phases
target_t = 30
norms = []
dirs= []
cycle_len=[]
gt_ind = []
gt_ind_rolled = []
xval = np.linspace(0,1,target_t)

for i,n in enumerate(nda_vects[:]):
    cardiac_cycle_length = int(gt_len[i,:,0].sum())
    cycle_len.append(cardiac_cycle_length)
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
    gt_onehot = gt[i][:cardiac_cycle_length]
    gt_ind.append(ind)
    
    dir_axis=0
    gt_ed = ind[0]
    #assert cardiac_cycle_length>=gt_ed
    temp = n[:cardiac_cycle_length]
    
    dim = temp.shape[1:-1]
    #print(dim)
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    #print('centers: ',c.dtype)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    # direction
    directions = flow2direction_lambda(temp)[...,0]
    # direction mean
    directions = np.mean(directions,axis=(1,2,3))
    # direction ed aligned
    directions = np.roll(directions, -1*gt_ed)
    # direction interpolate to unique length
    f = interp1d(np.linspace(0,1,directions.shape[0]), directions, kind='linear')
    directions = f(xval)
    # direction min/max normalised between -1,1

    directions = minmax_lambda([directions,lower,upper])
    
    # magnitude/norm as mean
    norm_full = np.linalg.norm(temp, axis=-1)
    norm_nda = norm_full.mean(axis=(1,2,3))
    # norm ed aligned
    norm_nda = np.roll(norm_nda, -1*gt_ed)
    # interpolate to unique length
    #f = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')
    #norm_nda = f(xval)
    norm_nda = np.interp(xval, np.linspace(0,1,norm_nda.shape[0]), norm_nda)
    # norm min/max aligned to 0,1
    norm_nda = minmax_lambda([norm_nda,mid,upper])
    
    # roll, scale, round and clip the gt indicies, to get an aligned distribution of the labels
    gt_onehot_rolled = np.roll(gt_onehot, -1*gt_ed, axis=0)
    resize_factor = target_t/cardiac_cycle_length
    gt_onehot_rolled = np.argmax(gt_onehot_rolled,axis=0)
    gt_onehot_rolled = np.clip(np.rint(gt_onehot_rolled * resize_factor),a_min=0,a_max=target_t-1)
    gt_ind_rolled.append(gt_onehot_rolled)
    
    norms.append(norm_nda)
    dirs.append(directions)
# Tue, 01 Mar 2022 10:46:21
# get the mean values
gt_ind_scaled = np.stack(gt_ind_rolled, axis=0)
gt_ind = np.stack(gt_ind,axis=0)
cycle_len = np.stack(cycle_len, axis=0)
norms = np.stack(norms, axis=0)
dirs = np.stack(dirs, axis=0)
norms_m = [norms.mean(axis=0)]
dirs_m=[dirs.mean(axis=0)]
# Tue, 01 Mar 2022 10:46:21
# plot the distribution of the aligned and scaled gt indicies
fig, (ax1,ax2, ax3) = plt.subplots(1,3, figsize=(15,5))
temp = pd.DataFrame(gt_ind_scaled, columns=phases)
# TODO check this outlier with MD ==2
temp = temp[temp['MD']>10]
print(temp.mean(), temp.std())
ax1 = sb.violinplot(data=temp, ax=ax1)
ax1 = sb.stripplot(data=temp, ax=ax1)
#_ = temp.plot(kind='box', ax=ax1)
_ = temp.plot(kind='line', ax=ax2, legend=None)
_ = temp.plot(kind='hist', ax=ax3, legend=None)
ax3.set_ylabel('')
# Tue, 01 Mar 2022 10:46:21
"""i = 0
cardiac_cycle_length = int(gt_len[i,:,0].sum())
cycle_len.append(cardiac_cycle_length)
ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
gt_ind.append(ind)
dir_axis=0
gt_ed = ind[0]
#assert cardiac_cycle_length>=gt_ed
temp = n[:cardiac_cycle_length]
norm_full = np.linalg.norm(temp, axis=-1)
norm_nda = norm_full.mean(axis=(1,2,3))"""
# Tue, 01 Mar 2022 10:46:21
get_ipython().run_cell_magic('time', '', "norm_nda = norm_full.mean(axis=(1,2,3))\nf = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')\nnorm_nda = f(xval)\n")
# Tue, 01 Mar 2022 10:46:21
get_ipython().run_cell_magic('time', '', 'norm_nda = norm_full.mean(axis=(1,2,3))\nnorm_nda = np.interp(xval, np.linspace(0,1,norm_nda.shape[0]), norm_nda)\n')
# Tue, 01 Mar 2022 10:46:21
# seaborn
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax.margins(0,0)
for n in dirs:
    _ = ax.plot(n, alpha=0.5, zorder=0)

df = pd.DataFrame(dirs).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='blue', label=r'$\alpha_t$',ci='sd',err_style='bars',zorder=2,legend=False,ax=ax)
#_ = ax.set_title('Mean direction +/- SD - aligned at ED')
_ = ax.set_xticks(np.rint(gt_ind_scaled.mean(axis=0)), minor=False)
_ = ax.set_xlabel('Time (t) - linear interpolated to 40 frames')
_ = ax.set_ylabel(r'Angle $\alpha_t$ of $\phi_t$')
_ = ax.legend(loc='upper left')
ax2 = ax.twinx()
ax2.margins(0,0)
df = pd.DataFrame(norms).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='black', label=r'|$\phi_t$|',ci='sd', linestyle='dashed',err_style='bars',zorder=2,legend=False,ax=ax2)
_ = ax2.set_ylabel(r'Norm |$\phi_t$|')

_ = ax2.legend(loc='upper right')
# Tue, 01 Mar 2022 10:46:22
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:46:23
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:46:24
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:46:25
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:46:26
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:46:27
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:46:27
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:46:28
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:46:29
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:46:30
folds_chooser = FileChooser(os.path.join(os.getcwd(),'/mnt/ssd/git/dynamic-cmr-models/exp/phasereg_v3/acdc/'), '')
display(folds_chooser)
# Tue, 01 Mar 2022 10:46:30
# define df and helper for pathology extractio
# this should have the same order as our inference data
#df_meta = pd.read_csv(folds_chooser.selected)
df_meta = pd.read_csv('/mnt/sds/sd20i001/sven/data/acdc/02_imported_4D_unfiltered/df_kfold.csv')
df_meta = df_meta.loc[:, ~df_meta.columns.str.contains('^Unnamed')]
df_meta = df_meta[df_meta.patient != 'patient090']  #we excluded this patient

def get_msk_for_pathology(df_, pathology='minf'):
    msk = []
    for f in [0,1,2,3,4]:
        patients = df_[df_.fold.isin([f])]
        patients = patients[patients['phase']=='ED']
        pat = patients[patients['modality'] == 'test']['patient'].str.lower().unique()
        sub_df = patients[patients.patient.isin(pat)].drop_duplicates(ignore_index=True, subset='patient')
        sub_df = sub_df.drop('x_path',axis=1).drop('y_path',axis=1)
        msk.append(sub_df['pathology'].str.lower()==pathology.lower())
    return np.concatenate(msk)
df_meta
# Tue, 01 Mar 2022 10:46:30
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:46:31
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:46:32
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:46:33
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:46:34
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:46:35
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:46:36
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax2 = ax.twinx()
ax2.margins(0,0)
ax.margins(0,0)
for p in df_meta.pathology.unique():
    msk = get_msk_for_pathology(df_meta, p)
    dirs_p = dirs[msk]
    norms_p = norms[msk]
    #for n in dirs_minf:
    #    _ = ax.plot(n, alpha=0.5, zorder=0)
    
    df = pd.DataFrame(dirs_p).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, label=r'$\alpha_t$ {}'.format(p),ci='sd',err_style=None,zorder=2,legend=False,ax=ax)
    #_ = ax.set_title('Mean direction +/- SD - aligned at ED')
    _ = ax.set_xlabel('Time (t) - linear interpolated to 40 frames')
    _ = ax.set_ylabel(r'Angle $\alpha_t$ of $\phi_t$')
    _ = ax.legend(loc='upper left')
    df = pd.DataFrame(norms_p).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, label=r'|$\phi_t$| {}'.format(p),ci='sd', linestyle='dashed',err_style=None,zorder=2,legend=False,ax=ax2)
    _ = ax2.set_ylabel(r'Norm |$\phi_t$|')
    _ = ax2.legend(loc='upper right')
    
plt.show()
# Tue, 01 Mar 2022 10:46:36
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:46:37
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:46:38
# predict per pathology, on aligned curves
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
ind_all = []

for p in df_meta.pathology.unique():
    msk = get_msk_for_pathology(df_meta, p)
    dirs_p = dirs[msk]
    norms_p = norms[msk]
    ind_ = np.stack([detect_phases(elem, len(elem)) for elem in dirs_p], axis=0)
    _ = pd.DataFrame(ind_, columns=phases).plot(kind='box')
    ind_all.extend(ind_)
# Tue, 01 Mar 2022 10:46:39
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:46:40
fig, ax = plt.subplots(1,5, figsize=(20,2))
_ = pd.DataFrame(gt_ind_scaled, columns=phases).hist(ax=ax)
_ = [a.set_xlim(0,40) for a in ax.flatten()]
# Tue, 01 Mar 2022 10:46:40
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:46:41
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:48:05
fig, ax = plt.subplots(1,5, figsize=(20,2))
_ = pd.DataFrame(gt_ind_scaled, columns=phases).hist(ax=ax)
_ = [a.set_xlim(0,40) for a in ax.flatten()]
# Tue, 01 Mar 2022 10:48:06
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:48:18
fig, ax = plt.subplots(1,5, figsize=(20,2))
_ = pd.DataFrame(gt_ind_scaled, columns=phases).hist(ax=ax)
_ = [a.set_xlim(0,30) for a in ax.flatten()]
# Tue, 01 Mar 2022 10:48:18
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:48:35
# scatterplot per pathology, t as dymension
import sklearn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
y = np.zeros(99)
for i,p in enumerate(df_meta.pathology.unique()):
    msk = get_msk_for_pathology(df_meta, p)
    y[msk] = i
low_dim = TSNE(n_components=2).fit_transform(norms)
#low_dim = PCA(n_components=2).fit_transform(norms)
_ =plt.scatter(low_dim[:,0],low_dim[:,1], c=y)
# Tue, 01 Mar 2022 10:48:35
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:48:38
# plot one line per patient - norm
fig, ax = plt.subplots(figsize=(25,5))
for n in norms:
    _ = plt.plot(n)
_ = ax.set_title('Magnitudes aligned at ED, resampling shape 40,')
# Tue, 01 Mar 2022 10:48:39
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:48:39
# plot one line per direction
fig, ax = plt.subplots(figsize=(25,5))
for n in dirs:
    _ = plt.plot(n)
_ = ax.set_title('Directions aligned at ED, resampling shape 40,')
# Tue, 01 Mar 2022 10:48:40
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:48:41
from src.visualization.Visualize import show_2D_or_3D
from src.data.Preprocess import clip_quantile
nda_temp = nda_vects[0]
# norm_1 = nda_temp[:,:,:32,:32]# top left
norm_1 = nda_temp[:,:,:32,32:]# top right
norm_1 = np.linalg.norm(norm_1, axis=-1)
nda_norm_ = np.linalg.norm(nda_temp, axis=-1)
plt.hist(nda_norm_.flatten());plt.show()
nda_norm = clip_quantile(nda_norm_, 0.99)
nda_norm = minmax_lambda([nda_norm,0,1])
plt.hist(nda_norm.flatten())
plt.show()
# mask phi with a threshold norm matrix 
nda_msk = (nda_norm>=0.2).astype(np.float32)
nda_temp = nda_norm * nda_msk
#plt.hist(nda_temp.flatten())
print(nda_temp.shape)
_ = show_2D_or_3D(nda_norm_[:,0], allow_slicing=True)
_ = show_2D_or_3D(nda_temp[:,0], allow_slicing=True)
_ = show_2D_or_3D(nda_temp[:,-1], allow_slicing=True)

# top right
_ = show_2D_or_3D(norm_1[:,0], allow_slicing=True,cmap='gray', interpolation='none')
# Tue, 01 Mar 2022 10:48:43
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:48:44
fig, ax = plt.subplots(figsize=(25,3))
plt.plot(np.argmax(gt[0], axis=1))
ax.set_yticks([0, 1, 2, 3, 4], minor=False)
phases = ['ED', 'MS', 'ES', 'PF', 'MD']
_ = ax.set_yticklabels(phases)
# Tue, 01 Mar 2022 10:48:44
# plot the mean/max norm for one patient oveer time
nda_1d_max = np.max(nda_temp,axis=(1,2,3))
nda_1d_mean = np.mean(nda_temp,axis=(1,2,3))
nda_1d_sum = np.sum(nda_temp,axis=(1,2,3))

fig, ax = plt.subplots(figsize=(25,3))
ax.set_title('max norm')
_ = plt.plot(nda_1d_max); plt.show()

fig, ax = plt.subplots(figsize=(25,3))
ax.set_title('mean norm')
_ = plt.plot(nda_1d_mean)

fig, ax = plt.subplots(figsize=(25,3))
ax.set_title('mean sum')
_ = plt.plot(nda_1d_sum)
#ax.set_ylim(0.0,0.15)
# Tue, 01 Mar 2022 10:48:44
temp = np.arange(10)
temp[:0]
# Tue, 01 Mar 2022 10:48:44
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:48:45
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:48:46
def detect_phases(dir_1d_mean, length):
    
    import scipy.signal as sig
    
        # MS
    # Global min of f(x)
    ms = np.argmin(dir_1d_mean)
    ms = ms -1 # take the bucket before the first min peak
    
    
    # ES
    # First time f(x)>0 after MS
    cycle = np.concatenate([dir_1d_mean[ms:], dir_1d_mean[:ms]])
    temp_ = 0
    es_found=False
    negative_slope = False
    for idx,elem in enumerate(cycle):
        if elem<0:
            negative_slope=True
            temp_ = idx
        elif elem>=0 and negative_slope:
            es_found = True
            #temp_ = idx
            break # stop after first zero-transition
    if es_found:
        es = ms + temp_
        #es = es-1
    else:
        es = ms + 1 # the frame after ms, fallback
    if es>=length:
        es = np.mod(es,length)
        print('ES overflow: {}, ms:{}'.format(es,ms))
     
    
    # PF
    # First peak after ES, min height 0.6
    seq = dir_1d_mean[es:]
    peaks, prop = sig.find_peaks(seq, height=0.6)#height=0.6 we normalise between -1 and 1, PF should be close to argmax

    if len(peaks>0):
        pf = es + peaks[0] # take the peak after es
        pf = pf -1
    else:
        print('pf not clear, set to ES {} + 1'.format(es))
        pf = es + 1
    pf = np.mod(pf, length)
      

    # ED
    # Between pf and ms: last time f(x) cross zero from positive to negative
    # a priori knowledge ED needs a minimal distance of 2 frames towards MS
    # CHANGED the minimal distance between ED and MS
    cycle = np.concatenate([dir_1d_mean[pf:], dir_1d_mean[:ms]])
    #print(cycle)
    ed_found = False
    last_idx_positive = True # we start at the pf, which is the peak(dir)
    for idx,elem in enumerate(cycle):
        
        if elem>=0:
            last_idx_positive = True
        # remember the last idx before the direction gets negative the last time before ms
        elif elem<0 and last_idx_positive: # first time direction negative
            ed_found = True # for fallbacks
            temp_ = idx # idx before negative direction
            #print('found transition at: {}'.format(idx))
            last_idx_positive = False # remember only the first idx after transition
        
    if ed_found:
        ed = pf + temp_
        #print('ed:{}, pf:{}, temp_:{}, lenght: {}'.format(ed,pf,temp_,length))
    else: 
        # if we dont find a transition from positive to negative, take the idx which is the closest to zero
        temp_ = np.argmin(np.abs(cycle)) # make sure we have a minimal distance
        ed = pf + temp_
        print('ED: no transition found between {}-{} , closest id to 0: {}, ed = {}'.format(pf,ms,temp_,ed))
    
    if ed>=length:
        #print('gt ed: {}, ed: {}, length: {}'.format(gted,ed,length))
        ed = np.mod(ed,length)
        #print('mod ed: {}'.format(ed))
    #ed = ed-1 # take the bucket before negative
        
    # MD
    # Middle between PF and ED
    ed_slice_idx = ed
    if ed_slice_idx<=pf: # ed overflow --> beginning of cmr stack
        ed_slice_idx = length + ed
    md = (pf+ed_slice_idx)//2 # the bucket after the middle
    md = md + 1
    md = np.mod(md,length)
    
    return np.array([ed,ms,es,pf,md])
# Tue, 01 Mar 2022 10:48:46
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:48:47
# try to predict the phases unsupervised just from the norm/direction curve

def get_phases_from_vects(vects_nda, length=-1, plot=True, dir_axis=0, gtind=[], figsize=(25,3)):
    import scipy.signal
    import scipy.signal as sig
    import scipy.ndimage
    from scipy.ndimage import gaussian_filter1d
    
    gted=gtind[0]
    
    # signal smoothing
    kernel_size = 4
    kernel = np.ones(kernel_size) / kernel_size
    z = 0
    
    
    dim_z = vects_nda.shape[1]
    #vects_nda = vects_nda[:int(length)]
    
    
    # define some helper lambdas
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    dim = vects_nda.shape[1:-1]
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]

    
    # norm of the vector
    norm_nda = np.linalg.norm(vects_nda[...,dir_axis:], axis=-1)
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    norm_nda_thres = clip_quantile(norm_nda, 0.9)
    norm_nda = minmax_lambda([norm_nda,0,1])
    norm_msk = norm_nda>=0.0 # this could be used to mask v_t by a threshold norm
    norm_nda = norm_nda * norm_msk
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    nda_1d_mean = np.mean(norm_nda,axis=(1,2,3))
    nda_1d_max = np.max(norm_nda,axis=(1,2,3))
    nda_1d_median = np.median(norm_nda,axis=(1,2,3))
    nda_1d_mean_smooth = np.convolve(nda_1d_mean, kernel, mode='same')
    #print(nda_1d_mean_smooth.shape, nda_1d_mean.shape)
    
    
    # direction relative to the center
    
    directions = flow2direction_lambda(vects_nda)[...,0].numpy()
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    directions = minmax_lambda([directions,lower,upper])
    #plt.hist(directions.flatten());plt.show()
    directions = directions * norm_msk
    #plt.hist(directions.flatten());plt.show()
    
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    dir_1d_mean = np.mean(directions,axis=(1,2,3))
    dir_1d_median = np.median(directions,axis=(1,2,3))

    
    # Find phases
    # 1st smooth the direction with a rolling averge (kernelsize=4)
    # or via gaussian filter
    # 2nd min/max normalise each direction vector in a range [-1:1]
    
    
    dir_1d_mean = gaussian_filter1d(dir_1d_mean, sigma=2, mode='wrap') # original: dir_1d_mean
    
    ################# CHANGED !!!!!! ######################## RECENTLY NO CHANGES
    # min/max scaling
    dir_1d_mean = minmax_lambda([dir_1d_mean,lower,upper])
    
    
#     # test with corner specific vector fields
#     y,x = directions.shape[-2:]
    
#     vects_mean = minmax_lambda([np.mean(vects_nda,axis=(1,2,3,4)),lower,upper])
#     vects_1 = vects_nda[:,:,:y//2,:x//2,:]
#     vects_1_mean = minmax_lambda([np.mean(vects_1,axis=(1,2,3,4)),lower,upper])
#     vects_2 = vects_nda[:,:,y//2:,:x//2,:]
#     vects_2_mean = minmax_lambda([np.mean(vects_2,axis=(1,2,3,4)),lower,upper])
#     vects_3 = vects_nda[:,:,y//2:,x//2:,:]
#     vects_3_mean = minmax_lambda([np.mean(vects_3,axis=(1,2,3,4)),lower,upper])
#     vects_4 = vects_nda[:,:,:y//2,x//2:,:]
#     vects_4_mean = minmax_lambda([np.mean(vects_4,axis=(1,2,3,4)),lower,upper])
    
    
#     # direction per corner
#     dir_1 = directions[:,:,:y//2,:x//2] # top left, plot
#     dir_1_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_2 = directions[:,:,y//2:,:x//2] # top right, plot
#     dir_2_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_3 = directions[:,:,y//2:,x//2:] # bottom left
#     dir_3_mean = minmax_lambda([np.mean(dir_3,axis=(1,2,3)),lower,upper])
#     dir_4 = directions[:,:,:y//2,x//2:] # bottom right
#     dir_4_mean = minmax_lambda([np.mean(dir_4,axis=(1,2,3)),lower,upper])
    

#     # norm per corner
#     norm_mean = minmax_lambda([nda_1d_mean, mid, upper])
#     norm_1 = norm_nda[:,:,:y//2,:x//2]
#     norm_1_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_2 = norm_nda[:,:,y//2:,:x//2]
#     norm_2_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_3 = norm_nda[:,:,y//2:,x//2:]
#     norm_3_mean = minmax_lambda([np.mean(norm_3,axis=(1,2,3)),mid,upper])
#     norm_4 = norm_nda[:,:,:y//2,x//2:]
#     norm_4_mean = minmax_lambda([np.mean(norm_4,axis=(1,2,3)),mid,upper])
    

    ind = detect_phases(dir_1d_mean=dir_1d_mean[:length], length=length)
    
    
    #print('ed: {}, ms: {}, es: {}, pf: {}, md: {}'.format(*ind))
    
        # plot the mean/max norm for one patient oveer time
    if plot:
        
        
        # VECT 2D+t
        vect_2d_t = vects_nda[:,z]
        fig1 = show_2D_or_3D(vect_2d_t,allow_slicing=False, f_size=(25,2))
        ax1 = fig1.get_axes()[0]
        _ = ax1.set_ylabel('phi_t \nmid-cavity')

        
        # DIR 2D+t
        dir_2d_t = directions[:,z]
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        #dir_2d_t = minmax_lambda([dir_2d_t,lower,upper])
        div_cmap = 'bwr'
        fig = show_2D_or_3D(dir_2d_t,allow_slicing=False, f_size=(25,2),cmap=div_cmap)
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('alpha_t \nmid-cavity')
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[len(ax.get_images())//2], cax=cax, orientation='horizontal')

        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # DIR 2D T x Z
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        directions_tz = directions.mean(axis=(2,3)) # original: norm()
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        _ = ax.imshow(directions_tz.T, aspect='auto', label='aasd', cmap=div_cmap, origin='lower')
        _ = ax.set_xticks(gtind, minor=False)
        ax2 = ax.twinx()
        _ = ax2.plot(dir_1d_mean, c='black',label='dir 1d+t')
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('dir z+t\napex:base')
        
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1
        
        #ax=fig.add_subplot(rows,1,pos)
        #_ = ax.plot(dir_1d_mean, c='black',label='dir 1d+t')
        #_ = ax.plot(dir_1_mean, label='top left')
        #_ = ax.plot(dir_2_mean, label='top right')
        #_ = ax.plot(dir_3_mean, label='bottom left')
        #_ = ax.plot(dir_4_mean, label='bottom right')
        #ax.legend(loc='upper left')
        #ax.set_xticks(gtind, minor=False)
        #ax.margins(0.022,0)
        #ax.label_outer()
        #pos=pos+1
        #plt.subplots_adjust(wspace=0, hspace=0.1)
        #plt.show()
        norm_cmap = 'hot'
        
        # NORM 2D + t
        norm_2d_t = norm_nda[:,z]
        norm_2d_t = minmax_lambda([norm_2d_t,mid,upper])
        fig = show_2D_or_3D(norm_2d_t,allow_slicing=False, f_size=(25,2),cmap=norm_cmap, interpolation='none')
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('norm 2d+t\nmid-cavity')
        #ax.margins(0,0)
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[0], cax=cax, orientation='horizontal')
        
        
        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # NORM 2D TxZ
        norm2d = minmax_lambda([norm_nda.mean(axis=(2,3)),mid,upper])
        _ = ax.imshow(norm2d.T, aspect='auto', origin='lower',cmap=norm_cmap, interpolation='none')
        _ = ax.set_xticks(gtind, minor=False)
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('norm z+t\napex:base')
        ax2 = ax.twinx()
        _ = ax2.plot(nda_1d_mean, c='black',label='norm 1d+t')
        #ax.spines['top'].set_visible(False)
        #_ = ax.legend(loc='upper left')
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1

        """ax=fig.add_subplot(rows,1,pos)
        _ = ax.plot(norm_mean, label='norm 1d+t')
        #_ = ax.plot(norm_1_mean, label='top left')
        #_ = ax.plot(norm_2_mean, label='top right')
        #_ = ax.plot(norm_3_mean, label='bottom left')
        #_ = ax.plot(norm_4_mean, label='bottom right')
        ax.legend(loc='upper left')
        ax.set_xticks(gtind, minor=False)
        ax.margins(0.022,0)
        ax.label_outer()
        pos = pos+1
        plt.subplots_adjust(wspace=0, hspace=0.1)"""
        #plt.tight_layout()
        plt.show()

        
    
    return ind
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
p = 2#215
gt_length = int(gt_len[p,:,0].sum())
ind = np.argmax(gt[p][:gt_length],axis=0)
get_phases_from_vects(nda_vects[p][:gt_length], length=gt_length,plot=True,dir_axis=0, gtind=ind)
# Tue, 01 Mar 2022 10:48:50
@interact
def compare_phases(i=(0,nda_vects.shape[0]-1), plot=True):
    import matplotlib as mpl
    mpl.rcParams.update(mpl.rcParamsDefault)
    figsize = (25,1)
    weight = 1
    z = 0
    
    phases = ['ED', 'MS', 'ES', 'PF', 'MD']
    cardiac_cycle_length = int(gt_len[i,:,-1].sum())
    # crop or not
    cardiac_cycle_length_ = gt[i].shape[0]-3 # plot the full length minus border for visualisation
    
    vects = nda_vects[i][:cardiac_cycle_length]
    
    
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0) # 
    ind_pred = np.argmax(pred[i][:],axis=0)
    
    temp = mov[i,:cardiac_cycle_length,z] # cardiac_cycle_length_
    fig = show_2D_or_3D(temp,allow_slicing=False)
    ax = fig.get_axes()[0]
    _ = ax.set_ylabel('CMR 2d+t)\nmid-cavity')
    plt.show()
    
    fig= plt.figure(figsize=figsize)
    ax=fig.add_subplot(111)
    ax.set_xticks(ind, minor=False)
    ax.set_yticks([0, 1, 2, 3, 4], minor=False)
    ax.set_yticklabels(phases, rotation=0)
    ind = np.array(ind)
    onehot_gt = np.zeros((ind.size, cardiac_cycle_length)) # cardiac_cycle_length_
    onehot_gt[np.arange(ind.size), ind] = weight
    #ax.imshow(gt[i][:cardiac_cycle_length].T,aspect='auto') # with uncertainity
    ax.imshow(onehot_gt,aspect='auto', cmap='bone') # gt as binary phase2time mapping
    _ = ax.set_yticklabels(phases)
    ax.set_title('gt')
    ax.margins(0,0)
    
    indices = get_phases_from_vects(vects, length=cardiac_cycle_length, plot=plot,dir_axis=0, gtind=ind, figsize=figsize)

    indices = np.array(indices)
    onehot = np.zeros((indices.size, cardiac_cycle_length)) # cardiac_cycle_length_
    onehot[np.arange(indices.size), indices] = weight
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xticks(indices, minor=False)
    ax.set_yticks([0, 1, 2, 3, 4], minor=False)
    ax.set_yticklabels(phases, rotation=0)
    plt.imshow(onehot,aspect='auto', cmap='bone')
    _ = ax.set_yticklabels(phases)
    ax.margins(0,0)
    #ax.set_title('prediction')
    plt.show()
    
    print(phases)
    print('gt:', ind)
    print('u: ', indices)
    
    
    # this would plot the supervised prediction
    #print('p: ', ind_pred)
    """fig, ax = plt.subplots(figsize=figsize)
    ax.set_xticks(ind_pred, minor=False)
    ax.set_yticks([0, 1, 2, 3, 4], minor=False)
    ax.set_yticklabels(phases, rotation=45)
    plt.imshow(pred[i][:cardiac_cycle_length].T,aspect='auto')
    _ = ax.set_yticklabels(phases);plt.show()"""
    # patient 107 gcn --> good curve for plotting
    # 200 no cut
    # 125 no cut
    # 194 moderate cut 
    # 134 strong cut-off
# Tue, 01 Mar 2022 10:48:53
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:48:54
pred_u = np.zeros_like(gt)
upred_ind = []
cycle_len=[]
print(pred_u.shape)
for i in range(pred_u.shape[0]):
    weight = 1
    
    cardiac_cycle_length = int(gt_len[i,:,0].sum())
    cycle_len.append(cardiac_cycle_length)
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
    #print(gt[i][:cardiac_cycle_length].T.shape)
    
    indices = get_phases_from_vects(nda_vects[i][:cardiac_cycle_length], length=cardiac_cycle_length, gtind=ind,plot=False,dir_axis=0)
    upred_ind.append(indices)
    indices = np.array(indices)
    onehot = np.zeros((indices.size, cardiac_cycle_length))
    onehot[np.arange(indices.size), indices] = weight
    pred_u[i][0:cardiac_cycle_length] = onehot.T
upred_ind=np.stack(upred_ind, axis=0)
cycle_len = np.stack(cycle_len, axis=0)
# re-create a compatible shape for the metric fn
gt_ = np.stack([gt,gt_len], axis=1)
pred_ = np.stack([pred_u,np.zeros_like(pred_u)], axis=1)

# create a dataframe for further plots
from src.utils.Metrics import meandiff
phases = ['ED', 'MS', 'ES', 'PF', 'MD']
res = meandiff(gt_, pred_, apply_sum=False, apply_average=False)
df = pd.DataFrame(res.numpy(), columns=phases)
# Tue, 01 Mar 2022 10:49:11
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:49:12
# get the patient idx, per label where this method fails the most
label = 0
error_thres = 6
if type(res) != np.ndarray: res.numpy()
print('label:{} \naFD>{}\nPatient idx: {}'.format(phases[label],error_thres,np.where(res[:,label]>error_thres)[0]))
print('gt length: {}'.format(cycle_len[res[:,label]>error_thres]))
print('gt indices: {}'.format(gt_ind[:,label][(res[:,label]>error_thres)]))
print('pred indices: {}'.format(upred_ind[:,label][(res[:,label]>error_thres)]))
print('aFD: {}'.format(res[:,label][res[:,label]>error_thres]))
# Tue, 01 Mar 2022 10:49:12
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:49:13
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:49:14
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:49:47
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:53:13
# load the vectors
pathstovectnpy = sorted(glob.glob(os.path.join(vects_chooser.selected, 'moved', '*vects*.npy')))
print(pathstovectnpy)
nda_vects = np.concatenate([np.load(path_) for path_ in pathstovectnpy],axis=0)
print(nda_vects.shape)
# Tue, 01 Mar 2022 10:53:36
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:53:37
# load the phase gt and pred
pred_path = os.path.join(vects_chooser.selected, 'pred')
pathtsophasenpy = sorted(glob.glob(os.path.join(pred_path, '*gtpred*.npy')))
print(pathtsophasenpy)
nda_phase = np.concatenate([np.load(path_) for path_ in pathtsophasenpy],axis=1)
print(nda_phase.shape)
gt_, pred_ = np.split(nda_phase, axis=0,indices_or_sections=2)
print(gt_.shape)
gt= gt_[0,:,0]
pred = pred_[0,:,0]
print(gt.shape)
gt_len = gt_[0,:,1]
# Tue, 01 Mar 2022 10:53:37
# load some moved examples for easier understanding of the dimensions
pathtomoved = sorted(glob.glob(os.path.join(vects_chooser.selected, 'moved','*moved*.npy')))
print(len(pathtomoved))
mov = np.concatenate([np.load(path_) for path_ in pathtomoved],axis=0)
print(mov.shape) # patient,time,spatial-z,y,x,channel, e.g.: 69,40,16,64,64,1
# Tue, 01 Mar 2022 10:53:46
# plot one moved timestep of one patient = one 3D volume 
# should go from apex to base
from src.visualization.Visualize import show_2D_or_3D
temp = mov[10,0,...,0]
_ = show_2D_or_3D(temp, cmap='gray', interpolation='none')
# Tue, 01 Mar 2022 10:53:47
import tensorflow as tf
from tensorflow.image import ssim
ssim(mov[10,0], mov[10,1],max_val=1.0, filter_size=11,
                          filter_sigma=1.5, k1=0.01, k2=0.03).numpy().mean()
# Tue, 01 Mar 2022 10:53:47
from skimage.metrics import structural_similarity as ssim
ssim(mov[10,0], mov[10,1],multichannel=True)
# Tue, 01 Mar 2022 10:53:47
plt.imshow(gt[0].T)
# Tue, 01 Mar 2022 10:53:47
plt.imshow(pred[0].T)
# Tue, 01 Mar 2022 10:53:47
import tensorflow as tf
import sys
# returns a matrix with the indicies as values, similar to np.indicies
def get_idxs_tf(x):
    return tf.cast(
        tf.reshape(tf.where(tf.ones((x[0], x[1], x[2]))), (x[0], x[1], x[2], 3)),
        tf.float32)


# returns a matrix with vectors pointing to the center
def get_centers_tf(x):
    return tf.cast(
        tf.tile(tf.convert_to_tensor([x[0] // 2, x[1] // 2, x[2] // 2])[tf.newaxis, tf.newaxis, tf.newaxis, ...],
                (x[0], x[1], x[2], 1)), tf.float32)

def get_angle_tf(a, b):
    # this should work for batches of n-dimensional vectors
    # α = arccos[(a · b) / (|a| * |b|)]
    # |v| = √(x² + y² + z²)
    """
    in 3D space
    If vectors a = [xa, ya, za], b = [xb, yb, zb], then:
    α = arccos[(xa * xb + ya * yb + za * zb) / (√(xa2 + ya2 + za2) * √(xb2 + yb2 + zb2))]
    """
    import math as m
    pi = tf.constant(m.pi)
    b = tf.cast(b, dtype=a.dtype)
    inner = tf.einsum('...i,...i->...', a, b)
    norms = tf.norm(a, ord='euclidean', axis=-1) * tf.norm(b, ord='euclidean', axis=-1)  # [...,None]
    cos = inner / (norms + sys.float_info.epsilon)
    #rad = tf.math.acos(tf.clip_by_value(cos, -1.0, 1.0))
    # rad2deg conversion
    #deg = rad * (180.0/pi)
    return cos[..., tf.newaxis]

dim = [16,64,64]
idx = get_idxs_tf(dim)
c = get_centers_tf(dim)
centers = c - idx
centers_tensor = centers[tf.newaxis, ...]
flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x, centers_tensor), name='flow2direction')
minmax_lambda = lambda x : x[1] + (((x[0] - np.min(x[0])) * (x[2]-x[1]))/(np.max(x[0]) - np.min(x[0])))
# define some central params
lower, mid, upper = -1,0, 1
# Tue, 01 Mar 2022 10:53:47
# align norm an direction by the cardiac phase ED
# resample all 1D feature vectors to the same length
# this should help to validate the pre-defined rules, and if both together explains the cardiac phases
target_t = 30
norms = []
dirs= []
cycle_len=[]
gt_ind = []
gt_ind_rolled = []
xval = np.linspace(0,1,target_t)

for i,n in enumerate(nda_vects[:]):
    cardiac_cycle_length = int(gt_len[i,:,0].sum())
    cycle_len.append(cardiac_cycle_length)
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
    gt_onehot = gt[i][:cardiac_cycle_length]
    gt_ind.append(ind)
    
    dir_axis=0
    gt_ed = ind[0]
    #assert cardiac_cycle_length>=gt_ed
    temp = n[:cardiac_cycle_length]
    
    dim = temp.shape[1:-1]
    #print(dim)
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    #print('centers: ',c.dtype)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    # direction
    directions = flow2direction_lambda(temp)[...,0]
    # direction mean
    directions = np.mean(directions,axis=(1,2,3))
    # direction ed aligned
    directions = np.roll(directions, -1*gt_ed)
    # direction interpolate to unique length
    f = interp1d(np.linspace(0,1,directions.shape[0]), directions, kind='linear')
    directions = f(xval)
    # direction min/max normalised between -1,1

    directions = minmax_lambda([directions,lower,upper])
    
    # magnitude/norm as mean
    norm_full = np.linalg.norm(temp, axis=-1)
    norm_nda = norm_full.mean(axis=(1,2,3))
    # norm ed aligned
    norm_nda = np.roll(norm_nda, -1*gt_ed)
    # interpolate to unique length
    #f = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')
    #norm_nda = f(xval)
    norm_nda = np.interp(xval, np.linspace(0,1,norm_nda.shape[0]), norm_nda)
    # norm min/max aligned to 0,1
    norm_nda = minmax_lambda([norm_nda,mid,upper])
    
    # roll, scale, round and clip the gt indicies, to get an aligned distribution of the labels
    gt_onehot_rolled = np.roll(gt_onehot, -1*gt_ed, axis=0)
    resize_factor = target_t/cardiac_cycle_length
    gt_onehot_rolled = np.argmax(gt_onehot_rolled,axis=0)
    gt_onehot_rolled = np.clip(np.rint(gt_onehot_rolled * resize_factor),a_min=0,a_max=target_t-1)
    gt_ind_rolled.append(gt_onehot_rolled)
    
    norms.append(norm_nda)
    dirs.append(directions)
# Tue, 01 Mar 2022 10:53:59
# get the mean values
gt_ind_scaled = np.stack(gt_ind_rolled, axis=0)
gt_ind = np.stack(gt_ind,axis=0)
cycle_len = np.stack(cycle_len, axis=0)
norms = np.stack(norms, axis=0)
dirs = np.stack(dirs, axis=0)
norms_m = [norms.mean(axis=0)]
dirs_m=[dirs.mean(axis=0)]
# Tue, 01 Mar 2022 10:53:59
# plot the distribution of the aligned and scaled gt indicies
fig, (ax1,ax2, ax3) = plt.subplots(1,3, figsize=(15,5))
temp = pd.DataFrame(gt_ind_scaled, columns=phases)
# TODO check this outlier with MD ==2
temp = temp[temp['MD']>10]
print(temp.mean(), temp.std())
ax1 = sb.violinplot(data=temp, ax=ax1)
ax1 = sb.stripplot(data=temp, ax=ax1)
#_ = temp.plot(kind='box', ax=ax1)
_ = temp.plot(kind='line', ax=ax2, legend=None)
_ = temp.plot(kind='hist', ax=ax3, legend=None)
ax3.set_ylabel('')
# Tue, 01 Mar 2022 10:53:59
# seaborn
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax.margins(0,0)
for n in dirs:
    _ = ax.plot(n, alpha=0.5, zorder=0)

df = pd.DataFrame(dirs).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='blue', label=r'$\alpha_t$',ci='sd',err_style='bars',zorder=2,legend=False,ax=ax)
#_ = ax.set_title('Mean direction +/- SD - aligned at ED')
_ = ax.set_xticks(np.rint(gt_ind_scaled.mean(axis=0)), minor=False)
_ = ax.set_xlabel('Time (t) - linear interpolated to 40 frames')
_ = ax.set_ylabel(r'Angle $\alpha_t$ of $\phi_t$')
_ = ax.legend(loc='upper left')
ax2 = ax.twinx()
ax2.margins(0,0)
df = pd.DataFrame(norms).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='black', label=r'|$\phi_t$|',ci='sd', linestyle='dashed',err_style='bars',zorder=2,legend=False,ax=ax2)
_ = ax2.set_ylabel(r'Norm |$\phi_t$|')

_ = ax2.legend(loc='upper right')
# Tue, 01 Mar 2022 10:53:59
folds_chooser = FileChooser(os.path.join(os.getcwd(),'/mnt/ssd/git/dynamic-cmr-models/exp/phasereg_v3/acdc/'), '')
display(folds_chooser)
# Tue, 01 Mar 2022 10:53:59
# define df and helper for pathology extractio
# this should have the same order as our inference data
#df_meta = pd.read_csv(folds_chooser.selected)
df_meta = pd.read_csv('/mnt/sds/sd20i001/sven/data/acdc/02_imported_4D_unfiltered/df_kfold.csv')
df_meta = df_meta.loc[:, ~df_meta.columns.str.contains('^Unnamed')]
df_meta = df_meta[df_meta.patient != 'patient090']  #we excluded this patient

def get_msk_for_pathology(df_, pathology='minf'):
    msk = []
    for f in [0,1,2,3,4]:
        patients = df_[df_.fold.isin([f])]
        patients = patients[patients['phase']=='ED']
        pat = patients[patients['modality'] == 'test']['patient'].str.lower().unique()
        sub_df = patients[patients.patient.isin(pat)].drop_duplicates(ignore_index=True, subset='patient')
        sub_df = sub_df.drop('x_path',axis=1).drop('y_path',axis=1)
        msk.append(sub_df['pathology'].str.lower()==pathology.lower())
    return np.concatenate(msk)
df_meta
# Tue, 01 Mar 2022 10:54:00
# interactive plot per pathology
@interact
def plot_per_pathology(p=df_meta.pathology.unique()):
    print(p)
    msk = get_msk_for_pathology(df_meta, p)

    import seaborn as sb
    sb.set_context('paper')
    sb.set(font_scale = 2)
    fig, ax = plt.subplots(figsize=(25,5))
    ax.margins(0,0)
    dirs_minf = dirs[msk]
    norms_minf = norms[msk]
    for n in dirs_minf:
        _ = ax.plot(n, alpha=0.5, zorder=0)

    df = pd.DataFrame(dirs).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, color='blue', label=r'$\alpha_t$',ci='sd',err_style='bars',zorder=2,legend=False,ax=ax)
    #_ = ax.set_title('Mean direction +/- SD - aligned at ED')
    _ = ax.set_xlabel('Time (t) - linear interpolated to 40 frames')
    _ = ax.set_ylabel(r'Angle $\alpha_t$ of $\phi_t$')
    _ = ax.legend(loc='upper left')
    ax2 = ax.twinx()
    ax2.margins(0,0)
    df = pd.DataFrame(norms).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, color='black', label=r'|$\phi_t$|',ci='sd', linestyle='dashed',err_style='bars',zorder=2,legend=False,ax=ax2)
    _ = ax2.set_ylabel(r'Norm |$\phi_t$|')

    _ = ax2.legend(loc='upper right')
# Tue, 01 Mar 2022 10:54:00
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax2 = ax.twinx()
ax2.margins(0,0)
ax.margins(0,0)
for p in df_meta.pathology.unique():
    msk = get_msk_for_pathology(df_meta, p)
    dirs_p = dirs[msk]
    norms_p = norms[msk]
    #for n in dirs_minf:
    #    _ = ax.plot(n, alpha=0.5, zorder=0)
    
    df = pd.DataFrame(dirs_p).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, label=r'$\alpha_t$ {}'.format(p),ci='sd',err_style=None,zorder=2,legend=False,ax=ax)
    #_ = ax.set_title('Mean direction +/- SD - aligned at ED')
    _ = ax.set_xlabel('Time (t) - linear interpolated to 40 frames')
    _ = ax.set_ylabel(r'Angle $\alpha_t$ of $\phi_t$')
    _ = ax.legend(loc='upper left')
    df = pd.DataFrame(norms_p).melt()
    _ = sb.lineplot(x='variable', y='value',data=df, label=r'|$\phi_t$| {}'.format(p),ci='sd', linestyle='dashed',err_style=None,zorder=2,legend=False,ax=ax2)
    _ = ax2.set_ylabel(r'Norm |$\phi_t$|')
    _ = ax2.legend(loc='upper right')
    
plt.show()
# Tue, 01 Mar 2022 10:54:00
# predict per pathology, on aligned curves
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
ind_all = []

for p in df_meta.pathology.unique():
    msk = get_msk_for_pathology(df_meta, p)
    dirs_p = dirs[msk]
    norms_p = norms[msk]
    ind_ = np.stack([detect_phases(elem, len(elem)) for elem in dirs_p], axis=0)
    _ = pd.DataFrame(ind_, columns=phases).plot(kind='box')
    ind_all.extend(ind_)
# Tue, 01 Mar 2022 10:54:01
fig, ax = plt.subplots(1,5, figsize=(20,2))
_ = pd.DataFrame(gt_ind_scaled, columns=phases).hist(ax=ax)
_ = [a.set_xlim(0,30) for a in ax.flatten()]
# Tue, 01 Mar 2022 10:54:01
# scatterplot per pathology, t as dymension
import sklearn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
y = np.zeros(99)
for i,p in enumerate(df_meta.pathology.unique()):
    msk = get_msk_for_pathology(df_meta, p)
    y[msk] = i
low_dim = TSNE(n_components=2).fit_transform(norms)
#low_dim = PCA(n_components=2).fit_transform(norms)
_ =plt.scatter(low_dim[:,0],low_dim[:,1], c=y)
# Tue, 01 Mar 2022 10:54:02
# plot one line per patient - norm
fig, ax = plt.subplots(figsize=(25,5))
for n in norms:
    _ = plt.plot(n)
_ = ax.set_title('Magnitudes aligned at ED, resampling shape 40,')
# Tue, 01 Mar 2022 10:54:02
# plot one line per direction
fig, ax = plt.subplots(figsize=(25,5))
for n in dirs:
    _ = plt.plot(n)
_ = ax.set_title('Directions aligned at ED, resampling shape 40,')
# Tue, 01 Mar 2022 10:54:03
from src.visualization.Visualize import show_2D_or_3D
from src.data.Preprocess import clip_quantile
nda_temp = nda_vects[0]
# norm_1 = nda_temp[:,:,:32,:32]# top left
norm_1 = nda_temp[:,:,:32,32:]# top right
norm_1 = np.linalg.norm(norm_1, axis=-1)
nda_norm_ = np.linalg.norm(nda_temp, axis=-1)
plt.hist(nda_norm_.flatten());plt.show()
nda_norm = clip_quantile(nda_norm_, 0.99)
nda_norm = minmax_lambda([nda_norm,0,1])
plt.hist(nda_norm.flatten())
plt.show()
# mask phi with a threshold norm matrix 
nda_msk = (nda_norm>=0.2).astype(np.float32)
nda_temp = nda_norm * nda_msk
#plt.hist(nda_temp.flatten())
print(nda_temp.shape)
_ = show_2D_or_3D(nda_norm_[:,0], allow_slicing=True)
_ = show_2D_or_3D(nda_temp[:,0], allow_slicing=True)
_ = show_2D_or_3D(nda_temp[:,-1], allow_slicing=True)

# top right
_ = show_2D_or_3D(norm_1[:,0], allow_slicing=True,cmap='gray', interpolation='none')
# Tue, 01 Mar 2022 10:54:04
fig, ax = plt.subplots(figsize=(25,3))
plt.plot(np.argmax(gt[0], axis=1))
ax.set_yticks([0, 1, 2, 3, 4], minor=False)
phases = ['ED', 'MS', 'ES', 'PF', 'MD']
_ = ax.set_yticklabels(phases)
# Tue, 01 Mar 2022 10:54:05
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:54:06
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:54:07
# plot the mean/max norm for one patient oveer time
nda_1d_max = np.max(nda_temp,axis=(1,2,3))
nda_1d_mean = np.mean(nda_temp,axis=(1,2,3))
nda_1d_sum = np.sum(nda_temp,axis=(1,2,3))

fig, ax = plt.subplots(figsize=(25,3))
ax.set_title('max norm')
_ = plt.plot(nda_1d_max); plt.show()

fig, ax = plt.subplots(figsize=(25,3))
ax.set_title('mean norm')
_ = plt.plot(nda_1d_mean)

fig, ax = plt.subplots(figsize=(25,3))
ax.set_title('mean sum')
_ = plt.plot(nda_1d_sum)
#ax.set_ylim(0.0,0.15)
# Tue, 01 Mar 2022 10:54:07
temp = np.arange(10)
temp[:0]
# Tue, 01 Mar 2022 10:54:07
def detect_phases(dir_1d_mean, length):
    
    import scipy.signal as sig
    
        # MS
    # Global min of f(x)
    ms = np.argmin(dir_1d_mean)
    ms = ms -1 # take the bucket before the first min peak
    
    
    # ES
    # First time f(x)>0 after MS
    cycle = np.concatenate([dir_1d_mean[ms:], dir_1d_mean[:ms]])
    temp_ = 0
    es_found=False
    negative_slope = False
    for idx,elem in enumerate(cycle):
        if elem<0:
            negative_slope=True
            temp_ = idx
        elif elem>=0 and negative_slope:
            es_found = True
            #temp_ = idx
            break # stop after first zero-transition
    if es_found:
        es = ms + temp_
        #es = es-1
    else:
        es = ms + 1 # the frame after ms, fallback
    if es>=length:
        es = np.mod(es,length)
        print('ES overflow: {}, ms:{}'.format(es,ms))
     
    
    # PF
    # First peak after ES, min height 0.6
    seq = dir_1d_mean[es:]
    peaks, prop = sig.find_peaks(seq, height=0.6)#height=0.6 we normalise between -1 and 1, PF should be close to argmax

    if len(peaks>0):
        pf = es + peaks[0] # take the peak after es
        pf = pf -1
    else:
        print('pf not clear, set to ES {} + 1'.format(es))
        pf = es + 1
    pf = np.mod(pf, length)
      

    # ED
    # Between pf and ms: last time f(x) cross zero from positive to negative
    # a priori knowledge ED needs a minimal distance of 2 frames towards MS
    # CHANGED the minimal distance between ED and MS
    cycle = np.concatenate([dir_1d_mean[pf:], dir_1d_mean[:ms]])
    #print(cycle)
    ed_found = False
    last_idx_positive = True # we start at the pf, which is the peak(dir)
    for idx,elem in enumerate(cycle):
        
        if elem>=0:
            last_idx_positive = True
        # remember the last idx before the direction gets negative the last time before ms
        elif elem<0 and last_idx_positive: # first time direction negative
            ed_found = True # for fallbacks
            temp_ = idx # idx before negative direction
            #print('found transition at: {}'.format(idx))
            last_idx_positive = False # remember only the first idx after transition
        
    if ed_found:
        ed = pf + temp_
        #print('ed:{}, pf:{}, temp_:{}, lenght: {}'.format(ed,pf,temp_,length))
    else: 
        # if we dont find a transition from positive to negative, take the idx which is the closest to zero
        temp_ = np.argmin(np.abs(cycle)) # make sure we have a minimal distance
        ed = pf + temp_
        print('ED: no transition found between {}-{} , closest id to 0: {}, ed = {}'.format(pf,ms,temp_,ed))
    
    if ed>=length:
        #print('gt ed: {}, ed: {}, length: {}'.format(gted,ed,length))
        ed = np.mod(ed,length)
        #print('mod ed: {}'.format(ed))
    #ed = ed-1 # take the bucket before negative
        
    # MD
    # Middle between PF and ED
    ed_slice_idx = ed
    if ed_slice_idx<=pf: # ed overflow --> beginning of cmr stack
        ed_slice_idx = length + ed
    md = (pf+ed_slice_idx)//2 # the bucket after the middle
    md = md + 1
    md = np.mod(md,length)
    
    return np.array([ed,ms,es,pf,md])
# Tue, 01 Mar 2022 10:54:07
# try to predict the phases unsupervised just from the norm/direction curve

def get_phases_from_vects(vects_nda, length=-1, plot=True, dir_axis=0, gtind=[], figsize=(25,3)):
    import scipy.signal
    import scipy.signal as sig
    import scipy.ndimage
    from scipy.ndimage import gaussian_filter1d
    
    gted=gtind[0]
    
    # signal smoothing
    kernel_size = 4
    kernel = np.ones(kernel_size) / kernel_size
    z = 0
    
    
    dim_z = vects_nda.shape[1]
    #vects_nda = vects_nda[:int(length)]
    
    
    # define some helper lambdas
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    dim = vects_nda.shape[1:-1]
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]

    
    # norm of the vector
    norm_nda = np.linalg.norm(vects_nda[...,dir_axis:], axis=-1)
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    norm_nda_thres = clip_quantile(norm_nda, 0.9)
    norm_nda = minmax_lambda([norm_nda,0,1])
    norm_msk = norm_nda>=0.0 # this could be used to mask v_t by a threshold norm
    norm_nda = norm_nda * norm_msk
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    nda_1d_mean = np.mean(norm_nda,axis=(1,2,3))
    nda_1d_max = np.max(norm_nda,axis=(1,2,3))
    nda_1d_median = np.median(norm_nda,axis=(1,2,3))
    nda_1d_mean_smooth = np.convolve(nda_1d_mean, kernel, mode='same')
    #print(nda_1d_mean_smooth.shape, nda_1d_mean.shape)
    
    
    # direction relative to the center
    
    directions = flow2direction_lambda(vects_nda)[...,0].numpy()
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    directions = minmax_lambda([directions,lower,upper])
    #plt.hist(directions.flatten());plt.show()
    directions = directions * norm_msk
    #plt.hist(directions.flatten());plt.show()
    
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    dir_1d_mean = np.mean(directions,axis=(1,2,3))
    dir_1d_median = np.median(directions,axis=(1,2,3))

    
    # Find phases
    # 1st smooth the direction with a rolling averge (kernelsize=4)
    # or via gaussian filter
    # 2nd min/max normalise each direction vector in a range [-1:1]
    
    
    dir_1d_mean = gaussian_filter1d(dir_1d_mean, sigma=2, mode='wrap') # original: dir_1d_mean
    
    ################# CHANGED !!!!!! ######################## RECENTLY NO CHANGES
    # min/max scaling
    dir_1d_mean = minmax_lambda([dir_1d_mean,lower,upper])
    
    
#     # test with corner specific vector fields
#     y,x = directions.shape[-2:]
    
#     vects_mean = minmax_lambda([np.mean(vects_nda,axis=(1,2,3,4)),lower,upper])
#     vects_1 = vects_nda[:,:,:y//2,:x//2,:]
#     vects_1_mean = minmax_lambda([np.mean(vects_1,axis=(1,2,3,4)),lower,upper])
#     vects_2 = vects_nda[:,:,y//2:,:x//2,:]
#     vects_2_mean = minmax_lambda([np.mean(vects_2,axis=(1,2,3,4)),lower,upper])
#     vects_3 = vects_nda[:,:,y//2:,x//2:,:]
#     vects_3_mean = minmax_lambda([np.mean(vects_3,axis=(1,2,3,4)),lower,upper])
#     vects_4 = vects_nda[:,:,:y//2,x//2:,:]
#     vects_4_mean = minmax_lambda([np.mean(vects_4,axis=(1,2,3,4)),lower,upper])
    
    
#     # direction per corner
#     dir_1 = directions[:,:,:y//2,:x//2] # top left, plot
#     dir_1_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_2 = directions[:,:,y//2:,:x//2] # top right, plot
#     dir_2_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_3 = directions[:,:,y//2:,x//2:] # bottom left
#     dir_3_mean = minmax_lambda([np.mean(dir_3,axis=(1,2,3)),lower,upper])
#     dir_4 = directions[:,:,:y//2,x//2:] # bottom right
#     dir_4_mean = minmax_lambda([np.mean(dir_4,axis=(1,2,3)),lower,upper])
    

#     # norm per corner
#     norm_mean = minmax_lambda([nda_1d_mean, mid, upper])
#     norm_1 = norm_nda[:,:,:y//2,:x//2]
#     norm_1_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_2 = norm_nda[:,:,y//2:,:x//2]
#     norm_2_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_3 = norm_nda[:,:,y//2:,x//2:]
#     norm_3_mean = minmax_lambda([np.mean(norm_3,axis=(1,2,3)),mid,upper])
#     norm_4 = norm_nda[:,:,:y//2,x//2:]
#     norm_4_mean = minmax_lambda([np.mean(norm_4,axis=(1,2,3)),mid,upper])
    

    ind = detect_phases(dir_1d_mean=dir_1d_mean[:length], length=length)
    
    
    #print('ed: {}, ms: {}, es: {}, pf: {}, md: {}'.format(*ind))
    
        # plot the mean/max norm for one patient oveer time
    if plot:
        
        
        # VECT 2D+t
        vect_2d_t = vects_nda[:,z]
        fig1 = show_2D_or_3D(vect_2d_t,allow_slicing=False, f_size=(25,2))
        ax1 = fig1.get_axes()[0]
        _ = ax1.set_ylabel('phi_t \nmid-cavity')

        
        # DIR 2D+t
        dir_2d_t = directions[:,z]
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        #dir_2d_t = minmax_lambda([dir_2d_t,lower,upper])
        div_cmap = 'bwr'
        fig = show_2D_or_3D(dir_2d_t,allow_slicing=False, f_size=(25,2),cmap=div_cmap)
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('alpha_t \nmid-cavity')
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[len(ax.get_images())//2], cax=cax, orientation='horizontal')

        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # DIR 2D T x Z
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        directions_tz = directions.mean(axis=(2,3)) # original: norm()
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        _ = ax.imshow(directions_tz.T, aspect='auto', label='aasd', cmap=div_cmap, origin='lower')
        _ = ax.set_xticks(gtind, minor=False)
        ax2 = ax.twinx()
        _ = ax2.plot(dir_1d_mean, c='black',label='dir 1d+t')
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('dir z+t\napex:base')
        
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1
        
        #ax=fig.add_subplot(rows,1,pos)
        #_ = ax.plot(dir_1d_mean, c='black',label='dir 1d+t')
        #_ = ax.plot(dir_1_mean, label='top left')
        #_ = ax.plot(dir_2_mean, label='top right')
        #_ = ax.plot(dir_3_mean, label='bottom left')
        #_ = ax.plot(dir_4_mean, label='bottom right')
        #ax.legend(loc='upper left')
        #ax.set_xticks(gtind, minor=False)
        #ax.margins(0.022,0)
        #ax.label_outer()
        #pos=pos+1
        #plt.subplots_adjust(wspace=0, hspace=0.1)
        #plt.show()
        norm_cmap = 'hot'
        
        # NORM 2D + t
        norm_2d_t = norm_nda[:,z]
        norm_2d_t = minmax_lambda([norm_2d_t,mid,upper])
        fig = show_2D_or_3D(norm_2d_t,allow_slicing=False, f_size=(25,2),cmap=norm_cmap, interpolation='none')
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('norm 2d+t\nmid-cavity')
        #ax.margins(0,0)
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[0], cax=cax, orientation='horizontal')
        
        
        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # NORM 2D TxZ
        norm2d = minmax_lambda([norm_nda.mean(axis=(2,3)),mid,upper])
        _ = ax.imshow(norm2d.T, aspect='auto', origin='lower',cmap=norm_cmap, interpolation='none')
        _ = ax.set_xticks(gtind, minor=False)
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('norm z+t\napex:base')
        ax2 = ax.twinx()
        _ = ax2.plot(nda_1d_mean, c='black',label='norm 1d+t')
        #ax.spines['top'].set_visible(False)
        #_ = ax.legend(loc='upper left')
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1

        """ax=fig.add_subplot(rows,1,pos)
        _ = ax.plot(norm_mean, label='norm 1d+t')
        #_ = ax.plot(norm_1_mean, label='top left')
        #_ = ax.plot(norm_2_mean, label='top right')
        #_ = ax.plot(norm_3_mean, label='bottom left')
        #_ = ax.plot(norm_4_mean, label='bottom right')
        ax.legend(loc='upper left')
        ax.set_xticks(gtind, minor=False)
        ax.margins(0.022,0)
        ax.label_outer()
        pos = pos+1
        plt.subplots_adjust(wspace=0, hspace=0.1)"""
        #plt.tight_layout()
        plt.show()

        
    
    return ind
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
p = 2#215
gt_length = int(gt_len[p,:,0].sum())
ind = np.argmax(gt[p][:gt_length],axis=0)
get_phases_from_vects(nda_vects[p][:gt_length], length=gt_length,plot=True,dir_axis=0, gtind=ind)
# Tue, 01 Mar 2022 10:54:10
@interact
def compare_phases(i=(0,nda_vects.shape[0]-1), plot=True):
    import matplotlib as mpl
    mpl.rcParams.update(mpl.rcParamsDefault)
    figsize = (25,1)
    weight = 1
    z = 0
    
    phases = ['ED', 'MS', 'ES', 'PF', 'MD']
    cardiac_cycle_length = int(gt_len[i,:,-1].sum())
    # crop or not
    cardiac_cycle_length_ = gt[i].shape[0]-3 # plot the full length minus border for visualisation
    
    vects = nda_vects[i][:cardiac_cycle_length]
    
    
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0) # 
    ind_pred = np.argmax(pred[i][:],axis=0)
    
    temp = mov[i,:cardiac_cycle_length,z] # cardiac_cycle_length_
    fig = show_2D_or_3D(temp,allow_slicing=False)
    ax = fig.get_axes()[0]
    _ = ax.set_ylabel('CMR 2d+t)\nmid-cavity')
    plt.show()
    
    fig= plt.figure(figsize=figsize)
    ax=fig.add_subplot(111)
    ax.set_xticks(ind, minor=False)
    ax.set_yticks([0, 1, 2, 3, 4], minor=False)
    ax.set_yticklabels(phases, rotation=0)
    ind = np.array(ind)
    onehot_gt = np.zeros((ind.size, cardiac_cycle_length)) # cardiac_cycle_length_
    onehot_gt[np.arange(ind.size), ind] = weight
    #ax.imshow(gt[i][:cardiac_cycle_length].T,aspect='auto') # with uncertainity
    ax.imshow(onehot_gt,aspect='auto', cmap='bone') # gt as binary phase2time mapping
    _ = ax.set_yticklabels(phases)
    ax.set_title('gt')
    ax.margins(0,0)
    
    indices = get_phases_from_vects(vects, length=cardiac_cycle_length, plot=plot,dir_axis=0, gtind=ind, figsize=figsize)

    indices = np.array(indices)
    onehot = np.zeros((indices.size, cardiac_cycle_length)) # cardiac_cycle_length_
    onehot[np.arange(indices.size), indices] = weight
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xticks(indices, minor=False)
    ax.set_yticks([0, 1, 2, 3, 4], minor=False)
    ax.set_yticklabels(phases, rotation=0)
    plt.imshow(onehot,aspect='auto', cmap='bone')
    _ = ax.set_yticklabels(phases)
    ax.margins(0,0)
    #ax.set_title('prediction')
    plt.show()
    
    print(phases)
    print('gt:', ind)
    print('u: ', indices)
    
    
    # this would plot the supervised prediction
    #print('p: ', ind_pred)
    """fig, ax = plt.subplots(figsize=figsize)
    ax.set_xticks(ind_pred, minor=False)
    ax.set_yticks([0, 1, 2, 3, 4], minor=False)
    ax.set_yticklabels(phases, rotation=45)
    plt.imshow(pred[i][:cardiac_cycle_length].T,aspect='auto')
    _ = ax.set_yticklabels(phases);plt.show()"""
    # patient 107 gcn --> good curve for plotting
    # 200 no cut
    # 125 no cut
    # 194 moderate cut 
    # 134 strong cut-off
# Tue, 01 Mar 2022 10:54:13
pred_u = np.zeros_like(gt)
upred_ind = []
cycle_len=[]
print(pred_u.shape)
for i in range(pred_u.shape[0]):
    weight = 1
    
    cardiac_cycle_length = int(gt_len[i,:,0].sum())
    cycle_len.append(cardiac_cycle_length)
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
    #print(gt[i][:cardiac_cycle_length].T.shape)
    
    indices = get_phases_from_vects(nda_vects[i][:cardiac_cycle_length], length=cardiac_cycle_length, gtind=ind,plot=False,dir_axis=0)
    upred_ind.append(indices)
    indices = np.array(indices)
    onehot = np.zeros((indices.size, cardiac_cycle_length))
    onehot[np.arange(indices.size), indices] = weight
    pred_u[i][0:cardiac_cycle_length] = onehot.T
upred_ind=np.stack(upred_ind, axis=0)
cycle_len = np.stack(cycle_len, axis=0)
# re-create a compatible shape for the metric fn
gt_ = np.stack([gt,gt_len], axis=1)
pred_ = np.stack([pred_u,np.zeros_like(pred_u)], axis=1)

# create a dataframe for further plots
from src.utils.Metrics import meandiff
phases = ['ED', 'MS', 'ES', 'PF', 'MD']
res = meandiff(gt_, pred_, apply_sum=False, apply_average=False)
df = pd.DataFrame(res.numpy(), columns=phases)
# Tue, 01 Mar 2022 10:54:30
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:54:31
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:54:32
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:54:33
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:54:34
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:54:35
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:54:36
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:54:37
# get the patient idx, per label where this method fails the most
label = 0
error_thres = 6
if type(res) != np.ndarray: res.numpy()
print('label:{} \naFD>{}\nPatient idx: {}'.format(phases[label],error_thres,np.where(res[:,label]>error_thres)[0]))
print('gt length: {}'.format(cycle_len[res[:,label]>error_thres]))
print('gt indices: {}'.format(gt_ind[:,label][(res[:,label]>error_thres)]))
print('pred indices: {}'.format(upred_ind[:,label][(res[:,label]>error_thres)]))
print('aFD: {}'.format(res[:,label][res[:,label]>error_thres]))
# Tue, 01 Mar 2022 10:54:37
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:54:37
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:54:38
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:54:39
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:54:40
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:54:41
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:54:42
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:54:43
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:54:44
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:54:45
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:54:46
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:54:47
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:54:48
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:54:48
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:54:49
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:54:50
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:54:51
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:54:52
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:54:53
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:54:54
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:54:55
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 10:54:56
# TOF unsupervised center 14_41 gaus 2
sb.set_context('paper')
sb.set(font_scale = 2)
_ =df.plot(kind='box')
ax = sb.violinplot(data=df)
ax = sb.stripplot(data=df)
pd.options.display.float_format = "{:,.2f}".format
df_summarized = pd.concat([df.mean(axis=0),df.std(axis=0), df.median(axis=0)], axis=1)
df_summarized.columns = ['mean', 'SD', 'meadian']
print(df_summarized.mean())
print(df_summarized)

sb.set_context('paper')
sb.set(font_scale = 1.8)
fig, axes = plt.subplots(1,5,figsize=(25,4))
axes = axes.flatten()
for i,ax in enumerate(axes):
    _ = ax.scatter(upred_ind[:,i],gt_ind[:,i])
    _ = ax.axline((1, 1), slope=1)
    _ = ax.set_title(phases[i])
    _ = ax.set_xlabel('Index - pred')
    _ = ax.set_ylabel('Index - gt')
    _ = ax.set_xlim([0,35])
    _ = ax.set_ylim([0,35])
plt.tight_layout()
# Tue, 01 Mar 2022 10:54:56
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 11:02:50
# get the patient idx, per label where this method fails the most
label = 4
error_thres = 10
if type(res) != np.ndarray: res.numpy()
print('label:{} \naFD>{}\nPatient idx: {}'.format(phases[label],error_thres,np.where(res[:,label]>error_thres)[0]))
print('gt length: {}'.format(cycle_len[res[:,label]>error_thres]))
print('gt indices: {}'.format(gt_ind[:,label][(res[:,label]>error_thres)]))
print('pred indices: {}'.format(upred_ind[:,label][(res[:,label]>error_thres)]))
print('aFD: {}'.format(res[:,label][res[:,label]>error_thres]))
# Tue, 01 Mar 2022 11:02:50
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 11:14:56
def detect_phases(dir_1d_mean, length):
    
    import scipy.signal as sig
    
        # MS
    # Global min of f(x)
    peaks, prop = sig.find_peaks(-1*dir_1d_mean, height=0.4)
    if len(peaks)>1
    ms = np.argmin(dir_1d_mean)
    ms = ms -1 # take the bucket before the first min peak
    
    
    # ES
    # First time f(x)>0 after MS
    cycle = np.concatenate([dir_1d_mean[ms:], dir_1d_mean[:ms]])
    temp_ = 0
    es_found=False
    negative_slope = False
    for idx,elem in enumerate(cycle):
        if elem<0:
            negative_slope=True
            temp_ = idx
        elif elem>=0 and negative_slope:
            es_found = True
            #temp_ = idx
            break # stop after first zero-transition
    if es_found:
        es = ms + temp_
        #es = es-1
    else:
        es = ms + 1 # the frame after ms, fallback
    if es>=length:
        es = np.mod(es,length)
        print('ES overflow: {}, ms:{}'.format(es,ms))
     
    
    # PF
    # First peak after ES, min height 0.6
    seq = dir_1d_mean[es:]
    peaks, prop = sig.find_peaks(seq, height=0.6)#height=0.6 we normalise between -1 and 1, PF should be close to argmax

    if len(peaks>0):
        pf = es + peaks[0] # take the peak after es
        pf = pf -1
    else:
        print('pf not clear, set to ES {} + 1'.format(es))
        pf = es + 1
    pf = np.mod(pf, length)
      

    # ED
    # Between pf and ms: last time f(x) cross zero from positive to negative
    # a priori knowledge ED needs a minimal distance of 2 frames towards MS
    # CHANGED the minimal distance between ED and MS
    cycle = np.concatenate([dir_1d_mean[pf:], dir_1d_mean[:ms]])
    #print(cycle)
    ed_found = False
    last_idx_positive = True # we start at the pf, which is the peak(dir)
    for idx,elem in enumerate(cycle):
        
        if elem>=0:
            last_idx_positive = True
        # remember the last idx before the direction gets negative the last time before ms
        elif elem<0 and last_idx_positive: # first time direction negative
            ed_found = True # for fallbacks
            temp_ = idx # idx before negative direction
            #print('found transition at: {}'.format(idx))
            last_idx_positive = False # remember only the first idx after transition
        
    if ed_found:
        ed = pf + temp_
        #print('ed:{}, pf:{}, temp_:{}, lenght: {}'.format(ed,pf,temp_,length))
    else: 
        # if we dont find a transition from positive to negative, take the idx which is the closest to zero
        temp_ = np.argmin(np.abs(cycle)) # make sure we have a minimal distance
        ed = pf + temp_
        print('ED: no transition found between {}-{} , closest id to 0: {}, ed = {}'.format(pf,ms,temp_,ed))
    
    if ed>=length:
        #print('gt ed: {}, ed: {}, length: {}'.format(gted,ed,length))
        ed = np.mod(ed,length)
        #print('mod ed: {}'.format(ed))
    #ed = ed-1 # take the bucket before negative
        
    # MD
    # Middle between PF and ED
    ed_slice_idx = ed
    if ed_slice_idx<=pf: # ed overflow --> beginning of cmr stack
        ed_slice_idx = length + ed
    md = (pf+ed_slice_idx)//2 # the bucket after the middle
    md = md + 1
    md = np.mod(md,length)
    
    return np.array([ed,ms,es,pf,md])
# Tue, 01 Mar 2022 11:14:56
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 11:15:28
def detect_phases(dir_1d_mean, length):
    
    import scipy.signal as sig
    
        # MS
    # Global min of f(x)
    peaks, prop = sig.find_peaks(-1*dir_1d_mean, height=0.4)
    if len(peaks)>1:
        ms = peaks[0]
    else:
        ms = np.argmin(dir_1d_mean)
    ms = ms -1 # take the bucket before the first min peak
    
    
    # ES
    # First time f(x)>0 after MS
    cycle = np.concatenate([dir_1d_mean[ms:], dir_1d_mean[:ms]])
    temp_ = 0
    es_found=False
    negative_slope = False
    for idx,elem in enumerate(cycle):
        if elem<0:
            negative_slope=True
            temp_ = idx
        elif elem>=0 and negative_slope:
            es_found = True
            #temp_ = idx
            break # stop after first zero-transition
    if es_found:
        es = ms + temp_
        #es = es-1
    else:
        es = ms + 1 # the frame after ms, fallback
    if es>=length:
        es = np.mod(es,length)
        print('ES overflow: {}, ms:{}'.format(es,ms))
     
    
    # PF
    # First peak after ES, min height 0.6
    seq = dir_1d_mean[es:]
    peaks, prop = sig.find_peaks(seq, height=0.6)#height=0.6 we normalise between -1 and 1, PF should be close to argmax

    if len(peaks>0):
        pf = es + peaks[0] # take the peak after es
        pf = pf -1
    else:
        print('pf not clear, set to ES {} + 1'.format(es))
        pf = es + 1
    pf = np.mod(pf, length)
      

    # ED
    # Between pf and ms: last time f(x) cross zero from positive to negative
    # a priori knowledge ED needs a minimal distance of 2 frames towards MS
    # CHANGED the minimal distance between ED and MS
    cycle = np.concatenate([dir_1d_mean[pf:], dir_1d_mean[:ms]])
    #print(cycle)
    ed_found = False
    last_idx_positive = True # we start at the pf, which is the peak(dir)
    for idx,elem in enumerate(cycle):
        
        if elem>=0:
            last_idx_positive = True
        # remember the last idx before the direction gets negative the last time before ms
        elif elem<0 and last_idx_positive: # first time direction negative
            ed_found = True # for fallbacks
            temp_ = idx # idx before negative direction
            #print('found transition at: {}'.format(idx))
            last_idx_positive = False # remember only the first idx after transition
        
    if ed_found:
        ed = pf + temp_
        #print('ed:{}, pf:{}, temp_:{}, lenght: {}'.format(ed,pf,temp_,length))
    else: 
        # if we dont find a transition from positive to negative, take the idx which is the closest to zero
        temp_ = np.argmin(np.abs(cycle)) # make sure we have a minimal distance
        ed = pf + temp_
        print('ED: no transition found between {}-{} , closest id to 0: {}, ed = {}'.format(pf,ms,temp_,ed))
    
    if ed>=length:
        #print('gt ed: {}, ed: {}, length: {}'.format(gted,ed,length))
        ed = np.mod(ed,length)
        #print('mod ed: {}'.format(ed))
    #ed = ed-1 # take the bucket before negative
        
    # MD
    # Middle between PF and ED
    ed_slice_idx = ed
    if ed_slice_idx<=pf: # ed overflow --> beginning of cmr stack
        ed_slice_idx = length + ed
    md = (pf+ed_slice_idx)//2 # the bucket after the middle
    md = md + 1
    md = np.mod(md,length)
    
    return np.array([ed,ms,es,pf,md])
# Tue, 01 Mar 2022 11:15:28
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 11:15:29
# try to predict the phases unsupervised just from the norm/direction curve

def get_phases_from_vects(vects_nda, length=-1, plot=True, dir_axis=0, gtind=[], figsize=(25,3)):
    import scipy.signal
    import scipy.signal as sig
    import scipy.ndimage
    from scipy.ndimage import gaussian_filter1d
    
    gted=gtind[0]
    
    # signal smoothing
    kernel_size = 4
    kernel = np.ones(kernel_size) / kernel_size
    z = 0
    
    
    dim_z = vects_nda.shape[1]
    #vects_nda = vects_nda[:int(length)]
    
    
    # define some helper lambdas
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    dim = vects_nda.shape[1:-1]
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]

    
    # norm of the vector
    norm_nda = np.linalg.norm(vects_nda[...,dir_axis:], axis=-1)
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    norm_nda_thres = clip_quantile(norm_nda, 0.9)
    norm_nda = minmax_lambda([norm_nda,0,1])
    norm_msk = norm_nda>=0.0 # this could be used to mask v_t by a threshold norm
    norm_nda = norm_nda * norm_msk
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    nda_1d_mean = np.mean(norm_nda,axis=(1,2,3))
    nda_1d_max = np.max(norm_nda,axis=(1,2,3))
    nda_1d_median = np.median(norm_nda,axis=(1,2,3))
    nda_1d_mean_smooth = np.convolve(nda_1d_mean, kernel, mode='same')
    #print(nda_1d_mean_smooth.shape, nda_1d_mean.shape)
    
    
    # direction relative to the center
    
    directions = flow2direction_lambda(vects_nda)[...,0].numpy()
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    directions = minmax_lambda([directions,lower,upper])
    #plt.hist(directions.flatten());plt.show()
    directions = directions * norm_msk
    #plt.hist(directions.flatten());plt.show()
    
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    dir_1d_mean = np.mean(directions,axis=(1,2,3))
    dir_1d_median = np.median(directions,axis=(1,2,3))

    
    # Find phases
    # 1st smooth the direction with a rolling averge (kernelsize=4)
    # or via gaussian filter
    # 2nd min/max normalise each direction vector in a range [-1:1]
    
    
    dir_1d_mean = gaussian_filter1d(dir_1d_mean, sigma=2, mode='wrap') # original: dir_1d_mean
    
    ################# CHANGED !!!!!! ######################## RECENTLY NO CHANGES
    # min/max scaling
    dir_1d_mean = minmax_lambda([dir_1d_mean,lower,upper])
    
    
#     # test with corner specific vector fields
#     y,x = directions.shape[-2:]
    
#     vects_mean = minmax_lambda([np.mean(vects_nda,axis=(1,2,3,4)),lower,upper])
#     vects_1 = vects_nda[:,:,:y//2,:x//2,:]
#     vects_1_mean = minmax_lambda([np.mean(vects_1,axis=(1,2,3,4)),lower,upper])
#     vects_2 = vects_nda[:,:,y//2:,:x//2,:]
#     vects_2_mean = minmax_lambda([np.mean(vects_2,axis=(1,2,3,4)),lower,upper])
#     vects_3 = vects_nda[:,:,y//2:,x//2:,:]
#     vects_3_mean = minmax_lambda([np.mean(vects_3,axis=(1,2,3,4)),lower,upper])
#     vects_4 = vects_nda[:,:,:y//2,x//2:,:]
#     vects_4_mean = minmax_lambda([np.mean(vects_4,axis=(1,2,3,4)),lower,upper])
    
    
#     # direction per corner
#     dir_1 = directions[:,:,:y//2,:x//2] # top left, plot
#     dir_1_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_2 = directions[:,:,y//2:,:x//2] # top right, plot
#     dir_2_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_3 = directions[:,:,y//2:,x//2:] # bottom left
#     dir_3_mean = minmax_lambda([np.mean(dir_3,axis=(1,2,3)),lower,upper])
#     dir_4 = directions[:,:,:y//2,x//2:] # bottom right
#     dir_4_mean = minmax_lambda([np.mean(dir_4,axis=(1,2,3)),lower,upper])
    

#     # norm per corner
#     norm_mean = minmax_lambda([nda_1d_mean, mid, upper])
#     norm_1 = norm_nda[:,:,:y//2,:x//2]
#     norm_1_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_2 = norm_nda[:,:,y//2:,:x//2]
#     norm_2_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_3 = norm_nda[:,:,y//2:,x//2:]
#     norm_3_mean = minmax_lambda([np.mean(norm_3,axis=(1,2,3)),mid,upper])
#     norm_4 = norm_nda[:,:,:y//2,x//2:]
#     norm_4_mean = minmax_lambda([np.mean(norm_4,axis=(1,2,3)),mid,upper])
    

    ind = detect_phases(dir_1d_mean=dir_1d_mean[:length], length=length)
    
    
    #print('ed: {}, ms: {}, es: {}, pf: {}, md: {}'.format(*ind))
    
        # plot the mean/max norm for one patient oveer time
    if plot:
        
        
        # VECT 2D+t
        vect_2d_t = vects_nda[:,z]
        fig1 = show_2D_or_3D(vect_2d_t,allow_slicing=False, f_size=(25,2))
        ax1 = fig1.get_axes()[0]
        _ = ax1.set_ylabel('phi_t \nmid-cavity')

        
        # DIR 2D+t
        dir_2d_t = directions[:,z]
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        #dir_2d_t = minmax_lambda([dir_2d_t,lower,upper])
        div_cmap = 'bwr'
        fig = show_2D_or_3D(dir_2d_t,allow_slicing=False, f_size=(25,2),cmap=div_cmap)
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('alpha_t \nmid-cavity')
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[len(ax.get_images())//2], cax=cax, orientation='horizontal')

        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # DIR 2D T x Z
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        directions_tz = directions.mean(axis=(2,3)) # original: norm()
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        _ = ax.imshow(directions_tz.T, aspect='auto', label='aasd', cmap=div_cmap, origin='lower')
        _ = ax.set_xticks(gtind, minor=False)
        ax2 = ax.twinx()
        _ = ax2.plot(dir_1d_mean, c='black',label='dir 1d+t')
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('dir z+t\napex:base')
        
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1
        
        #ax=fig.add_subplot(rows,1,pos)
        #_ = ax.plot(dir_1d_mean, c='black',label='dir 1d+t')
        #_ = ax.plot(dir_1_mean, label='top left')
        #_ = ax.plot(dir_2_mean, label='top right')
        #_ = ax.plot(dir_3_mean, label='bottom left')
        #_ = ax.plot(dir_4_mean, label='bottom right')
        #ax.legend(loc='upper left')
        #ax.set_xticks(gtind, minor=False)
        #ax.margins(0.022,0)
        #ax.label_outer()
        #pos=pos+1
        #plt.subplots_adjust(wspace=0, hspace=0.1)
        #plt.show()
        norm_cmap = 'hot'
        
        # NORM 2D + t
        norm_2d_t = norm_nda[:,z]
        norm_2d_t = minmax_lambda([norm_2d_t,mid,upper])
        fig = show_2D_or_3D(norm_2d_t,allow_slicing=False, f_size=(25,2),cmap=norm_cmap, interpolation='none')
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('norm 2d+t\nmid-cavity')
        #ax.margins(0,0)
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[0], cax=cax, orientation='horizontal')
        
        
        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # NORM 2D TxZ
        norm2d = minmax_lambda([norm_nda.mean(axis=(2,3)),mid,upper])
        _ = ax.imshow(norm2d.T, aspect='auto', origin='lower',cmap=norm_cmap, interpolation='none')
        _ = ax.set_xticks(gtind, minor=False)
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('norm z+t\napex:base')
        ax2 = ax.twinx()
        _ = ax2.plot(nda_1d_mean, c='black',label='norm 1d+t')
        #ax.spines['top'].set_visible(False)
        #_ = ax.legend(loc='upper left')
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1

        """ax=fig.add_subplot(rows,1,pos)
        _ = ax.plot(norm_mean, label='norm 1d+t')
        #_ = ax.plot(norm_1_mean, label='top left')
        #_ = ax.plot(norm_2_mean, label='top right')
        #_ = ax.plot(norm_3_mean, label='bottom left')
        #_ = ax.plot(norm_4_mean, label='bottom right')
        ax.legend(loc='upper left')
        ax.set_xticks(gtind, minor=False)
        ax.margins(0.022,0)
        ax.label_outer()
        pos = pos+1
        plt.subplots_adjust(wspace=0, hspace=0.1)"""
        #plt.tight_layout()
        plt.show()

        
    
    return ind
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
p = 2#215
gt_length = int(gt_len[p,:,0].sum())
ind = np.argmax(gt[p][:gt_length],axis=0)
get_phases_from_vects(nda_vects[p][:gt_length], length=gt_length,plot=True,dir_axis=0, gtind=ind)
# Tue, 01 Mar 2022 11:15:32
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 11:15:33
@interact
def compare_phases(i=(0,nda_vects.shape[0]-1), plot=True):
    import matplotlib as mpl
    mpl.rcParams.update(mpl.rcParamsDefault)
    figsize = (25,1)
    weight = 1
    z = 0
    
    phases = ['ED', 'MS', 'ES', 'PF', 'MD']
    cardiac_cycle_length = int(gt_len[i,:,-1].sum())
    # crop or not
    cardiac_cycle_length_ = gt[i].shape[0]-3 # plot the full length minus border for visualisation
    
    vects = nda_vects[i][:cardiac_cycle_length]
    
    
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0) # 
    ind_pred = np.argmax(pred[i][:],axis=0)
    
    temp = mov[i,:cardiac_cycle_length,z] # cardiac_cycle_length_
    fig = show_2D_or_3D(temp,allow_slicing=False)
    ax = fig.get_axes()[0]
    _ = ax.set_ylabel('CMR 2d+t)\nmid-cavity')
    plt.show()
    
    fig= plt.figure(figsize=figsize)
    ax=fig.add_subplot(111)
    ax.set_xticks(ind, minor=False)
    ax.set_yticks([0, 1, 2, 3, 4], minor=False)
    ax.set_yticklabels(phases, rotation=0)
    ind = np.array(ind)
    onehot_gt = np.zeros((ind.size, cardiac_cycle_length)) # cardiac_cycle_length_
    onehot_gt[np.arange(ind.size), ind] = weight
    #ax.imshow(gt[i][:cardiac_cycle_length].T,aspect='auto') # with uncertainity
    ax.imshow(onehot_gt,aspect='auto', cmap='bone') # gt as binary phase2time mapping
    _ = ax.set_yticklabels(phases)
    ax.set_title('gt')
    ax.margins(0,0)
    
    indices = get_phases_from_vects(vects, length=cardiac_cycle_length, plot=plot,dir_axis=0, gtind=ind, figsize=figsize)

    indices = np.array(indices)
    onehot = np.zeros((indices.size, cardiac_cycle_length)) # cardiac_cycle_length_
    onehot[np.arange(indices.size), indices] = weight
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xticks(indices, minor=False)
    ax.set_yticks([0, 1, 2, 3, 4], minor=False)
    ax.set_yticklabels(phases, rotation=0)
    plt.imshow(onehot,aspect='auto', cmap='bone')
    _ = ax.set_yticklabels(phases)
    ax.margins(0,0)
    #ax.set_title('prediction')
    plt.show()
    
    print(phases)
    print('gt:', ind)
    print('u: ', indices)
    
    
    # this would plot the supervised prediction
    #print('p: ', ind_pred)
    """fig, ax = plt.subplots(figsize=figsize)
    ax.set_xticks(ind_pred, minor=False)
    ax.set_yticks([0, 1, 2, 3, 4], minor=False)
    ax.set_yticklabels(phases, rotation=45)
    plt.imshow(pred[i][:cardiac_cycle_length].T,aspect='auto')
    _ = ax.set_yticklabels(phases);plt.show()"""
    # patient 107 gcn --> good curve for plotting
    # 200 no cut
    # 125 no cut
    # 194 moderate cut 
    # 134 strong cut-off
# Tue, 01 Mar 2022 11:15:36
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 11:17:27
pred_u = np.zeros_like(gt)
upred_ind = []
cycle_len=[]
print(pred_u.shape)
for i in range(pred_u.shape[0]):
    weight = 1
    
    cardiac_cycle_length = int(gt_len[i,:,0].sum())
    cycle_len.append(cardiac_cycle_length)
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
    #print(gt[i][:cardiac_cycle_length].T.shape)
    
    indices = get_phases_from_vects(nda_vects[i][:cardiac_cycle_length], length=cardiac_cycle_length, gtind=ind,plot=False,dir_axis=0)
    upred_ind.append(indices)
    indices = np.array(indices)
    onehot = np.zeros((indices.size, cardiac_cycle_length))
    onehot[np.arange(indices.size), indices] = weight
    pred_u[i][0:cardiac_cycle_length] = onehot.T
upred_ind=np.stack(upred_ind, axis=0)
cycle_len = np.stack(cycle_len, axis=0)
# re-create a compatible shape for the metric fn
gt_ = np.stack([gt,gt_len], axis=1)
pred_ = np.stack([pred_u,np.zeros_like(pred_u)], axis=1)

# create a dataframe for further plots
from src.utils.Metrics import meandiff
phases = ['ED', 'MS', 'ES', 'PF', 'MD']
res = meandiff(gt_, pred_, apply_sum=False, apply_average=False)
df = pd.DataFrame(res.numpy(), columns=phases)
# Tue, 01 Mar 2022 11:17:44
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 11:17:45
# get the patient idx, per label where this method fails the most
label = 4
error_thres = 10
if type(res) != np.ndarray: res.numpy()
print('label:{} \naFD>{}\nPatient idx: {}'.format(phases[label],error_thres,np.where(res[:,label]>error_thres)[0]))
print('gt length: {}'.format(cycle_len[res[:,label]>error_thres]))
print('gt indices: {}'.format(gt_ind[:,label][(res[:,label]>error_thres)]))
print('pred indices: {}'.format(upred_ind[:,label][(res[:,label]>error_thres)]))
print('aFD: {}'.format(res[:,label][res[:,label]>error_thres]))
# Tue, 01 Mar 2022 11:17:45
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 11:17:47
# TOF unsupervised center 14_41 gaus 2
sb.set_context('paper')
sb.set(font_scale = 2)
_ =df.plot(kind='box')
ax = sb.violinplot(data=df)
ax = sb.stripplot(data=df)
pd.options.display.float_format = "{:,.2f}".format
df_summarized = pd.concat([df.mean(axis=0),df.std(axis=0), df.median(axis=0)], axis=1)
df_summarized.columns = ['mean', 'SD', 'meadian']
print(df_summarized.mean())
print(df_summarized)

sb.set_context('paper')
sb.set(font_scale = 1.8)
fig, axes = plt.subplots(1,5,figsize=(25,4))
axes = axes.flatten()
for i,ax in enumerate(axes):
    _ = ax.scatter(upred_ind[:,i],gt_ind[:,i])
    _ = ax.axline((1, 1), slope=1)
    _ = ax.set_title(phases[i])
    _ = ax.set_xlabel('Index - pred')
    _ = ax.set_ylabel('Index - gt')
    _ = ax.set_xlim([0,35])
    _ = ax.set_ylim([0,35])
plt.tight_layout()
# Tue, 01 Mar 2022 11:17:47
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 11:18:27
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:15:48
# load the vectors
pathstovectnpy = sorted(glob.glob(os.path.join(vects_chooser.selected, 'moved', '*vects*.npy')))
print(pathstovectnpy)
nda_vects = np.concatenate([np.load(path_) for path_ in pathstovectnpy],axis=0)
print(nda_vects.shape)
# Tue, 01 Mar 2022 13:16:26
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:16:27
# load the phase gt and pred
pred_path = os.path.join(vects_chooser.selected, 'pred')
pathtsophasenpy = sorted(glob.glob(os.path.join(pred_path, '*gtpred*.npy')))
print(pathtsophasenpy)
nda_phase = np.concatenate([np.load(path_) for path_ in pathtsophasenpy],axis=1)
print(nda_phase.shape)
gt_, pred_ = np.split(nda_phase, axis=0,indices_or_sections=2)
print(gt_.shape)
gt= gt_[0,:,0]
pred = pred_[0,:,0]
print(gt.shape)
gt_len = gt_[0,:,1]
# Tue, 01 Mar 2022 13:16:27
# load some moved examples for easier understanding of the dimensions
pathtomoved = sorted(glob.glob(os.path.join(vects_chooser.selected, 'moved','*moved*.npy')))
print(len(pathtomoved))
mov = np.concatenate([np.load(path_) for path_ in pathtomoved],axis=0)
print(mov.shape) # patient,time,spatial-z,y,x,channel, e.g.: 69,40,16,64,64,1
# Tue, 01 Mar 2022 13:16:40
# plot one moved timestep of one patient = one 3D volume 
# should go from apex to base
from src.visualization.Visualize import show_2D_or_3D
temp = mov[10,0,...,0]
_ = show_2D_or_3D(temp, cmap='gray', interpolation='none')
# Tue, 01 Mar 2022 13:16:40
import tensorflow as tf
from tensorflow.image import ssim
ssim(mov[10,0], mov[10,1],max_val=1.0, filter_size=11,
                          filter_sigma=1.5, k1=0.01, k2=0.03).numpy().mean()
# Tue, 01 Mar 2022 13:16:40
from skimage.metrics import structural_similarity as ssim
ssim(mov[10,0], mov[10,1],multichannel=True)
# Tue, 01 Mar 2022 13:16:41
plt.imshow(gt[0].T)
# Tue, 01 Mar 2022 13:16:41
plt.imshow(pred[0].T)
# Tue, 01 Mar 2022 13:16:41
import tensorflow as tf
import sys
# returns a matrix with the indicies as values, similar to np.indicies
def get_idxs_tf(x):
    return tf.cast(
        tf.reshape(tf.where(tf.ones((x[0], x[1], x[2]))), (x[0], x[1], x[2], 3)),
        tf.float32)


# returns a matrix with vectors pointing to the center
def get_centers_tf(x):
    return tf.cast(
        tf.tile(tf.convert_to_tensor([x[0] // 2, x[1] // 2, x[2] // 2])[tf.newaxis, tf.newaxis, tf.newaxis, ...],
                (x[0], x[1], x[2], 1)), tf.float32)

def get_angle_tf(a, b):
    # this should work for batches of n-dimensional vectors
    # α = arccos[(a · b) / (|a| * |b|)]
    # |v| = √(x² + y² + z²)
    """
    in 3D space
    If vectors a = [xa, ya, za], b = [xb, yb, zb], then:
    α = arccos[(xa * xb + ya * yb + za * zb) / (√(xa2 + ya2 + za2) * √(xb2 + yb2 + zb2))]
    """
    import math as m
    pi = tf.constant(m.pi)
    b = tf.cast(b, dtype=a.dtype)
    inner = tf.einsum('...i,...i->...', a, b)
    norms = tf.norm(a, ord='euclidean', axis=-1) * tf.norm(b, ord='euclidean', axis=-1)  # [...,None]
    cos = inner / (norms + sys.float_info.epsilon)
    #rad = tf.math.acos(tf.clip_by_value(cos, -1.0, 1.0))
    # rad2deg conversion
    #deg = rad * (180.0/pi)
    return cos[..., tf.newaxis]

dim = [16,64,64]
idx = get_idxs_tf(dim)
c = get_centers_tf(dim)
centers = c - idx
centers_tensor = centers[tf.newaxis, ...]
flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x, centers_tensor), name='flow2direction')
minmax_lambda = lambda x : x[1] + (((x[0] - np.min(x[0])) * (x[2]-x[1]))/(np.max(x[0]) - np.min(x[0])))
# define some central params
lower, mid, upper = -1,0, 1
# Tue, 01 Mar 2022 13:16:41
# align norm an direction by the cardiac phase ED
# resample all 1D feature vectors to the same length
# this should help to validate the pre-defined rules, and if both together explains the cardiac phases
target_t = 30
norms = []
dirs= []
cycle_len=[]
gt_ind = []
gt_ind_rolled = []
xval = np.linspace(0,1,target_t)

for i,n in enumerate(nda_vects[:]):
    cardiac_cycle_length = int(gt_len[i,:,0].sum())
    cycle_len.append(cardiac_cycle_length)
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
    gt_onehot = gt[i][:cardiac_cycle_length]
    gt_ind.append(ind)
    
    dir_axis=0
    gt_ed = ind[0]
    #assert cardiac_cycle_length>=gt_ed
    temp = n[:cardiac_cycle_length]
    
    dim = temp.shape[1:-1]
    #print(dim)
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    #print('centers: ',c.dtype)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    # direction
    directions = flow2direction_lambda(temp)[...,0]
    # direction mean
    directions = np.mean(directions,axis=(1,2,3))
    # direction ed aligned
    directions = np.roll(directions, -1*gt_ed)
    # direction interpolate to unique length
    f = interp1d(np.linspace(0,1,directions.shape[0]), directions, kind='linear')
    directions = f(xval)
    # direction min/max normalised between -1,1

    directions = minmax_lambda([directions,lower,upper])
    
    # magnitude/norm as mean
    norm_full = np.linalg.norm(temp, axis=-1)
    norm_nda = norm_full.mean(axis=(1,2,3))
    # norm ed aligned
    norm_nda = np.roll(norm_nda, -1*gt_ed)
    # interpolate to unique length
    #f = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')
    #norm_nda = f(xval)
    norm_nda = np.interp(xval, np.linspace(0,1,norm_nda.shape[0]), norm_nda)
    # norm min/max aligned to 0,1
    norm_nda = minmax_lambda([norm_nda,mid,upper])
    
    # roll, scale, round and clip the gt indicies, to get an aligned distribution of the labels
    gt_onehot_rolled = np.roll(gt_onehot, -1*gt_ed, axis=0)
    resize_factor = target_t/cardiac_cycle_length
    gt_onehot_rolled = np.argmax(gt_onehot_rolled,axis=0)
    gt_onehot_rolled = np.clip(np.rint(gt_onehot_rolled * resize_factor),a_min=0,a_max=target_t-1)
    gt_ind_rolled.append(gt_onehot_rolled)
    
    norms.append(norm_nda)
    dirs.append(directions)
# Tue, 01 Mar 2022 13:17:08
# get the mean values
gt_ind_scaled = np.stack(gt_ind_rolled, axis=0)
gt_ind = np.stack(gt_ind,axis=0)
cycle_len = np.stack(cycle_len, axis=0)
norms = np.stack(norms, axis=0)
dirs = np.stack(dirs, axis=0)
norms_m = [norms.mean(axis=0)]
dirs_m=[dirs.mean(axis=0)]
# Tue, 01 Mar 2022 13:17:08
# plot the distribution of the aligned and scaled gt indicies
fig, (ax1,ax2, ax3) = plt.subplots(1,3, figsize=(15,5))
temp = pd.DataFrame(gt_ind_scaled, columns=phases)
# TODO check this outlier with MD ==2
temp = temp[temp['MD']>10]
print(temp.mean(), temp.std())
ax1 = sb.violinplot(data=temp, ax=ax1)
ax1 = sb.stripplot(data=temp, ax=ax1)
#_ = temp.plot(kind='box', ax=ax1)
_ = temp.plot(kind='line', ax=ax2, legend=None)
_ = temp.plot(kind='hist', ax=ax3, legend=None)
ax3.set_ylabel('')
# Tue, 01 Mar 2022 13:17:08
"""i = 0
cardiac_cycle_length = int(gt_len[i,:,0].sum())
cycle_len.append(cardiac_cycle_length)
ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
gt_ind.append(ind)
dir_axis=0
gt_ed = ind[0]
#assert cardiac_cycle_length>=gt_ed
temp = n[:cardiac_cycle_length]
norm_full = np.linalg.norm(temp, axis=-1)
norm_nda = norm_full.mean(axis=(1,2,3))"""
# Tue, 01 Mar 2022 13:17:08
get_ipython().run_cell_magic('time', '', "norm_nda = norm_full.mean(axis=(1,2,3))\nf = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')\nnorm_nda = f(xval)\n")
# Tue, 01 Mar 2022 13:17:08
get_ipython().run_cell_magic('time', '', 'norm_nda = norm_full.mean(axis=(1,2,3))\nnorm_nda = np.interp(xval, np.linspace(0,1,norm_nda.shape[0]), norm_nda)\n')
# Tue, 01 Mar 2022 13:17:08
# seaborn
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax.margins(0,0)
for n in dirs:
    _ = ax.plot(n, alpha=0.5, zorder=0)

df = pd.DataFrame(dirs).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='blue', label=r'$\alpha_t$',ci='sd',err_style='bars',zorder=2,legend=False,ax=ax)
#_ = ax.set_title('Mean direction +/- SD - aligned at ED')
_ = ax.set_xticks(np.rint(gt_ind_scaled.mean(axis=0)), minor=False)
_ = ax.set_xlabel('Time (t) - linear interpolated to 40 frames')
_ = ax.set_ylabel(r'Angle $\alpha_t$ of $\phi_t$')
_ = ax.legend(loc='upper left')
ax2 = ax.twinx()
ax2.margins(0,0)
df = pd.DataFrame(norms).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='black', label=r'|$\phi_t$|',ci='sd', linestyle='dashed',err_style='bars',zorder=2,legend=False,ax=ax2)
_ = ax2.set_ylabel(r'Norm |$\phi_t$|')

_ = ax2.legend(loc='upper right')
# Tue, 01 Mar 2022 13:17:09
folds_chooser = FileChooser(os.path.join(os.getcwd(),'/mnt/ssd/git/dynamic-cmr-models/exp/phasereg_v3/acdc/'), '')
display(folds_chooser)
# Tue, 01 Mar 2022 13:17:09
# define df and helper for pathology extractio
# this should have the same order as our inference data
#df_meta = pd.read_csv(folds_chooser.selected)
df_meta = pd.read_csv('/mnt/sds/sd20i001/sven/data/acdc/02_imported_4D_unfiltered/df_kfold.csv')
df_meta = df_meta.loc[:, ~df_meta.columns.str.contains('^Unnamed')]
df_meta = df_meta[df_meta.patient != 'patient090']  #we excluded this patient

def get_msk_for_pathology(df_, pathology='minf'):
    msk = []
    for f in [0,1,2,3,4]:
        patients = df_[df_.fold.isin([f])]
        patients = patients[patients['phase']=='ED']
        pat = patients[patients['modality'] == 'test']['patient'].str.lower().unique()
        sub_df = patients[patients.patient.isin(pat)].drop_duplicates(ignore_index=True, subset='patient')
        sub_df = sub_df.drop('x_path',axis=1).drop('y_path',axis=1)
        msk.append(sub_df['pathology'].str.lower()==pathology.lower())
    return np.concatenate(msk)
df_meta
# Tue, 01 Mar 2022 13:17:09
# plot one line per patient - norm
fig, ax = plt.subplots(figsize=(25,5))
for n in norms:
    _ = plt.plot(n)
_ = ax.set_title('Magnitudes aligned at ED, resampling shape 40,')
# Tue, 01 Mar 2022 13:17:09
# plot one line per direction
fig, ax = plt.subplots(figsize=(25,5))
for n in dirs:
    _ = plt.plot(n)
_ = ax.set_title('Directions aligned at ED, resampling shape 40,')
# Tue, 01 Mar 2022 13:17:10
from src.visualization.Visualize import show_2D_or_3D
from src.data.Preprocess import clip_quantile
nda_temp = nda_vects[0]
# norm_1 = nda_temp[:,:,:32,:32]# top left
norm_1 = nda_temp[:,:,:32,32:]# top right
norm_1 = np.linalg.norm(norm_1, axis=-1)
nda_norm_ = np.linalg.norm(nda_temp, axis=-1)
plt.hist(nda_norm_.flatten());plt.show()
nda_norm = clip_quantile(nda_norm_, 0.99)
nda_norm = minmax_lambda([nda_norm,0,1])
plt.hist(nda_norm.flatten())
plt.show()
# mask phi with a threshold norm matrix 
nda_msk = (nda_norm>=0.2).astype(np.float32)
nda_temp = nda_norm * nda_msk
#plt.hist(nda_temp.flatten())
print(nda_temp.shape)
_ = show_2D_or_3D(nda_norm_[:,0], allow_slicing=True)
_ = show_2D_or_3D(nda_temp[:,0], allow_slicing=True)
_ = show_2D_or_3D(nda_temp[:,-1], allow_slicing=True)

# top right
_ = show_2D_or_3D(norm_1[:,0], allow_slicing=True,cmap='gray', interpolation='none')
# Tue, 01 Mar 2022 13:17:12
fig, ax = plt.subplots(figsize=(25,3))
plt.plot(np.argmax(gt[0], axis=1))
ax.set_yticks([0, 1, 2, 3, 4], minor=False)
phases = ['ED', 'MS', 'ES', 'PF', 'MD']
_ = ax.set_yticklabels(phases)
# Tue, 01 Mar 2022 13:17:12
# plot the mean/max norm for one patient oveer time
nda_1d_max = np.max(nda_temp,axis=(1,2,3))
nda_1d_mean = np.mean(nda_temp,axis=(1,2,3))
nda_1d_sum = np.sum(nda_temp,axis=(1,2,3))

fig, ax = plt.subplots(figsize=(25,3))
ax.set_title('max norm')
_ = plt.plot(nda_1d_max); plt.show()

fig, ax = plt.subplots(figsize=(25,3))
ax.set_title('mean norm')
_ = plt.plot(nda_1d_mean)

fig, ax = plt.subplots(figsize=(25,3))
ax.set_title('mean sum')
_ = plt.plot(nda_1d_sum)
#ax.set_ylim(0.0,0.15)
# Tue, 01 Mar 2022 13:17:13
temp = np.arange(10)
temp[:0]
# Tue, 01 Mar 2022 13:17:13
def detect_phases(dir_1d_mean, length):
    
    import scipy.signal as sig
    
        # MS
    # Global min of f(x)
    peaks, prop = sig.find_peaks(-1*dir_1d_mean, height=0.4)
    if len(peaks)>1:
        ms = peaks[0]
    else:
        ms = np.argmin(dir_1d_mean)
    ms = ms -1 # take the bucket before the first min peak
    
    
    # ES
    # First time f(x)>0 after MS
    cycle = np.concatenate([dir_1d_mean[ms:], dir_1d_mean[:ms]])
    temp_ = 0
    es_found=False
    negative_slope = False
    for idx,elem in enumerate(cycle):
        if elem<0:
            negative_slope=True
            temp_ = idx
        elif elem>=0 and negative_slope:
            es_found = True
            #temp_ = idx
            break # stop after first zero-transition
    if es_found:
        es = ms + temp_
        #es = es-1
    else:
        es = ms + 1 # the frame after ms, fallback
    if es>=length:
        es = np.mod(es,length)
        print('ES overflow: {}, ms:{}'.format(es,ms))
     
    
    # PF
    # First peak after ES, min height 0.6
    seq = dir_1d_mean[es:]
    peaks, prop = sig.find_peaks(seq, height=0.6)#height=0.6 we normalise between -1 and 1, PF should be close to argmax

    if len(peaks>0):
        pf = es + peaks[0] # take the peak after es
        pf = pf -1
    else:
        print('pf not clear, set to ES {} + 1'.format(es))
        pf = es + 1
    pf = np.mod(pf, length)
      

    # ED
    # Between pf and ms: last time f(x) cross zero from positive to negative
    # a priori knowledge ED needs a minimal distance of 2 frames towards MS
    # CHANGED the minimal distance between ED and MS
    cycle = np.concatenate([dir_1d_mean[pf:], dir_1d_mean[:ms]])
    #print(cycle)
    ed_found = False
    last_idx_positive = True # we start at the pf, which is the peak(dir)
    for idx,elem in enumerate(cycle):
        
        if elem>=0:
            last_idx_positive = True
        # remember the last idx before the direction gets negative the last time before ms
        elif elem<0 and last_idx_positive: # first time direction negative
            ed_found = True # for fallbacks
            temp_ = idx # idx before negative direction
            #print('found transition at: {}'.format(idx))
            last_idx_positive = False # remember only the first idx after transition
        
    if ed_found:
        ed = pf + temp_
        #print('ed:{}, pf:{}, temp_:{}, lenght: {}'.format(ed,pf,temp_,length))
    else: 
        # if we dont find a transition from positive to negative, take the idx which is the closest to zero
        temp_ = np.argmin(np.abs(cycle)) # make sure we have a minimal distance
        ed = pf + temp_
        print('ED: no transition found between {}-{} , closest id to 0: {}, ed = {}'.format(pf,ms,temp_,ed))
    
    if ed>=length:
        #print('gt ed: {}, ed: {}, length: {}'.format(gted,ed,length))
        ed = np.mod(ed,length)
        #print('mod ed: {}'.format(ed))
    #ed = ed-1 # take the bucket before negative
        
    # MD
    # Middle between PF and ED
    ed_slice_idx = ed
    if ed_slice_idx<=pf: # ed overflow --> beginning of cmr stack
        ed_slice_idx = length + ed
    md = (pf+ed_slice_idx)//2 # the bucket after the middle
    md = md + 1
    md = np.mod(md,length)
    
    return np.array([ed,ms,es,pf,md])
# Tue, 01 Mar 2022 13:17:13
# try to predict the phases unsupervised just from the norm/direction curve

def get_phases_from_vects(vects_nda, length=-1, plot=True, dir_axis=0, gtind=[], figsize=(25,3)):
    import scipy.signal
    import scipy.signal as sig
    import scipy.ndimage
    from scipy.ndimage import gaussian_filter1d
    
    gted=gtind[0]
    
    # signal smoothing
    kernel_size = 4
    kernel = np.ones(kernel_size) / kernel_size
    z = 0
    
    
    dim_z = vects_nda.shape[1]
    #vects_nda = vects_nda[:int(length)]
    
    
    # define some helper lambdas
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    dim = vects_nda.shape[1:-1]
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]

    
    # norm of the vector
    norm_nda = np.linalg.norm(vects_nda[...,dir_axis:], axis=-1)
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    norm_nda_thres = clip_quantile(norm_nda, 0.9)
    norm_nda = minmax_lambda([norm_nda,0,1])
    norm_msk = norm_nda>=0.0 # this could be used to mask v_t by a threshold norm
    norm_nda = norm_nda * norm_msk
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    nda_1d_mean = np.mean(norm_nda,axis=(1,2,3))
    nda_1d_max = np.max(norm_nda,axis=(1,2,3))
    nda_1d_median = np.median(norm_nda,axis=(1,2,3))
    nda_1d_mean_smooth = np.convolve(nda_1d_mean, kernel, mode='same')
    #print(nda_1d_mean_smooth.shape, nda_1d_mean.shape)
    
    
    # direction relative to the center
    
    directions = flow2direction_lambda(vects_nda)[...,0].numpy()
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    directions = minmax_lambda([directions,lower,upper])
    #plt.hist(directions.flatten());plt.show()
    directions = directions * norm_msk
    #plt.hist(directions.flatten());plt.show()
    
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    dir_1d_mean = np.mean(directions,axis=(1,2,3))
    dir_1d_median = np.median(directions,axis=(1,2,3))

    
    # Find phases
    # 1st smooth the direction with a rolling averge (kernelsize=4)
    # or via gaussian filter
    # 2nd min/max normalise each direction vector in a range [-1:1]
    
    
    dir_1d_mean = gaussian_filter1d(dir_1d_mean, sigma=2, mode='wrap') # original: dir_1d_mean
    
    ################# CHANGED !!!!!! ######################## RECENTLY NO CHANGES
    # min/max scaling
    dir_1d_mean = minmax_lambda([dir_1d_mean,lower,upper])
    
    
#     # test with corner specific vector fields
#     y,x = directions.shape[-2:]
    
#     vects_mean = minmax_lambda([np.mean(vects_nda,axis=(1,2,3,4)),lower,upper])
#     vects_1 = vects_nda[:,:,:y//2,:x//2,:]
#     vects_1_mean = minmax_lambda([np.mean(vects_1,axis=(1,2,3,4)),lower,upper])
#     vects_2 = vects_nda[:,:,y//2:,:x//2,:]
#     vects_2_mean = minmax_lambda([np.mean(vects_2,axis=(1,2,3,4)),lower,upper])
#     vects_3 = vects_nda[:,:,y//2:,x//2:,:]
#     vects_3_mean = minmax_lambda([np.mean(vects_3,axis=(1,2,3,4)),lower,upper])
#     vects_4 = vects_nda[:,:,:y//2,x//2:,:]
#     vects_4_mean = minmax_lambda([np.mean(vects_4,axis=(1,2,3,4)),lower,upper])
    
    
#     # direction per corner
#     dir_1 = directions[:,:,:y//2,:x//2] # top left, plot
#     dir_1_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_2 = directions[:,:,y//2:,:x//2] # top right, plot
#     dir_2_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_3 = directions[:,:,y//2:,x//2:] # bottom left
#     dir_3_mean = minmax_lambda([np.mean(dir_3,axis=(1,2,3)),lower,upper])
#     dir_4 = directions[:,:,:y//2,x//2:] # bottom right
#     dir_4_mean = minmax_lambda([np.mean(dir_4,axis=(1,2,3)),lower,upper])
    

#     # norm per corner
#     norm_mean = minmax_lambda([nda_1d_mean, mid, upper])
#     norm_1 = norm_nda[:,:,:y//2,:x//2]
#     norm_1_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_2 = norm_nda[:,:,y//2:,:x//2]
#     norm_2_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_3 = norm_nda[:,:,y//2:,x//2:]
#     norm_3_mean = minmax_lambda([np.mean(norm_3,axis=(1,2,3)),mid,upper])
#     norm_4 = norm_nda[:,:,:y//2,x//2:]
#     norm_4_mean = minmax_lambda([np.mean(norm_4,axis=(1,2,3)),mid,upper])
    

    ind = detect_phases(dir_1d_mean=dir_1d_mean[:length], length=length)
    
    
    #print('ed: {}, ms: {}, es: {}, pf: {}, md: {}'.format(*ind))
    
        # plot the mean/max norm for one patient oveer time
    if plot:
        
        
        # VECT 2D+t
        vect_2d_t = vects_nda[:,z]
        fig1 = show_2D_or_3D(vect_2d_t,allow_slicing=False, f_size=(25,2))
        ax1 = fig1.get_axes()[0]
        _ = ax1.set_ylabel('phi_t \nmid-cavity')

        
        # DIR 2D+t
        dir_2d_t = directions[:,z]
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        #dir_2d_t = minmax_lambda([dir_2d_t,lower,upper])
        div_cmap = 'bwr'
        fig = show_2D_or_3D(dir_2d_t,allow_slicing=False, f_size=(25,2),cmap=div_cmap)
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('alpha_t \nmid-cavity')
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[len(ax.get_images())//2], cax=cax, orientation='horizontal')

        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # DIR 2D T x Z
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        directions_tz = directions.mean(axis=(2,3)) # original: norm()
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        _ = ax.imshow(directions_tz.T, aspect='auto', label='aasd', cmap=div_cmap, origin='lower')
        _ = ax.set_xticks(gtind, minor=False)
        ax2 = ax.twinx()
        _ = ax2.plot(dir_1d_mean, c='black',label='dir 1d+t')
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('dir z+t\napex:base')
        
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1
        
        #ax=fig.add_subplot(rows,1,pos)
        #_ = ax.plot(dir_1d_mean, c='black',label='dir 1d+t')
        #_ = ax.plot(dir_1_mean, label='top left')
        #_ = ax.plot(dir_2_mean, label='top right')
        #_ = ax.plot(dir_3_mean, label='bottom left')
        #_ = ax.plot(dir_4_mean, label='bottom right')
        #ax.legend(loc='upper left')
        #ax.set_xticks(gtind, minor=False)
        #ax.margins(0.022,0)
        #ax.label_outer()
        #pos=pos+1
        #plt.subplots_adjust(wspace=0, hspace=0.1)
        #plt.show()
        norm_cmap = 'hot'
        
        # NORM 2D + t
        norm_2d_t = norm_nda[:,z]
        norm_2d_t = minmax_lambda([norm_2d_t,mid,upper])
        fig = show_2D_or_3D(norm_2d_t,allow_slicing=False, f_size=(25,2),cmap=norm_cmap, interpolation='none')
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('norm 2d+t\nmid-cavity')
        #ax.margins(0,0)
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[0], cax=cax, orientation='horizontal')
        
        
        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # NORM 2D TxZ
        norm2d = minmax_lambda([norm_nda.mean(axis=(2,3)),mid,upper])
        _ = ax.imshow(norm2d.T, aspect='auto', origin='lower',cmap=norm_cmap, interpolation='none')
        _ = ax.set_xticks(gtind, minor=False)
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('norm z+t\napex:base')
        ax2 = ax.twinx()
        _ = ax2.plot(nda_1d_mean, c='black',label='norm 1d+t')
        #ax.spines['top'].set_visible(False)
        #_ = ax.legend(loc='upper left')
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1

        """ax=fig.add_subplot(rows,1,pos)
        _ = ax.plot(norm_mean, label='norm 1d+t')
        #_ = ax.plot(norm_1_mean, label='top left')
        #_ = ax.plot(norm_2_mean, label='top right')
        #_ = ax.plot(norm_3_mean, label='bottom left')
        #_ = ax.plot(norm_4_mean, label='bottom right')
        ax.legend(loc='upper left')
        ax.set_xticks(gtind, minor=False)
        ax.margins(0.022,0)
        ax.label_outer()
        pos = pos+1
        plt.subplots_adjust(wspace=0, hspace=0.1)"""
        #plt.tight_layout()
        plt.show()

        
    
    return ind
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
p = 2#215
gt_length = int(gt_len[p,:,0].sum())
ind = np.argmax(gt[p][:gt_length],axis=0)
get_phases_from_vects(nda_vects[p][:gt_length], length=gt_length,plot=True,dir_axis=0, gtind=ind)
# Tue, 01 Mar 2022 13:17:15
@interact
def compare_phases(i=(0,nda_vects.shape[0]-1), plot=True):
    import matplotlib as mpl
    mpl.rcParams.update(mpl.rcParamsDefault)
    figsize = (25,1)
    weight = 1
    z = 0
    
    phases = ['ED', 'MS', 'ES', 'PF', 'MD']
    cardiac_cycle_length = int(gt_len[i,:,-1].sum())
    # crop or not
    cardiac_cycle_length_ = gt[i].shape[0]-3 # plot the full length minus border for visualisation
    
    vects = nda_vects[i][:cardiac_cycle_length]
    
    
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0) # 
    ind_pred = np.argmax(pred[i][:],axis=0)
    
    temp = mov[i,:cardiac_cycle_length,z] # cardiac_cycle_length_
    fig = show_2D_or_3D(temp,allow_slicing=False)
    ax = fig.get_axes()[0]
    _ = ax.set_ylabel('CMR 2d+t)\nmid-cavity')
    plt.show()
    
    fig= plt.figure(figsize=figsize)
    ax=fig.add_subplot(111)
    ax.set_xticks(ind, minor=False)
    ax.set_yticks([0, 1, 2, 3, 4], minor=False)
    ax.set_yticklabels(phases, rotation=0)
    ind = np.array(ind)
    onehot_gt = np.zeros((ind.size, cardiac_cycle_length)) # cardiac_cycle_length_
    onehot_gt[np.arange(ind.size), ind] = weight
    #ax.imshow(gt[i][:cardiac_cycle_length].T,aspect='auto') # with uncertainity
    ax.imshow(onehot_gt,aspect='auto', cmap='bone') # gt as binary phase2time mapping
    _ = ax.set_yticklabels(phases)
    ax.set_title('gt')
    ax.margins(0,0)
    
    indices = get_phases_from_vects(vects, length=cardiac_cycle_length, plot=plot,dir_axis=0, gtind=ind, figsize=figsize)

    indices = np.array(indices)
    onehot = np.zeros((indices.size, cardiac_cycle_length)) # cardiac_cycle_length_
    onehot[np.arange(indices.size), indices] = weight
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xticks(indices, minor=False)
    ax.set_yticks([0, 1, 2, 3, 4], minor=False)
    ax.set_yticklabels(phases, rotation=0)
    plt.imshow(onehot,aspect='auto', cmap='bone')
    _ = ax.set_yticklabels(phases)
    ax.margins(0,0)
    #ax.set_title('prediction')
    plt.show()
    
    print(phases)
    print('gt:', ind)
    print('u: ', indices)
    
    
    # this would plot the supervised prediction
    #print('p: ', ind_pred)
    """fig, ax = plt.subplots(figsize=figsize)
    ax.set_xticks(ind_pred, minor=False)
    ax.set_yticks([0, 1, 2, 3, 4], minor=False)
    ax.set_yticklabels(phases, rotation=45)
    plt.imshow(pred[i][:cardiac_cycle_length].T,aspect='auto')
    _ = ax.set_yticklabels(phases);plt.show()"""
    # patient 107 gcn --> good curve for plotting
    # 200 no cut
    # 125 no cut
    # 194 moderate cut 
    # 134 strong cut-off
# Tue, 01 Mar 2022 13:17:17
pred_u = np.zeros_like(gt)
upred_ind = []
cycle_len=[]
print(pred_u.shape)
for i in range(pred_u.shape[0]):
    weight = 1
    
    cardiac_cycle_length = int(gt_len[i,:,0].sum())
    cycle_len.append(cardiac_cycle_length)
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
    #print(gt[i][:cardiac_cycle_length].T.shape)
    
    indices = get_phases_from_vects(nda_vects[i][:cardiac_cycle_length], length=cardiac_cycle_length, gtind=ind,plot=False,dir_axis=0)
    upred_ind.append(indices)
    indices = np.array(indices)
    onehot = np.zeros((indices.size, cardiac_cycle_length))
    onehot[np.arange(indices.size), indices] = weight
    pred_u[i][0:cardiac_cycle_length] = onehot.T
upred_ind=np.stack(upred_ind, axis=0)
cycle_len = np.stack(cycle_len, axis=0)
# re-create a compatible shape for the metric fn
gt_ = np.stack([gt,gt_len], axis=1)
pred_ = np.stack([pred_u,np.zeros_like(pred_u)], axis=1)

# create a dataframe for further plots
from src.utils.Metrics import meandiff
phases = ['ED', 'MS', 'ES', 'PF', 'MD']
res = meandiff(gt_, pred_, apply_sum=False, apply_average=False)
df = pd.DataFrame(res.numpy(), columns=phases)
# Tue, 01 Mar 2022 13:17:56
# get the patient idx, per label where this method fails the most
label = 4
error_thres = 10
if type(res) != np.ndarray: res.numpy()
print('label:{} \naFD>{}\nPatient idx: {}'.format(phases[label],error_thres,np.where(res[:,label]>error_thres)[0]))
print('gt length: {}'.format(cycle_len[res[:,label]>error_thres]))
print('gt indices: {}'.format(gt_ind[:,label][(res[:,label]>error_thres)]))
print('pred indices: {}'.format(upred_ind[:,label][(res[:,label]>error_thres)]))
print('aFD: {}'.format(res[:,label][res[:,label]>error_thres]))
# Tue, 01 Mar 2022 13:17:56
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:17:57
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:17:58
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:17:59
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:18:00
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:18:01
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:18:02
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:18:03
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:18:04
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:18:05
# TOF unsupervised center 14_41 gaus 2
sb.set_context('paper')
sb.set(font_scale = 2)
_ =df.plot(kind='box')
ax = sb.violinplot(data=df)
ax = sb.stripplot(data=df)
ax.set_ylim(0,10)
pd.options.display.float_format = "{:,.2f}".format
df_summarized = pd.concat([df.mean(axis=0),df.std(axis=0), df.median(axis=0)], axis=1)
df_summarized.columns = ['mean', 'SD', 'meadian']
print(df_summarized.mean())
print(df_summarized)

sb.set_context('paper')
sb.set(font_scale = 1.8)
fig, axes = plt.subplots(1,5,figsize=(25,4))
axes = axes.flatten()
for i,ax in enumerate(axes):
    _ = ax.scatter(upred_ind[:,i],gt_ind[:,i])
    _ = ax.axline((1, 1), slope=1)
    _ = ax.set_title(phases[i])
    _ = ax.set_xlabel('Index - pred')
    _ = ax.set_ylabel('Index - gt')
    _ = ax.set_xlim([0,35])
    _ = ax.set_ylim([0,35])
plt.tight_layout()
# Tue, 01 Mar 2022 13:18:05
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:18:06
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:18:07
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:18:08
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:18:09
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:18:10
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:18:11
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:18:12
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:18:13
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:18:14
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:18:15
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:18:16
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:18:17
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:18:18
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:18:19
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:18:19
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:18:20
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:18:21
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:18:22
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:18:23
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:18:24
# TOF unsupervised center 14_41 gaus 2
sb.set_context('paper')
sb.set(font_scale = 2)
_ =df.plot(kind='box')
ax = sb.violinplot(data=df)
ax = sb.stripplot(data=df)
ax.set_ylim(-1,10)
pd.options.display.float_format = "{:,.2f}".format
df_summarized = pd.concat([df.mean(axis=0),df.std(axis=0), df.median(axis=0)], axis=1)
df_summarized.columns = ['mean', 'SD', 'meadian']
print(df_summarized.mean())
print(df_summarized)

sb.set_context('paper')
sb.set(font_scale = 1.8)
fig, axes = plt.subplots(1,5,figsize=(25,4))
axes = axes.flatten()
for i,ax in enumerate(axes):
    _ = ax.scatter(upred_ind[:,i],gt_ind[:,i])
    _ = ax.axline((1, 1), slope=1)
    _ = ax.set_title(phases[i])
    _ = ax.set_xlabel('Index - pred')
    _ = ax.set_ylabel('Index - gt')
    _ = ax.set_xlim([0,35])
    _ = ax.set_ylim([0,35])
plt.tight_layout()
# Tue, 01 Mar 2022 13:18:25
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:18:43
# TOF unsupervised center 14_41 gaus 2
sb.set_context('paper')
sb.set(font_scale = 2)
_ =df.plot(kind='box')
ax = sb.violinplot(data=df)
ax = sb.stripplot(data=df)
ax.set_ylim(0,10)
pd.options.display.float_format = "{:,.2f}".format
df_summarized = pd.concat([df.mean(axis=0),df.std(axis=0), df.median(axis=0)], axis=1)
df_summarized.columns = ['mean', 'SD', 'meadian']
print(df_summarized.mean())
print(df_summarized)

sb.set_context('paper')
sb.set(font_scale = 1.8)
fig, axes = plt.subplots(1,5,figsize=(25,4))
axes = axes.flatten()
for i,ax in enumerate(axes):
    _ = ax.scatter(upred_ind[:,i],gt_ind[:,i])
    _ = ax.axline((1, 1), slope=1)
    _ = ax.set_title(phases[i])
    _ = ax.set_xlabel('Index - pred')
    _ = ax.set_ylabel('Index - gt')
    _ = ax.set_xlim([0,35])
    _ = ax.set_ylim([0,35])
plt.tight_layout()
# Tue, 01 Mar 2022 13:18:44
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:18:57
pred_u = np.zeros_like(gt)
upred_ind = []
cycle_len=[]
print(pred_u.shape)
for i in range(pred_u.shape[0]):
    weight = 1
    
    cardiac_cycle_length = int(gt_len[i,:,0].sum())
    cycle_len.append(cardiac_cycle_length)
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
    #print(gt[i][:cardiac_cycle_length].T.shape)
    
    indices = get_phases_from_vects(nda_vects[i][:cardiac_cycle_length], length=cardiac_cycle_length, gtind=ind,plot=False,dir_axis=0)
    upred_ind.append(indices)
    indices = np.array(indices)
    onehot = np.zeros((indices.size, cardiac_cycle_length))
    onehot[np.arange(indices.size), indices] = weight
    pred_u[i][0:cardiac_cycle_length] = onehot.T
upred_ind=np.stack(upred_ind, axis=0)
cycle_len = np.stack(cycle_len, axis=0)
# re-create a compatible shape for the metric fn
gt_ = np.stack([gt,gt_len], axis=1)
pred_ = np.stack([pred_u,np.zeros_like(pred_u)], axis=1)

# create a dataframe for further plots
from src.utils.Metrics import meandiff
phases = ['ED', 'MS', 'ES', 'PF', 'MD']
res = meandiff(gt_, pred_, apply_sum=False, apply_average=False)
df = pd.DataFrame(res.numpy(), columns=phases)
# Tue, 01 Mar 2022 13:19:37
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:19:38
# get the patient idx, per label where this method fails the most
label = 4
error_thres = 10
if type(res) != np.ndarray: res.numpy()
print('label:{} \naFD>{}\nPatient idx: {}'.format(phases[label],error_thres,np.where(res[:,label]>error_thres)[0]))
print('gt length: {}'.format(cycle_len[res[:,label]>error_thres]))
print('gt indices: {}'.format(gt_ind[:,label][(res[:,label]>error_thres)]))
print('pred indices: {}'.format(upred_ind[:,label][(res[:,label]>error_thres)]))
print('aFD: {}'.format(res[:,label][res[:,label]>error_thres]))
# Tue, 01 Mar 2022 13:19:38
# TOF unsupervised center 14_41 gaus 2
sb.set_context('paper')
sb.set(font_scale = 2)
_ =df.plot(kind='box')
ax = sb.violinplot(data=df)
ax = sb.stripplot(data=df)
ax.set_ylim(0,10)
pd.options.display.float_format = "{:,.2f}".format
df_summarized = pd.concat([df.mean(axis=0),df.std(axis=0), df.median(axis=0)], axis=1)
df_summarized.columns = ['mean', 'SD', 'meadian']
print(df_summarized.mean())
print(df_summarized)

sb.set_context('paper')
sb.set(font_scale = 1.8)
fig, axes = plt.subplots(1,5,figsize=(25,4))
axes = axes.flatten()
for i,ax in enumerate(axes):
    _ = ax.scatter(upred_ind[:,i],gt_ind[:,i])
    _ = ax.axline((1, 1), slope=1)
    _ = ax.set_title(phases[i])
    _ = ax.set_xlabel('Index - pred')
    _ = ax.set_ylabel('Index - gt')
    _ = ax.set_xlim([0,35])
    _ = ax.set_ylim([0,35])
plt.tight_layout()
# Tue, 01 Mar 2022 13:19:38
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:19:39
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:21:36
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:21:58
# load the vectors
pathstovectnpy = sorted(glob.glob(os.path.join(vects_chooser.selected, 'moved', '*vects*.npy')))
print(pathstovectnpy)
nda_vects = np.concatenate([np.load(path_) for path_ in pathstovectnpy],axis=0)
print(nda_vects.shape)
# Tue, 01 Mar 2022 13:22:19
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:22:20
# load the phase gt and pred
pred_path = os.path.join(vects_chooser.selected, 'pred')
pathtsophasenpy = sorted(glob.glob(os.path.join(pred_path, '*gtpred*.npy')))
print(pathtsophasenpy)
nda_phase = np.concatenate([np.load(path_) for path_ in pathtsophasenpy],axis=1)
print(nda_phase.shape)
gt_, pred_ = np.split(nda_phase, axis=0,indices_or_sections=2)
print(gt_.shape)
gt= gt_[0,:,0]
pred = pred_[0,:,0]
print(gt.shape)
gt_len = gt_[0,:,1]
# Tue, 01 Mar 2022 13:22:20
# load some moved examples for easier understanding of the dimensions
pathtomoved = sorted(glob.glob(os.path.join(vects_chooser.selected, 'moved','*moved*.npy')))
print(len(pathtomoved))
mov = np.concatenate([np.load(path_) for path_ in pathtomoved],axis=0)
print(mov.shape) # patient,time,spatial-z,y,x,channel, e.g.: 69,40,16,64,64,1
# Tue, 01 Mar 2022 13:22:27
# plot one moved timestep of one patient = one 3D volume 
# should go from apex to base
from src.visualization.Visualize import show_2D_or_3D
temp = mov[10,0,...,0]
_ = show_2D_or_3D(temp, cmap='gray', interpolation='none')
# Tue, 01 Mar 2022 13:22:27
import tensorflow as tf
from tensorflow.image import ssim
ssim(mov[10,0], mov[10,1],max_val=1.0, filter_size=11,
                          filter_sigma=1.5, k1=0.01, k2=0.03).numpy().mean()
# Tue, 01 Mar 2022 13:22:27
from skimage.metrics import structural_similarity as ssim
ssim(mov[10,0], mov[10,1],multichannel=True)
# Tue, 01 Mar 2022 13:22:27
plt.imshow(gt[0].T)
# Tue, 01 Mar 2022 13:22:27
plt.imshow(pred[0].T)
# Tue, 01 Mar 2022 13:22:28
import tensorflow as tf
import sys
# returns a matrix with the indicies as values, similar to np.indicies
def get_idxs_tf(x):
    return tf.cast(
        tf.reshape(tf.where(tf.ones((x[0], x[1], x[2]))), (x[0], x[1], x[2], 3)),
        tf.float32)


# returns a matrix with vectors pointing to the center
def get_centers_tf(x):
    return tf.cast(
        tf.tile(tf.convert_to_tensor([x[0] // 2, x[1] // 2, x[2] // 2])[tf.newaxis, tf.newaxis, tf.newaxis, ...],
                (x[0], x[1], x[2], 1)), tf.float32)

def get_angle_tf(a, b):
    # this should work for batches of n-dimensional vectors
    # α = arccos[(a · b) / (|a| * |b|)]
    # |v| = √(x² + y² + z²)
    """
    in 3D space
    If vectors a = [xa, ya, za], b = [xb, yb, zb], then:
    α = arccos[(xa * xb + ya * yb + za * zb) / (√(xa2 + ya2 + za2) * √(xb2 + yb2 + zb2))]
    """
    import math as m
    pi = tf.constant(m.pi)
    b = tf.cast(b, dtype=a.dtype)
    inner = tf.einsum('...i,...i->...', a, b)
    norms = tf.norm(a, ord='euclidean', axis=-1) * tf.norm(b, ord='euclidean', axis=-1)  # [...,None]
    cos = inner / (norms + sys.float_info.epsilon)
    #rad = tf.math.acos(tf.clip_by_value(cos, -1.0, 1.0))
    # rad2deg conversion
    #deg = rad * (180.0/pi)
    return cos[..., tf.newaxis]

dim = [16,64,64]
idx = get_idxs_tf(dim)
c = get_centers_tf(dim)
centers = c - idx
centers_tensor = centers[tf.newaxis, ...]
flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x, centers_tensor), name='flow2direction')
minmax_lambda = lambda x : x[1] + (((x[0] - np.min(x[0])) * (x[2]-x[1]))/(np.max(x[0]) - np.min(x[0])))
# define some central params
lower, mid, upper = -1,0, 1
# Tue, 01 Mar 2022 13:22:28
# align norm an direction by the cardiac phase ED
# resample all 1D feature vectors to the same length
# this should help to validate the pre-defined rules, and if both together explains the cardiac phases
target_t = 30
norms = []
dirs= []
cycle_len=[]
gt_ind = []
gt_ind_rolled = []
xval = np.linspace(0,1,target_t)

for i,n in enumerate(nda_vects[:]):
    cardiac_cycle_length = int(gt_len[i,:,0].sum())
    cycle_len.append(cardiac_cycle_length)
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
    gt_onehot = gt[i][:cardiac_cycle_length]
    gt_ind.append(ind)
    
    dir_axis=0
    gt_ed = ind[0]
    #assert cardiac_cycle_length>=gt_ed
    temp = n[:cardiac_cycle_length]
    
    dim = temp.shape[1:-1]
    #print(dim)
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    #print('centers: ',c.dtype)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    # direction
    directions = flow2direction_lambda(temp)[...,0]
    # direction mean
    directions = np.mean(directions,axis=(1,2,3))
    # direction ed aligned
    directions = np.roll(directions, -1*gt_ed)
    # direction interpolate to unique length
    f = interp1d(np.linspace(0,1,directions.shape[0]), directions, kind='linear')
    directions = f(xval)
    # direction min/max normalised between -1,1

    directions = minmax_lambda([directions,lower,upper])
    
    # magnitude/norm as mean
    norm_full = np.linalg.norm(temp, axis=-1)
    norm_nda = norm_full.mean(axis=(1,2,3))
    # norm ed aligned
    norm_nda = np.roll(norm_nda, -1*gt_ed)
    # interpolate to unique length
    #f = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')
    #norm_nda = f(xval)
    norm_nda = np.interp(xval, np.linspace(0,1,norm_nda.shape[0]), norm_nda)
    # norm min/max aligned to 0,1
    norm_nda = minmax_lambda([norm_nda,mid,upper])
    
    # roll, scale, round and clip the gt indicies, to get an aligned distribution of the labels
    gt_onehot_rolled = np.roll(gt_onehot, -1*gt_ed, axis=0)
    resize_factor = target_t/cardiac_cycle_length
    gt_onehot_rolled = np.argmax(gt_onehot_rolled,axis=0)
    gt_onehot_rolled = np.clip(np.rint(gt_onehot_rolled * resize_factor),a_min=0,a_max=target_t-1)
    gt_ind_rolled.append(gt_onehot_rolled)
    
    norms.append(norm_nda)
    dirs.append(directions)
# Tue, 01 Mar 2022 13:22:40
# get the mean values
gt_ind_scaled = np.stack(gt_ind_rolled, axis=0)
gt_ind = np.stack(gt_ind,axis=0)
cycle_len = np.stack(cycle_len, axis=0)
norms = np.stack(norms, axis=0)
dirs = np.stack(dirs, axis=0)
norms_m = [norms.mean(axis=0)]
dirs_m=[dirs.mean(axis=0)]
# Tue, 01 Mar 2022 13:22:40
# plot the distribution of the aligned and scaled gt indicies
fig, (ax1,ax2, ax3) = plt.subplots(1,3, figsize=(15,5))
temp = pd.DataFrame(gt_ind_scaled, columns=phases)
# TODO check this outlier with MD ==2
temp = temp[temp['MD']>10]
print(temp.mean(), temp.std())
ax1 = sb.violinplot(data=temp, ax=ax1)
ax1 = sb.stripplot(data=temp, ax=ax1)
#_ = temp.plot(kind='box', ax=ax1)
_ = temp.plot(kind='line', ax=ax2, legend=None)
_ = temp.plot(kind='hist', ax=ax3, legend=None)
ax3.set_ylabel('')
# Tue, 01 Mar 2022 13:22:40
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:22:41
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:22:42
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:22:43
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:22:44
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:22:45
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:22:46
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:22:47
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:22:48
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:22:49
# plot one line per patient - norm
fig, ax = plt.subplots(figsize=(25,5))
for n in norms:
    _ = plt.plot(n)
_ = ax.set_title('Magnitudes aligned at ED, resampling shape 40,')
# Tue, 01 Mar 2022 13:22:49
# plot one line per direction
fig, ax = plt.subplots(figsize=(25,5))
for n in dirs:
    _ = plt.plot(n)
_ = ax.set_title('Directions aligned at ED, resampling shape 40,')
# Tue, 01 Mar 2022 13:22:50
from src.visualization.Visualize import show_2D_or_3D
from src.data.Preprocess import clip_quantile
nda_temp = nda_vects[0]
# norm_1 = nda_temp[:,:,:32,:32]# top left
norm_1 = nda_temp[:,:,:32,32:]# top right
norm_1 = np.linalg.norm(norm_1, axis=-1)
nda_norm_ = np.linalg.norm(nda_temp, axis=-1)
plt.hist(nda_norm_.flatten());plt.show()
nda_norm = clip_quantile(nda_norm_, 0.99)
nda_norm = minmax_lambda([nda_norm,0,1])
plt.hist(nda_norm.flatten())
plt.show()
# mask phi with a threshold norm matrix 
nda_msk = (nda_norm>=0.2).astype(np.float32)
nda_temp = nda_norm * nda_msk
#plt.hist(nda_temp.flatten())
print(nda_temp.shape)
_ = show_2D_or_3D(nda_norm_[:,0], allow_slicing=True)
_ = show_2D_or_3D(nda_temp[:,0], allow_slicing=True)
_ = show_2D_or_3D(nda_temp[:,-1], allow_slicing=True)

# top right
_ = show_2D_or_3D(norm_1[:,0], allow_slicing=True,cmap='gray', interpolation='none')
# Tue, 01 Mar 2022 13:22:52
fig, ax = plt.subplots(figsize=(25,3))
plt.plot(np.argmax(gt[0], axis=1))
ax.set_yticks([0, 1, 2, 3, 4], minor=False)
phases = ['ED', 'MS', 'ES', 'PF', 'MD']
_ = ax.set_yticklabels(phases)
# Tue, 01 Mar 2022 13:22:52
# plot the mean/max norm for one patient oveer time
nda_1d_max = np.max(nda_temp,axis=(1,2,3))
nda_1d_mean = np.mean(nda_temp,axis=(1,2,3))
nda_1d_sum = np.sum(nda_temp,axis=(1,2,3))

fig, ax = plt.subplots(figsize=(25,3))
ax.set_title('max norm')
_ = plt.plot(nda_1d_max); plt.show()

fig, ax = plt.subplots(figsize=(25,3))
ax.set_title('mean norm')
_ = plt.plot(nda_1d_mean)

fig, ax = plt.subplots(figsize=(25,3))
ax.set_title('mean sum')
_ = plt.plot(nda_1d_sum)
#ax.set_ylim(0.0,0.15)
# Tue, 01 Mar 2022 13:22:52
temp = np.arange(10)
temp[:0]
# Tue, 01 Mar 2022 13:22:52
def detect_phases(dir_1d_mean, length):
    
    import scipy.signal as sig
    
        # MS
    # Global min of f(x)
    peaks, prop = sig.find_peaks(-1*dir_1d_mean, height=0.4)
    if len(peaks)>1:
        ms = peaks[0]
    else:
        ms = np.argmin(dir_1d_mean)
    ms = ms -1 # take the bucket before the first min peak
    
    
    # ES
    # First time f(x)>0 after MS
    cycle = np.concatenate([dir_1d_mean[ms:], dir_1d_mean[:ms]])
    temp_ = 0
    es_found=False
    negative_slope = False
    for idx,elem in enumerate(cycle):
        if elem<0:
            negative_slope=True
            temp_ = idx
        elif elem>=0 and negative_slope:
            es_found = True
            #temp_ = idx
            break # stop after first zero-transition
    if es_found:
        es = ms + temp_
        #es = es-1
    else:
        es = ms + 1 # the frame after ms, fallback
    if es>=length:
        es = np.mod(es,length)
        print('ES overflow: {}, ms:{}'.format(es,ms))
     
    
    # PF
    # First peak after ES, min height 0.6
    seq = dir_1d_mean[es:]
    peaks, prop = sig.find_peaks(seq, height=0.6)#height=0.6 we normalise between -1 and 1, PF should be close to argmax

    if len(peaks>0):
        pf = es + peaks[0] # take the peak after es
        pf = pf -1
    else:
        print('pf not clear, set to ES {} + 1'.format(es))
        pf = es + 1
    pf = np.mod(pf, length)
      

    # ED
    # Between pf and ms: last time f(x) cross zero from positive to negative
    # a priori knowledge ED needs a minimal distance of 2 frames towards MS
    # CHANGED the minimal distance between ED and MS
    cycle = np.concatenate([dir_1d_mean[pf:], dir_1d_mean[:ms]])
    #print(cycle)
    ed_found = False
    last_idx_positive = True # we start at the pf, which is the peak(dir)
    for idx,elem in enumerate(cycle):
        
        if elem>=0:
            last_idx_positive = True
        # remember the last idx before the direction gets negative the last time before ms
        elif elem<0 and last_idx_positive: # first time direction negative
            ed_found = True # for fallbacks
            temp_ = idx # idx before negative direction
            #print('found transition at: {}'.format(idx))
            last_idx_positive = False # remember only the first idx after transition
        
    if ed_found:
        ed = pf + temp_
        #print('ed:{}, pf:{}, temp_:{}, lenght: {}'.format(ed,pf,temp_,length))
    else: 
        # if we dont find a transition from positive to negative, take the idx which is the closest to zero
        temp_ = np.argmin(np.abs(cycle)) # make sure we have a minimal distance
        ed = pf + temp_
        print('ED: no transition found between {}-{} , closest id to 0: {}, ed = {}'.format(pf,ms,temp_,ed))
    
    if ed>=length:
        #print('gt ed: {}, ed: {}, length: {}'.format(gted,ed,length))
        ed = np.mod(ed,length)
        #print('mod ed: {}'.format(ed))
    #ed = ed-1 # take the bucket before negative
        
    # MD
    # Middle between PF and ED
    ed_slice_idx = ed
    if ed_slice_idx<=pf: # ed overflow --> beginning of cmr stack
        ed_slice_idx = length + ed
    md = (pf+ed_slice_idx)//2 # the bucket after the middle
    md = md + 1
    md = np.mod(md,length)
    
    return np.array([ed,ms,es,pf,md])
# Tue, 01 Mar 2022 13:22:52
# try to predict the phases unsupervised just from the norm/direction curve

def get_phases_from_vects(vects_nda, length=-1, plot=True, dir_axis=0, gtind=[], figsize=(25,3)):
    import scipy.signal
    import scipy.signal as sig
    import scipy.ndimage
    from scipy.ndimage import gaussian_filter1d
    
    gted=gtind[0]
    
    # signal smoothing
    kernel_size = 4
    kernel = np.ones(kernel_size) / kernel_size
    z = 0
    
    
    dim_z = vects_nda.shape[1]
    #vects_nda = vects_nda[:int(length)]
    
    
    # define some helper lambdas
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    dim = vects_nda.shape[1:-1]
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]

    
    # norm of the vector
    norm_nda = np.linalg.norm(vects_nda[...,dir_axis:], axis=-1)
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    norm_nda_thres = clip_quantile(norm_nda, 0.9)
    norm_nda = minmax_lambda([norm_nda,0,1])
    norm_msk = norm_nda>=0.0 # this could be used to mask v_t by a threshold norm
    norm_nda = norm_nda * norm_msk
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    nda_1d_mean = np.mean(norm_nda,axis=(1,2,3))
    nda_1d_max = np.max(norm_nda,axis=(1,2,3))
    nda_1d_median = np.median(norm_nda,axis=(1,2,3))
    nda_1d_mean_smooth = np.convolve(nda_1d_mean, kernel, mode='same')
    #print(nda_1d_mean_smooth.shape, nda_1d_mean.shape)
    
    
    # direction relative to the center
    
    directions = flow2direction_lambda(vects_nda)[...,0].numpy()
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    directions = minmax_lambda([directions,lower,upper])
    #plt.hist(directions.flatten());plt.show()
    directions = directions * norm_msk
    #plt.hist(directions.flatten());plt.show()
    
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    dir_1d_mean = np.mean(directions,axis=(1,2,3))
    dir_1d_median = np.median(directions,axis=(1,2,3))

    
    # Find phases
    # 1st smooth the direction with a rolling averge (kernelsize=4)
    # or via gaussian filter
    # 2nd min/max normalise each direction vector in a range [-1:1]
    
    
    dir_1d_mean = gaussian_filter1d(dir_1d_mean, sigma=2, mode='wrap') # original: dir_1d_mean
    
    ################# CHANGED !!!!!! ######################## RECENTLY NO CHANGES
    # min/max scaling
    dir_1d_mean = minmax_lambda([dir_1d_mean,lower,upper])
    
    
#     # test with corner specific vector fields
#     y,x = directions.shape[-2:]
    
#     vects_mean = minmax_lambda([np.mean(vects_nda,axis=(1,2,3,4)),lower,upper])
#     vects_1 = vects_nda[:,:,:y//2,:x//2,:]
#     vects_1_mean = minmax_lambda([np.mean(vects_1,axis=(1,2,3,4)),lower,upper])
#     vects_2 = vects_nda[:,:,y//2:,:x//2,:]
#     vects_2_mean = minmax_lambda([np.mean(vects_2,axis=(1,2,3,4)),lower,upper])
#     vects_3 = vects_nda[:,:,y//2:,x//2:,:]
#     vects_3_mean = minmax_lambda([np.mean(vects_3,axis=(1,2,3,4)),lower,upper])
#     vects_4 = vects_nda[:,:,:y//2,x//2:,:]
#     vects_4_mean = minmax_lambda([np.mean(vects_4,axis=(1,2,3,4)),lower,upper])
    
    
#     # direction per corner
#     dir_1 = directions[:,:,:y//2,:x//2] # top left, plot
#     dir_1_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_2 = directions[:,:,y//2:,:x//2] # top right, plot
#     dir_2_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_3 = directions[:,:,y//2:,x//2:] # bottom left
#     dir_3_mean = minmax_lambda([np.mean(dir_3,axis=(1,2,3)),lower,upper])
#     dir_4 = directions[:,:,:y//2,x//2:] # bottom right
#     dir_4_mean = minmax_lambda([np.mean(dir_4,axis=(1,2,3)),lower,upper])
    

#     # norm per corner
#     norm_mean = minmax_lambda([nda_1d_mean, mid, upper])
#     norm_1 = norm_nda[:,:,:y//2,:x//2]
#     norm_1_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_2 = norm_nda[:,:,y//2:,:x//2]
#     norm_2_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_3 = norm_nda[:,:,y//2:,x//2:]
#     norm_3_mean = minmax_lambda([np.mean(norm_3,axis=(1,2,3)),mid,upper])
#     norm_4 = norm_nda[:,:,:y//2,x//2:]
#     norm_4_mean = minmax_lambda([np.mean(norm_4,axis=(1,2,3)),mid,upper])
    

    ind = detect_phases(dir_1d_mean=dir_1d_mean[:length], length=length)
    
    
    #print('ed: {}, ms: {}, es: {}, pf: {}, md: {}'.format(*ind))
    
        # plot the mean/max norm for one patient oveer time
    if plot:
        
        
        # VECT 2D+t
        vect_2d_t = vects_nda[:,z]
        fig1 = show_2D_or_3D(vect_2d_t,allow_slicing=False, f_size=(25,2))
        ax1 = fig1.get_axes()[0]
        _ = ax1.set_ylabel('phi_t \nmid-cavity')

        
        # DIR 2D+t
        dir_2d_t = directions[:,z]
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        #dir_2d_t = minmax_lambda([dir_2d_t,lower,upper])
        div_cmap = 'bwr'
        fig = show_2D_or_3D(dir_2d_t,allow_slicing=False, f_size=(25,2),cmap=div_cmap)
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('alpha_t \nmid-cavity')
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[len(ax.get_images())//2], cax=cax, orientation='horizontal')

        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # DIR 2D T x Z
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        directions_tz = directions.mean(axis=(2,3)) # original: norm()
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        _ = ax.imshow(directions_tz.T, aspect='auto', label='aasd', cmap=div_cmap, origin='lower')
        _ = ax.set_xticks(gtind, minor=False)
        ax2 = ax.twinx()
        _ = ax2.plot(dir_1d_mean, c='black',label='dir 1d+t')
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('dir z+t\napex:base')
        
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1
        
        #ax=fig.add_subplot(rows,1,pos)
        #_ = ax.plot(dir_1d_mean, c='black',label='dir 1d+t')
        #_ = ax.plot(dir_1_mean, label='top left')
        #_ = ax.plot(dir_2_mean, label='top right')
        #_ = ax.plot(dir_3_mean, label='bottom left')
        #_ = ax.plot(dir_4_mean, label='bottom right')
        #ax.legend(loc='upper left')
        #ax.set_xticks(gtind, minor=False)
        #ax.margins(0.022,0)
        #ax.label_outer()
        #pos=pos+1
        #plt.subplots_adjust(wspace=0, hspace=0.1)
        #plt.show()
        norm_cmap = 'hot'
        
        # NORM 2D + t
        norm_2d_t = norm_nda[:,z]
        norm_2d_t = minmax_lambda([norm_2d_t,mid,upper])
        fig = show_2D_or_3D(norm_2d_t,allow_slicing=False, f_size=(25,2),cmap=norm_cmap, interpolation='none')
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('norm 2d+t\nmid-cavity')
        #ax.margins(0,0)
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[0], cax=cax, orientation='horizontal')
        
        
        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # NORM 2D TxZ
        norm2d = minmax_lambda([norm_nda.mean(axis=(2,3)),mid,upper])
        _ = ax.imshow(norm2d.T, aspect='auto', origin='lower',cmap=norm_cmap, interpolation='none')
        _ = ax.set_xticks(gtind, minor=False)
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('norm z+t\napex:base')
        ax2 = ax.twinx()
        _ = ax2.plot(nda_1d_mean, c='black',label='norm 1d+t')
        #ax.spines['top'].set_visible(False)
        #_ = ax.legend(loc='upper left')
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1

        """ax=fig.add_subplot(rows,1,pos)
        _ = ax.plot(norm_mean, label='norm 1d+t')
        #_ = ax.plot(norm_1_mean, label='top left')
        #_ = ax.plot(norm_2_mean, label='top right')
        #_ = ax.plot(norm_3_mean, label='bottom left')
        #_ = ax.plot(norm_4_mean, label='bottom right')
        ax.legend(loc='upper left')
        ax.set_xticks(gtind, minor=False)
        ax.margins(0.022,0)
        ax.label_outer()
        pos = pos+1
        plt.subplots_adjust(wspace=0, hspace=0.1)"""
        #plt.tight_layout()
        plt.show()

        
    
    return ind
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
p = 2#215
gt_length = int(gt_len[p,:,0].sum())
ind = np.argmax(gt[p][:gt_length],axis=0)
get_phases_from_vects(nda_vects[p][:gt_length], length=gt_length,plot=True,dir_axis=0, gtind=ind)
# Tue, 01 Mar 2022 13:22:56
@interact
def compare_phases(i=(0,nda_vects.shape[0]-1), plot=True):
    import matplotlib as mpl
    mpl.rcParams.update(mpl.rcParamsDefault)
    figsize = (25,1)
    weight = 1
    z = 0
    
    phases = ['ED', 'MS', 'ES', 'PF', 'MD']
    cardiac_cycle_length = int(gt_len[i,:,-1].sum())
    # crop or not
    cardiac_cycle_length_ = gt[i].shape[0]-3 # plot the full length minus border for visualisation
    
    vects = nda_vects[i][:cardiac_cycle_length]
    
    
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0) # 
    ind_pred = np.argmax(pred[i][:],axis=0)
    
    temp = mov[i,:cardiac_cycle_length,z] # cardiac_cycle_length_
    fig = show_2D_or_3D(temp,allow_slicing=False)
    ax = fig.get_axes()[0]
    _ = ax.set_ylabel('CMR 2d+t)\nmid-cavity')
    plt.show()
    
    fig= plt.figure(figsize=figsize)
    ax=fig.add_subplot(111)
    ax.set_xticks(ind, minor=False)
    ax.set_yticks([0, 1, 2, 3, 4], minor=False)
    ax.set_yticklabels(phases, rotation=0)
    ind = np.array(ind)
    onehot_gt = np.zeros((ind.size, cardiac_cycle_length)) # cardiac_cycle_length_
    onehot_gt[np.arange(ind.size), ind] = weight
    #ax.imshow(gt[i][:cardiac_cycle_length].T,aspect='auto') # with uncertainity
    ax.imshow(onehot_gt,aspect='auto', cmap='bone') # gt as binary phase2time mapping
    _ = ax.set_yticklabels(phases)
    ax.set_title('gt')
    ax.margins(0,0)
    
    indices = get_phases_from_vects(vects, length=cardiac_cycle_length, plot=plot,dir_axis=0, gtind=ind, figsize=figsize)

    indices = np.array(indices)
    onehot = np.zeros((indices.size, cardiac_cycle_length)) # cardiac_cycle_length_
    onehot[np.arange(indices.size), indices] = weight
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xticks(indices, minor=False)
    ax.set_yticks([0, 1, 2, 3, 4], minor=False)
    ax.set_yticklabels(phases, rotation=0)
    plt.imshow(onehot,aspect='auto', cmap='bone')
    _ = ax.set_yticklabels(phases)
    ax.margins(0,0)
    #ax.set_title('prediction')
    plt.show()
    
    print(phases)
    print('gt:', ind)
    print('u: ', indices)
    
    
    # this would plot the supervised prediction
    #print('p: ', ind_pred)
    """fig, ax = plt.subplots(figsize=figsize)
    ax.set_xticks(ind_pred, minor=False)
    ax.set_yticks([0, 1, 2, 3, 4], minor=False)
    ax.set_yticklabels(phases, rotation=45)
    plt.imshow(pred[i][:cardiac_cycle_length].T,aspect='auto')
    _ = ax.set_yticklabels(phases);plt.show()"""
    # patient 107 gcn --> good curve for plotting
    # 200 no cut
    # 125 no cut
    # 194 moderate cut 
    # 134 strong cut-off
# Tue, 01 Mar 2022 13:22:58
pred_u = np.zeros_like(gt)
upred_ind = []
cycle_len=[]
print(pred_u.shape)
for i in range(pred_u.shape[0]):
    weight = 1
    
    cardiac_cycle_length = int(gt_len[i,:,0].sum())
    cycle_len.append(cardiac_cycle_length)
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
    #print(gt[i][:cardiac_cycle_length].T.shape)
    
    indices = get_phases_from_vects(nda_vects[i][:cardiac_cycle_length], length=cardiac_cycle_length, gtind=ind,plot=False,dir_axis=0)
    upred_ind.append(indices)
    indices = np.array(indices)
    onehot = np.zeros((indices.size, cardiac_cycle_length))
    onehot[np.arange(indices.size), indices] = weight
    pred_u[i][0:cardiac_cycle_length] = onehot.T
upred_ind=np.stack(upred_ind, axis=0)
cycle_len = np.stack(cycle_len, axis=0)
# re-create a compatible shape for the metric fn
gt_ = np.stack([gt,gt_len], axis=1)
pred_ = np.stack([pred_u,np.zeros_like(pred_u)], axis=1)

# create a dataframe for further plots
from src.utils.Metrics import meandiff
phases = ['ED', 'MS', 'ES', 'PF', 'MD']
res = meandiff(gt_, pred_, apply_sum=False, apply_average=False)
df = pd.DataFrame(res.numpy(), columns=phases)
# Tue, 01 Mar 2022 13:23:15
# get the patient idx, per label where this method fails the most
label = 4
error_thres = 10
if type(res) != np.ndarray: res.numpy()
print('label:{} \naFD>{}\nPatient idx: {}'.format(phases[label],error_thres,np.where(res[:,label]>error_thres)[0]))
print('gt length: {}'.format(cycle_len[res[:,label]>error_thres]))
print('gt indices: {}'.format(gt_ind[:,label][(res[:,label]>error_thres)]))
print('pred indices: {}'.format(upred_ind[:,label][(res[:,label]>error_thres)]))
print('aFD: {}'.format(res[:,label][res[:,label]>error_thres]))
# Tue, 01 Mar 2022 13:23:15
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:23:16
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:23:17
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:23:18
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:23:19
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:23:20
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:23:21
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:23:22
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:23:23
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:23:23
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:23:24
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:23:25
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:23:26
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:23:27
# TOF unsupervised center 14_41 gaus 2
sb.set_context('paper')
sb.set(font_scale = 2)
_ =df.plot(kind='box')
ax = sb.violinplot(data=df)
ax = sb.stripplot(data=df)
ax.set_ylim(0,10)
pd.options.display.float_format = "{:,.2f}".format
df_summarized = pd.concat([df.mean(axis=0),df.std(axis=0), df.median(axis=0)], axis=1)
df_summarized.columns = ['mean', 'SD', 'meadian']
print(df_summarized.mean())
print(df_summarized)

sb.set_context('paper')
sb.set(font_scale = 1.8)
fig, axes = plt.subplots(1,5,figsize=(25,4))
axes = axes.flatten()
for i,ax in enumerate(axes):
    _ = ax.scatter(upred_ind[:,i],gt_ind[:,i])
    _ = ax.axline((1, 1), slope=1)
    _ = ax.set_title(phases[i])
    _ = ax.set_xlabel('Index - pred')
    _ = ax.set_ylabel('Index - gt')
    _ = ax.set_xlim([0,35])
    _ = ax.set_ylim([0,35])
plt.tight_layout()
# Tue, 01 Mar 2022 13:23:28
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:24:08
# TOF unsupervised center 14_41 gaus 2
sb.set_context('paper')
sb.set(font_scale = 2)
_ =df.plot(kind='box')
ax = sb.violinplot(data=df)
ax = sb.stripplot(data=df)
ax.set_ylim(0,11)
pd.options.display.float_format = "{:,.2f}".format
df_summarized = pd.concat([df.mean(axis=0),df.std(axis=0), df.median(axis=0)], axis=1)
df_summarized.columns = ['mean', 'SD', 'meadian']
print(df_summarized.mean())
print(df_summarized)

sb.set_context('paper')
sb.set(font_scale = 1.8)
fig, axes = plt.subplots(1,5,figsize=(25,4))
axes = axes.flatten()
for i,ax in enumerate(axes):
    _ = ax.scatter(upred_ind[:,i],gt_ind[:,i])
    _ = ax.axline((1, 1), slope=1)
    _ = ax.set_title(phases[i])
    _ = ax.set_xlabel('Index - pred')
    _ = ax.set_ylabel('Index - gt')
    _ = ax.set_xlim([0,35])
    _ = ax.set_ylim([0,35])
plt.tight_layout()
# Tue, 01 Mar 2022 13:24:08
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:24:38
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:24:57
# load the vectors
pathstovectnpy = sorted(glob.glob(os.path.join(vects_chooser.selected, 'moved', '*vects*.npy')))
print(pathstovectnpy)
nda_vects = np.concatenate([np.load(path_) for path_ in pathstovectnpy],axis=0)
print(nda_vects.shape)
# Tue, 01 Mar 2022 13:25:17
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:25:18
# load the phase gt and pred
pred_path = os.path.join(vects_chooser.selected, 'pred')
pathtsophasenpy = sorted(glob.glob(os.path.join(pred_path, '*gtpred*.npy')))
print(pathtsophasenpy)
nda_phase = np.concatenate([np.load(path_) for path_ in pathtsophasenpy],axis=1)
print(nda_phase.shape)
gt_, pred_ = np.split(nda_phase, axis=0,indices_or_sections=2)
print(gt_.shape)
gt= gt_[0,:,0]
pred = pred_[0,:,0]
print(gt.shape)
gt_len = gt_[0,:,1]
# Tue, 01 Mar 2022 13:25:18
# load some moved examples for easier understanding of the dimensions
pathtomoved = sorted(glob.glob(os.path.join(vects_chooser.selected, 'moved','*moved*.npy')))
print(len(pathtomoved))
mov = np.concatenate([np.load(path_) for path_ in pathtomoved],axis=0)
print(mov.shape) # patient,time,spatial-z,y,x,channel, e.g.: 69,40,16,64,64,1
# Tue, 01 Mar 2022 13:25:20
# plot one moved timestep of one patient = one 3D volume 
# should go from apex to base
from src.visualization.Visualize import show_2D_or_3D
temp = mov[10,0,...,0]
_ = show_2D_or_3D(temp, cmap='gray', interpolation='none')
# Tue, 01 Mar 2022 13:25:20
import tensorflow as tf
from tensorflow.image import ssim
ssim(mov[10,0], mov[10,1],max_val=1.0, filter_size=11,
                          filter_sigma=1.5, k1=0.01, k2=0.03).numpy().mean()
# Tue, 01 Mar 2022 13:25:20
from skimage.metrics import structural_similarity as ssim
ssim(mov[10,0], mov[10,1],multichannel=True)
# Tue, 01 Mar 2022 13:25:20
plt.imshow(gt[0].T)
# Tue, 01 Mar 2022 13:25:21
plt.imshow(pred[0].T)
# Tue, 01 Mar 2022 13:25:21
import tensorflow as tf
import sys
# returns a matrix with the indicies as values, similar to np.indicies
def get_idxs_tf(x):
    return tf.cast(
        tf.reshape(tf.where(tf.ones((x[0], x[1], x[2]))), (x[0], x[1], x[2], 3)),
        tf.float32)


# returns a matrix with vectors pointing to the center
def get_centers_tf(x):
    return tf.cast(
        tf.tile(tf.convert_to_tensor([x[0] // 2, x[1] // 2, x[2] // 2])[tf.newaxis, tf.newaxis, tf.newaxis, ...],
                (x[0], x[1], x[2], 1)), tf.float32)

def get_angle_tf(a, b):
    # this should work for batches of n-dimensional vectors
    # α = arccos[(a · b) / (|a| * |b|)]
    # |v| = √(x² + y² + z²)
    """
    in 3D space
    If vectors a = [xa, ya, za], b = [xb, yb, zb], then:
    α = arccos[(xa * xb + ya * yb + za * zb) / (√(xa2 + ya2 + za2) * √(xb2 + yb2 + zb2))]
    """
    import math as m
    pi = tf.constant(m.pi)
    b = tf.cast(b, dtype=a.dtype)
    inner = tf.einsum('...i,...i->...', a, b)
    norms = tf.norm(a, ord='euclidean', axis=-1) * tf.norm(b, ord='euclidean', axis=-1)  # [...,None]
    cos = inner / (norms + sys.float_info.epsilon)
    #rad = tf.math.acos(tf.clip_by_value(cos, -1.0, 1.0))
    # rad2deg conversion
    #deg = rad * (180.0/pi)
    return cos[..., tf.newaxis]

dim = [16,64,64]
idx = get_idxs_tf(dim)
c = get_centers_tf(dim)
centers = c - idx
centers_tensor = centers[tf.newaxis, ...]
flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x, centers_tensor), name='flow2direction')
minmax_lambda = lambda x : x[1] + (((x[0] - np.min(x[0])) * (x[2]-x[1]))/(np.max(x[0]) - np.min(x[0])))
# define some central params
lower, mid, upper = -1,0, 1
# Tue, 01 Mar 2022 13:25:21
# align norm an direction by the cardiac phase ED
# resample all 1D feature vectors to the same length
# this should help to validate the pre-defined rules, and if both together explains the cardiac phases
target_t = 30
norms = []
dirs= []
cycle_len=[]
gt_ind = []
gt_ind_rolled = []
xval = np.linspace(0,1,target_t)

for i,n in enumerate(nda_vects[:]):
    cardiac_cycle_length = int(gt_len[i,:,0].sum())
    cycle_len.append(cardiac_cycle_length)
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
    gt_onehot = gt[i][:cardiac_cycle_length]
    gt_ind.append(ind)
    
    dir_axis=0
    gt_ed = ind[0]
    #assert cardiac_cycle_length>=gt_ed
    temp = n[:cardiac_cycle_length]
    
    dim = temp.shape[1:-1]
    #print(dim)
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    #print('centers: ',c.dtype)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    # direction
    directions = flow2direction_lambda(temp)[...,0]
    # direction mean
    directions = np.mean(directions,axis=(1,2,3))
    # direction ed aligned
    directions = np.roll(directions, -1*gt_ed)
    # direction interpolate to unique length
    f = interp1d(np.linspace(0,1,directions.shape[0]), directions, kind='linear')
    directions = f(xval)
    # direction min/max normalised between -1,1

    directions = minmax_lambda([directions,lower,upper])
    
    # magnitude/norm as mean
    norm_full = np.linalg.norm(temp, axis=-1)
    norm_nda = norm_full.mean(axis=(1,2,3))
    # norm ed aligned
    norm_nda = np.roll(norm_nda, -1*gt_ed)
    # interpolate to unique length
    #f = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')
    #norm_nda = f(xval)
    norm_nda = np.interp(xval, np.linspace(0,1,norm_nda.shape[0]), norm_nda)
    # norm min/max aligned to 0,1
    norm_nda = minmax_lambda([norm_nda,mid,upper])
    
    # roll, scale, round and clip the gt indicies, to get an aligned distribution of the labels
    gt_onehot_rolled = np.roll(gt_onehot, -1*gt_ed, axis=0)
    resize_factor = target_t/cardiac_cycle_length
    gt_onehot_rolled = np.argmax(gt_onehot_rolled,axis=0)
    gt_onehot_rolled = np.clip(np.rint(gt_onehot_rolled * resize_factor),a_min=0,a_max=target_t-1)
    gt_ind_rolled.append(gt_onehot_rolled)
    
    norms.append(norm_nda)
    dirs.append(directions)
# Tue, 01 Mar 2022 13:25:46
# get the mean values
gt_ind_scaled = np.stack(gt_ind_rolled, axis=0)
gt_ind = np.stack(gt_ind,axis=0)
cycle_len = np.stack(cycle_len, axis=0)
norms = np.stack(norms, axis=0)
dirs = np.stack(dirs, axis=0)
norms_m = [norms.mean(axis=0)]
dirs_m=[dirs.mean(axis=0)]
# Tue, 01 Mar 2022 13:25:46
# plot the distribution of the aligned and scaled gt indicies
fig, (ax1,ax2, ax3) = plt.subplots(1,3, figsize=(15,5))
temp = pd.DataFrame(gt_ind_scaled, columns=phases)
# TODO check this outlier with MD ==2
temp = temp[temp['MD']>10]
print(temp.mean(), temp.std())
ax1 = sb.violinplot(data=temp, ax=ax1)
ax1 = sb.stripplot(data=temp, ax=ax1)
#_ = temp.plot(kind='box', ax=ax1)
_ = temp.plot(kind='line', ax=ax2, legend=None)
_ = temp.plot(kind='hist', ax=ax3, legend=None)
ax3.set_ylabel('')
# Tue, 01 Mar 2022 13:25:46
# plot one line per patient - norm
fig, ax = plt.subplots(figsize=(25,5))
for n in norms:
    _ = plt.plot(n)
_ = ax.set_title('Magnitudes aligned at ED, resampling shape 40,')
# Tue, 01 Mar 2022 13:25:47
# plot one line per direction
fig, ax = plt.subplots(figsize=(25,5))
for n in dirs:
    _ = plt.plot(n)
_ = ax.set_title('Directions aligned at ED, resampling shape 40,')
# Tue, 01 Mar 2022 13:25:47
from src.visualization.Visualize import show_2D_or_3D
from src.data.Preprocess import clip_quantile
nda_temp = nda_vects[0]
# norm_1 = nda_temp[:,:,:32,:32]# top left
norm_1 = nda_temp[:,:,:32,32:]# top right
norm_1 = np.linalg.norm(norm_1, axis=-1)
nda_norm_ = np.linalg.norm(nda_temp, axis=-1)
plt.hist(nda_norm_.flatten());plt.show()
nda_norm = clip_quantile(nda_norm_, 0.99)
nda_norm = minmax_lambda([nda_norm,0,1])
plt.hist(nda_norm.flatten())
plt.show()
# mask phi with a threshold norm matrix 
nda_msk = (nda_norm>=0.2).astype(np.float32)
nda_temp = nda_norm * nda_msk
#plt.hist(nda_temp.flatten())
print(nda_temp.shape)
_ = show_2D_or_3D(nda_norm_[:,0], allow_slicing=True)
_ = show_2D_or_3D(nda_temp[:,0], allow_slicing=True)
_ = show_2D_or_3D(nda_temp[:,-1], allow_slicing=True)

# top right
_ = show_2D_or_3D(norm_1[:,0], allow_slicing=True,cmap='gray', interpolation='none')
# Tue, 01 Mar 2022 13:25:49
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:25:50
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:25:51
fig, ax = plt.subplots(figsize=(25,3))
plt.plot(np.argmax(gt[0], axis=1))
ax.set_yticks([0, 1, 2, 3, 4], minor=False)
phases = ['ED', 'MS', 'ES', 'PF', 'MD']
_ = ax.set_yticklabels(phases)
# Tue, 01 Mar 2022 13:25:51
# plot the mean/max norm for one patient oveer time
nda_1d_max = np.max(nda_temp,axis=(1,2,3))
nda_1d_mean = np.mean(nda_temp,axis=(1,2,3))
nda_1d_sum = np.sum(nda_temp,axis=(1,2,3))

fig, ax = plt.subplots(figsize=(25,3))
ax.set_title('max norm')
_ = plt.plot(nda_1d_max); plt.show()

fig, ax = plt.subplots(figsize=(25,3))
ax.set_title('mean norm')
_ = plt.plot(nda_1d_mean)

fig, ax = plt.subplots(figsize=(25,3))
ax.set_title('mean sum')
_ = plt.plot(nda_1d_sum)
#ax.set_ylim(0.0,0.15)
# Tue, 01 Mar 2022 13:25:52
temp = np.arange(10)
temp[:0]
# Tue, 01 Mar 2022 13:25:52
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:25:53
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:25:54
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:25:55
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:25:55
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:25:56
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:25:57
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:25:58
def detect_phases(dir_1d_mean, length):
    
    import scipy.signal as sig
    
        # MS
    # Global min of f(x)
    peaks, prop = sig.find_peaks(-1*dir_1d_mean, height=0.4)
    if len(peaks)>1:
        ms = peaks[0]
    else:
        ms = np.argmin(dir_1d_mean)
    ms = ms -1 # take the bucket before the first min peak
    
    
    # ES
    # First time f(x)>0 after MS
    cycle = np.concatenate([dir_1d_mean[ms:], dir_1d_mean[:ms]])
    temp_ = 0
    es_found=False
    negative_slope = False
    for idx,elem in enumerate(cycle):
        if elem<0:
            negative_slope=True
            temp_ = idx
        elif elem>=0 and negative_slope:
            es_found = True
            #temp_ = idx
            break # stop after first zero-transition
    if es_found:
        es = ms + temp_
        #es = es-1
    else:
        es = ms + 1 # the frame after ms, fallback
    if es>=length:
        es = np.mod(es,length)
        print('ES overflow: {}, ms:{}'.format(es,ms))
     
    
    # PF
    # First peak after ES, min height 0.6
    seq = dir_1d_mean[es:]
    peaks, prop = sig.find_peaks(seq, height=0.6)#height=0.6 we normalise between -1 and 1, PF should be close to argmax

    if len(peaks>0):
        pf = es + peaks[0] # take the peak after es
        pf = pf -1
    else:
        print('pf not clear, set to ES {} + 1'.format(es))
        pf = es + 1
    pf = np.mod(pf, length)
      

    # ED
    # Between pf and ms: last time f(x) cross zero from positive to negative
    # a priori knowledge ED needs a minimal distance of 2 frames towards MS
    # CHANGED the minimal distance between ED and MS
    cycle = np.concatenate([dir_1d_mean[pf:], dir_1d_mean[:ms]])
    #print(cycle)
    ed_found = False
    last_idx_positive = True # we start at the pf, which is the peak(dir)
    for idx,elem in enumerate(cycle):
        
        if elem>=0:
            last_idx_positive = True
        # remember the last idx before the direction gets negative the last time before ms
        elif elem<0 and last_idx_positive: # first time direction negative
            ed_found = True # for fallbacks
            temp_ = idx # idx before negative direction
            #print('found transition at: {}'.format(idx))
            last_idx_positive = False # remember only the first idx after transition
        
    if ed_found:
        ed = pf + temp_
        #print('ed:{}, pf:{}, temp_:{}, lenght: {}'.format(ed,pf,temp_,length))
    else: 
        # if we dont find a transition from positive to negative, take the idx which is the closest to zero
        temp_ = np.argmin(np.abs(cycle)) # make sure we have a minimal distance
        ed = pf + temp_
        print('ED: no transition found between {}-{} , closest id to 0: {}, ed = {}'.format(pf,ms,temp_,ed))
    
    if ed>=length:
        #print('gt ed: {}, ed: {}, length: {}'.format(gted,ed,length))
        ed = np.mod(ed,length)
        #print('mod ed: {}'.format(ed))
    #ed = ed-1 # take the bucket before negative
        
    # MD
    # Middle between PF and ED
    ed_slice_idx = ed
    if ed_slice_idx<=pf: # ed overflow --> beginning of cmr stack
        ed_slice_idx = length + ed
    md = (pf+ed_slice_idx)//2 # the bucket after the middle
    md = md + 1
    md = np.mod(md,length)
    
    return np.array([ed,ms,es,pf,md])
# Tue, 01 Mar 2022 13:25:58
# try to predict the phases unsupervised just from the norm/direction curve

def get_phases_from_vects(vects_nda, length=-1, plot=True, dir_axis=0, gtind=[], figsize=(25,3)):
    import scipy.signal
    import scipy.signal as sig
    import scipy.ndimage
    from scipy.ndimage import gaussian_filter1d
    
    gted=gtind[0]
    
    # signal smoothing
    kernel_size = 4
    kernel = np.ones(kernel_size) / kernel_size
    z = 0
    
    
    dim_z = vects_nda.shape[1]
    #vects_nda = vects_nda[:int(length)]
    
    
    # define some helper lambdas
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    dim = vects_nda.shape[1:-1]
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]

    
    # norm of the vector
    norm_nda = np.linalg.norm(vects_nda[...,dir_axis:], axis=-1)
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    norm_nda_thres = clip_quantile(norm_nda, 0.9)
    norm_nda = minmax_lambda([norm_nda,0,1])
    norm_msk = norm_nda>=0.0 # this could be used to mask v_t by a threshold norm
    norm_nda = norm_nda * norm_msk
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    nda_1d_mean = np.mean(norm_nda,axis=(1,2,3))
    nda_1d_max = np.max(norm_nda,axis=(1,2,3))
    nda_1d_median = np.median(norm_nda,axis=(1,2,3))
    nda_1d_mean_smooth = np.convolve(nda_1d_mean, kernel, mode='same')
    #print(nda_1d_mean_smooth.shape, nda_1d_mean.shape)
    
    
    # direction relative to the center
    
    directions = flow2direction_lambda(vects_nda)[...,0].numpy()
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    directions = minmax_lambda([directions,lower,upper])
    #plt.hist(directions.flatten());plt.show()
    directions = directions * norm_msk
    #plt.hist(directions.flatten());plt.show()
    
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    dir_1d_mean = np.mean(directions,axis=(1,2,3))
    dir_1d_median = np.median(directions,axis=(1,2,3))

    
    # Find phases
    # 1st smooth the direction with a rolling averge (kernelsize=4)
    # or via gaussian filter
    # 2nd min/max normalise each direction vector in a range [-1:1]
    
    
    dir_1d_mean = gaussian_filter1d(dir_1d_mean, sigma=2, mode='wrap') # original: dir_1d_mean
    
    ################# CHANGED !!!!!! ######################## RECENTLY NO CHANGES
    # min/max scaling
    dir_1d_mean = minmax_lambda([dir_1d_mean,lower,upper])
    
    
#     # test with corner specific vector fields
#     y,x = directions.shape[-2:]
    
#     vects_mean = minmax_lambda([np.mean(vects_nda,axis=(1,2,3,4)),lower,upper])
#     vects_1 = vects_nda[:,:,:y//2,:x//2,:]
#     vects_1_mean = minmax_lambda([np.mean(vects_1,axis=(1,2,3,4)),lower,upper])
#     vects_2 = vects_nda[:,:,y//2:,:x//2,:]
#     vects_2_mean = minmax_lambda([np.mean(vects_2,axis=(1,2,3,4)),lower,upper])
#     vects_3 = vects_nda[:,:,y//2:,x//2:,:]
#     vects_3_mean = minmax_lambda([np.mean(vects_3,axis=(1,2,3,4)),lower,upper])
#     vects_4 = vects_nda[:,:,:y//2,x//2:,:]
#     vects_4_mean = minmax_lambda([np.mean(vects_4,axis=(1,2,3,4)),lower,upper])
    
    
#     # direction per corner
#     dir_1 = directions[:,:,:y//2,:x//2] # top left, plot
#     dir_1_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_2 = directions[:,:,y//2:,:x//2] # top right, plot
#     dir_2_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_3 = directions[:,:,y//2:,x//2:] # bottom left
#     dir_3_mean = minmax_lambda([np.mean(dir_3,axis=(1,2,3)),lower,upper])
#     dir_4 = directions[:,:,:y//2,x//2:] # bottom right
#     dir_4_mean = minmax_lambda([np.mean(dir_4,axis=(1,2,3)),lower,upper])
    

#     # norm per corner
#     norm_mean = minmax_lambda([nda_1d_mean, mid, upper])
#     norm_1 = norm_nda[:,:,:y//2,:x//2]
#     norm_1_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_2 = norm_nda[:,:,y//2:,:x//2]
#     norm_2_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_3 = norm_nda[:,:,y//2:,x//2:]
#     norm_3_mean = minmax_lambda([np.mean(norm_3,axis=(1,2,3)),mid,upper])
#     norm_4 = norm_nda[:,:,:y//2,x//2:]
#     norm_4_mean = minmax_lambda([np.mean(norm_4,axis=(1,2,3)),mid,upper])
    

    ind = detect_phases(dir_1d_mean=dir_1d_mean[:length], length=length)
    
    
    #print('ed: {}, ms: {}, es: {}, pf: {}, md: {}'.format(*ind))
    
        # plot the mean/max norm for one patient oveer time
    if plot:
        
        
        # VECT 2D+t
        vect_2d_t = vects_nda[:,z]
        fig1 = show_2D_or_3D(vect_2d_t,allow_slicing=False, f_size=(25,2))
        ax1 = fig1.get_axes()[0]
        _ = ax1.set_ylabel('phi_t \nmid-cavity')

        
        # DIR 2D+t
        dir_2d_t = directions[:,z]
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        #dir_2d_t = minmax_lambda([dir_2d_t,lower,upper])
        div_cmap = 'bwr'
        fig = show_2D_or_3D(dir_2d_t,allow_slicing=False, f_size=(25,2),cmap=div_cmap)
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('alpha_t \nmid-cavity')
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[len(ax.get_images())//2], cax=cax, orientation='horizontal')

        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # DIR 2D T x Z
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        directions_tz = directions.mean(axis=(2,3)) # original: norm()
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        _ = ax.imshow(directions_tz.T, aspect='auto', label='aasd', cmap=div_cmap, origin='lower')
        _ = ax.set_xticks(gtind, minor=False)
        ax2 = ax.twinx()
        _ = ax2.plot(dir_1d_mean, c='black',label='dir 1d+t')
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('dir z+t\napex:base')
        
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1
        
        #ax=fig.add_subplot(rows,1,pos)
        #_ = ax.plot(dir_1d_mean, c='black',label='dir 1d+t')
        #_ = ax.plot(dir_1_mean, label='top left')
        #_ = ax.plot(dir_2_mean, label='top right')
        #_ = ax.plot(dir_3_mean, label='bottom left')
        #_ = ax.plot(dir_4_mean, label='bottom right')
        #ax.legend(loc='upper left')
        #ax.set_xticks(gtind, minor=False)
        #ax.margins(0.022,0)
        #ax.label_outer()
        #pos=pos+1
        #plt.subplots_adjust(wspace=0, hspace=0.1)
        #plt.show()
        norm_cmap = 'hot'
        
        # NORM 2D + t
        norm_2d_t = norm_nda[:,z]
        norm_2d_t = minmax_lambda([norm_2d_t,mid,upper])
        fig = show_2D_or_3D(norm_2d_t,allow_slicing=False, f_size=(25,2),cmap=norm_cmap, interpolation='none')
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('norm 2d+t\nmid-cavity')
        #ax.margins(0,0)
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[0], cax=cax, orientation='horizontal')
        
        
        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # NORM 2D TxZ
        norm2d = minmax_lambda([norm_nda.mean(axis=(2,3)),mid,upper])
        _ = ax.imshow(norm2d.T, aspect='auto', origin='lower',cmap=norm_cmap, interpolation='none')
        _ = ax.set_xticks(gtind, minor=False)
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('norm z+t\napex:base')
        ax2 = ax.twinx()
        _ = ax2.plot(nda_1d_mean, c='black',label='norm 1d+t')
        #ax.spines['top'].set_visible(False)
        #_ = ax.legend(loc='upper left')
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1

        """ax=fig.add_subplot(rows,1,pos)
        _ = ax.plot(norm_mean, label='norm 1d+t')
        #_ = ax.plot(norm_1_mean, label='top left')
        #_ = ax.plot(norm_2_mean, label='top right')
        #_ = ax.plot(norm_3_mean, label='bottom left')
        #_ = ax.plot(norm_4_mean, label='bottom right')
        ax.legend(loc='upper left')
        ax.set_xticks(gtind, minor=False)
        ax.margins(0.022,0)
        ax.label_outer()
        pos = pos+1
        plt.subplots_adjust(wspace=0, hspace=0.1)"""
        #plt.tight_layout()
        plt.show()

        
    
    return ind
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
p = 2#215
gt_length = int(gt_len[p,:,0].sum())
ind = np.argmax(gt[p][:gt_length],axis=0)
get_phases_from_vects(nda_vects[p][:gt_length], length=gt_length,plot=True,dir_axis=0, gtind=ind)
# Tue, 01 Mar 2022 13:26:00
@interact
def compare_phases(i=(0,nda_vects.shape[0]-1), plot=True):
    import matplotlib as mpl
    mpl.rcParams.update(mpl.rcParamsDefault)
    figsize = (25,1)
    weight = 1
    z = 0
    
    phases = ['ED', 'MS', 'ES', 'PF', 'MD']
    cardiac_cycle_length = int(gt_len[i,:,-1].sum())
    # crop or not
    cardiac_cycle_length_ = gt[i].shape[0]-3 # plot the full length minus border for visualisation
    
    vects = nda_vects[i][:cardiac_cycle_length]
    
    
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0) # 
    ind_pred = np.argmax(pred[i][:],axis=0)
    
    temp = mov[i,:cardiac_cycle_length,z] # cardiac_cycle_length_
    fig = show_2D_or_3D(temp,allow_slicing=False)
    ax = fig.get_axes()[0]
    _ = ax.set_ylabel('CMR 2d+t)\nmid-cavity')
    plt.show()
    
    fig= plt.figure(figsize=figsize)
    ax=fig.add_subplot(111)
    ax.set_xticks(ind, minor=False)
    ax.set_yticks([0, 1, 2, 3, 4], minor=False)
    ax.set_yticklabels(phases, rotation=0)
    ind = np.array(ind)
    onehot_gt = np.zeros((ind.size, cardiac_cycle_length)) # cardiac_cycle_length_
    onehot_gt[np.arange(ind.size), ind] = weight
    #ax.imshow(gt[i][:cardiac_cycle_length].T,aspect='auto') # with uncertainity
    ax.imshow(onehot_gt,aspect='auto', cmap='bone') # gt as binary phase2time mapping
    _ = ax.set_yticklabels(phases)
    ax.set_title('gt')
    ax.margins(0,0)
    
    indices = get_phases_from_vects(vects, length=cardiac_cycle_length, plot=plot,dir_axis=0, gtind=ind, figsize=figsize)

    indices = np.array(indices)
    onehot = np.zeros((indices.size, cardiac_cycle_length)) # cardiac_cycle_length_
    onehot[np.arange(indices.size), indices] = weight
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xticks(indices, minor=False)
    ax.set_yticks([0, 1, 2, 3, 4], minor=False)
    ax.set_yticklabels(phases, rotation=0)
    plt.imshow(onehot,aspect='auto', cmap='bone')
    _ = ax.set_yticklabels(phases)
    ax.margins(0,0)
    #ax.set_title('prediction')
    plt.show()
    
    print(phases)
    print('gt:', ind)
    print('u: ', indices)
    
    
    # this would plot the supervised prediction
    #print('p: ', ind_pred)
    """fig, ax = plt.subplots(figsize=figsize)
    ax.set_xticks(ind_pred, minor=False)
    ax.set_yticks([0, 1, 2, 3, 4], minor=False)
    ax.set_yticklabels(phases, rotation=45)
    plt.imshow(pred[i][:cardiac_cycle_length].T,aspect='auto')
    _ = ax.set_yticklabels(phases);plt.show()"""
    # patient 107 gcn --> good curve for plotting
    # 200 no cut
    # 125 no cut
    # 194 moderate cut 
    # 134 strong cut-off
# Tue, 01 Mar 2022 13:26:03
pred_u = np.zeros_like(gt)
upred_ind = []
cycle_len=[]
print(pred_u.shape)
for i in range(pred_u.shape[0]):
    weight = 1
    
    cardiac_cycle_length = int(gt_len[i,:,0].sum())
    cycle_len.append(cardiac_cycle_length)
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
    #print(gt[i][:cardiac_cycle_length].T.shape)
    
    indices = get_phases_from_vects(nda_vects[i][:cardiac_cycle_length], length=cardiac_cycle_length, gtind=ind,plot=False,dir_axis=0)
    upred_ind.append(indices)
    indices = np.array(indices)
    onehot = np.zeros((indices.size, cardiac_cycle_length))
    onehot[np.arange(indices.size), indices] = weight
    pred_u[i][0:cardiac_cycle_length] = onehot.T
upred_ind=np.stack(upred_ind, axis=0)
cycle_len = np.stack(cycle_len, axis=0)
# re-create a compatible shape for the metric fn
gt_ = np.stack([gt,gt_len], axis=1)
pred_ = np.stack([pred_u,np.zeros_like(pred_u)], axis=1)

# create a dataframe for further plots
from src.utils.Metrics import meandiff
phases = ['ED', 'MS', 'ES', 'PF', 'MD']
res = meandiff(gt_, pred_, apply_sum=False, apply_average=False)
df = pd.DataFrame(res.numpy(), columns=phases)
# Tue, 01 Mar 2022 13:26:43
# get the patient idx, per label where this method fails the most
label = 4
error_thres = 10
if type(res) != np.ndarray: res.numpy()
print('label:{} \naFD>{}\nPatient idx: {}'.format(phases[label],error_thres,np.where(res[:,label]>error_thres)[0]))
print('gt length: {}'.format(cycle_len[res[:,label]>error_thres]))
print('gt indices: {}'.format(gt_ind[:,label][(res[:,label]>error_thres)]))
print('pred indices: {}'.format(upred_ind[:,label][(res[:,label]>error_thres)]))
print('aFD: {}'.format(res[:,label][res[:,label]>error_thres]))
# Tue, 01 Mar 2022 13:26:43
# TOF unsupervised center 14_41 gaus 2
sb.set_context('paper')
sb.set(font_scale = 2)
_ =df.plot(kind='box')
ax = sb.violinplot(data=df)
ax = sb.stripplot(data=df)
ax.set_ylim(0,11)
pd.options.display.float_format = "{:,.2f}".format
df_summarized = pd.concat([df.mean(axis=0),df.std(axis=0), df.median(axis=0)], axis=1)
df_summarized.columns = ['mean', 'SD', 'meadian']
print(df_summarized.mean())
print(df_summarized)

sb.set_context('paper')
sb.set(font_scale = 1.8)
fig, axes = plt.subplots(1,5,figsize=(25,4))
axes = axes.flatten()
for i,ax in enumerate(axes):
    _ = ax.scatter(upred_ind[:,i],gt_ind[:,i])
    _ = ax.axline((1, 1), slope=1)
    _ = ax.set_title(phases[i])
    _ = ax.set_xlabel('Index - pred')
    _ = ax.set_ylabel('Index - gt')
    _ = ax.set_xlim([0,35])
    _ = ax.set_ylim([0,35])
plt.tight_layout()
# Tue, 01 Mar 2022 13:26:43
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:26:44
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:26:45
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:26:46
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:26:47
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:26:48
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:26:49
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:26:50
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:26:51
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:26:52
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:26:53
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:26:54
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:26:55
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:26:56
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:27:08
# TOF unsupervised center 14_41 gaus 2
sb.set_context('paper')
sb.set(font_scale = 2)
_ =df.plot(kind='box')
ax = sb.violinplot(data=df)
ax = sb.stripplot(data=df)
ax.set_ylim(0,11)
pd.options.display.float_format = "{:,.2f}".format
df_summarized = pd.concat([df.mean(axis=0),df.std(axis=0), df.median(axis=0)], axis=1)
df_summarized.columns = ['mean', 'SD', 'meadian']
print(df_summarized.mean())
print(df_summarized)

sb.set_context('paper')
sb.set(font_scale = 1.8)
fig, axes = plt.subplots(1,5,figsize=(25,4))
axes = axes.flatten()
for i,ax in enumerate(axes):
    _ = ax.scatter(upred_ind[:,i],gt_ind[:,i])
    _ = ax.axline((1, 1), slope=1)
    _ = ax.set_title(phases[i])
    _ = ax.set_xlabel('Index - pred')
    _ = ax.set_ylabel('Index - gt')
    _ = ax.set_xlim([0,35])
    _ = ax.set_ylim([0,35])
plt.tight_layout()
# Tue, 01 Mar 2022 13:27:08
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:27:22
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:28:11
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:28:28
# load the vectors
pathstovectnpy = sorted(glob.glob(os.path.join(vects_chooser.selected, 'moved', '*vects*.npy')))
print(pathstovectnpy)
nda_vects = np.concatenate([np.load(path_) for path_ in pathstovectnpy],axis=0)
print(nda_vects.shape)
# Tue, 01 Mar 2022 13:28:30
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:28:31
# load the phase gt and pred
pred_path = os.path.join(vects_chooser.selected, 'pred')
pathtsophasenpy = sorted(glob.glob(os.path.join(pred_path, '*gtpred*.npy')))
print(pathtsophasenpy)
nda_phase = np.concatenate([np.load(path_) for path_ in pathtsophasenpy],axis=1)
print(nda_phase.shape)
gt_, pred_ = np.split(nda_phase, axis=0,indices_or_sections=2)
print(gt_.shape)
gt= gt_[0,:,0]
pred = pred_[0,:,0]
print(gt.shape)
gt_len = gt_[0,:,1]
# Tue, 01 Mar 2022 13:28:31
# load some moved examples for easier understanding of the dimensions
pathtomoved = sorted(glob.glob(os.path.join(vects_chooser.selected, 'moved','*moved*.npy')))
print(len(pathtomoved))
mov = np.concatenate([np.load(path_) for path_ in pathtomoved],axis=0)
print(mov.shape) # patient,time,spatial-z,y,x,channel, e.g.: 69,40,16,64,64,1
# Tue, 01 Mar 2022 13:28:32
# plot one moved timestep of one patient = one 3D volume 
# should go from apex to base
from src.visualization.Visualize import show_2D_or_3D
temp = mov[10,0,...,0]
_ = show_2D_or_3D(temp, cmap='gray', interpolation='none')
# Tue, 01 Mar 2022 13:28:33
import tensorflow as tf
from tensorflow.image import ssim
ssim(mov[10,0], mov[10,1],max_val=1.0, filter_size=11,
                          filter_sigma=1.5, k1=0.01, k2=0.03).numpy().mean()
# Tue, 01 Mar 2022 13:28:33
from skimage.metrics import structural_similarity as ssim
ssim(mov[10,0], mov[10,1],multichannel=True)
# Tue, 01 Mar 2022 13:28:33
plt.imshow(gt[0].T)
# Tue, 01 Mar 2022 13:28:33
plt.imshow(pred[0].T)
# Tue, 01 Mar 2022 13:28:33
import tensorflow as tf
import sys
# returns a matrix with the indicies as values, similar to np.indicies
def get_idxs_tf(x):
    return tf.cast(
        tf.reshape(tf.where(tf.ones((x[0], x[1], x[2]))), (x[0], x[1], x[2], 3)),
        tf.float32)


# returns a matrix with vectors pointing to the center
def get_centers_tf(x):
    return tf.cast(
        tf.tile(tf.convert_to_tensor([x[0] // 2, x[1] // 2, x[2] // 2])[tf.newaxis, tf.newaxis, tf.newaxis, ...],
                (x[0], x[1], x[2], 1)), tf.float32)

def get_angle_tf(a, b):
    # this should work for batches of n-dimensional vectors
    # α = arccos[(a · b) / (|a| * |b|)]
    # |v| = √(x² + y² + z²)
    """
    in 3D space
    If vectors a = [xa, ya, za], b = [xb, yb, zb], then:
    α = arccos[(xa * xb + ya * yb + za * zb) / (√(xa2 + ya2 + za2) * √(xb2 + yb2 + zb2))]
    """
    import math as m
    pi = tf.constant(m.pi)
    b = tf.cast(b, dtype=a.dtype)
    inner = tf.einsum('...i,...i->...', a, b)
    norms = tf.norm(a, ord='euclidean', axis=-1) * tf.norm(b, ord='euclidean', axis=-1)  # [...,None]
    cos = inner / (norms + sys.float_info.epsilon)
    #rad = tf.math.acos(tf.clip_by_value(cos, -1.0, 1.0))
    # rad2deg conversion
    #deg = rad * (180.0/pi)
    return cos[..., tf.newaxis]

dim = [16,64,64]
idx = get_idxs_tf(dim)
c = get_centers_tf(dim)
centers = c - idx
centers_tensor = centers[tf.newaxis, ...]
flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x, centers_tensor), name='flow2direction')
minmax_lambda = lambda x : x[1] + (((x[0] - np.min(x[0])) * (x[2]-x[1]))/(np.max(x[0]) - np.min(x[0])))
# define some central params
lower, mid, upper = -1,0, 1
# Tue, 01 Mar 2022 13:28:33
# align norm an direction by the cardiac phase ED
# resample all 1D feature vectors to the same length
# this should help to validate the pre-defined rules, and if both together explains the cardiac phases
target_t = 30
norms = []
dirs= []
cycle_len=[]
gt_ind = []
gt_ind_rolled = []
xval = np.linspace(0,1,target_t)

for i,n in enumerate(nda_vects[:]):
    cardiac_cycle_length = int(gt_len[i,:,0].sum())
    cycle_len.append(cardiac_cycle_length)
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
    gt_onehot = gt[i][:cardiac_cycle_length]
    gt_ind.append(ind)
    
    dir_axis=0
    gt_ed = ind[0]
    #assert cardiac_cycle_length>=gt_ed
    temp = n[:cardiac_cycle_length]
    
    dim = temp.shape[1:-1]
    #print(dim)
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    #print('centers: ',c.dtype)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    # direction
    directions = flow2direction_lambda(temp)[...,0]
    # direction mean
    directions = np.mean(directions,axis=(1,2,3))
    # direction ed aligned
    directions = np.roll(directions, -1*gt_ed)
    # direction interpolate to unique length
    f = interp1d(np.linspace(0,1,directions.shape[0]), directions, kind='linear')
    directions = f(xval)
    # direction min/max normalised between -1,1

    directions = minmax_lambda([directions,lower,upper])
    
    # magnitude/norm as mean
    norm_full = np.linalg.norm(temp, axis=-1)
    norm_nda = norm_full.mean(axis=(1,2,3))
    # norm ed aligned
    norm_nda = np.roll(norm_nda, -1*gt_ed)
    # interpolate to unique length
    #f = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')
    #norm_nda = f(xval)
    norm_nda = np.interp(xval, np.linspace(0,1,norm_nda.shape[0]), norm_nda)
    # norm min/max aligned to 0,1
    norm_nda = minmax_lambda([norm_nda,mid,upper])
    
    # roll, scale, round and clip the gt indicies, to get an aligned distribution of the labels
    gt_onehot_rolled = np.roll(gt_onehot, -1*gt_ed, axis=0)
    resize_factor = target_t/cardiac_cycle_length
    gt_onehot_rolled = np.argmax(gt_onehot_rolled,axis=0)
    gt_onehot_rolled = np.clip(np.rint(gt_onehot_rolled * resize_factor),a_min=0,a_max=target_t-1)
    gt_ind_rolled.append(gt_onehot_rolled)
    
    norms.append(norm_nda)
    dirs.append(directions)
# Tue, 01 Mar 2022 13:28:45
# get the mean values
gt_ind_scaled = np.stack(gt_ind_rolled, axis=0)
gt_ind = np.stack(gt_ind,axis=0)
cycle_len = np.stack(cycle_len, axis=0)
norms = np.stack(norms, axis=0)
dirs = np.stack(dirs, axis=0)
norms_m = [norms.mean(axis=0)]
dirs_m=[dirs.mean(axis=0)]
# Tue, 01 Mar 2022 13:28:45
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:28:46
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:28:47
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:28:48
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:28:49
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:28:50
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:28:51
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:28:52
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:28:53
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:28:54
# plot the distribution of the aligned and scaled gt indicies
fig, (ax1,ax2, ax3) = plt.subplots(1,3, figsize=(15,5))
temp = pd.DataFrame(gt_ind_scaled, columns=phases)
# TODO check this outlier with MD ==2
temp = temp[temp['MD']>10]
print(temp.mean(), temp.std())
ax1 = sb.violinplot(data=temp, ax=ax1)
ax1 = sb.stripplot(data=temp, ax=ax1)
#_ = temp.plot(kind='box', ax=ax1)
_ = temp.plot(kind='line', ax=ax2, legend=None)
_ = temp.plot(kind='hist', ax=ax3, legend=None)
ax3.set_ylabel('')
# Tue, 01 Mar 2022 13:28:55
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:28:56
# seaborn
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 2)
fig, ax = plt.subplots(figsize=(25,5))
ax.margins(0,0)
for n in dirs:
    _ = ax.plot(n, alpha=0.5, zorder=0)

df = pd.DataFrame(dirs).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='blue', label=r'$\alpha_t$',ci='sd',err_style='bars',zorder=2,legend=False,ax=ax)
#_ = ax.set_title('Mean direction +/- SD - aligned at ED')
_ = ax.set_xticks(np.rint(gt_ind_scaled.mean(axis=0)), minor=False)
_ = ax.set_xlabel('Time (t) - linear interpolated to 40 frames')
_ = ax.set_ylabel(r'Angle $\alpha_t$ of $\phi_t$')
_ = ax.legend(loc='upper left')
ax2 = ax.twinx()
ax2.margins(0,0)
df = pd.DataFrame(norms).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='black', label=r'|$\phi_t$|',ci='sd', linestyle='dashed',err_style='bars',zorder=2,legend=False,ax=ax2)
_ = ax2.set_ylabel(r'Norm |$\phi_t$|')

_ = ax2.legend(loc='upper right')
# Tue, 01 Mar 2022 13:28:56
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 13:28:57
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 14:08:49
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 14:09:27
# seaborn
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 3)
fig, ax = plt.subplots(figsize=(25,5))
ax.margins(0,0)
for n in dirs:
    _ = ax.plot(n, alpha=0.5, zorder=0)

df = pd.DataFrame(dirs).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='blue', label=r'$\alpha_t$',ci='sd',err_style='bars',zorder=2,legend=False,ax=ax)
#_ = ax.set_title('Mean direction +/- SD - aligned at ED')
_ = ax.set_xticks(np.rint(gt_ind_scaled.mean(axis=0)), minor=False)
_ = ax.set_xlabel('Time (t) - linear interpolated to 30 frames')
_ = ax.set_ylabel(r'Angle $\alpha_t$ of $\phi_t$')
_ = ax.legend(loc='upper left')
ax2 = ax.twinx()
ax2.margins(0,0)
df = pd.DataFrame(norms).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='black', label=r'|$\phi_t$|',ci='sd', linestyle='dashed',err_style='bars',zorder=2,legend=False,ax=ax2)
_ = ax2.set_ylabel(r'Norm |$\phi_t$|')

_ = ax2.legend(loc='upper right')
# Tue, 01 Mar 2022 14:09:28
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 14:10:02
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 14:10:19
# load the vectors
pathstovectnpy = sorted(glob.glob(os.path.join(vects_chooser.selected, 'moved', '*vects*.npy')))
print(pathstovectnpy)
nda_vects = np.concatenate([np.load(path_) for path_ in pathstovectnpy],axis=0)
print(nda_vects.shape)
# Tue, 01 Mar 2022 14:10:28
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 14:10:29
# load the phase gt and pred
pred_path = os.path.join(vects_chooser.selected, 'pred')
pathtsophasenpy = sorted(glob.glob(os.path.join(pred_path, '*gtpred*.npy')))
print(pathtsophasenpy)
nda_phase = np.concatenate([np.load(path_) for path_ in pathtsophasenpy],axis=1)
print(nda_phase.shape)
gt_, pred_ = np.split(nda_phase, axis=0,indices_or_sections=2)
print(gt_.shape)
gt= gt_[0,:,0]
pred = pred_[0,:,0]
print(gt.shape)
gt_len = gt_[0,:,1]
# Tue, 01 Mar 2022 14:10:30
# load some moved examples for easier understanding of the dimensions
pathtomoved = sorted(glob.glob(os.path.join(vects_chooser.selected, 'moved','*moved*.npy')))
print(len(pathtomoved))
mov = np.concatenate([np.load(path_) for path_ in pathtomoved],axis=0)
print(mov.shape) # patient,time,spatial-z,y,x,channel, e.g.: 69,40,16,64,64,1
# Tue, 01 Mar 2022 14:10:31
# plot one moved timestep of one patient = one 3D volume 
# should go from apex to base
from src.visualization.Visualize import show_2D_or_3D
temp = mov[10,0,...,0]
_ = show_2D_or_3D(temp, cmap='gray', interpolation='none')
# Tue, 01 Mar 2022 14:10:32
import tensorflow as tf
from tensorflow.image import ssim
ssim(mov[10,0], mov[10,1],max_val=1.0, filter_size=11,
                          filter_sigma=1.5, k1=0.01, k2=0.03).numpy().mean()
# Tue, 01 Mar 2022 14:10:32
from skimage.metrics import structural_similarity as ssim
ssim(mov[10,0], mov[10,1],multichannel=True)
# Tue, 01 Mar 2022 14:10:32
plt.imshow(gt[0].T)
# Tue, 01 Mar 2022 14:10:32
plt.imshow(pred[0].T)
# Tue, 01 Mar 2022 14:10:32
import tensorflow as tf
import sys
# returns a matrix with the indicies as values, similar to np.indicies
def get_idxs_tf(x):
    return tf.cast(
        tf.reshape(tf.where(tf.ones((x[0], x[1], x[2]))), (x[0], x[1], x[2], 3)),
        tf.float32)


# returns a matrix with vectors pointing to the center
def get_centers_tf(x):
    return tf.cast(
        tf.tile(tf.convert_to_tensor([x[0] // 2, x[1] // 2, x[2] // 2])[tf.newaxis, tf.newaxis, tf.newaxis, ...],
                (x[0], x[1], x[2], 1)), tf.float32)

def get_angle_tf(a, b):
    # this should work for batches of n-dimensional vectors
    # α = arccos[(a · b) / (|a| * |b|)]
    # |v| = √(x² + y² + z²)
    """
    in 3D space
    If vectors a = [xa, ya, za], b = [xb, yb, zb], then:
    α = arccos[(xa * xb + ya * yb + za * zb) / (√(xa2 + ya2 + za2) * √(xb2 + yb2 + zb2))]
    """
    import math as m
    pi = tf.constant(m.pi)
    b = tf.cast(b, dtype=a.dtype)
    inner = tf.einsum('...i,...i->...', a, b)
    norms = tf.norm(a, ord='euclidean', axis=-1) * tf.norm(b, ord='euclidean', axis=-1)  # [...,None]
    cos = inner / (norms + sys.float_info.epsilon)
    #rad = tf.math.acos(tf.clip_by_value(cos, -1.0, 1.0))
    # rad2deg conversion
    #deg = rad * (180.0/pi)
    return cos[..., tf.newaxis]

dim = [16,64,64]
idx = get_idxs_tf(dim)
c = get_centers_tf(dim)
centers = c - idx
centers_tensor = centers[tf.newaxis, ...]
flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x, centers_tensor), name='flow2direction')
minmax_lambda = lambda x : x[1] + (((x[0] - np.min(x[0])) * (x[2]-x[1]))/(np.max(x[0]) - np.min(x[0])))
# define some central params
lower, mid, upper = -1,0, 1
# Tue, 01 Mar 2022 14:10:32
# align norm an direction by the cardiac phase ED
# resample all 1D feature vectors to the same length
# this should help to validate the pre-defined rules, and if both together explains the cardiac phases
target_t = 30
norms = []
dirs= []
cycle_len=[]
gt_ind = []
gt_ind_rolled = []
xval = np.linspace(0,1,target_t)

for i,n in enumerate(nda_vects[:]):
    cardiac_cycle_length = int(gt_len[i,:,0].sum())
    cycle_len.append(cardiac_cycle_length)
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0)
    gt_onehot = gt[i][:cardiac_cycle_length]
    gt_ind.append(ind)
    
    dir_axis=0
    gt_ed = ind[0]
    #assert cardiac_cycle_length>=gt_ed
    temp = n[:cardiac_cycle_length]
    
    dim = temp.shape[1:-1]
    #print(dim)
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    #print('centers: ',c.dtype)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    # direction
    directions = flow2direction_lambda(temp)[...,0]
    # direction mean
    directions = np.mean(directions,axis=(1,2,3))
    # direction ed aligned
    directions = np.roll(directions, -1*gt_ed)
    # direction interpolate to unique length
    f = interp1d(np.linspace(0,1,directions.shape[0]), directions, kind='linear')
    directions = f(xval)
    # direction min/max normalised between -1,1

    directions = minmax_lambda([directions,lower,upper])
    
    # magnitude/norm as mean
    norm_full = np.linalg.norm(temp, axis=-1)
    norm_nda = norm_full.mean(axis=(1,2,3))
    # norm ed aligned
    norm_nda = np.roll(norm_nda, -1*gt_ed)
    # interpolate to unique length
    #f = interp1d(np.linspace(0,1,norm_nda.shape[0]), norm_nda, kind='linear')
    #norm_nda = f(xval)
    norm_nda = np.interp(xval, np.linspace(0,1,norm_nda.shape[0]), norm_nda)
    # norm min/max aligned to 0,1
    norm_nda = minmax_lambda([norm_nda,mid,upper])
    
    # roll, scale, round and clip the gt indicies, to get an aligned distribution of the labels
    gt_onehot_rolled = np.roll(gt_onehot, -1*gt_ed, axis=0)
    resize_factor = target_t/cardiac_cycle_length
    gt_onehot_rolled = np.argmax(gt_onehot_rolled,axis=0)
    gt_onehot_rolled = np.clip(np.rint(gt_onehot_rolled * resize_factor),a_min=0,a_max=target_t-1)
    gt_ind_rolled.append(gt_onehot_rolled)
    
    norms.append(norm_nda)
    dirs.append(directions)
# Tue, 01 Mar 2022 14:10:57
# get the mean values
gt_ind_scaled = np.stack(gt_ind_rolled, axis=0)
gt_ind = np.stack(gt_ind,axis=0)
cycle_len = np.stack(cycle_len, axis=0)
norms = np.stack(norms, axis=0)
dirs = np.stack(dirs, axis=0)
norms_m = [norms.mean(axis=0)]
dirs_m=[dirs.mean(axis=0)]
# Tue, 01 Mar 2022 14:10:57
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 14:10:58
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 14:10:59
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 14:11:00
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 14:11:01
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 14:11:02
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 14:11:04
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 14:11:05
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 14:11:06
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 14:11:07
# seaborn
import seaborn as sb
sb.set_context('paper')
sb.set(font_scale = 3)
fig, ax = plt.subplots(figsize=(25,5))
ax.margins(0,0)
for n in dirs:
    _ = ax.plot(n, alpha=0.5, zorder=0)

df = pd.DataFrame(dirs).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='blue', label=r'$\alpha_t$',ci='sd',err_style='bars',zorder=2,legend=False,ax=ax)
#_ = ax.set_title('Mean direction +/- SD - aligned at ED')
_ = ax.set_xticks(np.rint(gt_ind_scaled.mean(axis=0)), minor=False)
_ = ax.set_xlabel('Time (t) - linear interpolated to 30 frames')
_ = ax.set_ylabel(r'Angle $\alpha_t$ of $\phi_t$')
_ = ax.legend(loc='upper left')
ax2 = ax.twinx()
ax2.margins(0,0)
df = pd.DataFrame(norms).melt()
_ = sb.lineplot(x='variable', y='value',data=df, color='black', label=r'|$\phi_t$|',ci='sd', linestyle='dashed',err_style='bars',zorder=2,legend=False,ax=ax2)
_ = ax2.set_ylabel(r'Norm |$\phi_t$|')

_ = ax2.legend(loc='upper right')
# Tue, 01 Mar 2022 14:11:07
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 14:11:08
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 14:11:22
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 16:36:41
def detect_phases(dir_1d_mean, length):
    
    import scipy.signal as sig
    
        # MS
    # Global min of f(x)
    peaks, prop = sig.find_peaks(-1*dir_1d_mean, height=0.4)
    if len(peaks)>1:
        ms = peaks[0]
    else:
        ms = np.argmin(dir_1d_mean)
    ms = ms -1 # take the bucket before the first min peak
    
    
    # ES
    # First time f(x)>0 after MS
    cycle = np.concatenate([dir_1d_mean[ms:], dir_1d_mean[:ms]])
    temp_ = 0
    es_found=False
    negative_slope = False
    for idx,elem in enumerate(cycle):
        if elem<0:
            negative_slope=True
            temp_ = idx
        elif elem>=0 and negative_slope:
            es_found = True
            #temp_ = idx
            break # stop after first zero-transition
    if es_found:
        es = ms + temp_
        #es = es-1
    else:
        es = ms + 1 # the frame after ms, fallback
    if es>=length:
        es = np.mod(es,length)
        print('ES overflow: {}, ms:{}'.format(es,ms))
     
    
    # PF
    # First peak after ES, min height 0.6
    seq = dir_1d_mean[es:]
    peaks, prop = sig.find_peaks(seq, height=0.6)#height=0.6 we normalise between -1 and 1, PF should be close to argmax

    if len(peaks>0):
        pf = es + peaks[0] # take the peak after es
        pf = pf -1
    else:
        print('pf not clear, set to ES {} + 1'.format(es))
        pf = es + 1
    pf = np.mod(pf, length)
      

    # ED
    # Between pf and ms: last time f(x) cross zero from positive to negative
    # a priori knowledge ED needs a minimal distance of 2 frames towards MS
    # CHANGED the minimal distance between ED and MS
    cycle = np.concatenate([dir_1d_mean[pf:], dir_1d_mean[:ms]])
    #print(cycle)
    ed_found = False
    last_idx_positive = True # we start at the pf, which is the peak(dir)
    for idx,elem in enumerate(cycle):
        
        if elem>=0:
            last_idx_positive = True
        # remember the last idx before the direction gets negative the last time before ms
        elif elem<0 and last_idx_positive: # first time direction negative
            ed_found = True # for fallbacks
            temp_ = idx # idx before negative direction
            #print('found transition at: {}'.format(idx))
            last_idx_positive = False # remember only the first idx after transition
        
    if ed_found:
        ed = pf + temp_
        #print('ed:{}, pf:{}, temp_:{}, lenght: {}'.format(ed,pf,temp_,length))
    else: 
        # if we dont find a transition from positive to negative, take the idx which is the closest to zero
        temp_ = np.argmin(np.abs(cycle)) # make sure we have a minimal distance
        ed = pf + temp_
        print('ED: no transition found between {}-{} , closest id to 0: {}, ed = {}'.format(pf,ms,temp_,ed))
    
    if ed>=length:
        #print('gt ed: {}, ed: {}, length: {}'.format(gted,ed,length))
        ed = np.mod(ed,length)
        #print('mod ed: {}'.format(ed))
    #ed = ed-1 # take the bucket before negative
        
    # MD
    # Middle between PF and ED
    ed_slice_idx = ed
    if ed_slice_idx<=pf: # ed overflow --> beginning of cmr stack
        ed_slice_idx = length + ed
    md = (pf+ed_slice_idx)//2 # the bucket after the middle
    md = md + 1
    md = np.mod(md,length)
    
    return np.array([ed,ms,es,pf,md])
# Tue, 01 Mar 2022 16:36:41
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 16:36:42
# try to predict the phases unsupervised just from the norm/direction curve

def get_phases_from_vects(vects_nda, length=-1, plot=True, dir_axis=0, gtind=[], figsize=(25,3)):
    import scipy.signal
    import scipy.signal as sig
    import scipy.ndimage
    from scipy.ndimage import gaussian_filter1d
    
    gted=gtind[0]
    
    # signal smoothing
    kernel_size = 4
    kernel = np.ones(kernel_size) / kernel_size
    z = 0
    
    
    dim_z = vects_nda.shape[1]
    #vects_nda = vects_nda[:int(length)]
    
    
    # define some helper lambdas
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    dim = vects_nda.shape[1:-1]
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]

    
    # norm of the vector
    norm_nda = np.linalg.norm(vects_nda[...,dir_axis:], axis=-1)
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    norm_nda_thres = clip_quantile(norm_nda, 0.9)
    norm_nda = minmax_lambda([norm_nda,0,1])
    norm_msk = norm_nda>=0.0 # this could be used to mask v_t by a threshold norm
    norm_nda = norm_nda * norm_msk
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    nda_1d_mean = np.mean(norm_nda,axis=(1,2,3))
    nda_1d_max = np.max(norm_nda,axis=(1,2,3))
    nda_1d_median = np.median(norm_nda,axis=(1,2,3))
    nda_1d_mean_smooth = np.convolve(nda_1d_mean, kernel, mode='same')
    #print(nda_1d_mean_smooth.shape, nda_1d_mean.shape)
    
    
    # direction relative to the center
    
    directions = flow2direction_lambda(vects_nda)[...,0].numpy()
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    directions = minmax_lambda([directions,lower,upper])
    #plt.hist(directions.flatten());plt.show()
    directions = directions * norm_msk
    #plt.hist(directions.flatten());plt.show()
    
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    dir_1d_mean = np.mean(directions,axis=(1,2,3))
    dir_1d_median = np.median(directions,axis=(1,2,3))

    
    # Find phases
    # 1st smooth the direction with a rolling averge (kernelsize=4)
    # or via gaussian filter
    # 2nd min/max normalise each direction vector in a range [-1:1]
    
    
    dir_1d_mean = gaussian_filter1d(dir_1d_mean, sigma=2, mode='wrap') # original: dir_1d_mean
    
    ################# CHANGED !!!!!! ######################## RECENTLY NO CHANGES
    # min/max scaling
    dir_1d_mean = minmax_lambda([dir_1d_mean,lower,upper])
    
    
#     # test with corner specific vector fields
#     y,x = directions.shape[-2:]
    
#     vects_mean = minmax_lambda([np.mean(vects_nda,axis=(1,2,3,4)),lower,upper])
#     vects_1 = vects_nda[:,:,:y//2,:x//2,:]
#     vects_1_mean = minmax_lambda([np.mean(vects_1,axis=(1,2,3,4)),lower,upper])
#     vects_2 = vects_nda[:,:,y//2:,:x//2,:]
#     vects_2_mean = minmax_lambda([np.mean(vects_2,axis=(1,2,3,4)),lower,upper])
#     vects_3 = vects_nda[:,:,y//2:,x//2:,:]
#     vects_3_mean = minmax_lambda([np.mean(vects_3,axis=(1,2,3,4)),lower,upper])
#     vects_4 = vects_nda[:,:,:y//2,x//2:,:]
#     vects_4_mean = minmax_lambda([np.mean(vects_4,axis=(1,2,3,4)),lower,upper])
    
    
#     # direction per corner
#     dir_1 = directions[:,:,:y//2,:x//2] # top left, plot
#     dir_1_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_2 = directions[:,:,y//2:,:x//2] # top right, plot
#     dir_2_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_3 = directions[:,:,y//2:,x//2:] # bottom left
#     dir_3_mean = minmax_lambda([np.mean(dir_3,axis=(1,2,3)),lower,upper])
#     dir_4 = directions[:,:,:y//2,x//2:] # bottom right
#     dir_4_mean = minmax_lambda([np.mean(dir_4,axis=(1,2,3)),lower,upper])
    

#     # norm per corner
#     norm_mean = minmax_lambda([nda_1d_mean, mid, upper])
#     norm_1 = norm_nda[:,:,:y//2,:x//2]
#     norm_1_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_2 = norm_nda[:,:,y//2:,:x//2]
#     norm_2_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_3 = norm_nda[:,:,y//2:,x//2:]
#     norm_3_mean = minmax_lambda([np.mean(norm_3,axis=(1,2,3)),mid,upper])
#     norm_4 = norm_nda[:,:,:y//2,x//2:]
#     norm_4_mean = minmax_lambda([np.mean(norm_4,axis=(1,2,3)),mid,upper])
    

    ind = detect_phases(dir_1d_mean=dir_1d_mean[:length], length=length)
    
    
    #print('ed: {}, ms: {}, es: {}, pf: {}, md: {}'.format(*ind))
    
        # plot the mean/max norm for one patient oveer time
    if plot:
        
        
        # VECT 2D+t
        vect_2d_t = vects_nda[:,z]
        fig1 = show_2D_or_3D(vect_2d_t,allow_slicing=False, f_size=(25,2))
        ax1 = fig1.get_axes()[0]
        _ = ax1.set_ylabel('phi_t \nmid-cavity')

        
        # DIR 2D+t
        dir_2d_t = directions[:,z]
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        #dir_2d_t = minmax_lambda([dir_2d_t,lower,upper])
        div_cmap = 'bwr'
        fig = show_2D_or_3D(dir_2d_t,allow_slicing=False, f_size=(25,2),cmap=div_cmap)
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('alpha_t \nmid-cavity')
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[len(ax.get_images())//2], cax=cax, orientation='horizontal')

        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # DIR 2D T x Z
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        directions_tz = directions.mean(axis=(2,3)) # original: norm()
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        _ = ax.imshow(directions_tz.T, aspect='auto', label='aasd', cmap=div_cmap, origin='lower')
        _ = ax.set_xticks(gtind, minor=False)
        ax2 = ax.twinx()
        _ = ax2.plot(dir_1d_mean, c='black',label='dir 1d+t')
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('dir z+t\napex:base')
        
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1
        
        #ax=fig.add_subplot(rows,1,pos)
        #_ = ax.plot(dir_1d_mean, c='black',label='dir 1d+t')
        #_ = ax.plot(dir_1_mean, label='top left')
        #_ = ax.plot(dir_2_mean, label='top right')
        #_ = ax.plot(dir_3_mean, label='bottom left')
        #_ = ax.plot(dir_4_mean, label='bottom right')
        #ax.legend(loc='upper left')
        #ax.set_xticks(gtind, minor=False)
        #ax.margins(0.022,0)
        #ax.label_outer()
        #pos=pos+1
        #plt.subplots_adjust(wspace=0, hspace=0.1)
        #plt.show()
        norm_cmap = 'hot'
        
        # NORM 2D + t
        norm_2d_t = norm_nda[:,z]
        norm_2d_t = minmax_lambda([norm_2d_t,mid,upper])
        fig = show_2D_or_3D(norm_2d_t,allow_slicing=False, f_size=(25,2),cmap=norm_cmap, interpolation='none')
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('norm 2d+t\nmid-cavity')
        #ax.margins(0,0)
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[0], cax=cax, orientation='horizontal')
        
        
        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # NORM 2D TxZ
        norm2d = minmax_lambda([norm_nda.mean(axis=(2,3)),mid,upper])
        _ = ax.imshow(norm2d.T, aspect='auto', origin='lower',cmap=norm_cmap, interpolation='none')
        _ = ax.set_xticks(gtind, minor=False)
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('norm z+t\napex:base')
        ax2 = ax.twinx()
        _ = ax2.plot(nda_1d_mean, c='black',label='norm 1d+t')
        #ax.spines['top'].set_visible(False)
        #_ = ax.legend(loc='upper left')
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1

        """ax=fig.add_subplot(rows,1,pos)
        _ = ax.plot(norm_mean, label='norm 1d+t')
        #_ = ax.plot(norm_1_mean, label='top left')
        #_ = ax.plot(norm_2_mean, label='top right')
        #_ = ax.plot(norm_3_mean, label='bottom left')
        #_ = ax.plot(norm_4_mean, label='bottom right')
        ax.legend(loc='upper left')
        ax.set_xticks(gtind, minor=False)
        ax.margins(0.022,0)
        ax.label_outer()
        pos = pos+1
        plt.subplots_adjust(wspace=0, hspace=0.1)"""
        #plt.tight_layout()
        plt.show()

        
    
    return ind
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
p = 2#215
gt_length = int(gt_len[p,:,0].sum())
ind = np.argmax(gt[p][:gt_length],axis=0)
get_phases_from_vects(nda_vects[p][:gt_length], length=gt_length,plot=True,dir_axis=0, gtind=ind)
# Tue, 01 Mar 2022 16:36:44
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 16:36:45
@interact
def compare_phases(i=(0,nda_vects.shape[0]-1), plot=True):
    import matplotlib as mpl
    mpl.rcParams.update(mpl.rcParamsDefault)
    figsize = (25,1)
    weight = 1
    z = 0
    
    phases = ['ED', 'MS', 'ES', 'PF', 'MD']
    cardiac_cycle_length = int(gt_len[i,:,-1].sum())
    # crop or not
    cardiac_cycle_length_ = gt[i].shape[0]-3 # plot the full length minus border for visualisation
    
    vects = nda_vects[i][:cardiac_cycle_length]
    
    
    ind = np.argmax(gt[i][:cardiac_cycle_length],axis=0) # 
    ind_pred = np.argmax(pred[i][:],axis=0)
    
    temp = mov[i,:cardiac_cycle_length,z] # cardiac_cycle_length_
    fig = show_2D_or_3D(temp,allow_slicing=False)
    ax = fig.get_axes()[0]
    _ = ax.set_ylabel('CMR 2d+t)\nmid-cavity')
    plt.show()
    
    fig= plt.figure(figsize=figsize)
    ax=fig.add_subplot(111)
    ax.set_xticks(ind, minor=False)
    ax.set_yticks([0, 1, 2, 3, 4], minor=False)
    ax.set_yticklabels(phases, rotation=0)
    ind = np.array(ind)
    onehot_gt = np.zeros((ind.size, cardiac_cycle_length)) # cardiac_cycle_length_
    onehot_gt[np.arange(ind.size), ind] = weight
    #ax.imshow(gt[i][:cardiac_cycle_length].T,aspect='auto') # with uncertainity
    ax.imshow(onehot_gt,aspect='auto', cmap='bone') # gt as binary phase2time mapping
    _ = ax.set_yticklabels(phases)
    ax.set_title('gt')
    ax.margins(0,0)
    
    indices = get_phases_from_vects(vects, length=cardiac_cycle_length, plot=plot,dir_axis=0, gtind=ind, figsize=figsize)

    indices = np.array(indices)
    onehot = np.zeros((indices.size, cardiac_cycle_length)) # cardiac_cycle_length_
    onehot[np.arange(indices.size), indices] = weight
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xticks(indices, minor=False)
    ax.set_yticks([0, 1, 2, 3, 4], minor=False)
    ax.set_yticklabels(phases, rotation=0)
    plt.imshow(onehot,aspect='auto', cmap='bone')
    _ = ax.set_yticklabels(phases)
    ax.margins(0,0)
    #ax.set_title('prediction')
    plt.show()
    
    print(phases)
    print('gt:', ind)
    print('u: ', indices)
    
    
    # this would plot the supervised prediction
    #print('p: ', ind_pred)
    """fig, ax = plt.subplots(figsize=figsize)
    ax.set_xticks(ind_pred, minor=False)
    ax.set_yticks([0, 1, 2, 3, 4], minor=False)
    ax.set_yticklabels(phases, rotation=45)
    plt.imshow(pred[i][:cardiac_cycle_length].T,aspect='auto')
    _ = ax.set_yticklabels(phases);plt.show()"""
    # patient 107 gcn --> good curve for plotting
    # 200 no cut
    # 125 no cut
    # 194 moderate cut 
    # 134 strong cut-off
# Tue, 01 Mar 2022 16:36:48
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 16:46:12
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 16:48:00
# try to predict the phases unsupervised just from the norm/direction curve

def get_phases_from_vects(vects_nda, length=-1, plot=True, dir_axis=0, gtind=[], figsize=(25,3)):
    import scipy.signal
    import scipy.signal as sig
    import scipy.ndimage
    from scipy.ndimage import gaussian_filter1d
    
    gted=gtind[0]
    
    # signal smoothing
    kernel_size = 4
    kernel = np.ones(kernel_size) / kernel_size
    z = 0
    
    
    dim_z = vects_nda.shape[1]
    #vects_nda = vects_nda[:int(length)]
    
    
    # define some helper lambdas
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    dim = vects_nda.shape[1:-1]
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]

    
    # norm of the vector
    norm_nda = np.linalg.norm(vects_nda[...,dir_axis:], axis=-1)
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    norm_nda_thres = clip_quantile(norm_nda, 0.9)
    norm_nda = minmax_lambda([norm_nda,0,1])
    norm_msk = norm_nda>=0.0 # this could be used to mask v_t by a threshold norm
    norm_nda = norm_nda * norm_msk
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    nda_1d_mean = np.mean(norm_nda,axis=(1,2,3))
    nda_1d_max = np.max(norm_nda,axis=(1,2,3))
    nda_1d_median = np.median(norm_nda,axis=(1,2,3))
    nda_1d_mean_smooth = np.convolve(nda_1d_mean, kernel, mode='same')
    #print(nda_1d_mean_smooth.shape, nda_1d_mean.shape)
    
    
    # direction relative to the center
    
    directions = flow2direction_lambda(vects_nda)[...,0].numpy()
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    directions = minmax_lambda([directions,lower,upper])
    #plt.hist(directions.flatten());plt.show()
    directions = directions * norm_msk
    #plt.hist(directions.flatten());plt.show()
    
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    dir_1d_mean = np.mean(directions,axis=(1,2,3))
    dir_1d_median = np.median(directions,axis=(1,2,3))

    
    # Find phases
    # 1st smooth the direction with a rolling averge (kernelsize=4)
    # or via gaussian filter
    # 2nd min/max normalise each direction vector in a range [-1:1]
    
    
    dir_1d_mean = gaussian_filter1d(dir_1d_mean, sigma=2, mode='wrap') # original: dir_1d_mean
    
    ################# CHANGED !!!!!! ######################## RECENTLY NO CHANGES
    # min/max scaling
    dir_1d_mean = minmax_lambda([dir_1d_mean,lower,upper])
    
    
#     # test with corner specific vector fields
#     y,x = directions.shape[-2:]
    
#     vects_mean = minmax_lambda([np.mean(vects_nda,axis=(1,2,3,4)),lower,upper])
#     vects_1 = vects_nda[:,:,:y//2,:x//2,:]
#     vects_1_mean = minmax_lambda([np.mean(vects_1,axis=(1,2,3,4)),lower,upper])
#     vects_2 = vects_nda[:,:,y//2:,:x//2,:]
#     vects_2_mean = minmax_lambda([np.mean(vects_2,axis=(1,2,3,4)),lower,upper])
#     vects_3 = vects_nda[:,:,y//2:,x//2:,:]
#     vects_3_mean = minmax_lambda([np.mean(vects_3,axis=(1,2,3,4)),lower,upper])
#     vects_4 = vects_nda[:,:,:y//2,x//2:,:]
#     vects_4_mean = minmax_lambda([np.mean(vects_4,axis=(1,2,3,4)),lower,upper])
    
    
#     # direction per corner
#     dir_1 = directions[:,:,:y//2,:x//2] # top left, plot
#     dir_1_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_2 = directions[:,:,y//2:,:x//2] # top right, plot
#     dir_2_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_3 = directions[:,:,y//2:,x//2:] # bottom left
#     dir_3_mean = minmax_lambda([np.mean(dir_3,axis=(1,2,3)),lower,upper])
#     dir_4 = directions[:,:,:y//2,x//2:] # bottom right
#     dir_4_mean = minmax_lambda([np.mean(dir_4,axis=(1,2,3)),lower,upper])
    

#     # norm per corner
#     norm_mean = minmax_lambda([nda_1d_mean, mid, upper])
#     norm_1 = norm_nda[:,:,:y//2,:x//2]
#     norm_1_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_2 = norm_nda[:,:,y//2:,:x//2]
#     norm_2_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_3 = norm_nda[:,:,y//2:,x//2:]
#     norm_3_mean = minmax_lambda([np.mean(norm_3,axis=(1,2,3)),mid,upper])
#     norm_4 = norm_nda[:,:,:y//2,x//2:]
#     norm_4_mean = minmax_lambda([np.mean(norm_4,axis=(1,2,3)),mid,upper])
    

    ind = detect_phases(dir_1d_mean=dir_1d_mean[:length], length=length)
    
    
    #print('ed: {}, ms: {}, es: {}, pf: {}, md: {}'.format(*ind))
    
        # plot the mean/max norm for one patient oveer time
    if plot:
        
        
        # VECT 2D+t
        vect_2d_t = vects_nda[:,z]
        fig1 = show_2D_or_3D(vect_2d_t,allow_slicing=False, f_size=(25,2))
        ax1 = fig1.get_axes()[0]
        _ = ax1.set_ylabel('phi_t \nmid-cavity')

        
        # DIR 2D+t
        dir_2d_t = directions[:,z]
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        #dir_2d_t = minmax_lambda([dir_2d_t,lower,upper])
        div_cmap = 'bwr'
        fig = show_2D_or_3D(dir_2d_t,allow_slicing=False, f_size=(25,2),cmap=div_cmap)
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('phi_t \nmid-cavity') #'alpha_t \nmid-cavity'
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[len(ax.get_images())//2], cax=cax, orientation='horizontal')

        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # DIR 2D T x Z
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        directions_tz = directions.mean(axis=(2,3)) # original: norm()
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        _ = ax.imshow(directions_tz.T, aspect='auto', label='aasd', cmap=div_cmap, origin='lower')
        _ = ax.set_xticks(gtind, minor=False)
        ax2 = ax.twinx()
        _ = ax2.plot(dir_1d_mean, c='black',label='dir 1d+t')
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('dir z+t\napex:base')
        
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1
        
        #ax=fig.add_subplot(rows,1,pos)
        #_ = ax.plot(dir_1d_mean, c='black',label='dir 1d+t')
        #_ = ax.plot(dir_1_mean, label='top left')
        #_ = ax.plot(dir_2_mean, label='top right')
        #_ = ax.plot(dir_3_mean, label='bottom left')
        #_ = ax.plot(dir_4_mean, label='bottom right')
        #ax.legend(loc='upper left')
        #ax.set_xticks(gtind, minor=False)
        #ax.margins(0.022,0)
        #ax.label_outer()
        #pos=pos+1
        #plt.subplots_adjust(wspace=0, hspace=0.1)
        #plt.show()
        norm_cmap = 'hot'
        
        # NORM 2D + t
        norm_2d_t = norm_nda[:,z]
        norm_2d_t = minmax_lambda([norm_2d_t,mid,upper])
        fig = show_2D_or_3D(norm_2d_t,allow_slicing=False, f_size=(25,2),cmap=norm_cmap, interpolation='none')
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('norm 2d+t\nmid-cavity')
        #ax.margins(0,0)
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[0], cax=cax, orientation='horizontal')
        
        
        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # NORM 2D TxZ
        norm2d = minmax_lambda([norm_nda.mean(axis=(2,3)),mid,upper])
        _ = ax.imshow(norm2d.T, aspect='auto', origin='lower',cmap=norm_cmap, interpolation='none')
        _ = ax.set_xticks(gtind, minor=False)
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('norm z+t\napex:base')
        ax2 = ax.twinx()
        _ = ax2.plot(nda_1d_mean, c='black',label='norm 1d+t')
        #ax.spines['top'].set_visible(False)
        #_ = ax.legend(loc='upper left')
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1

        """ax=fig.add_subplot(rows,1,pos)
        _ = ax.plot(norm_mean, label='norm 1d+t')
        #_ = ax.plot(norm_1_mean, label='top left')
        #_ = ax.plot(norm_2_mean, label='top right')
        #_ = ax.plot(norm_3_mean, label='bottom left')
        #_ = ax.plot(norm_4_mean, label='bottom right')
        ax.legend(loc='upper left')
        ax.set_xticks(gtind, minor=False)
        ax.margins(0.022,0)
        ax.label_outer()
        pos = pos+1
        plt.subplots_adjust(wspace=0, hspace=0.1)"""
        #plt.tight_layout()
        plt.show()

        
    
    return ind
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
p = 2#215
gt_length = int(gt_len[p,:,0].sum())
ind = np.argmax(gt[p][:gt_length],axis=0)
get_phases_from_vects(nda_vects[p][:gt_length], length=gt_length,plot=True,dir_axis=0, gtind=ind)
# Tue, 01 Mar 2022 16:48:02
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 16:49:04
# try to predict the phases unsupervised just from the norm/direction curve

def get_phases_from_vects(vects_nda, length=-1, plot=True, dir_axis=0, gtind=[], figsize=(25,3)):
    import scipy.signal
    import scipy.signal as sig
    import scipy.ndimage
    from scipy.ndimage import gaussian_filter1d
    
    gted=gtind[0]
    
    # signal smoothing
    kernel_size = 4
    kernel = np.ones(kernel_size) / kernel_size
    z = 0
    
    
    dim_z = vects_nda.shape[1]
    #vects_nda = vects_nda[:int(length)]
    
    
    # define some helper lambdas
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    dim = vects_nda.shape[1:-1]
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]

    
    # norm of the vector
    norm_nda = np.linalg.norm(vects_nda[...,dir_axis:], axis=-1)
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    norm_nda_thres = clip_quantile(norm_nda, 0.9)
    norm_nda = minmax_lambda([norm_nda,0,1])
    norm_msk = norm_nda>=0.0 # this could be used to mask v_t by a threshold norm
    norm_nda = norm_nda * norm_msk
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    nda_1d_mean = np.mean(norm_nda,axis=(1,2,3))
    nda_1d_max = np.max(norm_nda,axis=(1,2,3))
    nda_1d_median = np.median(norm_nda,axis=(1,2,3))
    nda_1d_mean_smooth = np.convolve(nda_1d_mean, kernel, mode='same')
    #print(nda_1d_mean_smooth.shape, nda_1d_mean.shape)
    
    
    # direction relative to the center
    
    directions = flow2direction_lambda(vects_nda)[...,0].numpy()
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    directions = minmax_lambda([directions,lower,upper])
    #plt.hist(directions.flatten());plt.show()
    directions = directions * norm_msk
    #plt.hist(directions.flatten());plt.show()
    
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    dir_1d_mean = np.mean(directions,axis=(1,2,3))
    dir_1d_median = np.median(directions,axis=(1,2,3))

    
    # Find phases
    # 1st smooth the direction with a rolling averge (kernelsize=4)
    # or via gaussian filter
    # 2nd min/max normalise each direction vector in a range [-1:1]
    
    
    dir_1d_mean = gaussian_filter1d(dir_1d_mean, sigma=2, mode='wrap') # original: dir_1d_mean
    
    ################# CHANGED !!!!!! ######################## RECENTLY NO CHANGES
    # min/max scaling
    dir_1d_mean = minmax_lambda([dir_1d_mean,lower,upper])
    
    
#     # test with corner specific vector fields
#     y,x = directions.shape[-2:]
    
#     vects_mean = minmax_lambda([np.mean(vects_nda,axis=(1,2,3,4)),lower,upper])
#     vects_1 = vects_nda[:,:,:y//2,:x//2,:]
#     vects_1_mean = minmax_lambda([np.mean(vects_1,axis=(1,2,3,4)),lower,upper])
#     vects_2 = vects_nda[:,:,y//2:,:x//2,:]
#     vects_2_mean = minmax_lambda([np.mean(vects_2,axis=(1,2,3,4)),lower,upper])
#     vects_3 = vects_nda[:,:,y//2:,x//2:,:]
#     vects_3_mean = minmax_lambda([np.mean(vects_3,axis=(1,2,3,4)),lower,upper])
#     vects_4 = vects_nda[:,:,:y//2,x//2:,:]
#     vects_4_mean = minmax_lambda([np.mean(vects_4,axis=(1,2,3,4)),lower,upper])
    
    
#     # direction per corner
#     dir_1 = directions[:,:,:y//2,:x//2] # top left, plot
#     dir_1_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_2 = directions[:,:,y//2:,:x//2] # top right, plot
#     dir_2_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_3 = directions[:,:,y//2:,x//2:] # bottom left
#     dir_3_mean = minmax_lambda([np.mean(dir_3,axis=(1,2,3)),lower,upper])
#     dir_4 = directions[:,:,:y//2,x//2:] # bottom right
#     dir_4_mean = minmax_lambda([np.mean(dir_4,axis=(1,2,3)),lower,upper])
    

#     # norm per corner
#     norm_mean = minmax_lambda([nda_1d_mean, mid, upper])
#     norm_1 = norm_nda[:,:,:y//2,:x//2]
#     norm_1_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_2 = norm_nda[:,:,y//2:,:x//2]
#     norm_2_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_3 = norm_nda[:,:,y//2:,x//2:]
#     norm_3_mean = minmax_lambda([np.mean(norm_3,axis=(1,2,3)),mid,upper])
#     norm_4 = norm_nda[:,:,:y//2,x//2:]
#     norm_4_mean = minmax_lambda([np.mean(norm_4,axis=(1,2,3)),mid,upper])
    

    ind = detect_phases(dir_1d_mean=dir_1d_mean[:length], length=length)
    
    
    #print('ed: {}, ms: {}, es: {}, pf: {}, md: {}'.format(*ind))
    
        # plot the mean/max norm for one patient oveer time
    if plot:
        
        
        # VECT 2D+t
        vect_2d_t = vects_nda[:,z]
        fig1 = show_2D_or_3D(vect_2d_t,allow_slicing=False, f_size=(25,2))
        ax1 = fig1.get_axes()[0]
        _ = ax1.set_ylabel('phi_t \nmid-cavity')

        
        # DIR 2D+t
        dir_2d_t = directions[:,z]
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        #dir_2d_t = minmax_lambda([dir_2d_t,lower,upper])
        div_cmap = 'bwr'
        fig = show_2D_or_3D(dir_2d_t,allow_slicing=False, f_size=(25,2),cmap=div_cmap)
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel(r'$phi_t$ \nmid-cavity') #'alpha_t \nmid-cavity'
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[len(ax.get_images())//2], cax=cax, orientation='horizontal')

        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # DIR 2D T x Z
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        directions_tz = directions.mean(axis=(2,3)) # original: norm()
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        _ = ax.imshow(directions_tz.T, aspect='auto', label='aasd', cmap=div_cmap, origin='lower')
        _ = ax.set_xticks(gtind, minor=False)
        ax2 = ax.twinx()
        _ = ax2.plot(dir_1d_mean, c='black',label='dir 1d+t')
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('dir z+t\napex:base')
        
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1
        
        #ax=fig.add_subplot(rows,1,pos)
        #_ = ax.plot(dir_1d_mean, c='black',label='dir 1d+t')
        #_ = ax.plot(dir_1_mean, label='top left')
        #_ = ax.plot(dir_2_mean, label='top right')
        #_ = ax.plot(dir_3_mean, label='bottom left')
        #_ = ax.plot(dir_4_mean, label='bottom right')
        #ax.legend(loc='upper left')
        #ax.set_xticks(gtind, minor=False)
        #ax.margins(0.022,0)
        #ax.label_outer()
        #pos=pos+1
        #plt.subplots_adjust(wspace=0, hspace=0.1)
        #plt.show()
        norm_cmap = 'hot'
        
        # NORM 2D + t
        norm_2d_t = norm_nda[:,z]
        norm_2d_t = minmax_lambda([norm_2d_t,mid,upper])
        fig = show_2D_or_3D(norm_2d_t,allow_slicing=False, f_size=(25,2),cmap=norm_cmap, interpolation='none')
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('norm 2d+t\nmid-cavity')
        #ax.margins(0,0)
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[0], cax=cax, orientation='horizontal')
        
        
        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # NORM 2D TxZ
        norm2d = minmax_lambda([norm_nda.mean(axis=(2,3)),mid,upper])
        _ = ax.imshow(norm2d.T, aspect='auto', origin='lower',cmap=norm_cmap, interpolation='none')
        _ = ax.set_xticks(gtind, minor=False)
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('norm z+t\napex:base')
        ax2 = ax.twinx()
        _ = ax2.plot(nda_1d_mean, c='black',label='norm 1d+t')
        #ax.spines['top'].set_visible(False)
        #_ = ax.legend(loc='upper left')
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1

        """ax=fig.add_subplot(rows,1,pos)
        _ = ax.plot(norm_mean, label='norm 1d+t')
        #_ = ax.plot(norm_1_mean, label='top left')
        #_ = ax.plot(norm_2_mean, label='top right')
        #_ = ax.plot(norm_3_mean, label='bottom left')
        #_ = ax.plot(norm_4_mean, label='bottom right')
        ax.legend(loc='upper left')
        ax.set_xticks(gtind, minor=False)
        ax.margins(0.022,0)
        ax.label_outer()
        pos = pos+1
        plt.subplots_adjust(wspace=0, hspace=0.1)"""
        #plt.tight_layout()
        plt.show()

        
    
    return ind
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
p = 2#215
gt_length = int(gt_len[p,:,0].sum())
ind = np.argmax(gt[p][:gt_length],axis=0)
get_phases_from_vects(nda_vects[p][:gt_length], length=gt_length,plot=True,dir_axis=0, gtind=ind)
# Tue, 01 Mar 2022 16:49:06
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 16:49:37
# try to predict the phases unsupervised just from the norm/direction curve

def get_phases_from_vects(vects_nda, length=-1, plot=True, dir_axis=0, gtind=[], figsize=(25,3)):
    import scipy.signal
    import scipy.signal as sig
    import scipy.ndimage
    from scipy.ndimage import gaussian_filter1d
    
    gted=gtind[0]
    
    # signal smoothing
    kernel_size = 4
    kernel = np.ones(kernel_size) / kernel_size
    z = 0
    
    
    dim_z = vects_nda.shape[1]
    #vects_nda = vects_nda[:int(length)]
    
    
    # define some helper lambdas
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    dim = vects_nda.shape[1:-1]
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]

    
    # norm of the vector
    norm_nda = np.linalg.norm(vects_nda[...,dir_axis:], axis=-1)
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    norm_nda_thres = clip_quantile(norm_nda, 0.9)
    norm_nda = minmax_lambda([norm_nda,0,1])
    norm_msk = norm_nda>=0.0 # this could be used to mask v_t by a threshold norm
    norm_nda = norm_nda * norm_msk
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    nda_1d_mean = np.mean(norm_nda,axis=(1,2,3))
    nda_1d_max = np.max(norm_nda,axis=(1,2,3))
    nda_1d_median = np.median(norm_nda,axis=(1,2,3))
    nda_1d_mean_smooth = np.convolve(nda_1d_mean, kernel, mode='same')
    #print(nda_1d_mean_smooth.shape, nda_1d_mean.shape)
    
    
    # direction relative to the center
    
    directions = flow2direction_lambda(vects_nda)[...,0].numpy()
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    directions = minmax_lambda([directions,lower,upper])
    #plt.hist(directions.flatten());plt.show()
    directions = directions * norm_msk
    #plt.hist(directions.flatten());plt.show()
    
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    dir_1d_mean = np.mean(directions,axis=(1,2,3))
    dir_1d_median = np.median(directions,axis=(1,2,3))

    
    # Find phases
    # 1st smooth the direction with a rolling averge (kernelsize=4)
    # or via gaussian filter
    # 2nd min/max normalise each direction vector in a range [-1:1]
    
    
    dir_1d_mean = gaussian_filter1d(dir_1d_mean, sigma=2, mode='wrap') # original: dir_1d_mean
    
    ################# CHANGED !!!!!! ######################## RECENTLY NO CHANGES
    # min/max scaling
    dir_1d_mean = minmax_lambda([dir_1d_mean,lower,upper])
    
    
#     # test with corner specific vector fields
#     y,x = directions.shape[-2:]
    
#     vects_mean = minmax_lambda([np.mean(vects_nda,axis=(1,2,3,4)),lower,upper])
#     vects_1 = vects_nda[:,:,:y//2,:x//2,:]
#     vects_1_mean = minmax_lambda([np.mean(vects_1,axis=(1,2,3,4)),lower,upper])
#     vects_2 = vects_nda[:,:,y//2:,:x//2,:]
#     vects_2_mean = minmax_lambda([np.mean(vects_2,axis=(1,2,3,4)),lower,upper])
#     vects_3 = vects_nda[:,:,y//2:,x//2:,:]
#     vects_3_mean = minmax_lambda([np.mean(vects_3,axis=(1,2,3,4)),lower,upper])
#     vects_4 = vects_nda[:,:,:y//2,x//2:,:]
#     vects_4_mean = minmax_lambda([np.mean(vects_4,axis=(1,2,3,4)),lower,upper])
    
    
#     # direction per corner
#     dir_1 = directions[:,:,:y//2,:x//2] # top left, plot
#     dir_1_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_2 = directions[:,:,y//2:,:x//2] # top right, plot
#     dir_2_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_3 = directions[:,:,y//2:,x//2:] # bottom left
#     dir_3_mean = minmax_lambda([np.mean(dir_3,axis=(1,2,3)),lower,upper])
#     dir_4 = directions[:,:,:y//2,x//2:] # bottom right
#     dir_4_mean = minmax_lambda([np.mean(dir_4,axis=(1,2,3)),lower,upper])
    

#     # norm per corner
#     norm_mean = minmax_lambda([nda_1d_mean, mid, upper])
#     norm_1 = norm_nda[:,:,:y//2,:x//2]
#     norm_1_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_2 = norm_nda[:,:,y//2:,:x//2]
#     norm_2_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_3 = norm_nda[:,:,y//2:,x//2:]
#     norm_3_mean = minmax_lambda([np.mean(norm_3,axis=(1,2,3)),mid,upper])
#     norm_4 = norm_nda[:,:,:y//2,x//2:]
#     norm_4_mean = minmax_lambda([np.mean(norm_4,axis=(1,2,3)),mid,upper])
    

    ind = detect_phases(dir_1d_mean=dir_1d_mean[:length], length=length)
    
    
    #print('ed: {}, ms: {}, es: {}, pf: {}, md: {}'.format(*ind))
    
        # plot the mean/max norm for one patient oveer time
    if plot:
        
        
        # VECT 2D+t
        vect_2d_t = vects_nda[:,z]
        fig1 = show_2D_or_3D(vect_2d_t,allow_slicing=False, f_size=(25,2))
        ax1 = fig1.get_axes()[0]
        _ = ax1.set_ylabel('phi_t \nmid-cavity')

        
        # DIR 2D+t
        dir_2d_t = directions[:,z]
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        #dir_2d_t = minmax_lambda([dir_2d_t,lower,upper])
        div_cmap = 'bwr'
        fig = show_2D_or_3D(dir_2d_t,allow_slicing=False, f_size=(25,2),cmap=div_cmap)
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel(r'$phi_t$ \n mid-cavity') #'alpha_t \nmid-cavity'
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[len(ax.get_images())//2], cax=cax, orientation='horizontal')

        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # DIR 2D T x Z
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        directions_tz = directions.mean(axis=(2,3)) # original: norm()
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        _ = ax.imshow(directions_tz.T, aspect='auto', label='aasd', cmap=div_cmap, origin='lower')
        _ = ax.set_xticks(gtind, minor=False)
        ax2 = ax.twinx()
        _ = ax2.plot(dir_1d_mean, c='black',label='dir 1d+t')
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('dir z+t\napex:base')
        
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1
        
        #ax=fig.add_subplot(rows,1,pos)
        #_ = ax.plot(dir_1d_mean, c='black',label='dir 1d+t')
        #_ = ax.plot(dir_1_mean, label='top left')
        #_ = ax.plot(dir_2_mean, label='top right')
        #_ = ax.plot(dir_3_mean, label='bottom left')
        #_ = ax.plot(dir_4_mean, label='bottom right')
        #ax.legend(loc='upper left')
        #ax.set_xticks(gtind, minor=False)
        #ax.margins(0.022,0)
        #ax.label_outer()
        #pos=pos+1
        #plt.subplots_adjust(wspace=0, hspace=0.1)
        #plt.show()
        norm_cmap = 'hot'
        
        # NORM 2D + t
        norm_2d_t = norm_nda[:,z]
        norm_2d_t = minmax_lambda([norm_2d_t,mid,upper])
        fig = show_2D_or_3D(norm_2d_t,allow_slicing=False, f_size=(25,2),cmap=norm_cmap, interpolation='none')
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('norm 2d+t\nmid-cavity')
        #ax.margins(0,0)
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[0], cax=cax, orientation='horizontal')
        
        
        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # NORM 2D TxZ
        norm2d = minmax_lambda([norm_nda.mean(axis=(2,3)),mid,upper])
        _ = ax.imshow(norm2d.T, aspect='auto', origin='lower',cmap=norm_cmap, interpolation='none')
        _ = ax.set_xticks(gtind, minor=False)
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('norm z+t\napex:base')
        ax2 = ax.twinx()
        _ = ax2.plot(nda_1d_mean, c='black',label='norm 1d+t')
        #ax.spines['top'].set_visible(False)
        #_ = ax.legend(loc='upper left')
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1

        """ax=fig.add_subplot(rows,1,pos)
        _ = ax.plot(norm_mean, label='norm 1d+t')
        #_ = ax.plot(norm_1_mean, label='top left')
        #_ = ax.plot(norm_2_mean, label='top right')
        #_ = ax.plot(norm_3_mean, label='bottom left')
        #_ = ax.plot(norm_4_mean, label='bottom right')
        ax.legend(loc='upper left')
        ax.set_xticks(gtind, minor=False)
        ax.margins(0.022,0)
        ax.label_outer()
        pos = pos+1
        plt.subplots_adjust(wspace=0, hspace=0.1)"""
        #plt.tight_layout()
        plt.show()

        
    
    return ind
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
p = 2#215
gt_length = int(gt_len[p,:,0].sum())
ind = np.argmax(gt[p][:gt_length],axis=0)
get_phases_from_vects(nda_vects[p][:gt_length], length=gt_length,plot=True,dir_axis=0, gtind=ind)
# Tue, 01 Mar 2022 16:49:40
_jupyterlab_variableinspector_dict_list()
# Tue, 01 Mar 2022 16:50:48
# try to predict the phases unsupervised just from the norm/direction curve

def get_phases_from_vects(vects_nda, length=-1, plot=True, dir_axis=0, gtind=[], figsize=(25,3)):
    import scipy.signal
    import scipy.signal as sig
    import scipy.ndimage
    from scipy.ndimage import gaussian_filter1d
    
    gted=gtind[0]
    
    # signal smoothing
    kernel_size = 4
    kernel = np.ones(kernel_size) / kernel_size
    z = 0
    
    
    dim_z = vects_nda.shape[1]
    #vects_nda = vects_nda[:int(length)]
    
    
    # define some helper lambdas
    flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[...,dir_axis:], centers_tensor[...,dir_axis:]), name='flow2direction')
    
    dim = vects_nda.shape[1:-1]
    idx = get_idxs_tf(dim)
    c = get_centers_tf(dim)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]

    
    # norm of the vector
    norm_nda = np.linalg.norm(vects_nda[...,dir_axis:], axis=-1)
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    norm_nda_thres = clip_quantile(norm_nda, 0.9)
    norm_nda = minmax_lambda([norm_nda,0,1])
    norm_msk = norm_nda>=0.0 # this could be used to mask v_t by a threshold norm
    norm_nda = norm_nda * norm_msk
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    nda_1d_mean = np.mean(norm_nda,axis=(1,2,3))
    nda_1d_max = np.max(norm_nda,axis=(1,2,3))
    nda_1d_median = np.median(norm_nda,axis=(1,2,3))
    nda_1d_mean_smooth = np.convolve(nda_1d_mean, kernel, mode='same')
    #print(nda_1d_mean_smooth.shape, nda_1d_mean.shape)
    
    
    # direction relative to the center
    
    directions = flow2direction_lambda(vects_nda)[...,0].numpy()
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    directions = minmax_lambda([directions,lower,upper])
    #plt.hist(directions.flatten());plt.show()
    directions = directions * norm_msk
    #plt.hist(directions.flatten());plt.show()
    
    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
    
    dir_1d_mean = np.mean(directions,axis=(1,2,3))
    dir_1d_median = np.median(directions,axis=(1,2,3))

    
    # Find phases
    # 1st smooth the direction with a rolling averge (kernelsize=4)
    # or via gaussian filter
    # 2nd min/max normalise each direction vector in a range [-1:1]
    
    
    dir_1d_mean = gaussian_filter1d(dir_1d_mean, sigma=2, mode='wrap') # original: dir_1d_mean
    
    ################# CHANGED !!!!!! ######################## RECENTLY NO CHANGES
    # min/max scaling
    dir_1d_mean = minmax_lambda([dir_1d_mean,lower,upper])
    
    
#     # test with corner specific vector fields
#     y,x = directions.shape[-2:]
    
#     vects_mean = minmax_lambda([np.mean(vects_nda,axis=(1,2,3,4)),lower,upper])
#     vects_1 = vects_nda[:,:,:y//2,:x//2,:]
#     vects_1_mean = minmax_lambda([np.mean(vects_1,axis=(1,2,3,4)),lower,upper])
#     vects_2 = vects_nda[:,:,y//2:,:x//2,:]
#     vects_2_mean = minmax_lambda([np.mean(vects_2,axis=(1,2,3,4)),lower,upper])
#     vects_3 = vects_nda[:,:,y//2:,x//2:,:]
#     vects_3_mean = minmax_lambda([np.mean(vects_3,axis=(1,2,3,4)),lower,upper])
#     vects_4 = vects_nda[:,:,:y//2,x//2:,:]
#     vects_4_mean = minmax_lambda([np.mean(vects_4,axis=(1,2,3,4)),lower,upper])
    
    
#     # direction per corner
#     dir_1 = directions[:,:,:y//2,:x//2] # top left, plot
#     dir_1_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_2 = directions[:,:,y//2:,:x//2] # top right, plot
#     dir_2_mean = minmax_lambda([np.mean(dir_1,axis=(1,2,3)),lower,upper])
#     dir_3 = directions[:,:,y//2:,x//2:] # bottom left
#     dir_3_mean = minmax_lambda([np.mean(dir_3,axis=(1,2,3)),lower,upper])
#     dir_4 = directions[:,:,:y//2,x//2:] # bottom right
#     dir_4_mean = minmax_lambda([np.mean(dir_4,axis=(1,2,3)),lower,upper])
    

#     # norm per corner
#     norm_mean = minmax_lambda([nda_1d_mean, mid, upper])
#     norm_1 = norm_nda[:,:,:y//2,:x//2]
#     norm_1_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_2 = norm_nda[:,:,y//2:,:x//2]
#     norm_2_mean = minmax_lambda([np.mean(norm_1,axis=(1,2,3)),mid,upper])
#     norm_3 = norm_nda[:,:,y//2:,x//2:]
#     norm_3_mean = minmax_lambda([np.mean(norm_3,axis=(1,2,3)),mid,upper])
#     norm_4 = norm_nda[:,:,:y//2,x//2:]
#     norm_4_mean = minmax_lambda([np.mean(norm_4,axis=(1,2,3)),mid,upper])
    

    ind = detect_phases(dir_1d_mean=dir_1d_mean[:length], length=length)
    
    
    #print('ed: {}, ms: {}, es: {}, pf: {}, md: {}'.format(*ind))
    
        # plot the mean/max norm for one patient oveer time
    if plot:
        
        
        # VECT 2D+t
        vect_2d_t = vects_nda[:,z]
        fig1 = show_2D_or_3D(vect_2d_t,allow_slicing=False, f_size=(25,2))
        ax1 = fig1.get_axes()[0]
        _ = ax1.set_ylabel('phi_t \nmid-cavity')

        
        # DIR 2D+t
        dir_2d_t = directions[:,z]
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        #dir_2d_t = minmax_lambda([dir_2d_t,lower,upper])
        div_cmap = 'bwr'
        fig = show_2D_or_3D(dir_2d_t,allow_slicing=False, f_size=(25,2),cmap=div_cmap)
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel(r'$\phi_t \nmid-cavity$') #'alpha_t \nmid-cavity'
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[len(ax.get_images())//2], cax=cax, orientation='horizontal')

        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # DIR 2D T x Z
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        directions_tz = directions.mean(axis=(2,3)) # original: norm()
        ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######    ######## CHANGED #######
        _ = ax.imshow(directions_tz.T, aspect='auto', label='aasd', cmap=div_cmap, origin='lower')
        _ = ax.set_xticks(gtind, minor=False)
        ax2 = ax.twinx()
        _ = ax2.plot(dir_1d_mean, c='black',label='dir 1d+t')
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('dir z+t\napex:base')
        
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1
        
        #ax=fig.add_subplot(rows,1,pos)
        #_ = ax.plot(dir_1d_mean, c='black',label='dir 1d+t')
        #_ = ax.plot(dir_1_mean, label='top left')
        #_ = ax.plot(dir_2_mean, label='top right')
        #_ = ax.plot(dir_3_mean, label='bottom left')
        #_ = ax.plot(dir_4_mean, label='bottom right')
        #ax.legend(loc='upper left')
        #ax.set_xticks(gtind, minor=False)
        #ax.margins(0.022,0)
        #ax.label_outer()
        #pos=pos+1
        #plt.subplots_adjust(wspace=0, hspace=0.1)
        #plt.show()
        norm_cmap = 'hot'
        
        # NORM 2D + t
        norm_2d_t = norm_nda[:,z]
        norm_2d_t = minmax_lambda([norm_2d_t,mid,upper])
        fig = show_2D_or_3D(norm_2d_t,allow_slicing=False, f_size=(25,2),cmap=norm_cmap, interpolation='none')
        ax = fig.get_axes()[0]
        _ = ax.set_ylabel('norm 2d+t\nmid-cavity')
        #ax.margins(0,0)
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(ax.get_images()[0], cax=cax, orientation='horizontal')
        
        
        
        figsize=(25,2)
        rows=2
        pos=1
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(rows,1,pos)
        
        # NORM 2D TxZ
        norm2d = minmax_lambda([norm_nda.mean(axis=(2,3)),mid,upper])
        _ = ax.imshow(norm2d.T, aspect='auto', origin='lower',cmap=norm_cmap, interpolation='none')
        _ = ax.set_xticks(gtind, minor=False)
        _ = ax.set_yticks([])
        _ = ax.set_ylabel('norm z+t\napex:base')
        ax2 = ax.twinx()
        _ = ax2.plot(nda_1d_mean, c='black',label='norm 1d+t')
        #ax.spines['top'].set_visible(False)
        #_ = ax.legend(loc='upper left')
        #ax.margins(0,0)
        ax2.label_outer()
        pos = pos+1

        """ax=fig.add_subplot(rows,1,pos)
        _ = ax.plot(norm_mean, label='norm 1d+t')
        #_ = ax.plot(norm_1_mean, label='top left')
        #_ = ax.plot(norm_2_mean, label='top right')
        #_ = ax.plot(norm_3_mean, label='bottom left')
        #_ = ax.plot(norm_4_mean, label='bottom right')
        ax.legend(loc='upper left')
        ax.set_xticks(gtind, minor=False)
        ax.margins(0.022,0)
        ax.label_outer()
        pos = pos+1
        plt.subplots_adjust(wspace=0, hspace=0.1)"""
        #plt.tight_layout()
        plt.show()

        
    
    return ind
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
p = 2#215
gt_length = int(gt_len[p,:,0].sum())
ind = np.argmax(gt[p][:gt_length],axis=0)
get_phases_from_vects(nda_vects[p][:gt_length], length=gt_length,plot=True,dir_axis=0, gtind=ind)
# Tue, 01 Mar 2022 16:50:50
_jupyterlab_variableinspector_dict_list()
