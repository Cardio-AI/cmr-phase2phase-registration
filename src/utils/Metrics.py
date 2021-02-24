


import numpy as np
import tensorflow as tf


def meandiff( y_true, y_pred, batchsize=4):

    """
    Average over the batches
    the sum of the absolute difference between two arrays
    y_true and y_pred are one-hot vectors with the following shape
    batchsize * timesteps * phase classes
    e.g.: 4 * 36 * 5
    First for gt and pred:
    - get the timesteps per phase with the highest probability
    - get the absolute difference between gt and pred
    (- later we can slice each patient by the max value in the corresponding gt indices)
    - sum the diff per entity
    - calc the mean over all examples

    Parameters
    ----------
    y_true :
    y_pred :

    Returns tf.float32 scalar
    -------

    """
    print(y_true.shape)
    y_true, y_len_msk = tf.unstack(y_true,2,axis=1)
    y_pred, _ = tf.unstack(y_pred,2,axis=1)

    y_true = tf.cast(tf.convert_to_tensor(y_true), tf.float32)
    y_pred = tf.cast(tf.convert_to_tensor(y_pred), tf.float32)
    y_len_msk = tf.cast(tf.convert_to_tensor(y_len_msk), tf.float32)
    print(y_true.shape)
    print(y_pred.shape)
    print(y_len_msk.shape)

    # b, 36, 5


    # b, 5
    #gt_idxs = tf.math.argmax(y_true, axis=1)
    # b,
    # this is the length of the original cardiac cycle, we need to use this as cutoff point
    #gt_max = tf.cast(tf.reduce_max(gt_idxs, axis=1), tf.int32)
    #print(y_len)
    #print((y_len*-1)+36)
    # b, ones + zeros (36), 5
    # define ones with (y_length, 5)
    # pad first dim with (before,after) (0, 36-y_length) and the 2nd dim with (0,0)


    temp_pred = y_pred * y_len_msk
    temp_gt = y_true * y_len_msk

    # calculate the original length of each mask
    # b, 1
    y_len = tf.cast(tf.reduce_sum(y_len_msk[:,:,0], axis=1),dtype=tf.int32)
    #print('y_len shape: {}'.format(y_len.shape))
    # returns b, 5,
    gt_idx = tf.cast(tf.math.argmax(temp_gt, axis=1), dtype=tf.int32)
    pred_idx = tf.cast(tf.math.argmax(temp_pred, axis=1), dtype=tf.int32)
    filled_length = tf.repeat(tf.expand_dims(y_len,axis=1),5,axis=1)
    #print('gt_idx shape: {}'.format(gt_idx.shape))
    #print('pred_idx shape: {}'.format(pred_idx.shape))
    #print('filled shape: {}'.format(filled_length.shape))

    # b, 5, 3
    stacked = tf.cast(tf.stack([gt_idx, pred_idx, filled_length], axis=-1),tf.int32)

    # sum the error per entity, and calc the mean over the batches
    # for each batch ==> 5, 3 in stacked
    diffs = tf.map_fn(lambda x: get_min_dist_for_list(x), stacked, dtype=tf.int32)
    diffs = tf.cast(tf.reduce_sum(diffs, axis=0),tf.float32)
    diffs = tf.reduce_mean(diffs)
    return diffs



def get_min_dist_for_list(vals):
    # vals has the shape 5, 2
    # for each phase tuple (gt,pred,length)
    return tf.map_fn(lambda x :get_min_distance(x),vals, dtype=tf.int32)

def get_min_distance(vals):
    #assert(mod>(tf.reduce_max(a,b))), 'a: {}, b: {}, mod:{}, '.format(a,b,mod)

    decr_counter = tf.constant(0)
    incr_counter = tf.constant(0)
    print(vals.shape)
    smaller = tf.reduce_min(vals[0:2], keepdims=True)
    bigger = tf.reduce_max(vals[0:2], keepdims=True)
    mod = vals[2]

    i1 = bigger
    while (i1 != smaller):
        decr_counter = decr_counter + 1
        i1 = i1 - 1

    i1 = bigger
    while (i1 != smaller):
        incr_counter = incr_counter + 1
        i1 = tf.math.mod((i1 + 1), mod)

    return tf.reduce_min(tf.stack([decr_counter, incr_counter]))


def cce_wrapper(y_true,y_pred):
    y_true, y_len = tf.unstack(y_true,num=2,axis=1)
    y_pred, _ = tf.unstack(y_pred,num=2,axis=1)
    return tf.keras.losses.categorical_crossentropy(y_true,y_pred,label_smoothing=0.2)

def mse_wrapper(y_true,y_pred):
    y_true, y_len = tf.unstack(y_true, axis=1)
    y_pred, _ = tf.unstack(y_pred, axis=1)
    return tf.keras.losses.mse(y_true, y_pred)

def ca_wrapper(y_true, y_pred):
    y_true, y_len = tf.unstack(y_true, axis=1)
    y_pred, _ = tf.unstack(y_pred, axis=1)
    return tf.keras.metrics.CategoricalAccuracy()(y_true, y_pred)
