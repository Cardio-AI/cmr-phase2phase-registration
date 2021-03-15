


import numpy as np
import tensorflow as tf


def meandiff( y_true, y_pred, apply_sum=True, apply_average=True):

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
    #print(y_true.shape)
    y_true, y_len_msk = tf.unstack(y_true,2,axis=1)
    y_pred, _ = tf.unstack(y_pred,2,axis=1)

    y_true = tf.cast(tf.convert_to_tensor(y_true), tf.float32)
    y_pred = tf.cast(tf.convert_to_tensor(y_pred), tf.float32)
    y_len_msk = tf.cast(tf.convert_to_tensor(y_len_msk), tf.float32)
    #print(y_true.shape)
    #print(y_pred.shape)
    #print(y_len_msk.shape)

    # b, 36, 5
    temp_pred = y_pred * y_len_msk
    temp_gt = y_true * y_len_msk

    # calculate the original lengths of each mask in the current batch
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
    if apply_sum: diffs = tf.cast(tf.reduce_sum(diffs, axis=1),tf.float32)
    if apply_average: diffs = tf.reduce_mean(diffs)
    return diffs

def meandiff_transpose( y_true, y_pred):

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
    #print(y_true.shape)
    # b, 2, 5, 36
    y_true, y_len_msk = tf.unstack(y_true,2,axis=1)
    y_pred, _ = tf.unstack(y_pred,2,axis=1)

    y_true = tf.cast(tf.convert_to_tensor(y_true), tf.float32)
    y_pred = tf.cast(tf.convert_to_tensor(y_pred), tf.float32)
    y_len_msk = tf.cast(tf.convert_to_tensor(y_len_msk), tf.float32)

    # b, 5, 36
    temp_pred = y_pred * y_len_msk
    temp_gt = y_true * y_len_msk

    # calculate the original lengths of each mask in the current batch
    # b, 1
    # if we use modulo, we need to adjust the length by minus one
    # our argmax starts at 0,
    # our gt-length, which is the sum of ones in the mask starts at 1
    # we correct that by minus one
    y_len = tf.cast(tf.reduce_sum(y_len_msk[:,0,:], axis=1),dtype=tf.int32)
    # results in 3 x  b, 5,
    gt_idx = tf.cast(tf.math.argmax(temp_gt, axis=2), dtype=tf.int32)
    pred_idx = tf.cast(tf.math.argmax(temp_pred, axis=2), dtype=tf.int32)
    filled_length = tf.repeat(tf.expand_dims(y_len,axis=1),5,axis=1)

    # b, 5, 3
    stacked = tf.cast(tf.stack([gt_idx, pred_idx, filled_length], axis=-1),tf.int32)

    # sum the error per entity, and calc the mean over the batches
    # iterate over batches
    # for each batch ==> 5, 3 in stacked
    diffs = tf.map_fn(lambda x: get_min_dist_for_list(x), stacked, dtype=tf.int32)
    diffs = tf.cast(tf.reduce_sum(diffs, axis=1),tf.float32)
    diffs = tf.reduce_mean(diffs)
    return diffs

def get_min_dist_for_list(vals):
    # vals has the shape 5, 3
    # for each phase tuple (gt,pred,length)
    return tf.map_fn(lambda x :get_min_distance(x),vals, dtype=tf.int32)

def get_min_distance(vals):

    smaller = tf.reduce_min(vals[0:2], keepdims=True)
    bigger = tf.reduce_max(vals[0:2], keepdims=True)
    mod = vals[2]

    """decr_counter = tf.constant(0)
    incr_counter = tf.constant(0)
    i1 = bigger
    while (i1 != smaller):
        decr_counter = decr_counter + 1
        i1 = i1 - 1

    i1 = bigger
    while (i1 != smaller):
        incr_counter = incr_counter + 1
        i1 = tf.math.mod((i1 + 1), mod)"""
    diff = bigger - smaller
    diff_ring = mod - bigger + smaller
    return tf.reduce_min(tf.stack([diff, diff_ring]))

class Meandiff_loss(tf.keras.losses.Loss):

    def __init__(self):
        super().__init__(name='meandiff_loss')

    def __call__(self, y_true, y_pred, **kwargs):

        return -meandiff(y_true, y_pred)


import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer, Lambda



def DifferentiableArgmax(inputs):
    # if it doesnt sum to one: normalize
    
    def prob2oneHot(x):
        # len should be slightly larger than the length of x
        len = x.shape[-1] + 2
        a = K.pow(len * x, 10)
        sum_a = K.sum(a, axis=-1)
        sum_a = tf.expand_dims(sum_a, axis=1)
        onehot = tf.divide(a, sum_a)

        return onehot

    onehot = Lambda(prob2oneHot)(inputs)
    onehot = Lambda(prob2oneHot)(onehot)
    onehot = Lambda(prob2oneHot)(onehot)

    def onehot2token(x):
        cumsum = tf.cumsum(onehot, axis=-1, exclusive=True, reverse=True)
        rounding = 2 * (K.clip(cumsum, min_value=.5, max_value=1) - .5)
        token = tf.reduce_sum(rounding, axis=-1)
        return token

    token = Lambda(onehot2token)(onehot)
    return token

class CCE_combined(tf.keras.losses.Loss):

    def __init__(self, masked=False, smooth=0, transposed=False):

        super().__init__(name='cce_combined')
        self.masked = masked
        self.smooth = smooth
        self.transposed = transposed
        self.cce = CCE(masked=False,smooth=smooth,transposed=False)
        self.cce_t = CCE(masked=False, smooth=smooth, transposed=True)
        self.cce_weight = 0.5
        self.cce_t_weight = 0.5

    def __call__(self, y_true, y_pred, **kwargs):

        return self.cce_weight*self.cce(y_true, y_pred) + self.cce_t_weight * self.cce_t(y_true,y_pred)


class CCE(tf.keras.losses.Loss):

    def __init__(self, masked=False, smooth=0, transposed=False):

        super().__init__(name='cce_{}_{}_{}'.format(masked,smooth,transposed))
        self.masked = masked
        self.smooth = smooth
        self.transposed = transposed

    def __call__(self, y_pred, y_true, **kwargs):

        if y_true.shape[1] == 2: # this is a stacked onehot vector
            y_true, y_msk = tf.unstack(y_true, num=2, axis=1)
            y_pred, _ = tf.unstack(y_pred, num=2, axis=1)

            if self.masked:
                y_true = y_true * y_msk
                y_pred = y_pred * y_msk

        if self.transposed:
            y_true = tf.nn.softmax(tf.transpose(y_true, perm=[0, 2, 1]), axis=-1)
            y_pred = tf.nn.softmax(tf.transpose(y_pred, perm=[0, 2, 1]), axis=-1)

        loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred, label_smoothing=self.smooth)

        return loss




def cce_wrapper_transpose(y_true,y_pred, masked=False):
    y_true, y_msk = tf.unstack(y_true, num=2, axis=1)
    y_pred, _ = tf.unstack(y_pred, num=2, axis=1)

    if masked: y_true = y_true * y_msk
    if masked: y_pred = y_pred * y_msk

    y_true = tf.nn.softmax(tf.transpose(y_true, perm=[0, 2, 1]), axis=-1)
    y_true = tf.nn.softmax(tf.transpose(y_true, perm=[0, 2, 1]), axis=-1)

    loss = tf.keras.losses.categorical_crossentropy(y_true,y_pred, label_smoothing=0.5)
    loss = tf.transpose(loss, perm=[0,2,1])

    return loss


class MSE(tf.keras.losses.Loss):

    def __init__(self, masked=False):

        super().__init__(name='MSE_{}'.format(masked))
        self.masked = masked


    def __call__(self, y_pred, y_true, **kwargs):

        if y_true.shape[1] == 2: # this is a stacked onehot vector
            y_true, y_msk = tf.unstack(y_true, num=2, axis=1)
            y_pred, _ = tf.unstack(y_pred, num=2, axis=1)

            if self.masked:
                y_true = y_true * y_msk
                y_pred = y_pred * y_msk

        return tf.keras.losses.mse(y_true, y_pred)

def mse_wrapper(y_true,y_pred):
    y_true, y_len = tf.unstack(y_true,num=2, axis=1)
    y_pred, _ = tf.unstack(y_pred,num=2, axis=1)
    return tf.keras.losses.mse(y_true, y_pred)

def ca_wrapper(y_true, y_pred):
    y_true, y_len = tf.unstack(y_true,num=2, axis=1)
    y_pred, _ = tf.unstack(y_pred,num=2, axis=1)
    return tf.keras.metrics.categorical_accuracy(y_true, y_pred)
