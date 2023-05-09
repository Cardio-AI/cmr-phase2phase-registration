import logging
from src.models.KerasLayers import ConvEncoder, get_angle_tf, get_idxs_tf, get_centers_tf, ComposeTransform, \
    conv_layer_fn, ConvBlock

import sys
import numpy as np
import tensorflow
import tensorflow as tf
from tensorflow import keras
import keras.layers as KL
from keras.layers import Input
from keras.models import Model
from tensorflow.python.keras import metrics as metr
from keras.layers import Dropout, BatchNormalization, TimeDistributed
from keras.layers import LSTM, Bidirectional
import math
import atexit

from src.models.Unets import create_unet
from src.utils import Metrics as own_metr

from src.models.ModelUtils import get_optimizer

sys.path.append('src/ext/neuron')
sys.path.append('src/ext/pynd-lib')
sys.path.append('src/ext/pytools-lib')
import src.ext.neuron.neuron.layers as nrn_layers

def get_idxs_tf(x):
    return tf.cast(
        tf.reshape(tf.where(tf.ones((x[0], x[1], x[2]))), (x[0], x[1], x[2], 3)),
        tf.float32)


# ST to apply m to an volume
def create_affine_transformer_fixed(config, networkname='affine_transformer_fixed', fill_value=0,
                                    interp_method='linear'):
    """
    Apply a learned transformation matrix to an input image, no training possible
    :param config:  Key value pairs for image size and other network parameters
    :param networkname: string, name of this model scope
    :param fill_value:
    :return: compiled keras model
    """
    if tf.distribute.has_strategy():
        strategy = tf.distribute.get_strategy()
    else:
        # distribute the training with the mirrored data paradigm across multiple gpus if available, if not use gpu 0
        strategy = tf.distribute.MirroredStrategy(devices=config.get('GPUS', ["/gpu:0"]))

    with strategy.scope():

        inputs = Input((*config.get('DIM', [10, 224, 224]), 1))
        input_displacement = Input((*config.get('DIM', [10, 224, 224]), 3))
        indexing = config.get('INDEXING', 'ij')

        # warp the source with the flow
        y = nrn_layers.SpatialTransformer(interp_method=interp_method, indexing=indexing, ident=False,
                                          fill_value=fill_value)([inputs, input_displacement])

        model = Model(inputs=[inputs, input_displacement], outputs=[y, input_displacement], name=networkname)

        return model


def create_dense_compose(config, networkname='dense_compose_displacement'):
    """
    Compose a single transform from a series of transforms.
    Supports both dense and affine transforms, and returns a dense transform unless all
    inputs are affine. The list of transforms to compose should be in the order in which
    they would be individually applied to an image. For example, given transforms A, B,
    and C, to compose a single transform T, where T(x) = C(B(A(x))), the appropriate
    function call is:
    T = compose([A, B, C])
    :param config:  Key value pairs for image size and other network parameters
    :param networkname: string, name of this model scope
    :param fill_value:
    :return: compiled keras model
    """
    if tf.distribute.has_strategy():
        strategy = tf.distribute.get_strategy()
    else:
        # distribute the training with the mirrored data paradigm across multiple gpus if available, if not use gpu 0
        strategy = tf.distribute.MirroredStrategy(devices=config.get('GPUS', ["/gpu:0"]))

    with strategy.scope():

        inputs = Input((5, *config.get('DIM', [10, 224, 224]), 3))
        indexing = config.get('INDEXING', 'ij')
        reverse = config.get('REVERSE_COMPOSE', False)
        # warp the source with the flow
        flows = tf.unstack(inputs, axis=1)
        # reverse=True
        # we need to reverse the transforms as we register from t+1 to t.
        # we need to provide the order in which we would apply the compose transforms
        # ED, MS, ES, PF, MD
        # flows:
        # 0= MS->ED, 1=ES->MS, 2=PF->ES, 3=MD->PF, 4=ED->MD
        # e.g. for compose:
        # MS->ED = [0]
        # ES->ED = [1,0]
        # PF->ED = [2,1,0]
        # list(reversed())
        if reverse:
            y = [ComposeTransform(interp_method='linear', shift_center=True, indexing=indexing,
                                  name='Compose_transform{}'.format(i))(list(reversed(flows[:i]))) for i in
                 range(2, len(flows) + 1)]
        else:
            y = [ComposeTransform(interp_method='linear', shift_center=True, indexing=indexing,
                                  name='Compose_transform{}'.format(i))(flows[:i]) for i in range(2, len(flows) + 1)]
        y = tf.stack([flows[0], *y], axis=1)

        model = Model(inputs=[inputs], outputs=[y], name=networkname)

        return model
