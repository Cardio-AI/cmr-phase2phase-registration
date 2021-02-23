from src.models.KerasLayers import ConvEncoder
import numpy as np
import tensorflow
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from src.utils.Metrics import meandiff
from src.utils.Metrics import *

from src.models.ModelUtils import get_optimizer


def create_PhaseRegressionModel(config, networkname='PhaseRegressionModel'):


    if tf.distribute.has_strategy():
        strategy = tf.distribute.get_strategy()
    else:
        # distribute the training with the "mirrored data"-paradigm across multiple gpus if available, if not use gpu 0
        strategy = tf.distribute.MirroredStrategy(devices=config.get('GPUS', ["/gpu:0"]))
    with strategy.scope():

        input_shape = config.get('DIM', [10, 224, 224])
        T_SHAPE = config.get('T_SHAPE', 35)
        PHASES = config.get('PHASES', 5)
        input_tensor = Input((T_SHAPE, *input_shape, 1))
        # define standard values according to the convention over configuration paradigm
        activation = config.get('ACTIVATION', 'elu')
        batch_norm = config.get('BATCH_NORMALISATION', False)
        pad = config.get('PAD', 'same')
        kernel_init = config.get('KERNEL_INIT', 'he_normal')
        m_pool = config.get('M_POOL', (1, 2, 2))
        f_size = config.get('F_SIZE', (3, 3, 3))
        filters = config.get('FILTERS', 16)
        drop_1 = config.get('DROPOUT_min', 0.3)
        drop_3 = config.get('DROPOUT_max', 0.5)
        bn_first = config.get('BN_FIRST', False)
        ndims = len(config.get('DIM', [10, 224, 224]))
        depth = config.get('DEPTH', 4)


        # increase the dropout through the layer depth
        dropouts = list(np.linspace(drop_1, drop_3, depth))
        dropouts = [round(i, 1) for i in dropouts]

        spatial_encoder = ConvEncoder(activation=activation,
                              batch_norm=batch_norm,
                              bn_first=bn_first,
                              depth=depth,
                              drop_3=drop_3,
                              dropouts=dropouts,
                              f_size=f_size,
                              filters=filters,
                              kernel_init=kernel_init,
                              m_pool=m_pool,
                              ndims=ndims,
                              pad=pad)

        temporal_encoder = ConvEncoder(activation=activation,
                                      batch_norm=batch_norm,
                                      bn_first=bn_first,
                                      depth=depth,
                                      drop_3=drop_3,
                                      dropouts=dropouts,
                                      f_size=f_size,
                                      filters=filters,
                                      kernel_init=kernel_init,
                                      m_pool=m_pool,
                                      ndims=ndims,
                                      pad=pad)


        gap = tensorflow.keras.layers.GlobalAveragePooling3D()


        # unstack along the temporal axis
        # added shuffeling, which avoids the model to be biased by the order
        # unstack along t, yielding a list of 3D volumes
        """inputs_spatial = tf.unstack(input_tensor,axis=1)
        import random
        indicies = list(tf.range(len(inputs_spatial)))
        zipped = list(zip(inputs_spatial, indicies))
        random.shuffle(zipped)
        inputs_spatial, indicies = zip(*zipped)
        inputs_spatial = [spatial_encoder(vol)[0] for vol in inputs_spatial]
        print(inputs_spatial[0].shape)
        inputs_spatial = [gap(vol) for vol in inputs_spatial]  # m.shape --> batchsize, timesteps, 6
        print(inputs_spatial[0].shape)
        inputs_spatial, _ = zip(*sorted(zip(inputs_spatial, indicies), key=lambda tup: tup[1]))
        inputs_spatial = tf.stack(inputs_spatial, axis=1)
        print(inputs_spatial.shape)"""

        import random
        # unstack along Z yielding a list of 2D+t slices
        inputs_temporal = tf.unstack(input_tensor, axis=2)
        indicies = list(tf.range(len(inputs_temporal)))
        zipped = list(zip(inputs_temporal, indicies))
        random.shuffle(zipped)
        inputs_temporal, indicies = zip(*zipped)
        inputs_temporal = [temporal_encoder(vol)[0] for vol in inputs_temporal]
        inputs_temporal, _ = zip(*sorted(zip(inputs_temporal, indicies), key=lambda tup: tup[1]))
        inputs_temporal = tf.stack(inputs_temporal, axis=2)
        inputs_temporal = tf.unstack(inputs_temporal, axis=1)
        print(inputs_temporal[0].shape)
        inputs_temporal = [gap(vol) for vol in inputs_temporal]
        print(inputs_temporal[0].shape)
        inputs_temporal = tf.stack(inputs_temporal, axis=1)
        print(inputs_temporal.shape)

        # inputs = tf.keras.layers.concatenate([inputs_spatial, inputs_temporal], axis=-1)
        inputs = inputs_temporal
        print('encoder')
        print(inputs.shape)

        inputs = tf.keras.layers.BatchNormalization()(inputs)
        inputs = tf.keras.layers.Dropout(rate=0.5)(inputs)
        """inputs = tf.keras.layers.Conv1D(filters=PHASES, kernel_size=5, strides=1, padding='same',
                                        activation=activation)(inputs)
        inputs = tf.keras.layers.BatchNormalization()(inputs)
        print('conv')
        print(inputs.shape)"""

        from tensorflow.keras.layers import LSTM, Bidirectional
        forward_layer = LSTM(32,return_sequences=True)
        backward_layer = LSTM(32, activation=activation, return_sequences=True,go_backwards=True)
        inputs = Bidirectional(forward_layer, backward_layer=backward_layer,input_shape=(T_SHAPE, 5))(inputs)

        print('bi LSTM')
        print(inputs.shape)
        #inputs = tf.keras.layers.BatchNormalization()(inputs)
        """inputs = tf.keras.layers.Dropout(rate=0.5)(inputs)
        print('conv1d 32, 1, 1')
        print(inputs.shape)
        inputs = tf.keras.layers.Conv1D(filters=5, kernel_size=5, strides=1, padding='same', activation=activation)(inputs)
        inputs = tf.keras.layers.BatchNormalization()(inputs)"""
        print('conv1d 32 3,1')
        print(inputs.shape)
        inputs = tf.keras.layers.Conv1D(filters=PHASES, kernel_size=1, strides=1, padding='same', activation='softmax')(inputs)
        print(inputs.shape)
        outputs = [inputs]



        model = Model(inputs=[input_tensor], outputs=outputs, name=networkname)
        model.compile(optimizer=get_optimizer(config, networkname), loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2),
                      metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.mse, tf.keras.metrics.mae, meandiff])

        return model