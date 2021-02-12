from src.models.KerasLayers import ConvEncoder
import numpy as np
import tensorflow
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

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
        PHASES = config.get('PHASES', 6)
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
        dense1_weights = config.get('DENSE1_WEIGHTS', 256)
        dense2_weights = config.get('DENSE2_WEIGHTS', 9)
        dense3_weights = T_SHAPE*PHASES
        dense4_weights = T_SHAPE*PHASES

        T_SHAPE = config.get('T_SHAPE', 10)
        PHASES = config.get('PHASES', 5)

        # increase the dropout through the layer depth
        dropouts = list(np.linspace(drop_1, drop_3, depth))
        dropouts = [round(i, 1) for i in dropouts]

        encoder = ConvEncoder(activation=activation,
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
        inputs = tf.unstack(input_tensor,axis=1)
        inputs = [encoder(vol)[0] for vol in inputs]
        print(inputs[0].shape)
        # Shrink the encoding towards the euler angles and translation params,
        # no additional dense layers before the GAP layer
        inputs = [gap(vol) for vol in inputs]  # m.shape --> batchsize, timesteps, 6
        print('gap elem 0')
        print(inputs[0].shape)
        inputs = tf.stack(inputs, axis=1)
        print('concat all')
        print(inputs.shape)
        inputs = tf.keras.layers.Conv1D(filters=256, kernel_size=5, strides=1, padding='same', activation=activation)(inputs)
        print('conv1d 256, 5, 2')
        print(inputs.shape)
        inputs = tf.keras.layers.Conv1D(filters=256, kernel_size=3, strides=1, padding='same', activation=activation)(inputs)
        print(inputs.shape)
        inputs = tf.keras.layers.Conv1D(filters=6, kernel_size=1, strides=1, padding='same', activation='softmax')(inputs)
        print(inputs.shape)
        outputs = [inputs]

        model = Model(inputs=[input_tensor], outputs=outputs, name=networkname)
        model.compile(optimizer=get_optimizer(config, networkname), loss=tf.keras.losses.categorical_crossentropy)

        return model