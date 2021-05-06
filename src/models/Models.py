from tensorflow.python.keras.layers import LSTM, Bidirectional

from src.models.KerasLayers import ConvEncoder
import sys
import numpy as np
import tensorflow
import tensorflow as tf
import tensorflow.keras.layers as KL
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import metrics as metr

from src.models.Unets import create_unet
from src.utils import Metrics as own_metr

from src.models.ModelUtils import get_optimizer

sys.path.append('src/ext/neuron')
sys.path.append('src/ext/pynd-lib')
sys.path.append('src/ext/pytools-lib')
import src.ext.neuron.neuron.layers as nrn_layers


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
        input_tensor = Input(shape=(T_SHAPE, *input_shape, 1))
        # define standard values according to the convention over configuration paradigm
        activation = config.get('ACTIVATION', 'elu')
        batch_norm = config.get('BATCH_NORMALISATION', False)
        pad = config.get('PAD', 'same')
        kernel_init = config.get('KERNEL_INIT', 'he_normal')
        m_pool = config.get('M_POOL', (1, 2, 2))
        f_size = config.get('F_SIZE', (3, 3, 3))
        filters = config.get('FILTERS', 16)
        drop_1 = config.get('DROPOUT_MIN', 0.3)
        drop_3 = config.get('DROPOUT_MAX', 0.5)
        bn_first = config.get('BN_FIRST', False)
        ndims = len(config.get('DIM', [10, 224, 224]))
        depth = config.get('DEPTH', 4)
        batchsize = config.get('BATCHSIZE', 8)


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
        print('Shape after the temporal encoder')
        print(inputs_temporal.shape)
        inputs_temporal = tf.unstack(inputs_temporal, axis=1)
        inputs_temporal = [gap(vol) for vol in inputs_temporal]
        inputs_temporal = tf.stack(inputs_temporal, axis=1)

        """inputs_temporal = tf.keras.layers.Flatten()(inputs_temporal)
        inputs_temporal = tf.keras.layers.Dense(units=36*10,activation=activation)(inputs_temporal)
        inputs_temporal = tf.keras.layers.Reshape(target_shape=(36,10))(inputs_temporal)"""

        #inputs = tf.keras.layers.concatenate([inputs_spatial, inputs_temporal], axis=-1)
        inputs = inputs_temporal
        print('Shape after GAP')
        print(inputs.shape)
        # 36, 256
        """inputs = tf.keras.layers.BatchNormalization()(inputs)
        inputs = tf.keras.layers.Dropout(rate=0.5)(inputs)

        onehot_pre = tf.keras.layers.Conv1D(filters=64, kernel_size=5, strides=1, padding='same', activation=activation,
                                        name='conv_pre_final')(inputs)"""


        """forward_layer = LSTM(32,return_sequences=True)
        backward_layer = LSTM(32, activation='tanh', return_sequences=True,go_backwards=True) # maybe change to tanh
        inputs = Bidirectional(forward_layer, backward_layer=backward_layer,input_shape=inputs.shape)(inputs)
        """
        # 36,64
        print('Shape after Bi-LSTM layer')
        print(inputs.shape)
        #onehot = tf.keras.layers.Dense(units=5,activation='softmax', name='final_conv')(inputs)
        onehot = tf.keras.layers.Conv1D(filters=PHASES, kernel_size=1, strides=1, padding='same', activation=activation,
                                        name='final_conv')(inputs)

        # 36, 5
        print('Shape after final conv layer')
        print(onehot.shape)

        # add empty tensor with one-hot shape to align with gt
        zeros = tf.zeros_like(onehot)
        onehot = tf.stack([onehot, zeros], axis=1)

        outputs = [onehot]


        #losses = [own_metr.CCE(masked=True, smooth=0.2,transposed=True)]
        losses = [own_metr.MSE(masked=False)]
        #losses = [own_metr.Meandiff_loss()]

        model = Model(inputs=[input_tensor], outputs=outputs, name=networkname)
        model.compile(
            optimizer=get_optimizer(config, networkname),
            loss=losses,
            metrics=[own_metr.mse_wrapper, own_metr.ca_wrapper, own_metr.meandiff] #
        )

        return model


def create_RegistrationModel(config):
    """
    A registration wrapper for 3D image2image registration
    """

    if config is None:
        config = {}
    input_shape = config.get('DIM', [10, 224, 224])
    T_SHAPE = config.get('T_SHAPE', 5)
    PHASES = config.get('PHASES', 5)
    input_tensor = Input(shape=(T_SHAPE, *input_shape, 1))
    input_tensor_empty = Input(shape=(T_SHAPE, *input_shape, 3))
    # define standard values according to the convention over configuration paradigm
    activation = config.get('ACTIVATION', 'elu')
    batch_norm = config.get('BATCH_NORMALISATION', False)
    pad = config.get('PAD', 'same')
    kernel_init = config.get('KERNEL_INIT', 'he_normal')
    m_pool = config.get('M_POOL', (1, 2, 2))
    f_size = config.get('F_SIZE', (3, 3, 3))
    filters = config.get('FILTERS', 16)
    drop_1 = config.get('DROPOUT_MIN', 0.3)
    drop_3 = config.get('DROPOUT_MAX', 0.5)
    bn_first = config.get('BN_FIRST', False)
    ndims = len(input_shape)
    depth = config.get('DEPTH', 4)
    indexing = 'ij'
    interp_method = 'linear'
    Conv = getattr(KL, 'Conv{}D'.format(ndims))
    Conv_layer = Conv(ndims, kernel_size=3, padding='same',
                kernel_initializer=tensorflow.keras.initializers.RandomNormal(mean=0.0, stddev=1e-5), name='unet2flow')
    st_layer = nrn_layers.SpatialTransformer(interp_method=interp_method, indexing=indexing, ident=True,
                                      fill_value=0, name='deformable_layer')


    unet = create_unet(config, single_model=False)

    input_vols = tf.unstack(input_tensor, axis=1)
    print(input_vols[0].shape)
    import random
    indicies = list(tf.range(len(input_vols)))
    zipped = list(zip(input_vols, indicies))
    random.shuffle(zipped)
    input_vols_shuffled, indicies = zip(*zipped)
    pre_flows = [unet(vol) for vol in input_vols_shuffled]
    flows= [Conv_layer(vol) for vol in pre_flows]  # m.shape --> batchsize, timesteps, 6
    flows, _ = zip(*sorted(zip(flows, indicies), key=lambda tup: tup[1]))


    #pre_flows = unet(input_tensor)

    #flow = Conv(ndims, kernel_size=3, padding='same',
    #           kernel_initializer=kernel_init, name='unet2flow')(pre_flow)

    transformed = [st_layer([input_vol, flow]) for input_vol, flow in zip(input_vols, flows)]
    transformed = tf.stack(transformed, axis=1)
    flow = tf.stack(flows, axis=1)

    outputs = [transformed, flow]

    #super().__init__(name='simpleregister', inputs=[input_tensor, input_tensor_empty], outputs=outputs)
    return Model(name='simpleregister', inputs=[input_tensor, input_tensor_empty], outputs=outputs)







