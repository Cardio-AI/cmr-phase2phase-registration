import logging

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
        T_SHAPE = config.get('T_SHAPE', 36)
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
        add_bilstm = config.get('ADD_BILSTM', False)
        lstm_units = config.get('BILSTM_UNITS', 64)
        final_activation = config.get('FINAL_ACTIVATION', 'relu').lower()
        loss = config.get('LOSS', 'mse').lower()
        mask_loss = config.get('MASK_LOSS', False)
        pre_gap_conv = config.get('PRE_GAP_CONV', False)


        # increase the dropout through the layer depth
        dropouts = list(np.linspace(drop_1, drop_3, depth))
        dropouts = [round(i, 1) for i in dropouts]

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
        # added shuffling, which avoids the model to be biased by the order
        # unstack along t, yielding a list of 3D volumes

        import random
        # unstack along Z yielding a list of 2D+t slices
        inputs_temporal = tf.unstack(input_tensor, axis=2)
        indicies = list(tf.range(len(inputs_temporal)))
        zipped = list(zip(inputs_temporal, indicies))
        random.shuffle(zipped)
        inputs_temporal, indicies = zip(*zipped)
        # feed the 2D+T slices as 3D volumes into the temporal encoder (3D conv)
        inputs_temporal = [temporal_encoder(vol)[0] for vol in inputs_temporal]
        inputs_temporal, _ = zip(*sorted(zip(inputs_temporal, indicies), key=lambda tup: tup[1]))
        # stack the 2D+T encoding along the spatial axis (2, as we have b,t,z,x,y,c) --> 3D+T encoding
        inputs_temporal = tf.stack(inputs_temporal, axis=2)
        print('Shape after the temporal encoder')
        print(inputs_temporal.shape)
        # unstack each 3D volume encoding, apply global average pooling on each
        inputs_temporal = tf.unstack(inputs_temporal, axis=1)

        if pre_gap_conv:
            gap_conv = tf.keras.layers.Conv3D(filters=64, kernel_size=3, strides=(2, 1, 1), padding='valid',
                                              activation=activation, kernel_initializer=kernel_init)

            inputs_temporal = [tf.keras.layers.BatchNormalization()(elem) for elem in inputs_temporal]
            inputs_temporal = [tf.keras.layers.Dropout(rate=0.5)(elem) for elem in inputs_temporal]
            inputs_temporal = [gap_conv(vol) for vol in inputs_temporal]
        inputs_temporal = [gap(vol) for vol in inputs_temporal]
        inputs_temporal = tf.stack(inputs_temporal, axis=1)

        inputs = inputs_temporal
        print('Shape after GAP')
        print(inputs.shape)
        # 36, 256
        if add_bilstm:
            print('add a bilstm layer with: {} lstm units'.format(lstm_units))
            """inputs = tf.keras.layers.BatchNormalization()(inputs)
            inputs = tf.keras.layers.Dropout(rate=0.5)(inputs)
    
            onehot_pre = tf.keras.layers.Conv1D(filters=64, kernel_size=5, strides=1, padding='same', activation=activation,
                                            name='conv_pre_final')(inputs)"""
            forward_layer = LSTM(lstm_units,return_sequences=True)
            backward_layer = LSTM(lstm_units, activation='tanh', return_sequences=True,go_backwards=True) # maybe change to tanh
            inputs = Bidirectional(forward_layer, backward_layer=backward_layer,input_shape=inputs.shape)(inputs)

        # 36,64
        print('Shape after Bi-LSTM layer')
        print(inputs.shape)

        # input (36,encoding) output (36,5)
        # either 36,256 --> from the temp encoder or
        # 36,64 --> 64 --> number of BI-LSTM units
        # activation to linear
        onehot = tf.keras.layers.Conv1D(filters=PHASES, kernel_size=1, strides=1, padding='same', activation='linear',
                                        name='final_conv')(inputs)
        if final_activation == 'relu':
            onehot = tf.keras.activations.relu(onehot)
        elif final_activation == 'softmax':
            # axis -1 --> one class per timestep, as we repeat the phases its not possible to softmax the phase
            onehot = tf.keras.activations.softmax(onehot, axis=-1)
        elif final_activation == 'sigmoid':
            onehot = tf.keras.activations.sigmoid(onehot)
        else:
            logging.info('No final activation given! Please check the "FINAL_ACTIVATION" param!')

        # 36, 5
        print('Shape after final conv layer')
        print(onehot.shape)

        # add empty tensor with one-hot shape to align with gt
        zeros = tf.zeros_like(onehot)
        onehot = tf.stack([onehot, zeros], axis=1)

        outputs = [onehot]

        if loss == 'cce':
            losses = [own_metr.CCE(masked=mask_loss, smooth=0.8,transposed=False)]
        elif loss == 'meandiff':
            losses = [own_metr.Meandiff_loss()]
        else: # default fallback --> MSE - works the best
            losses = [own_metr.MSE(masked=mask_loss)]

        print('added loss: {}'.format(loss))
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







