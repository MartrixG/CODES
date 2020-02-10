import numpy as np
from keras.layers import Input, Activation, Concatenate, Add, Dropout, BatchNormalization, Conv2D, DepthwiseConv2D,\
    AveragePooling2D
from keras.engine import Layer
from keras.engine import InputSpec

from keras import backend as K
from keras.utils import conv_utils
from keras.models import Model
from keras.initializers import glorot_uniform
import tensorflow.image as tfimage


class BilinearUpsampling(Layer):
    """Just a simple bilinear upsampling layer. Works only with TF.
       Args:
           upsampling: tuple of 2 numbers > 0. The upsampling ratio for h and w
           output_size: used instead of upsampling arg if passed!
    """

    def __init__(self, upsampling=(2, 2), output_size=None, data_format=None, **kwargs):
        super(BilinearUpsampling, self).__init__(**kwargs)
        self.data_format = K.image_data_format()
        self.input_spec = InputSpec(ndim=4)
        if output_size:
            self.output_size = conv_utils.normalize_tuple(
                output_size, 2, 'output_size')
            self.upsampling = None
        else:
            self.output_size = None
            self.upsampling = conv_utils.normalize_tuple(
                upsampling, 2, 'upsampling')

    def compute_output_shape(self, input_shape):
        if self.upsampling:
            height = self.upsampling[0] * \
                     input_shape[1] if input_shape[1] is not None else None
            width = self.upsampling[1] * \
                    input_shape[2] if input_shape[2] is not None else None
        else:
            height = self.output_size[0]
            width = self.output_size[1]
        return (input_shape[0],
                height,
                width,
                input_shape[3])

    def call(self, inputs, **kwargs):
        if self.upsampling:
            return K.tf.image.resize_bilinear(inputs, (inputs.shape[1] * self.upsampling[0],
                                                       inputs.shape[2] * self.upsampling[1]),
                                              align_corners=True)
        else:
            return K.tf.image.resize_bilinear(inputs, (self.output_size[0],
                                                       self.output_size[1]),
                                              align_corners=True)

    def get_config(self):
        config = {'upsampling': self.upsampling,
                  'output_size': self.output_size,
                  'data_format': self.data_format}
        base_config = super(BilinearUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def SepConv_BN(x, filters, prefix, stride=1, k=3, r=1, depth_activation=False, e=1e-3):
    """ SepConv with BN between depthwise & pointwise. Optionally add activation after BN
        Implements right "same" padding for even kernel sizes
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            k: kernel size for depthwise convolution
            r: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & pointwise conv
            e: epsilon to use in BN layer
    """

    if not depth_activation:
        x = Activation('relu')(x)
    x = DepthwiseConv2D(kernel_size=k, strides=stride, dilation_rate=r, padding='same', name=prefix + '_depthwise')(x)
    x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=e)(x)
    if depth_activation:
        x = Activation('relu')(x)
    x = Conv2D(filters, 1, padding='same',
               use_bias=False, name=prefix + '_pointwise')(x)
    x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=e)(x)
    if depth_activation:
        x = Activation('relu')(x)

    return x


def sample_block(x, f, k, stride, prefix):
    rex = Conv2D(kernel_size=k, filters=f, strides=stride, name=prefix + '_conv', padding='same',
                 kernel_initializer=glorot_uniform(seed=0))(x)
    rex = BatchNormalization(name=prefix + '_BN')(rex)
    rex = Activation('relu')(rex)
    return rex


def DeepLabV3(input_shape=(512, 512, 3), classes=21, os=16):
    x_input = Input(input_shape)
    x = x_input
    if os == 16:
        atrous_rates = (12, 24, 36)
    else:
        atrous_rates = (6, 12, 18)

    if os == 16:
        x = sample_block(x, f=8, k=3, stride=2, prefix='sample_block')
        x = sample_block(x, f=32, k=3, stride=2, prefix='sample_block1')
        short_cut = x
        x = sample_block(x, f=64, k=3, stride=2, prefix='sample_block2')
        x = sample_block(x, f=128, k=3, stride=2, prefix='sample_block3')
    else:
        x = sample_block(x, f=16, k=3, stride=2, prefix='sample_block')
        x = sample_block(x, f=64, k=3, stride=2, prefix='sample_block1')
        short_cut = x
        x = sample_block(x, f=128, k=3, stride=2, prefix='sample_block2')

    x = ASPP(x, input_shape, atrous_rates, os)

    x = BilinearUpsampling(output_size=(int(np.ceil(input_shape[0] / 4)), int(np.ceil(input_shape[1] / 4))))(x)
    feature_projection = Conv2D(filters=48, kernel_size=1, padding='same', name='feature_projection0')(short_cut)
    feature_projection = BatchNormalization(name='feature_projection_BN', epsilon=1e-5)(feature_projection)
    feature_projection = Activation('relu')(feature_projection)
    x = Concatenate()([x, feature_projection])
    x = SepConv_BN(x, filters=256, prefix='decoder_conv0', depth_activation=True, e=1e-5)
    x = SepConv_BN(x, filters=256, prefix='decoder_conv1', depth_activation=True, e=1e-5)

    x = Conv2D(classes, (1, 1), padding='same', name='semantic')(x)
    x = BilinearUpsampling(output_size=(input_shape[0], input_shape[1]))(x)

    model = Model(x_input, x, name='DeepLabv3+')
    return model


def ASPP(x, input_shape, atrous_rates, os):
    b4 = AveragePooling2D(pool_size=(int(np.ceil(input_shape[0] / os)), int(np.ceil(input_shape[1] / os))))(x)
    b4 = Conv2D(256, (1, 1), padding='same', name='image_pooling')(b4)
    b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
    b4 = Activation('relu')(b4)
    b4 = BilinearUpsampling((int(np.ceil(input_shape[0] / os)), int(np.ceil(input_shape[1] / os))))(b4)

    b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
    b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
    b0 = Activation('relu')(b0)

    b1 = SepConv_BN(x, filters=256, prefix='aspp1', r=atrous_rates[0], depth_activation=True, e=1e-5)

    b2 = SepConv_BN(x, filters=256, prefix='aspp2', r=atrous_rates[1], depth_activation=True, e=1e-5)

    b3 = SepConv_BN(x, filters=256, prefix='aspp3', r=atrous_rates[2], depth_activation=True, e=1e-5)

    x = Concatenate()([b4, b0, b1, b2, b3])

    x = Conv2D(filters=256, kernel_size=1, padding='same', name='concatenate_layer')(x)
    x = BatchNormalization(name='concatenate_layer_BN', epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    return x
