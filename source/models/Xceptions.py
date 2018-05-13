'''
Xceptions网络
'''
from keras.layers import Activation, Convolution2D, Dropout, Conv2D
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers import Flatten
from keras.models import Model
from keras.layers import Input, MaxPooling2D, SeparableConv2D
from keras import layers
from keras.regularizers import l2

def mini_XCEPTION(input_shape, num_classes, l2_regularization=0.01):
    '''
    简单的XCEPTION网络
    '''
    regularization = l2(l2_regularization)
    
    # base
    img_input = Input(input_shape)
    x = Conv2D(
        filters=8, kernel_size=(3, 3), strides=(1, 1), # 卷积核数，核大小，卷积步长
        kernel_regularizer=regularization, # 权重上的正则项，正则项在优化过程中层的参数或层的激活值添加惩罚项
        use_bias=False
        )(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(
        filters=8, kernel_size=(3, 3), strides=(1, 1), 
        kernel_regularizer=regularization,
        use_bias=False
        )(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # module1
    residual = Conv2D(
        filters=16, kernel_size=(1, 1), strides=(2, 2),
        padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = SeparableConv2D(
        filters=16, kernel_size=(3, 3), padding='same',
        kernel_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(
        filters=16, kernel_size=(3, 3), padding='same',
        kernel_regularizer=regularization, use_bias=False
    )(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module2
    residual = Conv2D(
        filters=32, kernel_size=(1, 1), strides=(2, 2),
        padding='same', use_bias=False
    )(x)
    residual = BatchNormalization()(residual)
    x = SeparableConv2D(
        filters=32, kernel_size=(3, 3), padding='same',
        kernel_regularizer=regularization, use_bias=False
    )(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(
        filters=32, kernel_size=(3, 3), padding='same',
        kernel_regularizer=regularization, use_bias=False
    )(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 3
    residual = Conv2D(
        filters=64, kernel_size=(1, 1), strides=(2, 2),
        padding='same', use_bias=False
    )(x)
    residual = BatchNormalization()(residual)
    x = SeparableConv2D(
        filters=64, kernel_size=(3, 3), padding='same',
        kernel_regularizer=regularization, use_bias=False
    )(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(
        filters=64, kernel_size=(3, 3), padding='same',
        kernel_regularizer=regularization, use_bias=False
    )(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])
    
    # module4
    residual = Conv2D(
        filters=128, kernel_size=(1, 1), strides=(2, 2),
        padding='same', use_bias=False
    )(x)
    residual = BatchNormalization()(residual)
    x = SeparableConv2D(
        filters=128, kernel_size=(3, 3), padding='same',
        kernel_regularizer=regularization, use_bias=False
    )(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(
        filters=128, kernel_size=(3, 3), padding='same',
        kernel_regularizer=regularization, use_bias=False
    )(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    x = Conv2D(num_classes, (3, 3), padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax', name='predictions')(x)

    model = Model(img_input, output)
    return model
