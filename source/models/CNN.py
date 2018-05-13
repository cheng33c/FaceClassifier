from keras.layers import Activation, Convolution2D, Dropout, Conv2D
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers import Flatten
from keras.models import Model
from keras.layers import Input, MaxPooling2D, SeparableConv2D
from keras import layers
from keras.regularizers import l2

def CNN(input_shape, num_classes):
    '''
    CNN卷积神经网络实现
    '''
    # 设置序贯模型
    model = Sequential()
    # 添加一个Convolution2D卷积层
    model.add(Convolution2D(filters=16, kernel_size=(7, 7), padding='same',
                            name='image_array', input_shape=input_shape))
    model.add(BatchNormalization())
    # 添加激活函数Relu
    model.add(Activation('relu'))
    # 添加平均池化层
    model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
    # 添加Dropout层，机率设为0.5.
    model.add(Dropout(.5))

    model.add(Convolution2D(filters=32, kernel_size=(5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(filters=32, kernel_size=(5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.5))

    model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(.5))

    model.add(Convolution2D(filters=128, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(filters=128, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(.5))

    model.add(Convolution2D(filters=256, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(filters=num_classes, kernel_size=(3, 3), padding='same'))
    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax', name='predictions'))
    return model


def simpler_CNN(input_shape, num_classes):
    '''
    简单版本的CNN，去除池化层,加快训练速度
    '''
    model = Sequential()
    
    model.add(Convolution2D(filters=16, kernel_size=(5, 5), padding='same', name='image_array', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Convolution2D(filters=16, kernel_size=(5, 5), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(.25))

    model.add(Convolution2D(filters=32, kernel_size=(5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(filters=32, kernel_size=(5, 5), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(.25))

    model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(.25))

    model.add(Convolution2D(filters=64, kernel_size=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(.25))

    model.add(Convolution2D(filters=256, kernel_size=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same'))

    model.add(Convolution2D(filters=256, kernel_size=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(filters=num_classes, kernel_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(Flatten())
    model.add(Activation('softmax', name='predictions'))
    return model

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

if __name__ == '__main__':
    input_shape = (64, 64, 1)
    num_classes = 7
    #model = CNN(input_shape, num_classes)
    model = simpler_CNN(input_shape, num_classes)
    model.summary()