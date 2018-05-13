import numpy as np
from scipy.misc import imread, imresize

def preprocess_input(x, v2=True):
    ''' 对输入矩阵进行预处理到[0, 1]之间 避免溢出和大量运算 '''
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

def img_read(image_name):
    return imread(image_name)

def img_resize(image_name):
    return imresize(image_array, size)

def to_categorical(integer_classes, num_classes=2):
    integer_classes = np.asarray(integer_classes, dtype='int')
    num_samples = integer_classes.shape[0]
    categorical = np.zeros((num_samples, num_classes))
    categorical[np.arange(num_samples), integer_classes] = 1
    return categorical