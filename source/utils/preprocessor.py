import numpy as np

def preprocess_input(x, v2=True):
    ''' 对输入矩阵进行预处理到[0, 1]之间 避免溢出和大量运算 '''
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x