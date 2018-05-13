import sys
sys.path.append("../")
from Face import *

class TestFace(Face):
    '''
    Face.py 测试类
    '''
    def __init__(self):
        # 函数表，用于查找并调用函数。
        # 若需要添加测试方法，则首先添加到这张表中，再编写相应的逻辑代码
        self.function_table = {
            'detect': self.detect,                       # 人脸检测
            'cnn_detect': self.cnn_detect,               # 基于CNN的人脸检测
            'landmark_detect': self.landmark_detect,     # 人脸特征点检测
            'recognition': self.recognition,             # 人脸识别
            'alignment': self.alignment,                 # 人脸对齐
            'clustering': self.clustering,               # 人脸聚类
            'jitter': self.jitter,                       # 人脸抖动/增强
        }
    
    def call_function(self, function_name):
        image_list = load_images()
        function = self.function_table[function_name]
        function(image_list)

if __name__ == '__main__':
    tsface = TestFace()
    # 调用你想要测试的函数
    #tsface.call_function('clustering')
    #tsface.call_function('detect')
    tsface.call_function('detect')