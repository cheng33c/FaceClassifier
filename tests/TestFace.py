import sys
sys.path.append("../source")
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

    def test_Face(function_name=None, image_folder=None):
        """
        Face测试函数， 用于代替前面的测试函数
        function_name: 输入函数名
        image_folder: 图片文件夹
        """
        assert function_name, "function_name不能为空"
        cface = Face()
        function = function_name[function_name]
        if image_folder != None:
            image_list = load_images(image_folder)
        else:
            image_list = load_images()
        if function_name == 'detect':
            cface.detect(image_list)
        elif function_name == 'cnn_detect':
            cface.cnn_detect(image_list)
        elif function_name == 'landmark_detect':
            cface.landmark_detect(image_list)
        elif function_name == 'recognition':
            cface.recognition(image_list)
        elif function_name == 'alignment':
            cface.alignment(image_list)
        elif function_name == 'clustering':
            cface.clustering(image_list)
        elif function_name == 'jitter':
            cface.jitter(image_list)
        else:
            raise Exception('错误的函数名，请输入Face中的函数名进行测试')

if __name__ == '__main__':
    tsface = TestFace()
    # 调用你想要测试的函数
    #tsface.call_function('detect')
    tsface.call_function('clustering')
    #test_Face('landmark_detect')
    #test_Face('recognition')
    #test_Face('alignment')
    #test_Face('jitter')
    pass
