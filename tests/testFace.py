import sys
sys.path.append("../source")
from Face import *

def test_Face(function_name=None, image_folder=None):
    """
    Face测试函数， 用于代替前面的测试函数
    function_name: 输入函数名
    image_folder: 图片文件夹
    """
    assert function_name, "function_name不能为空"
    cface = Face()
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
    # 调用你想要测试的函数
    #test_Face('detect')
    #test_Face('clustering')
    #test_Face('landmark_detect')
    #test_Face('recognition')
    #test_Face('alignment')
    #test_Face('jitter')
    pass
