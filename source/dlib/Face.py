import os
import dlib
import glob
from skimage import io

class Face:
    
    def __init__(self):
        pass

    def detect(self, image_list=None):
        """
        人脸检测函数: 输入图片列表，输出人脸框,使用:
        1.经典梯度直方图(HOG,classic Histogram of Oriented Gradients)
        2.线性分类器
        这是种图像金字塔和滑动窗口检测方案，还可以检测其他半刚性物体
        无法识别人脸的图片(gen4ki): 老太婆的人脸无法识别
        """
        assert image_list,\
                "image_list不能为空,请载入一个image_list"
        detector = dlib.get_frontal_face_detector()

        for i in image_list:
            img = io.imread(i)
            # 探测器寻找每个人脸边界框并上传图片1次
            # 这样图像金字塔可以使图片放大以便探测更多的人脸
            dets = detector(img, 1)
            # 返回探测到的人脸数量k+1(从0开始计数), 探测框的坐标
            for k, d in enumerate(dets):
                show_position(k, d)
            show_score(img, detector)
            show_image(img, dets)

    def landmark_detect(self, image_list=None,
                        shape_predictor_path='../../models/shape_predictor_65_face_landmarks.dat'):
        """
        人脸关键点检测函数,关键点覆盖鼻子，嘴角，眉毛，眼睛等
        predictor_path: 预测器位置(默认采用68个关键点探测器)
        """
        assert os.path.exists(shape_predictor_path),\
                "predictor_path不能为空,请载入一个face landmark predictor"
        assert image_list,\
                "image_list不能为空，请载入一个image list"
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(shape_predictor_path)

        for i in image_list:
            img = io.imread(i)
            dets = detector(img, 1)
            for k, d in enumerate(dets):
                show_position(k, d)
                shape = predictor(img, d)
                show_parts(shape)
                show_position(k, d)
            show_image(img, dets, shape)

    def recognition(self, image_list=None, role_image=None,
                    shape_predictor_path='../../models/shape_predictor_65_face_landmarks.dat',
                    recognition_model_path='../../models/dlib_face_recognition_resnet_model_v1.dat'):
        """
        人脸识别函数: 判断多张图片是否是同一个人
        role_image: 主角图片，就是要比对的人物
        image_list: 和主角比对的图片
        recognition_model_path： 人脸识别模型
        """
        assert image_list,\
                "image_list不能为空，请载入一个image_list"
        assert os.path.exists(recognition_model_path),\
                "recognition_model_path不能为空，请载入一个人脸识别模型"
        
        if role_image == None:
            role_image = image_list[0]
        face_detector = dlib.get_frontal_face_detector()
        shape_predictor = dlib.shape_predictor(shape_predictor_path)
        recognition_model = dlib.face_recognition_model_v1(recognition_model_path)
        role_img = io.imread(role_image)
        role_dets = face_detector(role_img, 1)
        role_face_descriptor = None
        for k, d in enumerate(role_dets):
            shape = shape_predictor(role_img, d)
            role_face_descriptor = recognition_model.compute_face_descriptor(role_img, shape)


        for i in image_list:
            img = io.imread(i)
            dets = face_detector(img, 1)
            for k, d in enumerate(dets):
                shape = shape_predictor(img, d)
                # 计算两个人脸矩阵(img:128维矢量)的欧氏距离
                # 如果两张脸距离小于0.6就可以认为是同个人
                face_descriptor = recognition_model.compute_face_descriptor(img, shape)
                # 最后一个参数可设置更高以便获得更好的准确率
                #face_descriptor = recognition_model.compute_face_descriptor(img, shape, 10)
                distance = EuclideanDistances(role_face_descriptor, face_descriptor)
                print(distance)
                #if (face_descriptor < 0.6):
                #    print('同一个人')
                #else:
                #    print('不是同一个人')
                show_image(img, dets, shape)
            

""" helper function start """
def EuclideanDistances(matrixA, matrixB):
    """
    计算两个矩阵间的欧氏距离
    """
    BT = matrixB.transpose()
    vecProd = matrixA * BT
    SqA =  matrixA.getA()**2
    sumSqA = numpy.matrix(numpy.sum(SqA, axis=1))
    sumSqAEx = numpy.tile(sumSqA.transpose(), (1, vecProd.shape[1]))
    SqB = matrixB.getA()**2
    sumSqB = numpy.sum(SqB, axis=1)
    sumSqBEx = numpy.tile(sumSqB, (vecProd.shape[0], 1))
    SqED = sumSqBEx + sumSqAEx - 2*vecProd
    ED = (SqED.getA())**0.5
    return numpy.matrix(ED)
""" helper function end """


""" test utils start """
def show_image(img, dets, shape=None):
    """显示图片"""
    win = dlib.image_window()
    win.clear_overlay()
    win.set_image(img)
    if shape != None:
        win.add_overlay(shape)
    win.add_overlay(dets)
    dlib.hit_enter_to_continue()

def show_score(img, detector):
    """
    显示图片得分,分数越高置信度越高
    第三个参数是对检测阀值的可选调整，负值会返回更多的检测结果，正值会减少
    idx变量会告诉哪个脸部子检测器匹配，可用来广泛地识别不同方向的面孔
    """
    dets, scores, idx = detector.run(img, 1, -1)
    for i, d in enumerate(dets):
        print("检测 {}, 分数 {}, 人脸类型 {}".format(
            d, scores[i], idx[i]
        ))
        print("\n===================")

def show_position(i, d):
    """
    face detector测试函数
    输出人脸盒子的位置信息
    """
    print("detect {}, left: {}, top: {}, right: {}, bottom: {}".format(
        i, d.left(), d.top(), d.right(), d.bottom()
    ))

def show_parts(shape):
    """
    shape predictor测试函数
    输出landmark位置信息
    """
    print("部分0: {}, 部分1: {} ...".format(
        shape.part(0), shape.part(1)
    ))

def load_images(image_folder_path='faces'):
    """加载faces文件夹中的图片并返回一个图片列表"""
    assert os.path.exists(image_folder_path),\
            "图片文件夹不存在"
    image_list = []
    for root, dirs, files in os.walk('faces/', topdown=False):
        for file in files:
            image_list.append(os.path.join(root, file))
    return image_list

""" test utils end """

""" test functions start """
def test_Face_detect():
    """Face detect测试函数"""
    cface = Face()
    image_list = load_images()
    cface.detect(image_list)

def test_Face_landmark_detect():
    """Face landmark detect测试函数"""
    cface = Face()
    image_list = load_images()
    cface.landmark_detect(image_list)

def test_Face_recognition():
    """Face recognition测试函数"""
    cface = Face()
    image_list = load_images()
    cface.recognition(image_list)



""" test functions end """


# 供测试用
if __name__ == '__main__':
    # 调用你想要测试的函数
    #test_Face_detector()
    #test_Face_landmark_detect()
    test_Face_recognition()

