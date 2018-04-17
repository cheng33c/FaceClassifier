import os
import dlib
import glob
import cv2
from skimage import io
from asserts import *
from configs import *

class Face:
    """
    面部工具类: 有面部检测识别分类的各种工具
    TODO: 
    1.class Trainer， 训练各种分类器
    2.基于视频的各种分类器
    3.更好的人脸识别程序，可以通过一张照片识别某个特定的人脸
    """
    def __init__(self, image_list=None):
        pass

    def detect(self, image_list=None):
        """
        人脸检测函数: 输入图片列表，输出人脸框,使用:
        1.经典梯度直方图(HOG,classic Histogram of Oriented Gradients)
        2.线性分类器
        HOG: 这是种图像金字塔和滑动窗口检测方案，还可以检测其他半刚性物体
        无法识别人脸的图片(gen4ki): 老太婆的人脸无法识别
        """
        assert image_list,\
                "image_list不能为空,请载入一个image_list"
        detector = dlib.get_frontal_face_detector()
        for i in image_list:
            print('正在处理文件: {}'.format(i))
            img = io.imread(i)
            # 探测器寻找每个人脸边界框并上传图片1次
            # 这样图像金字塔可以使图片放大以便探测更多的人脸
            dets = detector(img, 1)
            # 返回探测到的人脸数量k+1(从0开始计数), 探测框的坐标
            for k, d in enumerate(dets):
                show_position(k, d)
            show_score(img, detector)
            show_image(img, dets)
    
    def cnn_detect(self, image_list=None,
                    detector_path=configs_cnn_detector_path):
        """
        基于CNN的人脸检测函数
        detector_path: 加载探测器路径
        CNN模型比HOG模型更精确些，但速度更慢。有GPU才能达到HOG的速度
        """
        assert image_list,\
                "image_list不能为空，请载入一个image_list"
        detector = dlib.cnn_face_detection_model_v1(detector_path)
        for i in image_list:
            print('正在处理文件: {}'.format(i))
            img = io.imread(i)
            dets = detector(img, 1)
            '''
            检测器返回一个mmod矩形对象，这个对象包含了一个列表的mmod矩形。
            只需遍历mmod_rectangles对象即可访问这些对象
            mmod_rectangle对象有两个成员变量，一个dlib.rectangle对象和一个置信度分数。
            也可以将图像列表传给检测器: 
            dets = cnn_face_detector([image_list], upsample_num, batch_size = 128)
            返回一个列表对象，存储一些存有图像列表的列表
            '''
            for k, d in enumerate(dets):
                show_position(k, d.rect)
            rects = dlib.rectangles()
            rects.extend([d.rect for d in dets])
            show_image(img, rects)

    def landmark_detect(self, image_list=None,
                        shape_predictor_path=configs_shape_predictor_path):
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
                    shape_predictor_path=configs_shape_predictor_path,
                    recognition_model_path=configs_recognition_model_path):
        """
        人脸识别函数: 判断多张图片是否是同一个人
        role_image: 主角图片，就是要比对的人物
        image_list: 和主角比对的图片
        recognition_model_path： 人脸识别模型
        TODO: 比对的功能还不够完全
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
                show_image(img, dets, shape)

    def alignment(self, image_list, 
                    shape_predictor_path=configs_shape_predictor_path):
        """图像对齐函数"""
        assert image_list,\
                'image_list不能为空'
        assert os.path.exists(shape_predictor_path),\
                'shape_predictor_path不存在'
    
        detector = dlib.get_frontal_face_detector()
        shape_predictor = dlib.shape_predictor(shape_predictor_path)
        # 使用opencv加载图片
        for i in image_list:
            bgr_img = cv2.imread(i)
            if i is None:
                print("无法加载{}".format(i))
                continue
            img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            dets = detector(img, 1)
            num_faces = len(dets)
            if num_faces == 0:
                print('没有找到人脸, 文件路径{}'.format(i))
                continue
            # 找到5个特征点进行校准
            faces = dlib.full_object_detections()
            for d in dets:
                faces.append(shape_predictor(img, d))
            # 得到校准人脸图片并显示
            # images = dlib.get_face_chips(img, faces, size=160, padding=0.25)
            images = dlib.get_face_chips(img, faces, size=320)
            for img in images:
                show_cv_image(img)

    def clustering(self, image_list=None, 
                    output_folder_path=configs_clustering_output_folder,
                    shape_predictor_path=configs_shape_predictor_path,
                    recognition_model_path=configs_recognition_model_path):
        """
        人脸聚类函数：使用聚类分析方式进行人脸识别，可以在某一群人中认出特定的人
        首先假设最大的群集将包含照片集中的普通人照片
        然后提取人脸图像保存150x150格式的最大聚类中
        TODO: 可以保存所有大于2的聚类到文件夹中
        这里图片不能使用黑白的，否则报错：
        RuntimeError: Unsupported image type, must be RGB image.
        """
        #assert_path(shape_predictor_path)
        #assert_path(recognition_model_path)
        #assert_imagelist(image_list)
        if not os.path.isdir(output_folder_path):
            os.makedirs(output_folder_path)

        detector = dlib.get_frontal_face_detector()
        shape_predictor = dlib.shape_predictor(shape_predictor_path)
        recognition_model = dlib.face_recognition_model_v1(recognition_model_path)
        
        descriptors = []
        images = []

        # 找到所有人脸并为每个人脸计算出128维人脸描述器
        for i in image_list:
            print('正在处理图片: {}'.format(i))
            img = io.imread(i)
            dets = detector(img, 1)
            num_faces = len(dets)
            if num_faces == 0:
                print("没有找到人脸，文件路径{}".format(i))
                continue
            print('检测到的人脸数: {}'.format(num_faces))
            
            for k, d in enumerate(dets):
                # 得到的人脸特征点/部分在矩形框d中
                shape = shape_predictor(img, d)
                # 计算128维向量描述的人脸形状
                face_descriptor = recognition_model.compute_face_descriptor(img, shape)
                descriptors.append(face_descriptor)
                images.append((img, shape))
        
        # 对人脸进行聚类
        labels = dlib.chinese_whispers_clustering(descriptors, 0.5)
        num_classes = len(set(labels))
        print("聚类的数量: {}".format(num_classes))
        # 找到人脸聚类最多的那个类
        biggest_class = None
        biggest_class_length = 0
        for i in range(0, num_classes):
            class_length = len([label for label in labels if label == i])
            if class_length > biggest_class_length:
                biggest_class_length = class_length
                biggest_class = i
        print("最大聚类的索引号: {}".format(biggest_class))
        print("最大聚类中存储的人脸数: {}".format(biggest_class_length))
        # 生成最大聚类生成索引
        indices = []
        for i, label in enumerate(labels):
            if label == biggest_class:
                indices.append(i)
        
        print("最大聚类中的图片索引：{}".format(str(indices)))
        # 确认输出字典的存在
        if not os.path.isdir(output_folder_path):
            os.makedirs(output_folder_path)
        # 保存提取出来的脸部
        print('正在保存最大s聚类到脸部文件夹{}'.format(output_folder_path))
        for i, index in enumerate(indices):
            img, shape = images[index]
            file_path = os.path.join(output_folder_path, 'face_' + str(i))
            # 大小(size)和填充(padding)参数默认设置为150x150, 0.25
            dlib.save_face_chip(img, shape, file_path, size=150, padding=0.25)
    
    def jitter(self, image_list,
                shape_predictor_path=configs_shape_predictor_path,):
        """
        面部抖动/增强来为面部识别模型创建训练数据
        当人脸与背景色十分相近甚至融合的时候，增强就很有用
        输入图片和干扰色，随机变换
        这里图片不能使用黑白的，否则报错：
        RuntimeError: Unsupported image type, must be RGB image.
        """
        detector = dlib.get_frontal_face_detector()
        shape_predictor = dlib.shape_predictor(predictor_path)

        bgr_img = cv2.imread()

        for i in image_list:
            img = io.imread(i)
            dets = detector(img, 1)
            num_faces = len(dets)
            if num_faces == 0:
                print('文件{}没有检测到人脸'.format(i))
                continue
            print('检测到的人脸数为{}'.format(num_faces))
            for k, d in enumerate(dets):
                show_position(k, d)
            show_image(img, dets)
        
        # 输出每次探测的分数（分数越高探测的置信度越高）
        # 第三个参数是对阈值的可选调整，负值会返回更多的监测结果，正值会减少
        # idx变量会告诉哪个脸部子检测器匹配，可用来广泛地识别不同方向的面孔
        for i in image_list:
            img = io.imread(i)
            dets, scores, idx = detector.run(img, 1, -1)
            for i, d in enumerate(dets):
                show_score(i, d)


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

def show_cv_image(img):
    cv_bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow('image', cv_bgr_img)
    cv2.waitKey(0)

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

def load_images(image_folder_path=configs_mini_faces_folder_path):
    """加载faces文件夹中的图片并返回一个图片列表"""
    assert os.path.exists(image_folder_path),\
            "图片文件夹不存在"
    image_list = []
    for root, dirs, files in os.walk(image_folder_path, topdown=False):
        for file in files:
            image_list.append(os.path.join(root, file))
    return image_list

""" test utils end """