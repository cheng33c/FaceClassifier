import sys
import cv2
import numpy
from keras.models import load_model
from utils.datasets import get_labels, detect_faces, draw_text
from utils.inference import draw_bounding_box, apply_offsets
from utils.inference import load_detection_model, load_image
from utils.preprocessor import preprocess_input

# 图片路径
image_path = sys.argv[1]
if image_path == None:
    print('image_path不能为空，调用方法:python3 image_emotion.py myimage.jpg')
    exit(1)
# 设置人脸识别模型，情绪识别模型，性别识别模型的路径
detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
gender_model_path = '../trained_models/gender_models/simple_CNN.81-0.96.hdf5'
# 加载fer2013标签作为情绪识别标签, imdb标签作为性别识别标签
emotion_labels = get_labels('fer2013')
gender_labels = get_labels('imdb')
# 加载模型
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
gender_classifier = load_model(gender_model_path, compile=False)
# 输入模型大小
emotion_target_size = emotion_classifier.input_shape[1:3]
gender_target_size = gender_classifier.input_shape[1:3]
# 加载图片
rgb_image = load_image(image_path, grayscale=False)
gray_image = load_image(image_path, grayscale=True)
# 删除一维项
gray_image = np.squeeze(gray_image)
# 类型转换为uint8
gray_image = gray_image.astype('uint8')
# 人脸框盒子的位置设置
# gender_offsets = (30, 60)
gender_offsets = (10, 10)
# emotion_offsets = (20, 40)
emotion_offsets = (0, 0)

# 检测人脸
faces = detect_faces(face_detection, gray_image)
for face_coordinates in faces:
    x1, x2, y1, y2 = apply_offsets(face_coordinates, gender_offsets)
    rgb_face = rgb_image[y1:y2, x1:x2]
    x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
    gray_face = gray_image[y1:y2, x1:x2]

    try:
        rgb_face = cv2.resize(rgb_face, gender_target_size)
        gray_face = cv2.resize(gray_face, emotion_target_size)
    except:
        continue
    