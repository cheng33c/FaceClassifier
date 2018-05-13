''' 人脸提取程序：将爬取到的人物图片进行人脸提取 '''

import cv2
import os
import dlib
from bloom_filter import BloomFilter

bf = BloomFilter(max_elements=10000, error_rate=0.001)
images_path = '../datasets/images1/'
save_path = '../datasets/extract_faces2/'
#save_path = './' # test
images_list = []

def detect(img, cascade='../datasets/haarcascade_frontalface_default.xml'):
    face_cascade = cv2.CascadeClassifier(cascade)
    rects = face_cascade.detectMultiScale(img, scaleFactor=1.3,\
        minNeighbors=4, minSize=(30, 30))
    if len(rects) == 0:
        return []
    rects[:, 2:] += rects[:, :2]
    return rects

def generate_images_list():
    for _, _, files in os.walk(images_path, topdown=False):
        for file in files:
            images_list.append(file)

def main():
    for i in images_list:
        print(i)
        savepath = save_path + i
        if savepath not in bf:
            img = cv2.imread(images_path+i)
            if img is None: # 图片损坏了直接continue
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            rects = detect(img)
            for x1, y1, x2, y2 in rects:
                crop = img[y1:y2, x1:x2]
                print(images_path+i)
                cv2.imwrite(savepath, crop)
                bf.add(savepath)

if __name__ == '__main__':
    #generate_images_list()
    images_list = ['2018-4-20-21-54-57.jpg']
    main()