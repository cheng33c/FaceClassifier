""" 人脸情绪分类器 """
import argparse
import os

images_list = []

def generate_images_list(folder_path):
    """ 传入文件夹路径，遍历文件夹生成文件列表images_list """
    for _, _, files in os.walk(folder_path, topdown=False):
        for file in files:
            images_list.append(file)

def main():
    parser = argparse.ArgumentParser("人脸情绪识别分类器")
    parser.add_argument("--file", type=str, help="指定需要探测人脸图片文件名")
    parser.add_argument("--folder", type=str, help="指定需要探测人脸图片的文件夹")
    parser.add_argument("--method", type=str, default="emotion_detect", 
                        help="指定所使用的方法:[METHOD: detect, cnn_detect, emotion_detect]")
    args = parser.parse_args()

    if args.folder != None:
        generate_images_list(args.folder)
    elif args.file != None:
        images_list.append(args.file)

    method = args.method
    print(method)

if __name__ == '__main__':
    main()