import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from network import VODataset, VOModel
from test import find_feature_matches

def create_video_from_images(image_folder, output_video_path, fps=10):
    """
    将图像文件夹中的图像创建为视频
    
    参数:
        image_folder (str): 图像文件夹路径
        output_video_path (str): 输出视频文件路径
        fps (int): 视频帧率
    """
    images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    video.release()

def visualize_feature_displacement(image_folder, output_video_path, fps=10):
    """
    可视化特征点位移并生成视频
    
    参数:
        image_folder (str): 图像文件夹路径
        output_video_path (str): 输出视频文件路径
        fps (int): 视频帧率
    """
    images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(".png")]
    tracked_points = None

    video = None

    for i in range(len(images) - 1):
        img1 = cv2.imread(os.path.join(image_folder, images[i]))
        img2 = cv2.imread(os.path.join(image_folder, images[i + 1]))

        matches_img, displacements, tracked_points, new_points = find_feature_matches(img1, img2)

        # 绘制特征点位移
        for pt1, pt2 in zip(tracked_points, new_points):
            cv2.arrowedLine(matches_img, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (0, 255, 0), 2)

        if video is None:
            height, width, layers = matches_img.shape
            video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))

        video.write(matches_img)

    if video is not None:
        video.release()

def main():
    image_folder = 'P001/image_left'
    output_video_path = 'feature_displacement_video2.avi'
    fps = 10

    visualize_feature_displacement(image_folder, output_video_path, fps)
    print(f'视频已保存到 {output_video_path}')

if __name__ == '__main__':
    main()
