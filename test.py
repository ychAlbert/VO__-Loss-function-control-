import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from network import VODataset, VOModel

def visualize_trajectory(poses, title='Trajectory'):
    """
    可视化轨迹
    """
    plt.figure()
    poses = np.array(poses)
    plt.plot(poses[:, 0], poses[:, 2], marker='o')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.grid()
    plt.show()

def test_vo_model(dataset_path, model_path):
    """
    测试视觉里程计模型并可视化位置偏移
    
    参数:
        dataset_path (str): 数据集路径
        model_path (str): 训练好的模型路径
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建数据集和数据加载器
    dataset = VODataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # 初始化模型
    model = VOModel().to(device)
    
    # 加载训练好的模型权重
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    # 存储预测的位姿
    predicted_poses = []
    current_pose = np.zeros(7)  # 初始位姿
    
    with torch.no_grad():
        for batch in dataloader:
            # 将数据移动到设备
            left_img = batch['left_img'].to(device)
            right_img = batch['right_img'].to(device)
            left_depth = batch['left_depth'].to(device)
            right_depth = batch['right_depth'].to(device)
            flow = batch['flow'].to(device)
            
            # 前向传播
            predicted_pose, _ = model(left_img, right_img, left_depth, right_depth, flow)
            predicted_pose = predicted_pose.cpu().numpy().flatten()
            
            # 更新当前位姿
            current_pose[:3] += predicted_pose[:3]  # 平移部分
            current_pose[3:] += predicted_pose[3:]  # 旋转部分
            
            # 存储位姿
            predicted_poses.append(current_pose.copy())
    
    # 可视化轨迹
    visualize_trajectory(predicted_poses, title='Predicted Trajectory')

def find_feature_matches(img1, img2):
    """
    在两张图像之间找到相同的特征点，并计算位移
    
    参数:
        img1 (numpy.ndarray): 第一张图像
        img2 (numpy.ndarray): 第二张图像
    
    返回:
        matches_img (numpy.ndarray): 拼接后的图像，显示匹配的特征点
        displacements (list): 特征点的位移
        tracked_points (list): 被跟踪的特征点
        new_points (list): 新捕捉的特征点
    """
    # 转换为灰度图
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # 使用ORB检测特征点
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    
    # 使用BFMatcher进行特征匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # 计算位移
    displacements = []
    tracked_points = []
    new_points = []
    for match in matches:
        pt1 = kp1[match.queryIdx].pt
        pt2 = kp2[match.trainIdx].pt
        displacement = (pt2[0] - pt1[0], pt2[1] - pt1[1])
        displacements.append(displacement)
        tracked_points.append(pt1)
        new_points.append(pt2)
    
    # 绘制匹配结果
    matches_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    return matches_img, displacements, tracked_points, new_points

def compute_reprojection_correction(model, left_img, right_img, left_depth, right_depth, flow, point):
    """
    计算重投影的改正值与权重
    
    参数:
        model (nn.Module): 训练好的模型
        left_img (torch.Tensor): 左图像张量
        right_img (torch.Tensor): 右图像张量
        left_depth (torch.Tensor): 左深度图张量
        right_depth (torch.Tensor): 右深度图张量
        flow (torch.Tensor): 光流张量
        point (tuple): 原始点坐标
    
    返回:
        corrected_point (tuple): 改正后的点坐标
        corrected_weight (float): 改正后的权重
    """
    model.eval()
    with torch.no_grad():
        # 前向传播
        updated_pose, sigma = model(left_img.unsqueeze(0), right_img.unsqueeze(0), left_depth.unsqueeze(0), right_depth.unsqueeze(0), flow.unsqueeze(0))
        
        # 提取改正值和权重
        delta = updated_pose.squeeze().cpu().numpy()
        weight = sigma.squeeze().cpu().numpy()
        
        # 计算改正后的点坐标和权重
        corrected_point = (point[0] + delta[0], point[1] + delta[1])
        corrected_weight = weight[0] * np.linalg.norm(delta[:2])
        
    return corrected_point, corrected_weight

def main():
    # 示例用法
    dataset_path = 'P001'
    model_path = 'vo_model.pth'
    test_vo_model(dataset_path, model_path)
    
    # 加载示例图像
    img1 = cv2.imread('P001\image_left\\000000_left.png')
    img2 = cv2.imread('P001\image_left\\000001_left.png')
    
    # 找到特征点匹配并计算位移
    matches_img, displacements, tracked_points, new_points = find_feature_matches(img1, img2)
    
    # 使用matplotlib显示匹配结果
    plt.imshow(matches_img)
    plt.title('Feature Matches')
    plt.show()
    
    # 输出被跟踪的特征点和新捕捉的特征点
    print(f'被跟踪的特征点: {tracked_points}')
    print(f'新捕捉的特征点: {new_points}')
    
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VOModel().to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    
    # 加载示例数据
    dataset = VODataset(dataset_path)
    sample = dataset[0]
    
    left_img = sample['left_img'].to(device)
    right_img = sample['right_img'].to(device)
    left_depth = sample['left_depth'].to(device)
    right_depth = sample['right_depth'].to(device)
    flow = sample['flow'].to(device)
    
    # 计算重投影改正值与权重
    point = (100, 100)
    corrected_point, corrected_weight = compute_reprojection_correction(model, left_img, right_img, left_depth, right_depth, flow, point)
    print(f'改正后的点: {corrected_point}, 改正后的权重: {corrected_weight}')

if __name__ == '__main__':
    main()
