import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
from tqdm import tqdm  # 用于进度条显示
import torch.nn.functional as F

########################################
# 数据集定义：只使用左图（连续帧）和左侧深度
########################################
class VODataset(Dataset):
    def __init__(self, trajectory_dir, transform=None):
        """
        自定义单目视觉里程计数据集（训练时使用深度监督）
        
        参数:
            trajectory_dir (str): 数据集根目录路径
            transform (callable, optional): 可选的对样本进行转换的操作
        """
        self.trajectory_dir = trajectory_dir
        self.transform = transform
        
        # 仅加载左图与对应深度（使用连续帧对）
        self.left_images = sorted([f for f in os.listdir(os.path.join(trajectory_dir, 'image_left')) if f.endswith('_left.png')])
        self.left_depths = sorted([f for f in os.listdir(os.path.join(trajectory_dir, 'depth_left')) if f.endswith('_left_depth.npy')])
        
        # 加载位姿文件（假设 pose_left.txt 中每行均为一帧位姿）
        self.poses = self._load_poses(os.path.join(trajectory_dir, 'pose_left.txt'))
    
    def _load_poses(self, pose_file):
        with open(pose_file, 'r') as f:
            poses = [list(map(float, line.strip().split())) for line in f]
        return np.array(poses)
    
    def __len__(self):
        # 使用相邻两帧构成一个样本
        return len(self.left_images) - 1
    
    def __getitem__(self, idx):
        # 加载当前帧和下一帧的左图
        curr_img_path = os.path.join(self.trajectory_dir, 'image_left', self.left_images[idx])
        next_img_path = os.path.join(self.trajectory_dir, 'image_left', self.left_images[idx + 1])
        curr_img = cv2.imread(curr_img_path)
        next_img = cv2.imread(next_img_path)
        
        # 加载当前帧的深度图（训练阶段监督信号）
        curr_depth_path = os.path.join(self.trajectory_dir, 'depth_left', self.left_depths[idx])
        curr_depth = np.load(curr_depth_path)  # 假设深度为单通道
        
        # 计算当前帧与下一帧之间的相对位姿
        curr_pose = self.poses[idx]
        next_pose = self.poses[idx + 1]
        relative_pose = self._compute_relative_pose(curr_pose, next_pose)
        
        # 预处理图像（调整大小、归一化、转换为 CxHxW）
        curr_img = self._preprocess_image(curr_img)
        next_img = self._preprocess_image(next_img)
        
        # 转换为张量
        curr_img = torch.from_numpy(curr_img).float()
        next_img = torch.from_numpy(next_img).float()
        curr_depth = torch.from_numpy(curr_depth).float()  # shape: H x W
        relative_pose = torch.from_numpy(relative_pose).float()
        
        return {
            'curr_img': curr_img,
            'next_img': next_img,
            'curr_depth': curr_depth,
            'relative_pose': relative_pose
        }
    
    def _preprocess_image(self, img):
        """
        图像预处理：调整大小（256×256）、归一化，并转换为 C×H×W 格式
        """
        img = cv2.resize(img, (256, 256))
        img = img / 255.0  # 归一化到 [0,1]
        img = np.transpose(img, (2, 0, 1))
        return img
    
    def _compute_relative_pose(self, curr_pose, next_pose):
        """
        简单计算相对位姿（实际中可能需要更复杂的 SE(3) 计算）
        """
        return np.array(next_pose) - np.array(curr_pose)

########################################
# 更新操作符（差分束调整层）保持基本结构
########################################
class UpdateOperator(nn.Module):
    def __init__(self):
        super(UpdateOperator, self).__init__()
        
        # 1D卷积层
        self.conv1d = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        
        # 自注意力机制
        self.self_attn = nn.MultiheadAttention(embed_dim=256, num_heads=8)
        
        # 过渡块
        self.transition = nn.Sequential(
            nn.Linear(256 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # 因子头：预测位姿修正和置信权重
        self.factor_head_delta = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 7)  # 7维修正（3平移+3旋转+1尺度或其他参数）
        )
        self.factor_head_sigma = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 7),
            nn.Sigmoid()  # 限制在 (0,1) 范围内
        )
        
        # 束调整层：模拟全局优化过程
        self.ba_layer = nn.Linear(7, 7)
    
    def forward(self, correlation_features, context_features):
        # 1D卷积
        x = self.conv1d(correlation_features)  # (batch, 256, N)
        
        # 自注意力
        x = x.permute(2, 0, 1)  # (N, batch, 256)
        x, _ = self.self_attn(x, x, x)
        x = x.permute(1, 2, 0)  # (batch, 256, N)
        
        # 过渡块：将所有 patch 特征展平后输入全连接网络
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1)
        x = self.transition(x)
        
        # 分支：位姿修正和置信权重
        delta = self.factor_head_delta(x)
        sigma = self.factor_head_sigma(x)
        
        # 束调整层
        updated_pose = self.ba_layer(delta)
        
        return updated_pose, sigma

########################################
# 单目视觉里程计模型：仅使用连续两帧左图，并预测深度（训练时监督）
########################################
class VOModel(nn.Module):
    def __init__(self):
        super(VOModel, self).__init__()
        
        # 公用的图像特征提取网络
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # 深度预测分支：从当前帧特征预测深度（训练时监督，测试时模型自动生成）
        self.depth_predictor = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )
        
        # 拼接当前帧与下一帧特征后用于位姿估计的降维网络
        # 输入通道数调整为 128+128 = 256
        self.combined_conv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # 位姿回归网络
        self.pose_regressor = nn.Sequential(
            nn.Linear(128 * 2 * 2, 512),  # 根据 combined_conv 的输出尺寸设定（此处假设输出尺寸为 2×2）
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 7)  # 预测 7 维参数（如 3 平移 + 3 旋转 [+1尺度/其他]）
        )
        
        # 更新操作符（差分束调整层），利用特征相关性进行全局优化
        self.update_operator = UpdateOperator()
    
    def forward(self, curr_img, next_img, curr_depth=None):
        """
        参数:
            curr_img (Tensor): 当前帧左图，形状 (B,3,256,256)
            next_img (Tensor): 下一帧左图，形状 (B,3,256,256)
            curr_depth (Tensor, optional): 当前帧深度（训练时作为监督），形状 (B,H,W)
            
        返回:
            updated_pose: 通过 BA 得到的位姿更新
            sigma: 置信权重
            pred_depth: 预测的当前帧深度（训练时监督，测试时自动估计）
        """
        # 提取当前帧与下一帧特征
        curr_features = self.feature_extractor(curr_img)   # (B,128,H',W')
        next_features = self.feature_extractor(next_img)       # (B,128,H',W')
        
        # 深度预测：仅使用当前帧特征
        pred_depth = self.depth_predictor(curr_features)       # (B,1,H',W')
        
        # 拼接两帧特征以获取跨帧信息，用于位姿估计
        combined_features = torch.cat([curr_features, next_features], dim=1)  # (B,256,H',W')
        combined_features = self.combined_conv(combined_features)
        
        # 展平后预测初始位姿
        flattened = combined_features.view(combined_features.size(0), -1)
        pose = self.pose_regressor(flattened)
        
        # 利用局部 patch 特征进行更新（模拟光流估计及 BA 优化）
        correlation_features = combined_features.view(combined_features.size(0), combined_features.size(1), -1)
        context_features = flattened.unsqueeze(2)
        updated_pose, sigma = self.update_operator(correlation_features, context_features)
        
        return updated_pose, sigma, pred_depth

########################################
# 捆绑调整损失：最小化重投影误差（位姿损失示例）
########################################
def bundle_adjustment_loss(predicted_pose, relative_pose, delta, sigma):
    """
    束调整损失函数，最小化重投影误差
    
    参数:
        predicted_pose (Tensor): 预测的位姿
        relative_pose (Tensor): 真实相对位姿
        delta (Tensor): 修正量（这里示例为 predicted_pose - relative_pose）
        sigma (Tensor): 置信权重
        
    返回:
        loss (Tensor): 束调整损失
    """
    reprojection_error = predicted_pose - (relative_pose + delta)
    weighted_error = torch.sum((reprojection_error ** 2) * sigma, dim=1)
    loss = torch.mean(weighted_error) * 1e6  # 放大损失
    return loss

########################################
# 训练函数
########################################
def train_vo_model(dataset_path, epochs=20, batch_size=32, learning_rate=1e-4, depth_loss_weight=1.0):
    """
    训练单目视觉里程计模型
    
    参数:
        dataset_path (str): 数据集根目录路径
        epochs (int): 训练轮数
        batch_size (int): 批次大小
        learning_rate (float): 学习率
        depth_loss_weight (float): 深度监督损失权重
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = VODataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = VOModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 定义深度损失（L1 损失）
    depth_loss_fn = nn.L1Loss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        with tqdm(total=len(dataloader), desc=f'训练轮数 {epoch+1}/{epochs}', unit='batch') as pbar:
            for batch in dataloader:
                curr_img = batch['curr_img'].to(device)
                next_img = batch['next_img'].to(device)
                curr_depth = batch['curr_depth'].to(device)  # shape: (B, H, W)
                relative_pose = batch['relative_pose'].to(device)
                
                optimizer.zero_grad()
                
                # 前向传播
                predicted_pose, sigma, pred_depth = model(curr_img, next_img, curr_depth)
                
                # 计算位姿损失（这里 delta 简单使用 predicted_pose - relative_pose）
                pose_loss = bundle_adjustment_loss(predicted_pose, relative_pose, predicted_pose - relative_pose, sigma)
                
                # 为了计算深度损失，将监督深度下采样到预测深度的分辨率
                curr_depth = curr_depth.unsqueeze(1)  # (B,1,H,W)
                curr_depth_ds = F.interpolate(curr_depth, size=pred_depth.shape[2:], mode='bilinear', align_corners=False)
                depth_loss = depth_loss_fn(pred_depth, curr_depth_ds)
                
                total_batch_loss = pose_loss + depth_loss_weight * depth_loss
                total_batch_loss.backward()
                optimizer.step()
                
                total_loss += total_batch_loss.item()
                pbar.set_postfix({'损失': f'{total_batch_loss.item():.4f}'})
                pbar.update(1)
        
        print(f'轮次 [{epoch+1}/{epochs}], 平均损失: {total_loss/len(dataloader):.4f}')
    
    torch.save(model.state_dict(), 'vo_model.pth')
    print("模型训练完成并保存。")

########################################
# 主函数入口
########################################
def main():
    dataset_path = 'P001'  # 根据实际数据集路径修改
    train_vo_model(dataset_path)

if __name__ == '__main__':
    main()
