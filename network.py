import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
from tqdm import tqdm  # 导入tqdm库用于进度条显示

class VODataset(Dataset):
    def __init__(self, trajectory_dir, transform=None):
        """
        自定义视觉里程计数据集
        
        参数:
            trajectory_dir (str): 轨迹目录的路径
            transform (callable, optional): 可选的对样本进行转换的操作
        """
        self.trajectory_dir = trajectory_dir
        self.transform = transform
        
        # 准备文件列表
        self.left_images = sorted([f for f in os.listdir(os.path.join(trajectory_dir, 'image_left')) if f.endswith('_left.png')])
        self.right_images = sorted([f for f in os.listdir(os.path.join(trajectory_dir, 'image_right')) if f.endswith('_right.png')])
        self.left_depths = sorted([f for f in os.listdir(os.path.join(trajectory_dir, 'depth_left')) if f.endswith('_left_depth.npy')])
        self.right_depths = sorted([f for f in os.listdir(os.path.join(trajectory_dir, 'depth_right')) if f.endswith('_right_depth.npy')])
        self.flows = sorted([f for f in os.listdir(os.path.join(trajectory_dir, 'flow')) if f.endswith('_flow.npy')])
        
        # 加载位姿文件
        self.poses = self._load_poses(os.path.join(trajectory_dir, 'pose_left.txt'))
    
    def _load_poses(self, pose_file):
        """
        从文本文件加载位姿
        
        返回:
            位姿的numpy数组
        """
        with open(pose_file, 'r') as f:
            poses = [list(map(float, line.strip().split())) for line in f]
        return np.array(poses)
    
    def __len__(self):
        return len(self.left_images) - 1  # 我们需要成对的帧
    
    def __getitem__(self, idx):
        # 加载左图和右图
        left_img_path = os.path.join(self.trajectory_dir, 'image_left', self.left_images[idx])
        right_img_path = os.path.join(self.trajectory_dir, 'image_right', self.right_images[idx])
        
        left_img = cv2.imread(left_img_path)
        right_img = cv2.imread(right_img_path)
        
        # 加载深度图
        left_depth_path = os.path.join(self.trajectory_dir, 'depth_left', self.left_depths[idx])
        right_depth_path = os.path.join(self.trajectory_dir, 'depth_right', self.right_depths[idx])
        
        left_depth = np.load(left_depth_path)
        right_depth = np.load(right_depth_path)
        
        # 加载光流
        flow_path = os.path.join(self.trajectory_dir, 'flow', self.flows[idx])
        flow = np.load(flow_path)
        
        # 提取相对位姿
        curr_pose = self.poses[idx]
        next_pose = self.poses[idx + 1]
        relative_pose = self._compute_relative_pose(curr_pose, next_pose)
        
        # 预处理图像
        left_img = self._preprocess_image(left_img)
        right_img = self._preprocess_image(right_img)
        
        # 转换为张量
        left_img = torch.from_numpy(left_img).float()
        right_img = torch.from_numpy(right_img).float()
        left_depth = torch.from_numpy(left_depth).float()
        right_depth = torch.from_numpy(right_depth).float()
        flow = torch.from_numpy(flow).float()
        relative_pose = torch.from_numpy(relative_pose).float()
        
        return {
            'left_img': left_img,
            'right_img': right_img,
            'left_depth': left_depth,
            'right_depth': right_depth,
            'flow': flow,
            'relative_pose': relative_pose
        }
    
    def _preprocess_image(self, img):
        """
        预处理图像：调整大小，归一化
        """
        img = cv2.resize(img, (256, 256))  # 调整为固定大小
        img = img / 255.0  # 归一化到[0,1]
        img = np.transpose(img, (2, 0, 1))  # 转换为CxHxW
        return img
    
    def _compute_relative_pose(self, curr_pose, next_pose):
        """
        计算两帧之间的相对位姿
        """
        # 这是一个简化版本。实际实现取决于位姿表示
        return np.array(next_pose) - np.array(curr_pose)

class UpdateOperator(nn.Module):
    def __init__(self):
        super(UpdateOperator, self).__init__()
        
        # 1D卷积层
        self.conv1d = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        
        # 自注意力机制
        self.self_attn = nn.MultiheadAttention(embed_dim=256, num_heads=8)
        
        # 过渡块：调整输入维度为256*4=1024
        self.transition = nn.Sequential(
            nn.Linear(256 * 4, 128),  # 修正输入特征数
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # 因子头
        self.factor_head_delta = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 7)  # 预测七维修正值以匹配relative_pose
        )
        self.factor_head_sigma = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 7),  # 预测置信权重
            nn.Sigmoid()  # 限制在(0,1)范围内
        )
        
        # 束调整层（BA）
        self.ba_layer = nn.Linear(7, 7)  # 调整输出为7维度以匹配relative_pose

    def forward(self, correlation_features, context_features):
        # 1D卷积
        x = self.conv1d(correlation_features)  # 输入形状：(batch, 128, N) → 输出形状：(batch, 256, N)
        
        # 自注意力机制
        x = x.permute(2, 0, 1)  # 调整为(seq_len, batch, embed_dim) → (N, batch, 256)
        x, _ = self.self_attn(x, x, x)  # 输出形状保持(N, batch, 256)
        x = x.permute(1, 2, 0)  # 调整回(batch, 256, N)
        
        # 过渡块
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1)  # 展平后的形状：(batch, 256*N)
        x = self.transition(x)  # 输入应匹配256*N → 修正后的Linear层处理
        
        # 因子头
        delta = self.factor_head_delta(x)
        sigma = self.factor_head_sigma(x)
        
        # 束调整层
        updated_pose = self.ba_layer(delta)
        
        return updated_pose, sigma

def bundle_adjustment_loss(predicted_pose, relative_pose, delta, sigma):
    """
    束调整损失函数，最小化重投影误差
    
    参数:
        predicted_pose (torch.Tensor): 预测的位姿
        relative_pose (torch.Tensor): 相对位姿
        delta (torch.Tensor): 预测的轨迹修正值
        sigma (torch.Tensor): 置信权重
    
    返回:
        loss (torch.Tensor): 束调整损失
    """
    # 计算重投影误差
    reprojection_error = predicted_pose - (relative_pose + delta)
    
    # 加权平方和
    weighted_error = torch.sum((reprojection_error ** 2) * sigma, dim=1)
    
    # 计算损失并放大
    loss = torch.mean(weighted_error) * 1e6  # 放大损失值
    
    return loss

class VOModel(nn.Module):
    def __init__(self):
        super(VOModel, self).__init__()
        
        # 特征提取网络
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # 深度特征提取
        self.depth_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # 光流特征提取
        self.flow_extractor = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # 新增的降维层
        self.combined_conv = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # 位姿回归网络
        self.pose_regressor = nn.Sequential(
            nn.Linear(128 * 2 * 2, 512),  # 调整输入大小为降维后的特征数
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 7)  # 3个平移，3个旋转
        )
        
        # 更新操作符
        self.update_operator = UpdateOperator()
    
    def forward(self, left_img, right_img, left_depth, right_depth, flow):
        # 从图像中提取特征
        left_img_features = self.feature_extractor(left_img)
        right_img_features = self.feature_extractor(right_img)
        
        # 从深度图中提取特征
        left_depth_features = self.depth_extractor(left_depth.unsqueeze(1))
        right_depth_features = self.depth_extractor(right_depth.unsqueeze(1))
        
        # 从光流中提取特征
        flow_features = self.flow_extractor(flow.permute(0, 3, 1, 2))  # 调整flow的维度
        
        # 确保所有特征的尺寸匹配
        min_size = min(left_img_features.size(2), right_img_features.size(2), 
                      left_depth_features.size(2), right_depth_features.size(2), 
                      flow_features.size(2))
        left_img_features = left_img_features[:, :, :min_size, :min_size]
        right_img_features = right_img_features[:, :, :min_size, :min_size]
        left_depth_features = left_depth_features[:, :, :min_size, :min_size]
        right_depth_features = right_depth_features[:, :, :min_size, :min_size]
        flow_features = flow_features[:, :, :min_size, :min_size]
        
        # 拼接所有特征
        combined_features = torch.cat([
            left_img_features, 
            right_img_features, 
            left_depth_features, 
            right_depth_features, 
            flow_features
        ], dim=1)
        
        # 应用降维卷积层
        combined_features = self.combined_conv(combined_features)
        
        # 展平
        flattened = combined_features.view(combined_features.size(0), -1)
        
        # 预测位姿
        pose = self.pose_regressor(flattened)
        
        # 更新操作符
        correlation_features = combined_features.view(combined_features.size(0), combined_features.size(1), -1)
        context_features = flattened.unsqueeze(2)
        updated_pose, sigma = self.update_operator(correlation_features, context_features)
        
        return updated_pose, sigma

def train_vo_model(dataset_path, epochs=20, batch_size=32, learning_rate=1e-4):
    """
    训练视觉里程计模型
    
    参数:
        dataset_path (str): 数据集路径
        epochs (int): 训练轮数
        batch_size (int): 训练批次大小
        learning_rate (float): 优化器的学习率
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建数据集和数据加载器
    dataset = VODataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 初始化模型
    model = VOModel().to(device)
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练循环
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # 使用tqdm显示进度条
        with tqdm(total=len(dataloader), desc=f'训练轮数 {epoch+1}/{epochs}', unit='batch') as pbar:
            for batch in dataloader:
                # 将数据移动到设备
                left_img = batch['left_img'].to(device)
                right_img = batch['right_img'].to(device)
                left_depth = batch['left_depth'].to(device)
                right_depth = batch['right_depth'].to(device)
                flow = batch['flow'].to(device)
                relative_pose = batch['relative_pose'].to(device)
                
                # 清零梯度
                optimizer.zero_grad()
                
                # 前向传播
                predicted_pose, sigma = model(left_img, right_img, left_depth, right_depth, flow)
                
                # 计算束调整损失
                loss = bundle_adjustment_loss(predicted_pose, relative_pose, predicted_pose - relative_pose, sigma)
                
                # 反向传播
                loss.backward()
                
                # 优化
                optimizer.step()
                
                total_loss += loss.item()
                
                # 更新进度条
                pbar.set_postfix({'损失': f'{loss.item():.4f}'})
                pbar.update(1)
        
        # 打印每轮统计信息
        print(f'轮数 [{epoch+1}/{epochs}], 平均损失: {total_loss/len(dataloader):.4f}')
    
    # 保存模型
    torch.save(model.state_dict(), 'vo_model.pth')
    print("模型训练完成并保存。")

def main():
    dataset_path = 'P001'
    train_vo_model(dataset_path)

if __name__ == '__main__':
    main()