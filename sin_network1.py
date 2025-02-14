import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
from tqdm import tqdm
import torch.nn.functional as F

########################################
# 数据集定义：加载左右图、深度和位姿，同时生成 patch 坐标和 COR
########################################
class VODataset(Dataset):
    def __init__(self, trajectory_dir, num_patches=100, transform=None):
        """
        自定义单目视觉里程计数据集（训练时使用深度监督，同时生成 patch 坐标和 COR）
        
        参数:
            trajectory_dir (str): 数据集根目录路径
            num_patches (int): 每帧随机采样的 patch 数量
            transform (callable, optional): 对样本进行转换的操作
        """
        self.trajectory_dir = trajectory_dir
        self.transform = transform
        self.num_patches = num_patches
        
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
        
        # 计算当前帧与下一帧之间的相对位姿（此处仅作示例）
        curr_pose = self.poses[idx]
        next_pose = self.poses[idx + 1]
        relative_pose = np.array(next_pose) - np.array(curr_pose)
        
        # 预处理图像（调整大小、归一化、转换为 C×H×W）
        curr_img = self._preprocess_image(curr_img)
        next_img = self._preprocess_image(next_img)
        
        # 随机采样 patch 坐标和生成 COR
        # 归一化坐标范围设定为 [-0.8, 0.8]，避免边缘问题；同时为下一帧加入小随机偏移
        N = self.num_patches
        patch_coords = []
        cor_list = []
        for i in range(N):
            x_curr = np.random.uniform(-0.8, 0.8)
            y_curr = np.random.uniform(-0.8, 0.8)
            offset = np.random.uniform(-0.1, 0.1, size=2)  # 模拟重投影偏移
            x_next = x_curr + offset[0]
            y_next = y_curr + offset[1]
            patch_coords.append([x_curr, y_curr, x_next, y_next])
            cor_list.append(offset)
        patch_coords = np.array(patch_coords)  # (N,4)
        cor = np.array(cor_list)  # (N,2)
        
        sample = {
            'curr_img': torch.from_numpy(curr_img).float(),   # (3,256,256)
            'next_img': torch.from_numpy(next_img).float(),       # (3,256,256)
            'curr_depth': torch.from_numpy(curr_depth).float(),   # (H,W)
            'relative_pose': torch.from_numpy(relative_pose).float(),
            'patch_coords': torch.from_numpy(patch_coords).float(),  # (N,4)
            'cor': torch.from_numpy(cor).float()                   # (N,2)
        }
        return sample
    
    def _preprocess_image(self, img):
        """
        图像预处理：调整大小（256×256）、归一化，并转换为 C×H×W 格式
        """
        img = cv2.resize(img, (256, 256))
        img = img / 255.0  # 归一化到 [0,1]
        img = np.transpose(img, (2, 0, 1))
        return img

########################################
# 更新操作符（利用 patch 特征进行重投影修正及置信度估计）
########################################
class UpdateOperator(nn.Module):
    def __init__(self, input_dim=256):
        """
        input_dim: 拼接后的特征维度（correlation + context），本例为 192+64=256
        """
        super(UpdateOperator, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=3, padding=1)
        # 使用 batch_first=True 简化维度转换（输入形状 (B, N, 256)）
        self.self_attn = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)
        
        self.transition = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        # 因子头：预测 2D 修正 delta 和 1D 置信权重 sigma
        self.factor_head_delta = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # 2D 修正
        )
        self.factor_head_sigma = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 输出 (0,1)
        )
        # 可微分束调整层，用于对预测修正进行进一步优化
        self.ba_layer = nn.Linear(2, 2)
    
    def forward(self, correlation_features, context_features):
        """
        参数:
            correlation_features: (B, N, D_corr) 例如 (B, N, 192)
            context_features: (B, N, D_context) 例如 (B, N, 64)
        返回:
            global_correction: 通过 SoftMax 聚合后的全局重投影修正，形状 (B, 2)
            sigma: 每个 patch 的置信权重，形状 (B, N, 1)
        """
        # 拼接特征: (B, N, 192+64=256)
        x = torch.cat([correlation_features, context_features], dim=-1)
        # 1D 卷积要求 (B, channels, N)
        x_conv = self.conv1d(x.transpose(1, 2)).transpose(1, 2)  # (B, N, 256)
        # 自注意力
        attn_out, _ = self.self_attn(x_conv, x_conv, x_conv)  # (B, N, 256)
        # 对每个 patch 单独更新
        patch_features = self.transition(attn_out)  # (B, N, 128)
        # 分支预测 delta 和 sigma
        delta = self.factor_head_delta(patch_features)  # (B, N, 2)
        sigma = self.factor_head_sigma(patch_features)  # (B, N, 1)
        # BA 层对每个 patch 的 delta 进行修正
        delta_refined = self.ba_layer(delta)  # (B, N, 2)
        # SoftMax 聚合：使用 sigma 作为权重对各 patch 修正进行加权求和
        sigma_soft = torch.softmax(sigma, dim=1)  # (B, N, 1)
        global_correction = torch.sum(delta_refined * sigma_soft, dim=1)  # (B, 2)
        return global_correction, sigma

########################################
# 单目视觉里程计模型（改）：
# – 分别使用匹配和上下文特征提取网络
# – 利用双线性插值提取 patch 特征，并构造相关性特征
# – 输入 patch 坐标和 COR，输出全局重投影修正和置信权重
########################################
class VOModel(nn.Module):
    def __init__(self, patch_size=3):
        super(VOModel, self).__init__()
        self.patch_size = patch_size
        
        # 匹配特征提取网络（示例中的轻量卷积网络，可替换为残差结构）
        self.matching_feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # 上下文特征提取网络
        self.context_feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # 构造特征金字塔（本例中通过平均池化获得1/4分辨率，可根据需要扩展）
        self.avg_pool = nn.AvgPool2d(kernel_size=4)
        
        # 对 COR（2D）进行线性投影到 64 维
        self.cor_proj = nn.Linear(2, 64)
        
        # 更新操作符：输入为拼接后的相关性（匹配）特征和上下文特征
        # 本例中：匹配特征：当前帧 64 + 下一帧 64，再加上 COR 投影 64 => 192，
        # 上下文 patch 特征为 64，拼接后输入 UpdateOperator 的维度为 256
        self.update_operator = UpdateOperator(input_dim=256)
    
    def forward(self, curr_img, next_img, patch_coords, cor):
        """
        参数:
            curr_img (Tensor): 当前帧图像，形状 (B,3,256,256)
            next_img (Tensor): 下一帧图像，形状 (B,3,256,256)
            patch_coords (Tensor): 每个样本中 N 个 patch 坐标，形状 (B, N, 4)
                                     每行 [x_curr, y_curr, x_next, y_next]（归一化坐标）
            cor (Tensor): 每个 patch 的重投影结果，形状 (B, N, 2)
            
        返回:
            global_correction: 全局重投影修正，形状 (B,2)
            sigma: 每个 patch 的置信权重，形状 (B,N,1)
        """
        # 提取匹配特征（当前帧和下一帧）
        match_feat_curr = self.matching_feature_extractor(curr_img)   # (B,64,H,W)
        match_feat_next = self.matching_feature_extractor(next_img)     # (B,64,H,W)
        
        # 提取上下文特征（仅使用当前帧，可根据需要调整）
        context_feat_curr = self.context_feature_extractor(curr_img)      # (B,64,H,W)
        
        # 构造特征金字塔（本例仅使用全分辨率，可扩展使用 avg_pool 得到低分辨率特征）
        # 如：match_feat_curr_pyr = [match_feat_curr, self.avg_pool(match_feat_curr)]
        match_feat_curr_pyr = match_feat_curr
        match_feat_next_pyr = match_feat_next
        context_feat_curr_pyr = context_feat_curr
        
        # patch_coords: (B, N, 4) 分为两部分：当前帧和下一帧的归一化坐标
        coords_curr = patch_coords[:, :, :2]  # (B, N, 2)
        coords_next = patch_coords[:, :, 2:]   # (B, N, 2)
        
        # 从匹配特征中采样 patch 特征（利用双线性插值），并对每个 patch 进行平均池化得到向量表示
        patch_feat_curr = self.extract_patches(match_feat_curr_pyr, coords_curr, self.patch_size)  # (B, N, 64)
        patch_feat_next = self.extract_patches(match_feat_next_pyr, coords_next, self.patch_size)  # (B, N, 64)
        
        # 从上下文特征中采样 patch 特征（当前帧）
        patch_feat_context = self.extract_patches(context_feat_curr_pyr, coords_curr, self.patch_size)  # (B, N, 64)
        
        # 对 COR 进行线性投影（从 2D -> 64D）
        cor_proj = self.cor_proj(cor)  # (B, N, 64)
        
        # 构造匹配（相关性）特征：拼接当前帧、下一帧的匹配特征及 COR 投影，得到 (B, N, 192)
        correlation_features = torch.cat([patch_feat_curr, patch_feat_next, cor_proj], dim=-1)
        
        # 调用更新操作符，输出全局重投影修正和每个 patch 的置信权重
        global_correction, sigma = self.update_operator(correlation_features, patch_feat_context)
        
        return global_correction, sigma

    def extract_patches(self, feature_map, coords, patch_size):
        """
        利用双线性插值从特征图中提取 patch 特征，并对 patch 内特征进行平均池化。
        
        参数:
            feature_map: Tensor, 形状 (B, C, H, W)
            coords: Tensor, 形状 (B, N, 2) —— 每个坐标均为归一化坐标（范围 [-1,1]）
            patch_size: int, patch 的尺寸（例如 3）
            
        返回:
            patch_features: Tensor, 形状 (B, N, C)
        """
        B, C, H, W = feature_map.shape
        N = coords.shape[1]
        device = feature_map.device
        
        # 构造相对采样网格：在归一化坐标系下，一个 patch 覆盖的范围较小，
        # 此处取每个 patch 周围 ±(1/H) 的范围（近似一个像素的大小）
        delta = 1.0 / H
        lin_coords = torch.linspace(-delta, delta, steps=patch_size, device=device)
        grid_y, grid_x = torch.meshgrid(lin_coords, lin_coords, indexing='ij')
        grid = torch.stack((grid_x, grid_y), dim=-1)  # (patch_size, patch_size, 2)
        grid = grid.unsqueeze(0).unsqueeze(0)  # (1,1,patch_size,patch_size,2)
        grid = grid.expand(B, N, patch_size, patch_size, 2)  # (B, N, patch_size, patch_size,2)
        
        # 将中心坐标扩展为 (B, N, 1, 1, 2)
        coords = coords.unsqueeze(2).unsqueeze(3)
        sampling_grid = coords + grid  # (B, N, patch_size, patch_size,2)
        sampling_grid = sampling_grid.view(B * N, patch_size, patch_size, 2)
        
        # 将 feature_map 扩展到每个 patch
        feature_map_rep = feature_map.unsqueeze(1).expand(B, N, C, H, W).contiguous().view(B * N, C, H, W)
        patches = F.grid_sample(feature_map_rep, sampling_grid, align_corners=True)  # (B*N, C, patch_size, patch_size)
        # 对 patch 内进行平均池化
        patch_features = patches.view(B * N, C, -1).mean(dim=-1)  # (B*N, C)
        patch_features = patch_features.view(B, N, C)  # (B, N, C)
        return patch_features

########################################
# 捆绑调整损失：最小化全局修正与真实重投影结果之间的误差
########################################
def reprojection_loss(predicted_correction, cor_gt):
    """
    参数:
        predicted_correction (Tensor): 模型输出的全局重投影修正，形状 (B,2)
        cor_gt (Tensor): 每个样本中 patch 重投影结果的平均值，形状 (B,2)
    返回:
        loss (Tensor): 均方误差损失
    """
    loss = F.mse_loss(predicted_correction, cor_gt)
    return loss

########################################
# 训练函数
########################################
def train_vo_model(dataset_path, epochs=20, batch_size=32, learning_rate=1e-4):
    """
    训练单目视觉里程计模型（基于 patch 相关性和可微分束调整）
    
    参数:
        dataset_path (str): 数据集根目录路径
        epochs (int): 训练轮数
        batch_size (int): 批次大小
        learning_rate (float): 学习率
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = VODataset(dataset_path, num_patches=100)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = VOModel(patch_size=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        with tqdm(total=len(dataloader), desc=f'Epoch {epoch+1}/{epochs}', unit='batch') as pbar:
            for batch in dataloader:
                curr_img = batch['curr_img'].to(device)       # (B,3,256,256)
                next_img = batch['next_img'].to(device)
                patch_coords = batch['patch_coords'].to(device) # (B, N, 4)
                cor = batch['cor'].to(device)                   # (B, N, 2)
                
                optimizer.zero_grad()
                
                # 前向传播：输出全局重投影修正和每个 patch 的置信权重
                predicted_correction, sigma = model(curr_img, next_img, patch_coords, cor)
                
                # 以每个样本中所有 patch 的重投影结果均值作为 ground truth
                cor_gt = torch.mean(cor, dim=1)  # (B,2)
                loss = reprojection_loss(predicted_correction, cor_gt)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                pbar.update(1)
        
        print(f'Epoch [{epoch+1}/{epochs}], Avg Loss: {total_loss/len(dataloader):.4f}')
    
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
