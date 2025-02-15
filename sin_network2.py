import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
from tqdm import tqdm
import torch.nn.functional as F

#############################################
# 数据集定义：增加 patch 采样与 COR 模拟
#############################################
class DPVODataset(Dataset):
    def __init__(self, trajectory_dir, transform=None):
        """
        自定义单目视觉里程计数据集，额外采样图像块的位置和 COR（重投影结果）
        """
        self.trajectory_dir = trajectory_dir
        self.transform = transform
        
        # 加载左图与对应深度（使用连续帧对）
        self.left_images = sorted([f for f in os.listdir(os.path.join(trajectory_dir, 'image_left')) if f.endswith('_left.png')])
        self.left_depths = sorted([f for f in os.listdir(os.path.join(trajectory_dir, 'depth_left')) if f.endswith('_left_depth.npy')])
        self.poses = self._load_poses(os.path.join(trajectory_dir, 'pose_left.txt'))
    
    def _load_poses(self, pose_file):
        with open(pose_file, 'r') as f:
            poses = [list(map(float, line.strip().split())) for line in f]
        return np.array(poses)
    
    def __len__(self):
        return len(self.left_images) - 1
    
    def __getitem__(self, idx):
        # 加载当前帧和下一帧图像
        curr_img_path = os.path.join(self.trajectory_dir, 'image_left', self.left_images[idx])
        next_img_path = os.path.join(self.trajectory_dir, 'image_left', self.left_images[idx+1])
        curr_img = cv2.imread(curr_img_path)
        next_img = cv2.imread(next_img_path)
        
        # 加载深度和位姿（此处监督信号可保留）
        curr_depth_path = os.path.join(self.trajectory_dir, 'depth_left', self.left_depths[idx])
        curr_depth = np.load(curr_depth_path)
        curr_pose = self.poses[idx]
        next_pose = self.poses[idx+1]
        relative_pose = np.array(next_pose) - np.array(curr_pose)
        
        # 图像预处理：调整大小到256×256，归一化，并转换为 C×H×W
        curr_img = self._preprocess_image(curr_img)
        next_img = self._preprocess_image(next_img)
        
        # 随机采样图像块位置（避免边界区域）
        H, W = 256, 256
        margin = 16
        x = np.random.randint(margin, W - margin)
        y = np.random.randint(margin, H - margin)
        patch_pos_curr = np.array([x, y], dtype=np.float32)
        
        # 模拟真实的下一帧 patch 位置：添加一个随机偏移（模拟真实运动）
        gt_offset = np.random.uniform(-5, 5, size=(2,)).astype(np.float32)
        patch_pos_next = patch_pos_curr + gt_offset
        
        # 模拟重投影结果 COR：在当前帧位置上添加噪声（与真实偏移存在差异）
        noise = np.random.uniform(-2, 2, size=(2,)).astype(np.float32)
        cor = patch_pos_curr + noise
        
        # 拼接 patch 位置，格式为 [x_curr, y_curr, x_next, y_next]
        patch_positions = np.concatenate([patch_pos_curr, patch_pos_next], axis=0)
        # 真实的修正值（监督）：应使 COR 修正为真实位置，即 gt_corr = patch_pos_next - cor
        gt_corr = patch_pos_next - cor
        
        # 转换为张量
        curr_img = torch.from_numpy(curr_img).float()
        next_img = torch.from_numpy(next_img).float()
        curr_depth = torch.from_numpy(curr_depth).float()
        relative_pose = torch.from_numpy(relative_pose).float()
        patch_positions = torch.from_numpy(patch_positions).float()  # shape (4,)
        cor = torch.from_numpy(cor).float()  # shape (2,)
        gt_corr = torch.from_numpy(gt_corr).float()  # shape (2,)
        
        sample = {
            'curr_img': curr_img,
            'next_img': next_img,
            'curr_depth': curr_depth,
            'relative_pose': relative_pose,
            'patch_positions': patch_positions,
            'cor': cor,
            'gt_corr': gt_corr
        }
        return sample
    
    def _preprocess_image(self, img):
        img = cv2.resize(img, (256, 256))
        img = img / 255.0  # 归一化到 [0,1]
        img = np.transpose(img, (2, 0, 1))
        return img

#############################################
# 特征提取网络：匹配特征与上下文特征（两个残差结构示意）
#############################################
class MatchingFeatureExtractor(nn.Module):
    def __init__(self):
        super(MatchingFeatureExtractor, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class ContextFeatureExtractor(nn.Module):
    def __init__(self):
        super(ContextFeatureExtractor, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

#############################################
# 更新操作符：递归网络结合可微分束调整层
#############################################
class RecurrentBAUpdate(nn.Module):
    def __init__(self, hidden_dim=128):
        super(RecurrentBAUpdate, self).__init__()
        # 输入包括 patch 位置（当前帧和下一帧各 2 维，共 4 维）和 COR（2 维），共 6 维
        self.gru = nn.GRUCell(input_size=6, hidden_size=hidden_dim)
        self.transition = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        # 因子头：预测 2D 修正（改正值）
        self.factor_head_corr = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2)  # 输出 2D 改正值
        )
        # 因子头：预测置信权重（输出范围限定在 (0,1)）
        self.factor_head_sigma = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        # 可微分束调整层（示意）：这里简单用一个线性层，如果需要可以扩展为更复杂的 BA 优化模块
        self.ba_layer = nn.Linear(2, 7)
    
    def forward(self, patch_positions, cor, hidden_state):
        """
        参数:
            patch_positions: (B, 4) 当前帧与下一帧 patch 坐标 [x_curr, y_curr, x_next, y_next]
            cor: (B, 2) 重投影结果
            hidden_state: (B, hidden_dim)
        返回:
            correction: (B, 2) 2D 修正值
            sigma: (B, 1) 置信权重
            new_hidden: (B, hidden_dim) 更新后的隐藏状态
        """
        # 拼接输入特征
        x = torch.cat([patch_positions, cor], dim=1)  # (B,6)
        h = self.gru(x, hidden_state)  # 更新隐藏状态
        h_trans = self.transition(h)
        correction = self.factor_head_corr(h_trans)
        sigma = self.factor_head_sigma(h_trans)
        # 可选：通过束调整层进一步优化（此处示意，不参与后续 loss 计算）
        # updated_pose = self.ba_layer(correction)
        return correction, sigma, h

#############################################
# 单目视觉里程计模型：结合特征提取和递归更新操作符
#############################################
class RecurrentVOModel(nn.Module):
    def __init__(self, hidden_dim=128):
        super(RecurrentVOModel, self).__init__()
        self.matching_extractor = MatchingFeatureExtractor()
        self.context_extractor = ContextFeatureExtractor()
        # 构建特征金字塔（此处简单使用平均池化获得低分辨率特征，示意用途）
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        # 更新操作符：递归网络结合可微分束调整层
        self.recurrent_update = RecurrentBAUpdate(hidden_dim=hidden_dim)
    
    def forward(self, curr_img, next_img, patch_positions, cor, hidden_state=None):
        """
        参数:
            curr_img, next_img: (B, 3, 256, 256)
            patch_positions: (B, 4) [x_curr, y_curr, x_next, y_next]
            cor: (B, 2) 重投影结果
            hidden_state: (B, hidden_dim)，若为 None 则初始化为 0
        返回:
            correction: (B, 2) 2D 修正值
            sigma: (B, 1) 置信权重
            new_hidden: (B, hidden_dim)
        """
        # 提取匹配特征和上下文特征（后续可用于局部 patch 特征提取，此处仅做示意）
        matching_feat_curr = self.matching_extractor(curr_img)
        matching_feat_next = self.matching_extractor(next_img)
        context_feat_curr = self.context_extractor(curr_img)
        context_feat_next = self.context_extractor(next_img)
        # 构建特征金字塔（示意，未在更新中直接使用）
        context_feat_down = self.avg_pool(context_feat_curr)
        
        # 初始化隐藏状态
        if hidden_state is None:
            batch_size = curr_img.size(0)
            device = curr_img.device
            hidden_state = torch.zeros(batch_size, self.recurrent_update.gru.hidden_size, device=device)
        
        # 递归更新：这里示意为单步更新，可扩展为多步迭代
        correction, sigma, new_hidden = self.recurrent_update(patch_positions, cor, hidden_state)
        return correction, sigma, new_hidden

#############################################
# 损失函数定义：最小化预测的改正值与真实改正值之间的误差
#############################################
def update_loss(pred_corr, gt_corr, sigma):
    """
    使用加权均方误差作为更新损失。
    """
    loss = torch.mean(((pred_corr - gt_corr) ** 2) * sigma)
    return loss

#############################################
# 训练函数
#############################################
def train_vo_model(dataset_path, epochs=20, batch_size=32, learning_rate=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = DPVODataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    model = RecurrentVOModel(hidden_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        with tqdm(total=len(dataloader), desc=f'Epoch {epoch+1}/{epochs}', unit='batch') as pbar:
            for batch in dataloader:
                curr_img = batch['curr_img'].to(device)         # (B,3,256,256)
                next_img = batch['next_img'].to(device)
                patch_positions = batch['patch_positions'].to(device)  # (B,4)
                cor = batch['cor'].to(device)                     # (B,2)
                gt_corr = batch['gt_corr'].to(device)             # (B,2)
                
                optimizer.zero_grad()
                # 前向传播
                pred_corr, sigma, _ = model(curr_img, next_img, patch_positions, cor)
                loss = update_loss(pred_corr, gt_corr, sigma)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                pbar.update(1)
        print(f'Epoch [{epoch+1}/{epochs}], Avg Loss: {total_loss/len(dataloader):.4f}')
    
    torch.save(model.state_dict(), 'recurrent_vo_model.pth')
    print("训练完成并保存模型。")

#############################################
# 主函数入口
#############################################
def main():
    dataset_path = 'P001'  # 根据实际数据集路径修改
    train_vo_model(dataset_path, epochs=20, batch_size=16)

if __name__ == '__main__':
    main()
