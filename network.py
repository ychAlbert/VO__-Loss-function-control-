import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
from tqdm import tqdm  # 导入tqdm库用于进度条显示
import torch.nn.functional as F

#############################################
# 数据集定义（不变）
#############################################
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

#############################################
# 更新操作符：包含自注意力、束调整层及新增的深度预测头
#############################################
class UpdateOperator(nn.Module):
    def __init__(self):
        super(UpdateOperator, self).__init__()

        # 1D卷积层
        self.conv1d = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)

        # 自注意力机制
        self.self_attn = nn.MultiheadAttention(embed_dim=256, num_heads=8)

        # 过渡块：调整输入维度为256*4=1024（后续映射到64维特征）
        self.transition = nn.Sequential(
            nn.Linear(256 * 4, 128),  # 修正输入特征数
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        # 因子头：预测轨迹修正值
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

        # 束调整层：调整输出为7维以匹配relative_pose
        self.ba_layer = nn.Linear(7, 7)

        # 新增深度预测头：预测补丁的深度（这里输出7x7个深度值，后续reshape为低分辨率深度图）
        self.depth_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 49)  # 7x7 = 49
        )

    def forward(self, correlation_features, context_features):
        # 1D卷积
        x = self.conv1d(correlation_features)  # 输入形状：(batch, 128, N) → 输出形状：(batch, 256, N)

        # 自注意力机制
        x = x.permute(2, 0, 1)  # 调整为(seq_len, batch, embed_dim)
        x, _ = self.self_attn(x, x, x)  # 输出形状保持 (N, batch, 256)
        x = x.permute(1, 2, 0)  # 恢复为 (batch, 256, N)

        # 过渡块：展平后映射到64维
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1)  # (batch, 256*N)
        x = self.transition(x)  # 输出形状：(batch, 64)

        # 因子头：预测delta和sigma
        delta = self.factor_head_delta(x)
        sigma = self.factor_head_sigma(x)

        # 束调整层：输出更新后的位姿
        updated_pose = self.ba_layer(delta)

        # 新增深度预测头
        predicted_depth = self.depth_head(x)
        predicted_depth = predicted_depth.view(batch_size, 1, 7, 7)  # 重塑为 (batch, 1, 7, 7)

        return updated_pose, sigma, predicted_depth

#############################################
# 束调整损失函数：同时计算位姿和深度的重投影误差
#############################################
def bundle_adjustment_loss(predicted_pose, relative_pose, delta, sigma, predicted_depth=None, gt_depth=None):
    """
    束调整损失函数，最小化重投影误差，同时对深度进行监督

    参数:
        predicted_pose (torch.Tensor): 预测的位姿
        relative_pose (torch.Tensor): 相对位姿
        delta (torch.Tensor): 预测的轨迹修正值
        sigma (torch.Tensor): 置信权重
        predicted_depth (torch.Tensor, optional): 预测的低分辨率深度图 (batch, 1, 7, 7)
        gt_depth (torch.Tensor, optional): 真值深度图 (batch, H, W)

    返回:
        loss (torch.Tensor): 束调整损失
    """
    # 位姿重投影误差
    reprojection_error = predicted_pose - (relative_pose + delta)
    weighted_error = torch.sum((reprojection_error ** 2) * sigma, dim=1)
    pose_loss = torch.mean(weighted_error) * 1e6  # 放大损失值

    if predicted_depth is not None and gt_depth is not None:
        # 将真值深度下采样到 (7,7) 大小
        gt_depth_ds = F.adaptive_avg_pool2d(gt_depth.unsqueeze(1), (7, 7))
        depth_loss = F.mse_loss(predicted_depth, gt_depth_ds)
        loss = pose_loss + 1e3 * depth_loss  # 深度损失权重可根据实际情况调整
    else:
        loss = pose_loss

    return loss

#############################################
# 视觉里程计模型：支持训练和测试两种模式
#############################################
class VOModel(nn.Module):
    def __init__(self):
        super(VOModel, self).__init__()

        # 图像特征提取网络
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 深度特征提取（训练时使用监督深度）
        self.depth_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 光流特征提取（训练时使用预先计算的光流）
        self.flow_extractor = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 新增：测试时通过左右图像计算相关性得到光流特征
        self.flow_conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 新增：测试时通过图像特征动态估计深度特征（替代预先提供的深度）
        self.dynamic_depth_estimator = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # 降维层：拼接特征后降维
        # 训练时拼接：左图(128) + 右图(128) + 左深度(32) + 右深度(32) + 光流(64) = 384通道
        # 测试时拼接：左图(128) + 右图(128) + 左动态深度(32) + 右动态深度(32) + 光流(64) = 384通道
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
            nn.Linear(128 * 2 * 2, 512),  # 根据combined_conv输出尺寸调整
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 7)  # 3个平移，4个旋转参数（或其他表示）
        )

        # 更新操作符：包含差分束调整层和深度预测头
        self.update_operator = UpdateOperator()

    def compute_correlation(self, feat1, feat2):
        """
        计算左右图像特征之间的相关性（简单的逐元素乘积求和）
        参数:
            feat1, feat2: (batch, C, H, W)
        返回:
            correlation_features: (batch, 64, H', W')
        """
        b, c, h, w = feat1.size()
        # 将特征展平 (batch, C, H*W)
        feat1_flat = feat1.view(b, c, -1)
        feat2_flat = feat2.view(b, c, -1)
        # 逐元素乘积后沿通道求和得到相关性 (batch, H*W)
        correlation = (feat1_flat * feat2_flat).sum(dim=1)
        correlation = correlation.view(b, 1, h, w)
        # 利用卷积层映射到64通道
        correlation_features = self.flow_conv(correlation)
        return correlation_features

    def forward(self, left_img, right_img, left_depth=None, right_depth=None, flow=None):
        """
        参数:
            left_img, right_img: (batch, 3, H, W)
            left_depth, right_depth: (batch, H, W) —— 训练时提供，测试时可为None
            flow: (batch, H, W, 2) —— 训练时提供，测试时可为None
        """
        # 提取左右图像特征
        left_img_features = self.feature_extractor(left_img)
        right_img_features = self.feature_extractor(right_img)

        if self.training:
            # 训练模式：使用提供的深度和光流
            left_depth_features = self.depth_extractor(left_depth.unsqueeze(1))
            right_depth_features = self.depth_extractor(right_depth.unsqueeze(1))
            flow_features = self.flow_extractor(flow.permute(0, 3, 1, 2))  # 调整光流维度
        else:
            # 测试模式：动态估计深度和光流
            flow_features = self.compute_correlation(left_img_features, right_img_features)
            left_depth_features = self.dynamic_depth_estimator(left_img_features)
            right_depth_features = self.dynamic_depth_estimator(right_img_features)

        # 为确保各分支特征尺寸一致，取最小尺寸
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

        # 展平后预测位姿
        flattened = combined_features.view(combined_features.size(0), -1)
        pose = self.pose_regressor(flattened)

        # 为更新操作符准备特征
        correlation_features = combined_features.view(combined_features.size(0), combined_features.size(1), -1)
        context_features = flattened.unsqueeze(2)
        updated_pose, sigma, predicted_depth = self.update_operator(correlation_features, context_features)

        return updated_pose, sigma, predicted_depth

#############################################
# 训练函数
#############################################
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

                # 前向传播（训练时传入深度和光流）
                predicted_pose, sigma, predicted_depth = model(left_img, right_img, left_depth, right_depth, flow)

                # 计算束调整损失（这里delta暂时用 predicted_pose - relative_pose 表示）
                loss = bundle_adjustment_loss(predicted_pose, relative_pose, predicted_pose - relative_pose, sigma,
                                              predicted_depth, left_depth)
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
