# 视觉里程计（VO）项目

## 目录结构
```
VOproject/
├── network.py          # 主要的视觉里程计模型定义和训练脚本
├── network_ori.py      # 原始版本的网络定义（备份）
├── demo2.py            # 可视化特征点位移并生成视频
├── test.py             # 测试模型并可视化轨迹
├── P001/               # 示例数据集目录
└── vo_model.pth        # 训练好的模型权重文件
```

## 项目简介
本项目实现了基于深度学习的视觉里程计（Visual Odometry, VO），通过融合图像、深度图和光流信息来预测相机或车辆的相对位姿变化。项目包括数据预处理、模型训练、测试和结果可视化等功能。

## 环境配置

### 依赖项
确保已安装以下依赖库：
- Python 3.8+
- PyTorch 1.9+ (支持CUDA)
- OpenCV
- NumPy
- tqdm

### 数据准备
请将数据集放置在 `P001` 文件夹中，具体结构如下：
```
P001/
├── image_left/
│   ├── 000000_left.png
│   ├── 000001_left.png
│   └── ...
├── image_right/
│   ├── 000000_right.png
│   ├── 000001_right.png
│   └── ...
├── depth_left/
│   ├── 000000_left_depth.npy
│   ├── 000001_left_depth.npy
│   └── ...
├── depth_right/
│   ├── 000000_right_depth.npy
│   ├── 000001_right_depth.npy
│   └── ...
├── flow/
│   ├── 000000_flow.npy
│   ├── 000001_flow.npy
│   └── ...
└── pose_left.txt       # 包含每帧的位姿信息
```

## 使用说明

### 模型训练

运行以下命令开始训练模型：

```bash
python network.py
```

训练过程中会显示进度条，并在每个epoch结束后打印平均损失值。训练完成后，模型权重将保存为 `vo_model.pth`。

### 模型测试与可视化

#### 可视化轨迹

运行以下命令测试模型并可视化预测的轨迹：

```bash
python test.py
```

这将加载训练好的模型并绘制出预测的轨迹图。

#### 可视化特征点位移

运行以下命令生成特征点位移的视频：

```bash
python demo2.py
```

这将生成一个包含特征点位移箭头的视频文件 `feature_displacement_video2.avi`。

### 自定义参数

可以在 `network.py` 和 `test.py` 中修改以下参数以适应不同的需求：
- `dataset_path`: 数据集路径
- `epochs`: 训练轮数
- `batch_size`: 批次大小
- `learning_rate`: 学习率

## 代码结构

### `network.py`
- **VODataset**: 自定义数据集类，用于加载和预处理图像、深度图、光流和位姿数据。
- **UpdateOperator**: 更新操作符模块，结合卷积、自注意力机制和全连接层来生成位姿修正量和置信度权重。
- **VOModel**: 主要模型类，包含特征提取器、位姿回归网络和更新操作符。
- **train_vo_model**: 训练循环函数，使用束调整损失函数进行优化。
- **main**: 入口函数，调用训练函数。

### `test.py`
- **visualize_trajectory**: 绘制预测的轨迹图。
- **test_vo_model**: 测试模型并可视化位置偏移。
- **find_feature_matches**: 在两张图像之间找到相同的特征点并计算位移。
- **compute_reprojection_correction**: 计算重投影的改正值与权重。

### `demo2.py`
- **create_video_from_images**: 将图像文件夹中的图像创建为视频。
- **visualize_feature_displacement**: 可视化特征点位移并生成视频。

## 注意事项

1. **GPU 支持**：如果使用 GPU 进行训练，请确保 PyTorch 已正确安装 CUDA 支持。
2. **数据预处理**：确保数据集格式正确，特别是图像和深度图的命名和路径。
3. **模型保存**：训练后的模型权重会保存为 `vo_model.pth`，可以根据需要更改保存路径。

本仓库的network使用专门用于视觉里程计中的束调整（Bundle Adjustment），最小化重投影误差。我还提供了一个采用损失函数（位姿回归的MSE）的版本network_ori，用于自行参考和比对。


如果有任何问题或建议，请提交 GitHub Issue。
