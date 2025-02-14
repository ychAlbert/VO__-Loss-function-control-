import torch
import cv2
import numpy as np
from sin_network1 import VOModel

def test_vo_model():
    # 加载测试图像
    img1_path = 'P001/image_left/000000_left.png'
    img2_path = 'P001/image_left/000001_left.png'
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    # 预处理图像
    def preprocess_image(img):
        img = cv2.resize(img, (256, 256))
        img = img / 255.0
        img = np.transpose(img, (2, 0, 1))
        return img
    
    img1 = preprocess_image(img1)
    img2 = preprocess_image(img2)
    
    # 转换为 Tensor
    img1_tensor = torch.from_numpy(img1).unsqueeze(0).float()  # (1, 3, 256, 256)
    img2_tensor = torch.from_numpy(img2).unsqueeze(0).float()  # (1, 3, 256, 256)
    
    # 定义 patch 坐标和 COR
    patch_coords = torch.tensor([[[0.1, 0.1, 0.2, 0.2]]])  # (1, 1, 4)
    cor = torch.tensor([[[0.05, 0.05]]])  # (1, 1, 2)
    
    # 加载模型
    model = VOModel(patch_size=3)
    model.load_state_dict(torch.load('vo_model.pth'))
    model.eval()
    
    # 前向传播
    with torch.no_grad():
        global_correction, sigma = model(img1_tensor, img2_tensor, patch_coords, cor)
    
    print("Global Correction:", global_correction)
    print("Sigma:", sigma)

if __name__ == '__main__':
    test_vo_model()
