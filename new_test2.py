import torch
import cv2
import numpy as np
from sin_network2 import RecurrentVOModel

def load_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256))
    img = img / 255.0  # 归一化到 [0,1]
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).float().unsqueeze(0)  # 增加 batch 维度
    return img

def test_vo_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RecurrentVOModel(hidden_dim=128).to(device)
    model.load_state_dict(torch.load('recurrent_vo_model.pth'))
    model.eval()

    # 加载测试图像
    curr_img = load_image('P001/image_left/000000_left.png').to(device)
    next_img = load_image('P001/image_left/000001_left.png').to(device)

    # 定义测试输入
    patch_positions = torch.tensor([[100.0, 100.0, 105.0, 105.0]], dtype=torch.float32).to(device)  # (B, 4)
    cor = torch.tensor([[102.0, 102.0]], dtype=torch.float32).to(device)  # (B, 2)
    hidden_state = torch.zeros(1, 128).to(device)  # (B, hidden_dim)

    # 前向传播
    with torch.no_grad():
        correction, sigma, _ = model(curr_img, next_img, patch_positions, cor, hidden_state)

    print(f'Correction: {correction.cpu().numpy()}')
    print(f'Sigma: {sigma.cpu().numpy()}')

if __name__ == '__main__':
    test_vo_model()
