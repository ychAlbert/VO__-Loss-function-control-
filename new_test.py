import torch
import cv2
import numpy as np
from sin_network import VOModel

def load_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256))
    img = img / 255.0  # 归一化到 [0,1]
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).float().unsqueeze(0)  # 增加 batch 维度
    return img

def main():
    # 加载模型
    model = VOModel()
    model.load_state_dict(torch.load('vo_model.pth'))
    model.eval()

    # 加载测试图像
    curr_img_path = 'P001/image_left/000000_left.png'
    next_img_path = 'P001/image_left/000001_left.png'
    curr_img = load_image(curr_img_path)
    next_img = load_image(next_img_path)

    # 前向传播
    with torch.no_grad():
        updated_pose, sigma, pred_depth = model(curr_img, next_img)

    print("Updated Pose:", updated_pose)
    print("Sigma:", sigma)
    print("Predicted Depth:", pred_depth)

if __name__ == '__main__':
    main()
