# -*- coding: utf-8 -*-
"""
Created on Mon May 20 15:39:11 2024

@author: yisss
"""

# 将256*256 图像切割成 16个 64*64图像并保存
import numpy as np
import cv2
import os

print("Current working directory:", os.getcwd())

# 使用绝对路径
folder_path = 'images'
image_name = '8-bit-256-x-256-Grayscale-Lena-Image.png'
image_path = os.path.abspath(os.path.join(folder_path, image_name))

# 打印完整路径
print(f"Full image path: {image_path}")

# 读取图像
src = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 检查图像是否正确读取
if src is None:
    print(f"Error: Could not read the image from path: {image_path}")
else:
    print("Image read successfully.")
    print(f"Image shape: {src.shape}")

    # 定义切割参数
    sub_image_num = 4  # 切割成4x4小块
    sub_images = []
    src_height, src_width = src.shape[0], src.shape[1]
    sub_height = src_height // sub_image_num
    sub_width = src_width // sub_image_num

    # 切割图像
    for j in range(sub_image_num):
        for i in range(sub_image_num):
            # 计算每个子图像的边界
            start_row = j * sub_height
            end_row = (j + 1) * sub_height if j < sub_image_num - 1 else src_height
            start_col = i * sub_width
            end_col = (i + 1) * sub_width if i < sub_image_num - 1 else src_width
            
            # 提取子图像
            image_roi = src[start_row:end_row, start_col:end_col]
            sub_images.append(image_roi)

    # 保存子图像
    for i, img in enumerate(sub_images):
        sub_image_path = os.path.join(folder_path, f'sub_img_{i}.png')
        cv2.imwrite(sub_image_path, img)
        print(f"Saved sub-image {i} at: {sub_image_path}")
        

# sub images reconstruction        
# 读取子图像
loaded_sub_images = []
for i in range(sub_image_num * sub_image_num):
    sub_image_path = os.path.join(folder_path, f'sub_img_{i}.png')
    img = cv2.imread(sub_image_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        loaded_sub_images.append(img)
    else:
        print(f"Error: Could not read the sub-image from path: {sub_image_path}")

# 检查是否读取了所有子图像
if len(loaded_sub_images) != sub_image_num * sub_image_num:
    print(f"Error: Expected {sub_image_num * sub_image_num} sub-images but read {len(loaded_sub_images)}")
else:
    # 拼接子图像
    rows = []
    for j in range(sub_image_num):
        row = np.hstack(loaded_sub_images[j * sub_image_num:(j + 1) * sub_image_num])
        rows.append(row)
    reconstructed_image = np.vstack(rows)

    # 检查重建图像的尺寸
    print(f"Reconstructed image shape: {reconstructed_image.shape}")

    # 保存重建的图像
    reconstructed_image_path = os.path.join(folder_path, 'reconstructed_image.png')
    cv2.imwrite(reconstructed_image_path, reconstructed_image) #, [cv2.IMWRITE_PNG_COMPRESSION, 0])  add these content, the reconstructed image will have higher quality
    print(f"Reconstructed image saved at: {reconstructed_image_path}")                               # 0 -- 9, 9 is the largest compression

    # 显示重建的图像
    cv2.imshow('Reconstructed Image', reconstructed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    
    
    





"""
cv2.IMREAD_COLOR：加载彩色图片，这个是默认参数，可以直接写1。
cv2.IMREAD_GRAYSCALE：以灰度模式加载图片，可以直接写0。
cv2.IMREAD_UNCHANGED：包括alpha(包括透明度通道)，可以直接写-1
"""