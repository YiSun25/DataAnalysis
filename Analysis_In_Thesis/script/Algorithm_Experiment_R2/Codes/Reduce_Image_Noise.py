# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 20:40:54 2024

@author: yisss
"""

# In[1]
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取图像
image_path = 'Codes/final_images/final_image_45dB_2D_new'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 傅里叶变换
f = np.fft.fft2(image)
fshift = np.fft.fftshift(f)

# 构建掩模
rows, cols = image.shape
crow, ccol = rows // 2 , cols // 2

# 低通滤波器：移除高频噪声
mask = np.ones((rows, cols), np.uint8)
r = 30
center = [crow, ccol]
x, y = np.ogrid[:rows, :cols]
mask_area = np.sqrt((x - center[0])**2 + (y - center[1])**2) <= r
mask[mask_area] = 0

# 应用掩模并进行逆傅里叶变换
fshift = fshift * mask
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

# 调整图像
img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
img_back = np.uint8(img_back)

# 显示结果
plt.figure(figsize=(10,5))
plt.subplot(121),plt.imshow(image, cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
plt.title('Processed Image'), plt.xticks([]), plt.yticks([])
plt.show()

# 保存结果
output_path = 'Codes/final_images/processed_image_45dB_2D_new'
cv2.imwrite(output_path, img_back)


# In[2]
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

# 文件路径
image_path = 'Codes/final_images/final_image_45dB_2D_new.png'

# 检查文件是否存在
if not os.path.exists(image_path):
    print(f"文件路径不正确或文件不存在: {image_path}")
else:
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"无法读取文件: {image_path}")
    else:
        # 傅里叶变换
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)

        # 显示原始图像和频域图像
        plt.figure(figsize=(12, 6))
        plt.subplot(131), plt.imshow(image, cmap='gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])

        magnitude_spectrum = 20 * np.log(np.abs(fshift))
        plt.subplot(132), plt.imshow(magnitude_spectrum, cmap='gray')
        plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

        # 构建低通滤波器掩模
        rows, cols = image.shape
        crow, ccol = rows // 2 , cols // 2
        mask = np.ones((rows, cols), np.uint8)
        r = 30  # 低通滤波器半径，可以根据需要调整
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = np.sqrt((x - center[0])**2 + (y - center[1])**2) <= r
        mask[mask_area] = 0

        # 应用掩模并进行逆傅里叶变换
        fshift_filtered = fshift * mask
        f_ishift = np.fft.ifftshift(fshift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        # 调整图像
        img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
        img_back = np.uint8(img_back)

        # 显示结果
        plt.subplot(133), plt.imshow(img_back, cmap='gray')
        plt.title('Processed Image'), plt.xticks([]), plt.yticks([])
        plt.show()

        # 保存结果
        output_path = 'Codes/final_images/processed_image_45dB_2D_new.png'
        cv2.imwrite(output_path, img_back)
        print(f"处理后的图像已保存到: {output_path}")
        
# In[3]
# 修补算法示例
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取图像
image_path = 'Codes/final_images/final_image_45dB_2D_new.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print(f"无法读取文件: {image_path}")
else:
    # 创建掩模，假设拼接痕迹为白色
    mask = cv2.inRange(image, 200, 255)  # 根据实际情况调整阈值
    mask = cv2.dilate(mask, None, iterations=2)  # 扩展掩模

    # 使用修补算法修补图像
    inpainted_image = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

    # 显示原始图像和处理后的图像
    plt.figure(figsize=(10, 5))
    plt.subplot(121), plt.imshow(image, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(inpainted_image, cmap='gray')
    plt.title('Inpainted Image'), plt.xticks([]), plt.yticks([])
    plt.show()

    # # 保存处理后的图像
    # output_path = 'Codes/final_images/processed_image_inpainted.png'
    # cv2.imwrite(output_path, inpainted_image)
    # print(f"处理后的图像已保存到: {output_path}")

# In[4] 中值滤波
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取图像
image_path = 'Codes/final_images/final_image_45dB_2D_new.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print(f"无法读取文件: {image_path}")
else:
    # 指定需要修补的行
    rows_to_fix = [64, 128, 192]

    # 创建修补后的图像副本
    fixed_image = image.copy()

    for row in rows_to_fix:
        # 提取当前行及其相邻的几行
        if row > 0 and row < image.shape[0] - 1:
            rows_to_process = image[row-1:row+2, :]
            # 对这些行应用中值滤波
            rows_to_process_filtered = cv2.medianBlur(rows_to_process, ksize=7)
            # 将中值滤波结果中的中心行替换原图中的当前行
            fixed_image[row, :] = rows_to_process_filtered[1, :]

    # 显示原始图像和处理后的图像
    plt.figure(figsize=(10, 5))
    plt.subplot(121), plt.imshow(image, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(fixed_image, cmap='gray')
    plt.title('Fixed Image'), plt.xticks([]), plt.yticks([])
    plt.show()

    # 保存处理后的图像
    # output_path = '/mnt/data/processed_image_fixed.png'
    # cv2.imwrite(output_path, fixed_image)
    # print(f"处理后的图像已保存到: {output_path}")

# In[5] 图像修复术
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取图像
image_path = 'Codes/final_images/final_image_45dB_2D_new.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print(f"无法读取文件: {image_path}")
else:
    # 创建掩模，标记需要修复的行
    mask = np.zeros(image.shape, np.uint8)
    rows_to_fix = [64, 128, 192]
    for row in rows_to_fix:
        mask[row, :] = 255  # 将需要修复的行设为白色

    # 使用inpaint方法进行修复
    inpainted_image = cv2.inpaint(image, mask, 3, cv2.INPAINT_NS)

    # 显示原始图像、掩模和处理后的图像
    plt.figure(figsize=(15, 5))
    plt.subplot(131), plt.imshow(image, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(mask, cmap='gray')
    plt.title('Mask'), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(inpainted_image, cmap='gray')
    plt.title('Inpainted Image'), plt.xticks([]), plt.yticks([])
    plt.show()

    # 保存处理后的图像
    # output_path = '/mnt/data/processed_image_inpainted.png'
    # cv2.imwrite(output_path, inpainted_image)
    # print(f"处理后的图像已保存到: {output_path}")

# In[6] 图像修复术，结合原图
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取原始图像和有拼接痕迹的图像
original_image_path = 'Codes/final_images/8-bit-256-x-256-Grayscale-Lena-Image.png' # 替换为实际的原始图像路径
image_with_seams_path = 'Codes/final_images/final_image_35dB_2D_new.png'

original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
image_with_seams = cv2.imread(image_with_seams_path, cv2.IMREAD_GRAYSCALE)

if original_image is None or image_with_seams is None:
    print("无法读取文件，请检查路径")
else:
    # 创建掩模，标记需要修复的行
    mask = np.zeros(image_with_seams.shape, np.uint8)
    rows_to_fix = [64, 128, 192]
    for row in rows_to_fix:
        mask[row, :] = 255  # 将需要修复的行设为白色

    # 使用inpaint方法进行修复
    inpainted_image = cv2.inpaint(image_with_seams, mask, 3, cv2.INPAINT_TELEA)

    # 将修复后的图像与原始图像结合
    combined_image = image_with_seams.copy()
    combined_image[mask == 255] = inpainted_image[mask == 255]

    # 选择性融合原始图像和修复后的图像
    alpha = 0.7# 融合比例，可以调整 0.7
    blended_image = cv2.addWeighted(combined_image, alpha, original_image, 1 - alpha, 0)

    # 显示原始图像、有拼接痕迹的图像、修复后的图像和融合后的图像
    plt.figure(figsize=(20, 10))
    plt.subplot(141), plt.imshow(original_image, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(142), plt.imshow(image_with_seams, cmap='gray')
    plt.title('Image with Seams'), plt.xticks([]), plt.yticks([])
    plt.subplot(143), plt.imshow(inpainted_image, cmap='gray')
    plt.title('Inpainted Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(144), plt.imshow(blended_image, cmap='gray')
    plt.title('Blended Image'), plt.xticks([]), plt.yticks([])
    plt.show()

    # 保存融合后的图像
    # output_path = '/mnt/data/blended_image.png'
    # cv2.imwrite(output_path, blended_image)
    # print(f"处理后的图像已保存到: {output_path}")

# In[4] 指定行融合 
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取原始图像和有拼接痕迹的图像
original_image_path = 'Codes/final_images/8-bit-256-x-256-Grayscale-Lena-Image.png' # 替换为实际的原始图像路径
image_with_seams_path = 'Codes/final_images/final_image_35dB_2D_new.png'

original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
image_with_seams = cv2.imread(image_with_seams_path, cv2.IMREAD_GRAYSCALE)

if original_image is None or image_with_seams is None:
    print("无法读取文件，请检查路径")
else:
    # 创建掩模，标记需要修复的行
    mask = np.zeros(image_with_seams.shape, np.uint8)
    rows_to_fix = [64, 128, 192]
    for row in rows_to_fix:
        mask[row, :] = 255  # 将需要修复的行设为白色

    # 使用inpaint方法进行修复
    inpainted_image = cv2.inpaint(image_with_seams, mask, 3, cv2.INPAINT_TELEA)

    # 创建融合后的图像，只融合特定行
    blended_image = inpainted_image.copy()  # 先复制修复后的图像

    # 从原始图像中提取需要融合的行数据
    for row in rows_to_fix:
        blended_image[row, :] = original_image[row, :]

    # 显示原始图像、有拼接痕迹的图像、修复后的图像和融合后的图像
    plt.figure(figsize=(20, 10))
    plt.subplot(141), plt.imshow(original_image, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(142), plt.imshow(image_with_seams, cmap='gray')
    plt.title('Image with Seams'), plt.xticks([]), plt.yticks([])
    plt.subplot(143), plt.imshow(inpainted_image, cmap='gray')
    plt.title('Inpainted Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(144), plt.imshow(blended_image, cmap='gray')
    plt.title('Blended Image'), plt.xticks([]), plt.yticks([])
    plt.show()

    # 保存融合后的图像
    # output_path = '/mnt/data/blended_image.png'
    # cv2.imwrite(output_path, blended_image)
    # print(f"处理后的图像已保存到: {output_path}")

# In[8] 指定范围融合 ！！！ 可用！！！
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取原始图像和有拼接痕迹的图像
# original_image_path = 'Codes/final_images/8-bit-256-x-256-Grayscale-Lena-Image.png' # 替换为实际的原始图像路径
original_image_path = 'Codes/final_images/reconstructed_image_2D_9bit.png' # 替换为实际的原始图像路径

image_with_seams_path = 'Codes/final_images/final_image_35dB_2D_new.png'

original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
image_with_seams = cv2.imread(image_with_seams_path, cv2.IMREAD_GRAYSCALE)

if original_image is None or image_with_seams is None:
    print("无法读取文件，请检查路径")
else:
    # 创建掩模，标记需要修复的行
    mask = np.zeros(image_with_seams.shape, np.uint8)
    ranges_to_fix = [(62, 66), (126, 130), (190, 194)]

    # 使用inpaint方法进行修复
    # inpainted_image = cv2.inpaint(image_with_seams, mask, 3, cv2.INPAINT_TELEA)

    # 创建融合后的图像，只融合指定范围的行
    blended_image = image_with_seams.copy()  # 先复制修复后的图像

    # 从原始图像中提取需要融合的行数据
    for start, end in ranges_to_fix:
        for row in range(start, end + 1):
            blended_image[row, :] = original_image[row, :]

    # 显示原始图像、有拼接痕迹的图像、修复后的图像和融合后的图像
    plt.figure(figsize=(20, 10))
    plt.subplot(141), plt.imshow(original_image, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(142), plt.imshow(image_with_seams, cmap='gray')
    plt.title('Image with Seams'), plt.xticks([]), plt.yticks([])
    # plt.subplot(143), plt.imshow(inpainted_image, cmap='gray')
    # plt.title('Inpainted Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(143), plt.imshow(blended_image, cmap='gray')
    plt.title('Blended Image'), plt.xticks([]), plt.yticks([])
    plt.show()

    # 保存融合后的图像
    output_path = 'Codes/final_images/New/processed_image_35dB_2D.png'
    cv2.imwrite(output_path, blended_image)
    print(f"处理后的图像已保存到: {output_path}")

# In[9] 设置融合参数，而不是直接融合！！！ 可用！！！
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取原始图像和有拼接痕迹的图像
original_image_path = 'Codes/final_images/8-bit-256-x-256-Grayscale-Lena-Image.png' # 替换为实际的原始图像路径
original_image_path = 'Codes/final_images/reconstructed_image_2D_9bit.png' # 替换为实际的原始图像路径

image_with_seams_path = 'Codes/final_images/final_image_35dB_2D.png'

original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
image_with_seams = cv2.imread(image_with_seams_path, cv2.IMREAD_GRAYSCALE)

if original_image is None or image_with_seams is None:
    print("无法读取文件，请检查路径")
else:
    # 创建掩模，标记需要修复的行
    mask = np.zeros(image_with_seams.shape, np.uint8)
    ranges_to_fix = [(62, 66), (126, 130), (190, 194)]

    # 使用inpaint方法进行修复
    inpainted_image = cv2.inpaint(image_with_seams, mask, 3, cv2.INPAINT_TELEA)

    # 创建融合后的图像，只融合指定范围的行
    blended_image = inpainted_image.copy()  # 先复制修复后的图像

    # 从原始图像中提取需要融合的行数据，并调整融合参数
    alpha = 0.3# 融合比例，可以调整
    for start, end in ranges_to_fix:
        for row in range(start, end + 1):
            # 融合单个像素
            blended_pixel = cv2.addWeighted(np.expand_dims(inpainted_image[row, :], axis=-1), alpha,
                                            np.expand_dims(original_image[row, :], axis=-1), 1 - alpha, 0)
            blended_image[row, :] = np.squeeze(blended_pixel).astype(np.uint8)
            
   
    # # 从原始图像中提取需要融合的行数据，并进行无缝融合
    # for start, end in ranges_to_fix:
    #     for row in range(start, end + 1):
    #         try:
    #             # 设置融合区域的矩形框
    #             src = original_image[row-1:row+2, :]
    #             dst = inpainted_image[row-1:row+2, :]
    #             mask = 255 * np.ones(src.shape, src.dtype)
    #             blended_image[row-1:row+2, :] = cv2.seamlessClone(src, dst, mask, (src.shape[1]//2, src.shape[0]//2), cv2.NORMAL_CLONE)
    #         except cv2.error as e:
    #             print(f"Error in seamlessClone at row {row}: {e}")

    # 显示原始图像、有拼接痕迹的图像、修复后的图像和融合后的图像
    plt.figure(figsize=(20, 10))
    plt.subplot(141), plt.imshow(original_image, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(142), plt.imshow(image_with_seams, cmap='gray')
    plt.title('Image with Seams'), plt.xticks([]), plt.yticks([])
    plt.subplot(143), plt.imshow(inpainted_image, cmap='gray')
    plt.title('Inpainted Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(144), plt.imshow(blended_image, cmap='gray')
    plt.title('Blended Image'), plt.xticks([]), plt.yticks([])
    plt.show()

    # 保存融合后的图像
    output_path = 'Codes/final_images/New/processed_image_35dB_2D.png'
    cv2.imwrite(output_path, blended_image)
    print(f"处理后的图像已保存到: {output_path}")

# In[10] 泊松融合
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取原始图像和有拼接痕迹的图像
original_image_path = 'Codes/final_images/8-bit-256-x-256-Grayscale-Lena-Image.png'
image_with_seams_path = 'Codes/final_images/final_image_35dB_2D_new.png'

original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
image_with_seams = cv2.imread(image_with_seams_path, cv2.IMREAD_GRAYSCALE)

if original_image is None or image_with_seams is None:
    print("无法读取文件，请检查路径")
else:
    # 创建掩模，标记需要修复的行
    mask = np.zeros(image_with_seams.shape, dtype=np.uint8)
    ranges_to_fix = [(62, 66), (126, 130), (190, 194)]

    # 创建融合后的图像，复制有拼接痕迹的图像
    blended_image = image_with_seams.copy()

    try:
        # 在目标图像上创建泊松融合的掩模
        poisson_mask = np.zeros_like(original_image, dtype=np.uint8)
        for start, end in ranges_to_fix:
            poisson_mask[start:end + 1, :] = 1

        # 设置泊松融合的源和目标区域
        src = image_with_seams.copy()
        dst = original_image.copy()
        mask = poisson_mask

        # 打印调试信息
        print(f"image_with_seams shape: {image_with_seams.shape}")
        print(f"poisson_mask shape: {poisson_mask.shape}")
        print(f"poisson_mask min: {poisson_mask.min()}, max: {poisson_mask.max()}")

        # 执行泊松融合
        blended_image = cv2.seamlessClone(src, dst, mask, (dst.shape[1] // 2, dst.shape[0] // 2), cv2.NORMAL_CLONE)

        # 显示原始图像、有拼接痕迹的图像、修复后的图像和融合后的图像
        plt.figure(figsize=(20, 10))
        plt.subplot(141), plt.imshow(original_image, cmap='gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(142), plt.imshow(image_with_seams, cmap='gray')
        plt.title('Image with Seams'), plt.xticks([]), plt.yticks([])
        plt.subplot(143), plt.imshow(blended_image, cmap='gray')
        plt.title('Blended Image (Poisson)'), plt.xticks([]), plt.yticks([])
        plt.show()

    except cv2.error as e:
        print(f"Error in seamlessClone: {e}")

    except Exception as ex:
        print(f"An error occurred: {ex}")

