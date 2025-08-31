# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 14:19:54 2024

@author: yisss
"""
# In[1] 1D approximate FFT
import cv2
import os
import numpy as np
from tqdm import tqdm  # 导入 tqdm 库
from Class_DIF_R22 import R22_DIF_FFT

N = 64

################ All 16 bits ##########################
# precision = '16bits'
# fixed_FFT = R22_DIF_FFT(N, 16, 16)
# In_n_word = 16
# Data_n_word = [16,16,16,16,16,16]
# TF_n_word = [16,16,16,16,16,16]
# fixed_FFT.TF_Gen()
#######################################################

# ############### 55 dB approximate FFT ###############
# precision = '55dB'
# fixed_FFT = R22_DIF_FFT(N, 16, 16)
# In_n_word = 12
# Data_n_word = [12,13,13,16,16,16]
# TF_n_word = [13,13,13,13,16,16]
# fixed_FFT.TF_Gen()
# #####################################################

############### 45 dB approximate FFT ###############
# precision = '45dB'
# fixed_FFT = R22_DIF_FFT(N, 16, 16)
# In_n_word = 10
# Data_n_word = [10,12,12,13,13,16]
# TF_n_word = [12,12,13,13,16,16]
# fixed_FFT.TF_Gen()
#####################################################

############### 35 dB approximate FFT ###############
# precision = '35dB'
# fixed_FFT = R22_DIF_FFT(N, 16, 16)
# In_n_word = 8
# Data_n_word = [8,11,11,13,13,16]
# TF_n_word = [10,10,10,10,16,16]
# fixed_FFT.TF_Gen()
#####################################################

############### 35 dB approximate FFT ###############
precision = '8bits'
fixed_FFT = R22_DIF_FFT(N, 16, 16)
In_n_word = 8
Data_n_word = [8,8,8,8,8,16]
TF_n_word = [8,8,8,8,8,8]
fixed_FFT.TF_Gen()
#####################################################

def img_1d_appr_fft(normalized_image):
    # Split the image into columns
    image_columns = [normalized_image[:, col] for col in range(normalized_image.shape[1])]

    # Perform fixed-point FFT on each column
    fft_results_fixed = []
    for column in image_columns:
        fixed_FFT.FFT_Fixed(column, In_n_word, Data_n_word, TF_n_word)
        fft_result_fixed = fixed_FFT.final_res_fxp * 64  # Multiply the result by 64
        fft_results_fixed.append(fft_result_fixed)   # fft using scaling methods

    ###############################################################    
    # 将列变换结果按列重新组合为一个矩阵
    fft_columns_combined = np.column_stack(fft_results_fixed)
    ###############################################################

    # Perform standard FFT on each row of the combined matrix using numpy's FFT
    fft_results_standard_rows = []
    for row in fft_columns_combined:
        fft_result_standard_row = np.fft.fft(row)  # Using numpy's standard FFT
        fft_results_standard_rows.append(fft_result_standard_row)

    ###############################################################
    # Combine the row FFT results into the final 2D FFT result
    fft_2d = np.vstack(fft_results_standard_rows)

    # 使用2D逆FFT还原图像
    reconstructed_image = np.fft.ifft2(fft_2d).real

    # Clip the reconstructed image to the valid range
    reconstructed_image = np.clip(reconstructed_image, 0, 1)
    
    return reconstructed_image

sub_images_folder = 'sub_images'
recon_images_folder = f'recon_images/{precision}_1D_appr'

# 创建重建图像文件夹（如果不存在）
os.makedirs(recon_images_folder, exist_ok=True)


# 读取和处理每个子图像
images_fft_ifft = []
for i in tqdm(range(16), desc='Processing Images', unit='image'):
    # 读取子图像
    # 生成子图像路径
    sub_img_path = os.path.join(sub_images_folder, f'sub_img_{i}.png')
    sub_img = cv2.imread(sub_img_path, cv2.IMREAD_GRAYSCALE)
    normalized_img = sub_img.astype(np.float32) / 255.0
    
    # 对每个子图像应用设计的2D FFT处理
    reconstructed_img = img_1d_appr_fft(normalized_img)
    
    # 保存重建的图像
    recon_img_path = os.path.join(recon_images_folder, f'recon_img_{i}.png')
    cv2.imwrite(recon_img_path, (reconstructed_img * 255).astype(np.uint8))
    
    # 将重建的图像添加到列表中
    images_fft_ifft.append(reconstructed_img)
#%%
images_fft_ifft = []
for i in range(16):
    # 读取重建的图像
    recon_img_path = os.path.join(recon_images_folder, f'recon_img_{i}.png')
    reconstructed_img = cv2.imread(recon_img_path, cv2.IMREAD_GRAYSCALE)
    norm_recon_img =  reconstructed_img.astype(np.float32) / 255.0
    
    # 将重建的图像添加到列表中
    images_fft_ifft.append(norm_recon_img)
    
    overlap_size = 16

    # 初始化重叠图像
    overlap_image = np.zeros((256, 256), dtype=np.float32)

    # 创建一个加权矩阵用于平滑过渡
    # weight_matrix = np.linspace(0, 1, 64)
    # weight_matrix = np.outer(weight_matrix, weight_matrix)


    # 对每个图像块进行拼接
    for i, image in enumerate(images_fft_ifft):
        # 将图像块加入重叠图像
        y_offset = (i // 4) * 64  # 计算垂直偏移量
        x_offset = (i % 4) * 64  # 计算水平偏移量
        overlap_image[y_offset:y_offset+64, x_offset:x_offset+64] += image#*weight_matrix
        
# 仅在64、128、192三条水平线附近进行平滑处理
smooth_range = 3  # 扩展平滑范围为上下各3个像素
for y in [64, 128, 192]:
    overlap_image[y-smooth_range:y+smooth_range+1, :] = (
        overlap_image[y-smooth_range-1:y+smooth_range, :] + overlap_image[y-smooth_range+1:y+smooth_range+2, :]
    ) / 2
        

# 将重叠图像裁剪为原始尺寸
final_image = overlap_image[:256, :256]

# 定义保存图像的文件夹路径
output_folder = 'final_images'

# 创建文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 定义要保存的文件名
final_image_filename = f'final_image_{precision}_1D_appr.png'

# 组合文件夹路径和文件名
final_image_path = os.path.join(output_folder, final_image_filename)

# 保存最终图像
cv2.imwrite(final_image_path, (final_image * 255).astype(np.uint8))

print(f"Final image saved to {final_image_path}")
##############

# # 保存最终图像
# cv2.imwrite('final_image.png', final_image * 255)

# 显示最终图像
cv2.imshow('Final Image', final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# In[2] approximate 2D 
import cv2
import os
import numpy as np
from tqdm import tqdm  # 导入 tqdm 库
from Class_DIF_R22 import R22_DIF_FFT

N = 64

################ All 16 bits ##########################
precision = '16bits'
fixed_FFT = R22_DIF_FFT(N, 16, 16)
In_n_word = 16
Data_n_word = [16,16,16,16,16,16]
TF_n_word = [16,16,16,16,16,16]
fixed_FFT.TF_Gen()
#######################################################

# ############### 55 dB approximate FFT ###############
# precision = '55dB'
# fixed_FFT = R22_DIF_FFT(N, 16, 16)
# In_n_word = 12
# Data_n_word = [12,13,13,16,16,16]
# TF_n_word = [13,13,13,13,16,16]
# fixed_FFT.TF_Gen()
# #####################################################

############### 45 dB approximate FFT ###############
# precision = '45dB'
# fixed_FFT = R22_DIF_FFT(N, 16, 16)
# In_n_word = 10
# Data_n_word = [10,12,12,13,13,16]
# TF_n_word = [12,12,13,13,16,16]
# fixed_FFT.TF_Gen()
#####################################################

############### 35 dB approximate FFT ###############
# precision = '35dB'
# fixed_FFT = R22_DIF_FFT(N, 16, 16)
# In_n_word = 8
# Data_n_word = [8,11,11,13,13,16]
# TF_n_word = [10,10,10,10,16,16]
# fixed_FFT.TF_Gen()
#####################################################

############### 35 dB approximate FFT ###############
# precision = '8bits'
# fixed_FFT = R22_DIF_FFT(N, 16, 16)
# In_n_word = 8
# Data_n_word = [8,8,8,8,8,16]
# TF_n_word = [8,8,8,8,8,8]
# fixed_FFT.TF_Gen()
# #####################################################


def img_2d_fft(normalized_image):
    # Split the image into columns
    image_columns = [normalized_image[:, col] for col in range(normalized_image.shape[1])]

    # Perform fixed-point FFT on each column
    fft_results_fixed = []
    for column in image_columns:
        fixed_FFT.FFT_Fixed(column, In_n_word, Data_n_word, TF_n_word)
        fft_result_fixed = fixed_FFT.final_res_fxp   #If use 1D approximate FFT, here times number of the fft points is very important! Because this is the true value!
        fft_results_fixed.append(fft_result_fixed)   # fft using scaling methods

    ###############################################################    
    # 将列变换结果按列重新组合为一个矩阵
    fft_columns_combined = np.column_stack(fft_results_fixed)
    ###############################################################

    # Perform fixed-point FFT on each row of the combined matrix
    fft_results_fixed_rows = []
    for row in fft_columns_combined:
        fixed_FFT.FFT_Fixed(row, In_n_word, Data_n_word, TF_n_word)
        fft_result_fixed_row = fixed_FFT.final_res_fxp
        fft_results_fixed_rows.append(fft_result_fixed_row)
    ###############################################################
    # # 对重新组合后的矩阵按行进行标准的1D FFT
    # fft_2d = np.fft.fft(fft_columns_combined, axis=1)  # 用近似FFT替换标准FFT
    
    # Combine the row FFT results into the final 2D FFT result
    fft_2d = np.vstack(fft_results_fixed_rows)
    fft_2d *= (64 * 64)

    # 使用2D逆FFT还原图像
    reconstructed_image = np.fft.ifft2(fft_2d).real
    
    
    ###############################################################
    # # Perform inverse FFT on fixed-point FFT results
    # inverse_fft_results = [np.fft.ifft(result).real for result in fft_results_fixed]

    # # Reconstruct the image from inverse FFT results
    # reconstructed_image = np.column_stack(inverse_fft_results)
    reconstructed_image = np.clip(reconstructed_image, 0, 1)
    
    return reconstructed_image

sub_images_folder = 'sub_images'
recon_images_folder = f'recon_images/{precision}_2D'

# 创建重建图像文件夹（如果不存在）
os.makedirs(recon_images_folder, exist_ok=True)


# 读取和处理每个子图像
images_fft_ifft = []
for i in tqdm(range(16), desc='Processing Images', unit='image'):
    # 读取子图像
    # 生成子图像路径
    sub_img_path = os.path.join(sub_images_folder, f'sub_img_{i}.png')
    sub_img = cv2.imread(sub_img_path, cv2.IMREAD_GRAYSCALE)
    normalized_img = sub_img.astype(np.float32) / 255.0
    
    # 对每个子图像应用设计的2D FFT处理
    reconstructed_img = img_2d_fft(normalized_img)
    
    # 保存重建的图像
    recon_img_path = os.path.join(recon_images_folder, f'recon_img_{i}.png')
    cv2.imwrite(recon_img_path, (reconstructed_img * 255).astype(np.uint8))
    
    # 将重建的图像添加到列表中
    images_fft_ifft.append(reconstructed_img)
    
#%%
images_fft_ifft = []
for i in range(16):
    # 读取重建的图像
    recon_img_path = os.path.join(recon_images_folder, f'recon_img_{i}.png')
    reconstructed_img = cv2.imread(recon_img_path, cv2.IMREAD_GRAYSCALE)
    norm_recon_img =  reconstructed_img.astype(np.float32) / 255.0
    
    # 将重建的图像添加到列表中
    images_fft_ifft.append(norm_recon_img)
    

# 定义重叠区域大小
overlap_size = 16

# 初始化重叠图像
overlap_image = np.zeros((256, 256), dtype=np.float32)

# 对每个图像块进行拼接
for i, image in enumerate(images_fft_ifft):
    # 将图像块加入重叠图像
    y_offset = (i // 4) * 64  # 计算垂直偏移量
    x_offset = (i % 4) * 64  # 计算水平偏移量
    overlap_image[y_offset:y_offset+64, x_offset:x_offset+64] += image

    
# # 仅在64、128、192三条水平线附近进行平滑处理
# smooth_range = 3  # 扩展平滑范围为上下各3个像素
# for y in [64, 128, 192]:
#     overlap_image[y-smooth_range:y+smooth_range+1, :] = (
#         overlap_image[y-smooth_range-1:y+smooth_range, :] + overlap_image[y-smooth_range+1:y+smooth_range+2, :]
#     ) / 2
    
# 将重叠图像裁剪为原始尺寸
final_image = overlap_image[:256, :256]

# 定义保存图像的文件夹路径
output_folder = 'final_images'

# 创建文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 定义要保存的文件名
final_image_filename = f'final_image_{precision}_2D.png'

# 组合文件夹路径和文件名
final_image_path = os.path.join(output_folder, final_image_filename)

# 保存最终图像
cv2.imwrite(final_image_path, (final_image * 255).astype(np.uint8))

print(f"Final image saved to {final_image_path}")
##############

# # 保存最终图像
# cv2.imwrite('final_image.png', final_image * 255)

# 显示最终图像
cv2.imshow('Final Image', final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
    
    
# In[3] New try for spectrum combinition        
import cv2
import os
import numpy as np
from tqdm import tqdm  # 导入 tqdm 库
from Class_DIF_R22 import R22_DIF_FFT

N = 64

################ All 16 bits ##########################
# precision = '16bits'
# fixed_FFT = R22_DIF_FFT(N, 16, 16)
# In_n_word = 16
# Data_n_word = [16,16,16,16,16,16]
# TF_n_word = [16,16,16,16,16,16]
# fixed_FFT.TF_Gen()
#######################################################

# ############### 55 dB approximate FFT ###############
# precision = '55dB'
# fixed_FFT = R22_DIF_FFT(N, 16, 16)
# In_n_word = 12
# Data_n_word = [12,13,13,16,16,16]
# TF_n_word = [13,13,13,13,16,16]
# fixed_FFT.TF_Gen()
# #####################################################

############### 45 dB approximate FFT ###############
# precision = '45dB'
# fixed_FFT = R22_DIF_FFT(N, 16, 16)
# In_n_word = 10
# Data_n_word = [10,12,12,13,13,16]
# TF_n_word = [12,12,13,13,16,16]
# fixed_FFT.TF_Gen()
#####################################################


############### 35 dB approximate FFT ###############
precision = '35dB'
fixed_FFT = R22_DIF_FFT(N, 16, 16)
In_n_word = 8
Data_n_word = [8,11,11,13,13,16]
TF_n_word = [10,10,10,10,16,16]
fixed_FFT.TF_Gen()
#####################################################

def img_2d_fft(normalized_image):
    # Perform 2D FFT on the image columns and rows using fixed-point FFT
    image_columns = [normalized_image[:, col] for col in range(normalized_image.shape[1])]

    fft_results_fixed = []
    for column in image_columns:
        fixed_FFT.FFT_Fixed(column, In_n_word, Data_n_word, TF_n_word)
        fft_result_fixed = fixed_FFT.final_res_fxp
        fft_results_fixed.append(fft_result_fixed)

    fft_columns_combined = np.column_stack(fft_results_fixed)

    fft_results_fixed_rows = []
    for row in fft_columns_combined:
        fixed_FFT.FFT_Fixed(row, In_n_word, Data_n_word, TF_n_word)
        fft_result_fixed_row = fixed_FFT.final_res_fxp
        fft_results_fixed_rows.append(fft_result_fixed_row)

    fft_2d = np.vstack(fft_results_fixed_rows)
    fft_2d *= (64 * 64)
    
    # The result is the fixed-point 2D FFT of a 64x64 image
    return fft_2d

sub_images_folder = 'sub_images'
recon_images_folder = 'recon_images/35dB_2D_new'
fft_spectra_folder = 'fft_spectra'  # 文件夹用于保存频谱图
# fft_combined_path = 'fft_combined.npy'  # 文件路径用于保存拼接后的频谱图

# 创建重建图像文件夹（如果不存在）
os.makedirs(recon_images_folder, exist_ok=True)
os.makedirs(fft_spectra_folder, exist_ok=True)

# 创建256x256频谱图
fft_256x256 = np.zeros((256, 256), dtype=np.complex64)

# 创建一个加权矩阵用于平滑过渡
weight_matrix = np.linspace(0, 1, 64)
weight_matrix = np.outer(weight_matrix, weight_matrix)

# 读取和处理每个子图像，并将它们的频谱拼接到256x256频谱图中
for i in tqdm(range(16), desc='Processing Images', unit='image'):
    # 读取子图像
    sub_img_path = os.path.join(sub_images_folder, f'sub_img_{i}.png')
    sub_img = cv2.imread(sub_img_path, cv2.IMREAD_GRAYSCALE)
    normalized_img = sub_img.astype(np.float32) / 255.0
    
    # 对每个子图像应用设计的2D FFT处理
    fft_sub_img = img_2d_fft(normalized_img)
    
    # # 将频谱图保存到文件中
    fft_sub_img_path = os.path.join(fft_spectra_folder, f'fft_sub_img_{i}.png')
    # np.save(fft_sub_img_path, fft_sub_img)
    
    # 根据索引将小频谱图拼接到完整的256x256频谱图中
    row_idx = (i // 4) * 64  # 计算子图像在256x256图中的行起始索引
    col_idx = (i % 4) * 64   # 计算子图像在256x256图中的列起始索引
    
    # fft_256x256[row_idx:row_idx+64, col_idx:col_idx+64] = fft_sub_img
    fft_256x256[row_idx:row_idx+64, col_idx:col_idx+64] += fft_sub_img * weight_matrix

# 保存拼接后的256x256频谱图
# np.save(fft_combined_path, fft_256x256)
# 对拼接后的256x256频谱图进行标准的2D逆变换
reconstructed_image = np.fft.ifft2(fft_256x256).real
reconstructed_image = np.clip(reconstructed_image, 0, 1)

# 保存最终的256x256重建图像
recon_img_path = os.path.join(recon_images_folder, 'recon_img_256x256.png')
reconstructed_image_final = (reconstructed_image * 255).astype(np.uint8)
cv2.imwrite(recon_img_path, reconstructed_image_final)
cv2.imshow('Reconstructed 256x256 Image', reconstructed_image_final)
cv2.waitKey(0)  # 等待用户按键关闭窗口
cv2.destroyAllWindows()