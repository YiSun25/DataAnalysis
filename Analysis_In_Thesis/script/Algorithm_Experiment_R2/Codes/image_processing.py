# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 19:38:41 2024

@author: yisss
"""
# In[1]
from Class_DIF_R2 import R2_DIF_FFT
from fxpmath import Fxp
import os
import numpy as np
import cv2

N = 64

fixed_FFT = R2_DIF_FFT(N,16,16)

# image 
# image path
folder_path = 'Codes/images/'
# image name
image_name = '8-bit-256-x-256-Grayscale-Lena-Image.png'
# Build complete image paths
image_path = folder_path + image_name


# Read the image and convert it to a grey scale image
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 获取图像的大小
height, width = image.shape

block_size = 64

# 归一化图像灰度值到0到1之间
normalized_image = image / 255.0

# 将纵向像素进行定点化
fixed_point_columns = []
for col in range(normalized_image.shape[1]):
    # 获取当前列的像素值
    column_pixels = normalized_image[:, col]
    # # 将归一化后的像素值转换为定点数表示
    # fixed_point_column = Fxp(normalized_column, True, 16, 15) # n_word = 16, n_frac = 15
    fixed_point_columns.append(column_pixels)
    
# 在定点FFT函数中处理纵向像素数据
fft_results = []
Data_n_word = [10, 11, 12, 13, 14, 16]
TF_n_word = [11, 16, 16, 16, 16, 16]
for column in fixed_point_columns:
    fft_result = fixed_FFT.FFT_Fixed(column_pixels, 8, Data_n_word, TF_n_word)
    fft_results.append(fft_result)

# 对FFT操作后的结果进行逆FFT变换
inverse_fft_results = []
for result in fft_results:
    inverse_fft_result =  np.fft.ifft(fft_result)
    inverse_fft_results.append(inverse_fft_result)
    
# 将逆变换后的纵向像素数据与原始图像的横向像素数据进行平均处理
restored_image = np.mean([inverse_fft_result.np_object() + normalized_image[:, idx] for idx, inverse_fft_result in enumerate(inverse_fft_results)], axis=0)

# 显示还原后的图像
cv2.imshow('Restored Image', (restored_image * 255).astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()


# # show the image
# cv2.imshow('Image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# In[2]

from Codes.Class_DIF_R2 import R2_DIF_FFT
from fxpmath import Fxp
import os
import numpy as np
import cv2

N = 256

fixed_FFT = R2_DIF_FFT(N,16,16)
In_n_word = 16
Data_n_word = [16,16,16,16,16]
TF_n_word = [16,16,16,16,16,16]

# image 
# image path
folder_path = 'Codes/images/'
# image name
image_name = '8-bit-256-x-256-Grayscale-Lena-Image.png'
# Build complete image paths
image_path = folder_path + image_name


# Read the image and convert it to a grey scale image
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 获取图像的大小
height, width = image.shape

block_size = 64

# 定义存储处理后图像的数组
processed_image = np.zeros_like(image)

# 对图像进行分块处理
for i in range(0, height, block_size):
    for j in range(0, width, block_size):
        # 获取当前小块的像素数据
        block_pixels = image[i:i+block_size, j:j+block_size]
        
        # 将像素值归一化到 0 到 1 之间
        normalized_block = block_pixels / 255.0
        
        # 对每列的像素进行近似定点 FFT 正变换
        fft_result = np.zeros_like(normalized_block, dtype=np.complex128)
        for col in range(block_size):
            fixed_FFT.TF_Gen()
            fft_result[:, col] = fixed_FFT.FFT_Fixed(normalized_block[:,col], In_n_word, Data_n_word, TF_n_word) #custom_forward_fft(normalized_block[:, col])
        
        # 对得到的频域数据进行相应的处理（此处省略）
        
        # 对处理后的频域数据进行逆变换
        processed_block = np.zeros_like(normalized_block)
        for col in range(block_size):
            processed_block[:, col] = np.fft.ifft(fft_result[:, col]).real
        
        # 将处理后的小块放回图像中相应的位置
        processed_image[i:i+block_size, j:j+block_size] = processed_block

# 显示处理后的图像
cv2.imshow('Processed Image', processed_image.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()




# In[3] !!! Can be used !!!(but not very optimal, lack "time 256") [5] and [6] are better
from Codes.Class_DIF_R2 import R2_DIF_FFT
from fxpmath import Fxp
import os
import numpy as np
import cv2

N = 256

fixed_FFT = R2_DIF_FFT(N,6,6)
In_n_word = 6
Data_n_word = [6,6,6,6,6,6,6,6]
TF_n_word = [6,6,6,6,6,6,6,6]
fixed_FFT.TF_Gen()

# image 
# image path
folder_path = 'Codes/images/'
# image name
image_name = '8-bit-256-x-256-Grayscale-Lena-Image.png'
# Build complete image paths
image_path = folder_path + image_name


# Read the image and convert it to a grey scale image
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 将像素值归一化到 0 到 1 之间
normalized_image = image / 255.0

image_columns = []
for col in range(normalized_image.shape[1]):
    column_pixels = normalized_image[:,col]
    image_columns.append(column_pixels)

fft_results = []
for column in image_columns:
    
    fixed_FFT.FFT_Fixed(column, In_n_word, Data_n_word, TF_n_word)
    fft_result = fixed_FFT.final_res_fxp
    fft_results.append(fft_result)
    
inverse_fft_results = []
for result in fft_results:
    inverse_fft_result = np.fft.ifft(fft_result)
    inverse_fft_results.append(inverse_fft_result)
    
# 将逆变换后的纵向像素数据与原始图像的横向像素数据进行结合
combined_pixels = []
for row, inverse_fft_result in zip(range(normalized_image.shape[0]), inverse_fft_results):
    # 将逆变换后的纵向像素数据与原始图像的横向像素数据进行结合
    combined_row = (inverse_fft_result + normalized_image[row, :]) / 2
    combined_pixels.append(combined_row)

# 将纵向像素与横向像素结合后的数据重新组合成图像
result_image = np.array(combined_pixels)

# 显示还原后的图像
cv2.imshow('Restored Image', (result_image * 255).astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
    

# # 对每列的像素进行 1D 近似定点 FFT 正变换
# fft_result = fixed_FFT.FFT_Fixed(normalized_image, In_n_word, Data_n_word, TF_n_word)

# # 对得到的频域数据进行相应的处理（例如频域滤波、频域增强等）（此处省略）

# # 对处理后的频域数据进行 1D 近似定点 FFT 逆变换
# processed_image = np.fft.ifft(fft_result, axis=0).real

# # 显示处理后的图像
# cv2.imshow('Processed Image', (processed_image * 255).astype(np.uint8))
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# In[4] generate black figure
import numpy as np
import cv2
from Codes.Class_DIF_R2 import R2_DIF_FFT


N = 256

fixed_FFT = R2_DIF_FFT(N, 16, 16)
In_n_word = 16
Data_n_word = [16, 16, 16, 16, 16, 16, 16, 16]
TF_n_word = [16, 16, 16, 16, 16, 16, 16, 16]
fixed_FFT.TF_Gen()

# image 
# image path
folder_path = 'Codes/images/'
# image name
image_name = '8-bit-256-x-256-Grayscale-Lena-Image.png'
# Build complete image paths
image_path = folder_path + image_name

# Read the image and convert it to a grey scale image
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 将像素值归一化到 0 到 1 之间
normalized_image = image / 255.0

fft_results = []
for col in range(normalized_image.shape[1]):
    column_pixels = normalized_image[:, col]
    list_column = list(column_pixels)
    
    fixed_FFT.FFT_Fixed(list_column, In_n_word, Data_n_word, TF_n_word)
    fft_result = fixed_FFT.final_res_fxp
    fft_results.append(fft_result)

inverse_fft_results = []
for fft_result in fft_results:
    inverse_fft_result = np.fft.ifft(fft_result).real  # 取实部
    inverse_fft_results.append(inverse_fft_result)

# 将逆变换后的数据重新组合成图像
result_image = np.vstack(inverse_fft_results)

# 显示还原后的图像
cv2.imshow('Restored Image', (result_image * 255).astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()

# In[5] !!!work successfully!!!!  column transformation
import numpy as np
import cv2
from fxpmath import Fxp
from Codes.Class_DIF_R2 import R2_DIF_FFT

N = 256

fixed_FFT = R2_DIF_FFT(N, 9, 9)
In_n_word = 9
Data_n_word = [9, 9, 9, 9, 9, 9, 9, 9]
TF_n_word = [9, 9, 9, 9, 9, 9, 9, 9]
fixed_FFT.TF_Gen()

# image 
folder_path = 'Codes/images/'
image_name = '8-bit-256-x-256-Grayscale-Lena-Image.png'
image_path = folder_path + image_name

# Read and normalize the image
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
normalized_image = image.astype(np.float32) / 255.0

# Split the image into columns
image_columns = [normalized_image[:, col] for col in range(normalized_image.shape[1])]

# Perform fixed-point FFT on each column
fft_results_fixed = []
for column in image_columns:
    fixed_FFT.FFT_Fixed(column, In_n_word, Data_n_word, TF_n_word)
    fft_result_fixed = fixed_FFT.final_res_fxp*256                          # Here times 256 is very important! Because this is the true value!
    fft_results_fixed.append(fft_result_fixed)

# Perform inverse FFT on fixed-point FFT results
inverse_fft_results = [np.fft.ifft(result).real for result in fft_results_fixed]

# Reconstruct the image from inverse FFT results
reconstructed_image = np.column_stack(inverse_fft_results)
reconstructed_image = np.clip(reconstructed_image, 0, 1)

# Display the original and reconstructed images
cv2.imshow('Original Image', (normalized_image * 255).astype(np.uint8))
cv2.imshow('Reconstructed Image', (reconstructed_image * 255).astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save reconstructed image
reconstructed_image_path = os.path.join(folder_path, 'reconstructed_image_try.png')
cv2.imwrite(reconstructed_image_path, (reconstructed_image * 255).astype(np.uint8))




# In[6] row transformation
import numpy as np
import cv2
from fxpmath import Fxp
from Codes.Class_DIF_R2 import R2_DIF_FFT

N = 256

fixed_FFT = R2_DIF_FFT(N, 8, 8)
In_n_word = 8
Data_n_word = [8, 8, 8, 8, 8, 8, 8, 8]
TF_n_word = [8, 8, 8, 8, 8, 8, 8, 8]
fixed_FFT.TF_Gen()


# Image path
folder_path = 'Codes/images/'
image_name = '8-bit-256-x-256-Grayscale-Lena-Image.png'
image_path = folder_path + image_name
# image_path = os.path.join(folder_path, image_name)

# Read and normalize the image
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
normalized_image = image.astype(np.float32) / 255.0

# Split the image into rows
image_rows = [normalized_image[row, :] for row in range(normalized_image.shape[0])]

# Placeholder for FFT results
fft_results_fixed = []

# Perform fixed-point FFT on each row
for row in image_rows:
    # Assuming fixed_FFT.FFT_Fixed modifies fixed_FFT.final_res_fxp
    fixed_FFT.FFT_Fixed(row, In_n_word, Data_n_word, TF_n_word)
    fft_result_fixed = fixed_FFT.final_res_fxp * 256  # Adjust scaling as necessary
    fft_results_fixed.append(fft_result_fixed)

# Perform inverse FFT on fixed-point FFT results
inverse_fft_results = [np.fft.ifft(result).real for result in fft_results_fixed]

# Reconstruct the image from inverse FFT results
reconstructed_image = np.row_stack(inverse_fft_results)
reconstructed_image = np.clip(reconstructed_image, 0, 1)

# Display the original and reconstructed images
cv2.imshow('Original Image', (normalized_image * 255).astype(np.uint8))
cv2.imshow('Reconstructed Image', (reconstructed_image * 255).astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save reconstructed image
reconstructed_image_path = os.path.join(folder_path, 'reconstructed_image_try.png')
cv2.imwrite(reconstructed_image_path, (reconstructed_image * 255).astype(np.uint8))


# In[7] # Try 2D FFT #
import numpy as np
import cv2
from fxpmath import Fxp
from Codes.Class_DIF_R2 import R2_DIF_FFT

N = 256

fixed_FFT = R2_DIF_FFT(N, 16, 16)
In_n_word = 16
Data_n_word = [16, 16, 16, 16, 16, 16, 16, 16]
TF_n_word = [16, 16, 16, 16, 16, 16, 16, 16]
fixed_FFT.TF_Gen()

# normalization
def normalize(data, factor):
    return data / factor

# denormalization
def denormalize(data, factor):
    return data * factor


# In[8] 2D FFT, 1D with approximate fft, 1D with np.fft.fft
import numpy as np
import cv2
import os
from fxpmath import Fxp
from Codes.Class_DIF_R2 import R2_DIF_FFT

N = 256

fixed_FFT = R2_DIF_FFT(N, 9, 9)
In_n_word = 9
Data_n_word = [9, 9, 9, 9, 9, 9, 9, 9]
TF_n_word = [9, 9, 9, 9, 9, 9, 9, 9]
fixed_FFT.TF_Gen()

# image 
folder_path = 'Codes/images/'
image_name = '8-bit-256-x-256-Grayscale-Lena-Image.png'
image_path = folder_path + image_name

# Read and normalize the image
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
normalized_image = image.astype(np.float32) / 255.0

# Split the image into columns
image_columns = [normalized_image[:, col] for col in range(normalized_image.shape[1])]

# Perform fixed-point FFT on each column
fft_results_fixed = []
for column in image_columns:
    fixed_FFT.FFT_Fixed(column, In_n_word, Data_n_word, TF_n_word)
    fft_result_fixed = fixed_FFT.final_res_fxp*256                          # Here times 256 is very important! Because this is the true value!
    fft_results_fixed.append(fft_result_fixed)

###############################################################    
# 将列变换结果按列重新组合为一个矩阵
fft_columns_combined = np.column_stack(fft_results_fixed)
###############################################################

###############################################################
# 对重新组合后的矩阵按行进行标准的1D FFT
fft_2d = np.fft.fft(fft_columns_combined, axis=1)

# 使用2D逆FFT还原图像
reconstructed_image = np.fft.ifft2(fft_2d).real
###############################################################
# # Perform inverse FFT on fixed-point FFT results
# inverse_fft_results = [np.fft.ifft(result).real for result in fft_results_fixed]

# # Reconstruct the image from inverse FFT results
# reconstructed_image = np.column_stack(inverse_fft_results)
reconstructed_image = np.clip(reconstructed_image, 0, 1)

# Display the original and reconstructed images
cv2.imshow('Original Image', (normalized_image * 255).astype(np.uint8))
cv2.imshow('Reconstructed Image', (reconstructed_image * 255).astype(np.uint8))

cv2.waitKey(0)
cv2.destroyAllWindows()

# Save reconstructed image
reconstructed_image_path = os.path.join(folder_path, 'reconstructed_image_2D_9bit.png')
cv2.imwrite(reconstructed_image_path, (reconstructed_image * 255).astype(np.uint8))

# In[1D approximate FFT]
import cv2
import os
import numpy as np
from Codes.Class_DIF_R2 import R2_DIF_FFT

N = 64

################ All 16 bits ##########################
# fixed_FFT = R2_DIF_FFT(N, 16, 16)
# In_n_word = 16
# Data_n_word = [16, 16, 16, 16, 16, 16]
# TF_n_word = [16, 16, 16, 16, 16, 16]
# fixed_FFT.TF_Gen()
#######################################################


# ############### 55 dB approximate FFT ###############
# fixed_FFT = R2_DIF_FFT(N, 16, 16)
# In_n_word = 12
# Data_n_word = [13, 14, 15, 16, 16, 16]
# TF_n_word = [14, 14, 16, 16, 16, 16]
# fixed_FFT.TF_Gen()
# #####################################################


############### 45 dB approximate FFT ###############
# fixed_FFT = R2_DIF_FFT(N, 16, 16)
# In_n_word = 10
# Data_n_word = [11, 12, 16, 16, 16, 16]
# TF_n_word = [16, 15, 16, 16, 16, 16]
# fixed_FFT.TF_Gen()
#####################################################


############### 35 dB approximate FFT ###############
fixed_FFT = R2_DIF_FFT(N, 16, 16)
In_n_word = 8
Data_n_word = [10, 11, 12, 14, 14, 16]
TF_n_word = [13, 14, 13, 16, 16, 16]
fixed_FFT.TF_Gen()
#####################################################


def img_2d_fft(normalized_image):
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

sub_images_folder = 'Codes/sub_images'
recon_images_folder = 'Codes/recon_images/35dB_1D_new'

# 创建重建图像文件夹（如果不存在）
os.makedirs(recon_images_folder, exist_ok=True)


# 读取和处理每个子图像
images_fft_ifft = []
for i in range(16):
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

# In[10] 16个64*64图像的图像处理,并拼接，消除块效应!!!
import cv2
import os
import numpy as np
from Codes.Class_DIF_R2 import R2_DIF_FFT

N = 64

################ All 16 bits ##########################
# fixed_FFT = R2_DIF_FFT(N, 16, 16)
# In_n_word = 16
# Data_n_word = [16, 16, 16, 16, 16, 16]
# TF_n_word = [16, 16, 16, 16, 16, 16]
# fixed_FFT.TF_Gen()
#######################################################


# ############### 55 dB approximate FFT ###############
# fixed_FFT = R2_DIF_FFT(N, 16, 16)
# In_n_word = 12
# Data_n_word = [13, 14, 15, 16, 16, 16]
# TF_n_word = [14, 14, 16, 16, 16, 16]
# fixed_FFT.TF_Gen()
# #####################################################


############### 45 dB approximate FFT ###############
# fixed_FFT = R2_DIF_FFT(N, 16, 16)
# In_n_word = 10
# Data_n_word = [11, 12, 16, 16, 16, 16]
# TF_n_word = [16, 15, 16, 16, 16, 16]
# fixed_FFT.TF_Gen()
#####################################################


############### 35 dB approximate FFT ###############
fixed_FFT = R2_DIF_FFT(N, 16, 16)
In_n_word = 8
Data_n_word = [10, 11, 12, 14, 14, 16]
TF_n_word = [13, 14, 13, 16, 16, 16]
fixed_FFT.TF_Gen()
#####################################################


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

sub_images_folder = 'Codes/sub_images'
recon_images_folder = 'Codes/recon_images/35dB_2D_new'

# 创建重建图像文件夹（如果不存在）
os.makedirs(recon_images_folder, exist_ok=True)


# 读取和处理每个子图像
images_fft_ifft = []
for i in range(16):
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

# folder_path = 'Codes/images/'
# image name
# image_name = '8-bit-256-x-256-Grayscale-Lena-Image.png'
# # Build complete image paths
# image_path = folder_path + image_name

# # Read the image and convert it to a grey scale image
# original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# # 检查图像是否成功加载
# if original_image is None:
#     print("Failed to load original image ")
# else:
#     print("Original image loaded successfully")
# 读取重建的图像块并存储在列表中
images_fft_ifft = []
for i in range(16):
    # 读取重建的图像
    recon_img_path = os.path.join(recon_images_folder, f'recon_img_{i}.png')
    reconstructed_img = cv2.imread(recon_img_path, cv2.IMREAD_GRAYSCALE)
    norm_recon_img =  reconstructed_img.astype(np.float32) / 255.0
    
    # 将重建的图像添加到列表中
    images_fft_ifft.append(norm_recon_img)
    

# 双边滤波函数
def apply_bilateral_filter(image, d, sigma_color, sigma_space):
    filtered_image = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    return filtered_image


############## 平滑过渡法  ############################
# # 确定图像块的数量
# num_blocks = len(images_fft_ifft)
# block_size = 64
# overlap_size = 4  # 重叠区域的大小
# blocks_per_row = 4  # 每行图像块的数量

# # 计算重叠图像的大小
# rows = (num_blocks + blocks_per_row - 1) // blocks_per_row  # 向上取整
# overlap_image_height = rows * (block_size - overlap_size) + overlap_size
# overlap_image_width = blocks_per_row * (block_size - overlap_size) + overlap_size

# # 初始化重叠图像和权重图像
# overlap_image = np.zeros((overlap_image_height, overlap_image_width))
# weight_image = np.zeros((overlap_image_height, overlap_image_width))

# # 创建权重矩阵
# weight_matrix = np.ones((block_size, block_size))
# # 创建平滑的过渡区域
# smooth_transition = np.linspace(0, 1, overlap_size)
# weight_matrix[:overlap_size, :] *= smooth_transition[:, None]
# weight_matrix[-overlap_size:, :] *= smooth_transition[::-1][:, None]
# weight_matrix[:, :overlap_size] *= smooth_transition[None, :]
# weight_matrix[:, -overlap_size:] *= smooth_transition[None, ::-1]

# # 对每个图像块进行拼接
# for i, image in enumerate(images_fft_ifft):
#     y_offset = (i // blocks_per_row) * (block_size - overlap_size)  # 计算垂直偏移量
#     x_offset = (i % blocks_per_row) * (block_size - overlap_size)  # 计算水平偏移量
    
#     # 将图像块加入重叠图像，并在重叠区域进行加权平均
#     overlap_image[y_offset:y_offset + block_size, x_offset:x_offset + block_size] += image * weight_matrix
#     weight_image[y_offset:y_offset + block_size, x_offset:x_offset + block_size] += weight_matrix

# # 避免除以零
# weight_image[weight_image == 0] = 1

# # 归一化处理
# overlap_image /= weight_image
######################################################################################


#####################################平滑过渡法，高斯权重矩阵 #######################
# # 确定图像块的数量
# block_size = 64
# overlap_size = 16  # 重叠区域的大小（增加宽度以获得更平滑的过渡）
# blocks_per_row = 4  # 每行图像块的数量
# num_blocks = blocks_per_row * blocks_per_row  # 假设图像块的数量是16个

# # # 分割图像为块
# # images_fft_ifft = [image[y:y+block_size, x:x+block_size] for y in range(0, image.shape[0], block_size) for x in range(0, image.shape[1], block_size)]

# # 计算重叠图像的大小
# rows = (num_blocks + blocks_per_row - 1) // blocks_per_row  # 向上取整
# overlap_image_height = rows * (block_size - overlap_size) + overlap_size
# overlap_image_width = blocks_per_row * (block_size - overlap_size) + overlap_size

# # 初始化重叠图像和权重图像
# overlap_image = np.zeros((overlap_image_height, overlap_image_width))
# weight_image = np.zeros((overlap_image_height, overlap_image_width))

# # 创建高斯权重矩阵
# def create_gaussian_weight_matrix(block_size, overlap_size, sigma=5):
#     weight_matrix = np.ones((block_size, block_size))
#     gaussian_1d = cv2.getGaussianKernel(overlap_size*2, sigma)
#     gaussian_2d = gaussian_1d @ gaussian_1d.T
#     weight_matrix[:overlap_size, :overlap_size] = gaussian_2d[overlap_size:, overlap_size:]
#     weight_matrix[-overlap_size:, :overlap_size] = gaussian_2d[:overlap_size, overlap_size:]
#     weight_matrix[:overlap_size, -overlap_size:] = gaussian_2d[overlap_size:, :overlap_size]
#     weight_matrix[-overlap_size:, -overlap_size:] = gaussian_2d[:overlap_size, :overlap_size]
#     return weight_matrix

# weight_matrix = create_gaussian_weight_matrix(block_size, overlap_size)

# # 对每个图像块进行拼接
# for i, image in enumerate(images_fft_ifft):
#     y_offset = (i // blocks_per_row) * (block_size - overlap_size)  # 计算垂直偏移量
#     x_offset = (i % blocks_per_row) * (block_size - overlap_size)  # 计算水平偏移量
    
#     # 将图像块加入重叠图像，并在重叠区域进行加权平均
#     overlap_image[y_offset:y_offset + block_size, x_offset:x_offset + block_size] += image * weight_matrix
#     weight_image[y_offset:y_offset + block_size, x_offset:x_offset + block_size] += weight_matrix

# # 避免除以零
# weight_image[weight_image == 0] = 1

# # 归一化处理
# overlap_image /= weight_image
###########################################################################################


########################## previous useful method ######################################
# 定义重叠区域大小
overlap_size = 16

# 初始化重叠图像
overlap_image = np.zeros((256, 256), dtype=np.float32)

# # 创建一个加权矩阵用于平滑过渡
# weight_matrix = np.linspace(0, 1, 64)
# weight_matrix = np.outer(weight_matrix, weight_matrix)


# 对每个图像块进行拼接
for i, image in enumerate(images_fft_ifft):
    # 将图像块加入重叠图像
    y_offset = (i // 4) * 64  # 计算垂直偏移量
    x_offset = (i % 4) * 64  # 计算水平偏移量
    overlap_image[y_offset:y_offset+64, x_offset:x_offset+64] += image #* weight_matrix
    
    
  ############################################################################################  
    
    
    
    
# # 确保图像大小和类型一致
# assert original_image.shape == overlap_image.shape, "Images must have the same dimensions"

# # 定义要融合的行索引
# rows_to_blend = [64, 128, 192]

# # 创建掩码
# mask = np.zeros_like(original_image, dtype=np.uint8)
# for row in rows_to_blend:
#     mask[row:row+1, :] = 255

# # 执行泊松融合
# result = cv2.seamlessClone(original_image, overlap_image, mask, (0, 0), cv2.NORMAL_CLONE)

# #显示或保存结果
# cv2.imshow('Blended Image', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# rows_of_interest = [64, 128, 192]  # 感兴趣的行索引

# for row in rows_of_interest:
#     # 从原始图像中提取当前行的像素
#     original_row = original_image[row, :]
    
#     # 将原始行像素复制到 overlap_image 对应的行上
#     overlap_image[row, :] = original_row  # 直接替换整行的像素值
    
# # 仅在64、128、192三条水平线附近进行简单平滑处理
# for y in [64, 128, 192]:
#     overlap_image[y-1:y+2, :] = (overlap_image[y-2:y+1, :] + overlap_image[y:y+3, :]) / 2.0
    
# 仅在64、128、192三条水平线附近进行平滑处理
smooth_range = 3  # 扩展平滑范围为上下各3个像素
for y in [64, 128, 192]:
    overlap_image[y-smooth_range:y+smooth_range+1, :] = (
        overlap_image[y-smooth_range-1:y+smooth_range, :] + overlap_image[y-smooth_range+1:y+smooth_range+2, :]
    ) / 2
    
# smooth_range_gaussian = 1  # 扩展高斯平滑范围为上下各5个像素
# for y in [64, 128, 192]:
#     overlap_image[y-smooth_range_gaussian:y+smooth_range_gaussian, :] = cv2.GaussianBlur(
#         overlap_image[y-smooth_range_gaussian:y+smooth_range_gaussian, :], (3, 3), 0
#    )


# smooth_range = 3  # 扩展平滑范围为上下各5个像素
# for y in [64, 128, 192]:
#     overlap_image[y-smooth_range:y+smooth_range+1, :] = apply_bilateral_filter(
#         overlap_image[y-smooth_range:y+smooth_range+1, :], d=1, sigma_color=20, sigma_space=20
#     )
    
    # # Apply edge smoothing
    # if i % 4 != 0:
    #     overlap_image[y_offset:y_offset+64, x_offset:x_offset+overlap_size] /= 2
    # if i >= 4:
    #     overlap_image[y_offset:y_offset+overlap_size, x_offset:x_offset+64] /= 2

# final_image = apply_bilateral_filter(overlap_image, d=9, sigma_color=20, sigma_space=20)

# Apply Gaussian smoothing
# final_image = cv2.GaussianBlur(overlap_image, (3, 3), 0)  # change number to change gaussian kernel

# # Clip the final image to [0, 1]
# final_image = np.clip(final_image, 0, 1)
# final_image = apply_bilateral_filter(overlap_image, d=3, sigma_color=10, sigma_space=10)






# 将重叠图像裁剪为原始尺寸
final_image = overlap_image[:256, :256]

# 定义保存图像的文件夹路径
output_folder = 'Codes/final_images'

# 创建文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 定义要保存的文件名
final_image_filename = 'final_image_35dB_2D_new.png'

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

# In[11] another method to reduce the blocking effect, 权重矩阵
# Read and process each sub-image
images_fft_ifft = []
for i in range(16):
    # Read the sub-image
    sub_img_path = os.path.join(sub_images_folder, f'sub_img_{i}.png')
    sub_img = cv2.imread(sub_img_path, cv2.IMREAD_GRAYSCALE)
    normalized_img = sub_img.astype(np.float32) / 255.0
    
    # Apply the designed 2D FFT process to each sub-image
    reconstructed_img = img_2d_fft(normalized_img)
    
    # Save the reconstructed image
    recon_img_path = os.path.join(recon_images_folder, f'recon_img_{i}.png')
    cv2.imwrite(recon_img_path, (reconstructed_img * 255).astype(np.uint8))
    
    # Add the reconstructed image to the list
    images_fft_ifft.append(reconstructed_img)

# Initialize the overlap image and weight matrix
overlap_image = np.zeros((256, 256), dtype=np.float32)
weight_matrix = np.zeros((256, 256), dtype=np.float32)

# Define the weight mask for blending
# weight_mask = np.ones((64, 64), dtype=np.float32)
# for i in range(64):
#     for j in range(64):
#         weight_mask[i, j] = min(i / 63.0, j / 63.0, (63 - i) / 63.0, (63 - j) / 63.0)

# Define the weight masks for blending
x = np.linspace(0, np.pi, 64)
weight_mask_x = np.sin(x) ** 2
weight_mask_y = weight_mask_x.reshape(-1, 1)
weight_mask = weight_mask_y @ weight_mask_x.reshape(1, -1)

# Combine images with gradient weights at the edges
for i, image in enumerate(images_fft_ifft):
    y_offset = (i // 4) * 64
    x_offset = (i % 4) * 64
    
    overlap_image[y_offset:y_offset+64, x_offset:x_offset+64] += image * weight_mask
    weight_matrix[y_offset:y_offset+64, x_offset:x_offset+64] += weight_mask

# Normalize the overlap image by the weight matrix
overlap_image /= weight_matrix

# Clip the final image to [0, 1]
final_image = np.clip(overlap_image, 0, 1)

# Save the final image
output_folder = 'Codes/final_images'
os.makedirs(output_folder, exist_ok=True)
final_image_filename = 'final_image_45dB_2D.png'
final_image_path = os.path.join(output_folder, final_image_filename)
cv2.imwrite(final_image_path, (final_image * 255).astype(np.uint8))

print(f"Final image saved to {final_image_path}")

# Display the final image
cv2.imshow('Final Image', final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# In[12] 高斯函数生成权重
# Read and process each sub-image
images_fft_ifft = []
for i in range(16):
    # Read the sub-image
    sub_img_path = os.path.join(sub_images_folder, f'sub_img_{i}.png')
    sub_img = cv2.imread(sub_img_path, cv2.IMREAD_GRAYSCALE)
    normalized_img = sub_img.astype(np.float32) / 255.0
    
    # Apply the designed 2D FFT process to each sub-image
    reconstructed_img = img_2d_fft(normalized_img)
    
    # Save the reconstructed image
    recon_img_path = os.path.join(recon_images_folder, f'recon_img_{i}.png')
    cv2.imwrite(recon_img_path, (reconstructed_img * 255).astype(np.uint8))
    
    # Add the reconstructed image to the list
    images_fft_ifft.append(reconstructed_img)

# Initialize the overlap image and weight matrix
overlap_image = np.zeros((256, 256), dtype=np.float32)
weight_matrix = np.zeros((256, 256), dtype=np.float32)

# Define the weight masks for blending 高斯
# def create_weight_mask(size):
#     # Create a Gaussian weight mask
#     sigma = size / 10.0
#     gauss_1d = np.exp(-0.5 * (np.linspace(-5, 5, size) / sigma) ** 2)
#     weight_mask = np.outer(gauss_1d, gauss_1d)
#     return weight_mask / weight_mask.max()  # Normalize to have max value 1

# 双线性
def create_weight_mask(size):
    linear_1d = np.linspace(0, 1, size)
    weight_mask = np.outer(linear_1d, linear_1d) * np.outer(1 - linear_1d, 1 - linear_1d)
    return weight_mask / weight_mask.max()  # Normalize to have max value 1


# Combine images with gradient weights at the edges
for i, image in enumerate(images_fft_ifft):
    y_offset = (i // 4) * 64
    x_offset = (i % 4) * 64

    weight_mask = create_weight_mask(64)
    
    overlap_image[y_offset:y_offset+64, x_offset:x_offset+64] += image * weight_mask
    weight_matrix[y_offset:y_offset+64, x_offset:x_offset+64] += weight_mask

# Normalize the overlap image by the weight matrix
with np.errstate(divide='ignore', invalid='ignore'):
    overlap_image = np.true_divide(overlap_image, weight_matrix)
    overlap_image[~np.isfinite(overlap_image)] = 0  # -inf inf NaN

# Clip the final image to [0, 1]
final_image = np.clip(overlap_image, 0, 1)

# Save the final image
output_folder = 'Codes/final_images'
os.makedirs(output_folder, exist_ok=True)
final_image_filename = 'final_image_45dB_2D.png'
final_image_path = os.path.join(output_folder, final_image_filename)
cv2.imwrite(final_image_path, (final_image * 255).astype(np.uint8))

print(f"Final image saved to {final_image_path}")

# Display the final image
cv2.imshow('Final Image', final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# In[13] # Poisson Image Editing
import numpy as np
import cv2
import os

# 读取图像块的文件夹路径
recon_images_folder = 'Codes/recon_images/8bit'


# 读取16个图像块并添加到列表中
images_fft_ifft = []
for i in range(16):
    recon_img_path = os.path.join(recon_images_folder, f'recon_img_{i}.png')
    reconstructed_img = cv2.imread(recon_img_path, cv2.IMREAD_GRAYSCALE)
    
    normalized_img = reconstructed_img.astype(np.float32) / 255.0
    images_fft_ifft.append(normalized_img)

# 初始化最终图像
final_image = np.zeros((256, 256), dtype=np.float32)

# 拼接图像块
for i, image in enumerate(images_fft_ifft):
    y_offset = (i // 4) * 64
    x_offset = (i % 4) * 64

    # 将图像块粘贴到最终图像中
    final_image[y_offset:y_offset+64, x_offset:x_offset+64] = image
    

# 使用泊松融合消除拼接痕迹
result = cv2.seamlessClone(
    (final_image * 255).astype(np.uint8),   # 源图像
    (final_image * 255).astype(np.uint8),   # 目标图像
    np.full((256, 256), 255, dtype=np.uint8),  # 掩码
    (128, 128),  # 中心位置
    cv2.MIXED_CLONE
)

# 保存最终图像
output_folder = 'Codes/final_images'
os.makedirs(output_folder, exist_ok=True)
final_image_path = os.path.join(output_folder, 'final_image_try.png')
cv2.imwrite(final_image_path, result)

print(f"Final image saved to {final_image_path}")

# 显示最终图像
cv2.imshow('Final Image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# In[14]
import cv2
import numpy as np

# 读取图像
folder_path = 'Codes/images/'
image_name = '8-bit-256-x-256-Grayscale-Lena-Image.png'
image_path = folder_path + image_name
image = cv2.imread(image_path)


# 应用高斯模糊
blurred_image = cv2.GaussianBlur(image, (3, 3), 0)  # 参数 (15, 15) 是模糊核的大小，可以根据需要调整

# # 添加高斯噪声
# mean = 0
# sigma = 25  # 噪声的标准差，可以根据需要调整
# gauss_noise = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
# noisy_image = cv2.add(blurred_image, gauss_noise)  # 将高斯噪声添加到模糊图像中

# 显示原始图像、模糊后的图像和添加噪声后的图像
cv2.imshow('Original Image', image)
cv2.imshow('Blurred Image', blurred_image)
# cv2.imshow('Noisy Image', noisy_image)

output_path = 'Codes/final_images/blur_image2.png'
cv2.imwrite(output_path, blurred_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# In[15]
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取原始图像
original_image_path = 'Codes/images/8-bit-256-x-256-Grayscale-Lena-Image.png'
original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)

if original_image is None:
    print("无法读取原始图像，请检查路径。")
else:
    # 进行FFT
    f_original = np.fft.fft2(original_image)
    fshift_original = np.fft.fftshift(f_original)

    # 生成35dB信噪比的噪声
    mean = 0
    sigma = 0.06 * np.max(original_image)  # 计算标准差
    gauss_noise = np.random.normal(mean, sigma, original_image.shape).astype(np.float64)

    # 添加噪声到频域
    f_noisy = fshift_original + gauss_noise

    # 逆FFT
    f_noisy_shifted = np.fft.ifftshift(f_noisy)
    noisy_image = np.abs(np.fft.ifft2(f_noisy_shifted))

    # 显示原始图像和生成的带噪声的图像
    plt.figure(figsize=(10, 6))
    plt.subplot(121), plt.imshow(original_image, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(noisy_image, cmap='gray')
    plt.title('Noisy Image (approx. 35dB SNR)'), plt.xticks([]), plt.yticks([])
    plt.show()

# In[16]
import cv2

# 读取图像
image_path = 'Codes/final_images/final_image_35dB_2D.png'
image = cv2.imread(image_path)

# # 创建掩膜，将需要模糊的行标记为白色（255）
# mask = np.zeros(image.shape[:2], dtype=np.uint8)
# rows_to_blur = [64, 128, 192]
# for row in rows_to_blur:
#     mask[row, :] = 255

# # 应用高斯模糊到标记的区域
# # 应用高斯模糊到标记的区域
# blurred_image = image.copy()
# blurred_image[mask == 255] = cv2.GaussianBlur(blurred_image[mask == 255], (15, 15), 0)

# image = cv2.imread(image_path)

if image is None:
    print("无法读取文件，请检查路径")
else:
    # 创建模糊后的图像
    blurred_image = image.copy()

    # 指定需要模糊的行范围
    ranges_to_blur = [(62, 66), (126, 130), (190, 194)]

    for start, end in ranges_to_blur:
        for row in range(start, end + 1):
            # 对指定行的每一行应用高斯模糊
            blurred_image[row, :] = cv2.GaussianBlur(image[row-1:row+2, :], (7, 7), 0)[1, :]


# 显示原始图像和模糊后的图像
cv2.imshow('Original Image', image)
cv2.imshow('Blurred Image', blurred_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# In[17] New Try
import cv2
import os
import numpy as np
from Codes.Class_DIF_R2 import R2_DIF_FFT

N = 64

################ All 16 bits ##########################
# fixed_FFT = R2_DIF_FFT(N, 16, 16)
# In_n_word = 16
# Data_n_word = [16, 16, 16, 16, 16, 16]
# TF_n_word = [16, 16, 16, 16, 16, 16]
# fixed_FFT.TF_Gen()
#######################################################


# ############### 55 dB approximate FFT ###############
# fixed_FFT = R2_DIF_FFT(N, 16, 16)
# In_n_word = 12
# Data_n_word = [13, 14, 15, 16, 16, 16]
# TF_n_word = [14, 14, 16, 16, 16, 16]
# fixed_FFT.TF_Gen()
# #####################################################


############### 45 dB approximate FFT ###############
# fixed_FFT = R2_DIF_FFT(N, 16, 16)
# In_n_word = 10
# Data_n_word = [11, 12, 16, 16, 16, 16]
# TF_n_word = [16, 15, 16, 16, 16, 16]
# fixed_FFT.TF_Gen()
#####################################################


############### 35 dB approximate FFT ###############
fixed_FFT = R2_DIF_FFT(N, 16, 16)
In_n_word = 8
Data_n_word = [10, 11, 12, 14, 14, 16]
TF_n_word = [13, 14, 13, 16, 16, 16]
fixed_FFT.TF_Gen()
#####################################################


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

sub_images_folder = 'Codes/sub_images'
recon_images_folder = 'Codes/recon_images/35dB_2D_new'

# 创建重建图像文件夹（如果不存在）
os.makedirs(recon_images_folder, exist_ok=True)


# 读取和处理每个子图像
images_fft_ifft = []
for i in range(16):
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


# In[18]
"""
import cv2
import numpy as np
from scipy.fft import fft, ifft

# 读取图像并将其转换为灰度图像
image = cv2.imread('example_image.jpg', cv2.IMREAD_GRAYSCALE)

# 对图像的纵向像素进行FFT变换
fft_result = np.fft.fft(image, axis=0)

# 将FFT变换后的结果进行逆变换，恢复纵向像素
inverse_fft_result = np.fft.ifft(fft_result, axis=0).real

# 将逆变换后的纵向像素数据与原始图像的横向像素数据进行结合
combined_image = np.hstack((image[:, :image.shape[1]//2], inverse_fft_result[:, image.shape[1]//2:]))

# 显示还原后的图像
cv2.imshow('Restored Image', combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()



###################################################################################
import cv2
import numpy as np
from fxpmath import Fxp

# 读取图像并将其转换为灰度图像
image = cv2.imread('example_image.jpg', cv2.IMREAD_GRAYSCALE)

# 归一化图像灰度值到0到1之间
normalized_image = image / 255.0

# 定义定点数位数
integer_bits = 8
fractional_bits = 8

# 只对图像的纵向像素进行定点化
fixed_point_columns = []
for col in range(normalized_image.shape[1]):
    # 获取当前列的像素值
    column_pixels = normalized_image[:, col]
    # 将像素值转换为定点数表示
    fixed_point_column = Fxp(column_pixels, signed=True, n_word=integer_bits+fractional_bits, n_frac=fractional_bits)
    fixed_point_columns.append(fixed_point_column)

# 在定点FFT函数中处理定点化后的纵向像素数据
fft_results = []
for column in fixed_point_columns:
    fft_result = custom_fft(column)
    fft_results.append(fft_result)

# 对FFT操作后的结果进行逆FFT变换
inverse_fft_results = []
for result in fft_results:
    inverse_fft_result = inverse_fft(result)
    inverse_fft_results.append(inverse_fft_result)

# 将逆变换后的纵向像素数据与原始图像的横向像素数据进行平均处理
restored_image = np.mean([inverse_fft_result.np_object() + normalized_image[:, idx] for idx, inverse_fft_result in enumerate(inverse_fft_results)], axis=0)

# 显示还原后的图像
cv2.imshow('Restored Image', (restored_image * 255).astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()


###########################################################################################
# 将纵向像素进行定点化
vertical_pixels = []
for col in range(image.shape[1]):
    # 获取当前列的像素值
    column_pixels = image[:, col]
    # 将像素值归一化到0到1之间
    normalized_column = column_pixels / 255.0
    # # 将归一化后的像素值转换为定点数表示
    # fixed_point_column = Fxp(normalized_column, True, 16, 15) # n_word = 16, n_frac = 15
    # vertical_pixels.append(fixed_point_column)
    
################### 用64点的fft来处理 512*512的图像，应该分块处理 ###################
import numpy as np
import cv2

# 读取图像
image = cv2.imread('example_image.jpg', cv2.IMREAD_GRAYSCALE)

# 定义小块的大小
block_size = 64

# 获取图像的大小
height, width = image.shape

# 定义存储处理后图像的数组
processed_image = np.zeros_like(image)

# 对图像进行分块处理
for i in range(0, height, block_size):
    for j in range(0, width, block_size):
        # 获取当前小块的像素数据
        block = image[i:i+block_size, j:j+block_size]
        
        # 对小块进行 FFT 变换
        fft_result = np.fft.fft2(block)
        
        # 对 FFT 结果进行相应的处理（此处省略）
        
        # 对处理后的 FFT 结果进行逆变换
        processed_block = np.fft.ifft2(fft_result).real
        
        # 将处理后的小块放回图像中相应的位置
        processed_image[i:i+block_size, j:j+block_size] = processed_block

# 显示处理后的图像
cv2.imshow('Processed Image', processed_image.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()


"""