# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 14:05:55 2024

@author: yisss
"""

# In[1] Adjust input data wordlength
from Codes.Class_DIF_R22 import R22_DIF_FFT
import numpy as np
# from fxpmath import Fxp
# import cmath
# import math
import os
# import random
import pandas as pd
# import matplotlib.pyplot as plt

FFT_R22 = R22_DIF_FFT(64, 16, 16)

SQNR_Req = 35
initial_n_word = FFT_R22.n_Word
# print(initial_n_word)
In_n_word = FFT_R22.n_Word
Data_n_word = [initial_n_word]*(FFT_R22.stages) # first cell is used for the 0.stage addition results wordlength
# print(Data_n_word)
TF_n_word = [initial_n_word]*FFT_R22.stages

mean_SQNR_flp_fxp =0    

# initialise the data
# creat 100 groups differnt random inputs data
random_inputs=[]
seed_num = 100
for _ in range (seed_num):
    random_inputs.append(np.array(FFT_R22.random_vector(_), dtype=np.cdouble))   
# print(random_inputs)

In_Wordlen_list = []
mean_SQNR_flp_fxp_list = []


for In_Wordlen in range(16, 0, -1):
    Data_n_word[0] = In_Wordlen
    SQNR_list=[]
    for In_vec in random_inputs:
        FFT_R22.TF_Gen()
        FFT_R22.FFT_Accuracy(In_vec)
        FFT_R22.FFT_Fixed(In_vec, In_Wordlen, Data_n_word, TF_n_word)
        FFT_R22.sqnr_calculation()
        # print(FFT_R2.SNR_flp_fxp)
        SQNR_list.append(FFT_R22.SNR_flp_fxp)
        
    mean_SQNR_flp_fxp =  sum(SQNR_list) / len(SQNR_list)  # calculate average results of 100 groups
    In_Wordlen_list.append(In_Wordlen)
    mean_SQNR_flp_fxp_list.append(mean_SQNR_flp_fxp)
    # print('Mean_Value', mean_SQNR_flp_fxp)
    # print(In_Wordlen)
    
    if mean_SQNR_flp_fxp < SQNR_Req:
        In_n_word = In_Wordlen + 1
        Data_n_word[0] = In_Wordlen + 1
        break

# put In_Wordlen_list and mean_SQNR_flp_fxp_list in one array
# result_array = np.column_stack((In_Wordlen_list, mean_SQNR_flp_fxp_list))
result1 = [[a, b] for a, b in zip(In_Wordlen_list, mean_SQNR_flp_fxp_list)]
print(result1)



# save results1
res_folder = f"Results_Folder_AftMid/Data_Results1_AftMid_RHU/N{FFT_R22.N}"
if not os.path.exists(res_folder):
    os.makedirs(res_folder)
df_result1 = pd.DataFrame(result1)
filename_res1 = f"res1_{SQNR_Req}dB.csv"
file_path = os.path.join(res_folder, filename_res1)
df_result1.to_csv(file_path)





# In[2] Adjust multiplication output wordlength

print('*********',In_n_word)
print('*********',Data_n_word)
# FFT_R2.FFT_Accuracy(In_vec)
# FFT_R2.FFT_Fixed(In_vec, In_n_word, Data_n_word, TF_n_word)
# FFT_R2.sqnr_calculation()
# print(FFT_R2.SNR_flp_fxp)

Data_n_word_list=[]
mean_SQNR_flp_fxp_list2=[]

for stage in [1,3]: 
    for D_Wordlen in range(16, 0, -1):
        Data_n_word[stage] = D_Wordlen
        Data_n_word[stage+1] = D_Wordlen
        SQNR_list2 = []
        # print(Data_n_word)
        for In_vec in random_inputs:
            FFT_R22.FFT_Accuracy(In_vec) # Not to be omitted here. 
            FFT_R22.FFT_Fixed(In_vec, In_n_word, Data_n_word, TF_n_word)
            FFT_R22.sqnr_calculation()
            # print(FFT_R2.SNR_flp_fxp)
            SQNR_list2.append(FFT_R22.SNR_flp_fxp)
            

        mean_SQNR_flp_fxp2 = sum(SQNR_list2) / len(SQNR_list2)  # 计算均值
        # print('Average SNR',mean_SQNR_flp_fxp2)
        Data_n_word_list.append(Data_n_word.copy())  # 复制当前的 Data_n_word, 将Data_n_word 这个list 放入 Data_n_word_list中
        mean_SQNR_flp_fxp_list2.append(mean_SQNR_flp_fxp2)

        if mean_SQNR_flp_fxp2 < SQNR_Req:
            Data_n_word[stage] = D_Wordlen + 1
            Data_n_word[stage+1] = D_Wordlen + 1
            break
        else:
            continue
        
# 打包成元组列表
# result_array = np.column_stack((In_Wordlen_list, mean_SQNR2_flp_fxp_list))
# print(Data_n_word_list)
# print(mean_SQNR_flp_fxp_list2)

result2 = [[a, b] for a, b in zip(Data_n_word_list, mean_SQNR_flp_fxp_list2)]
print(result2)

# save results2
res_folder = f"Results_Folder_AftMid/Data_Results1_AftMid_RHU/N{FFT_R22.N}"
if not os.path.exists(res_folder):
    os.makedirs(res_folder)
df_result2 = pd.DataFrame(result2)
filename_res2 = f"res2_{SQNR_Req}dB.csv"
file_path = os.path.join(res_folder, filename_res2)
df_result2.to_csv(file_path)


# In[3] Adjust TF wordlength

print('#######', In_n_word)
print('#######', Data_n_word)

TF_n_word_list = []
mean_SQNR_flp_fxp_list3 = []

for stage in [1,3]: 
    for TF_Wordlen in range(16, 0, -1):
        TF_n_word[stage] = TF_Wordlen
        SQNR_list3 = []
        # print(TF_n_word)
        for In_vec in random_inputs:
            FFT_R22.FFT_Accuracy(In_vec)
            FFT_R22.FFT_Fixed(In_vec, In_n_word, Data_n_word, TF_n_word)
            FFT_R22.sqnr_calculation()
            SQNR_list3.append(FFT_R22.SNR_flp_fxp)
        
        mean_SQNR_flp_fxp3 =  sum(SQNR_list3) / len(SQNR_list3)  # 计算均值
        # print('Average_SNR', mean_SQNR_flp_fxp3)
        TF_n_word_list.append(TF_n_word.copy())
        mean_SQNR_flp_fxp_list3.append(mean_SQNR_flp_fxp3)
        
        if mean_SQNR_flp_fxp3 < SQNR_Req:
            TF_n_word[stage] = TF_Wordlen + 1
            break
        else:
            continue
# print(TF_n_word_list)
# print(mean_SQNR_flp_fxp_list3)


result3 = [[a, b] for a, b in zip(TF_n_word_list, mean_SQNR_flp_fxp_list3)]
print(result3)

# save results3
res_folder = f"Results_Folder_AftMid/Data_Results1_AftMid_RHU/N{FFT_R22.N}"
if not os.path.exists(res_folder):
    os.makedirs(res_folder)
df_result3 = pd.DataFrame(result3)
filename_res3 = f"res3_{SQNR_Req}dB.csv"
file_path = os.path.join(res_folder, filename_res3)
df_result3.to_csv(file_path)


# # In[4] Check another method
# #for In_vec in random_inputs:
# FFT_R22 = R22_DIF_FFT(64, 16, 16)
# In_n_word = 8
# Data_n_word = [10, 11, 12, 13, 14, 16]
# TF_n_word = [25, 16, 16, 16, 16, 16]

# SQNR_list3 = []
# random_inputs=[]
# seed_num = 100
# for _ in range (seed_num):
#     random_inputs.append(np.array(FFT_R22.random_vector(_), dtype=np.cdouble))

# FFT_R22.TF_Gen()


# for In_vec in random_inputs:
#     FFT_R22.FFT_Accuracy(In_vec)
#     FFT_R22.FFT_Fixed(In_vec, In_n_word, Data_n_word, TF_n_word)
#     FFT_R22.sqnr_calculation()
#     SQNR_list3.append(FFT_R22.SNR_flp_fxp)

# mean_SQNR_flp_fxp3 =  sum(SQNR_list3) / len(SQNR_list3)  # 计算均值
# # # print('Average_SNR', mean_SQNR_flp_fxp3)
# # TF_n_word_list.append(TF_n_word.copy())
# # mean_SQNR_flp_fxp_list3.append(mean_SQNR_flp_fxp3)
# # print(mean_SQNR_flp_fxp_list3
# print(mean_SQNR_flp_fxp3)

# # [17, 16, 16, 16, 16, 16]  35.342448424618105
# # [20, 16, 16, 16, 16, 16]  35.352224567509104
# # [17, 17, 17, 17, 17, 17]  35.35172421498667
# # [17, 17, 17, 17, 17, 17]  35.352224567509104