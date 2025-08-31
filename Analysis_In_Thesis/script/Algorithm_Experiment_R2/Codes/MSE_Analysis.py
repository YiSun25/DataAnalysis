# -*- coding: utf-8 -*-
"""
Created on Tue May  7 16:51:13 2024

@author: yisss
"""

# In[1] Adjust input data wordlength
from Codes.Class_DIF_R2 import R2_DIF_FFT
import numpy as np
# from fxpmath import Fxp
# import cmath
import math
import os
# import random
import pandas as pd
# import matplotlib.pyplot as plt

FFT_R2 = R2_DIF_FFT(64, 16, 16)

SQNR_Req = 55
initial_n_word = FFT_R2.n_Word
# print(initial_n_word)
In_n_word = FFT_R2.n_Word
Data_n_word = [initial_n_word]*(FFT_R2.stages) # first cell is used for the 0.stage addition results wordlength
# print(Data_n_word)
TF_n_word = [initial_n_word]*FFT_R2.stages

mean_SQNR_flp_fxp =0    

# initialise the data
# creat 100 groups differnt random inputs data
random_inputs=[]
seed_num = 100
for _ in range (seed_num):
    random_inputs.append(np.array(FFT_R2.random_vector(_), dtype=np.cdouble))   
# print(random_inputs)

In_Wordlen_list = []
mean_SQNR_flp_fxp_list = []
# average_MSE_list = []
average_SQNR_list = []


for In_Wordlen in range(16, 0, -1):
    SQNR_list=[]
    MSE_list=[]
    SQNR_stage_list = []
    for In_vec in random_inputs:
        FFT_R2.TF_Gen()
        Accu = FFT_R2.FFT_Accuracy(In_vec)
        Appr = FFT_R2.FFT_Fixed(In_vec, In_Wordlen, Data_n_word, TF_n_word)
        FFT_R2.sqnr_calculation()
        # print(FFT_R2.SNR_flp_fxp)
        SQNR_list.append(FFT_R2.SNR_flp_fxp)
        
        # SQNR each stage
        signal_power = np.sum(np.square(np.abs(Accu)), axis = 0)
        noise_power = np.sum(np.square(np.abs(Accu-Appr)), axis = 0)
        # SQNR_stage = signal_power/noise_power
        SQNR_stage = 10*np.log10(signal_power/noise_power)
        SQNR_stage_list.append(SQNR_stage)
        
        # MSE
        MSE = np.sum(np.square(np.abs(Accu-Appr)), axis = 0)/FFT_R2.N
        MSE_list.append(MSE)
        
    mean_SQNR_flp_fxp =  sum(SQNR_list) / len(SQNR_list)  # 计算均值
    MSE_average = [sum(col) / len(MSE_list) for col in zip(*MSE_list)]
    SQNR_stage_average = [sum(col) / len(SQNR_stage_list) for col in zip(*SQNR_stage_list)]
    
    In_Wordlen_list.append(In_Wordlen)
    mean_SQNR_flp_fxp_list.append(mean_SQNR_flp_fxp)
    # average_MSE_list.append(MSE_average)
    average_SQNR_list.append(SQNR_stage_average)
    # print('Mean_Value', mean_SQNR_flp_fxp)
    # print(In_Wordlen)
    
    if mean_SQNR_flp_fxp < SQNR_Req:
        In_n_word = In_Wordlen + 1
        break

# put In_Wordlen_list and mean_SQNR_flp_fxp_list in one array
# result_array = np.column_stack((In_Wordlen_list, mean_SQNR_flp_fxp_list))
# result1 = [[a, b] for a, b in zip(In_Wordlen_list, mean_SQNR_flp_fxp_list)]
# print(result1)
# print(average_MSE_list)

# save result
res_folder = f"Results_Folder/Data_Results1/N{FFT_R2.N}"
if not os.path.exists(res_folder):
    os.makedirs(res_folder)
df_result1 = pd.DataFrame(average_SQNR_list)
filename_res1 = f"res1_{SQNR_Req}dB_SQNR_stage.csv"
file_path = os.path.join(res_folder, filename_res1)
df_result1.to_csv(file_path)



# # save results1
# res_folder = f"Results_Folder/Data_Results1/N{FFT_R2.N}"
# if not os.path.exists(res_folder):
#     os.makedirs(res_folder)
# df_result1 = pd.DataFrame(result1)
# filename_res1 = f"res1_{SQNR_Req}dB_MSE.csv"
# file_path = os.path.join(res_folder, filename_res1)
# df_result1.to_csv(file_path)
# In[2] Adjust multiplication output wordlength

print('*********',In_n_word)
# FFT_R2.FFT_Accuracy(In_vec)
# FFT_R2.FFT_Fixed(In_vec, In_n_word, Data_n_word, TF_n_word)
# FFT_R2.sqnr_calculation()
# print(FFT_R2.SNR_flp_fxp)

Data_n_word_list=[]
mean_SQNR_flp_fxp_list2=[]
# average_MSE_list2=[]
average_SQNR_list2 = []


for stage in range(FFT_R2.stages-1): # skip last stage
    for D_Wordlen in range(16, 0, -1):
        Data_n_word[stage] = D_Wordlen
        SQNR_list2 = []
        # MSE_list2 = []
        SQNR_stage_list2 = []
        # print(Data_n_word)
        for In_vec in random_inputs:
            Accu = FFT_R2.FFT_Accuracy(In_vec) # Not to be omitted here. 
            Appr = FFT_R2.FFT_Fixed(In_vec, In_n_word, Data_n_word, TF_n_word)
            FFT_R2.sqnr_calculation()
            # print(FFT_R2.SNR_flp_fxp)
            SQNR_list2.append(FFT_R2.SNR_flp_fxp)
            
            signal_power = np.sum(np.square(np.abs(Accu)), axis = 0)
            noise_power = np.sum(np.square(np.abs(Accu-Appr)), axis = 0)
            # SQNR_stage = signal_power/noise_power
            SQNR_stage = 10*np.log10(signal_power/noise_power)
            SQNR_stage_list2.append(SQNR_stage)
            
            # MSE = np.sum(np.square(np.abs(Accu-Appr)), axis = 0)/FFT_R2.N
            # MSE_list2.append(MSE)
            

        mean_SQNR_flp_fxp2 = sum(SQNR_list2) / len(SQNR_list2)  # calculate average results of 100 groups
        # MSE_average2 = [sum(col) / len(MSE_list2) for col in zip(*MSE_list2)]
        SQNR_stage_average2 = [sum(col) / len(SQNR_stage_list2) for col in zip(*SQNR_stage_list2)]
        # print('Average SNR',mean_SQNR_flp_fxp2)
        Data_n_word_list.append(Data_n_word.copy())  # 复制当前的 Data_n_word, 将Data_n_word 这个list 放入 Data_n_word_list中
        mean_SQNR_flp_fxp_list2.append(mean_SQNR_flp_fxp2)
        # average_MSE_list2.append(MSE_average2)
        average_SQNR_list2.append(SQNR_stage_average2)
        

        if mean_SQNR_flp_fxp2 < SQNR_Req:
            Data_n_word[stage] = D_Wordlen + 1
            break
        else:
            continue
        
# 打包成元组列表
# result_array = np.column_stack((In_Wordlen_list, mean_SQNR2_flp_fxp_list))
# print(Data_n_word_list)
# print(mean_SQNR_flp_fxp_list2)
# result2 = [[a, b] for a, b in zip(Data_n_word_list, mean_SQNR_flp_fxp_list2)]
# print(result2)
# print(average_MSE_list2)

# save results2
res_folder = f"Results_Folder/Data_Results1/N{FFT_R2.N}"
if not os.path.exists(res_folder):
    os.makedirs(res_folder)
df_result2 = pd.DataFrame(average_SQNR_list2)
filename_res2 = f"res2_{SQNR_Req}dB_SQNR_stage.csv"
file_path = os.path.join(res_folder, filename_res2)
df_result2.to_csv(file_path)


# In[3] Adjust TF wordlength

print('#######', In_n_word)
print('#######', Data_n_word)

TF_n_word_list = []
mean_SQNR_flp_fxp_list3 = []
# average_MSE_list3 = []
average_SQNR_list3 = []


for stage in range(FFT_R2.stages-2): # skip last two stages
    for TF_Wordlen in range(16, 0, -1):
        TF_n_word[stage] = TF_Wordlen
        SQNR_list3 = []
        # MSE_list3 = []
        SQNR_stage_list3 = []
        
        # print(TF_n_word)
        for In_vec in random_inputs:
            Accu = FFT_R2.FFT_Accuracy(In_vec)
            Appr = FFT_R2.FFT_Fixed(In_vec, In_n_word, Data_n_word, TF_n_word)
            FFT_R2.sqnr_calculation()
            SQNR_list3.append(FFT_R2.SNR_flp_fxp)
            # MSE = np.sum(np.square(np.abs(Accu-Appr)), axis = 0)/FFT_R2.N
            # MSE_list3.append(MSE)
            
            signal_power = np.sum(np.square(np.abs(Accu)), axis = 0)
            noise_power = np.sum(np.square(np.abs(Accu-Appr)), axis = 0)
            # SQNR_stage = signal_power/noise_power
            SQNR_stage = 10*np.log10(signal_power/noise_power)
            SQNR_stage_list3.append(SQNR_stage)
        
        mean_SQNR_flp_fxp3 =  sum(SQNR_list3) / len(SQNR_list3)  # 计算均值
        # MSE_average3 = [sum(col) / len(MSE_list3) for col in zip(*MSE_list3)]
        SQNR_stage_average3 = [sum(col) / len(SQNR_stage_list3) for col in zip(*SQNR_stage_list3)]
        
        # print('Average_SNR', mean_SQNR_flp_fxp3)
        TF_n_word_list.append(TF_n_word.copy())
        mean_SQNR_flp_fxp_list3.append(mean_SQNR_flp_fxp3)
        # average_MSE_list3.append(MSE_average3)
        average_SQNR_list3.append(SQNR_stage_average3)
        
        
        if mean_SQNR_flp_fxp3 < SQNR_Req:
            TF_n_word[stage] = TF_Wordlen + 1
            break
        else:
            continue
# print(TF_n_word_list)
# print(mean_SQNR_flp_fxp_list3)


# result3 = [[a, b] for a, b in zip(TF_n_word_list, mean_SQNR_flp_fxp_list3)]
# print(result3)
# print(average_MSE_list3)

# save results3
res_folder = f"Results_Folder/Data_Results1/N{FFT_R2.N}"
if not os.path.exists(res_folder):
    os.makedirs(res_folder)
df_result3 = pd.DataFrame(average_SQNR_list3)
filename_res3 = f"res3_{SQNR_Req}dB_SQNR_stage.csv"
file_path = os.path.join(res_folder, filename_res3)
df_result3.to_csv(file_path)


# In[4] Check another method
#for In_vec in random_inputs:
FFT_R2 = R2_DIF_FFT(64, 16, 16)
In_n_word = 8
Data_n_word = [10, 11, 12, 13, 14, 16]
TF_n_word = [25, 16, 16, 16, 16, 16]

SQNR_list3 = []
random_inputs=[]
seed_num = 100
for _ in range (seed_num):
    random_inputs.append(np.array(FFT_R2.random_vector(_), dtype=np.cdouble))

FFT_R2.TF_Gen()


for In_vec in random_inputs:
    FFT_R2.FFT_Accuracy(In_vec)
    FFT_R2.FFT_Fixed(In_vec, In_n_word, Data_n_word, TF_n_word)
    FFT_R2.sqnr_calculation()
    SQNR_list3.append(FFT_R2.SNR_flp_fxp)

mean_SQNR_flp_fxp3 =  sum(SQNR_list3) / len(SQNR_list3)  # 计算均值
# # print('Average_SNR', mean_SQNR_flp_fxp3)
# TF_n_word_list.append(TF_n_word.copy())
# mean_SQNR_flp_fxp_list3.append(mean_SQNR_flp_fxp3)
# print(mean_SQNR_flp_fxp_list3
print(mean_SQNR_flp_fxp3)