# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 15:04:00 2024

@author: yisss
"""

# In[1] Change input wordlength and TF 0 stege wordlength together
# Baseline Experiment
from Codes.Class_DIF_R22 import R22_DIF_FFT
import numpy as np
import os
import pandas as pd


FFT_R22 = R22_DIF_FFT(64, 16, 16)

# required SQNR
SQNR_Req = 35
initial_n_word = FFT_R22.n_Word
# parameter initialization
In_n_word = FFT_R22.n_Word
Data_n_word = [initial_n_word]*(FFT_R22.stages) # first cell is used for the 0.stage addition results wordlength
TF_n_word = [initial_n_word]*FFT_R22.stages

mean_SQNR_flp_fxp =0    

# creat 100 groups differnt random inputs data
random_inputs=[]
seed_num = 100
for _ in range (seed_num):
    random_inputs.append(np.array(FFT_R22.random_vector(_), dtype=np.cdouble))   
# print(random_inputs)

In_Wordlen_list = []
TF_n_word_list = []
mean_SQNR_flp_fxp_list = []
# SQNR_list=[]

# change input data wordlength and 0.stage TF wordllength together
for WordLen in range(16, 0, -1):
    Data_n_word[0] = WordLen
    TF_n_word[1] = WordLen
    SQNR_list=[]
    for In_vec in random_inputs:
        FFT_R22.TF_Gen()
        FFT_R22.FFT_Accuracy(In_vec)
        FFT_R22.FFT_Fixed(In_vec, WordLen, Data_n_word, TF_n_word)
        FFT_R22.sqnr_calculation()
        # print(FFT_R2.SNR_flp_fxp)
        SQNR_list.append(FFT_R22.SNR_flp_fxp) # 将每一个不同的输入计算得来的SQNR放入list中
        
    mean_SQNR_flp_fxp =  sum(SQNR_list) / len(SQNR_list)  # 计算均值
    In_Wordlen_list.append(WordLen)
    TF_n_word_list.append(TF_n_word.copy())  # 复制当前的TF_n_word, 存入TF_Wordlen_list中
    mean_SQNR_flp_fxp_list.append(mean_SQNR_flp_fxp)
    # print('Mean_Value', mean_SQNR_flp_fxp)
    # print(In_Wordlen)
    
    if mean_SQNR_flp_fxp < SQNR_Req:
        In_n_word = WordLen + 1
        Data_n_word[0] = WordLen + 1
        TF_n_word[1] = WordLen + 1
        break
    else:
        continue

# put In_Wordlen_list and mean_SQNR_flp_fxp_list in one array
# result_array = np.column_stack((In_Wordlen_list, mean_SQNR_flp_fxp_list))
result1 = [[a, b, c] for a, b, c in zip(In_Wordlen_list, TF_n_word_list, mean_SQNR_flp_fxp_list)]
print(result1)
# save results1
res_folder = f"Results_Folder_AftMid/Baseline1_Results_AftMid/N{FFT_R22.N}"
if not os.path.exists(res_folder):
    os.makedirs(res_folder)
df_result1 = pd.DataFrame(result1)
filename_res1 = f"res1_{SQNR_Req}dB.csv"
file_path = os.path.join(res_folder, filename_res1)
df_result1.to_csv(file_path)
print(In_n_word)
print(TF_n_word)

# In[2] Change output wordlength and TF wordlength together
print('*********',In_n_word)
print('*********',Data_n_word)
print('*********', TF_n_word)
# FFT_R2.FFT_Accuracy(In_vec)
# FFT_R2.FFT_Fixed(In_vec, In_n_word, Data_n_word, TF_n_word)
# FFT_R2.sqnr_calculation()
# print(FFT_R2.SNR_flp_fxp)

Data_n_word_list=[]
TF_n_word_list=[]
mean_SQNR_flp_fxp_list2=[]

for stage in [1,3]: 
    
    for D_Wordlen in range(16, 0, -1):
        # print(stage+1)
        TF_n_word[stage+2] = D_Wordlen # change TF wordlength and Output wordlength together
        # print(TF_n_word)
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
        Data_n_word_list.append(Data_n_word.copy())  # 复制当前的 Data_n_word 存入 Data_n_word_list
        TF_n_word_list.append(TF_n_word.copy())      # 复制当前的 TF_n_word 存入TF_n_word_list
        # print('******',Data_n_word_list)
        # print('######',TF_n_word_list)
        mean_SQNR_flp_fxp_list2.append(mean_SQNR_flp_fxp2)

        if mean_SQNR_flp_fxp2 < SQNR_Req:
            Data_n_word[stage] = D_Wordlen + 1
            Data_n_word[stage+1] = D_Wordlen + 1
            TF_n_word[stage+2] = D_Wordlen + 1
            break
        else:
            continue
        
# 打包成元组列表
# result_array = np.column_stack((In_Wordlen_list, mean_SQNR2_flp_fxp_list))
# print(Data_n_word_list)
# print(mean_SQNR_flp_fxp_list2)
result2 = [[a, b, c] for a, b, c in zip(Data_n_word_list, TF_n_word_list, mean_SQNR_flp_fxp_list2)]
print(result2)
print(Data_n_word)
print(TF_n_word)

# save results2
res_folder = f"Results_Folder_AftMid/Baseline1_Results_AftMid/N{FFT_R22.N}"
if not os.path.exists(res_folder):
    os.makedirs(res_folder)
df_result2 = pd.DataFrame(result2)
filename_res2 = f"res2_{SQNR_Req}dB.csv"
file_path = os.path.join(res_folder, filename_res2)
df_result2.to_csv(file_path)
