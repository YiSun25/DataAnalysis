# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 18:53:43 2024

@author: yisss
"""

# In[1] Change All Data Wordlength together firstly
from Codes.Class_DIF_R22 import R22_DIF_FFT
import numpy as np
import os
import pandas as pd

FFT_R22 = R22_DIF_FFT(64, 16, 16)

# required SQNR
SQNR_Req = 35
# parameter initialization
initial_n_word = FFT_R22.n_Word
In_n_word = FFT_R22.n_Word
Data_n_word = [initial_n_word]*(FFT_R22.stages) # first cell is used for the 0.stage addition results wordlength
# print(Data_n_word)
TF_n_word = [initial_n_word]*FFT_R22.stages

mean_SQNR_flp_fxp = 0    

# initialise the data
# creat 100 groups differnt random inputs data
random_inputs=[]
seed_num = 100
for _ in range (seed_num):
    random_inputs.append(np.array(FFT_R22.random_vector(_), dtype=np.cdouble))   
# print(random_inputs)

In_Wordlen_list = []
Data_n_word_list = []
mean_SQNR_flp_fxp_list = []


for WordLen in range(16, 0, -1):   # Input Wordlen 逐位递减
    Data_n_word[:5] = [WordLen]*5  # Data_n_word 前五个位宽一起减
    SQNR_list=[]
    for In_vec in random_inputs:
        FFT_R22.TF_Gen()
        FFT_R22.FFT_Accuracy(In_vec)
        FFT_R22.FFT_Fixed(In_vec, WordLen, Data_n_word, TF_n_word)
        FFT_R22.sqnr_calculation()
        # print(FFT_R2.SNR_flp_fxp)
        SQNR_list.append(FFT_R22.SNR_flp_fxp)
        
    mean_SQNR_flp_fxp =  sum(SQNR_list) / len(SQNR_list)  # 计算均值
    In_Wordlen_list.append(WordLen)
    Data_n_word_list.append(Data_n_word.copy())  # 复制当前的TF_n_word, 存入TF_Wordlen_list中
    mean_SQNR_flp_fxp_list.append(mean_SQNR_flp_fxp)
    # print('Mean_Value', mean_SQNR_flp_fxp)
    # print(In_Wordlen)
    
    if mean_SQNR_flp_fxp < SQNR_Req:
        In_n_word = WordLen + 1
        Data_n_word[:5] = [WordLen+1]*5
        break
    else:
        continue

# put In_Wordlen_list and mean_SQNR_flp_fxp_list in one array
# result_array = np.column_stack((In_Wordlen_list, mean_SQNR_flp_fxp_list))
result1 = [[a, b, c] for a, b, c in zip(In_Wordlen_list, Data_n_word_list, mean_SQNR_flp_fxp_list)]
print(result1)
# print(In_n_word)
# print(Data_n_word)

# save results
res_folder = f"Results_Folder_new/Data_Results2/N{FFT_R22.N}"
if not os.path.exists(res_folder):
    os.makedirs(res_folder)
df_result1 = pd.DataFrame(result1)
filename_res1 = f"res1_{SQNR_Req}dB.csv"
file_path = os.path.join(res_folder, filename_res1)
df_result1.to_csv(file_path)
print(In_n_word)
print(TF_n_word)

# In[2] Change Input Wordlength for each stage
# print(In_Wordlen_list)
print(In_n_word)
print(Data_n_word)
start_word = In_n_word
mean_SQNR_flp_fxp2 = 0

In_Wordlen_list = []
mean_SQNR_flp_fxp_list2 = []

for In_Wordlen in range(start_word, 0, -1):
    Data_n_word[0] = In_Wordlen
    SQNR_list2=[]
    for In_vec in random_inputs:
        FFT_R22.TF_Gen()
        FFT_R22.FFT_Accuracy(In_vec)
        FFT_R22.FFT_Fixed(In_vec, In_Wordlen, Data_n_word, TF_n_word)
        FFT_R22.sqnr_calculation()
        # print(FFT_R2.SNR_flp_fxp)
        SQNR_list2.append(FFT_R22.SNR_flp_fxp)
        
    mean_SQNR_flp_fxp2 =  sum(SQNR_list2) / len(SQNR_list2)  # 计算均值
    In_Wordlen_list.append(In_Wordlen)
    mean_SQNR_flp_fxp_list2.append(mean_SQNR_flp_fxp2)
    # print('Mean_Value', mean_SQNR_flp_fxp)
    # print(In_Wordlen)
    
    if mean_SQNR_flp_fxp2 < SQNR_Req:
        In_n_word = In_Wordlen + 1
        Data_n_word[0] = In_Wordlen + 1
        break
    else:
        continue

result2 = [[a, b] for a, b in zip(In_Wordlen_list, mean_SQNR_flp_fxp_list2)]
print(result2)
# print(In_n_word

# save results
res_folder = f"Results_Folder_new/Data_Results2/N{FFT_R22.N}"
if not os.path.exists(res_folder):
    os.makedirs(res_folder)
df_result2 = pd.DataFrame(result2)
filename_res2 = f"res2_{SQNR_Req}dB.csv"
file_path = os.path.join(res_folder, filename_res2)
df_result2.to_csv(file_path)


# In[3]
# change data wordlength for each stage
print('*********',In_n_word)
# FFT_R2.FFT_Accuracy(In_vec)
# FFT_R2.FFT_Fixed(In_vec, In_n_word, Data_n_word, TF_n_word)
# FFT_R2.sqnr_calculation()
# print(FFT_R2.SNR_flp_fxp)

Data_n_word_list=[]
mean_SQNR_flp_fxp_list3=[]

for stage in [1,3]: 
    for D_Wordlen in range(start_word, 0, -1):
        Data_n_word[stage] = D_Wordlen
        Data_n_word[stage+1] = D_Wordlen
        SQNR_list3 = []
        # print(Data_n_word)
        for In_vec in random_inputs:
            FFT_R22.FFT_Accuracy(In_vec) # Not to be omitted here. 
            FFT_R22.FFT_Fixed(In_vec, In_n_word, Data_n_word, TF_n_word)
            FFT_R22.sqnr_calculation()
            # print(FFT_R2.SNR_flp_fxp)
            SQNR_list3.append(FFT_R22.SNR_flp_fxp)
            

        mean_SQNR_flp_fxp3 = sum(SQNR_list3) / len(SQNR_list3)  # 计算均值
        # print('Average SNR',mean_SQNR_flp_fxp2)
        Data_n_word_list.append(Data_n_word.copy())  # 复制当前的 Data_n_word, 将Data_n_word 这个list 放入 Data_n_word_list中
        mean_SQNR_flp_fxp_list3.append(mean_SQNR_flp_fxp3)

        if mean_SQNR_flp_fxp3 < SQNR_Req:
            Data_n_word[stage] = D_Wordlen + 1
            Data_n_word[stage+1] = D_Wordlen + 1
            break
        else:
            continue
        
# 打包成元组列表
# result_array = np.column_stack((In_Wordlen_list, mean_SQNR2_flp_fxp_list))
# print(Data_n_word_list)
# print(mean_SQNR_flp_fxp_list2)
result3 = [[a, b] for a, b in zip(Data_n_word_list, mean_SQNR_flp_fxp_list3)]
print(result3)
print(In_n_word)

# save results3
res_folder = f"Results_Folder_new/Data_Results2/N{FFT_R22.N}"
if not os.path.exists(res_folder):
    os.makedirs(res_folder)
df_result3 = pd.DataFrame(result3)
filename_res3 = f"res3_{SQNR_Req}dB.csv"
file_path = os.path.join(res_folder, filename_res3)
df_result3.to_csv(file_path)

# In[4] Change TF Wordlength stage by stage at last
print('#######', In_n_word)
print('#######', Data_n_word)
print('#######', TF_n_word)

TF_n_word_list = []
mean_SQNR_flp_fxp_list4 = []

for stage in [1,3]:  
    for TF_Wordlen in range(16, 0, -1):
        TF_n_word[stage] = TF_Wordlen
        SQNR_list4 = []
        # print(TF_n_word)
        for In_vec in random_inputs:
            FFT_R22.FFT_Accuracy(In_vec)
            FFT_R22.FFT_Fixed(In_vec, In_n_word, Data_n_word, TF_n_word)
            FFT_R22.sqnr_calculation()
            SQNR_list4.append(FFT_R22.SNR_flp_fxp)
        
        mean_SQNR_flp_fxp4 =  sum(SQNR_list4) / len(SQNR_list4)  # 计算均值
        # print('Average_SNR', mean_SQNR_flp_fxp3)
        TF_n_word_list.append(TF_n_word.copy())
        mean_SQNR_flp_fxp_list4.append(mean_SQNR_flp_fxp4)
        
        if mean_SQNR_flp_fxp4 < SQNR_Req:
            TF_n_word[stage] = TF_Wordlen + 1
            break
        else:
            continue
# print(TF_n_word_list)
# print(mean_SQNR_flp_fxp_list3)


result4 = [[a, b] for a, b in zip(TF_n_word_list, mean_SQNR_flp_fxp_list4)]
print(result4)

# save results3
res_folder = f"Results_Folder_new/Data_Results2/N{FFT_R22.N}"
if not os.path.exists(res_folder):
    os.makedirs(res_folder)
df_result4 = pd.DataFrame(result4)
filename_res4 = f"res4_{SQNR_Req}dB.csv"
file_path = os.path.join(res_folder, filename_res4)
df_result4.to_csv(file_path)