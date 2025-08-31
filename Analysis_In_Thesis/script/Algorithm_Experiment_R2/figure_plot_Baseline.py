# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 20:59:59 2024

@author: yisss
"""

# In[1] Get SNR values
import pandas as pd
import os

SQNR_Req = 55

# 获取当前脚本所在的目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)

# 拼接 Test_Data 文件夹的路径
# data_folder = os.path.join(current_dir, "Results_Folder/Baseline1_Results/N64")
data_folder = os.path.join(current_dir, "Results_Folder_AftMid/Baseline1_Results_AftMid/N64")

# get first file path
first_file_path = os.path.join(data_folder, f"res1_{SQNR_Req}dB.csv")

# get second file path
second_file_path = os.path.join(data_folder, f"res2_{SQNR_Req}dB.csv")

# # get third file path
# third_file_path = os.path.join(data_folder, f"res3_{SQNR_Req}dB.csv")

# # get fourth file path
# fourth_file_path = os.path.join(data_folder, f"res4_{SQNR_Req}db.csv")


# 使用 pandas 读取文件内容
first_data = pd.read_csv(first_file_path)
second_data = pd.read_csv(second_file_path)
# third_data = pd.read_csv(third_file_path)
# fourth_data = pd.read_csv(fourth_file_path)

# read first file
df1 = pd.read_csv(first_file_path)

# read second file
df2 = pd.read_csv(second_file_path)

# read third file
# df3 = pd.read_csv(third_file_path)

# read fourth file
# df4 = pd.read_csv(fourth_file_path)

# get value of the first column
SQNR_values = df1.iloc[:,3].tolist()
SQNR_values += df2.iloc[:,3].tolist()
# SQNR_values += df3.iloc[:,2].tolist()
# SQNR_values += df4.iloc[:,2].tolist()



# # 将第二个 DataFrame 的数据追加到第一个 DataFrame 中, if need all information,then add these several lines
# df1 = df1.append(df2, ignore_index=True)
# df1 = df1.append(df3, ignore_index=True)
# df1 = df1.append(df4, ignore_index=True)

# # 将 DataFrame 转换为 Python 列表
# data_list = df1.values.tolist()
# del(SQNR_values[-1])  # delete the last value
print(SQNR_values)


#%%
import matplotlib.pyplot as plt
import numpy as np

# 生成迭代次数列表作为横坐标
iterations = list(range(len(SQNR_values))) # index starts from 0
# iterations = list(range(1, len(SQNR_values) + 1)) # index starts from 1


# 绘制图形
plt.figure(figsize=(10, 5)) # 设置图片大小为 10x5 英寸
plt.plot(iterations, SQNR_values, marker='o', linestyle='-', markersize=5) # 绘制图形
plt.xlabel('Iteration', fontsize = 12)
plt.ylabel('SNR', fontsize =12)
plt.title(f'SNR_Req = {SQNR_Req}dB', fontsize = 14, fontweight ='bold')

# 设置横坐标范围
plt.xlim(-1, 18)

# 设置横坐标刻度间距
plt.xticks(np.arange(0, 18, 3),fontsize = 12)  # 设置从 0 到 10，间隔为 1 的刻度
plt.yticks(fontsize = 12)
# 在纵坐标为SQNR_Req的位置添加水平线
plt.axhline(y=SQNR_Req, color='r', linestyle='--')
# 添加SQNR_Req标签
plt.text(iterations[-1], SQNR_Req, 'SNR_Req', color='r', va='bottom', fontsize = 12)


# add "Change TF Wordlength" line
# specific_x_value = 20 # iteration number
# plt.axvline(x=specific_x_value, color='g', linestyle='--')
# # plt.text(specific_x_value, SQNR_values[-1], 'Change TF Wordlength', color='g', ha='center', va='bottom')# 文字会紧贴着虚线
# y_coordinate_for_text = 63.5
# plt.text(specific_x_value, y_coordinate_for_text, 'W_TF changing stage by stage', color='g', ha='left', va='bottom')

# # add " Change Data Wordlength stage by stage"
# specific_x_value = 7  # iteration number
# plt.axvline(x=specific_x_value, color='orange', linestyle='--')
# # plt.text(specific_x_value, SQNR_values[-1], 'Change TF Wordlength', color='g', ha='center', va='bottom')# 文字会紧贴着虚线
# y_coordinate_for_text = 64
# plt.text(specific_x_value, y_coordinate_for_text, 'W_D changing stage by stage ', color='orange', ha='left', va='bottom') # y=64的位置添加文字





# add comment for one point
# target_value0 = SQNR_values[0]
# plt.text(0, target_value0, ' 63.792 ', fontsize=8, color='red', ha='left', va="bottom") #给某个点加入文字注释
# plt.plot(0, target_value0, marker='o', color='red', markersize = 5) # 给某个点改变颜色

# target_value2 = SQNR_values[2]
# plt.text(2, target_value2, ' 52.950 ', fontsize=8, color='red', ha='left', va="top") #给某个点加入文字注释
# plt.plot(2, target_value2, marker='o', color='red', markersize = 5) # 给某个点改变颜色

# target_value3 = SQNR_values[3]
# plt.text(3, target_value3, ' 47.047 ', fontsize=8, color='red', ha='left', va="bottom") #给某个点加入文字注释
# plt.plot(3, target_value3, marker='o', color='red', markersize = 5) # 给某个点改变颜色

# target_value5 = SQNR_values[5]
# plt.text(5, target_value5, ' 51.331 ', fontsize=8, color='red', ha='left', va="top") #给某个点加入文字注释
# plt.plot(5, target_value5, marker='o', color='red', markersize = 5) # 给某个点改变颜色

# target_value6 = SQNR_values[6]
# plt.text(6, target_value6, ' 56.279 ', fontsize=8, color='red', ha='left', va="bottom") #给某个点加入文字注释
# plt.plot(6, target_value6, marker='o', color='red', markersize = 5) # 给某个点改变颜色

# target_value7 = SQNR_values[7]
# plt.text(7, target_value7, ' 35.142 ', fontsize=8, color='red', ha='left', va="bottom") #给某个点加入文字注释
# plt.plot(7, target_value7, marker='o', color='red', markersize = 5) # 给某个点改变颜色

# target_value8 = SQNR_values[8]
# plt.text(8, target_value8, ' 34.715 ', fontsize=8, color='red', ha='left',va="top" ) #给某个点加入文字注释
# plt.plot(8, target_value8, marker='o', color='red', markersize = 5) # 给某个点改变颜色

# target_value9 = SQNR_values[9]
# plt.text(9, target_value9, ' 54.119 ', fontsize=8, color='red', ha='left',va="top" ) #给某个点加入文字注释
# plt.plot(9, target_value9, marker='o', color='red', markersize = 5) # 给某个点改变颜色


# target_value10 = SQNR_values[10]
# plt.text(10, target_value10, ' 55.512 ', fontsize=8, color='red', ha='left',va="bottom" ) #给某个点加入文字注释
# plt.plot(10, target_value10, marker='o', color='red', markersize = 5) # 给某个点改变颜色

# target_value11 = SQNR_values[11]
# plt.text(11, target_value11, ' 56.416 ', fontsize=8, color='red', ha='left',va="bottom" ) #给某个点加入文字注释
# plt.plot(11, target_value11, marker='o', color='red', markersize = 5) # 给某个点改变颜色

# target_value12 = SQNR_values[12]
# plt.text(12, target_value12, ' 54.504 ', fontsize=8, color='red', ha='left', va="top") #给某个点加入文字注释
# plt.plot(12, target_value12, marker='o', color='red', markersize = 5) # 给某个点改变颜色

# target_value13 = SQNR_values[13]
# plt.text(13, target_value13, ' 55.210 ', fontsize=8, color='red', ha='left', va="bottom") #给某个点加入文字注释
# plt.plot(13, target_value13, marker='o', color='red', markersize = 5) # 给某个点改变颜色

# target_value14 = SQNR_values[14]
# plt.text(14, target_value14, ' 53.670 ', fontsize=8, color='red', ha='left', va="top") #给某个点加入文字注释
# plt.plot(14, target_value14, marker='o', color='red', markersize = 5) # 给某个点改变颜色

# target_value15 = SQNR_values[15]
# plt.text(15, target_value15, ' 55.615 ', fontsize=8, color='red', ha='left', va="bottom") #给某个点加入文字注释
# plt.plot(15, target_value15, marker='o', color='red', markersize = 5) # 给某个点改变颜色

# target_value17 = SQNR_values[17]
# plt.text(17, target_value17, ' 54.197 ', fontsize=8, color='red', ha='left', va="top") #给某个点加入文字注释
# plt.plot(17, target_value17, marker='o', color='red', markersize = 5) # 给某个点改变颜色

# target_value18 = SQNR_values[18]
# plt.text(18, target_value18, ' 30.724 ', fontsize=8, color='red', ha='left', va="bottom") #给某个点加入文字注释
# plt.plot(18, target_value18, marker='o', color='red', markersize = 5) # 给某个点改变颜色

# target_value19 = SQNR_values[19]
# plt.text(19, target_value19, ' 35.142 ', fontsize=8, color='red', ha='left', va="bottom") #给某个点加入文字注释
# plt.plot(19, target_value19, marker='o', color='red', markersize = 5) # 给某个点改变颜色


# for i in range(len(iterations)): # add value for each point, not workable
#     plt.annotate(str(SQNR_values[i]), (iterations[i], SQNR_values[i]), textcoords="offset points", xytext=(0,10), ha='center')

plt.grid(True)

# save figure
figure_folder = "Results_Folder_AftMid/Baseline1_Results_AftMid"
file_name = f"N64_{SQNR_Req}dB_Base1_R2.pdf"
if not os.path.exists(figure_folder):
    os.makedirs(figure_folder)

file_path = os.path.join(figure_folder, file_name)
plt.savefig(file_path,bbox_inches='tight')

plt.show()