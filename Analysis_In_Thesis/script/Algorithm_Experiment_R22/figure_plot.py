# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 14:58:18 2024

@author: yisss
"""

import pandas as pd
import os

SQNR_Req = 35

# 获取当前脚本所在的目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)

# 拼接 Test_Data 文件夹的路径
# data_folder = os.path.join(current_dir, "Results_Folder/Data_Results/N64")
data_folder = os.path.join(current_dir, "Results_Folder_new/Data_Results1_new/N64")

# get first file path
first_file_path = os.path.join(data_folder, f"res1_{SQNR_Req}dB.csv")

# get second file path
second_file_path = os.path.join(data_folder, f"res2_{SQNR_Req}dB.csv")

# # get third file path
third_file_path = os.path.join(data_folder, f"res3_{SQNR_Req}dB.csv")

# # # get fourth file path
# fourth_file_path = os.path.join(data_folder, f"res4_{SQNR_Req}db.csv")


# 使用 pandas 读取文件内容
first_data = pd.read_csv(first_file_path)
second_data = pd.read_csv(second_file_path)
third_data = pd.read_csv(third_file_path)
# fourth_data = pd.read_csv(fourth_file_path)

# read first file
df1 = pd.read_csv(first_file_path)

# read second file
df2 = pd.read_csv(second_file_path)

# read third file
df3 = pd.read_csv(third_file_path)

# # read fourth file
# df4 = pd.read_csv(fourth_file_path)

# get value of the first column
SQNR_values = df1.iloc[:,2].tolist()
SQNR_values += df2.iloc[:,2].tolist()
SQNR_values += df3.iloc[:,2].tolist()
# SQNR_values += df4.iloc[:,2].tolist()



# # 将第二个 DataFrame 的数据追加到第一个 DataFrame 中, if need all information,then add these several lines
# df1 = df1.append(df2, ignore_index=True)
# df1 = df1.append(df3, ignore_index=True)
# df1 = df1.append(df4, ignore_index=True)

# # 将 DataFrame 转换为 Python 列表
# data_list = df1.values.tolist()

print(SQNR_values)


#%%
import matplotlib.pyplot as plt

# 生成迭代次数列表作为横坐标
iterations = list(range(len(SQNR_values))) # index starts from 0
# iterations = list(range(1, len(SQNR_values) + 1)) # index starts from 1


# 设置图片大小为 10x5 英寸
# 绘制图形
plt.figure(figsize=(10, 5))
plt.plot(iterations, SQNR_values, marker='o', linestyle='-', markersize=5)
plt.xlabel('Iteration')
plt.ylabel('SNR')
plt.title(f'SNR_Req = {SQNR_Req}dB')

# 在纵坐标为SQNR_Req的位置添加水平线
plt.axhline(y=SQNR_Req, color='r', linestyle='--')
# 添加SQNR_Req标签
plt.text(iterations[-1], SQNR_Req, 'SNR_Req', color='r', va='bottom')


## add "Change TF Wordlength" line
specific_x_value = 21  # iteration number
plt.axvline(x=specific_x_value, color='g', linestyle='--')
# plt.text(specific_x_value, SQNR_values[-1], 'Change TF Wordlength', color='g', ha='center', va='bottom')# 文字会紧贴着虚线
y_coordinate_for_text = 63.5
plt.text(specific_x_value, y_coordinate_for_text, 'W_TF changing stage by stage', color='g', ha='left', va='bottom')

# # add " Change Data Wordlength stage by stage"
# specific_x_value = 3  # iteration number
# plt.axvline(x=specific_x_value, color='orange', linestyle='--')
# # plt.text(specific_x_value, SQNR_values[-1], 'Change TF Wordlength', color='g', ha='center', va='bottom')# 文字会紧贴着虚线
# y_coordinate_for_text = 64
# plt.text(specific_x_value, y_coordinate_for_text, 'W_D changing stage by stage ', color='orange', ha='left', va='bottom')



# add comment for one point
target_value0 = SQNR_values[0]
plt.text(0, target_value0, ' 64.084 ', fontsize=8, color='red', ha='left', va="bottom") #给某个点加入文字注释
plt.plot(0, target_value0, marker='o', color='red', markersize = 5) # 给某个点改变颜色

# # target_value2 = SQNR_values[2]
# # plt.text(2, target_value2, ' 52.950 ', fontsize=8, color='red', ha='left', va="top") #给某个点加入文字注释
# # plt.plot(2, target_value2, marker='o', color='red', markersize = 5) # 给某个点改变颜色

# target_value3 = SQNR_values[3]
# plt.text(3, target_value3, ' 59.462 ', fontsize=8, color='red', ha='left', va="bottom") #给某个点加入文字注释
# plt.plot(3, target_value3, marker='o', color='red', markersize = 5) # 给某个点改变颜色

# target_value5 = SQNR_values[5]
# plt.text(5, target_value5, ' 50.640 ', fontsize=8, color='red', ha='left', va="top") #给某个点加入文字注释
# plt.plot(5, target_value5, marker='o', color='red', markersize = 5) # 给某个点改变颜色

# target_value6 = SQNR_values[6]
# plt.text(6, target_value6, ' 55.502 ', fontsize=8, color='red', ha='left', va="bottom") #给某个点加入文字注释
# plt.plot(6, target_value6, marker='o', color='red', markersize = 5) # 给某个点改变颜色

# target_value7 = SQNR_values[7]
# plt.text(7, target_value7, ' 39.496 ', fontsize=8, color='red', ha='left', va="top") #给某个点加入文字注释
# plt.plot(7, target_value7, marker='o', color='red', markersize = 5) # 给某个点改变颜色

target_value8 = SQNR_values[8]
plt.text(8, target_value8, ' 33.643 ', fontsize=8, color='red', ha='left',va="top" ) #给某个点加入文字注释
plt.plot(8, target_value8, marker='o', color='red', markersize = 5) # 给某个点改变颜色

target_value9 = SQNR_values[9]
plt.text(9, target_value9, ' 39.472 ', fontsize=8, color='red', ha='left', va="bottom") #给某个点加入文字注释
plt.plot(9, target_value9, marker='o', color='red', markersize = 5) # 给某个点改变颜色

target_value15 = SQNR_values[15]
plt.text(15, target_value15, ' 33.617 ', fontsize=8, color='red', ha='left', va="top") #给某个点加入文字注释
plt.plot(15, target_value15, marker='o', color='red', markersize = 5) # 给某个点改变颜色

target_value16 = SQNR_values[16]
plt.text(16, target_value16, ' 36.523 ', fontsize=8, color='red', ha='left', va="bottom") #给某个点加入文字注释
plt.plot(16, target_value16, marker='o', color='red', markersize = 5) # 给某个点改变颜色

target_value20 = SQNR_values[20]
plt.text(20, target_value20, ' 34.541 ', fontsize=8, color='red', ha='left', va="top") #给某个点加入文字注释
plt.plot(20, target_value20, marker='o', color='red', markersize = 5) # 给某个点改变颜色

target_value21 = SQNR_values[21]
plt.text(21, target_value21, ' 35.700 ', fontsize=8, color='red', ha='left', va="bottom") #给某个点加入文字注释
plt.plot(21, target_value21, marker='o', color='red', markersize = 5) # 给某个点改变颜色



# 55.502





## add comment for one point
# target_value = SQNR_values[0]
# plt.text(0, target_value, '[16, 16, 16, 16, 16, 16]', fontsize=8, color='red', ha='left') #给某个点加入文字注释
# plt.plot(0, target_value, marker='o', color='black', markersize = 5) # 给某个点改变颜色

# for i in range(len(iterations)):
#     plt.annotate(str(SQNR_values[i]), (iterations[i], SQNR_values[i]), textcoords="offset points", xytext=(0,10), ha='center')

plt.grid(True)

# # save figure
figure_folder = "Results_Folder_new/Data_Results1_new"
file_name = f"N64_{SQNR_Req}dB_E1_R22"
if not os.path.exists(figure_folder):
    os.makedirs(figure_folder)

file_path = os.path.join(figure_folder, file_name)
plt.savefig(file_path)

plt.show()


