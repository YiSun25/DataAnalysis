# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 19:32:06 2023

@author: yisss
"""
# In[1]
import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

N=64

# 定义文件夹路径
base_folder = f'Test_Results_4/Data_Results/N{N}'
# base_folder = 'Test_Results_4/Data_Results_ComplexIn/N64' # Results with complex inputs
# 存储数据
all_data = pd.DataFrame()

# 遍历含有 N_64 关键字的文件夹
for n_64_folder in glob.glob(os.path.join(base_folder, f'*N_{N}*')):
    # 输出当前处理的文件夹路径
    # print(f"Processing folder: {n_64_folder}")

    # 遍历 SQNR 文件
    for snr_file in glob.glob(os.path.join(n_64_folder, '*SQNR_f*')):
        # 输出找到的 SNR 文件路径
        # print(f"Found SNR file: {snr_file}")

        # 读取文件中的数据并存储到同一个 DataFrame
        data = pd.read_csv(snr_file, header=None, names=['X', 'Y'], skiprows=1)
        
        # 获取 frac 后缀
        frac_match = re.search(r'frac_(\d+)', snr_file)
        

        # 检查是否匹配成功
        if frac_match:
            frac_suffix = frac_match.group(1)
            
            # 添加 Label 列，用于标识不同的 frac
            data['Label'] = f'Frac_{frac_suffix}'    #f'SNR_frac_{frac_suffix}'
            
            # 合并到总数据表
            all_data = pd.concat([all_data, data], ignore_index=True)
            print(all_data)

# 检查是否有数据
# if not all_data.empty:
# 绘制折线图
plt.figure(figsize=(10, 6))

# 根据 Label 区分不同数据集，选择不同标志, 根据fraction个数不同调整makers的个数
marker_dict = {}
markers = ['o', 's', '^', 'v', '<', '>', 'D',]# 'p', 'P']

for i in range(1, 6):
    label = f'SNR_frac_{i}'    #f'SNR_frac_{i}'
    marker = markers[i-1] if i <= len(markers) else 'o'  # 如果 markers 不够用，使用 'o'
    marker_dict[label] = marker


markersize = 2
# linestyle = '--'
linestyle ='dashed'
# linestyle ='dotted'

legend_elements = []  # 定义一个列表用于存储标志示意图的元素


for label, group in all_data.groupby('Label'):
    marker = marker_dict.get(label, 'o')  # 使用字典映射，如果找不到，默认使用 'o'
    plt.scatter(group['X'], group['Y'], label=label, marker=marker, linewidth=1, s=markersize)
    # plt.plot(group['X'], group['Y'], label=label, marker=marker, linewidth=1,linestyle=linestyle,markersize=markersize)
 
###########################################################################################   
# plot average data
avg_files = glob.glob(f'Test_Results_4/Average_Data/N{N}/SQNR_avg/SQNR*')
# avg_files = glob.glob('Test_Results_4/Average_Data_ComplexIn/N64/SQNR_avg/SQNR*')

file_paths_list = []
for file_path in avg_files:
    file_paths_list.append(file_path)
avg_data = pd.DataFrame()
for i, file_path in enumerate(file_paths_list):
    # import data from csv files
    data_avg = pd.read_csv(file_path, header=None, names=['X', 'Y'], skiprows = 1)
    avg_data = pd.concat([avg_data, data_avg], ignore_index=True)
    # plot figures
    # plt.scatter(data_avg['X'], data_avg['Y'])#, label=f'Fraction {i+1}')
    plt.plot(data_avg['X'], data_avg['Y'])#, label=f'Fraction {i+1}')
############################################################################################
# 画fxp的SQNR横线
# 读取CSV数据
csv_file_path = f'Test_Results_4/Average_Data/N{N}/SQNR_16bit/SQNR_word_16_average.csv'  
# csv_file_path = 'Test_Results_4/Average_Data_ComplexIn/N64/SQNR_16bit/SQNR_word_16_average.csv'  

df = pd.read_csv(csv_file_path)

# 获取文件夹名
folder_name = os.path.basename(os.path.dirname(csv_file_path))

# 将数据复制5次
data_array = np.tile(df['Average'].values, 6)  # 替换 'Average' 为你的数据列的列名

# 绘制点线图
plt.plot(data_array, marker='o', linestyle='dashed', label='Average_16fxp', markersize=5)  # 添加 label 参数
plt.legend(labels=['Average'], loc='upper right')
# plt.xlabel('Index')
# plt.ylabel('Values')
# plt.title(f'Repeated Data Line Plot - Folder: {folder_name}')
# plt.show()

############################################################################################

plt.xlabel('Stages')
plt.ylabel('SQNR')
plt.title(f'SQNR Data for N={N} (with scaling)')

# 调整 x 轴的间距
plt.xticks(all_data['X'].unique())

# 检查是否有数据，然后再调整标签的位置
if 'Label' in all_data.columns:
    plt.legend(bbox_to_anchor=(1.05, 1.04), loc='upper left')

# 保存图片
# plt.savefig(f'Test_Results_4/Analysis_Figures/SQNR/N_{N}_DIF_withAVG.png', bbox_inches='tight')     # store the figure with real inputs
# plt.savefig('Test_Results_4/Analysis_Figures/SQNR_CpxIn/N_64_DIF_withAVG.png', bbox_inches='tight') # store the figure with complex inputs


plt.show()

# In[2]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

N = 64

# Mocking the 'all_data' DataFrame for demonstration
# Replace this with your actual data
all_data = pd.DataFrame({
    'X': np.random.randint(1, 8, size=100),  # Random values between 1 and 7 for demonstration
    'Y': np.random.rand(100) * 50,  # Random Y values for demonstration
    'Label': np.random.choice(['SNR_frac_1', 'SNR_frac_2', 'SNR_frac_3', 'SNR_frac_4', 'SNR_frac_5'], size=100)
})

# 根据 Label 区分不同数据集，选择不同标志, 根据fraction个数不同调整makers的个数
marker_dict = {}
markers = ['o', 's', '^', 'v', '<', '>', 'D']  # 'p', 'P']

for i in range(1, 8):  # 修改范围为1到8
    label = f'SNR_frac_{i}'  # f'SNR_frac_{i}'
    marker = markers[i - 1] if i <= len(markers) else 'o'  # 如果 markers 不够用，使用 'o'
    marker_dict[label] = marker

markersize = 2
# linestyle = '--'
linestyle = 'dashed'
# linestyle ='dotted'

legend_elements = []  # 定义一个列表用于存储标志示意图的元素

for label, group in all_data.groupby('Label'):
    marker = marker_dict.get(label, 'o')  # 使用字典映射，如果找不到，默认使用 'o'
    plt.scatter(group['X'], group['Y'], label=label, marker=marker, linewidth=1, s=markersize)
    # plt.plot(group['X'], group['Y'], label=label, marker=marker, linewidth=1,linestyle=linestyle,markersize=markersize)

###########################################################################################
# plot average data
avg_files = glob.glob(f'Test_Results_4/Average_Data/N{N}/SQNR_avg/SQNR*')
# avg_files = glob.glob('Test_Results_4/Average_Data_ComplexIn/N64/SQNR_avg/SQNR*')

file_paths_list = []
for file_path in avg_files:
    file_paths_list.append(file_path)
avg_data = pd.DataFrame()
for i, file_path in enumerate(file_paths_list):
    # import data from csv files
    data_avg = pd.read_csv(file_path, header=None, names=['X', 'Y'], skiprows=1)
    avg_data = pd.concat([avg_data, data_avg], ignore_index=True)
    # plot figures
    # plt.scatter(data_avg['X'], data_avg['Y'])#, label=f'Fraction {i+1}')
    plt.plot(data_avg['X'], data_avg['Y'])  # , label=f'Fraction {i+1}')
############################################################################################
# 画fxp的SQNR横线
# 读取CSV数据
csv_file_path = f'Test_Results_4/Average_Data/N{N}/SQNR_16bit/SQNR_word_16_average.csv'
# csv_file_path = 'Test_Results_4/Average_Data_ComplexIn/N64/SQNR_16bit/SQNR_word_16_average.csv'

df = pd.read_csv(csv_file_path)

# 获取文件夹名
folder_name = os.path.basename(os.path.dirname(csv_file_path))

# 将数据复制5次
data_array = np.tile(df['Average'].values, 8)  # 替换 'Average' 为你的数据列的列名

# 绘制点线图
plt.plot(data_array, marker='o', linestyle='dashed', label='Average_16fxp', markersize=5)  # 添加 label 参数
plt.legend(labels=['Average'], loc='upper right')
# plt.xlabel('Index')
# plt.ylabel('Values')
# plt.title(f'Repeated Data Line Plot - Folder: {folder_name}')
# plt.show()

############################################################################################

plt.xlabel('Fractions')  # 修改横坐标标签
plt.ylabel('SQNR')
plt.title(f'SQNR Data for N={N} (with scaling)')

# 调整 x 轴的间距
plt.xticks(range(1, 8))  # 设置 x 轴的刻度为 1 到 7

# 检查是否有数据，然后再调整标签的位置
if 'Label' in all_data.columns:
    plt.legend(bbox_to_anchor=(1.05, 1.04), loc='upper left')

# 保存图片
# plt.savefig(f'Test_Results_4/Analysis_Figures/SQNR/N_{N}_DIF_withAVG.png', bbox_inches='tight')  # store the figure with real inputs
# plt.savefig('Test_Results_4/Analysis_Figures/SQNR_CpxIn/N_64_DIF_withAVG.png', bbox_inches='tight') # store the figure with complex inputs

plt.show()


# In[3] Get all data and save them in data_array, in order to change Frac as X-axis and stages as legends.

import pandas as pd
import numpy as np
import os

# 定义文件夹路径
# folder_path = "FFT_Radix2_DIF/Test_Results_4/Data_Results/N64/"
folder_path = "FFT_Radix2_DIF/Test_Results_scaling_new/Data_Results_AftMid/N64_Trun/"


# 初始化一个空的三维数组，用于存储数据
data_array = np.empty((6, 7, 100)) # 6对应stage的数值，7对应frac的数值，100 对应100个Seed文件夹
# data_array = np.empty((6, 8, 100))   # 相比于上一行代码添加了16 bit fixed point的SQNR


# 遍历每个文件夹
for seed in range(100):
    seed_folder = os.path.join(folder_path, f"N_64_Seed_{seed}")
    
    
    # 遍历每个文件
    for word in range(9, 16): # previous (1,9)
        # file_path = os.path.join(seed_folder, f"SQNR_frac_{frac}.csv")
        file_path = os.path.join(seed_folder, f"SQNR_word_{word}.csv")
        
        # 读取CSV文件
        df = pd.read_csv(file_path, header=None, skiprows=1)
        
        # 将数据存储到三维数组中
        data_array[:, word-9, seed] = df.iloc[:, 1].values  # 读取索引之后的值，跳过第一列的索引
        # 读取 Fxp_16.csv 中的值
        
# 如果不需要16 point fixed point的结果，则注释掉 # 中间的代码，并将data_array定义处的第二个维度改为 7  In[4]中的数据也根据实际情况改变横坐标的长度
###################################################################################  
    # fxp_16_file_path = os.path.join(seed_folder, "SQNR_lx_word_16.csv")
    # df_fxp_16 = pd.read_csv(fxp_16_file_path)
    # value_at_wordlength_16 = df_fxp_16.iloc[0, 1] + 0.5  # 假设文件中的值在第一行第二列  0.5 可以不加。

    # # 复制值到数组中
    # value_array = np.tile(value_at_wordlength_16, 6)

    # # 将复制的值添加到 data_array 中
    # data_array[:, 7, seed] = value_array
####################################################################################    

# print the data that are needed
# data_all_stages = data_array[:, 6, 99] 
# print(data_all_stages)

# Check the shape of the array
print("Shape of the Array:", data_array.shape)

# In[4] Plot 100 average results !!! IMPORTANT !!! Please run In[3] first, to get all data that to be ploted.
import matplotlib.pyplot as plt

# 创建图表
plt.figure(figsize=(10, 6))

# 初始化一个空字典，用于存储每个 stage 对应的图例
stage_legends = {}

# 遍历每个 stage
for stage in range(1,7):
    # 如果当前 stage 还没有添加图例
    if stage not in stage_legends:
        # 绘制当前 stage 对应的数据，并指定图例和标志
        
        # plt.scatter(range(1, 8), data_array[stage].mean(axis=1), label=f"Stage {stage}", marker='o') # mean求取了100组的均值
        # scatter chart
        plt.scatter(range(9, 16), data_array[stage-1].mean(axis=1), label=f"Stage {stage}", marker='o') # mean求取了100组的均值
        
        # line chart
        # plt.plot(range(9, 16), data_array[stage].mean(axis=1), label=f"Stage {stage}") # mean求取了100组的均值
        
        # combination of scatter chart and line chart
        # plt.plot(range(9, 16), data_array[stage].mean(axis=1), label=f"Stage {stage}", marker='o', linestyle='-') # mean求取了100组的均值
        # 将当前 stage 添加到已添加的图例中，并存储对应的图例对象
        stage_legends[stage] = plt.legend()
        
# 添加图例
# plt.legend()
# plt.gca().add_artist(stage_legends[0])  # 将第一个 stage 的图例重新添加到图表上


# plt.xlabel("Frac")    # use Frac number to show the x axix
plt.xlabel("Wordlength")

plt.ylabel("SNR(dB)")
plt.title("SNR Distribution (Radix2 DIF, Scaling, Truncation)")

# plt.xticks(range(1, 8)) # range of frac nuber
plt.xticks(range(9, 16))  # 设置 x 轴的刻度标签为 9 到 15

plt.grid(False)

# plt.savefig(f'FFT_Radix2_DIF/Test_Results_4/Analysis_Figures/SQNR/N_{N}_DIF_WLasX_new.png', bbox_inches='tight')  # store the figure with real inputs

plt.show()


# In[5] Use different shapes to plot the scatter
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
N=64

# 创建图表
plt.figure(figsize=(10.5, 6.5))

# 初始化一个空字典，用于存储每个 stage 对应的图例和标志形状
stage_legends = {}

# 设置散点的大小
point_size = 50

# 设置不同的标志形状
marker_shapes = ['o', 's', '^', 'v', 'D', 'x','h']

# 遍历每个 stage
for stage in range(1,7):
    # 如果当前 stage 还没有添加图例
    if stage not in stage_legends:
        # 绘制当前 stage 对应的数据，并指定图例和标志
        plt.scatter(range(9, 16), data_array[stage-1].mean(axis=1), label=f"Stage {stage}", marker=marker_shapes[stage], s=point_size)
        
        # 将当前 stage 添加到已添加的图例中，并存储对应的图例对象和标志形状
        stage_legends[stage] = (plt.legend(), marker_shapes[stage])

# 添加图例
plt.legend(fontsize=16)
# plt.gca().add_artist(stage_legends[0][0])  # 将第一个 stage 的图例重新添加到图表上
plt.xlabel("Stage Output Wordlength",fontsize=16)
plt.ylabel("SNR(dB)",fontsize=16)
plt.title("SNR Distribution (Scaling, Radix-2 DIF)", fontsize=17, fontweight = 'bold')
plt.xticks(range(9, 16), fontsize=16)
plt.yticks(fontsize=16)
plt.ylim(24, 67)  # 将纵坐标的范围设置为 20 到 50
# 设置主要刻度线间隔为 5
plt.gca().yaxis.set_major_locator(MultipleLocator(5))

plt.grid(False)

plt.savefig(f'A_Results/Figure/N_{N}_R2_DIF_Scaling.pdf', bbox_inches='tight')  # store the figure with real inputs

plt.show()

#%%
'''
'o' 圆圈（默认）
'^' 向上的三角形
's' 正方形
'd' 菱形
'+' 十字
'x' X 形
'p' 五角星
'h' 六边形
'v' 向下的三角形
'<' 向左的三角形
'>' 向右的三角形
'''
# In[6] Plot separate 100 results
import matplotlib.pyplot as plt

# 创建图表
plt.figure(figsize=(10, 6))

# 初始化一个空字典，用于存储每个 stage 对应的图例
stage_legends = {}

# 设置散点的大小
point_size = 10

# 遍历每个 stage
for stage in range(6):
    # 如果当前 stage 还没有添加图例
    if stage not in stage_legends:
        # 绘制当前 stage 对应的数据，并指定图例和标志
        for seed in range(100):
            plt.scatter(range(1, 8), data_array[stage, :, seed], label=f"Stage {stage}" if seed == 0 else "", marker='o', s = point_size)
        
        # 将当前 stage 添加到已添加的图例中，并存储对应的图例对象
        stage_legends[stage] = f"Stage {stage}"

# 添加图例
plt.legend([legend for _, legend in stage_legends.items()])#, fontsize = 10 to change the size of the legend
plt.xlabel("Frac")
plt.ylabel("Data")
plt.title("Data Distribution by Frac and Stage")
plt.xticks(range(1, 8))
plt.grid(True) # change to false, the grid will disappear.
plt.show()


# In[7] Update for In[4]
import matplotlib.pyplot as plt
import os

# 文件夹路径
folder_path = "Test_Results_4/Average_Data/N64/SQNR_16bit"

# 构建 Fxp_16.csv 文件的完整路径
fxp_16_file_path = os.path.join(folder_path, "SQNR_word_16_average.csv")

# 从 Fxp_16.csv 文件中读取值
df_fxp_16 = pd.read_csv(fxp_16_file_path)#, header=None)
value_at_wordlength_16 = df_fxp_16["Average"].iloc[0]  # 假设文件中的值在第一行第二列
print(value_at_wordlength_16)

# 创建图表
plt.figure(figsize=(10, 6))

# 初始化一个空字典，用于存储每个 stage 对应的图例
stage_legends = {}

# 遍历每个 stage
for stage in range(6):
    # 如果当前 stage 还没有添加图例
    if stage not in stage_legends:
        # 绘制当前 stage 对应的数据，并指定图例和标志
        plt.scatter(range(9, 16), data_array[stage].mean(axis=1), label=f"Stage {stage}", marker='o') # mean求取了100组的均值
        
        # 将当前 stage 添加到已添加的图例中，并存储对应的图例对象
        stage_legends[stage] = plt.legend()

# 确保图例已经添加到图表中之后再更新图例位置
# 确保图例已经添加到图表中之后再更新图例位置
for stage in range(6):
    # 获取当前 stage 的图例对象
    stage_legend = stage_legends[stage]
    
    # 获取当前 stage 对应的轨迹
    lines = stage_legend.get_lines()
    
    # 如果当前 stage 对应的轨迹不为空
    if lines:
        # 获取当前 stage 对应的标记
        marker = lines[0].get_marker()

        # 找到当前 stage 对应的标记并更新其位置
        for line in lines:
            xdata, ydata = line.get_data()
            xdata = np.append(xdata, 16)  # 添加 x 轴坐标为 16
            ydata = np.append(ydata, value_at_wordlength_16)  # 添加 y 轴坐标为读取到的值
            line.set_data(xdata, ydata)

# 添加图例
plt.legend()
plt.xlabel("Wordlength")
plt.ylabel("Data")
plt.title("Data Distribution by Wordlength and Stage")
plt.xticks(range(9, 17))  # 设置 x 轴的刻度标签为 9 到 16
plt.grid(True)
plt.show()
