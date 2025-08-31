# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 14:21:30 2024

@author: yisss
"""

# radix-22 spectrum analysis_RHU

import numpy as np
from Codes.Class_DIF_R22 import R22_DIF_FFT
import matplotlib.pyplot as plt

# fixed point FFT
N = 64 # total points of FFT
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

# 生成测试信号
fs = 64  # 采样频率
t = np.arange(N) / fs
f0, f1 = 10, 30  # 信号频率
signal = 0.3*np.sin(2 * np.pi * f0 * t) + 0.4*np.cos(2 * np.pi * f1 * t)

# 生成幅值为0.1的白噪声
np.random.seed(0)
noise_amplitude = 0.1
noise = noise_amplitude * np.random.normal(0, 1, N)

# 将白噪声添加到信号中
noisy_signal = signal #+ noise

# use high sampling frequency to restore the time-domain signal
fs_high = 800 # high sampling frequency
t_high = np.linspace(0, 1, fs_high, endpoint=False)  # generate fs_high point between 0 and 1 second
signal_high = 0.3 * np.sin(2 * np.pi * f0 * t_high) + 0.4 * np.cos(2 * np.pi * f1 * t_high)
noise_high = noise_amplitude * np.random.normal(0, 1, fs_high)  # has same number of points as high sampling points
noisy_signal_high = signal_high + noise_high  # high sampling frequency noisy signal

# ideal FFT using np.fft.fft
ideal_fft_result = np.fft.fft(noisy_signal)

# approximate FFT
fixed_FFT.FFT_Fixed(noisy_signal, In_n_word, Data_n_word, TF_n_word)
fft_result_fixed = fixed_FFT.final_res_fxp * 64

appr_magnitude = np.abs(fft_result_fixed)
appr_phase = np.angle(ideal_fft_result)

ideal_magnitude = np.abs(ideal_fft_result)
ideal_phase = np.angle(ideal_fft_result)

# 计算幅度误差和相位误差
magnitude_error = np.abs(ideal_magnitude - appr_magnitude)
phase_error = np.abs(ideal_phase - appr_phase)

# 计算信噪比
signal_power = np.sum(ideal_magnitude ** 2)
noise_power = np.sum(magnitude_error ** 2)
snr = 10 * np.log10(signal_power / noise_power)
print(f"SNR: {snr:.2f} dB")

# 频率轴
frequencies = np.fft.fftfreq(N, 1/fs)

# 计算逆FFT
restored_signal_ideal = np.fft.ifft(ideal_fft_result).real
restored_signal_approx = np.fft.ifft(fft_result_fixed).real

# 计算恢复信号与原始信号的差异
# Time_Domain_Signal_Error = restored_signal_ideal - restored_signal_approx
Time_Domain_Signal_Error = noisy_signal - restored_signal_approx
# diff_signal_approx = noisy_signal - restored_signal_approx

################################ PLOTS ###############################
#%%

# 图0： 绘制原信号
plt.figure(figsize=(10, 2))
plt.plot(t_high, signal_high)
plt.title("s(t)",fontsize = 15, fontweight = 'bold')# fontsize=14, fontweight = 'bold')
plt.xlabel("Time (s)", fontsize = 13)
plt.ylabel("Amplitude", fontsize = 13)
plt.xticks(fontsize = 13)  # 设置从 0 到 10，间隔为 1 的刻度
plt.yticks(fontsize = 13)
plt.grid()
plt.savefig("Codes/Spectrum_Analysis/Original_Signal.pdf",bbox_inches='tight', format='pdf')
plt.show()


# 图1: 绘制噪声的信号
plt.figure(figsize=(10, 2))
plt.plot(t_high, noise_high)
plt.title("n(t)", fontsize = 15, fontweight = 'bold')
plt.xlabel("Time (s)", fontsize = 13)
plt.ylabel("Amplitude", fontsize = 13)
plt.xticks(fontsize = 13)  # 设置从 0 到 10，间隔为 1 的刻度
plt.yticks(fontsize = 13)
plt.grid()
plt.savefig("Codes/Spectrum_Analysis/White_Noise_Signal.pdf", bbox_inches='tight',format='pdf')
plt.show()


# 图2: 绘制带噪声的信号
plt.figure(figsize=(10, 2))
plt.plot(t_high, noisy_signal_high)
plt.title("y(t)", fontsize = 15, fontweight = 'bold')
plt.xlabel("Time (s)", fontsize = 13)
plt.ylabel("Amplitude", fontsize = 13)
plt.xticks(fontsize = 13)  # 设置从 0 到 10，间隔为 1 的刻度
plt.yticks(fontsize = 13)
plt.grid()
plt.savefig("Codes/Spectrum_Analysis/Noisy_Signal_Time_Domain.pdf", bbox_inches='tight', format='pdf')
plt.show()


# 图3: 绘制带噪声时域信号(low sampling signal)
plt.figure(figsize=(10, 2))
plt.plot(t, noisy_signal)
plt.title("Noisy Time Domain Signal (Sampling Rate: 64Hz)", fontsize = 15, fontweight = 'bold')
plt.xlabel("Time (s)", fontsize = 13)
plt.ylabel("Amplitude", fontsize = 13)
plt.xticks(fontsize = 13)  # 设置从 0 到 10，间隔为 1 的刻度
plt.yticks(fontsize = 13)
plt.grid()
plt.savefig("Codes/Spectrum_Analysis/Noisy_Time_Domain_Signal_Low_Fre.pdf", bbox_inches='tight', format='pdf')
plt.show()

#%%
# 图4: 绘制幅度谱的 stem 图
plt.figure(figsize=(10, 2))
plt.stem(frequencies[:N//2], ideal_magnitude[:N//2], label='Ideal FFT', use_line_collection=True, basefmt='')
plt.stem(frequencies[:N//2], appr_magnitude[:N//2], label='Approximate FFT', use_line_collection=True, linefmt='C1-', markerfmt='o', basefmt='orange')
plt.title("Magnitude Spectrum Comparison", fontsize = 15, fontweight = 'bold')
plt.xlabel("Frequency (Hz)", fontsize = 13)
plt.ylabel("Magnitude", fontsize = 13)
plt.xticks(fontsize = 13)  # 设置从 0 到 10，间隔为 1 的刻度
plt.yticks(fontsize = 13)
plt.legend(fontsize = 13)# fontsize
plt.grid()
plt.savefig(f"Codes/Spectrum_Analysis/{precision}_Magnitude_Spectrum_Stem.pdf", bbox_inches='tight', format='pdf')
plt.show()

#%%
# 图5: 绘制幅度谱的 plot 图
plt.figure(figsize=(10, 2))
plt.plot(frequencies[:N//2], ideal_magnitude[:N//2], label='Ideal FFT') # add [[:N//2]] is one side spectrum
plt.plot(frequencies[:N//2], appr_magnitude[:N//2], label='Approximate FFT')
plt.title("Magnitude Spectrum Comparison")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.legend()
plt.grid()
# plt.savefig(f"Codes/Spectrum_Analysis/{precision}_Magnitude_Spectrum_Comparison.pdf", bbox_inches='tight', format='pdf')
plt.show()

#%%
# 图6: 绘制双边幅度误差
plt.figure(figsize=(10, 2))
plt.plot(frequencies[:N//2], magnitude_error[:N//2])
plt.title("Magnitude Error")
plt.xlabel("Frequency (Hz)",fontsize = 15, fontweight = 'bold')
plt.ylabel("Error", fontsize = 13)
plt.xticks(fontsize = 13)  # 设置从 0 到 10，间隔为 1 的刻度
plt.yticks(fontsize = 13)
plt.grid()
plt.savefig(f"Codes/Spectrum_Analysis/{precision}_Magnitude_Error.pdf", bbox_inches='tight', format='pdf')
plt.show()

#%%
# 图7: 绘制逆变换的时域信号
plt.figure(figsize=(10, 2))
plt.plot(t, noisy_signal, label='Original Noisy Signal')
plt.plot(t, restored_signal_ideal, label='Restored Signal (Ideal FFT)', linestyle='--')
plt.plot(t, restored_signal_approx, label='Restored Signal (Approx FFT)', linestyle=':')
plt.title("Restored Time Domain Signal from IFFT")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
# plt.savefig(f"Codes/Spectrum_Analysis/{precision}_with_IFFT_Restored_Signal.pdf", bbox_inches='tight', format='pdf')
plt.show()

#%%
# 图8: 绘制恢复信号与原始信号的差异
plt.figure(figsize=(10, 2))
plt.plot(t, Time_Domain_Signal_Error)#, label='Time Domain Signal Error')
plt.title("Difference Between Restored Time Domain Signals",fontsize = 15, fontweight = 'bold')
plt.xlabel("Time (s)", fontsize = 13)
plt.ylabel("Difference", fontsize = 13)
plt.xticks(fontsize = 13)  # 设置从 0 到 10，间隔为 1 的刻度
plt.yticks(fontsize = 13)
plt.grid(True)
# plt.legend()
# plt.grid()
plt.savefig(f"Codes/Spectrum_Analysis/{precision}_with_IFFT_Signal_Error.pdf", bbox_inches='tight', format='pdf')

# 显示所有图
plt.show()


#%% New(plot all figures 6 in one figure, and all figures 8 in one figure)
import numpy as np
from Codes.Class_DIF_R22 import R22_DIF_FFT
import matplotlib.pyplot as plt

# fixed point FFT
N = 64  # total points of FFT
fs = 64  # 采样频率
t = np.arange(N) / fs
f0, f1 = 10, 30  # 信号频率
signal = 0.3 * np.sin(2 * np.pi * f0 * t) + 0.4 * np.cos(2 * np.pi * f1 * t)

# 生成幅值为0.1的白噪声
np.random.seed(0)
noise_amplitude = 0.1
noise = noise_amplitude * np.random.normal(0, 1, N)

# 将白噪声添加到信号中
noisy_signal = signal   + noise

# ideal FFT using np.fft.fft
ideal_fft_result = np.fft.fft(noisy_signal)
ideal_magnitude = np.abs(ideal_fft_result)
ideal_phase = np.angle(ideal_fft_result)

# 需要比较的不同 precision 配置
precision_configs = {
    '16-bit': {
        'In_n_word': 16,
        'Data_n_word': [16, 16, 16, 16, 16, 16],
        'TF_n_word': [16, 16, 16, 16, 16, 16]
    },
    '55 dB': {
        'In_n_word': 12,
        'Data_n_word': [12, 13, 13, 16, 16, 16],
        'TF_n_word': [13, 13, 13, 13, 16, 16]
    },
    '45 dB': {
        'In_n_word': 10,
        'Data_n_word': [10, 12, 12, 13, 13, 16],
        'TF_n_word': [12, 12, 13, 13, 16, 16]
    },
    '35 dB': {
        'In_n_word': 8,
        'Data_n_word': [8, 11, 11, 13, 13, 16],
        'TF_n_word': [10, 10, 10, 10, 16, 16]
    }
}

# 创建一个新的图表用于显示 "图6" (幅度误差)
plt.figure(figsize=(12, 5))
frequencies = np.fft.fftfreq(N, 1/fs)

# 循环不同的 precision 进行 FFT 计算和误差计算
for precision, config in precision_configs.items():
    # 创建 FFT 对象
    fixed_FFT = R22_DIF_FFT(N, 16, 16)
    fixed_FFT.TF_Gen()

    # 执行近似 FFT
    fixed_FFT.FFT_Fixed(noisy_signal, config['In_n_word'], config['Data_n_word'], config['TF_n_word'])
    fft_result_fixed = fixed_FFT.final_res_fxp * 64

    # 计算幅度误差
    appr_magnitude = np.abs(fft_result_fixed)
    magnitude_error = np.abs(ideal_magnitude - appr_magnitude)

    # 绘制“幅度误差”图 (图6)
    plt.plot(frequencies[:N//2], magnitude_error[:N//2], label=f'{precision}')

# 设置图表标题和标签
plt.title("Magnitude Error", fontsize=15, fontweight='bold')
plt.xlabel("Frequency (Hz)", fontsize=13)
plt.ylabel("Error", fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.grid(True)
plt.legend(fontsize=13)

# 保存“图6” 并显示
plt.savefig("Codes/Spectrum_Analysis/Comparison_Magnitude_Error.pdf", bbox_inches='tight', format='pdf')
plt.show()

# 创建一个新的图表用于显示 "图8" (时域信号恢复误差)
plt.figure(figsize=(12, 5))

# 循环不同的 precision 进行逆 FFT 计算
for precision, config in precision_configs.items():
    # 创建 FFT 对象
    fixed_FFT = R22_DIF_FFT(N, 16, 16)
    fixed_FFT.TF_Gen()

    # 执行近似 FFT
    fixed_FFT.FFT_Fixed(noisy_signal, config['In_n_word'], config['Data_n_word'], config['TF_n_word'])
    fft_result_fixed = fixed_FFT.final_res_fxp * 64

    # 计算逆变换的时域信号
    restored_signal_approx = np.fft.ifft(fft_result_fixed).real

    # 计算恢复信号与原始信号的差异
    Time_Domain_Signal_Error = noisy_signal - restored_signal_approx

    # 绘制“恢复信号误差”图 (图8)
    plt.plot(t, Time_Domain_Signal_Error, label=f'{precision}')

# 设置图表标题和标签
plt.title("Difference Between Restored Time Domain Signals", fontsize=15, fontweight='bold')
plt.xlabel("Time (s)", fontsize=13)
plt.ylabel("Difference", fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.grid(True)
plt.legend(fontsize=13)

# 保存“图8” 并显示
plt.savefig("Codes/Spectrum_Analysis/Comparison_IFFT_Signal_Error.pdf", bbox_inches='tight', format='pdf')
plt.show()
