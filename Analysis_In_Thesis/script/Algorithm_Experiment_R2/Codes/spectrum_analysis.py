# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 18:30:04 2024

@author: yisss
"""

# In[1]
from Codes.Class_DIF_R2 import R2_DIF_FFT
import numpy as np
import matplotlib.pyplot as plt



# 生成测试信号
N = 64
t = np.linspace(0, 1, N, endpoint=False)
signal = np.sin(2 * np.pi * 7 * t) + np.sin(2 * np.pi * 13 * t)

# 精确FFT计算
fft_exact = np.fft.fft(signal)

# 近似FFT计算（例如通过降低精度实现）
signal_approx = np.round(signal, 1)  # 简单的低精度近似
fft_approx = np.fft.fft(signal_approx)

# 计算频谱误差
spectrum_error = np.abs(fft_exact - fft_approx)

# 绘制频谱和误差
plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.title("Original Signal")
plt.plot(t, signal)

plt.subplot(3, 1, 2)
plt.title("Exact FFT")
plt.plot(np.abs(fft_exact))

plt.subplot(3, 1, 3)
plt.title("Approximate FFT")
plt.plot(np.abs(fft_approx))

plt.figure()
plt.title("Spectrum Error")
plt.plot(spectrum_error)
plt.show()

# 计算信噪比
signal_power = np.sum(np.abs(fft_exact) ** 2)
noise_power = np.sum(np.abs(spectrum_error) ** 2)
SNR = 10 * np.log10(signal_power / noise_power)
print(f"SNR: {SNR:.2f} dB")

# 计算其他误差指标
MSE = np.mean(spectrum_error ** 2)
PSNR = 10 * np.log10(np.max(np.abs(fft_exact)) ** 2 / MSE)
print(f"MSE: {MSE:.4f}")
print(f"PSNR: {PSNR:.2f} dB")


# In[2]  one side spectrum
import numpy as np
from Codes.Class_DIF_R2 import R2_DIF_FFT
import matplotlib.pyplot as plt


# fixed point FFT
N = 64

fixed_FFT = R2_DIF_FFT(N, 16, 16)
In_n_word = 10
Data_n_word = [11, 12, 16, 16, 16, 16]
TF_n_word = [16, 16, 16, 16, 16, 16]
fixed_FFT.TF_Gen()

# fixed_FFT = R2_DIF_FFT(N, 16, 16)
# In_n_word = 12
# Data_n_word = [12, 12, 12, 12, 12, 16]
# TF_n_word = [12, 12, 12, 12, 12, 12]
# fixed_FFT.TF_Gen()


# 生成测试信号
fs = 64 # 采样频率
N = 64    # 采样点数
t = np.arange(N) / fs

f0, f1 = 10, 30  # 信号频率
signal = 0.3*np.sin(2 * np.pi * f0 * t) + 0.4*np.cos(2 * np.pi * f1 * t)

# ideal FFT
ideal_fft_result = np.fft.fft(signal)

# approximate FFT
fixed_FFT.FFT_Fixed(signal, In_n_word, Data_n_word, TF_n_word)
fft_result_fixed = fixed_FFT.final_res_fxp*64                 

appr_magnitude = np.abs(fft_result_fixed)
appr_phase = np.angle(ideal_fft_result)

ideal_magnitude = np.abs(ideal_fft_result)
ideal_phase = np.angle(ideal_fft_result)

# # 近似FFT（示例：减少计算精度）
# approx_signal = np.round(signal, decimals=1)  # 简单近似：减少信号精度
# approx_fft_result = np.fft.fft(approx_signal)
# approx_magnitude = np.abs(approx_fft_result)
# approx_phase = np.angle(approx_fft_result)

# 计算幅度误差和相位误差
magnitude_error = np.abs(ideal_magnitude - appr_magnitude)
phase_error = np.abs(ideal_phase - appr_phase)

# 频率轴
frequencies = np.fft.fftfreq(N, 1/fs)

# 绘制频谱对比
plt.figure(figsize=(12, 10))

# 绘制幅度谱对比
plt.subplot(3, 1, 1)
# 单边谱
plt.plot(frequencies[:N//2], ideal_magnitude[:N//2], label='Ideal FFT')
plt.plot(frequencies[:N//2], appr_magnitude[:N//2], label='Approximate FFT')
#双边谱
# plt.plot(frequencies, ideal_magnitude, label='Ideal FFT')
# plt.plot(frequencies, appr_magnitude, label='Approximate FFT')

plt.title("Magnitude Spectrum Comparison")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.legend()
plt.grid()

# 绘制幅度误差
plt.subplot(3, 1, 2)
plt.plot(frequencies[:N//2], magnitude_error[:N//2])
plt.title("Magnitude Error")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Error")
plt.grid()

# 绘制相位误差
plt.subplot(3, 1, 3)
plt.plot(frequencies[:N//2], phase_error[:N//2])
plt.title("Phase Error")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Error (radians)")
plt.grid()

plt.tight_layout()
plt.show()

# In[3] when use two side just delete [:N//2], this part of codes can be used
import numpy as np
from Codes.Class_DIF_R2 import R2_DIF_FFT
import matplotlib.pyplot as plt

# fixed point FFT
N = 64 # total points of FFT

#######################################################
#     Change different approximate FFT processor      #
#######################################################

################ All 16 bits ##########################
# fixed_FFT = R2_DIF_FFT(N, 16, 16)
# In_n_word = 16
# Data_n_word = [16, 16, 16, 16, 16, 16]
# TF_n_word = [16, 16, 16, 16, 16, 16]
# fixed_FFT.TF_Gen()
#######################################################


# ############### 55 dB approximate FFT ###############
fixed_FFT = R2_DIF_FFT(N, 16, 16)
In_n_word = 12
Data_n_word = [13, 14, 15, 16, 16, 16]
TF_n_word = [14, 14, 16, 16, 16, 16]
fixed_FFT.TF_Gen()
# #####################################################


############### 45 dB approximate FFT ###############
# fixed_FFT = R2_DIF_FFT(N, 16, 16)
# In_n_word = 10
# Data_n_word = [11, 12, 16, 16, 16, 16]
# TF_n_word = [16, 15, 16, 16, 16, 16]
# fixed_FFT.TF_Gen()
#####################################################


############### 35 dB approximate FFT ###############
# fixed_FFT = R2_DIF_FFT(N, 16, 16)
# In_n_word = 8
# Data_n_word = [10, 11, 12, 14, 14, 16]
# TF_n_word = [13, 14, 13, 16, 16, 16]
# fixed_FFT.TF_Gen()
#####################################################



# 生成测试信号
fs = 64  # 采样频率
# N = 64    # 采样点数
t = np.arange(N) / fs
f0, f1 = 10, 30  # 信号频率
signal = 0.3*np.sin(2 * np.pi * f0 * t) + 0.4*np.cos(2 * np.pi * f1 * t)


# 生成幅值为0.1的白噪声
np.random.seed(0)
noise_amplitude = 0.1
noise = noise_amplitude * np.random.normal(0, 1, N)

# 将白噪声添加到信号中
noisy_signal = signal + noise

############################################################################################
# use high sampling frequency to restore the time-domain signal
fs_high = 800 # high sampling frequency
t_high = np.linspace(0, 1, fs_high, endpoint=False)  # generate fs_high point between 0 and 1 second
signal_high = 0.3 * np.sin(2 * np.pi * f0 * t_high) + 0.4 * np.cos(2 * np.pi * f1 * t_high)
noise_high = noise_amplitude * np.random.normal(0, 1, fs_high)  # has same number of points as high sampling points
noisy_signal_high = signal_high + noise_high  # high sampling frequency noisy signal

##############################################################################################
# ideal FFT using np.fft.fft
ideal_fft_result = np.fft.fft(noisy_signal)
# print(ideal_fft_result)

# ideal FFt using fft that established by myself
# fixed_FFT.FFT_Accuracy(noisy_signal)
# ideal_fft_result = fixed_FFT.final_res_accu * 64
# print(ideal_fft_result)

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

# 绘制频谱对比
plt.figure(figsize=(10, 10))

# Magnitude stem 
#use_line_collection=True
plt.subplot(6, 1, 4)
plt.stem(frequencies[:N//2], ideal_magnitude[:N//2], label='Ideal FFT', use_line_collection=True, basefmt='')
plt.stem(frequencies[:N//2], appr_magnitude[:N//2], label='Approximate FFT', use_line_collection=True, linefmt='C1-', markerfmt='o', basefmt='orange')

plt.title("Magnitude Spectrum  Stem Comparison")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.legend()
plt.grid()

# Magnitude plot
plt.subplot(6, 1, 5)
plt.plot(frequencies[:N//2], ideal_magnitude[:N//2], label='Ideal FFT') # add [[:N//2]] is one side spectrum
plt.plot(frequencies[:N//2], appr_magnitude[:N//2], label='Approximate FFT')


# plt.title("Two-Sided Magnitude Spectrum Comparison")
plt.title("Magnitude Spectrum Comparison")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.legend()
plt.grid()

# 绘制双边幅度误差
plt.subplot(6, 1, 6)
plt.plot(frequencies[:N//2], magnitude_error[:N//2])
# plt.title("Two-Sided Magnitude Error")
plt.title("Magnitude Error")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Error")
plt.grid()

# # 绘制相位误差
# plt.subplot(6, 1, 3)
# plt.plot(frequencies[:N//2], phase_error[:N//2])
# plt.title("Phase Error")
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Error (radians)")
# plt.grid()

# 绘制时域信号
plt.subplot(6, 1, 3)
plt.plot(t, noisy_signal)
plt.title("Noisy Time Domain Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()


# 绘噪声的信号
plt.subplot(6, 1, 1)
plt.plot(t_high, noise_high)
plt.title("White Noise Signal (High Sampling Rate)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()

# 绘制带噪声的信号
plt.subplot(6, 1, 2)
plt.plot(t_high, noisy_signal_high)
plt.title("Noisy Time Domain Signal (High Sampling Rate)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()


plt.tight_layout()
plt.show()



# In[4] 棉签图和折线图（展示频谱基线）
import numpy as np
from Codes.Class_DIF_R2 import R2_DIF_FFT
import matplotlib.pyplot as plt

# fixed point FFT
N = 64

# fixed_FFT = R2_DIF_FFT(N, 16, 16)
# In_n_word = 10
# Data_n_word = [11, 12, 16, 16, 16, 16]
# TF_n_word = [16, 16, 16, 16, 16, 16]
# fixed_FFT.TF_Gen()

fixed_FFT = R2_DIF_FFT(N, 16, 16)
In_n_word = 16
Data_n_word = [16, 16, 16, 16, 16, 16]
TF_n_word = [16, 16, 16, 16, 16, 16]
fixed_FFT.TF_Gen()

# 生成测试信号
fs = 64  # 采样频率
N = 64   # 采样点数
t = np.arange(N) / fs
f0, f1 = 10, 30  # 信号频率
signal = 0.3 * np.sin(2 * np.pi * f0 * t) + 0.4 * np.cos(2 * np.pi * f1 * t)

# 生成幅值为0.1的白噪声
noise_amplitude = 0.1
noise = noise_amplitude * np.random.normal(0, 1, N)

# 将白噪声添加到信号中
noisy_signal = signal #noise# +

# ideal FFT
ideal_fft_result = np.fft.fft(noisy_signal)
ideal_magnitude = np.abs(ideal_fft_result)

# approximate FFT
fixed_FFT.FFT_Fixed(noisy_signal, In_n_word, Data_n_word, TF_n_word)
fft_result_fixed = fixed_FFT.final_res_fxp * 64
appr_magnitude = np.abs(fft_result_fixed)

# 频率轴
frequencies = np.fft.fftfreq(N, 1/fs)

# 绘制频谱
plt.figure(figsize=(12, 10))

# 绘制双边幅度谱对比
plt.subplot(2, 1, 1)
# plt.stem(frequencies, ideal_magnitude, use_line_collection=True, label='Ideal FFT')
# plt.stem(frequencies, appr_magnitude, use_line_collection=True, label='Approximate FFT', linefmt='r-', markerfmt='ro')
plt.plot(frequencies, ideal_magnitude, 'b', alpha=0.5)
plt.plot(frequencies, appr_magnitude, 'r', alpha=0.5)
plt.title("Two-Sided Magnitude Spectrum Comparison")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.legend()
plt.grid()

# 绘制双边幅度误差
magnitude_error = np.abs(ideal_magnitude - appr_magnitude)
plt.subplot(2, 1, 2)
plt.stem(frequencies, magnitude_error, use_line_collection=True)
plt.plot(frequencies, magnitude_error, 'b', alpha=0.5)
plt.title("Two-Sided Magnitude Error")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Error")
plt.grid()

plt.tight_layout()
plt.show()

# In[5]
import numpy as np
from Codes.Class_DIF_R2 import R2_DIF_FFT
import matplotlib.pyplot as plt

# fixed point FFT
N = 64 # total points of FFT

#######################################################
#     Change different approximate FFT processor      #
#######################################################

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



# 生成测试信号
fs = 64  # 采样频率
# N = 64    # 采样点数
t = np.arange(N) / fs
f0, f1 = 10, 30  # 信号频率
signal = 0.3*np.sin(2 * np.pi * f0 * t) + 0.4*np.cos(2 * np.pi * f1 * t)


# 生成幅值为0.1的白噪声
np.random.seed(0)
noise_amplitude = 0.1
noise = noise_amplitude * np.random.normal(0, 1, N)

# 将白噪声添加到信号中
noisy_signal = signal + noise

############################################################################################
# use high sampling frequency to restore the time-domain signal
fs_high = 800 # high sampling frequency
t_high = np.linspace(0, 1, fs_high, endpoint=False)  # generate fs_high point between 0 and 1 second
signal_high = 0.3 * np.sin(2 * np.pi * f0 * t_high) + 0.4 * np.cos(2 * np.pi * f1 * t_high)
noise_high = noise_amplitude * np.random.normal(0, 1, fs_high)  # has same number of points as high sampling points
noisy_signal_high = signal_high + noise_high  # high sampling frequency noisy signal

##############################################################################################
# ideal FFT using np.fft.fft
ideal_fft_result = np.fft.fft(noisy_signal)
# print(ideal_fft_result)

# ideal FFt using fft that established by myself
# fixed_FFT.FFT_Accuracy(noisy_signal)
# ideal_fft_result = fixed_FFT.final_res_accu * 64
# print(ideal_fft_result)

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

# 绘制频谱对比
plt.figure(figsize=(10, 12))

# Magnitude stem 
#use_line_collection=True
plt.subplot(7, 1, 4)
plt.stem(frequencies[:N//2], ideal_magnitude[:N//2], label='Ideal FFT', use_line_collection=True, basefmt='')
plt.stem(frequencies[:N//2], appr_magnitude[:N//2], label='Approximate FFT', use_line_collection=True, linefmt='C1-', markerfmt='o', basefmt='orange')

plt.title("Magnitude Spectrum Stem Comparison")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.legend()
plt.grid()

# Magnitude plot
plt.subplot(7, 1, 5)
plt.plot(frequencies[:N//2], ideal_magnitude[:N//2], label='Ideal FFT') # add [[:N//2]] is one side spectrum
plt.plot(frequencies[:N//2], appr_magnitude[:N//2], label='Approximate FFT')


# plt.title("Two-Sided Magnitude Spectrum Comparison")
plt.title("Magnitude Spectrum Comparison")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.legend()
plt.grid()

# 绘制双边幅度误差
plt.subplot(7, 1, 6)
plt.plot(frequencies[:N//2], magnitude_error[:N//2])
# plt.title("Two-Sided Magnitude Error")
plt.title("Magnitude Error")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Error")
plt.grid()

# # 绘制相位误差
# plt.subplot(7, 1, 3)
# plt.plot(frequencies[:N//2], phase_error[:N//2])
# plt.title("Phase Error")
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Error (radians)")
# plt.grid()

# 绘制时域信号
plt.subplot(7, 1, 3)
plt.plot(t, noisy_signal)
plt.title("Noisy Time Domain Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()


# 绘噪声的信号
plt.subplot(7, 1, 1)
plt.plot(t_high, noise_high)
plt.title("White Noise Signal (High Sampling Rate)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()

# 绘制带噪声的信号
plt.subplot(7, 1, 2)
plt.plot(t_high, noisy_signal_high)
plt.title("Noisy Time Domain Signal (High Sampling Rate)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()

# 绘制逆变换的时域信号
plt.subplot(7, 1, 7)
plt.plot(t, noisy_signal, label='Original Noisy Signal')
plt.plot(t, restored_signal_ideal, label='Restored Signal (Ideal FFT)', linestyle='--')
plt.plot(t, restored_signal_approx, label='Restored Signal (Approx FFT)', linestyle=':')
plt.title("Restored Time Domain Signal from IFFT")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

plt.tight_layout()

plt.savefig("Codes/Spectrum_Analysis/35dB_with_IFFT.pdf", format='pdf')
plt.show()

# In[6] observe distortion
import numpy as np
from Codes.Class_DIF_R2 import R2_DIF_FFT
import matplotlib.pyplot as plt

# fixed point FFT
N = 64 # total points of FFT
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
fixed_FFT = R2_DIF_FFT(N, 16, 16)
In_n_word = 10
Data_n_word = [11, 12, 16, 16, 16, 16]
TF_n_word = [16, 15, 16, 16, 16, 16]
fixed_FFT.TF_Gen()
#####################################################


############### 35 dB approximate FFT ###############
# fixed_FFT = R2_DIF_FFT(N, 16, 16)
# In_n_word = 8
# Data_n_word = [10, 11, 12, 14, 14, 16]
# TF_n_word = [13, 14, 13, 16, 16, 16]
# fixed_FFT.TF_Gen()
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
noisy_signal = signal + noise

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
Time_Domain_Signal_Error = restored_signal_ideal - restored_signal_approx
# diff_signal_approx = noisy_signal - restored_signal_approx

# 绘制频谱对比
plt.figure(figsize=(10, 15))

# Magnitude stem 
plt.subplot(8, 1, 4)
plt.stem(frequencies[:N//2], ideal_magnitude[:N//2], label='Ideal FFT', use_line_collection=True, basefmt='')
plt.stem(frequencies[:N//2], appr_magnitude[:N//2], label='Approximate FFT', use_line_collection=True, linefmt='C1-', markerfmt='o', basefmt='orange')
plt.title("Magnitude Spectrum Stem Comparison")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.legend()
plt.grid()

# Magnitude plot
plt.subplot(8, 1, 5)
plt.plot(frequencies[:N//2], ideal_magnitude[:N//2], label='Ideal FFT')
plt.plot(frequencies[:N//2], appr_magnitude[:N//2], label='Approximate FFT')
plt.title("Magnitude Spectrum Comparison")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.legend()
plt.grid()

# 双边幅度误差
plt.subplot(8, 1, 6)
plt.plot(frequencies[:N//2], magnitude_error[:N//2])
plt.title("Magnitude Error")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Error")
plt.grid()

# 绘制时域信号
plt.subplot(8, 1, 3)
plt.plot(t, noisy_signal)
plt.title("Noisy Time Domain Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()

# 绘噪声的信号
plt.subplot(8, 1, 1)
plt.plot(t_high, noise_high)
plt.title("White Noise Signal (High Sampling Rate)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()

# 绘制带噪声的信号
plt.subplot(8, 1, 2)
plt.plot(t_high, noisy_signal_high)
plt.title("Noisy Time Domain Signal (High Sampling Rate)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()

# 绘制逆变换的时域信号
plt.subplot(8, 1, 7)
# plt.plot(t, noisy_signal, label='Original Noisy Signal')
plt.plot(t, restored_signal_ideal, label='Restored Signal (Ideal FFT)', linestyle='--')
plt.plot(t, restored_signal_approx, label='Restored Signal (Approx FFT)', linestyle=':')
plt.title("Restored Time Domain Signal from IFFT")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

# 绘制恢复信号与原始信号的差异
plt.subplot(8, 1, 8)
plt.plot(t, Time_Domain_Signal_Error, label='Time Domain Signal Error')
#plt.plot(t, diff_signal_approx, label='Difference Signal (Approx FFT)', linestyle=':')
plt.title("Difference Between Restored Time Domain Signals")
plt.xlabel("Time (s)")
plt.ylabel("Difference")
plt.legend()
plt.grid()

plt.tight_layout()
# plt.savefig("Codes/Spectrum_Analysis/35dB_with_IFFT_and_Difference.pdf", format='pdf')
plt.show()


# In[7] Spectrum Analysis New (including distortion) !!! New
import numpy as np
from Codes.Class_DIF_R2 import R2_DIF_FFT
import matplotlib.pyplot as plt

# fixed point FFT
N = 64 # total points of FFT
################ All 16 bits ##########################
# precision = '16bits'
# fixed_FFT = R2_DIF_FFT(N, 16, 16)
# In_n_word = 16
# Data_n_word = [16, 16, 16, 16, 16, 16]
# TF_n_word = [16, 16, 16, 16, 16, 16]
# fixed_FFT.TF_Gen()
#######################################################


# ############### 55 dB approximate FFT ###############
# precision = '55dB'
# fixed_FFT = R2_DIF_FFT(N, 16, 16)
# In_n_word = 12
# Data_n_word = [13, 14, 15, 16, 16, 16]
# TF_n_word = [14, 14, 16, 16, 16, 16]
# fixed_FFT.TF_Gen()
# #####################################################


############### 45 dB approximate FFT ###############
# precision = '45dB'
# fixed_FFT = R2_DIF_FFT(N, 16, 16)
# In_n_word = 10
# Data_n_word = [11, 12, 16, 16, 16, 16]
# TF_n_word = [16, 15, 16, 16, 16, 16]
# fixed_FFT.TF_Gen()
#####################################################


############### 35 dB approximate FFT ###############
precision = '35dB'
fixed_FFT = R2_DIF_FFT(N, 16, 16)
In_n_word = 8
Data_n_word = [10, 11, 12, 14, 14, 16]
TF_n_word = [13, 14, 13, 16, 16, 16]
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
noisy_signal = signal + noise

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
plt.title("Original Signal (High Sampling Rate)",)# fontsize=14, fontweight = 'bold')
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()
plt.savefig("Codes/Spectrum_Analysis/Original_Signal.pdf",bbox_inches='tight', format='pdf')
plt.show()


# 图1: 绘制噪声的信号
plt.figure(figsize=(10, 2))
plt.plot(t_high, noise_high)
plt.title("White Noise Signal (High Sampling Rate)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()
plt.savefig("Codes/Spectrum_Analysis/White_Noise_Signal.pdf", bbox_inches='tight',format='pdf')
plt.show()


# 图2: 绘制带噪声的信号
plt.figure(figsize=(10, 2))
plt.plot(t_high, noisy_signal_high)
plt.title("Noisy Time Domain Signal (High Sampling Rate)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()
plt.savefig("Codes/Spectrum_Analysis/Noisy_Signal_Time_Domain.pdf", bbox_inches='tight', format='pdf')
plt.show()


# 图3: 绘制带噪声时域信号(low sampling signal)
plt.figure(figsize=(10, 2))
plt.plot(t, noisy_signal)
plt.title("Noisy Time Domain Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()
plt.savefig("Codes/Spectrum_Analysis/Noisy_Time_Domain_Signal_Low_Fre.pdf", bbox_inches='tight', format='pdf')
plt.show()

#%%
# 图4: 绘制幅度谱的 stem 图
plt.figure(figsize=(10, 2))
plt.stem(frequencies[:N//2], ideal_magnitude[:N//2], label='Ideal FFT', use_line_collection=True, basefmt='')
plt.stem(frequencies[:N//2], appr_magnitude[:N//2], label='Approximate FFT', use_line_collection=True, linefmt='C1-', markerfmt='o', basefmt='orange')
plt.title("Magnitude Spectrum Stem Comparison")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.legend()# fontsize
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
plt.savefig(f"Codes/Spectrum_Analysis/{precision}_Magnitude_Spectrum_Comparison.pdf", bbox_inches='tight', format='pdf')
plt.show()

#%%
# 图6: 绘制双边幅度误差
plt.figure(figsize=(10, 2))
plt.plot(frequencies[:N//2], magnitude_error[:N//2])
plt.title("Magnitude Error")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Error")
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
plt.savefig(f"Codes/Spectrum_Analysis/{precision}_with_IFFT_Restored_Signal.pdf", bbox_inches='tight', format='pdf')
plt.show()

#%%
# 图8: 绘制恢复信号与原始信号的差异
plt.figure(figsize=(10, 2))
plt.plot(t, Time_Domain_Signal_Error, label='Time Domain Signal Error')
plt.title("Difference Between Restored Time Domain Signals")
plt.xlabel("Time (s)")
plt.ylabel("Difference")
plt.legend()
plt.grid()
plt.savefig(f"Codes/Spectrum_Analysis/{precision}_with_IFFT_Signal_Error.pdf", bbox_inches='tight', format='pdf')

# 显示所有图
plt.show()


# In[8]

def float_to_binary(f):
    # Determine the sign bit (assume positive here)
    sign_bit = '0'
    if f < 0:
        sign_bit = '1'
        f = abs(f)
    
    # Initialize variables
    binary_representation = []
    fraction = f
    
    # Convert the fraction part to binary (15 bits)
    for _ in range(15):
        fraction *= 2
        if fraction >= 1:
            binary_representation.append('1')
            fraction -= 1
        else:
            binary_representation.append('0')
    
    # Construct the final binary string
    binary_string = sign_bit + ''.join(binary_representation)
    
    return binary_string

# Example usage
float_value = -0.03775024
binary_representation = float_to_binary(float_value)
print(f"Binary representation of {float_value}: {binary_representation}")
