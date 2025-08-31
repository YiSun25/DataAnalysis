# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 14:41:29 2024

@author: yisss
"""

import numpy as np
from fxpmath import Fxp
import cmath
import math
# import os
import random
# import pandas as pd
# import matplotlib.pyplot as plt

class R2_DIF_FFT:
    def __init__(self, N, n_Word, n_Word_TF):
        self.N = N
        self.stages = int(math.log(N,2)) # total stages
        self.n_Word = n_Word
        self.n_Word_TF = n_Word_TF
        
        self.method = 'trunc'
        # self.method = 'around'
        # self.method1 = 'trunc'
        self.epsilon = 0
        
        
    # generate twiddle factor (floating point and fixed point)
    def TF_Gen(self):
        self.TF_array_accu = np.zeros((self.stages,self.N >> 1), dtype = np.cdouble)
        for i in range(self.stages):
            for j in range((2**(self.stages-1))//(2**i)):
                self.TF_array_accu[i][j] = cmath.exp((-1j) * 2 * math.pi * j / (2**(self.stages-i))) 
        # change TF_array to fixed point
        # self.TF_array_fixed =  Fxp(self.TF_array_accu, True, self.n_Word_TF, (self.n_Word_TF - 2)) 
        # return TF_array_accu, TF_array_fixed
    
    # DIF radix-2 BF
    def BF_r2(self, in0, in1, TF):
        # print(in0, in1, TF)
        out0 = complex(in0 + in1)/2 
        out1 = complex(in0 - in1)/2*TF
        # print(out0,out1)
        return out0, out1
    
    def bit_reverse(self,n):                             
        result = 0
        for i in range (self.stages):                   
            result <<= 1
            result |= n & 1
            n >>= 1
        return result  # e.g. if n = 1, result = 4
    
    def FFT_Accuracy(self, IN_vec):   # maybe later can also add the variable of fraction number , r=1
        OUT_vec_accu = np.zeros((self.N, self.stages+1), dtype = np.cdouble)
        
        IN_vec = np.array(IN_vec)  # 将列表转换为 NumPy 数组
        IN_vec_modified = IN_vec + self.epsilon # 进行元素加法
        # OUT_vec_accu[:,0] = IN_vec
        OUT_vec_accu[:,0] = Fxp(IN_vec_modified, True, 16, 15, rounding = self.method)
        

        for i in range(self.stages-1, -1, -1): # if N = 128, i = 6,5,4,3,2,1,0
            n = 2**(i+1) # total numbers of input
            step_len = 2**i
            index_c = int(math.log((self.N/n),2)) # index used for column
            for j in range (self.N//n):
                for k in range(step_len):
                    OUT_vec_accu[n*j+k][index_c+1], OUT_vec_accu[n*j+k+step_len][index_c+1] = self.BF_r2(OUT_vec_accu[n*j+k][index_c], OUT_vec_accu[n*j+k+step_len][index_c], self.TF_array_accu[index_c][k])
                    
        self.final_res_accu = np.zeros(self.N,dtype = np.cdouble) # complex128,float64
        for m in range (self.N):
            self.final_res_accu[m] = OUT_vec_accu[self.bit_reverse(m)][index_c+1]      
        
        OUT_vec_accu[:,-1] = self.final_res_accu
        
        return OUT_vec_accu
    
    
    def FFT_Fixed(self, In_vec, In_n_word, Data_n_word, TF_n_word, tf_change = 0, appt_stage = 0, approx = 0, Frac_Appr =0): # n_word 应该是一个数组 # TF_array,
     # self.n_Word = n_word
     OUT_vec_fxp = np.zeros((self.N,self.stages+1), dtype = np.cdouble)
     
     In_vec = np.array(In_vec)  # 将列表转换为 NumPy 数组
     In_vec_modified = In_vec + self.epsilon  # 进行元素加法
     
     OUT_vec_fxp[:,0] = Fxp(In_vec_modified, True, In_n_word, In_n_word-1, rounding = self.method)  # change input data to fixed point
     
     # print('fxp', OUT_vec_fxp[:,0])
     
     TF_array = np.zeros((self.stages,self.N >> 1), dtype = np.cdouble)
     
     counter = 0  # counter for stage number
     appr = approx     # if appr = 1, do approximation.
     
     for i in range(self.stages-1, -1, -1): # if N = 128, i = 6,5,4,3,2,1,0
         #print(i)
         n = 2**(i+1) # total numbers of input
         step_len = 2**i
         index_c = int(math.log((self.N/n),2)) # index used for column
         ## if(i == stage_to_modify )
         ##   n_word = .....
         ## elseif i != stage_to_modify
         ##   n_word = default
         for j in range (self.N//n):
             for k in range(step_len):
                 # previous code
                 # OUT_vec_fxp[n*j+k][index_c+1], OUT_vec_fxp[n*j+k+step_len][index_c+1] = BF_r2(OUT_vec_fxp[n*j+k][index_c], OUT_vec_fxp[n*j+k+step_len][index_c],TF_array[index_c][k])
                  if (counter == 0): # i = self.stage, counter = 0
                     # scaling
                     OUT_vec_fxp[n*j+k][index_c+1] = (OUT_vec_fxp[n*j+k][index_c] + OUT_vec_fxp[n*j+k+step_len][index_c])*0.5
                     OUT_vec_fxp[n*j+k+step_len][index_c+1] = (OUT_vec_fxp[n*j+k][index_c] - OUT_vec_fxp[n*j+k+step_len][index_c])*0.5
                     
                     # align the addition output datawordlenth with the In_n_word
                     OUT_vec_fxp[n*j+k][index_c+1] = Fxp(OUT_vec_fxp[n*j+k][index_c+1]+(1+1j)*self.epsilon, True, In_n_word, In_n_word - 1, rounding = self.method)
                     OUT_vec_fxp[n*j+k+step_len][index_c+1] = Fxp(OUT_vec_fxp[n*j+k+step_len][index_c+1]+(1+1j)*self.epsilon, True, In_n_word, In_n_word -1, rounding = self.method)
                     
                     
                     # multiplication
                     TF_array[index_c][k] = Fxp(self.TF_array_accu[index_c][k], True, TF_n_word[counter], TF_n_word[counter]-2, rounding = self.method) # counter = 0 时，对所乘的TF_n_word进行调整
                     # previous
                     # OUT_vec_fxp[n*j+k+step_len][index_c+1] = OUT_vec_fxp[n*j+k+step_len][index_c+1] * TF_array[index_c][k] # TF_array wordlength 也需要改
                     rr, ri, ir, ii = self.multiplication(OUT_vec_fxp[n*j+k+step_len][index_c+1],TF_array[index_c][k])
                     # print(rr,ri,ir,ii)
                     
                     rr = Fxp(rr+self.epsilon, True, Data_n_word[counter], Data_n_word[counter]-1, rounding = self.method)
                     ri = Fxp(ri+self.epsilon, True, Data_n_word[counter], Data_n_word[counter]-1, rounding = self.method)
                     ir = Fxp(ir+self.epsilon, True, Data_n_word[counter], Data_n_word[counter]-1, rounding = self.method)
                     ii = Fxp(ii+self.epsilon, True, Data_n_word[counter], Data_n_word[counter]-1, rounding = self.method)
                     
                     # OUT_vec_fxp[n*j+k][index_c+1] = Fxp(OUT_vec_fxp[n*j+k][index_c+1], True, Data_n_word[counter], Data_n_word[counter] - 1, rounding = self.method) # Data_n_word[counter+1]对应的是第一个stage的乘法输出的位宽
                     OUT_vec_fxp[n*j+k+step_len][index_c+1] = (rr-ii)+1j*(ri+ir) #Fxp((, True, Data_n_word[counter], Data_n_word[counter] - 1, rounding = self.method)
                     # OUT_vec_fxp[n*j+k][index_c+1] = Fxp(OUT_vec_fxp[n*j+k][index_c+1], True, Data_n_word[counter], Data_n_word[counter] - 1) # Data_n_word[counter+1]对应的是第一个stage的乘法输出的位宽
                     # OUT_vec_fxp[n*j+k+step_len][index_c+1] = Fxp(OUT_vec_fxp[n*j+k+step_len][index_c+1], True, Data_n_word[counter], Data_n_word[counter] - 1)
                     # print(Data_n_word[counter])
                     
                  else:
                    # scaling
                    OUT_vec_fxp[n*j+k][index_c+1] = (OUT_vec_fxp[n*j+k][index_c] + OUT_vec_fxp[n*j+k+step_len][index_c])*0.5
                    OUT_vec_fxp[n*j+k+step_len][index_c+1] =(OUT_vec_fxp[n*j+k][index_c] - OUT_vec_fxp[n*j+k+step_len][index_c])*0.5
                    # align the addition output datawordlenth with the In_n_word
                    OUT_vec_fxp[n*j+k][index_c+1] = Fxp(OUT_vec_fxp[n*j+k][index_c+1]+(1+1j)*self.epsilon, True, Data_n_word[counter-1], Data_n_word[counter-1]-1, rounding = self.method)
                    OUT_vec_fxp[n*j+k+step_len][index_c+1] = Fxp(OUT_vec_fxp[n*j+k+step_len][index_c+1]+(1+1j)*self.epsilon, True, Data_n_word[counter-1], Data_n_word[counter-1]-1, rounding = self.method)
        
        
                    # multiplication
                    TF_array[index_c][k] = Fxp(self.TF_array_accu[index_c][k], True, TF_n_word[counter], TF_n_word[counter]-2) # Array for twiddle factor 原始位宽是2位整数，14位小数
                    # OUT_vec_fxp[n*j+k+step_len][index_c+1] = OUT_vec_fxp[n*j+k+step_len][index_c+1] * TF_array[index_c][k] # TF_array wordlength 也需要改
               
                    rr, ri, ir, ii = self.multiplication(OUT_vec_fxp[n*j+k+step_len][index_c+1],TF_array[index_c][k])
                    rr = Fxp(rr+self.epsilon, True, Data_n_word[counter], Data_n_word[counter]-1, rounding = self.method)
                    ri = Fxp(ri+self.epsilon, True, Data_n_word[counter], Data_n_word[counter]-1, rounding = self.method)
                    ir = Fxp(ir+self.epsilon, True, Data_n_word[counter], Data_n_word[counter]-1, rounding = self.method)
                    ii = Fxp(ii+self.epsilon, True, Data_n_word[counter], Data_n_word[counter]-1, rounding = self.method)
                    # print("result")
                    # print(rr,ri,ir,ii)
                    # OUT_vec_fxp[n*j+k][index_c+1] = Fxp(OUT_vec_fxp[n*j+k][index_c+1], True, Data_n_word[counter], Data_n_word[counter] - 1, rounding = self.method) # Data_n_word[counter+1]对应的是第一个stage的乘法输出的位宽
                    OUT_vec_fxp[n*j+k+step_len][index_c+1] = (rr-ii)+1j*(ri+ir) #Fxp(, True, Data_n_word[counter], Data_n_word[counter] - 1, rounding = self.method)
                    # quantization for the multiplication output
                    # OUT_vec_fxp[n*j+k][index_c+1] = Fxp(OUT_vec_fxp[n*j+k][index_c+1], True, Data_n_word[counter], Data_n_word[counter] - 1) # 第counter个stage的位宽
                    # OUT_vec_fxp[n*j+k+step_len][index_c+1] = Fxp(OUT_vec_fxp[n*j+k+step_len][index_c+1], True, Data_n_word[counter], Data_n_word[counter] - 1)
                        
         counter = counter + 1  
         # print("counter", counter)          
                                     
     # bit reverse order of output
     self.final_res_fxp = np.zeros(self.N,dtype = np.cdouble) 
     # final_res_fxp = Fxp(final_res, True, n_word, n_frac)
     for m in range (self.N):
         self.final_res_fxp[m] = OUT_vec_fxp[self.bit_reverse(m)][index_c+1]
     
     
     OUT_vec_fxp[:,-1] = self.final_res_fxp   # this is not true final_res, because of scaling
                                              # to get real final result, should time FFT total point (N), when use this class
     return OUT_vec_fxp
 
        # generate input vector
    def random_vector(self, seed_num, start=-1, end=1): # range [-1,1)
       random_in = []
       random.seed(seed_num)
       for _ in range(self.N):
           rand_num = random.uniform(start, end)
           random_in.append(rand_num)
       return random_in
       
    def sqnr_calculation(self):
        # get flp_res and fxp_res
        # flp_res = self.final_res_accu
        # fxp_res = self.final_res_fxp
        err = self.final_res_accu - self.final_res_fxp
        
        # signal power
        accu_vec_abs = np.abs(self.final_res_accu)
        squa_accu_vec_abs = np.square(accu_vec_abs)
        sum_squa_accu = sum(squa_accu_vec_abs)
        
        # noise power
        sum_err_flpfxp = np.sum(np.square(np.abs(err)))
        
        # sqnr
        self.SNR_flp_fxp  = 10*math.log10(sum_squa_accu/sum_err_flpfxp)
        
    def multiplication(self, a, b):
        # print("real")
        # print(a.real, a.imag, b.real, b.imag)
        
        arbr = a.real * b.real
        arbi = a.real * b.imag
        aibr = a.imag * b.real
        aibi = a.imag * b.imag
        
        
        return arbr, arbi, aibr, aibi
    


# FFT_R2 = R2_DIF_FFT(64,16,16)
# initial_n_word = FFT_R2.n_Word


# TF_n_word = [16]*FFT_R2.stages
# In_n_word = 16
# Data_n_word = [initial_n_word]*(FFT_R2.stages) # first cell is used for the 0.stage addition results wordlength


# IN_vec = FFT_R2.random_vector(20) 
# print(IN_vec)

# FFT_R2.TF_Gen()

# FFT_R2.FFT_Accuracy(IN_vec)
# FFT_R2.FFT_Fixed(IN_vec, In_n_word, Data_n_word, TF_n_word)

# FFT_R2.sqnr_calculation()


# print(FFT_R2.SNR_flp_fxp)
# print(FFT_R2.final_res_fxp)


##################################################################
# # In[2] Important！！！ use to change back to binary！！！
# # get real part and imaginary part
# real_part = np.real(FFT_R2.final_res_fxp)
# imaginary_part = np.imag(FFT_R2.final_res_fxp)

# real_array = np.array(real_part)
# imaginary_array = np.array(imaginary_part)

# fxp_real_array = Fxp(real_array, signed=True, n_word=16, n_frac=15)
# fxp_imaginary_array = Fxp(imaginary_array, signed=True, n_word=16, n_frac=15)
# print("Fixed-point real part array:", fxp_real_array.bin())
# print("Fixed-point imaginary part array:", fxp_imaginary_array.bin())

# # 将固定点表示转换为整数表示
# fxp_real_int = fxp_real_array.val.astype(int)
# fxp_imaginary_int = fxp_imaginary_array.val.astype(int)

# # 将整数表示转换为16位二进制表示
# fxp_real_bin = [format(x & 0xFFFF, '016b') for x in fxp_real_int]
# fxp_imaginary_bin = [format(x & 0xFFFF, '016b') for x in fxp_imaginary_int]

# # 打印结果
# for i in range(len(fxp_real_bin)):
#     print(f"{fxp_real_bin[i]}  {fxp_imaginary_bin[i]}  //  {i}")


# print(FFT_R2.final_res_accu)

# FFT_R2.sqnr_calculation()

# print(FFT_R2.SNR_flp_fxp)