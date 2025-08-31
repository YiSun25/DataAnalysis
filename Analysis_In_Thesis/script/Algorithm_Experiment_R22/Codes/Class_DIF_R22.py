# -*- coding: utf-8 -*-
"""
Created on Wed May  8 15:21:44 2024

@author: yisss
"""

import numpy as np
from fxpmath import Fxp
import random
import cmath
import math

# import os
# import struct
# import random
# import pandas as pd
# import matplotlib.pyplot as plt


class R22_DIF_FFT: 
    def __init__(self, N, n_Word, n_Word_TF):
        self.N = N
        self.stages = int(math.log(N,2)) # total stages
        self.n_Word = n_Word
        self.n_Word_TF = n_Word_TF
        self.method1 = 'trunc'
        self.method2 = 'around'
        self.epsilon = 1e-16
   
    # generate twiddle factor
    def TF_Gen(self):
        self.TF_array_accu = np.zeros((self.stages,self.N), dtype = np.cdouble)
        for i in range(self.stages):
            if (i%2 == 0):
                # print(i)
                group_num = int(self.N/(2**(self.stages-i)))
                # print(group_num)
                group_mem = int(self.N/(2**i))
                # print(group_mem)
                self.TF_array_accu[i,:] = 1
                for j in range(group_num):
                    for k in range (int(self.N//(2**i)*3/4), self.N//(2**i)):
                        # print(i,j*group_mem + k )
                        self.TF_array_accu[i][j*group_mem + k] = -1j
                # print(TF_array_accu)
            
            if (i%2 == 1):
                # print(i)
                group_mem1 = self.N/(2**(i-1))
                group_num1 = self.N/group_mem1
                step = self.N/(2**(i+1))
                for g in range (int(group_num1)):
                    for l in range (2):
                        for m in range (2):
                            for s in range (int(step)):
                                self.TF_array_accu[i][int(step*(2*l+m)+s)+g*int(group_mem1)] = cmath.exp((-1j) * 2 * math.pi * s * (l+2*m) / (2**(self.stages-(i-1))))
        # print(TF_array_accu)
        # TF_array_fixed =  Fxp(TF_array_accu, True, n_word_TF, n_frac_TF)
        # return TF_array_accu, TF_array_fixed
        
               
    # DIF radix-2 BF
    def BF_r2(self, in0, in1):
        # print(in0, in1, TF)
        out0 = complex(in0 + in1)/2 # multiply 1/2 for scaling, to avoid overflow, when do scaling in fixed point, note the 
        out1 = complex(in0 - in1)/2 
        # print(out0,out1)
        return out0, out1
    
    def bit_reverse(self,n):                             
        result = 0
        for i in range (self.stages):                    
            result <<= 1
            result |= n & 1
            n >>= 1
        return result  # e.g. if n = 1, result = 4
    
    # Floating point FFT implementation
    def FFT_Accuracy(self, IN_vec):   # maybe later can also add the variable of fraction number , r=1
        OUT_vec_accu = np.zeros((self.N,self.stages+1), dtype = np.cdouble)
        # OUT_vec_accu[:,0] = IN_vec
        IN_vec = np.array(IN_vec)  # 将列表转换为 NumPy 数组
        IN_vec_modified = IN_vec + self.epsilon  # 进行元素加法
        
        OUT_vec_accu[:,0] = Fxp(IN_vec_modified, True, 16, 15, rounding = self.method2)
        
        # print('In_vec', IN_vec)

        for i in range(self.stages-1, -1, -1): # if N = 128, i = 6,5,4,3,2,1,0
            n = 2**(i+1) # total numbers of input
            step_len = 2**i
            index_c = int(math.log((self.N/n),2)) # index used for column
            for j in range (self.N//n):
                for k in range(step_len):
                    BF_res = self.BF_r2(OUT_vec_accu[n*j+k][index_c], OUT_vec_accu[n*j+k+step_len][index_c])
                    # print('BF_res', BF_res)
                    y0, y1 = BF_res[0], BF_res[1]
                    OUT_vec_accu[n*j+k][index_c+1] = y0*self.TF_array_accu[index_c][n*j+k]
                    OUT_vec_accu[n*j+k+step_len][index_c+1] = y1*self.TF_array_accu[index_c][n*j+k+step_len]
                    # print('OUT_vec_accu', OUT_vec)
        self.final_res_accu = np.zeros(self.N,dtype = np.csingle) # complex64
        for m in range (self.N):
            self.final_res_accu[m] = OUT_vec_accu[self.bit_reverse(m)][index_c+1]
        
        OUT_vec_accu[:,-1] = self.final_res_accu
        
        return OUT_vec_accu
    
    # Fixed point FFT implementation    
    def FFT_Fixed(self, IN_vec, In_n_word, Data_n_word, TF_n_word, approx=0, appt_stage=0, Frac_Appr=0):
        OUT_vec_fxp = np.zeros((self.N, self.stages+1), dtype = np.cdouble)
        # OUT_vec_fxp = Fxp(OUT_vec_fxp, True, n_word, n_frac) 
       
        IN_vec = np.array(IN_vec)  # 将列表转换为 NumPy 数组
        IN_vec_modified = IN_vec + self.epsilon# 进行元素加法
        
        OUT_vec_fxp[:,0] = Fxp(IN_vec_modified, True, In_n_word, In_n_word-1, rounding = self.method2)
        
        TF_array = np.zeros((self.stages,self.N), dtype = np.cdouble)
       
        counter = 0  # counter for stage number
        appr = approx     # if appr = 1, do approximation.
        
        for i in range(self.stages-1, -1, -1): # if N = 128, i = 6,5,4,3,2,1,0
            #print(i)
            n = 2**(i+1) # total numbers of input
            step_len = 2**i
            index_c = int(math.log((self.N/n),2)) # index used for column
            for j in range (self.N//n):
                for k in range(step_len):
                    
                    if (counter == 0):
                        # BF_res = self.BF_r2(OUT_vec_fxp[n*j+k][index_c], OUT_vec_fxp[n*j+k+step_len][index_c])
                        
                        OUT_vec_fxp[n*j+k][index_c+1] = (OUT_vec_fxp[n*j+k][index_c] + OUT_vec_fxp[n*j+k+step_len][index_c])*0.5
                        OUT_vec_fxp[n*j+k+step_len][index_c+1] = (OUT_vec_fxp[n*j+k][index_c] - OUT_vec_fxp[n*j+k+step_len][index_c])*0.5
                        # align the addition output datawordlenth with the In_n_word
                        OUT_vec_fxp[n*j+k][index_c+1] = Fxp(OUT_vec_fxp[n*j+k][index_c+1]+(1+1j)*self.epsilon, True, In_n_word, In_n_word - 1, rounding = self.method2)
                        OUT_vec_fxp[n*j+k+step_len][index_c+1] = Fxp(OUT_vec_fxp[n*j+k+step_len][index_c+1]+(1+1j)*self.epsilon, True, In_n_word, In_n_word -1, rounding = self.method2)
                        
                        TF_array[index_c][n*j+k] = Fxp(self.TF_array_accu[index_c][n*j+k], True, TF_n_word[counter], TF_n_word[counter]-2)
                        # print(TF_array[index_c][n*j+k])
                        TF_array[index_c][n*j+k+step_len] = Fxp(self.TF_array_accu[index_c][n*j+k+step_len], True, TF_n_word[counter], TF_n_word[counter]-2)
                        
                        # OUT_vec_fxp[n*j+k][index_c+1], OUT_vec_fxp[n*j+k+step_len][index_c+1] = OUT_vec_fxp[n*j+k][index_c+1]*TF_array[index_c][n*j+k], OUT_vec_fxp[n*j+k+step_len][index_c+1]*TF_array[index_c][n*j+k+step_len]
                        
                        rr0, ri0, ir0, ii0 = self.multiplication(OUT_vec_fxp[n*j+k][index_c+1],TF_array[index_c][n*j+k])
                        rr0 = Fxp(rr0+self.epsilon, True, Data_n_word[counter], Data_n_word[counter]-1, rounding = self.method2)
                        ri0 = Fxp(ri0+self.epsilon, True, Data_n_word[counter], Data_n_word[counter]-1, rounding = self.method2)
                        ir0 = Fxp(ir0+self.epsilon, True, Data_n_word[counter], Data_n_word[counter]-1, rounding = self.method2)
                        ii0 = Fxp(ii0+self.epsilon, True, Data_n_word[counter], Data_n_word[counter]-1, rounding = self.method2)
                        # print(rr0,ri0,ir0,ii0)
                        
                        rr1, ri1, ir1, ii1 = self.multiplication(OUT_vec_fxp[n*j+k+step_len][index_c+1],TF_array[index_c][n*j+k+step_len])
                        rr1 = Fxp(rr1+self.epsilon, True, Data_n_word[counter], Data_n_word[counter]-1, rounding = self.method2)
                        ri1 = Fxp(ri1+self.epsilon, True, Data_n_word[counter], Data_n_word[counter]-1, rounding = self.method2)
                        ir1 = Fxp(ir1+self.epsilon, True, Data_n_word[counter], Data_n_word[counter]-1, rounding = self.method2)
                        ii1 = Fxp(ii1+self.epsilon, True, Data_n_word[counter], Data_n_word[counter]-1, rounding = self.method2)
                        # print(rr1,ri1,ir1,ii1)
                        
                        OUT_vec_fxp[n*j+k][index_c+1] = (rr0-ii0)+1j*(ri0+ir0)#Fxp((rr0-ii0)+1j*(ri0+ir0), True, Data_n_word[counter], Data_n_word[counter] - 1)
                        OUT_vec_fxp[n*j+k+step_len][index_c+1] = (rr1-ii1)+1j*(ri1+ir1)#Fxp((rr1-ii1)+1j*(ri1+ir1), True, Data_n_word[counter], Data_n_word[counter] - 1)
                        
                        # OUT_vec_fxp[n*j+k][index_c+1] = Fxp(OUT_vec_fxp[n*j+k][index_c+1], True, Data_n_word[counter], Data_n_word[counter] - 1)
                        # OUT_vec_fxp[n*j+k+step_len][index_c+1] = Fxp(OUT_vec_fxp[n*j+k+step_len][index_c+1], True, Data_n_word[counter], Data_n_word[counter] - 1)
                        
                    else:
                        # BF_res = self.BF_r2(OUT_vec_fxp[n*j+k][index_c], OUT_vec_fxp[n*j+k+step_len][index_c])

                        OUT_vec_fxp[n*j+k][index_c+1] = (OUT_vec_fxp[n*j+k][index_c] + OUT_vec_fxp[n*j+k+step_len][index_c])*0.5
                        OUT_vec_fxp[n*j+k+step_len][index_c+1] =(OUT_vec_fxp[n*j+k][index_c] - OUT_vec_fxp[n*j+k+step_len][index_c])*0.5
                        # align the addition output datawordlenth with the In_n_word
                        OUT_vec_fxp[n*j+k][index_c+1] = Fxp(OUT_vec_fxp[n*j+k][index_c+1]+(1+1j)*self.epsilon, True, Data_n_word[counter-1], Data_n_word[counter-1]-1, rounding = self.method2)
                        OUT_vec_fxp[n*j+k+step_len][index_c+1] = Fxp(OUT_vec_fxp[n*j+k+step_len][index_c+1]+(1+1j)*self.epsilon, True, Data_n_word[counter-1], Data_n_word[counter-1]-1, rounding = self.method2)                        
                        
                        TF_array[index_c][n*j+k] = Fxp(self.TF_array_accu[index_c][n*j+k], True, TF_n_word[counter], TF_n_word[counter]-2)
                        TF_array[index_c][n*j+k+step_len] = Fxp(self.TF_array_accu[index_c][n*j+k+step_len], True, TF_n_word[counter], TF_n_word[counter]-2)
                        
                        rr0, ri0, ir0, ii0 = self.multiplication(OUT_vec_fxp[n*j+k][index_c+1],TF_array[index_c][n*j+k])
                        rr0 = Fxp(rr0+self.epsilon, True, Data_n_word[counter], Data_n_word[counter]-1, rounding = self.method2)
                        ri0 = Fxp(ri0+self.epsilon, True, Data_n_word[counter], Data_n_word[counter]-1, rounding = self.method2)
                        ir0 = Fxp(ir0+self.epsilon, True, Data_n_word[counter], Data_n_word[counter]-1, rounding = self.method2)
                        ii0 = Fxp(ii0+self.epsilon, True, Data_n_word[counter], Data_n_word[counter]-1, rounding = self.method2)
                        
                        rr1, ri1, ir1, ii1 = self.multiplication(OUT_vec_fxp[n*j+k+step_len][index_c+1], TF_array[index_c][n*j+k+step_len])
                        rr1 = Fxp(rr1+self.epsilon, True, Data_n_word[counter], Data_n_word[counter]-1, rounding = self.method2)
                        ri1 = Fxp(ri1+self.epsilon, True, Data_n_word[counter], Data_n_word[counter]-1, rounding = self.method2)
                        ir1 = Fxp(ir1+self.epsilon, True, Data_n_word[counter], Data_n_word[counter]-1, rounding = self.method2)
                        ii1 = Fxp(ii1+self.epsilon, True, Data_n_word[counter], Data_n_word[counter]-1, rounding = self.method2)
                        
                        OUT_vec_fxp[n*j+k][index_c+1] = (rr0-ii0)+1j*(ri0+ir0)#Fxp((rr0-ii0)+1j*(ri0+ir0), True, Data_n_word[counter], Data_n_word[counter] - 1)
                        OUT_vec_fxp[n*j+k+step_len][index_c+1] = (rr1-ii1)+1j*(ri1+ir1) #Fxp((rr1-ii1)+1j*(ri1+ir1), True, Data_n_word[counter], Data_n_word[counter] - 1)
                        
                        # OUT_vec_fxp[n*j+k][index_c+1], OUT_vec_fxp[n*j+k+step_len][index_c+1] = OUT_vec_fxp[n*j+k][index_c+1]*TF_array[index_c][n*j+k], OUT_vec_fxp[n*j+k+step_len][index_c+1]*TF_array[index_c][n*j+k+step_len]
                        # OUT_vec_fxp[n*j+k][index_c+1] = Fxp(OUT_vec_fxp[n*j+k][index_c+1], True, Data_n_word[counter], Data_n_word[counter] - 1)
                        # OUT_vec_fxp[n*j+k+step_len][index_c+1] = Fxp(OUT_vec_fxp[n*j+k+step_len][index_c+1], True, Data_n_word[counter], Data_n_word[counter] - 1)
                        
                    
            counter = counter + 1  
            # print("counter", counter)          
                                        
        # bit reverse order of output
        self.final_res_fxp = np.zeros(self.N,dtype = np.cdouble) 
        # final_res_fxp = Fxp(final_res, True, n_word, n_frac)
        for m in range (self.N):
            self.final_res_fxp[m] = OUT_vec_fxp[self.bit_reverse(m)][index_c+1]
        
        
        OUT_vec_fxp[:,-1] = self.final_res_fxp
       
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
        
        arbr = a.real * b.real
        arbi = a.real * b.imag
        aibr = a.imag * b.real
        aibi = a.imag * b.imag
        
        
        return arbr, arbi, aibr, aibi
        
        
# FFT_R22 = R22_DIF_FFT(64,16,16)
# initial_n_word = FFT_R22.n_Word
# # parameter initialization
# In_n_word = 16
# # Data_n_word = [initial_n_word]*(FFT_R22.stages) # first cell is used for the 0.stage addition results wordlength
# Data_n_word = [16, 16, 16, 16, 16, 16]
# TF_n_word = [initial_n_word]*FFT_R22.stages

# IN_vec = FFT_R22.random_vector(0)
# print(IN_vec)
# FFT_R22.TF_Gen()
# FFT_R22.FFT_Accuracy(IN_vec)
# # print(FFT_R22.final_res_accu)
# FFT_R22.FFT_Fixed(IN_vec, In_n_word, Data_n_word, TF_n_word)
# # print(FFT_R22.final_res_fxp)
# FFT_R22.sqnr_calculation()

# print(FFT_R22.SNR_flp_fxp)






# # In[2] Important！！！ use to change back to binary！！！
# # get real part and imaginary part
# real_part = np.real(FFT_R22.final_res_fxp)
# imaginary_part = np.imag(FFT_R22.final_res_fxp)

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


# # print(FFT_R22.final_res_accu)

# FFT_R22.sqnr_calculation()

# print(FFT_R22.SNR_flp_fxp)