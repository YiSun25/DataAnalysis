# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 20:41:29 2024

@author: yisss
"""
import numpy as np
from fxpmath import Fxp
import cmath
import math
import os
import random
import pandas as pd
import matplotlib.pyplot as plt

class R2_DIF_FFT:
    def __init__(self, N, stages, n_Word, n_Word_TF, method):
        self.N = N
        self.stages = int(math.log(N,2)) # total stages
        self.n_Word = n_Word
        self.n_Word_TF = n_Word_TF
        self.method = method
        
    # generate twiddle factor (floating point and fixed point)
    def TF_Gen(self):
        TF_array_accu = np.zeros((self.stages,self.N >> 1), dtype = np.cdouble)
        for i in range(self.stages):
            for j in range((2**(self.stages-1))//(2**i)):
                TF_array_accu[i][j] = cmath.exp((-1j) * 2 * math.pi * j / (2**(self.stages-i))) 
        # change TF_array to fixed point
        TF_array_fixed =  Fxp(TF_array_accu, True, self.n_Word_TF, (self.n_Word_TF - 2)) 
        return TF_array_accu, TF_array_fixed
    
    # DIF radix-2 BF
    def BF_r2(self, in0, in1, TF):
        # print(in0, in1, TF)
        out0 = complex(in0 + in1)/2 # multiply 1/2 for scaling, to avoid overflow, when do scaling in fixed point, note the 
        out1 = complex(in0 - in1)/2*TF
        # print(out0,out1)
        return out0, out1
    
    def bit_reverse(self,n):                             ### 这里本来是 bit_reverse(n), 但是n的值是变量self.N,所以这里是否需要修改？？？
        result = 0
        for i in range (self.stages):                  ### 所有的 stages 还有N都可以写在最前面吗？
            result <<= 1
            result |= n & 1
            n >>= 1
        return result  # e.g. if n = 1, result = 4
    
    def FFT_Accuracy(self, IN_vec, TF_array):   # maybe later can also add the variable of fraction number , r=1
        OUT_vec_accu = np.zeros((self.N, self.stages+1), dtype = np.cdouble)
        OUT_vec_accu[:,0] = IN_vec
        

        for i in range(self.stages-1, -1, -1): # if N = 128, i = 6,5,4,3,2,1,0
            n = 2**(i+1) # total numbers of input
            step_len = 2**i
            index_c = int(math.log((self.N/n),2)) # index used for column
            for j in range (self.N//n):
                for k in range(step_len):
                    OUT_vec_accu[n*j+k][index_c+1], OUT_vec_accu[n*j+k+step_len][index_c+1] = self.BF_r2(OUT_vec_accu[n*j+k][index_c], OUT_vec_accu[n*j+k+step_len][index_c],TF_array[index_c][k])
                    
        final_res_accu = np.zeros(self.N,dtype = np.csingle) # complex64
        for m in range (self.N):
            final_res_accu[m] = OUT_vec_accu[self.bit_reverse(m)][index_c+1]      # ？？？ bit_reverse 函数的写法？？？
        
        OUT_vec_accu[:,-1] = final_res_accu
        
        return OUT_vec_accu
    
    def FFT_Fixed(self, In_vec, TF_array_default, TF_array_changed, n_word_addition, tf_change = 0, appt_stage = 0, approx = 0, Frac_Appr =0):
        self.n_Word = n_word
        OUT_vec_fxp = np.zeros((self.N,self.stages+1), dtype = np.cdouble)
        # OUT_vec_fxp = Fxp(OUT_vec_fxp, True, n_word, n_frac) 
        OUT_vec_fxp[:,0] = In_vec
        
       
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
                    
                    if (appr == 1):
                        if(counter == appt_stage):
                            OUT_vec_fxp[n*j+k][index_c+1] = Fxp((OUT_vec_fxp[n*j+k][index_c] + OUT_vec_fxp[n*j+k+step_len][index_c])*0.5, True, n_word , n_word-2)
                            OUT_vec_fxp[n*j+k+step_len][index_c+1] = Fxp((OUT_vec_fxp[n*j+k][index_c] - OUT_vec_fxp[n*j+k+step_len][index_c])*0.5, True, n_word, n_word-2)
                            # quantization for addition results
                            OUT_vec_fxp[n*j+k][index_c+1] = Fxp(OUT_vec_fxp[n*j+k][index_c+1], True, n_word - Frac_Appr, n_word - Frac_Appr -2, rounding = method)
                            OUT_vec_fxp[n*j+k+step_len][index_c+1] = Fxp(OUT_vec_fxp[n*j+k+step_len][index_c+1], True, n_word - Frac_Appr, n_word - Frac_Appr -2, rounding = method)
                                                                
                            OUT_vec_fxp[n*j+k+step_len][index_c+1] = OUT_vec_fxp[n*j+k+step_len][index_c+1] * TF_array[index_c][k]
                            
                            OUT_vec_fxp[n*j+k][index_c+1] = Fxp(OUT_vec_fxp[n*j+k][index_c+1], True, n_word, n_word - 2)
                            OUT_vec_fxp[n*j+k+step_len][index_c+1] = Fxp(OUT_vec_fxp[n*j+k+step_len][index_c+1], True, n_word, n_word -2)
                            
                        else:
                            OUT_vec_fxp[n*j+k][index_c+1] = Fxp((OUT_vec_fxp[n*j+k][index_c] + OUT_vec_fxp[n*j+k+step_len][index_c])*0.5, True, n_word, n_word - 2)
                            OUT_vec_fxp[n*j+k+step_len][index_c+1] = Fxp((OUT_vec_fxp[n*j+k][index_c] - OUT_vec_fxp[n*j+k+step_len][index_c])*0.5, True, n_word , n_word -2)
                            OUT_vec_fxp[n*j+k+step_len][index_c+1] = OUT_vec_fxp[n*j+k+step_len][index_c+1] * TF_array[index_c][k]
                            
                            OUT_vec_fxp[n*j+k][index_c+1] = Fxp(OUT_vec_fxp[n*j+k][index_c+1], True, n_word, n_word - 2)
                            OUT_vec_fxp[n*j+k+step_len][index_c+1] = Fxp(OUT_vec_fxp[n*j+k+step_len][index_c+1], True, n_word, n_word -2)
                            
                    else:
                        OUT_vec_fxp[n*j+k][index_c+1] = Fxp((OUT_vec_fxp[n*j+k][index_c] + OUT_vec_fxp[n*j+k+step_len][index_c])*0.5, True, n_word, n_word  -2)
                        OUT_vec_fxp[n*j+k+step_len][index_c+1] = Fxp((OUT_vec_fxp[n*j+k][index_c] - OUT_vec_fxp[n*j+k+step_len][index_c])*0.5, True, n_word , n_word -2)
                        OUT_vec_fxp[n*j+k+step_len][index_c+1] = OUT_vec_fxp[n*j+k+step_len][index_c+1] * TF_array[index_c][k]
                        
                        OUT_vec_fxp[n*j+k][index_c+1] = Fxp(OUT_vec_fxp[n*j+k][index_c+1], True, n_word, n_word - 2)
                        OUT_vec_fxp[n*j+k+step_len][index_c+1] = Fxp(OUT_vec_fxp[n*j+k+step_len][index_c+1], True, n_word, n_word -2)
            counter = counter + 1  
            # print("counter", counter)          
                                        
        # bit reverse order of output
        final_res_fxp = np.zeros(self.N,dtype = np.cdouble) 
        # final_res_fxp = Fxp(final_res, True, n_word, n_frac)
        for m in range (self.N):
            final_res_fxp[m] = OUT_vec_fxp[self.bit_reverse(m)][index_c+1]
        
        
        OUT_vec_fxp[:,-1] = final_res_fxp
       
        return OUT_vec_fxp, final_res_fxp
    
    
    # generate input vector
    def random_vector(self, fft_length, start, end, seed_num): 
       random_in = []
       random.seed(seed_num)
       for _ in range(fft_length):
           # real input
            rand_num = random.uniform(start, end)  # random.uniform includes the upper bound and lower bounder, but np.random.uniform doesn't includes the upper bound
            random_in.append(rand_num)
           
           # complex input
            # real_part = random.uniform(start, end)
            # imag_part = random.uniform(start, end)
            # complex_num = complex(real_part, imag_part)
            # random_in.append(complex_num)
            
       return random_in
   
    # def adjust_wordlength(self, TF_array, initial_n_word = 16, target_SQNR = 50):
        
    #     n_word_array = [initial_n_word]*self.stages
        
        
    #     for i in range (self.stages):
            
        
   
    # # def get_xxx(self):
    # #     return
    
    # # def set_xxx(self):
    # #     xxx = xxx