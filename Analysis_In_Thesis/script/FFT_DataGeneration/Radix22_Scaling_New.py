# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 17:21:31 2024

@author: yisss
"""

import numpy as np
from fxpmath import Fxp
import cmath
import math
import os
import struct
import random
import pandas as pd
import matplotlib.pyplot as plt


# parameters(should be changed in different simulation)
##############################################
N = 64
stages = int(math.log(N,2)) # total stages

n_word = 16
n_frac = 15 

n_word_TF = 16
n_frac_TF = 14 # 2 bits for sign and integer, 14 bits for fraction

##############################################
n_frac_appr = 8
# n_word_new = n_word - (n_frac - n_frac_appr) # to maintain the bits of integer parts 

tiny_bias = 0
# seed_num = 0
method = 'trunc'
# method = 'around' # for round half up, should change tiny_bias to 1e-15 or 1e-16
###############################################


def TF22_gen(n):
    TF_array_accu = np.zeros((stages,N), dtype = np.cdouble)
    for i in range(stages):
        if (i%2 == 0):
            # print(i)
            group_num = int(N/(2**(stages-i)))
            # print(group_num)
            group_mem = int(N/(2**i))
            # print(group_mem)
            TF_array_accu[i,:] = 1
            for j in range(group_num):
                for k in range (int(N//(2**i)*3/4), N//(2**i)):
                    # print(i,j*group_mem + k )
                    TF_array_accu[i][j*group_mem + k] = -1j
            # print(TF_array_accu)
        
        if (i%2 == 1):
            # print(i)
            group_mem1 = N/(2**(i-1))
            group_num1 = N/group_mem1
            step = N/(2**(i+1))
            for g in range (int(group_num1)):
                for l in range (2):
                    for m in range (2):
                        for s in range (int(step)):
                            TF_array_accu[i][int(step*(2*l+m)+s)+g*int(group_mem1)] = cmath.exp((-1j) * 2 * math.pi * s * (l+2*m) / (2**(stages-(i-1))))
    # print(TF_array_accu)
    # TF_array_fixed =  Fxp(TF_array_accu, True, n_word_TF, n_frac_TF)
    return TF_array_accu#, TF_array_fixed


       
# DIF radix-2 BF
def BF_r2(in0, in1):
    # print(in0, in1, TF)
    out0 = complex(in0 + in1)/2 # multiply 1/2 for scaling, to avoid overflow, when do scaling in fixed point, note the 
    out1 = complex(in0 - in1)/2 
    # print(out0,out1)
    return out0, out1

def multiplication(a, b):
    # print(a,b)
    # print("real")
    # print(a.real, a.imag, b.real, b.imag)
    arbr = a.real * b.real
    arbi = a.real * b.imag
    aibr = a.imag * b.real
    aibi = a.imag * b.imag
    
    # print("arbr:", arbr, "arbi:", arbi, "aibr:", aibr, "aibi:", aibi)  # Debug print
    return arbr, arbi, aibr, aibi


# bit reverse order
def bit_reverse(n):
    result = 0
    for i in range (stages):
        result <<= 1
        result |= n & 1
        n >>= 1
    return result  # e.g. if n = 1, result = 4


# Floating point FFT implementation
def FFT_Accuracy(IN_vec, TF_array):   # maybe later can also add the variable of fraction number , r=1
    OUT_vec_accu = np.zeros((N,stages+1), dtype = np.cdouble)
    # OUT_vec_accu[:,0] = IN_vec
    OUT_vec_accu[:,0] = Fxp(IN_vec+tiny_bias, True, 16, 15, rounding = method)
    
    # print('In_vec', IN_vec)

    for i in range(stages-1, -1, -1): # if N = 128, i = 6,5,4,3,2,1,0
        n = 2**(i+1) # total numbers of input
        step_len = 2**i
        index_c = int(math.log((N/n),2)) # index used for column
        for j in range (N//n):
            for k in range(step_len):
                BF_res = BF_r2(OUT_vec_accu[n*j+k][index_c], OUT_vec_accu[n*j+k+step_len][index_c])
                # print('BF_res', BF_res)
                OUT_vec_accu[n*j+k][index_c+1], OUT_vec_accu[n*j+k+step_len][index_c+1] = BF_res[0]*TF_array[index_c][n*j+k], BF_res[1]*TF_array[index_c][n*j+k+step_len]
                # print('OUT_vec_accu', OUT_vec)
    final_res_accu = np.zeros(N,dtype = np.cdouble) # complex64
    for m in range (N):
        final_res_accu[m] = OUT_vec_accu[bit_reverse(m)][index_c+1]
    
    OUT_vec_accu[:,-1] = final_res_accu
    
    return OUT_vec_accu

    
# Fixed point FFT implementation    
def FFT_Fixed(IN_vec, TF_array, approx=0, appt_stage=0, Frac_Appr=0):
    OUT_vec_fxp = np.zeros((N,stages+1), dtype = np.cdouble)
    # OUT_vec_fxp = Fxp(OUT_vec_fxp, True, n_word, n_frac) 
    OUT_vec_fxp[:,0] = IN_vec
   
    counter = 0  # counter for stage number
    appr = approx     # if appr = 1, do approximation.
    
    for i in range(stages-1, -1, -1): # if N = 128, i = 6,5,4,3,2,1,0
        #print(i)
        n = 2**(i+1) # total numbers of input
        step_len = 2**i
        index_c = int(math.log((N/n),2)) # index used for column
        for j in range (N//n):
            for k in range(step_len):
                
                OUT_vec_fxp[n*j+k][index_c+1] = (OUT_vec_fxp[n*j+k][index_c] + OUT_vec_fxp[n*j+k+step_len][index_c])*0.5
                OUT_vec_fxp[n*j+k+step_len][index_c+1] = (OUT_vec_fxp[n*j+k][index_c] - OUT_vec_fxp[n*j+k+step_len][index_c])*0.5
                # align the addition output datawordlenth with the In_n_word
                OUT_vec_fxp[n*j+k][index_c+1] = Fxp(OUT_vec_fxp[n*j+k][index_c+1]+(1+1j)*tiny_bias, True, n_word, n_word - 1, rounding = method)
                OUT_vec_fxp[n*j+k+step_len][index_c+1] = Fxp(OUT_vec_fxp[n*j+k+step_len][index_c+1]+(1+1j)*tiny_bias, True, n_word, n_word -1, rounding = method)
                
                TF_array[index_c][n*j+k] = Fxp(TF_array[index_c][n*j+k], True, n_word_TF, n_frac_TF)
                TF_array[index_c][n*j+k+step_len] = Fxp(TF_array[index_c][n*j+k+step_len], True, n_word_TF, n_frac_TF)
                
                # OUT_vec_fxp[n*j+k][index_c+1], OUT_vec_fxp[n*j+k+step_len][index_c+1] = OUT_vec_fxp[n*j+k][index_c+1]*TF_array[index_c][n*j+k], OUT_vec_fxp[n*j+k+step_len][index_c+1]*TF_array[index_c][n*j+k+step_len]
                
                rr0, ri0, ir0, ii0 = multiplication(OUT_vec_fxp[n*j+k][index_c+1],TF_array[index_c][n*j+k])
                
                rr0 = rr0 + tiny_bias#2**(-(n_word))
                ri0 = ri0 + tiny_bias#2**(-(n_word))
                ir0 = ir0 + tiny_bias#2**(-(n_word))
                ii0 = ii0 + tiny_bias#2**(-(n_word))
                rr0 = Fxp(rr0, True, n_word, n_word - 1, rounding = method)
                ri0 = Fxp(ri0, True, n_word, n_word - 1, rounding = method)
                ir0 = Fxp(ir0, True, n_word, n_word - 1, rounding = method)
                ii0 = Fxp(ii0, True, n_word, n_word - 1, rounding = method)
                # print(rr0,ri0,ir0,ii0)
                
                rr1, ri1, ir1, ii1 = multiplication(OUT_vec_fxp[n*j+k+step_len][index_c+1],TF_array[index_c][n*j+k+step_len])
                
                rr1 = rr1 + tiny_bias#2**(-(n_word))
                ri1 = ri1 + tiny_bias#2**(-(n_word))
                ir1 = ir1 + tiny_bias#2**(-(n_word))
                ii1 = ii1 + tiny_bias#2**(-(n_word))
                rr1 = Fxp(rr1, True, n_word, n_word - 1, rounding = method)
                ri1 = Fxp(ri1, True, n_word, n_word - 1, rounding = method)
                ir1 = Fxp(ir1, True, n_word, n_word - 1, rounding = method)
                ii1 = Fxp(ii1, True, n_word, n_word - 1, rounding = method)
                # print(rr1,ri1,ir1,ii1)
                
                OUT_vec_fxp[n*j+k][index_c+1] = (rr0-ii0)+1j*(ri0+ir0) #Fxp(, True, n_word, n_word - 1)
                OUT_vec_fxp[n*j+k+step_len][index_c+1] = (rr1-ii1)+1j*(ri1+ir1) #Fxp(, True, n_word, n_word - 1)
                
                # BF_res = BF_r2(OUT_vec_fxp[n*j+k][index_c], OUT_vec_fxp[n*j+k+step_len][index_c])
                # OUT_vec_fxp[n*j+k][index_c+1], OUT_vec_fxp[n*j+k+step_len][index_c+1] = BF_res[0]*TF_array[index_c][n*j+k], BF_res[1]*TF_array[index_c][n*j+k+step_len]
                # # change it back to n_word, n_frac fixed point
                # OUT_vec_fxp[n*j+k][index_c+1] = Fxp(OUT_vec_fxp[n*j+k][index_c+1], True, n_word, n_frac-counter-1)
                # OUT_vec_fxp[n*j+k+step_len][index_c+1] = Fxp(OUT_vec_fxp[n*j+k+step_len][index_c+1], True, n_word, n_frac-counter-1)
             
                if (appr == 1):
                    if (counter == appt_stage):  # approximation stage number
                        OUT_vec_fxp[n*j+k][index_c+1] = OUT_vec_fxp[n*j+k][index_c+1] + (1+1j)*tiny_bias
                        OUT_vec_fxp[n*j+k+step_len][index_c+1] = OUT_vec_fxp[n*j+k+step_len][index_c+1] +  (1+1j)*tiny_bias
                    
                        OUT_vec_fxp[n*j+k][index_c+1], OUT_vec_fxp[n*j+k+step_len][index_c+1] = Fxp(OUT_vec_fxp[n*j+k][index_c+1], True, Frac_Appr+1, Frac_Appr, rounding = method), Fxp(OUT_vec_fxp[n*j+k+step_len][index_c+1], True, Frac_Appr+1, Frac_Appr, rounding = method)
                        # print("1")                                                     
                        # print(OUT_vec_fxp[n*j+k][index_c+1].bin(frac_dot=(True)), OUT_vec_fxp[n*j+k+step_len][index_c+1].bin(frac_dot=(True)))
        counter = counter + 1  
        # print("counter", counter)          
                                    
    # bit reverse order of output
    final_res_fxp = np.zeros(N,dtype = np.cdouble) 
    # final_res_fxp = Fxp(final_res, True, n_word, n_frac)
    for m in range (N):
        final_res_fxp[m] = OUT_vec_fxp[bit_reverse(m)][index_c+1]
    
    
    OUT_vec_fxp[:,-1] = final_res_fxp
   
    return OUT_vec_fxp, final_res_fxp


# generate input vector
def random_vector(fft_length, start, end, seed_num): 
   random_in = []
   random.seed(seed_num)
   for _ in range(fft_length):
       # real input
        rand_num = random.uniform(start, end)
        random_in.append(rand_num)
   
       # complex input
       # real_part = random.uniform(start, end)
       # imag_part = random.uniform(start, end)
       # complex_num = complex(real_part, imag_part)
       # random_in.append(complex_num)
       
   return random_in


# check correction of fft results
def check():
        poly_vec = np.array(random_vector(N, -1, 1, 0), dtype = np.cdouble)
        # poly_vec_fixed = Fxp(poly_vec, True, n_word, n_frac)
        
        TF_array_accu = TF22_gen(N)[0]
        flp_matri = FFT_Accuracy(poly_vec, TF_array_accu)
        # check if the results of fft are correct
        expected_result = np.fft.fft(poly_vec)  # Use NumPy FFT as a reference
        is_correct = np.allclose(flp_matri[:,-1], expected_result)
        print(flp_matri[:,-1])
        print(expected_result)
        print("Is the result correct?", is_correct)
        
        return 0
        

def main():
    for seed in range (100):
        seed_folder = f"Test_Results_scaling_new/Data_Results_AftMid/N{N}_Trunc/N_{N}_Seed_{seed}"
        if not os.path.exists(seed_folder):
            os.makedirs(seed_folder)
        
        # input vector
        # floating point input vector
        poly_vec = np.array(random_vector(N, -1, 1,seed), dtype = np.cdouble)
        
        # fixed point input vector with n_word and n_frac
        poly_vec_fixed = Fxp(poly_vec+tiny_bias, True, n_word, n_frac, rounding = method)
        
        # twiddle factor
        # floating point twiddle factor and fixed point twiddle factor
        TF_array = TF22_gen(N)
        # TF_array_accu = TF_array[0]
        # TF_array_fixed = TF_array[1]
        
        # accuracy result (floating point results)
        flp_matri = FFT_Accuracy(poly_vec, TF_array)
        # save flp_matri
        df_accu = pd.DataFrame(flp_matri) 
        filename_accu = "Accu_.csv"
        file_path = os.path.join(seed_folder, filename_accu)
        df_accu.to_csv(file_path)#, index = False
        
        
        # fixed point result
        # fixed point with n_word bit in total and n_frac fraction bits
        fixed_matri = FFT_Fixed(poly_vec_fixed, TF_array)[0]
        # save fxp_matri
        df_fxp = pd.DataFrame(fixed_matri) 
        filename_fxp = "Fxp_16.csv"
        file_path = os.path.join(seed_folder, filename_fxp)
        df_fxp.to_csv(file_path)#, index = False
        
        
        # Abs of floating point final result
        accu_vec_abs = np.abs(flp_matri[:,-1])
        # save accu_vec_abs
        df_accu_abs = pd.DataFrame(accu_vec_abs)
        filename_accu_abs = "Accu_vec_abs.csv"
        file_path = os.path.join(seed_folder, filename_accu_abs)
        df_accu_abs.to_csv(file_path)
        
        accu_res = flp_matri[:,-1]
        accu_res_fxp = fixed_matri[:,-1]
    
        # sum of abusolute accuracy  values per column  sum(||x_real||^2)
        squa_accu_vec_abs = np.square(accu_vec_abs)
        sum_squa_accu = sum(squa_accu_vec_abs)   
        
        # error between flp results and fxp results (with each stage 16 bits data)
        flp_fxp_error = accu_res - accu_res_fxp
        sum_err_flpfxp = np.sum(np.square(np.abs(flp_fxp_error)))
        
        # max abs fixed point result(16 bits)
        # max_fxp_res = np.amax(np.abs(fixed_matri[:,-1]))
        # square of the max_fxp_res  (max(APR_fxp))^2
        # squa_max_fxp_res = np.square(max_fxp_res)
        
        # SNR, MSE, PSNR between floating point results and fixed point results with 16 bits data
        SNR_flp_fxp = 10*math.log10(sum_squa_accu/sum_err_flpfxp)
        # MSE_flp_fxp = sum_err_flpfxp/N
        # PSNR_flp_fxp = 10*math.log10(squa_max_fxp_res/MSE_flp_fxp)
        
        df_SQNR_flp_fxp = pd.DataFrame(SNR_flp_fxp, index=['Result'], columns=['SQNR'])   # SNR_flp_fxp
        # df_MSE_flp_fxp = pd.DataFrame(MSE_flp_fxp, index=['Result'], columns=['MSE'])     # MSE_flp_fxp
        # df_PSNR_flp_fxp = pd.DataFrame(PSNR_flp_fxp, index=['Result'], columns=['PSNR'])  # PSNR_flp_fxp
        
        filename_SQNR_flp_fxp = f"SQNR_lx_word_{n_word}.csv"
        # filename_MSE_flp_fxp = f"MSE_lx_word_{n_word}.csv"
        # filename_PSNR_flp_fxp = f"PSNR_lx_word_{n_word}.csv"
        
        file_path = os.path.join(seed_folder, filename_SQNR_flp_fxp)
        df_SQNR_flp_fxp.to_csv(file_path)
        # file_path = os.path.join(seed_folder, filename_MSE_flp_fxp)
        # df_MSE_flp_fxp.to_csv(file_path)
        # file_path = os.path.join(seed_folder, filename_PSNR_flp_fxp)
        # df_PSNR_flp_fxp.to_csv(file_path)
        
        
    
        # fixed point with approximation in each stage(i),and with different fraction(j) (reduce word length in different stages)
        array_size = (N, stages)
        appr_list = [np.zeros(array_size, dtype = np.cdouble)for _ in range(n_frac_appr)]
        # print(array_list)
        for j in range (8,15): # if max n_frac_appr equals to 7, then j is from 1 to 7
            for i in range (stages):
                appr_list[j-8][:,i] = FFT_Fixed(poly_vec_fixed, TF_array, 1, i, j)[1]
            
            appr_final = appr_list[j-8]
            df_appr = pd.DataFrame(appr_final)
            filename_appr = f"Appr_word_{j+1}.csv"
            file_path = os.path.join(seed_folder, filename_appr)
            df_appr.to_csv(file_path)
            
            
            #################################### get results ######################################
            # max absolute results of appr_final
            # appr_final_abs = np.abs(appr_final)
            # max_appr_abs = np.amax(appr_final_abs, axis = 0)
            # (max(APR))^2
            # squa_max_appr_abs = np.square(max_appr_abs)
            
            
            ###################################### get error #####################################
            # error between accu result and approximate results
            error = np.zeros((N, stages), dtype = np.cdouble)
            for m in range (stages):
                error[:,m] = accu_res - appr_final[:,m] 
            # absolute value of error array
            abs_err = np.abs(error)
            
            # df_abs_error = pd.DataFrame(abs_err)
            # filename_absErr = f"AbsErr_frac_{j}.csv"
            # file_path = os.path.join(seed_folder, filename_absErr)
            # df_abs_error.to_csv(file_path)
            
             
            # maximum value of the absolute error from each approximation results
            # max_absErr = np.amax(abs_err, axis = 0) 
            
            # df_max_absErr = pd.DataFrame(max_absErr) 
            # filename_max_absErr = f"Max_absErr_frac_{j}.csv"
            # file_path = os.path.join(seed_folder, filename_max_absErr)
            # df_max_absErr.to_csv(file_path)
            
            
            # sum (||abs_error||^2)
            squa_abs_err = np.square(abs_err)
            # print(squa_abs_err)
            sum_squa_abs_err = np.sum(squa_abs_err, axis = 0) # list
            # print(sum_squa_abs_err)
            
            
            ##################################### get metrics ######################################
            # SNR, MSE, PSNR
            SNR = np.zeros(stages) 
            # MSE = np.zeros(stages)
            # PSNR = np.zeros(stages)
            for k in range (stages):
                SNR[k] = 10*math.log10(sum_squa_accu/sum_squa_abs_err[k])  #sum_squa_real
                # MSE[k] = sum_squa_abs_err[k]/N
                # PSNR[k] = 10*math.log10(squa_max_appr_abs[k]/MSE[k])
                
            
            df_SQNR = pd.DataFrame(SNR)                              # SNR
            # df_MSE = pd.DataFrame(MSE)                               # MSE
            # df_PSNR = pd.DataFrame(PSNR)                             # PSNR
            
            
            filename_SQNR = f"SQNR_word_{j+1}.csv"
            # filename_MSE = f"MSE_frac_{j}.csv"
            # filename_PSNR = f"PSNR_frac_{j}.csv"
            # filename_SQNR_flp_fxp = f"SQNR_lx_word_{n_word}.csv"
            # filename_MSE_flp_fxp = f"MSE_lx_word_{n_word}.csv"
            # filename_PSNR_flp_fxp = f"PSNR_lx_word_{n_word}.csv"
            
            file_path = os.path.join(seed_folder, filename_SQNR)
            df_SQNR.to_csv(file_path)

            # file_path = os.path.join(seed_folder, filename_MSE)
            # df_MSE.to_csv(file_path)

            # file_path = os.path.join(seed_folder, filename_PSNR)
            # df_PSNR.to_csv(file_path)
            
            
            
            
        # print(appr_list[0][:,0]) # list 0 infer j = 1
            
        
        
       # appr_final = np.zeros((n_frac-1, N, stages), dtype = np.cdouble) # colunm 0 stored the approximate final result when do approximation in stage 0
        # for i in range (stages):
        #      appr_final[:,i] = FFT_Fixed( poly_vec_fixed, TF_array_fixed, 1, i, j)[1]
             
        # print (array_list)
    
    
    return flp_matri, fixed_matri#, appr_final

if __name__ == "__main__":
    main()
# check()