# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 19:13:31 2024

@author: yisss
"""

import numpy as np
from fxpmath import Fxp
import random
import cmath
import math
import os
import pandas as pd
# import matplotlib.pyplot as plt

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
n_word_new = n_word - (n_frac - n_frac_appr) # to maintain the bits of integer parts 

tiny_bias = 1e-16
# seed_num = 0

# method = 'trunc'
method = 'around'
###############################################

# twiddle factor
def TF_gen(n):
    TF_array_accu = np.zeros((stages,n >> 1), dtype = np.cdouble)
    for i in range(stages):
        for j in range(2**i):
            TF_array_accu[i][j] = cmath.exp((-1j) * 2 * math.pi * j / (2**(i+1))) 
    # change TF_array to fixed point
    # TF_array_fixed =  Fxp(TF_array_accu, True, n_word_TF, n_frac_TF)
    # print(TF_array)
    return TF_array_accu#, TF_array_fixed

            
# DIT radix-2 Butterfly
def BF_r2(in0, in1, TF):
    temp = in1 * complex(TF)
    #print(in1, temp, TF)
    out0 = complex(in0 + temp)/2
    out1 = complex(in0 - temp)/2
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

def bit_reverse(n):
    result = 0
    for i in range (stages):
        result <<= 1
        result |= n & 1
        n >>= 1
    return result  # e.g. if n = 1, result = 4


def FFT_Accu_DIT(IN_vec, TF_array):
    OUT_vec = np.zeros((N,stages+1), dtype = np.cdouble)
    # OUT_vec[:,0] = IN_vec
    OUT_vec[:,0] = Fxp(IN_vec+tiny_bias, True, n_word, n_word-1, rounding = method)
   
    for i in range(stages):
        n = 2**(i+1)
        step_len = 2**i
        for j in range(N//n):
            for k in range (step_len):
                OUT_vec[n*j+k][i+1], OUT_vec[n*j+k+step_len][i+1] =  BF_r2(OUT_vec[n*j+k][i], OUT_vec[n*j+k+step_len][i],TF_array[i][k])
                
    final_res = OUT_vec[:,-1]
    
    return OUT_vec, final_res

def FFT_Fixed_DIT(IN_vec, TF_array, approx=0, appt_stage=0, Frac_Appr=0):
    OUT_vec_fxp = np.zeros((N,stages+1), dtype = np.cdouble)
    OUT_vec_fxp[:,0] = IN_vec
   
    counter = 0  # counter for stage number
    appr = approx     # if appr = 1, do approximation.
    for i in range(stages):
        n = 2**(i+1)
        step_len = 2**i
        for j in range(N//n):
            for k in range (step_len):
                tf = Fxp(TF_array[i][k], True, n_word_TF, n_frac_TF)
                rr, ri, ir, ii = multiplication(OUT_vec_fxp[n*j+k+step_len][i], tf)
                
                rr = rr + tiny_bias
                ri = ri + tiny_bias
                ir = ir + tiny_bias
                ii = ii + tiny_bias
                
                rr = Fxp(rr, True, n_word, n_word-1, rounding = method)
                ri = Fxp(ri, True, n_word, n_word-1, rounding = method)
                ir = Fxp(ir, True, n_word, n_word-1, rounding = method)
                ii = Fxp(ii, True, n_word, n_word-1, rounding = method)
                
                # print(counter,rr, ri, ir, ii)
                temp = (rr-ii)+1j*(ri+ir)#Fxp((rr-ii)+1j*(ri+ir), True, n_word, n_word-1, rounding = method)
                OUT_vec_fxp[n*j+k][i+1] = OUT_vec_fxp[n*j+k][i] + temp
                OUT_vec_fxp[n*j+k+step_len][i+1] = OUT_vec_fxp[n*j+k][i] - temp
                
                OUT_vec_fxp[n*j+k][i+1] = Fxp(OUT_vec_fxp[n*j+k][i+1]*0.5+(1+1j)*tiny_bias, True, n_word, n_word - 1, rounding = method)
                OUT_vec_fxp[n*j+k+step_len][i+1] = Fxp(OUT_vec_fxp[n*j+k+step_len][i+1]*0.5+(1+1j)*tiny_bias, True, n_word, n_word - 1, rounding = method)
                # print(OUT_vec_fxp[n*j+k][i+1], OUT_vec_fxp[n*j+k+step_len][i+1] )
            
                # OUT_vec_fxp[n*j+k][i+1], OUT_vec_fxp[n*j+k+step_len][i+1] =  BF_r2(OUT_vec_fxp[n*j+k][i], OUT_vec_fxp[n*j+k+step_len][i],TF_array[i][k])
                # OUT_vec_fxp[n*j+k][i+1] = Fxp(OUT_vec_fxp[n*j+k][i+1], True, n_word, n_word-1)
                # OUT_vec_fxp[n*j+k+step_len][i+1] = Fxp(OUT_vec_fxp[n*j+k+step_len][i+1], True, n_word, n_word-1)
                # approximation
                if (counter == appt_stage):
                    if (appr == 1):  # approximation stage number
                        OUT_vec_fxp[n*j+k][i+1] =  OUT_vec_fxp[n*j+k][i+1] + (1+1j)*tiny_bias#(2**(-(Frac_Appr+1)))
                        OUT_vec_fxp[n*j+k+step_len][i+1] = OUT_vec_fxp[n*j+k+step_len][i+1] + (1+1j)*tiny_bias
                        
                        OUT_vec_fxp[n*j+k][i+1], OUT_vec_fxp[n*j+k+step_len][i+1] = Fxp(OUT_vec_fxp[n*j+k][i+1], True, Frac_Appr+1, Frac_Appr, rounding = method), Fxp(OUT_vec_fxp[n*j+k+step_len][i+1], True, Frac_Appr + 1 , Frac_Appr, rounding = method)
                        
        counter = counter + 1
    # print(OUT_vec)
    # store the final result
    # res = pd.DataFrame(OUT_vec)
    # res.to_csv("Radix2_DIT_Res1.csv")
    final_res_fxp = OUT_vec_fxp[:,-1]
    
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


# main function 
def main():
    for seed in range (100):
        seed_folder = f"Test_Results_Scaling_new/Data_Results_AftMid/N{N}_RHU_NEW/N_{N}_Seed_{seed}"
        if not os.path.exists(seed_folder):
            os.makedirs(seed_folder)
        
        # input vector
        # floating point input vector
        poly_vec = np.array(random_vector(N, -1, 1,seed), dtype = np.cdouble)
        Xn_R = np.zeros(N, dtype = np.cdouble)
        
        # bit reverse input
        for i in range (len(poly_vec)):
            Xn_R[bit_reverse(i)] = poly_vec[i]
            
        # fixed point input vector with n_word and n_frac
        Xn_R_fixed = Fxp(Xn_R+tiny_bias, True, n_word, n_word-1, rounding = method)
        
        # twiddle factor
        # floating point twiddle factor
        TF_array = TF_gen(N)
        
        # accuracy result (floating point results)
        flp_matri = FFT_Accu_DIT(Xn_R, TF_array)[0]
        # save flp_matri
        df_accu = pd.DataFrame(flp_matri) 
        filename_accu = "Accu_.csv"
        file_path = os.path.join(seed_folder, filename_accu)
        df_accu.to_csv(file_path)#, index = False
        
        # check if the results of fft are correct
        # expected_result = np.fft.fft(poly_vec)  # Use NumPy FFT as a reference
        # is_correct = np.allclose(accu_matri[:,-1], expected_result)
        # print("Is the result correct?", is_correct)
        
        
        # fixed point result
        # fixed point with n_word bit in total and n_frac fraction bits
        fixed_matri = FFT_Fixed_DIT(Xn_R_fixed, TF_array)[0]
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
    
       
        # fixed point with approximation in each stage(i),and with different fraction(j) (reduce word length in different stages)
        array_size = (N, stages)
        appr_list = [np.zeros(array_size, dtype = np.cdouble)for _ in range(n_frac_appr)]
        # print(array_list)
        for j in range (8,15): # if max n_frac_appr equals to 7, then j is from 1 to 7
            for i in range (stages):
                appr_list[j-8][:,i] = FFT_Fixed_DIT(Xn_R_fixed, TF_array, 1, i, j)[1]
            
            appr_final = appr_list[j-8]
            df_appr = pd.DataFrame(appr_final)
            filename_appr = f"Appr_word_{j+1}.csv"
            file_path = os.path.join(seed_folder, filename_appr)
            df_appr.to_csv(file_path)
            
            accu_res = flp_matri[:,-1]
            accu_res_fxp = fixed_matri[:,-1]
            
            #################################### get results ######################################
            # max absolute results of appr_final
            # appr_final_abs = np.abs(appr_final)
            # max_appr_abs = np.amax(appr_final_abs, axis = 0)
            # (max(APR))^2
            # squa_max_appr_abs = np.square(max_appr_abs)
            
            
            # max abs fixed point result(16 bits)
            # max_fxp_res = np.amax(np.abs(fixed_matri[:,-1]))
            # square of the max_fxp_res  (max(APR_fxp))^2
            # squa_max_fxp_res = np.square(max_fxp_res)
            
            
            ###################################### get error #####################################
            # error between accu result and approximate results
            error = np.zeros((N, stages), dtype = np.cdouble)
            for m in range (stages):
                error[:,m] = accu_res - appr_final[:,m] 
            # absolute value of error array
            abs_err = np.abs(error)
            
            # df_abs_error = pd.DataFrame(abs_err)
            # filename_absErr = f"AbsErr_word_{j+1}.csv"
            # file_path = os.path.join(seed_folder, filename_absErr)
            # df_abs_error.to_csv(file_path)
            
             
            # maximum value of the absolute error from each approximation results
            # max_absErr = np.amax(abs_err, axis = 0) 
            
            # df_max_absErr = pd.DataFrame(max_absErr) 
            # filename_max_absErr = f"Max_absErr_word_{j+n_word}.csv"
            # file_path = os.path.join(seed_folder, filename_max_absErr)
            # df_max_absErr.to_csv(file_path)
            
            # sum of abusolute accuracy  values per column  sum(||x_real||^2)
            squa_accu_vec_abs = np.square(accu_vec_abs)
            sum_squa_accu = sum(squa_accu_vec_abs)
            
            # sum (||abs_error||^2)
            squa_abs_err = np.square(abs_err)
            # print(squa_abs_err)
            sum_squa_abs_err = np.sum(squa_abs_err, axis = 0) # list
            # print(sum_squa_abs_err)
            
            # error between flp results and fxp results (with each stage 16 bits data)
            flp_fxp_error = accu_res - accu_res_fxp
            sum_err_flpfxp = np.sum(np.square(np.abs(flp_fxp_error)))
            
            ##################################### get metrics ######################################
            # SNR, MSE, PSNR
            SNR = np.zeros(stages) 
            # MSE = np.zeros(stages)
            # PSNR = np.zeros(stages)
            for k in range (stages):
                SNR[k] = 10*math.log10(sum_squa_accu/sum_squa_abs_err[k])  #sum_squa_real
                # MSE[k] = sum_squa_abs_err[k]/N
                # PSNR[k] = 10*math.log10(squa_max_appr_abs[k]/MSE[k])
                
            # SNR, MSE, PSNR between floating point results and fixed point results with 16 bits data
            SNR_flp_fxp = 10*math.log10(sum_squa_accu/sum_err_flpfxp)
            # MSE_flp_fxp = sum_err_flpfxp/N
            # PSNR_flp_fxp = 10*math.log10(squa_max_fxp_res/MSE_flp_fxp)
            
            df_SQNR = pd.DataFrame(SNR)                              # SNR
            # df_MSE = pd.DataFrame(MSE)                               # MSE
            # df_PSNR = pd.DataFrame(PSNR)                             # PSNR
            df_SQNR_flp_fxp = pd.DataFrame(SNR_flp_fxp, index=['Result'], columns=['SQNR'])   # SNR_flp_fxp
            # df_MSE_flp_fxp = pd.DataFrame(MSE_flp_fxp, index=['Result'], columns=['MSE'])     # MSE_flp_fxp
            # df_PSNR_flp_fxp = pd.DataFrame(PSNR_flp_fxp, index=['Result'], columns=['PSNR'])  # PSNR_flp_fxp
            
            
            filename_SQNR = f"SQNR_word_{j+1}.csv"
            # filename_MSE = f"MSE_word_{j+n_word}.csv"
            # filename_PSNR = f"PSNR_word_{j+n_word}.csv"
            filename_SQNR_flp_fxp = f"SQNR_lx_word_{n_word}.csv"
            # filename_MSE_flp_fxp = f"MSE_lx_word_{n_word}.csv"
            # filename_PSNR_flp_fxp = f"PSNR_lx_word_{n_word}.csv"
            
            file_path = os.path.join(seed_folder, filename_SQNR)
            df_SQNR.to_csv(file_path)

            # file_path = os.path.join(seed_folder, filename_MSE)
            # df_MSE.to_csv(file_path)

            # file_path = os.path.join(seed_folder, filename_PSNR)
            # df_PSNR.to_csv(file_path)


            #####
            file_path = os.path.join(seed_folder, filename_SQNR_flp_fxp)
            df_SQNR_flp_fxp.to_csv(file_path)

            # file_path = os.path.join(seed_folder, filename_MSE_flp_fxp)
            # df_MSE_flp_fxp.to_csv(file_path)

            # file_path = os.path.join(seed_folder, filename_PSNR_flp_fxp)
            # df_PSNR_flp_fxp.to_csv(file_path)
            
            
            
            
        # print(appr_list[0][:,0]) # list 0 infer j = 1
            
        
        
       # appr_final = np.zeros((n_frac-1, N, stages), dtype = np.cdouble) # colunm 0 stored the approximate final result when do approximation in stage 0
        # for i in range (stages):
        #      appr_final[:,i] = FFT_Fixed( poly_vec_fixed, TF_array_fixed, 1, i, j)[1]
             
        # print (array_list)
    
    
    return flp_matri, fixed_matri#, appr_final

if __name__ == "__main__":
        main()