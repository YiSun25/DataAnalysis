# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 15:45:43 2023

@author: yisss
"""

import numpy as np
import pandas as pd
# import Radix2_DIF_forloop
import FFT
# import FFT2
# import csv
import os
import math

# # when the seed in file Radix_DIF_forloop changes, the variable_value should also be changed.
# #####################################################
total_num = FFT.N
stage_num = FFT.stages
seed_ = FFT.seed_num

word_len = FFT.n_word
frac_len = FFT.n_frac
appr_frac_len = FFT.n_frac_appr  # for fraction number
#######################################################
FFT_results = FFT.main()

# real matrix, store accuracy results in each steps, last column is the final result
flp_matri = FFT_results[0]
# final results of accuracy calculation, last column is the final result
accu_vec_abs = np.abs(flp_matri[:,-1])

# stroe the fixed point result, each stage with 16 bits
fixed_matri = FFT_results[1]

# store the approximate final results, when do approximation in different stages
appr_final = FFT_results[2]



def get_results():
    
    # max absolute results of appr_final
    appr_final_abs = np.abs(appr_final)
    max_appr_abs = np.amax(appr_final_abs, axis = 0)
    # (max(APR))^2
    squa_max_appr_abs = np.square(max_appr_abs)
    
    
    # max abs fixed point result(16 bits)
    max_fxp_res = np.amax(np.abs(fixed_matri[:,-1]))
    # square of the max_fxp_res  (max(APR_fxp))^2
    squa_max_fxp_res = np.square(max_fxp_res)
    
   
    return max_appr_abs, squa_max_appr_abs, squa_max_fxp_res


def get_error():
    
    accu_res = flp_matri[:,-1] 
    accu_res_fxp = fixed_matri[:,-1]
    
    
    # errors betwwen accuracy result(floating point) and approximate results(fixed point results with approximation)
    error = np.zeros((total_num, stage_num), dtype = np.cdouble)
    for i in range (stage_num):
        error[:,i] = accu_res - appr_final[:,i]  
        
        
    # absolute value of error array
    abs_err = np.abs(error)
    # maximum value of the absolute error from each approximation results
    max_absErr = np.amax(abs_err, axis = 0) 
    
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
    
    return abs_err, max_absErr, sum_squa_accu, sum_squa_abs_err, sum_err_flpfxp



def calculation_Metrics(): 
    # get results
    Results = get_results()
    # get error
    Err_Res = get_error()
    
    
    # SNR, MSE, PSNR
    SNR = np.zeros(stage_num) 
    MSE = np.zeros(stage_num)
    PSNR = np.zeros(stage_num)
    
    for j in range (stage_num):
        SNR[j] = 10*math.log10(Err_Res[2]/Err_Res[3][j])  #sum_squa_real
        MSE[j] = Err_Res[3][j]/total_num
        PSNR[j] = 10*math.log10(Results[1][j]/MSE[j]) 
     
        
    # SNR, MSE, PSNR between floating point results and fixed point results with 16 bits data
    SNR_flp_fxp = 10*math.log10(Err_Res[2]/Err_Res[4])
    MSE_flp_fxp = Err_Res[4]/total_num
    PSNR_flp_fxp = 10*math.log10(Results[2]/MSE_flp_fxp)
    
    return SNR, MSE, PSNR, SNR_flp_fxp, MSE_flp_fxp, PSNR_flp_fxp


# print(get_results()[0][:,-1])
# print(get_results()[4][:,-1])

# print(calculation_Metrics()[0])
# print(np.square(get_results()[5]))
# print(calculation_Metrics()[5])

# for check
# get_results()
# print(get_results()[5])
# print(get_error()[2])
# print(get_error()[5])
# print(calculation_Metrics()[0])
# print(calculation_Metrics()[5])   
# check 
# print(calculation_Metrics()[1])
# print(calculation_Metrics()[1].dtype)
# print(calculation_Metrics()[2])
# print(get_error()[2])
# print(get_results()[0][:,-1])
# print(get_results()[1])
# print(get_error()[2])
# print(get_results()[0].dtype)
# print(get_results()[1].dtype)
# print(get_results()[2].dtype)
# print(get_results()[3].dtype)


# generate data frame

df_accu = pd.DataFrame(flp_matri)                        # accu_matri
df_appr = pd.DataFrame(appr_final)                       # appr_vec
df_fxp  = pd.DataFrame(fixed_matri)                      # fxp_matri
df_accu_abs = pd.DataFrame(accu_vec_abs)                 # accu_vec_abs  

Err = get_error()
df_abs_error = pd.DataFrame(Err[0])                      # abs_err
df_max_absErr = pd.DataFrame(Err[1])                     # max_absErr

df_SQNR = pd.DataFrame(calculation_Metrics()[0])         # SNR
df_MSE = pd.DataFrame(calculation_Metrics()[1])          # MSE
df_PSNR = pd.DataFrame(calculation_Metrics()[2])         # PSNR

df_SQNR_flp_fxp = pd.DataFrame(calculation_Metrics()[3],index=['Result'],columns=['SQNR'])  # SNR_flp_fxp
df_MSE_flp_fxp = pd.DataFrame(calculation_Metrics()[4],index=['Result'],columns=['MSE'])  # MSE_flp_fxp
df_PSNR_flp_fxp = pd.DataFrame(calculation_Metrics()[5],index=['Result'],columns=['PSNR']) # PSNR_flp_fxp




# #####################################################
filename_accu = "Accu_.csv"
filename_appr = f"Appr_frac_{appr_frac_len}.csv"
filename_accu_abs = "Accu_vec_abs.csv"
filename_fxp = "Fxp_16.csv"


filename_absErr = f"AbsErr_frac_{appr_frac_len}.csv"
filename_max_absErr = f"Max_absErr_frac_{appr_frac_len}.csv"

filename_SQNR = f"SQNR_frac_{appr_frac_len}.csv"
filename_MSE = f"MSE_frac_{appr_frac_len}.csv"
filename_PSNR = f"PSNR_frac_{appr_frac_len}.csv"

filename_SQNR_flp_fxp = f"SQNR_lx_word_{word_len}.csv"
filename_MSE_flp_fxp = f"MSE_lx_word_{word_len}.csv"
filename_PSNR_flp_fxp = f"PSNR_lx_word_{word_len}.csv"

# filename_absErr_sum = f"Sum_Squa_AbsErr_frac_{appr_frac_len}.csv"

#####################################################
# for FFT
folder_name = f'Test_Results_1/Data_Results/N_{total_num}_Word_{word_len}_Frac_{frac_len}_Seed_{seed_}' # remember to change when the N and fraction in another code.

# for FFT2
# folder_name = f'Test_Results_2/Data_Results/N_{total_num}_Word_{word_len}_Frac_{frac_len}_Seed_{seed_}' 
# #####################################################
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

####
file_path = os.path.join(folder_name, filename_accu)
df_accu.to_csv(file_path)#, index = False

file_path = os.path.join(folder_name, filename_appr)
df_appr.to_csv(file_path)

file_path = os.path.join(folder_name, filename_accu_abs)
df_accu_abs.to_csv(file_path)

file_path = os.path.join(folder_name, filename_fxp)
df_fxp.to_csv(file_path)#, index = False




####
file_path = os.path.join(folder_name, filename_absErr)
df_abs_error.to_csv(file_path)

file_path = os.path.join(folder_name, filename_max_absErr)
df_max_absErr.to_csv(file_path)


#####
file_path = os.path.join(folder_name, filename_SQNR)
df_SQNR.to_csv(file_path)

file_path = os.path.join(folder_name, filename_MSE)
df_MSE.to_csv(file_path)

file_path = os.path.join(folder_name, filename_PSNR)
df_PSNR.to_csv(file_path)


#####
file_path = os.path.join(folder_name, filename_SQNR_flp_fxp)
df_SQNR_flp_fxp.to_csv(file_path)

file_path = os.path.join(folder_name, filename_MSE_flp_fxp)
df_MSE_flp_fxp.to_csv(file_path)

file_path = os.path.join(folder_name, filename_PSNR_flp_fxp)
df_PSNR_flp_fxp.to_csv(file_path)









