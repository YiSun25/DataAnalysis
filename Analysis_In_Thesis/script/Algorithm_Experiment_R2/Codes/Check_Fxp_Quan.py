# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 17:25:10 2024

@author: yisss
"""
# In[1]
import numpy as np
import matplotlib.pyplot as plt
from fxpmath import Fxp
from matplotlib.ticker import MultipleLocator

# In[2]
fs = 1e5  # sampling frequency
n_samples = 2**10  # number of samples
f_sig = 2  # signal frequency
A = 1#0.8 # amplitude

n = np.arange(n_samples)
t = n / fs

sig = A * np.sin(2*np.pi*f_sig*n/n_samples)

# In[3]
plt.figure(figsize=(12,6))
plt.plot(n, sig)

# 设置纵坐标间隔为0.125
plt.gca().yaxis.set_major_locator(MultipleLocator(0.125))
plt.show()

# In[4]
signed = True
n_word = 8
n_frac = 2

roundings = ['around']#, 'trunc']

plt.figure(figsize=(12,6))
plt.plot(n, sig)
for r in roundings:
    sig_fxp = Fxp(sig+1e-12, signed=signed, n_word=n_word, n_frac=n_frac, rounding=r)
    plt.plot(n, sig_fxp, label=r)

# 设置纵坐标间隔为0.125
plt.gca().yaxis.set_major_locator(MultipleLocator(0.125))    

plt.legend()
plt.show()

# In[5] Banker's Rounding
a = -0.875 + (1e-15)*0.5
res = Fxp(a, signed=True, n_word=8, n_frac=2, rounding='around')
# print(res*res)
# res = res+2**(-4)
print(res)

# print(round(3.5))

# from fxpmath import Fxp
# import numpy as np

# def custom_rounding(data, signed=False, n_word=16, n_frac=8, rounding='round_half_up'):
#     # Create an Fxp object to get precision
#     fxp_obj = Fxp(signed=signed, n_word=n_word, n_frac=n_frac, rounding=rounding)
#     precision = fxp_obj.precision
    
#     if isinstance(data, np.ndarray):
#         if isinstance(data.item(0), complex):
#             # Round real and imaginary parts using Fxp with round_half_up rounding
#             data = data.real.round(precision=precision) + 1j * data.imag.round(precision=precision)
    
#     return Fxp(data, signed=signed, n_word=n_word, n_frac=n_frac, rounding=rounding)

# # Example usage
# data = np.array([0.125, 0.625])

# rounded_data = custom_rounding(data)

# print("Original data:")
# print(data)
# print("\nRounded data using round half up:")
# print(rounded_data)

# from fxpmath import Fxp
# import numpy as np

# def custom_rounding(data, signed=False, n_word=16, n_frac=2, rounding='around'):
#    precision = Fxp(signed=signed, n_word=n_word, n_frac=n_frac, rounding=rounding).precision
#    print('1111',precision)
#    if isinstance(data, np.ndarray):
#        if isinstance(data.item(0), complex):
#            data = data + np.sign(data.real)*precision*1e-8 + 1j*np.sign(data.imag)*precision*1e-8
#            return Fxp(data, signed=signed, n_word=n_word, n_frac=n_frac, rounding=rounding)

# # Example usage
# data = np.array([0.125, 0.125])

# rounded_data = custom_rounding(data)

# print("Original data:")
# # print(data)
# print("\nRounded data using round half up (modified Banker's rounding with bias):")
# print(rounded_data)

#%%
from fxpmath import Fxp
import numpy as np

def custom_rounding(data, signed=False, n_word=16, n_frac=2, rounding='around'):
    precision = Fxp(signed=signed, n_word=n_word, n_frac=n_frac, rounding=rounding).precision
    epsilon = precision * 1e-8
    if isinstance(data, np.ndarray):
        if isinstance(data.item(0), complex):
            # Add a small positive bias to data
            
            data = data + np.sign(data.real)*epsilon + 1j*np.sign(data.imag)*epsilon
        else:
            # If data is not complex, handle each element individually
            data = data + np.sign(data)*epsilon
    
    # Return as Fxp object
    return Fxp(data, signed=signed, n_word=n_word, n_frac=n_frac, rounding=rounding)

# Example usage
# data = np.array([0.125, 0.125])

rounded_data = custom_rounding(0.125+1j*0.125)

print("Original data:")
# print(data)
print("\nRounded data using round half up (modified Banker's rounding with bias):")
print(rounded_data)

#%%
from fxpmath import Fxp
import numpy as np

def custom_rounding(data, signed=False, n_word=16, n_frac=2, rounding='around'):
    precision = Fxp(signed=signed, n_word=n_word, n_frac=n_frac, rounding=rounding).precision
    epsilon = precision * 1e-8
    
    if isinstance(data, complex):
        # Handle complex numbers
        real_part = data.real + np.sign(data.real) * epsilon
        imag_part = data.imag + np.sign(data.imag) * epsilon
        rounded_data = complex(Fxp(real_part, signed=signed, n_word=n_word, n_frac=n_frac, rounding=rounding),
                               Fxp(imag_part, signed=signed, n_word=n_word, n_frac=n_frac, rounding=rounding))
    elif isinstance(data, np.ndarray):
        if isinstance(data.item(0), complex):
            # Handle complex numpy array
            real_part = data.real + np.sign(data.real) * epsilon
            imag_part = data.imag + np.sign(data.imag) * epsilon
            rounded_data = real_part + 1j * imag_part
        else:
            # Handle real numpy array
            rounded_data = data + np.sign(data) * epsilon
    else:
        # Handle single real number
        rounded_data = data + np.sign(data) * epsilon
    
    # Return as Fxp object
    return Fxp(rounded_data, signed=signed, n_word=n_word, n_frac=n_frac, rounding=rounding)

# Example usage
data = -0.625

rounded_data = custom_rounding(data)

print("Original data:")
print(data)
print("\nRounded data using round half up (modified Banker's rounding with bias):")
print(rounded_data)

#%%
from fxpmath import Fxp
import numpy as np

def custom_rounding(data, signed=False, n_word=16, n_frac=2, rounding='around'):
    # Determine precision based on Fxp configuration
    precision = Fxp(signed=signed, n_word=n_word, n_frac=n_frac, rounding=rounding).precision
    
    # Calculate epsilon for adjustment
    epsilon = precision * 1e-8
    
    if isinstance(data, complex):
        # Handle complex numbers
        real_part = data.real
        imag_part = data.imag
        
        # Apply Banker's rounding with epsilon adjustment
        if np.sign(real_part) == -1:
            real_part = np.round(real_part + np.sign(real_part) * epsilon)
        else:
            real_part = np.round(real_part)
        
        if np.sign(imag_part) == -1:
            imag_part = np.round(imag_part + np.sign(imag_part) * epsilon)
        else:
            imag_part = np.round(imag_part)
        
        # Create rounded complex number
        rounded_data = complex(real_part, imag_part)
        
    elif isinstance(data, np.ndarray):
        # Handle numpy arrays
        if isinstance(data.item(0), complex):
            # Handle complex numpy array
            real_part = data.real
            imag_part = data.imag
            
            real_part = np.where(np.sign(real_part) == -1,
                                 np.round(real_part + np.sign(real_part) * epsilon),
                                 np.round(real_part))
            
            imag_part = np.where(np.sign(imag_part) == -1,
                                 np.round(imag_part + np.sign(imag_part) * epsilon),
                                 np.round(imag_part))
            
            rounded_data = real_part + 1j * imag_part
            
        else:
            # Handle real numpy array
            rounded_data = np.where(np.sign(data) == -1,
                                    np.round(data + np.sign(data) * epsilon),
                                    np.round(data))
    
    else:
        # Handle single real number
        if np.sign(data) == -1:
            rounded_data = np.round(data + np.sign(data) * epsilon)
        else:
            rounded_data = np.round(data)
    
    # Return as Fxp object
    return Fxp(rounded_data, signed=signed, n_word=n_word, n_frac=n_frac, rounding=rounding)

# Example usage
data = -0.625

rounded_data = custom_rounding(data)

print("Original data:")
print(data)
print("\nRounded data using round half up (modified Banker's rounding with bias):")
print(rounded_data)


#%%
from numfi import numfi
import numpy as np
# from numfi import numqi as fi 

a = numfi(0.125, 1,5,1)
# print(a.bin())
print(a)

#%%
from fxpmath import Fxp
import numpy as np
epsilon = 1e-15
a = -3.3125
if (a>0):
    a = Fxp(a+epsilon, True, 8, 3, rounding = 'around')
else:
    a = Fxp(a-epsilon, True, 8, 3, rounding = 'around')
print(a)
# b = (a + 2**(-15))*0.5
# print(b)
# b_fi = Fxp(b, True, 16, 15)
# print(b_fi, )

# check2 (-0.0241241455078125-0.0052642822265625j)
# check3 (-0.02410888671875-0.0052490234375j)

#%%
from fxpmath import Fxp
import numpy as np
def simple_round_complex(c, n_word,n_frac):
    epsilon = 1e-15  # a tiny bias to activate the round
    def simple_round(number):
        if number > 0:
            return Fxp(number+epsilon, True, n_word, n_frac, rounding = 'around')
            # return int(number + 0.5 + epsilon)
        elif number < 0:
            return Fxp(number-epsilon, True, n_word, n_frac, rounding = 'around')
            # return int(number + 0.5 - epsilon)
        else:
            return 0
    
    real_part = simple_round(c.real)
    imag_part = simple_round(c.imag)
    
    return complex(real_part, imag_part)

print(simple_round_complex(-3.375,8,2))
print(simple_round_complex(3.375,8,2))

#%% !!! useful!!!
from fxpmath import Fxp
import numpy as np
import matplotlib.pyplot as plt

def simple_round_complex(value, n_word,n_frac):
    epsilon = 1e-15  # a tiny bias to activate the round
    def simple_round(number):
        if number > 0:
            return Fxp(number+epsilon, True, n_word, n_frac, rounding = 'around')
            
            # return int(number + 0.5 + epsilon)
        elif number < 0:
            return Fxp(number-epsilon, True, n_word, n_frac, rounding = 'around')
            # return int(number + 0.5 - epsilon)
        else:
            return 0
        
    if isinstance(value, complex):
        real_part = simple_round(value.real)
        imag_part = simple_round(value.imag)
        return complex(real_part, imag_part)
    
    elif isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            real_parts = np.real(value)
            imag_parts = np.imag(value)
            
            rounded_real_parts = np.vectorize(simple_round)(real_parts)
            rounded_imag_parts = np.vectorize(simple_round)(imag_parts)

            rounded_arr = rounded_real_parts + 1j * rounded_imag_parts
            return rounded_arr
        else:
            raise ValueError("Input array must contain complex numbers.")
    else:
        raise TypeError("Input must be a complex number or a numpy array of complex numbers.")
        
print(simple_round_complex(complex(0.125),8,2))

#%%
from fxpmath import Fxp
import numpy as np
import matplotlib.pyplot as plt

n_frac = 2
n_int = 0
n_word = n_int + n_frac
overflow = 'saturate'

fxp_ref = Fxp(None, signed=True, n_int=n_int, n_frac=n_frac, overflow=overflow)

fxp_ref.info(3)


def simple_round_complex(value, n_word, n_frac):#, n_word,n_frac):
    epsilon = 1e-15  # a tiny bias to activate the round
    def simple_round(number):
        if number > 0:
            #return Fxp(number+epsilon, rounding = 'around', like=fxp_ref)
            return Fxp(number+epsilon, True, n_word, n_frac, rounding = 'around')
            
            # return int(number + 0.5 + epsilon)
        elif number < 0:
            # return Fxp(number-epsilon, rounding = 'around', like=fxp_ref)
            return Fxp(number-epsilon, True, n_word, n_frac, rounding = 'around')
        else:
            return 0
        
    if isinstance(value, complex):
        real_part = simple_round(value.real)
        imag_part = simple_round(value.imag)
        return complex(real_part, imag_part)
    
    elif isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            real_parts = np.real(value)
            imag_parts = np.imag(value)
            
            rounded_real_parts = np.vectorize(simple_round)(real_parts)
            rounded_imag_parts = np.vectorize(simple_round)(imag_parts)

            rounded_arr = rounded_real_parts + 1j * rounded_imag_parts
            return rounded_arr
        else:
            raise ValueError("Input array must contain complex numbers.")
    else:
        raise TypeError("Input must be a complex number or a numpy array of complex numbers.")
        
# print(simple_round_complex(complex(-1.5),8,0))

#%%
ratio = 4
float_precision = fxp_ref.precision / ratio
x = np.arange(fxp_ref.lower - (ratio*float_precision), fxp_ref.upper + ratio*float_precision, step=float_precision)
x = x.astype(complex)
print(x)

indices_to_remove = [0, 1, 2, 3, -1, -2]

# 使用 np.delete() 函数删除指定索引的元素
x = np.delete(x, indices_to_remove)

print(x)

#%%

# Half away from zero
rounding_method= 'Half away from zero'
fxp_var = simple_round_complex(x,3,2)

# # Banker's rounding
# rounding = 'around'
# rounding_method= 'Half to even'
# fxp_var = fxp_var = Fxp(x, rounding=rounding, like=fxp_ref)

# Round half up
# rounding = 'around'
# rounding_method= 'Round half up '
# fxp_var = fxp_var = Fxp(x+1e-15, rounding=rounding, like=fxp_ref)

# truncate
# rounding = 'trunc'
# rounding_method = 'Truncate'
# fxp_var = fxp_var = Fxp(x, rounding=rounding, like=fxp_ref)



plt.figure(figsize=(8,8))
plt.plot(x, x, marker='x', label='float')
plt.plot(x, fxp_var, marker='*', label=f'{rounding_method}')
plt.grid()
plt.title(f'{rounding_method}')
plt.legend()
plt.show()

#%%
from fxpmath import Fxp
import numpy as np
def simple_round_complex(value, n_word, n_frac):#, n_word,n_frac):
    epsilon = 1e-15  # a tiny bias to activate the round
    def simple_round(number):
        if number > 0:
            #return Fxp(number+epsilon, rounding = 'around', like=fxp_ref)
            return Fxp(number-epsilon, True, n_word, n_frac, rounding = 'around')
            
            # return int(number + 0.5 + epsilon)
        elif number < 0:
            # return Fxp(number-epsilon, rounding = 'around', like=fxp_ref)
            return Fxp(number+epsilon, True, n_word, n_frac, rounding = 'around')
        else:
            return 0
        
    if isinstance(value, complex):
        real_part = simple_round(value.real)
        imag_part = simple_round(value.imag)
        return complex(real_part, imag_part)
    
    elif isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            real_parts = np.real(value)
            imag_parts = np.imag(value)
            
            rounded_real_parts = np.vectorize(simple_round)(real_parts)
            rounded_imag_parts = np.vectorize(simple_round)(imag_parts)

            rounded_arr = rounded_real_parts + 1j * rounded_imag_parts
            return rounded_arr
        else:
            raise ValueError("Input array must contain complex numbers.")
    else:
        raise TypeError("Input must be a complex number or a numpy array of complex numbers.")
b = -0.00000000023283064365386962890625
        
print(simple_round_complex(complex(b),32,31))

a = Fxp(b, True, 2, 1, rounding='around')
print(a)

c = Fxp(b+1e-15, True, 32, 31, rounding = 'around' )
print(c)

#%% rounding towards zero
from fxpmath import Fxp
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import FuncFormatter

# Function to format tick labels
def format_func(value, tick_number):
    return f'{value:.2f}'  # Adjust the number of decimal places here

n_frac = 2
n_int = 0
n_word = n_int + n_frac
overflow = 'saturate'

fxp_ref = Fxp(None, signed=True, n_int=n_int, n_frac=n_frac, overflow=overflow)

fxp_ref.info(3)


def simple_round_complex(value, n_word, n_frac):#, n_word,n_frac):
    epsilon = 1e-15  # a tiny bias to activate the round
    def simple_round(number):
        if number > 0:
            #return Fxp(number+epsilon, rounding = 'around', like=fxp_ref)
            return Fxp(number-epsilon, True, n_word, n_frac, rounding = 'around')
            
            # return int(number + 0.5 + epsilon)
        elif number < 0:
            # return Fxp(number-epsilon, rounding = 'around', like=fxp_ref)
            return Fxp(number+epsilon, True, n_word, n_frac, rounding = 'around')
        else:
            return 0
        
    if isinstance(value, complex):
        real_part = simple_round(value.real)
        imag_part = simple_round(value.imag)
        return complex(real_part, imag_part)
    
    elif isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            real_parts = np.real(value)
            imag_parts = np.imag(value)
            
            rounded_real_parts = np.vectorize(simple_round)(real_parts)
            rounded_imag_parts = np.vectorize(simple_round)(imag_parts)

            rounded_arr = rounded_real_parts + 1j * rounded_imag_parts
            return rounded_arr
        else:
            raise ValueError("Input array must contain complex numbers.")
    else:
        raise TypeError("Input must be a complex number or a numpy array of complex numbers.")
        
# print(simple_round_complex(complex(-1.5),8,0))

#%%
ratio = 4
float_precision = fxp_ref.precision / ratio
x = np.arange(fxp_ref.lower - (ratio*float_precision), fxp_ref.upper + ratio*float_precision, step=float_precision)
x = x.astype(complex)
print(x)

indices_to_remove = [0, 1, 2, 3, -1, -2]

# 使用 np.delete() 函数删除指定索引的元素
x = np.delete(x, indices_to_remove)

print(x)

#%%
from matplotlib.ticker import FuncFormatter

# Half towards zero
# rounding_method= 'Round Half Towards Zero'
# fxp_var = simple_round_complex(x,3,2)

# Half away from zero
# rounding_method = 'Round Half Away From Zero'
# fxp_var = simple_round_complex(x,3,2)

# Half Up
# rounding_method = 'Round Half Up'
# fxp_var = simple_round_complex(x,3,2)


# # # Banker's rounding
rounding = 'around'
rounding_method= 'Round Half To Even'
fxp_var = Fxp(x, rounding=rounding, like=fxp_ref)




plt.figure(figsize=(8,8))
plt.plot(x, x, marker='x', label='input signal (fractional bits: 4)')
plt.plot(x, fxp_var, marker='*', label='quantized signal (fractional bits: 2)')

# Apply formatting to the axes
plt.gca().xaxis.set_major_formatter(FuncFormatter(format_func))
plt.gca().yaxis.set_major_formatter(FuncFormatter(format_func))

# Add labels and title with fractional bits info
# plt.xlabel(f'Value (Fractional bits: {n_frac})')
plt.xlabel('Input Value', fontsize = 13)
plt.ylabel('Output Value', fontsize = 13)
# plt.title(f'{rounding_method} with {n_frac} Fractional Bits')

#改变横纵坐标刻度值的字体大小
plt.tick_params(axis='both', which='major', labelsize=13)  # 你可以调整 'labelsize' 的大小

plt.grid()
plt.title(f'{rounding_method}', fontsize = 15, fontweight ='bold')
plt.legend(fontsize = 13)
plt.savefig('A_Results/Figure/Half_To_Even.pdf', bbox_inches='tight')
plt.show()



