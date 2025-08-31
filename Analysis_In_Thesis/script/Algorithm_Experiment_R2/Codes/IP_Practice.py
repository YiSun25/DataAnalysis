# -*- coding: utf-8 -*-
"""
Created on Wed May 22 17:42:12 2024

@author: yisss
"""

import numpy as np
import cv2
 
path = 'images/bear_example.jpg'
 
img = cv2.imread(path)
 
# the image height
sum_rows = img.shape[0]
# the image length
sum_cols = img.shape[1]
part1 = img[0:sum_rows, 0:sum_cols // 2]
part2 = img[0:sum_rows, sum_cols // 2:sum_cols]
 
cv2.imshow('part1', part1)
cv2.imshow('part2', part2)
 
cv2.waitKey()
cv2.imwrite('1_1.jpg', part1)
cv2.imwrite('1_2.jpg', part2)


#%%
# 
sum_rows = 1080
sum_cols = 1920
# new image
final_matrix = np.zeros((sum_rows, sum_cols, 3), np.uint8)
 
path1 = 'images/1_1.jpg'
path2 = 'images/1_2.jpg'
 
img1 = cv2.imread(path1)
img2 = cv2.imread(path2)
 
 
# change
final_matrix[0:sum_rows, 0:sum_cols // 2] = img1
final_matrix[0:sum_rows, sum_cols // 2:sum_cols] = img2
 
cv2.imshow('image', final_matrix)
cv2.waitKey()
