#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 10:54:54 2025

@author: segu
"""


import numpy as np

def init_dcov(data):
    std = data[:, 2] 
    covmatrix = np.diag(std ** 2) 

    covmatrix_inv = np.linalg.inv(covmatrix)  # 역행렬 계산
    return covmatrix, covmatrix_inv

# def upd_dcov(residual, shrink=0.0, min_eig=1e-8):
    
             
    




# def dcov(data, covtype='full', numsample=10000):
#     # peri = data[:,0]
#     mean = data[:,1]
#     std = data[:,2]
    
#     if covtype == 'full': # full covariance 고려
#         _dat = np.array([np.random.normal(loc=_mean, scale=_std, size=numsample)
#                          for _mean, _std in zip(mean, std)]).T
#         # covmatrix = np.cov(_dat, rowvar=False)
#         ccmatrix = np.corrcoef(_dat, rowvar=False)
#         covmatrix = np.outer(std, std) * ccmatrix
        
#     elif covtype == 'diag': # diagonal 만 고려
#         std = data[:, 2] 
#         covmatrix = np.diag(std ** 2) 
    
#     covmatrix_inv = np.linalg.inv(covmatrix)  # 역행렬 계산

#     return covmatrix, covmatrix_inv
