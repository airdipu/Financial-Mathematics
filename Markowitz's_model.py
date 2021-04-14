#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 16:23:40 2021

@author: arahman
"""


# Mean-variance problem with d = 3, 𝜇 = (0.18, 0.15, 0.12), c = 0.3

import numpy as np
from scipy.optimize import *

# Optimal Asset Allocation for 3 assets 


sigma = np.array([[0.5, 0, -0.4], [0, 0.6, 0], [-0.4, 0, 0.4]]) # Covariance matrix
mu = np.array([0.3,0.15,0.12]) # Mean value of assets

def f(x):
    return -x.dot(mu.T)
    

ineq_c = {'type': 'ineq', 'fun': lambda x: -(x.transpose().dot(sigma)).dot(x) + 0.3}

eq_c = {'type': 'eq', 'fun': lambda x: -np.sum(x)+1 }

bnd = [[0, 1], [0, 1], [0, 1]]

x0 = np.array([0.3, 0.4, 0.3])
res = minimize(f,x0, method = 'SLSQP', constraints = [ineq_c,eq_c], bounds=bnd)
np.set_printoptions(suppress = True)
print(res)

print("Portfolio return for optimal allocation")
Pret = np.sum(res.x.dot(mu))
Pvar = (res.x.transpose().dot(sigma)).dot(res.x)

print("Pret = {:.4f}, Pvar = {:.4f}".format(Pret, Pvar))

    
print("Comparison of Portfolio variance")

Pstd = np.sqrt(Pvar)
Pind = np.sum(np.sqrt(np.diagonal(sigma)).dot(res.x))

print("Pstd = {:.4f}, Pind = {:.4f}".format(Pstd,Pind))
