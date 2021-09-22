#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 07:55:54 2021

@author: arahman
"""

import numpy as np
import matplotlib.pyplot as plt
import random

# Simulation of a Stochastic integral with simple process

n = 2000

W0 = 0                          # initial value of the Brownian motion
X = np.random.uniform(-2,2,4)   # Generate uniformly distributed random heights of the step function 
tk = random.sample(range(0, n), 3)
ti = np.sort(tk)                # length of the interval for simple process

k1 = ti[0]
dW1 = np.random.normal(0,1,k1)*np.sqrt(1/k1)
I1 = X[0] * (np.cumsum(dW1))

k2 = ti[1]-ti[0]
dW2 = np.random.normal(0,1,k2)*np.sqrt(1/k2)
I2 = X[1] * (np.cumsum(dW2))

S1 = np.append(I1,I2)

k3 = ti[2]-ti[1]
dW3 = np.random.normal(0,1,k3)*np.sqrt(1/k3)
I3 = X[2] * (np.cumsum(dW3))

S2 = np.append(S1, I3)

k3 = n-ti[2]
dW4 = np.random.normal(0,1,k3)*np.sqrt(1/k3)
I4 = X[3] * (np.cumsum(dW4))

S3 = np.append(S2, I4)

I = np.append(W0,S3)

# Plot the path of the stochastic integral

t = np.linspace(0, 1, num=n+1)

plt.plot(t,I, 'r')
plt.show()