#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 20:54:58 2021

@author: arahman
"""

import numpy as np
from scipy.stats import norm

# Black-Scholes formula for European calls and puts

def BSformula(S0_, r_, sigma_, K_, T_, opt):  
    S0 = np.float32(S0_)  # Initial stock price
    K = np.float32(K_)  # Strike price
    T = np.float32(T_)  # Option maturity in years
    r = np.float32(r_)  # Interest rate
    sigma = np.float32(sigma_) # Stock volatility    
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) /(sigma * np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    Call = S0 * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)   
    Put = K * np.exp(-r*T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)    
    return Call if opt==1 else Put

    
print('\n' "Black-Scholes formula for option pricing" '\n')
BScall = BSformula(200, 0.05, 0.1, 200, 1, 1)
print('European call price: {:.4f}'.format(BScall), '\n')
BSput = BSformula(200, 0.05, 0.1, 200, 1, 0)
print('European put price: {:.4f}'.format(BSput))