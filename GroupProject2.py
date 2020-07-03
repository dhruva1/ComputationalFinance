#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 10:38:44 2019

@author: dhruv
"""


#Answer 1:

import numpy as np
import numpy.matlib


#Initializing the variables from previous section
r = 0.08    # risk-free rate 
T = 1               # Time to maturity of option
K = 100        #	Strike price of option
S0 = 100             #Initial stock value 
sigma = 0.3             #	Stock volatility

k_log = np.log(K)


#Initializing the variable values given in this project
v0 = 0.06
kappa = 9
theta = 0.06
rho = -0.4

#Approximation information
t_max = 30
N = 100

#Characteristic function code

a = sigma**2/2

def b(u):
    return kappa - rho*sigma*1j*u

def c(u):
    return -(u**2+1j*u)/2

def d(u):
    return np.sqrt(b(u)**2-4*a*c(u))

def xminus(u):
    return (b(u)-d(u))/(2*a)

def xplus(u):
    return (b(u)+d(u))/(2*a)

def g(u):
    return xminus(u)/xplus(u)

def C(u):
    val1 = T*xminus(u) - np.log((1-g(u)*np.exp(-T*d(u)))/(1-g(u)))/a
    return r*T*1j*u +theta*kappa*val1

def D(u):
    val1 = 1 - np.exp(-T*d(u))
    val2 = 1 - g(u)*np.exp(-T*d(u))
    return (val1/val2)*xminus(u)

def log_char(u):
    return np.exp(C(u)+ D(u)*v0 + 1j*u*np.log(S0))

def adj_char(u):
    return log_char(u-1j)/log_char(-1j)


delta_t = t_max/N
from_1_to_N = np.linspace(1,N,N)
t_n = (from_1_to_N -1/2)*delta_t

first_integral = sum((((np.exp(-1j*t_n*k_log)*adj_char(t_n)).imag)/t_n)*delta_t)
second_integral = sum((((np.exp(-1j*t_n*k_log)*log_char(t_n)).imag)/t_n)*delta_t)

fourier_call_val = S0*(1/2+first_integral/np.pi) - np.exp(-r*T)*K*(1/2+second_integral/np.pi)

fourier_call_val
#Answer is 13.734

#Answer 2:

import matplotlib.pyplot as plt
from scipy.stats import ncx2
from scipy.stats import norm

def terminal_value(St, r, sigma, Z,T, gamma):
    
    sigmat = sigma*((St)**(gamma - 1))
    return St*np.exp((r  - sigmat**2/2)*T + sigmat*np.sqrt(T)*Z)
#Calculating next month's stock value using previous month's stock value, 
#risk free rate, sigma,simulated matrix of N(0,1) random variables, 
#time frame of 1 month (from the previos month), and gamma
    
def option_payoff(S_T,K):
    return np.maximum(S_T - K,0)
#The option_payoff function finds the payoff for a call option at terminal 
#time for given terminal stock price and strike price.

np.random.seed(0) #fixing the seed for result replication

moption_estimates = [None]*50 #monte carlo option estimates not incorporating default risk
moption_std = [None]*50    #standard deviation of monte carlo option estimates

gamma = 0.75

for i in range (1,51):
 
    size = i*1000 #sample sizes set ranging from 1000 to 50,000 with increments of 1000
    #in each loop this is the number of paths simulated
    
    stock_val = np.zeros(size)
    # The value of stock is initialized to zero for each simulation.
    
    prevstock_val = S0
    
    for month in range(1,12*T+1):     #simulation from month 1 to month 12*T 
                                        #where T is number of years (here T = 1)
        
            norm_matrix = norm.rvs(size = np.array([1,size]))
            
            stock_val = terminal_value(prevstock_val,r, sigma,
                                   norm_matrix,1/12.0, gamma)
    
            #We calculate the stock's value for this month using 
            #previos month's stock value, risk free rate, sigma, 
            #simulated matrix of N(0,1) random variables, time frame of 1 month
            # (from the previos month), and gamma
            
            prevstock_val = stock_val
            
    option_val = np.zeros(size)        
    for j in range(0,size):
        option_val = np.exp(-r*T)*option_payoff(stock_val,K)    
         #finds the discounted call payoff

    
    moption_estimates[i-1] = np.mean(option_val)
    moption_std[i-1] = np.std(option_val)/np.sqrt(size)


#Plotting the graph for Monte Carlo estimates
plt.plot(moption_estimates,'.')
plt.plot(np.mean(moption_estimates)+np.array(moption_std)*3, 'r')
plt.plot(np.mean(moption_estimates)-np.array(moption_std)*3, 'r')
plt.xlabel("Sample Size in Thousands")
plt.ylabel("Monte Carlo value of the option")


#Closed form solution to value of call option with CEV
z = 2 + 1/(1-gamma)

def CF(t,K):
    kappa = 2*r/(sigma**2*(1-gamma)*(np.exp(2*r*(1-gamma)*t)-1))
    x = kappa*S0**(2*(1-gamma))*np.exp(2*r*(1-gamma)*t)
    y = kappa*K**(2*(1-gamma))
    return S0*(1-ncx2.cdf(y,z,x)) - K*np.exp(-r*t)*ncx2.cdf(x,z-2,y)


CF(T,K)