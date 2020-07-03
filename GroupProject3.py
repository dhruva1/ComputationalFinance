#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 13:42:34 2019

@author: dhruv
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
import random

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



#Initializing the variables 
T = 1               # Time to maturity of option
strike = 100        #	Strike price of option
S0 = 100             #Initial stock value 
BarrierLevel=150        #up and out barrier level for the option is 150
debt = 175              #Counterparty Firm debt
recovery_rate = 0.25    #	Recovery rate from counterparty in event of default
correlation =  0.2          #Counterparty firm and stock correlation
V0 = 200 # the firm value of the counterparty in the beginning is 200
sigma = 0.3             #	Stock and firm volatility
gamma = 0.75

delta = 1/12   #it is the time step difference (1 month or 1/12 years) between 
              #two bond prices of closest maturity dates

bond_prices = [99.38,98.76,98.15, 97.54, 96.94, 96.34, 95.74, 95.16, 94.57, 93.99, 93.42, 92.85]
#zero coupon bond prices with maturities 1 month, 2 months, 3 months, .......12 months

M1r = np.log(100/bond_prices[0])/delta #risk free rate for month 1 

r = M1r    # dummy initial value for risk free rate

np.random.seed(0)       #fixing the seed for result replication

moption_estimates = [None] #monte carlo option estimates not incorporating default risk
moption_std = [None]    #standard deviation of monte carlo option estimates
cva_estimates = [None] #cva estimates 
cva_std = [None]        # standard deviation of cva estimates
CVAadjustedOptionval = [None] #Option value incorporating default risk
CVAadjustedOptionval_std = [None] #standard deviation of Option value incorporating default risk

np.random.seed(0) #fixing the seed for result replication

size = 100000 #sample size

prevstock_val = S0
prevfirm_val = V0

n_steps = 12 #number of steps - different bond prices that are there (12 in this case)
n_simulations = size # number of simulations is equal to sample size in this case

# The value of stock, firm, maxST ( maximum stock value for each simulation path) 
# and option value without default risk of counterparty is initialized to zero

maxST = np.zeros(size)    
stock_val = np.zeros(size)
firm_val = np.zeros(size)                
option_val = np.zeros(size)



#Initializing variables for the  predictor corrector method

mc_forward = np.ones([n_simulations,n_steps-1])* np.divide(np.subtract(bond_prices[:-1],bond_prices[1:]), np.multiply(delta,bond_prices[1:]))
predcorr_forward = np.ones([n_simulations,n_steps-1])* np.divide(np.subtract(bond_prices[:-1],bond_prices[1:]), np.multiply(delta,bond_prices[1:]))
predcorr_capfac = np.ones([n_simulations,n_steps])
mc_capfac = np.ones([n_simulations,n_steps])

Delta = np.ones([n_simulations,n_steps - 1])*(1/12) #Vectorizing Delta
sigmaj =0.2 #Using value of sigmaj = 0.2 from notes


#For each time step, we implement the predictor-corrector method. We set our 
#initial forward rates to the previous times’ forward rates, and create temporary 
#simulated forward rates using our first drift estimate. We then use these new 
#rates to create a second drift estimate. Finally, we use the average of these 
#two drift estimates to simulate the next time step forward rates from our 
#previous forward rates


for i in range(1,n_steps):
    Z = norm.rvs(size = [n_simulations,1])
    
    # Explicit Monte Carlo simulation
    muhat = np.cumsum(Delta[:,i:]*mc_forward[:,i:]*sigmaj**2/(1+Delta[:,i:]*mc_forward[:,i:]),axis = 1)
    mc_forward[:,i:] = mc_forward[:,i:]*np.exp((muhat-sigmaj**2/2)*Delta[:,i:]+sigmaj*np.sqrt(Delta[:,i:])*Z)
    
    # Predictor-Corrector Montecarlo simulation
    mu_initial = np.cumsum(Delta[:,i:]*predcorr_forward[:,i:]*sigmaj**2/(1+Delta[:,i:]*predcorr_forward[:,i:]),axis = 1)
    for_temp = predcorr_forward[:,i:]*np.exp((mu_initial-sigmaj**2/2)*Delta[:,i:]+sigmaj*np.sqrt(Delta[:,i:])*Z)
    mu_term = np.cumsum(Delta[:,i:]*for_temp*sigmaj**2/(1+Delta[:,i:]*for_temp),axis = 1)
    predcorr_forward[:,i:] = predcorr_forward[:,i:]*np.exp((mu_initial+mu_term-sigmaj**2)*Delta[:,i:]/2+sigmaj*np.sqrt(Delta[:,i:])*Z)


  

  
for month in range(1,12*T+1):     #simulation from month 1 to month 12*T 
                                        #where T is number of years (here T = 1)
    
    if month == 1 :
        r = M1r
    else :
        r = np.log(1+(predcorr_forward[:, (month-2)])*delta)/delta
        
    corr_matrix = np.array([[1,correlation],[correlation,1]])
    norm_matrix = norm.rvs(size = np.array([2,size]))
    corr_norm_matrix = np.matmul(np.linalg.cholesky(corr_matrix), 
                                         norm_matrix)
            
    #The statement above creates a 2 × size array of standard 
    #normal random variables with a given correlation    
        
            
    stock_val = terminal_value(prevstock_val,r, sigma,corr_norm_matrix[0,],1/12.0, gamma)
    
    #We calculate the stock's value for this month using previous month's stock 
    #value, risk free rate, sigma, simulated matrix of N(0,1) random
    # variables, time frame of 1 month (from the previous month), and gamma
                               
    firm_val = terminal_value(prevfirm_val,r, sigma, corr_norm_matrix[1,],1/12.0, gamma)
                                   
    #We calculate the counterparty firm's value for this month using a similar 
    #mechanism that we employed for calculating the stock's value for this month
 
    prevstock_val = stock_val         
    prevfirm_val = firm_val
    
    #Values of stock and firm for the prior month are initialized
    
    for j in range(0,size):
            if(stock_val[j] > maxST[j]):
                    maxST[j] = stock_val[j]
         #updating maxST to the max value of the respective stock price path 
         #(in case this month’s value is higher than the maxST on record)

DF = bond_prices[12*T-1]/100 #one year discount factor

M0M12r =  np.log(100/bond_prices[12*T-1])/T # calculating risk free rate continuously compounded from month 0 to month 12
 
r = M0M12r #updating r for calculation of option value and CVA
         
    
for j in range(0,size):
    
    if(maxST[j] > BarrierLevel):
        option_val[j] = 0 
         #if stock price has exceeded the barrier level then option value is 0    
    else:
        option_val[j] = np.exp(-r*T)*option_payoff(stock_val[j],strike)    
        #finds the discounted call payoff
    


amount_lost = np.exp(-r*T)*(1-recovery_rate)*(firm_val < debt)*option_val

#estimates for CVA based on firm value 
cva_estimates = np.mean(amount_lost)
# We calculate the CVA estimate to be the mean of the amount_lost across 
# simulations and also calculate its standard deviation. 
cva_std = np.std(amount_lost)/np.sqrt(size)
    
#default-free value of the option
moption_estimates = np.mean(option_val)
moption_std = np.std(option_val)/np.sqrt(size)
    
#Monte Carlo estimates for the price of the option incorporating counterparty risk, given by the default-free price less the CVA
#We calculate CVA Adjusted option value to be the default free value of the option minus the CVA estimate
CVAadjustedOptionval = moption_estimates - cva_estimates
    
CVAadjustedOptionval_std = np.std(option_val - amount_lost)/np.sqrt(size)
                


print("The CVA based on firm value is ",cva_estimates)
print("The option value with no default risk is ",moption_estimates)
print("The option value with counterparty default risk is ",CVAadjustedOptionval)

