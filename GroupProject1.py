import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
import random

def terminal_value(S_0, risk_free_rate, sigma, Z,T):
    return S_0*np.exp((risk_free_rate  - sigma**2/2)*T + sigma*np.sqrt(T)*Z)

#We are going to be using the terminal_value function to transform our initial 
#stock and counterparty firm value into terminal values for a given risk free 
#rate, volatility (sigma), random path array (Z), and time to maturity. 

def option_payoff(S_T,K):
    return np.maximum(S_T - K,0)
#The option_payoff function finds the payoff for a call option at terminal 
#time for given terminal stock price and strike price.


#Initializing the variables 
risk_free = 0.08    # risk-free rate 
T = 1               # Time to maturity of option
strike = 100        #	Strike price of option
S_0 = 100             #Initial stock value 
BarrierLevel=150        #up and out barrier level for the option is 150
sigma = 0.3             #	Stock volatility
sigma_firm = 0.25       #Counterparty Firm volatility
debt = 175              #Counterparty Firm debt
recovery_rate = 0.25    #	Recovery rate from counterparty in event of default
correlation =  0.2          #Counterparty firm and stock correlation
V_0 = 275 #assuming the value of the counterparty in the beginning is 275

np.random.seed(0)       #fixing the seed for result replication

moption_estimates = [None]*50 #monte carlo option estimates not incorporating default risk
moption_std = [None]*50    #standard deviation of monte carlo option estimates
cva_estimates = [None]*50 #cva estimates 
cva_std = [None]*50        # standard deviation of cva estimates
CVAadjustedOptionval = [None]*50 #Option value incorporating default risk
CVAadjustedOptionval_std = [None]*50 #standard deviation of Option value incorporating default risk

for i in range (1,51):
    
    size = i*1000 #sample sizes set ranging from 1000 to 50,000 with increments of 1000
    #in each loop this is the number of paths simulated
    
    maxST = np.zeros(size)  #max stock price for each simulation    
    stock_val = np.zeros(size)
    firm_val = np.zeros(size)                
    option_val = np.zeros(size)
    
    # The value of stock, firm, maxST ( maximum stock value for each simulation path) 
    # and option value without default risk of counterparty is initialized to zero for each simulation.
    
    for month in range(1,12*T+1):     #simulation from month 1 to month 12*T 
                                        #where T is number of years (here T = 1)
        
        
        corr_matrix = np.array([[1,correlation],[correlation,1]])
        norm_matrix = norm.rvs(size = np.array([2,size]))
        corr_norm_matrix = np.matmul(np.linalg.cholesky(corr_matrix), 
                                         norm_matrix)
           
        #The statement above creates a 2 ×50 000 array of standard 
        #normal random variables with a given correlation    
        
        stock_val = terminal_value(S_0,risk_free, sigma,
                                   corr_norm_matrix[0,],month/12.0)
        
        
        firm_val = terminal_value(V_0,risk_free, sigma_firm, 
                                   corr_norm_matrix[1,],month/12.0)
        
        #We calculate the stock and firm’s final value using the terminal value function 
        #initialized in the beginning by passing in the appropriate parameters.
        
        for j in range(0,size):
            if(stock_val[j] > maxST[j]):
                    maxST[j] = stock_val[j]
         #updating maxST to the max value of the respective stock price path 
         #(in case this month’s value is higher than the maxST on record)
    
    
    for j in range(0,size):
        if(maxST[j] > BarrierLevel):
            option_val[j] = 0 
            #if stock price has exceeded the barrier level then option value is 0    
        else:
            option_val[j] = np.exp(-risk_free*T)*option_payoff(stock_val[j],strike)    
             #finds the discounted call payoff
    
         
    amount_lost = np.exp(-risk_free*T)*(1-recovery_rate)*(firm_val < debt)*option_val
    #estimates for CVA based on firm value 
    cva_estimates[i-1] = np.mean(amount_lost)
    # We calculate the CVA estimate to be the mean of the amount_lost across 
    # simulations and also calculate its standard deviation. 
    cva_std[i-1] = np.std(amount_lost)/np.sqrt(size)
    
    #default-free value of the option
    moption_estimates[i-1] = np.mean(option_val)
    moption_std[i-1] = np.std(option_val)/np.sqrt(size)
    
    #Monte Carlo estimates for the price of the option incorporating counterparty risk, given by the default-free price less the CVA
    #We calculate CVA Adjusted option value to be the default free value of the option minus the CVA estimate
    CVAadjustedOptionval[i-1] = moption_estimates[i-1] - cva_estimates[i-1]
    
    CVAadjustedOptionval_std[i-1] = np.std(option_val - amount_lost)/np.sqrt(size)
    

#Plotting the graph for default-free value of the option
plt.plot(moption_estimates,'.')
plt.plot(np.mean(moption_estimates)+np.array(moption_std)*3, 'r')
plt.plot(np.mean(moption_estimates)-np.array(moption_std)*3, 'r')
plt.xlabel("Sample Size in Thousands")
plt.ylabel("default-free value of the option")


#Plotting the graph for CVA
plt.plot(cva_estimates,'.')
plt.plot(np.mean(cva_estimates)+np.array(cva_std)*3, 'r')
plt.plot(np.mean(cva_estimates)-np.array(cva_std)*3, 'r')
plt.xlabel("Sample Size in Thousands")
plt.ylabel("Value of CVA")

#Plotting the graph for CVA adjusted option value
plt.plot(CVAadjustedOptionval,'.')
plt.plot(np.mean(CVAadjustedOptionval)+np.array(CVAadjustedOptionval_std)*3, 'r')
plt.plot(np.mean(CVAadjustedOptionval)-np.array(CVAadjustedOptionval_std)*3, 'r')
plt.xlabel("Sample Size in Thousands")
plt.ylabel("CVA Adjusted Option Value")

