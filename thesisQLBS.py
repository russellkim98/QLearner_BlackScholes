
import numpy as np
import pandas as pd
from scipy.stats import norm
import random
import copy
import sys
import os
os.chdir("/Users/therealrussellkim/QLearner_BlackScholes")


import time
import matplotlib.pyplot as plt
import bspline
import bspline.splinelab as splinelab
from pathlib import Path

###############################################################################
###############################################################################
# CONTROL ENVIRONMENT




# The Black-Scholes prices
def bs_put(S0,sigma,r,T,t,K):
    d1 = (np.log(S0/K) + (r + 1/2 * sigma**2) * (T-t)) / sigma / np.sqrt(T-t)
    d2 = (np.log(S0/K) + (r - 1/2 * sigma**2) * (T-t)) / sigma / np.sqrt(T-t)
    price = K * np.exp(-r * (T-t)) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    return price

def bs_call(S0,mu,sigma,r,T,t,K):
    d1 = (np.log(S0/K) + (r + 1/2 * sigma**2) * (T-t)) / sigma / np.sqrt(T-t)
    d2 = (np.log(S0/K) + (r - 1/2 * sigma**2) * (T-t)) / sigma / np.sqrt(T-t)
    price = S0 * norm.cdf(d1) - K * np.exp(-r * (T-t)) * norm.cdf(d2)
    return price
    




###############################################################################
###############################################################################
# TESTING ENVIRONEMENT
    

init = [100,0.05,0.15,0.03,1,10,0.001,10000,100, 42]
# Import main helper function
from QLBSHelper import QLBS_EPUT

def testingPhase1(initParams):
   S0 = initParams[0]    # initial stock price
   mu = initParams[1]    # drift
   sigma = initParams[2] # volatility
   r = initParams[3]     # risk-free rate
   M = initParams[4]     # maturity
   T = initParams[5]     # number of time steps
   risk_lambda = initParams[6]   # risk aversion
   N_MC = initParams[7] # number of paths
   K = initParams[8] # Strike price
   delta_t = M / T                # time interval
   gamma = np.exp(- r * delta_t)  # discount factor
   
   
   rand_seed = initParams[9]
   
   
   q_price = QLBS_EPUT(S0,mu,sigma,r,M,T,risk_lambda,N_MC,delta_t,gamma,K, rand_seed)
   QLBS_price = -np.mean(q_price[0]) #average over all simulations
   
   EBS_put_price = bs_put(S0,sigma,r,M,0,K)
   compTime = q_price[1]
   
   print('---------------------------------')
   print('       QLBS RL Option Pricing       ')
   print('---------------------------------\n')
   print('Type:' + 'European put')
   print('%-25s' % ('Initial Stock Price:'), S0)
   print('%-25s' % ('Drift of Stock:'), mu)
   print('%-25s' % ('Volatility of Stock:'), sigma)
   print('%-25s' % ('Risk-free Rate:'), r)
   print('%-25s' % ('Risk aversion parameter :'), risk_lambda)
   print('%-25s' % ('Strike:'), K)
   print('%-25s' % ('Maturity:'), M)
   print('%-26s %.4f' % ('\nThe QLBS Price:', QLBS_price))
   print('%-26s %.4f' % ('\nBlack-Scholes Price:', EBS_put_price))
   print('%-25s' % ('Random Seed:'), rand_seed)
   print('Computational time for Q-Learning:', compTime, 'seconds')
   print('\n')
   
   ###############################################################################
   ###############################################################################
   

 
   
   return(QLBS_price)
 

test = testingPhase1(init)






'''

# add here calculation of different MC runs (6 repetitions of action randomization)

# on-policy values
y1_onp = 5.0211 # 4.9170
y2_onp = 4.7798 # 7.6500

# QLBS_price_on_policy = 4.9004 +/- 0.1206

# these are the results for noise eta = 0.15
# p1 = np.array([5.0174, 4.9249, 4.9191, 4.9039, 4.9705, 4.6216 ])
# p2 = np.array([6.3254, 8.6733, 8.0686, 7.5355, 7.1751, 7.1959 ])

p1 = np.array([5.0485, 5.0382, 5.0211, 5.0532, 5.0184])
p2 = np.array([4.7778, 4.7853, 4.7781,4.7805, 4.7828])

# results for eta = 0.25
# p3 = np.array([4.9339, 4.9243, 4.9224, 5.1643, 5.0449, 4.9176 ])
# p4 = np.array([7.7696,8.1922, 7.5440,7.2285, 5.6306, 12.6072])

p3 = np.array([5.0147, 5.0445, 5.1047, 5.0644, 5.0524])
p4 = np.array([4.7842,4.7873, 4.7847, 4.7792, 4.7796])

# eta = 0.35 
# p7 = np.array([4.9718, 4.9528, 5.0170, 4.7138, 4.9212, 4.6058])
# p8 = np.array([8.2860, 7.4012, 7.2492, 8.9926, 6.2443, 6.7755])

p7 = np.array([5.1342, 5.2288, 5.0905, 5.0784, 5.0013 ])
p8 = np.array([4.7762, 4.7813,4.7789, 4.7811, 4.7801])

# results for eta = 0.5
# p5 = np.array([4.9446, 4.9894,6.7388, 4.7938,6.1590, 4.5935 ])
# p6 = np.array([7.5632, 7.9250, 6.3491, 7.3830, 13.7668, 14.6367 ])

p5 = np.array([3.1459, 4.9673, 4.9348, 5.2998, 5.0636 ])
p6 = np.array([4.7816, 4.7814, 4.7834, 4.7735, 4.7768])

# print(np.mean(p1), np.mean(p3), np.mean(p5))
# print(np.mean(p2), np.mean(p4), np.mean(p6))
# print(np.std(p1), np.std(p3), np.std(p5))
# print(np.std(p2), np.std(p4), np.std(p6))

x = np.array([0.15, 0.25, 0.35, 0.5])
y1 = np.array([np.mean(p1), np.mean(p3), np.mean(p7), np.mean(p5)])
y2 = np.array([np.mean(p2), np.mean(p4), np.mean(p8), np.mean(p6)])
y_err_1 = np.array([np.std(p1), np.std(p3),np.std(p7),  np.std(p5)])
y_err_2 = np.array([np.std(p2), np.std(p4), np.std(p8), np.std(p6)])

# plot it 
f, axs = plt.subplots(nrows=2, ncols=2, sharex=True)

f.subplots_adjust(hspace=.5)
f.set_figheight(6.0)
f.set_figwidth(8.0)

ax = axs[0,0]
ax.plot(x, y1)
ax.axhline(y=y1_onp,linewidth=2, color='r')
textstr = 'On-policy value = %2.2f'% (y1_onp)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)                      
# place a text box in upper left in axes coords
ax.text(0.05, 0.15, textstr, fontsize=11,transform=ax.transAxes, verticalalignment='top', bbox=props)
ax.set_title('Mean option price')
ax.set_xlabel('Noise level')

ax = axs[0,1]
ax.plot(x, y2)
ax.axhline(y=y2_onp,linewidth=2, color='r')
textstr = 'On-policy value = %2.2f'% (y2_onp)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)                      
# place a text box in upper left in axes coords
ax.text(0.35, 0.95, textstr, fontsize=11,transform=ax.transAxes, verticalalignment='top', bbox=props)
ax.set_title('Mean option price')
ax.set_xlabel('Noise level')

ax = axs[1,0]
ax.plot(x, y_err_1)
ax.set_title('Std of option price')
ax.set_xlabel('Noise level')

ax = axs[1,1]
ax.plot(x, y_err_2)
ax.set_title('Std of option price')
ax.set_xlabel('Noise level')

f.suptitle('Mean and std of option price vs noise level')

plt.savefig('Option_price_vs_noise_level.png', dpi=600)
plt.show()

#[]:
'''



