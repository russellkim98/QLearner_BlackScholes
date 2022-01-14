#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 18:43:25 2019

@author: therealrussellkim
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import random
import copy
import sys

sys.path.append("..")

import time
import matplotlib.pyplot as plt
import bspline
import bspline.splinelab as splinelab


def QLBS_EPUT(S0, mu, sigma, r, M, T, risk_lambda, N_MC, delta_t, gamma, K, rand_seed):

    ###############################################################################
    ###############################################################################

    # make a dataset

    np.random.seed(rand_seed)  # Fix random seed
    # stock price
    S = pd.DataFrame([], index=range(1, N_MC + 1), columns=range(T + 1))
    S.loc[:, 0] = S0

    # standard normal random numbers
    RN = pd.DataFrame(
        np.random.randn(N_MC, T), index=range(1, N_MC + 1), columns=range(1, T + 1)
    )

    for t in range(1, T + 1):
        S.loc[:, t] = S.loc[:, t - 1] * np.exp(
            (mu - 1 / 2 * sigma ** 2) * delta_t
            + sigma * np.sqrt(delta_t) * RN.loc[:, t]
        )

    delta_S = S.loc[:, 1:T].values - np.exp(r * delta_t) * S.loc[:, 0 : T - 1]
    delta_S_hat = delta_S.apply(lambda x: x - np.mean(x), axis=0)

    # state variable
    X = -(mu - 1 / 2 * sigma ** 2) * np.arange(T + 1) * delta_t + np.log(
        S
    )  # delta_t here is due to their conventions

    # plot 10 paths
    step_size = N_MC // 10
    idx_plot = np.arange(step_size, N_MC, step_size)
    plt.plot(S.T.iloc[:, idx_plot])
    plt.xlabel("Time Steps")
    plt.title("Stock Price Sample Paths")
    plt.ylabel("State Variable")
    plt.show()

    plt.plot(X.T.iloc[:, idx_plot])
    plt.xlabel("Time Steps")
    plt.ylabel("State Variable")
    plt.title("State Variable Sample Paths")
    plt.show()

    ###############################################################################
    ###############################################################################
    # Define function *terminal_payoff* to compute the terminal payoff of a European put option.

    def terminal_payoff(ST, K):
        # ST   final stock price
        # K    strike
        payoff = max(K - ST, 0)
        return payoff

    ###############################################################################
    ###############################################################################

    # Define spline basis functions

    import bspline
    import bspline.splinelab as splinelab

    X_min = np.min(np.min(X))
    X_max = np.max(np.max(X))

    p = 4  # order of spline (as-is; 3 = cubic, 4: B-spline?)
    ncolloc = 12

    tau = np.linspace(
        X_min, X_max, ncolloc
    )  # These are the sites to which we would like to interpolate

    # k is a knot vector that adds endpoints repeats as appropriate for a spline of order p
    # To get meaninful results, one should have ncolloc >= p+1
    k = splinelab.aptknt(tau, p)

    # Spline basis of order p on knots k
    basis = bspline.Bspline(k, p)
    f = plt.figure()

    # Spline basis functions
    plt.title("Basis Functions to be Used For This Iteration")
    basis.plot()

    plt.savefig("Basis_functions.png", dpi=600)

    ###############################################################################
    ###############################################################################

    # ### Make data matrices with feature values
    #
    # "Features" here are the values of basis functions at data points
    # The outputs are 3D arrays of dimensions num_tSteps x num_MC x num_basis

    num_t_steps = T + 1
    num_basis = ncolloc  # len(k) #

    data_mat_t = np.zeros((num_t_steps, N_MC, num_basis))

    # fill it, expand function in finite dimensional space
    # in neural network the basis is the neural network itself
    t_0 = time.time()
    for i in np.arange(num_t_steps):
        x = X.values[:, i]
        data_mat_t[i, :, :] = np.array([basis(el) for el in x])

    t_end = time.time()

    # save these data matrices for future re-use
    np.save("data_mat_m=r_A_%d" % N_MC, data_mat_t)

    ###############################################################################
    ###############################################################################

    # ## Dynamic Programming solution for QLBS

    risk_lambda = 0.001  # 0.001 # 0.0001            # risk aversion
    K = 100  #

    # functions to compute optimal hedges
    def function_A_vec(t, delta_S_hat, data_mat, reg_param):
        """
        function_A_vec - compute the matrix A_{nm} from Eq. (52) (with a regularization!)
        Eq. (52) in QLBS Q-Learner in the Black-Scholes-Merton article

        Arguments:
        t - time index, a scalar, an index into time axis of data_mat
        delta_S_hat - pandas.DataFrame of dimension N_MC x T
        data_mat - pandas.DataFrame of dimension T x N_MC x num_basis
        reg_param - a scalar, regularization parameter

        Return:
        - np.array, i.e. matrix A_{nm} of dimension num_basis x num_basis
        """
        ### START CODE HERE ### (≈ 5-6 lines of code)
        # A_mat = your code goes here ...
        X_mat = data_mat[t, :, :]
        num_basis_funcs = X_mat.shape[1]
        this_dS = delta_S_hat.loc[:, t]
        hat_dS2 = (this_dS ** 2).values.reshape(-1, 1)
        A_mat = np.dot(X_mat.T, X_mat * hat_dS2) + reg_param * np.eye(num_basis_funcs)
        ### END CODE HERE ###
        return A_mat

    def function_B_vec(
        t,
        Pi_hat,
        delta_S_hat=delta_S_hat,
        S=S,
        data_mat=data_mat_t,
        gamma=gamma,
        risk_lambda=risk_lambda,
    ):
        """
        function_B_vec - compute vector B_{n} from Eq. (52) QLBS Q-Learner in the Black-Scholes-Merton article

        Arguments:
        t - time index, a scalar, an index into time axis of delta_S_hat
        Pi_hat - pandas.DataFrame of dimension N_MC x T of portfolio values
        delta_S_hat - pandas.DataFrame of dimension N_MC x T
        S - pandas.DataFrame of simulated stock prices
        data_mat - pandas.DataFrame of dimension T x N_MC x num_basis
        gamma - one time-step discount factor $exp(-r \delta t)$
        risk_lambda - risk aversion coefficient, a small positive number

        Return:
        B_vec - np.array() of dimension num_basis x 1
        """
        # coef = 1.0/(2 * gamma * risk_lambda)
        # override it by zero to have pure risk hedge
        coef = 0.0  # keep it

        tmp = Pi_hat.loc[:, t + 1] * delta_S_hat.loc[:, t]
        X_mat = data_mat[t, :, :]  # matrix of dimension N_MC x num_basis
        B_vec = np.dot(X_mat.T, tmp)

        return B_vec

    ###############################################################################
    ###############################################################################

    # ## Compute optimal hedge and portfolio value

    starttime = time.time()

    # portfolio value
    Pi = pd.DataFrame([], index=range(1, N_MC + 1), columns=range(T + 1))
    Pi.iloc[:, -1] = S.iloc[:, -1].apply(lambda x: terminal_payoff(x, K))

    Pi_hat = pd.DataFrame([], index=range(1, N_MC + 1), columns=range(T + 1))
    Pi_hat.iloc[:, -1] = Pi.iloc[:, -1] - np.mean(Pi.iloc[:, -1])

    # optimal hedge
    a = pd.DataFrame([], index=range(1, N_MC + 1), columns=range(T + 1))
    a.iloc[:, -1] = 0

    reg_param = 1e-3
    for t in range(T - 1, -1, -1):

        A_mat = function_A_vec(t, delta_S_hat, data_mat_t, reg_param)
        B_vec = function_B_vec(t, Pi_hat, delta_S_hat, S, data_mat_t)

        # print ('t =  A_mat.shape = B_vec.shape = ', t, A_mat.shape, B_vec.shape)
        phi = np.dot(np.linalg.inv(A_mat), B_vec)

        a.loc[:, t] = np.dot(data_mat_t[t, :, :], phi)
        Pi.loc[:, t] = gamma * (Pi.loc[:, t + 1] - a.loc[:, t] * delta_S.loc[:, t])
        Pi_hat.loc[:, t] = Pi.loc[:, t] - np.mean(Pi.loc[:, t])

    a = a.astype("float")
    Pi = Pi.astype("float")
    Pi_hat = Pi_hat.astype("float")
    endtime = time.time()

    # Plots of 10 optimal hedge $a_t^\star$ and portfolio value $\Pi_t$ paths are shown below.

    # plot 10 paths
    plt.plot(a.T.iloc[:, idx_plot])
    plt.xlabel("Time Steps")
    plt.title("Optimal Hedge")
    plt.show()

    plt.plot(Pi.T.iloc[:, idx_plot])
    plt.xlabel("Time Steps")
    plt.title("Portfolio Value")
    plt.show()

    ###############################################################################
    ###############################################################################

    # ## Part 2: Compute the optimal Q-function with the DP approach

    def function_C_vec(t, data_mat, reg_param):
        """
        function_C_vec - calculate C_{nm} matrix  (with a regularization!)

        Arguments:
        t - time index, a scalar, an index into time axis of data_mat
        data_mat - pandas.DataFrame of values of basis functions of dimension T x N_MC x num_basis
        reg_param - regularization parameter, a scalar

        Return:
        C_mat - np.array of dimension num_basis x num_basis
        """
        ### START CODE HERE ### (≈ 5-6 lines of code)
        # C_mat = your code goes here ....
        X_mat = data_mat[t, :, :]
        num_basis_funcs = X_mat.shape[1]
        C_mat = np.dot(X_mat.T, X_mat) + reg_param * np.eye(num_basis_funcs)
        ### END CODE HERE ###

        return C_mat

    def function_D_vec(t, Q, R, data_mat, gamma=gamma):
        """
        function_D_vec - calculate D_{nm} vector (with a regularization!)

        Arguments:
        t - time index, a scalar, an index into time axis of data_mat
        Q - pandas.DataFrame of Q-function values of dimension N_MC x T
        R - pandas.DataFrame of rewards of dimension N_MC x T
        data_mat - pandas.DataFrame of values of basis functions of dimension T x N_MC x num_basis
        gamma - one time-step discount factor $exp(-r \delta t)$

        Return:
        D_vec - np.array of dimension num_basis x 1
        """
        ### START CODE HERE ### (≈ 2-3 lines of code)
        # D_vec = your code goes here ...
        X_mat = data_mat[t, :, :]
        D_vec = np.dot(X_mat.T, R.loc[:, t] + gamma * Q.loc[:, t + 1])
        ### END CODE HERE ###

        return D_vec

    ###############################################################################
    ###############################################################################

    # Implement a batch-mode off-policy model-free Q-Learning by Fitted Q-Iteration.
    # The only data available is given by a set of $N_{MC}$ paths for the underlying state
    # variable $X_t$, hedge position $a_t$, instantaneous reward $R_t$ and the
    # next-time value $X_{t+1}$.

    starttime = time.time()

    eta = 0.5  #  0.5 # 0.25 # 0.05 # 0.5 # 0.1 # 0.25 # 0.15
    reg_param = 1e-3
    np.random.seed(42)  # Fix random seed

    # disturbed optimal actions to be computed
    a_op = pd.DataFrame([], index=range(1, N_MC + 1), columns=range(T + 1))
    a_op.iloc[:, -1] = 0

    # also make portfolios and rewards
    # portfolio value
    Pi_op = pd.DataFrame([], index=range(1, N_MC + 1), columns=range(T + 1))
    Pi_op.iloc[:, -1] = S.iloc[:, -1].apply(lambda x: terminal_payoff(x, K))

    Pi_op_hat = pd.DataFrame([], index=range(1, N_MC + 1), columns=range(T + 1))
    Pi_op_hat.iloc[:, -1] = Pi_op.iloc[:, -1] - np.mean(Pi_op.iloc[:, -1])

    # reward function
    R_op = pd.DataFrame([], index=range(1, N_MC + 1), columns=range(T + 1))
    R_op.iloc[:, -1] = -risk_lambda * np.var(Pi_op.iloc[:, -1])

    # The backward loop
    for t in range(T - 1, -1, -1):

        # 1. Compute the optimal policy, and write the result to a_op
        a_op.loc[:, t] = a.loc[:, t]
        # 2. Now disturb these values by a random noise
        a_op.loc[:, t] *= np.random.uniform(1 - eta, 1 + eta, size=a_op.shape[0])

        # 3. Compute portfolio values corresponding to observed actions
        Pi_op.loc[:, t] = gamma * (
            Pi_op.loc[:, t + 1] - a_op.loc[:, t] * delta_S.loc[:, t]
        )
        Pi_hat.loc[:, t] = Pi_op.loc[:, t] - np.mean(Pi_op.loc[:, t])

        # 4. Compute rewards corrresponding to observed actions
        R_op.loc[:, t] = gamma * a_op.loc[:, t] * delta_S.loc[
            :, t
        ] - risk_lambda * np.var(Pi_op.loc[:, t])

    # Plot 10 reward functions
    plt.plot(R_op.iloc[idx_plot, :])
    plt.xlabel("Time Steps")
    plt.title("Reward Function")
    plt.show()

    ###############################################################################
    ###############################################################################

    # Override on-policy data with off-policy data
    a = copy.deepcopy(a_op)  # distrubed actions
    Pi = copy.deepcopy(Pi_op)  # disturbed portfolio values
    Pi_hat = copy.deepcopy(Pi_hat)
    R = copy.deepcopy(R_op)

    # make matrix A_t of shape (3 x num_MC x num_steps)
    num_MC = a.shape[0]  # number of simulated paths
    num_TS = a.shape[1]  # number of time steps
    a_1_1 = a.values.reshape((1, num_MC, num_TS))

    a_1_2 = 0.5 * a_1_1 ** 2
    ones_3d = np.ones((1, num_MC, num_TS))

    A_stack = np.vstack((ones_3d, a_1_1, a_1_2))

    data_mat_swap_idx = np.swapaxes(data_mat_t, 0, 2)

    # expand dimensions of matrices to multiply element-wise
    A_2 = np.expand_dims(A_stack, axis=1)  # becomes (3,1,10000,25)
    data_mat_swap_idx = np.expand_dims(
        data_mat_swap_idx, axis=0
    )  # becomes (1,12,10000,25)

    Psi_mat = np.multiply(
        A_2, data_mat_swap_idx
    )  # this is a matrix of size 3 x num_basis x num_MC x num_steps

    # now concatenate columns along the first dimension
    # Psi_mat = Psi_mat.reshape(-1, a.shape[0], a.shape[1], order='F')
    Psi_mat = Psi_mat.reshape(-1, N_MC, T + 1, order="F")

    ###############################################################################
    ###############################################################################

    # make matrix S_t

    Psi_1_aux = np.expand_dims(Psi_mat, axis=1)
    Psi_2_aux = np.expand_dims(Psi_mat, axis=0)

    S_t_mat = np.sum(np.multiply(Psi_1_aux, Psi_2_aux), axis=2)

    # clean up some space
    del Psi_1_aux, Psi_2_aux, data_mat_swap_idx, A_2

    ###############################################################################
    ###############################################################################

    def function_S_vec(t, S_t_mat, reg_param):
        """
        function_S_vec - calculate S_{nm} matrix from Eq. (75) (with a regularization!)
        Eq. (75) in QLBS Q-Learner in the Black-Scholes-Merton article

        num_Qbasis = 3 x num_basis, 3 because of the basis expansion (1, a_t, 0.5 a_t^2)

        Arguments:
        t - time index, a scalar, an index into time axis of S_t_mat
        S_t_mat - pandas.DataFrame of dimension num_Qbasis x num_Qbasis x T
        reg_param - regularization parameter, a scalar
        Return:
        S_mat_reg - num_Qbasis x num_Qbasis
        """
        ### START CODE HERE ### (≈ 4-5 lines of code)
        # S_mat_reg = your code goes here ...
        num_Qbasis = S_t_mat.shape[0]
        S_mat_reg = S_t_mat[:, :, t] + reg_param * np.eye(num_Qbasis)
        ### END CODE HERE ###
        return S_mat_reg

    def function_M_vec(t, Q_star, R, Psi_mat_t, gamma=gamma):
        """
        function_S_vec - calculate M_{nm} vector from Eq. (75) (with a regularization!)
        Eq. (75) in QLBS Q-Learner in the Black-Scholes-Merton article

        num_Qbasis = 3 x num_basis, 3 because of the basis expansion (1, a_t, 0.5 a_t^2)

        Arguments:
        t- time index, a scalar, an index into time axis of S_t_mat
        Q_star - pandas.DataFrame of Q-function values of dimension N_MC x T
        R - pandas.DataFrame of rewards of dimension N_MC x T
        Psi_mat_t - pandas.DataFrame of dimension num_Qbasis x N_MC
        gamma - one time-step discount factor $exp(-r \delta t)$
        Return:
        M_t - np.array of dimension num_Qbasis x 1
        """
        ### START CODE HERE ### (≈ 2-3 lines of code)
        # M_t = your code goes here ...
        M_t = np.dot(Psi_mat_t, R.loc[:, t] + gamma * Q_star.loc[:, t + 1])
        ### END CODE HERE ###

        return M_t

    ###############################################################################
    ###############################################################################

    # Call *function_S* and *function_M* for $t=T-1,...,0$ together with vector $\vec\Psi\left(X_t,a_t\right)$ to compute $\vec W_t$ and learn the Q-function $Q_t^\star\left(X_t,a_t\right)=\mathbf A_t^T\mathbf U_W\left(t,X_t\right)$ implied by the input data backward recursively with terminal condition $Q_T^\star\left(X_T,a_T=0\right)=-\Pi_T\left(X_T\right)-\lambda Var\left[\Pi_T\left(X_T\right)\right]$.

    # Plots of 5 optimal action $a_t^\star\left(X_t\right)$, optimal Q-function with optimal action $Q_t^\star\left(X_t,a_t^\star\right)$ and implied Q-function $Q_t^\star\left(X_t,a_t\right)$ paths are shown below.

    # ## Fitted Q Iteration (FQI)

    # implied Q-function by input data (using the first form in Eq.(68))
    Q_RL = pd.DataFrame([], index=range(1, N_MC + 1), columns=range(T + 1))
    Q_RL.iloc[:, -1] = -Pi.iloc[:, -1] - risk_lambda * np.var(Pi.iloc[:, -1])

    # optimal action
    a_opt = np.zeros((N_MC, T + 1))
    a_star = pd.DataFrame([], index=range(1, N_MC + 1), columns=range(T + 1))
    a_star.iloc[:, -1] = 0

    # optimal Q-function with optimal action
    Q_star = pd.DataFrame([], index=range(1, N_MC + 1), columns=range(T + 1))
    Q_star.iloc[:, -1] = Q_RL.iloc[:, -1]

    # max_Q_star_next = Q_star.iloc[:,-1].values
    max_Q_star = np.zeros((N_MC, T + 1))
    max_Q_star[:, -1] = Q_RL.iloc[:, -1].values

    num_basis = data_mat_t.shape[2]

    reg_param = 1e-3
    hyper_param = 1e-1

    # The backward loop
    for t in range(T - 1, -1, -1):

        # calculate vector W_t
        S_mat_reg = function_S_vec(t, S_t_mat, reg_param)
        M_t = function_M_vec(t, Q_star, R, Psi_mat[:, :, t], gamma)
        W_t = np.dot(
            np.linalg.inv(S_mat_reg), M_t
        )  # this is an 1D array of dimension 3M

        # reshape to a matrix W_mat
        W_mat = W_t.reshape((3, num_basis), order="F")  # shape 3 x M

        # make matrix Phi_mat
        Phi_mat = data_mat_t[t, :, :].T  # dimension M x N_MC

        # compute matrix U_mat of dimension N_MC x 3
        U_mat = np.dot(W_mat, Phi_mat)

        # compute vectors U_W^0,U_W^1,U_W^2 as rows of matrix U_mat
        U_W_0 = U_mat[0, :]
        U_W_1 = U_mat[1, :]
        U_W_2 = U_mat[2, :]

        # IMPORTANT!!! Instead, use hedges computed as in DP approach:
        # in this way, errors of function approximation do not back-propagate.
        # This provides a stable solution, unlike
        # the first method that leads to a diverging solution
        A_mat = function_A_vec(t, delta_S_hat, data_mat_t, reg_param)
        B_vec = function_B_vec(t, Pi_hat, delta_S_hat, S, data_mat_t)
        # print ('t =  A_mat.shape = B_vec.shape = ', t, A_mat.shape, B_vec.shape)
        phi = np.dot(np.linalg.inv(A_mat), B_vec)

        a_opt[:, t] = np.dot(data_mat_t[t, :, :], phi)
        a_star.loc[:, t] = a_opt[:, t]

        """    
        print("test "+str(t))
        print(str(Q_star.head())) 
        """

        max_Q_star[:, t] = (
            U_W_0 + a_opt[:, t] * U_W_1 + 0.5 * (a_opt[:, t] ** 2) * U_W_2
        )
        Q_star.iloc[:, t] = max_Q_star[:, t]

        # update dataframes
        # update the Q_RL solution given by a dot product of two matrices W_t Psi_t
        Psi_t = Psi_mat[:, :, t].T  # dimension N_MC x 3M
        Q_RL.loc[:, t] = np.dot(Psi_t, W_t)

        # trim outliers for Q_RL
        up_percentile_Q_RL = 95  # 95
        low_percentile_Q_RL = 5  # 5

        low_perc_Q_RL, up_perc_Q_RL = np.percentile(
            Q_RL.loc[:, t], [low_percentile_Q_RL, up_percentile_Q_RL]
        )

        # print('t = %s low_perc_Q_RL = %s up_perc_Q_RL = %s' % (t, low_perc_Q_RL, up_perc_Q_RL))

        # trim outliers in values of max_Q_star:
        flag_lower = Q_RL.loc[:, t].values < low_perc_Q_RL
        flag_upper = Q_RL.loc[:, t].values > up_perc_Q_RL
        Q_RL.loc[flag_lower, t] = low_perc_Q_RL
        Q_RL.loc[flag_upper, t] = up_perc_Q_RL

    endtime = time.time()

    ###############################################################################
    ###############################################################################
    # plot both simulations
    f, axarr = plt.subplots(3, 1)
    f.subplots_adjust(hspace=0.5)
    f.set_figheight(8.0)
    f.set_figwidth(8.0)

    step_size = N_MC // 10
    idx_plot = np.arange(step_size, N_MC, step_size)
    axarr[0].plot(a_star.T.iloc[:, idx_plot])
    axarr[0].set_xlabel("Time Steps")
    axarr[0].set_title(r"Optimal action $a_t^{\star}$")

    axarr[1].plot(Q_RL.T.iloc[:, idx_plot])
    axarr[1].set_xlabel("Time Steps")
    axarr[1].set_title(r"Q-function $Q_t^{\star} (X_t, a_t)$")

    axarr[2].plot(Q_star.T.iloc[:, idx_plot])
    axarr[2].set_xlabel("Time Steps")
    axarr[2].set_title(r"Optimal Q-function $Q_t^{\star} (X_t, a_t^{\star})$")
    plt.show()
    plt.savefig("QLBS_FQI_off_policy_summary_ATM_eta_%d.png" % (100 * eta), dpi=600)

    # Note that a from the DP method and a_star from the RL method are now identical by construction
    # plot 1 path

    num_path = 300  # 430 #  510

    plt.plot(a.T.iloc[:, num_path], label="DP Action")
    plt.plot(a_star.T.iloc[:, num_path], label="RL Action")
    plt.legend()
    plt.xlabel("Time Steps")
    plt.title("Optimal Action Comparison Between DP and RL for a sample path")
    plt.show()

    compTime = endtime - starttime

    return [Q_star.iloc[:, 0], compTime]
