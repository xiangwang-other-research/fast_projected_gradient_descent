# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 09:26:01 2019
Apply the Frank Wolf algorithm and Projected gradient descent to solve optimization problem 
min|X - Y|_F^2 s.t. \sum max(|X_i|) < k
@author: wx100
"""
import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
import time
# The Frank Wollfe algorithm 
"""
Input: original m*n matrix Y of real value and constraint k

Output: X
"""
def Frank_Wolfe(Y, epsilon):
    nrow = len(Y)
    ncol = len(Y[0])
    X = np.zeros((nrow, ncol)) # X store the final output value.
    record_X = np.ones((nrow,ncol)) # record_X store the X value in the last loop to judge if X converges.
    k = 0;  # k is the loop index
    # compute S = min S * |X - Y| s.t. \sum max(|S_i|) < k
    while np.linalg.norm(X - record_X, 'fro') > 0.01:
        k = k + 1;
        temp = X - Y
        abs_temp = np.abs(temp)
        # find the column vector in matrix X-Y with biggest norm-1
        temp_sum = np.sum(abs_temp,axis = 0)
        S = np.zeros((nrow,ncol))
        S_value = np.amax(temp_sum)
        S_index = np.where(temp_sum == S_value)
        for i in range(0,len(S_index)):
            S[:, S_index[i][0]] = - epsilon / len(S_index) * np.sign(temp[:, S_index[i][0]])
        #update X
        mylambda = 2 / (k+1)
        record_X = X
        X = (1 - mylambda) * X + mylambda * S
    return X

def cvx(Y,epsilon):
    nrow = len(Y)
    ncol = len(Y[0])
    X = cp.Variable((nrow, ncol))
    constraints = 0
    for i in range(0,ncol):
        constraints += cp.norm(X[:,i], 'inf')
        
    prob = cp.Problem(cp.Minimize(cp.norm(X-Y,"fro")), constraints <= epsilon)
    prob.solve()
    return X.value

start = 10 
end = 250
step = 10
epsilon = 1
loop_time = 100
length = int((end - start -1) / step) + 1
run_time1 = np.zeros(length) # store the computation time of Frank Wollfe algorithm
run_time2 = np.empty(length) # store the computation time of convex package
matrix_size = np.arange(start,end,step)
for loop_index in range(0,loop_time):
    index = 0
    for i in matrix_size:
        # create the n*n random matrix Y and then compute the corresponded projected matrix X
        Y = 5 * np.random.rand(i,i)
        begin_time = time.clock()
        X1 = Frank_Wolfe(Y,epsilon)
        run_time1[index] += time.clock() - begin_time    
    #    X2 = cvx(Y,epsilon)
    #    run_time2[index] = time.clock() - run_time1[index] - begin_time
        index = index + 1
ave_run_time1 = run_time1 / loop_time
plt.plot(matrix_size, run_time1, label='Frank_Wolfe')
#plt.plot(matrix_size, run_time2, label='cvx')
#plt.plot(matrix_size, run_time3, label='cvx')
plt.xlabel('matrix size n')
plt.ylabel('run time/s')
plt.title("Run time comparision")
plt.legend()
plt.show()
#epsilon = 1
#Y = 0.1 * np.ones((5,5))
#X = Frank_Wolfe(Y,epsilon)