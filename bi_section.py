# -*- coding: utf-8 -*-
"""
This algorithm compares the three different methods to compute the projection of a matrix. 
"""
import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
import time


"""
This function applies bisection algorithm to compute the project gradient descent.
Input: Y, a n times n matrix of real number.
k, the constrains the out matrix X need to obey. |X|_inf < k

Output: the matrix X that follow the rules X = argmin|X - Y|_F s.t. |X|_inf <= k
"""
def bisection(Y,k):
    # transfer the original problem into argmin|X - Y|_F s.t. Y >= 0, X >=0, |X|_inf = k
    Y_sgn = np.sign(Y)
    abs_Y = np.abs(Y)
    temp_Y = abs_Y.sum(axis = 1) 
    index = np.where(temp_Y > 1)
    index = np.array(index)
    # if |Y|_infty < k. Then directly set X = Y
    if not len(index):
        return Y
    # solve problem argmin|X - Y|_F s.t. Y >= 0, X >=0, |X|_inf = k
    Q = abs_Y[index,]
    Q = np.transpose(np.squeeze(Q))
    # compute the optimal value of Lagrange multiplier of the origional problem.
    lambda_lower = np.zeros(len(index[0]));
    lambda_upper = Q.max(axis = 0);
    while np.linalg.norm(lambda_lower - lambda_upper) > 0.01:
        lambda_middle = (lambda_lower + lambda_upper) / 2 
        temp_Q = np.maximum(Q - lambda_middle, 0)
        temp_Q_sum = np.sum(temp_Q, axis = 0)
        index_upper = np.where(temp_Q_sum <= k)
        lambda_upper[index_upper] = lambda_middle[index_upper];
        index_lower = np.where(temp_Q_sum > k)
        lambda_lower[index_lower] = lambda_middle[index_lower];
    Q = np.transpose(np.maximum(Q - lambda_middle,0))
    X_output = np.array(Y);
    X_output[index, :] = np.multiply(Q, Y_sgn[index,:]);
    return X_output;


"""
This function applies bisection algorithm to compute the project gradient descent.
Input: Y, a n times n matrix of real number.
k, the constrains the out matrix X need to obey. |X|_inf < k

Output: the matrix X that follow the rules X = argmin|X - Y|_F s.t. |X|_inf <= k
"""
def order(Y,k):
    # transfer the original problem into argmin|X - Y|_F s.t. Y >= 0, X >=0, |X|_inf = k
    Y_sgn = np.sign(Y)
    abs_Y = np.abs(Y)
    temp_Y = abs_Y.sum(axis = 1) 
    index = np.where(temp_Y > 1)
    index = np.array(index)
    Q = np.squeeze(abs_Y[index,])
    # solve problem argmin|X - Y|_F s.t. Y >= 0, X >=0, |X|_inf = k
    size = np.shape(Q)
    Q = np.squeeze(Q)
    P = np.sort(Q, axis = 1)
    P = np.transpose(P)
    P = P[::-1]
    P = np.transpose(P)
    P_temp = np.matmul((k - np.cumsum(P, axis = 1)),np.diag(np.divide(1, np.arange(1,size[1]+1))))
    P_lambda = P + P_temp
    lambda_index = np.argmax(P_lambda < 0, axis = 1)
    lambda_index = np.mod(lambda_index - 1, size[1])
    P = np.transpose(np.maximum(np.transpose(Q) + P_temp[np.arange(0,size[0]),lambda_index], 0))
    P_output = np.array(Y)
    P_output[index,:] = np.multiply(P, Y_sgn[index,:])
    return P_output
    
        
"""
This function applies cvxpy to compute the projection of a matrix on a convex set.
Input: Y, a n times n matrix of real number.
k, the constrains the out matrix X need to obey. |X|_inf < k

Output: the matrix X that follow the rules X = argmin|X - Y|_F s.t. |X|_inf <= k
"""
def cvx(Y,k):
    nrow = len(Y)
    ncol = len(Y[0])
    X = cp.Variable((nrow, ncol))
    prob = cp.Problem(cp.Minimize(cp.norm(X-Y,"fro")),[cp.norm(X,"inf") <= k])
    prob.solve()
    return X.value

#Y = np.array([[1,1,1,1],[0.1,0.2,0.3,0.4],[-1,-1,-1,-1],[-0.1,-0.2,-0.3,-0.4]])
# we try to compute the matrix computation time from initial 10*10 matrix to final 1000*1000 matrix.
start = 10 
end = 100
step = 10
length = int((end - start -1) / step) + 1
run_time1 = np.empty(length) # store the computation time of bisection based algorithm
run_time2 = np.empty(length) # store the computation time of order based algorithm
run_time3 = np.empty(length) # store the computation time of using cvxpy package
matrix_size = np.arange(start,end,step)
index = 0
k = 1
for i in matrix_size:
    # create the n*n random matrix Y and then compute the corresponded projected matrix X
    Y = 5 * np.random.rand(i,i)
    begin_time = time.clock()
    X1 = bisection(Y,k)
    run_time1[index] = time.clock() - begin_time    
    X2 = order(Y,k)
    run_time2[index] = time.clock() - run_time1[index] - begin_time
#    X3 = cvx(Y,k)
#    run_time3[index] = time.clock() - run_time2[index] - run_time1[index] - begin_time
    index = index + 1
plt.plot(matrix_size, run_time1, label='bisection')
plt.plot(matrix_size, run_time2, label='order')
#plt.plot(matrix_size, run_time3, label='cvx')
plt.xlabel('matrix size n')
plt.ylabel('run time/s')
plt.title("Run time comparision")
plt.legend()
plt.show()