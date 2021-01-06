import numpy as np
import matplotlib.pyplot as plt
import random
import csv

file_name = "winequality-red.csv"

with open(file_name,'r') as csvfile:
    reader = csv.DictReader(csvfile, restkey = None, restval = None, dialect = 'excel')
    data = list(reader)
    random.shuffle(data)

list_len = len(data)
print('Size of total data : {}'.format(list_len))

X = np.zeros((list_len,11), dtype = float)
Y = np.zeros((list_len,1), dtype = float)

for i in range(0,list_len):
    values = list(data[i].values())

    X[i,0] = values[0]
    X[i,1] = values[1]
    X[i,2] = values[2]
    X[i,3] = values[3]
    X[i,4] = values[4]
    X[i,5] = values[5]
    X[i,6] = values[6]
    X[i,7] = values[7]
    X[i,8] = values[8]
    X[i,9] = values[9]
    X[i,10] = values[10]
    Y[i,0] = values[11]

print(X)

print(np.amin(X,axis=0))

print(np.amax(X,axis=0))

X = (X - np.amin(X,axis=0))/(np.amax(X,axis=0) - np.amin(X,axis=0))
X = X.T

print("Data head :")
print(X)
print("\n")

train_split = 7

X_train = X[:,0:train_split*(X.shape[1])//10]
X_test = X[:,train_split*X.shape[1]//10:X.shape[1]]

Y_train = Y[:,0:train_split*(Y.shape[1])//10]
Y_test = Y[:,train_split*(Y.shape[1])//10:Y.shape[1]]

assert((X_train.shape[1] + X_test.shape[1]) == list_len)

X_train_data_len = X_train.shape
print(' X Train data shape : {}'.format(X_train_data_len))
X_test_data_len = X_test.shape
print(' X Test data shape : {}'.format(X_test_data_len))
Y_train_data_len = Y_train.shape
print(' Y Train data shape : {}'.format(Y_train_data_len))
Y_test_data_len = Y_test.shape
print(' Y Test data shape : {}'.format(Y_test_data_len))

def sigmoid(Z):

    A = 1/(1 + np.exp(-Z))

    return A

def sigmoid_der(A):

    dAdZ = A*(1-A)

    return dAdZ

def relu(Z):

    A = np.maximum(0,Z)

    return A

def relu_der(A):

    dAdZ = np.where(A>0,1,0)

    return dAdZ

def initialize_parameters(dims):

    n = len(dims)
    params = {}

    for l in range(1,n):
        params["W"+str(l)] = np.random.randn(dims[l],dims[l-1])*0.1
        params["b"+str(l)] = np.zeros((dims[l],1),dtype=float)

    return params


def forward_propagation(params,A):

    L = len(params) // 2
    caches = []

    for l in range(1,L):
        A_prev = A
        Z = np.dot(params["W"+str(l)],A_prev) + params["b"+str(l)]
        linear_cache = (A_prev,params["W"+str(l)],params["b"+str(l)])
        A = relu(Z)
        cache = (linear_cache,A)
        caches.append(cache)

    A_prev = A
    Z = np.dot(params["W"+str(L)],A_prev) + params["b"+str(L)]
    linear_cache = (A_prev,params["W"+str(L)],params["b"+str(L)])
    AL = sigmoid(Z)
    cache = (linear_cache,AL)
    caches.append(cache)

    return AL, caches


def compute_cost(AL,Y):

    m = Y.shape[1]

    cost = -np.sum(np.multiply(Y,np.log(AL)) + np.multiply(1-Y,np.log(1-AL)), axis=1, keepdims=True)/m

    cost = np.squeeze(cost)

    return cost

def backward_propagation(AL,Y,caches):

    grads = {}
    L = len(caches)
    m = Y.shape[1]

    dAL = -(np.divide(Y,AL) - np.divide(1-Y,1-AL))/m
    last_cache = caches[L-1]
    linear_cache, A = last_cache
    A_prev, W ,b = linear_cache
    grads["dA"+str(L)] = dAL
    grads["dW"+str(L)] = np.dot(np.multiply(grads["dA"+str(L)],sigmoid_der(A)),A_prev.T)
    grads["db"+str(L)] = np.sum(np.multiply(grads["dA"+str(L)],sigmoid_der(A)),axis=1,keepdims=True)

    for l in range(L-1,0,-1):
        if l == L-1:
            grads["dA"+str(l)] = np.dot(W.T,np.multiply(grads["dA"+str(l+1)],sigmoid_der(A)))
        else:
            grads["dA"+str(l)] = np.dot(W.T,np.multiply(grads["dA"+str(l+1)],relu_der(A)))
        current_cache = caches[l-1]
        linear_cache, A = current_cache
        A_prev, W, b = linear_cache
        grads["dW"+str(l)] = np.dot(grads["dA"+str(l)]*relu_der(A),A_prev.T)
        grads["db"+str(l)] = np.sum(grads["dA"+str(l)]*relu_der(A),axis=1,keepdims=True)

    return grads

def update_parameters_rmsprop(params, grads, learning_rate, v_w, v_b, beta, e):

    L = len(params) // 2

    for l in range(1,L):
        v_w = beta * v_w + (1 - beta) * grads["dW"+str(l)]**2
        v_b = beta * v_b + (1 - beta) * grads["db"+str(l)]**2
        params["W"+str(l)] = params["W"+str(l)] - (learning_rate/np.sqrt(v_w + e)) * grads["dW"+str(l)]
        params["b"+str(l)] = params["b"+str(l)] - (learning_rate/np.sqrt(v_b + e)) * grads["db"+str(l)]

    return params, v_w, v_b, beta, e

def predict(params,X):

    AL,cache = forward_propagation(params,X)
    predictions = np.where(AL>0.5,1,0)

def model(X,Y,dims,learning_rate,num_iterations,print_cost=True):

    costs = []
    params = initialize_parameters(dims)
    v_w, v_b, beta, e = 0, 0, 0.9, 1e-8

    for i in range(0, num_iterations):

        AL, caches = forward_propagation(params,X)

        cost = compute_cost(AL,Y)

        grads = backward_propagation(AL,Y,caches)

        params,v_w, v_b, beta, e = update_parameters(params,grads,learning_rate,v_w, v_b, beta, e)

        if print_cost and i%100 == 0:
            print("Cost after iteration %i: %f" %(i,cost))
            costs.append(cost)

    return params

dims = [11,11,11,1]
params = model(X_train,Y_train,dims,learning_rate=0.5,num_iterations=5000,print_cost=True)

# Enter the following values

# Fixed acidity range (4.6 - 15.9) ; max = 15.9
FA = 10
# Volatile acidity range (0.12 - 1.58) ; max = 1.58
VA = 0.5
# Citric acid range (0 - 1) ; max = 1
CA = 0.8
# Residual sugars (0.9 - 15.5) ; max = 15.5
RS = 10
# Chlorides (0.01 - 0.61) ;  max = 0.61
C = 0.03
# Free sulphur dioxide (1 - 72) ; max = 72
FS = 30
# Total sulphure dioxide (6 - 289) ; max = 289
TS = 50
# Density (0.99 - 1) ; max = 1
D = 0.9997
# pH (2.74 - 4.01) ; max = 4.01
PH = 3.5
# Sulphates (0.33 - 2) ; max = 2
S = 1.5
# alcohol (8.4 - 14.9) ; max = 14.9
A = 12

user_X = np.zeros((11,1), dtype = float)
user_X[0] = FA/15.9
user_X[1] = VA/1.58
user_X[2] = CA/1
user_X[3] = RS/15.5
user_X[4] = C/0.61
user_X[5] = FS/72
user_X[6] = TS/289
user_X[7] = D/1
user_X[8] = PH/4.01
user_X[9] = S/2
user_X[10] = A/14.9
