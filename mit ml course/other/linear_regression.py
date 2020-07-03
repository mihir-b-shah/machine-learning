
import numpy as np

#gen data on a V=IR plot
i = 0.0459
sigma = 1
dataset_size = 10000

def gen_data():
    x = np.random.uniform(-1000000,1000000,dataset_size)
    noise_samples = np.random.normal(0,sigma,dataset_size)
    y = i*x + noise_samples

    x_train = x[0:8000]
    y_train = y[0:8000]
    x_test = x[8000:]
    y_test = y[8000:]
    return x_train, y_train, x_test, y_test

def gen_toy_data():
    x = np.linspace(-20, 20, 41)
    y = 2*x
    x_train = np.transpose(np.matrix(x[0:30]))
    y_train = np.transpose(np.matrix(y[0:30]))
    x_test = np.transpose(np.matrix(x[30:]))
    y_test = np.transpose(np.matrix(y[30:]))
    return x_train, y_train, x_test, y_test

def converge(X, Y, n, reg_param):
    xT = np.transpose(X)
    O = np.ones((n,n))
    return np.linalg.inv(n*reg_param + xT @ X) @ (xT @ Y)
    #return np.linalg.inv(reg_param*n + (xT @ X) + ((1/n)*xT @ O @ Y)) @ ((xT @ Y) - ((1/n)*xT @ O @ Y))

def run():
    Xtr, Ytr, Xts, Yts = gen_toy_data()
    print(Xtr.shape)
    print(Ytr.shape)
    theta = converge(Xtr, Ytr, 30, 0)
    print(theta)
