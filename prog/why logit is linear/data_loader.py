import numpy as np

def load_data(data_filename):
    arr = np.genfromtxt(data_filename, delimiter=",")
    n = arr.shape[0]
    X1, X2, Y = list(map(lambda x: x.reshape(n,1), arr.T))
    Y = Y.astype('bool').reshape((n,))
    X = np.concatenate([X1,X2], axis = 1)

    return X,Y