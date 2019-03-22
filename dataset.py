import numpy as np
import tensorflow as tf

class MnistDataset():
    """
    Handler class for the MNIST  datset.
    Supports batch iteration as iterator.
    
    If n_iter is not specified, the iterator only does one pass on the data.
    """
    def __init__(self, batch_size, n_iter=None):
        mnist = tf.keras.datasets.mnist
        (Xtrain, ytrain), (Xtest, ytest) = mnist.load_data()
        X = np.concatenate((Xtrain, Xtest), axis=0)
        self.X = np.expand_dims(X, axis=-1) # add channel
        self.batch_size = batch_size
        self.size = len(X)
        if n_iter:
            self.n_iter = n_iter # max number of iterations
        else:
            self.n_iter = int(len(X)/self.batch_size)

    def __iter__(self):
        self.batch_idx = 0
        return self
    
    def __next__(self):
        if self.batch_idx >= self.n_iter:
            raise StopIteration
        inf = self.batch_idx * self.batch_size % self.size
        sup = (self.batch_idx + 1) * self.batch_size % self.size
        qinf = self.batch_idx * self.batch_size // self.size
        qsup = (self.batch_idx + 1) * self.batch_size // self.size
        self.batch_idx += 1
        #print(qinf, qsup)
        if qinf == qsup:
            return self.X[inf:sup]
        else:
            return np.concatenate((self.X[inf:], self.X[:sup]), axis=0)
    
    def __getattr__(self, attr):
        """Supports numpy array indexing""" # or does it ?
        return self.X[attr]
    
    def iterations(self, n_iter):
        """Change max number of iterations"""
        self.n_iter = n_iter
    
    






























        
