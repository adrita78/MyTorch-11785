import numpy as np

# TODO: Implement this code similar to how you did for HW1P1 or HW2P1.

class Linear:

    def __init__(self, in_features, out_features, debug=False):

        self.W = np.zeros((out_features, in_features), dtype="f")
        self.b = np.zeros((out_features, 1), dtype="f")
        self.dLdW = np.zeros((out_features, in_features), dtype="f")
        self.dLdb = np.zeros((out_features, 1), dtype="f")

        self.debug = debug

    def __call__(self, A):
        return self.forward(A)    

    def forward(self, A):

        self.A = A
        self.N = A.shape[0]
        self.Ones = np.ones((self.N, 1), dtype="f")
        Z = np.dot(self.A, self.W.T) + np.dot(self.Ones, self.b.T)

        return Z

    def backward(self, dLdZ):

        dZdA = self.W.T
        dZdW = self.A
        
        dZdb = self.Ones
        dLdA = np.dot(dLdZ, dZdA.T)
        dLdW = np.dot(dLdZ.T, dZdW)
    
        dLdb = np.dot(dLdZ.T, dZdb)
        self.dLdW = dLdW / self.N
        self.dLdb = dLdb / self.N

        if self.debug:

            self.dZdA = dZdA
            self.dZdW = dZdW
            
            self.dZdb = dZdb
            self.dLdA = dLdA
            
            

        return dLdA
