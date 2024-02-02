# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np


class BatchNorm2d:

    def __init__(self, num_features, alpha=0.9):
        
        self.alpha = alpha
        self.eps = 1e-8

        self.Z = None
        self.NZ = None
        self.BZ = None

        self.BW = np.ones((1, num_features, 1, 1))
        self.Bb = np.zeros((1, num_features, 1, 1))
        self.dLdBW = np.zeros((1, num_features, 1, 1))
        self.dLdBb = np.zeros((1, num_features, 1, 1))

        self.M = np.zeros((1, num_features, 1, 1))
        self.V = np.ones((1, num_features, 1, 1))

        # inference parameters
        self.running_M = np.zeros((1, num_features, 1, 1))
        self.running_V = np.ones((1, num_features, 1, 1))

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, Z, eval=False):
        """
        The eval parameter is to indicate whether we are in the
        training phase of the problem or are we in the inference phase.
        So see what values you need to recompute when eval is True.
        """
        self.Z = Z
        self.N = self.Z.shape[0]
     

        if eval:

            # inference mode
            NZ = (self.Z - self.running_M) / np.sqrt(self.running_V + self.eps)
            BZ = self.BW * NZ + self.Bb

            return BZ
            
      
            
        # training mode

        self.M = np.mean(self.Z, axis=(0, 2, 3), keepdims=True)
        self.V = np.var(self.Z, axis=(0, 2, 3), keepdims=True)
        self.NZ = (self.Z - self.M) / np.sqrt(self.V + self.eps)
        self.BZ = self.BW * self.NZ + self.Bb

        self.running_M = self.alpha * self.running_M + (1 - self.alpha) * self.M
        self.running_V = self.alpha * self.running_V + (1 - self.alpha) * self.V
            

        return self.BZ


    def backward(self, dLdBZ):
        self.dLdBW = np.sum((dLdBZ * self.NZ), axis=(0, 2, 3), keepdims=True)
        self.dLdBb = np.sum(dLdBZ, axis=(0, 2, 3), keepdims=True)

        dLdNZ = (dLdBZ * self.BW)
        dLdV = (-1 / 2 )* np.sum(dLdNZ * (self.Z - self.M) * ((self.V + self.eps) ** (-1.5)), axis=(0, 2, 3), keepdims=True)
        dLdM = -np.sum(dLdNZ * ((self.V + self.eps) ** (-0.5)), axis=(0, 2, 3), keepdims=True) -(2 / self.N )* dLdV * np.sum(self.Z - self.M, axis=(0, 2, 3), keepdims=True)

        dLdZ = dLdNZ * ((self.V + self.eps) ** (-0.5)) +( 2 / (self.N * dLdBZ.shape[2]*dLdBZ.shape[3]))*dLdV * (self.Z - self.M) + dLdM / (self.N * dLdBZ.shape[2]*dLdBZ.shape[3])
        

        return dLdZ