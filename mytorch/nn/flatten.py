import numpy as np

class Flatten():

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, in_width)
        Return:
            Z (np.array): (batch_size, in_channels * in width)
        """

        self.A_shape = A.shape
        N = A.shape[0]
        C = A.shape[1]
        Win = A.shape[2]
        Z = A.reshape(A.shape[0], -1)




        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch size, in channels * in width)
        Return:
            dLdA (np.array): (batch size, in channels, in width)
        """

        self.dLdZ = dLdZ
        dLdA = dLdZ.reshape(self.A_shape)
        

        return dLdA
