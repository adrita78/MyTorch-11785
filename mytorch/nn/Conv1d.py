# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from resampling import *


class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A = A

        self.A = A
        N, Cin, Win = A.shape
        K = self.kernel_size
        Cout = self.out_channels
        Wout = Win - K + 1

        Z = np.zeros((N, Cout, Wout))
        for n in range(N):
            for c in range(Cout):
                for w in range(Wout):
                    Z[n, c, w] = np.sum(self.W[c] * A[n, :, w:w+K]) + self.b[c]

        return Z
        

        

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        N, Cout, Wout = dLdZ.shape
        K = self.kernel_size
        Cin = self.in_channels

        dLdA = np.zeros((N, Cin, Wout+K-1))

        self.dLdW = np.zeros((Cout, Cin, K))
        self.dLdb = np.zeros((Cout,))

        # backward pass
        for n in range(N):
            for c in range(Cout):
                for w in range(Wout):
                    self.dLdb[c] += dLdZ[n, c, w]
                    self.dLdW[c] += dLdZ[n, c, w] * self.A[n, :, w:w+K]
                    dLdA[n, :, w:w+K] += dLdZ[n, c, w] * self.W[c]

        return dLdA



        
    
    


        


class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.stride = stride
        self.kernel_size = kernel_size

        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels,kernel_size,weight_init_fn,bias_init_fn)
        self.downsample1d = Downsample1d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A = A

        # Call Conv1d_stride1
        Z = self.conv1d_stride1.forward(A)

        # downsample
        Z = self.downsample1d.forward(Z)

        
        

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        self.dLdZ = dLdZ
        # Call downsample1d backward
        dLdZ = self.downsample1d.backward(dLdZ)
        dLdA = self.conv1d_stride1.backward(dLdZ)
        
        return dLdA




 
