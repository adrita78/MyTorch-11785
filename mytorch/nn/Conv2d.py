import numpy as np
from resampling import *


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels,
                 kernel_size, weight_init_fn=None, bias_init_fn=None):

        # Do not modify this method

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        self.A = A

        self.A = A
        N, Cin, Hin, Win = A.shape
        K = self.kernel_size
        Cout = self.out_channels
        Hout = Hin - K + 1
        Wout = Win - K + 1

        Z = np.zeros((N, Cout, Hout, Wout))
        for n in range(N):
            for c in range(Cout):
                for h in range(Hout):
                    for w in range(Wout):
                        Z[n, c, h, w] = np.sum(self.W[c] * A[n, :, h:h+K, w:w+K]) + self.b[c]

        return Z


    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        N, Cout, Hout, Wout = dLdZ.shape
        K = self.kernel_size
        Cin = self.in_channels

        dLdA = np.zeros((N, Cin, Hout+K-1, Wout+K-1))

        self.dLdW = np.zeros((Cout, Cin, K, K))
        self.dLdb = np.zeros((Cout,))

        # backward pass
        for n in range(N):
            for c in range(Cout):
                for h in range(Hout):
                    for w in range(Wout):
                        self.dLdb[c] += dLdZ[n, c, h, w]
                        self.dLdW[c] += dLdZ[n, c, h, w] * self.A[n, :, h:h+K, w:w+K]
                        dLdA[n, :, h:h+K, w:w+K] += dLdZ[n, c, h, w] * self.W[c]

        return dLdA









class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride

        # Initialize Conv2d() and Downsample2d() isntance
        self.stride = stride
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)
        self.downsample2d = Downsample2d(stride)
       

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        # Call Conv2d_stride1
        # # Call Conv2d_stride1 forward
        Z = self.conv2d_stride1.forward(A)
        
        # Call downsample2d forward
        Z = self.downsample2d.forward(Z)
        
        return Z

        # downsample
        

        

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        # Call downsample1d backward
        dLdZ = self.downsample2d.backward(dLdZ)
        
        # Call Conv2d_stride1 backward
        dLdA = self.conv2d_stride1.backward(dLdZ)

        # Call Conv1d_stride1 backward
        

        return dLdA
