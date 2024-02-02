import numpy as np
from resampling import *


class MaxPool2d_stride1():

    def __init__(self, kernel):
      
        self.kernel = kernel
        

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        N,C, Hin, Win = A.shape
        Kh = self.kernel
        Kw = self.kernel

        Sh, Sw = 1, 1
        Hout = int((Hin - Kh) / Sh) + 1
        Wout = int((Win - Kw) / Sw) + 1
        
        # Initialize output tensor
        self.argmax = np.zeros((N,C, Hout, Wout))
        Z = np.zeros((N, C, Hout, Wout))
        
        # Compute max pooling
        for n in range(N):
            for c in range(C):
                for i in range(Hout):
                    for j in range(Wout):
                        Z[n, c, i, j] = np.max(A[n, c, i*Sh:i*Sh+Kh, j*Sw:j*Sw+Kw])
                        self.argmax[n, c, i, j] = np.argmax(A[n, c, i*Sh:i*Sh+Kh, j*Sw:j*Sw+Kw])
                        
                        
        return Z



        

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        self.dLdZ = dLdZ
        Kh = self.kernel
        Kw = self.kernel
        N,C, Hin, Win = self.A.shape

        Sh, Sw = 1, 1
        Hout = int((Hin - Kh) / Sh) + 1
        Wout = int((Win - Kw) / Sw) + 1
        
        # Initialize gradient tensor
        dLdA = np.zeros(self.A.shape)
        
        # Compute gradient of max poolingaa =
        for n in range(N):
            for c in range(C):
                for i in range(Hout):
                    for j in range(Wout):
                        sub_region = self.A[n, c, i*Sh:i*Sh+Kh, j*Sw:j*Sw+Kw]
                        mask = (sub_region == np.max(sub_region))
                        assert(Kh == Kw)
                        if np.sum(mask)!=1:
                            print("Error")
                        assert(mask.shape == (Kw,Kh))
                        if np.argmax(mask) != self.argmax[n,c,i,j]: # As suggested by TA
                            print("Error")
                        # x_in, y_in =np.unravel_index(int(self.argmax[n,c,i,j]), (Kh,Kw))
                        # dLdA[n, c, i+int(x_in), j+int(y_in)] += dLdZ[n, c, i, j]
                        dLdA[n, c, i*Sh:i*Sh+Kh, j*Sw:j*Sw+Kw] += (mask * dLdZ[n, c, i, j])
                        
        return dLdA







        


        


class MeanPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        N,C,Hin,Win = A.shape
        Kh , Kw = self.kernel, self.kernel
        Sh , Sw = 1,1
        Hout = int((Hin - Kh) /Sh) + 1
        Wout = int((Win - Kw)/ Sw) + 1

        Z = np.zeros((N, C, Hout, Wout))

        for n in range(N):
            for c in range(C):
                for i in range(Hout):
                    for j in range(Wout):
                        Z[n,c,i,j] = np.mean(A[n,c,i*Sh:i*Sh+Kh,j*Sw:j*Sw+Kw])
                        
        return Z                

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        self.dLdZ = dLdZ    
        N, C, Hin, Win = self.A.shape
        Kh, Kw = self.kernel, self.kernel
        Sh , Sw = 1,1
        Hout = int((Hin - Kh)/Sh) + 1
        Wout = int((Win - Kw)/Sw) + 1

        dLdA = np.zeros(self.A.shape)

        for n in range(N):
            for c in range(C):
                for i in range(Hout):
                    for j in range(Wout):
                        dLdA[n,c, i*Sh:i*Sh+Kh, j*Sw:j*Sw+Kw] += dLdZ[n,c,i,j] / (Kh*Kw)
        return dLdA                


        


class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        Z1 = self.maxpool2d_stride1.forward(A)
        Z2 = self.downsample2d.forward(Z1)

        return Z2



        

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdZ1 = self.downsample2d.backward(dLdZ)

        dLdA = self.maxpool2d_stride1.backward(dLdZ1)
        
        return dLdA

        


class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        Z1 = self.meanpool2d_stride1.forward(A)
        Z2 = self.downsample2d.forward(Z1)
        return Z2


        

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        

        dLdZ1 = self.downsample2d.backward(dLdZ)

        dLdA = self.meanpool2d_stride1.backward(dLdZ1)

        return dLdA
        
