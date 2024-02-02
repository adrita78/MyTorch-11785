import numpy as np


class Upsample1d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        self.A = A
        N = A.shape[0]
        C = A.shape[1]
        Win = A.shape[2]
        Wout = self.upsampling_factor * (Win-1) + 1
        Z = np.zeros((N,C,Wout))
        for i in range(Win):
            Z[:, :, i*self.upsampling_factor] = A[:, :, i]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """

        self.dLdZ = dLdZ
        N = dLdZ.shape[0]
        C = dLdZ.shape[1]
        Wout = dLdZ.shape[2] 
        Win = int((Wout -1) / self.upsampling_factor) +1
        dLdA = np.zeros((N,C,Win))
        for i in range(Win):
            dLdA[:,:,i] = dLdZ[:,:,i*self.upsampling_factor]

        return dLdA


class Downsample1d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """

        self.A = A
        N = A.shape[0]
        C = A.shape[1]
        Win = A.shape[2]
        Wout = (Win-1)// self.downsampling_factor + 1
        Z = np.zeros((N, C, Wout))

        for i in range(Wout):
            if i*self.downsampling_factor < A.shape[2]:
                Z[: , :, i] = A[:,:,i*self.downsampling_factor]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        self.dLdZ = dLdZ
        dLdA = np.zeros(self.A.shape)
        for i in range(dLdZ.shape[2]):
            if i* self.downsampling_factor < dLdA.shape[2]:
                dLdA[:,:, i*self.downsampling_factor] = dLdZ[:,:,i]
            

        

        return dLdA


class Upsample2d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """

        self.A = A
        N = A.shape[0]
        C = A.shape[1]
        Hin = A.shape[2]
        Win = A.shape[3]
        Hout = self.upsampling_factor * (Hin-1)+1
        Wout = self.upsampling_factor * (Win-1) + 1
        Z = np.zeros((N,C,Hout,Wout))
        for i in range(Hin):
            for j in range(Win):
                Z[:,:, i*self.upsampling_factor, j*self.upsampling_factor] = A[:,:,i,j]


        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        self.dLdZ = dLdZ
        N = dLdZ.shape[0]
        C = dLdZ.shape[1]
        Hout = dLdZ.shape[2]
        Wout = dLdZ.shape[3]
        Hin = (Hout -1) // self.upsampling_factor + 1
        Win = (Wout -1) // self.upsampling_factor + 1
        dLdA = np.zeros((N, C, Hin, Win))
        for i in range(Hin):
            for j in range(Win):
                dLdA[:,:,i,j] = dLdZ[:,:,i*self.upsampling_factor, j*self.upsampling_factor]


        return dLdA


class Downsample2d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """

        self.A = A
        N = A.shape[0]
        C= A.shape[1]
        Hin= A.shape[2]
        Win = A.shape[3]
        Hout = (Hin-1)//self.downsampling_factor + 1
        Wout = (Win -1)//self.downsampling_factor + 1
        Z = np.zeros((N,C,Hout,Wout))
        for i in range(Hout):
            for j in range(Wout):

                Z[: , : , i, j] = A[:, : , i*self.downsampling_factor, j*self.downsampling_factor]
        return Z


        

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        self.dLdZ = dLdZ
       

       
        dLdA = np.zeros(self.A.shape)

        for i in range(dLdZ.shape[2]):
            for j in range(dLdZ.shape[3]):
                if i * self.downsampling_factor < dLdA.shape[2] and j*self.downsampling_factor < dLdA.shape[3]:
                    dLdA[:,:, i*self.downsampling_factor, j*self.downsampling_factor] = dLdZ[:,:,i,j]
                



        return dLdA
