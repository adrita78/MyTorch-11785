import numpy as np


class MSELoss:

    def forward(self, A, Y):
        """
        Calculate the Mean Squared error
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: MSE Loss(scalar)

        """

        self.A = A
        self.Y = Y
        self.N = self.A.shape[0]
        self.C = self.A.shape[1]
        Ones_N = np.ones((self.N,1))
        Ones_C = np.ones((self.C,1))

        se = np.square(np.subtract(self.A,self.Y))
        sse = Ones_N.T@se@Ones_C
        mse = 1/2 * np.divide(sse, np.dot(self.N, self.C))

        return mse

    def backward(self):

        dLdA = (self.A-self.Y)/(self.N*self.C)

        return dLdA


class CrossEntropyLoss:

    def forward(self, A, Y):
        """
        Calculate the Cross Entropy Loss
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: CrossEntropyLoss(scalar)

        Refer the the writeup to determine the shapes of all the variables.
        Use dtype ='f' whenever initializing with np.zeros()
        """
        self.A = A
        self.Y = Y
        N = A.shape[0]
        C = A.shape[1]

        Ones_C = np.ones((C,1))
        Ones_N = np.ones((N,1))

        self.softmax = np.divide(np.exp(self.A), np.sum(np.exp(self.A),axis=1)[:, np.newaxis])
        crossentropy = (-self.Y*np.log(self.softmax))@Ones_C
        sum_crossentropy = np.dot(Ones_N.T,crossentropy)
        L = sum_crossentropy/N

        return L

    def backward(self):

        dLdA = (self.softmax-self.Y)

        return dLdA
