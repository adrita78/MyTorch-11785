import numpy as np
from activation import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, in_dim, hidden_dim):
        self.d = in_dim
        self.h = hidden_dim
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.brx = np.random.randn(h)
        self.bzx = np.random.randn(h)
        self.bnx = np.random.randn(h)

        self.brh = np.random.randn(h)
        self.bzh = np.random.randn(h)
        self.bnh = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbrx = np.zeros((h))
        self.dbzx = np.zeros((h))
        self.dbnx = np.zeros((h))

        self.dbrh = np.zeros((h))
        self.dbzh = np.zeros((h))
        self.dbnh = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx
        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h_prev_t: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h_prev_t

        x = np.expand_dims(x,1)
        h_prev_t = np.expand_dims(h_prev_t,1)
        #self.r = r_t
        #self.z = z_t
        #self.n = n_t
        
        # Add your code here.
        # Define your variables based on the writeup using the corresponding
        # names below.
        Wrx, Wzx, Wnx = self.Wrx, self.Wzx, self.Wnx
        Wrh , Wzh, Wnh = self.Wrh, self.Wzh, self.Wnh
        brx , bzx, bnx = self.brx, self.bzx, self.bnx
        brh , bzh, bnh = self.brh, self.bzh, self.bnh

        r_act, z_act, h_act = self.r_act, self.z_act, self.h_act

        self.z = self.z_act(np.matmul(self.Wzx, x.reshape((-1, 1))) + self.bzx.reshape((-1,1)) + np.matmul(self.Wzh, h_prev_t.reshape((-1, 1))) + self.bzh.reshape((-1,1)))

        self.r = self.r_act(self.Wrx @ x + self.brx.reshape((-1, 1)) + self.Wrh @ h_prev_t + self.brh.reshape(-1, 1))
        self.n = self.h_act(self.Wnx @ x + self.bnx.reshape((-1, 1)) + self.r * (self.Wnh @ h_prev_t + self.bnh.reshape(-1, 1)))
        h_t = (1 - self.z) * self.n + self.z * h_prev_t

        self.r = np.squeeze(self.r)
        self.z = np.squeeze(self.z)
        self.n = np.squeeze(self.n)
        h_t = np.squeeze(h_t)

        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        assert h_t.shape == (self.h,) # h_t is the final output of you GRU cell.

        # return h_t
        return h_t
      

    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (1, input_dim)
            derivative of the loss wrt the input x.

        dh_prev_t: (1, hidden_dim)
            derivative of the loss wrt the input hidden h.

        """
        # 1) Reshape self.x and self.hidden to (input_dim, 1) and (hidden_dim, 1) respectively
        #    when computing self.dWs...
        # 2) Transpose all calculated dWs...
        # 3) Compute all of the derivatives
        # 4) Know that the autograder grades the gradients in a certain order, and the
        #    local autograder will tell you which gradient you are currently failing.

        # ADDITIONAL TIP:
        # Make sure the shapes of the calculated dWs and dbs  match the
        # initalized shapes accordingly

        delta = np.reshape(delta, (-1,1))

        r = self.r.reshape((-1,1))
        z = self.z.reshape((-1,1))
        n = self.n.reshape((-1,1))

        h_prev_t = self.hidden.reshape((-1,1))
        x = self.x.reshape((-1, 1)).transpose() 

        dn = delta * (1 - z)
        dz = delta * (-n + h_prev_t)

        grad_n = dn * self.h_act.backward(n)  
        self.dWnx = grad_n @ x
        self.dbnx = np.squeeze(grad_n)
        dr = grad_n * (self.Wnh @ (np.expand_dims(self.hidden, 1)) + self.bnh.reshape(-1, 1))
        self.dWnh = grad_n * r @ h_prev_t.transpose()
        self.dbnh = np.squeeze(grad_n * r)

        grad_z = dz * self.z_act.backward()
        self.dWzx = grad_z @ x
        self.dbzx = np.squeeze(grad_z)
        self.dWzh = grad_z @ h_prev_t.transpose()
        self.dbzh = np.squeeze(grad_z)

        grad_r = dr * self.r_act.backward()
        self.dWrx = grad_r @ x
        self.dbrx = np.squeeze(grad_r)
        self.dWrh = grad_r @ h_prev_t.transpose()
        self.dbrh = np.squeeze(grad_r)

        dx = np.zeros((1, self.d))
        dx += grad_n.transpose() @ self.Wnx
        dx += grad_z.transpose() @ self.Wzx
        dx += grad_r.transpose() @ self.Wrx

        dh_prev_t = np.zeros((1, self.h))
        dh_prev_t += (delta * z).transpose()
        dh_prev_t += (grad_n * r).transpose() @ self.Wnh
        dh_prev_t += grad_z.transpose() @ self.Wzh
        dh_prev_t += grad_r.transpose() @ self.Wrh

      


        

        assert dx.shape == (1, self.d)
        assert dh_prev_t.shape == (1, self.h)

        return dx, dh_prev_t

        

       
        
