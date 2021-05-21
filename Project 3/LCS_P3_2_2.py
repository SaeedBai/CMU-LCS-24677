import numpy as np
from scipy import signal, linalg
import math

if __name__ == '__main__':
        lr = 3.32
        lf = 1.01
        Ca = 20000
        Iz = 29526.2
        m = 4500
        g = 9.81
        delT = 0.032
        #Set velocity
        xdot = 20
        def diag(a, b, c, d):
            # get a 4x4 diagonal matrix
            diagonal_matrix = np.array([
                [a, 0, 0, 0],
                [0, b, 0, 0],
                [0, 0, c, 0],
                [0, 0, 0, d]
            ])
            return diagonal_matrix
        # State matrix
        A = np.array([
                [0, 1, 0, 0],
                [0, -4 * Ca / (m * xdot), 4 * Ca / m, -2 * Ca * (lf - lr) / (m * xdot)],
                [0, 0, 0, 1],
                [0, -2 * Ca * (lf - lr) / (Iz * xdot), 2 * Ca * (lf - lr) / Iz,
                 -2 * Ca * (lf ** 2 + lr ** 2) / (Iz * xdot)]
            ])

        # Input matrix
        B = np.array([
                [0],
                [2 * Ca / m],
                [0],
                [2 * Ca * lf / Iz]
            ])

        # Output matrix
        C = np.eye(4)

        # Feedthrough matrix
        D = np.zeros((4, 1))

        # Continuous system is not working
        # Discrete system
        lti = signal.StateSpace(A, B, C, D)
        sys = lti.to_discrete(delT)
        Anew = sys.A
        Bnew = sys.B
        #Setup Q
        Q = diag(0.0000001, 70, 300, 0.1)
        #Setup R
        R = 1500
        #Setup S
        S = diag(0.1, 0.1, 0.1, 0.1)
        #Setup steps
        N = 50
        #Setup initial K
        K = [0] * 4

        #Finite Horizon
        for i in range(N,1,-1):
            V = np.matrix(-1 * linalg.inv(R + np.transpose(Bnew) @ S @ Bnew) @ (np.transpose(Bnew) @ S @ Anew))
            K = np.vstack((K,V))
            S = np.matrix(np.transpose(Anew + Bnew @ V)) @ S @ np.matrix(Anew + Bnew @ V) + Q + \
                (np.transpose(V) * R) @ V
        K = K[-1]
        print(K)

        '''
        1.
        The function numpy.vstack requires initial value for stacking two different arrays, K initial was set to [0 0 0 0]
        since The first step won't lead to a huge difference to the system and The second K help the vehicle back on track
        2.
        Q and R were chose from LQR controller and were chosen as 'the best' performance Q and R
        3.
        N was set initially 20 and increased from 20 to 200 with a increment of 10. Any value over 200 will make the system
        slow to test due to gigantic test in each time step. N was chosen to be 50 as making the best performance
        '''