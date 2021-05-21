
import numpy as np
import math
class EKF_SLAM():
    def __init__(self, init_mu, init_P, dt, W, V, n):
        """Initialize EKF SLAM

        Create and initialize an EKF SLAM to estimate the robot's pose and
        the location of map features

        Args:
            init_mu: A numpy array of size (3+2*n, ). Initial guess of the mean
            of state.
            init_P: A numpy array of size (3+2*n, 3+2*n). Initial guess of
            the covariance of state.
            dt: A double. The time step.
            W: A numpy array of size (3+2*n, 3+2*n). Process noise
            V: A numpy array of size (2*n, 2*n). Observation noise
            n: A int. Number of map features


        Returns:
            An EKF SLAM object.
        """
        self.mu = init_mu  # initial guess of state mean
        self.P = init_P  # initial guess of state covariance
        self.dt = dt  # time step
        self.W = W  # process noise
        self.V = V  # observation noise
        self.n = n  # number of map features


    def _f(self, x, u):
        """Non-linear dynamic function.

        Compute the state at next time step according to the nonlinear dynamics f.

        Args:
            x: A numpy array of size (3+2*n, ). State at current time step.
            u: A numpy array of size (3, ). The control input [\dot{x}, \dot{y}, \dot{\psi}]

        Returns:
            x_next: A numpy array of size (3+2*n, ). The state at next time step
        """
        xdot = u[0]
        ydot = u[1]
        psidot = u[2]

        X = x[0]
        Y = x[1]
        Psii = x[2]
        psi = self._wrap_to_pi(x[2])

        # numerical update
        x_next_x = X + (xdot * math.cos(psi) - ydot * math.sin(psi)) * self.dt
        x_next_y = Y + (xdot * math.sin(psi) + ydot * math.cos(psi)) * self.dt
        x_next_psi = self._wrap_to_pi(Psii + psidot * self.dt)
        x_next = np.array([x_next_x,x_next_y,x_next_psi])

        for i in range(3, 3 + 2 * self.n):
            x_next = np.append(x_next,np.array([x[i]]))
        return x_next

    def _h(self, x):
        """Non-linear measurement function.

        Compute the sensor measurement according to the nonlinear function h.

        Args:
            x: A numpy array of size (3+2*n, ). State at current time step.

        Returns:
            y: A numpy array of size (2*n, ). The sensor measurement.
        """
        X = x[0]
        Y = x[1]
        Psii = x[2]

        y = np.zeros(2 * self.n)

        for i in range(self.n):
            x_n = x[2 * i + 3]
            y_n = x[2 * i + 4]
            y[i] = math.sqrt((x_n - X)**2 + (y_n - Y)**2)
        for cc in range(self.n,2 * self.n):
            i = cc - self.n
            x_n = x[2 * i + 3]
            y_n = x[2 * i + 4]
            y[cc] = self._wrap_to_pi(math.atan2(y_n - Y, x_n - X) - Psii)
        return y


    def _compute_F(self, u):
        """Compute Jacobian of f

        You will use self.mu in this function.

        Args:
            u: A numpy array of size (3, ). The control input [\dot{x}, \dot{y}, \dot{\psi}]

        Returns:
            F: A numpy array of size (3+2*n, 3+2*n). The jacobian of f evaluated at x_k.
        """
        F = np.zeros((3 + 2 * self.n, 3+ 2 * self.n))
        psi = self._wrap_to_pi(self.mu[2])
        F = np.eye(F.shape[0])

        F[0, 2] = -self.dt * (u[0] * math.sin(psi) + u[1] * math.cos(psi))
        F[1, 2] =  self.dt * (u[0] * math.cos(psi) - u[1] * math.sin(psi))

        return F


    def _compute_H(self):
        """Compute Jacobian of h

        You will use self.mu in this function.

        Args:

        Returns:
            H: A numpy array of size (2*n, 3+2*n). The jacobian of h evaluated at x_k.
        """
        H = np.zeros((2 * self.n, 3 + 2 * self.n))
        X = self.mu[0]
        Y = self.mu[1]

        # distance sensor
        # first n rows
        for i in range(self.n):
            x_n = self.mu[2 * i + 3]
            y_n = self.mu[2 * i + 4]
            d_n = math.sqrt((X - x_n) ** 2 + (Y - y_n) ** 2)
            # first column
            H[i,0] = (X - x_n) / d_n
            # second column
            H[i,1] = (Y - y_n) / d_n
            # third column
            H[i,2] = 0
            # columns with x
            H[i,2*i+3] = (x_n - X) / d_n
            # columns with y
            H[i,2*i+4] = (y_n - Y) / d_n

        # bearing sensor
        # rest n rows
        for cc in range(self.n,2*self.n):
            # get correct index
            i = cc - self.n

            x_n = self.mu[2 * i + 3]
            y_n = self.mu[2 * i + 4]
            d_n = math.sqrt((X - x_n) ** 2 + (Y - y_n) ** 2)
            # first column
            H[cc,0] = (y_n - Y) / d_n ** 2
            # second column
            H[cc,1] = (x_n - X) / d_n ** 2
            # third column
            H[cc,2] = -1
            # columns with x
            H[cc, 2 * i + 4] = (x_n - X) / d_n ** 2
            # columns with y
            H[cc, 2 * i + 3] = (Y - y_n) / d_n ** 2
        return H


    def predict_and_correct(self, y, u):
        """Predice and correct step of EKF

        You will use self.mu in this function. You must update self.mu in this function.

        Args:
            y: A numpy array of size (2*n, ). The measurements according to the project description.
            u: A numpy array of size (3, ). The control input [\dot{x}, \dot{y}, \dot{\psi}]

        Returns:
            self.mu: A numpy array of size (3+2*n, ). The corrected state estimation
            self.P: A numpy array of size (3+2*n, 3+2*n). The corrected state covariance
        """

        # compute F and H matrix
        F = self._compute_F(u)
        H = self._compute_H()
        # last_mu = self.mu
        #***************** Predict step *****************#
        # predict the state
        xbar = self._f(self.mu,u)
        # predict the error covariance
        pbar = F @ self.P @ np.transpose(F) + self.W
        #***************** Correct step *****************#
        # compute the Kalman gain
        L = pbar @ np.transpose(H) @ np.linalg.inv(H @ pbar @ np.transpose(H) + self.V)
        # update estimation with new measurement
        self.mu = xbar + L @ self._wrap_to_pi(y - self._h(xbar))
        self.mu[2] = self._wrap_to_pi(self.mu[2])
        # update the error covariance
        self.P = (np.eye(pbar.shape[0]) - L @ H) @ pbar

        return self.mu, self.P


    def _wrap_to_pi(self, angle):
        angle = angle - 2*np.pi*np.floor((angle+np.pi )/(2*np.pi))
        return angle


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    m = np.array([[0.,  0.],
                  [0.,  20.],
                  [20., 0.],
                  [20., 20.],
                  [0,  -20],
                  [-20, 0],
                  [-20, -20],
                  [-50, -50]]).reshape(-1)

    dt = 0.01
    T = np.arange(0, 20, dt)
    n = int(len(m)/2)
    W = np.zeros((3+2*n, 3+2*n))
    W[0:3, 0:3] = dt**2 * 1 * np.eye(3)
    V = 0.1*np.eye(2*n)
    V[n:,n:] = 0.01*np.eye(n)

    # EKF estimation
    mu_ekf = np.zeros((3+2*n, len(T)))
    mu_ekf[0:3,0] = np.array([2.2, 1.8, 0.])
    # mu_ekf[3:,0] = m + 0.1
    mu_ekf[3:,0] = m + np.random.multivariate_normal(np.zeros(2*n), 0.5*np.eye(2*n))
    init_P = 1*np.eye(3+2*n)

    # initialize EKF SLAM
    slam = EKF_SLAM(mu_ekf[:,0], init_P, dt, W, V, n)

    # real state
    mu = np.zeros((3+2*n, len(T)))
    mu[0:3,0] = np.array([2, 2, 0.])
    mu[3:,0] = m

    y_hist = np.zeros((2*n, len(T)))
    for i, t in enumerate(T):
        if i > 0:
            # real dynamics
            u = [-5, 2*np.sin(t*0.5), 1*np.sin(t*3)]
            # u = [0.5, 0.5*np.sin(t*0.5), 0]
            # u = [0.5, 0.5, 0]
            mu[:,i] = slam._f(mu[:,i-1], u) + \
                np.random.multivariate_normal(np.zeros(3+2*n), W)

            # measurements
            y = slam._h(mu[:,i]) + np.random.multivariate_normal(np.zeros(2*n), V)
            y_hist[:,i] = (y-slam._h(slam.mu))
            # apply EKF SLAM
            mu_est, _ = slam.predict_and_correct(y, u)
            mu_ekf[:,i] = mu_est


    plt.figure(1, figsize=(10,6))
    ax1 = plt.subplot(121, aspect='equal')
    ax1.plot(mu[0,:], mu[1,:], 'b')
    ax1.plot(mu_ekf[0,:], mu_ekf[1,:], 'r--')
    mf = m.reshape((-1,2))
    ax1.scatter(mf[:,0], mf[:,1])
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')

    ax2 = plt.subplot(322)
    ax2.plot(T, mu[0,:], 'b')
    ax2.plot(T, mu_ekf[0,:], 'r--')
    ax2.set_xlabel('t')
    ax2.set_ylabel('X')

    ax3 = plt.subplot(324)
    ax3.plot(T, mu[1,:], 'b')
    ax3.plot(T, mu_ekf[1,:], 'r--')
    ax3.set_xlabel('t')
    ax3.set_ylabel('Y')

    ax4 = plt.subplot(326)
    ax4.plot(T, mu[2,:], 'b')
    ax4.plot(T, mu_ekf[2,:], 'r--')
    ax4.set_xlabel('t')
    ax4.set_ylabel('psi')

    plt.figure(2)
    ax1 = plt.subplot(211)
    ax1.plot(T, y_hist[0:n, :].T)
    ax2 = plt.subplot(212)
    ax2.plot(T, y_hist[n:, :].T)

    plt.show()

