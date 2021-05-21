# Fill in the respective function to implement the LQR/EKF SLAM controller

# Import libraries
import numpy as np
from base_controller import BaseController
from scipy import signal, linalg
from scipy.spatial.transform import Rotation
from util import *
from ekf_slam import EKF_SLAM
import math as math
# CustomController class (inherits from BaseController)
class CustomController(BaseController):

    def __init__(self, trajectory):

        super().__init__(trajectory)

        # Define constants
        # These can be ignored in P1
        self.lr = 3.32
        self.lf = 1.01
        self.Ca = 20000
        self.Iz = 29526.2
        self.m = 4500
        self.g = 9.81
        
        self.counter = 0
        np.random.seed(99)

        # Add additional member variables according to your need here.
        self.F_max = 16000.0
        self.F_min = 0.0
        self.delta_min = -math.pi / 6
        self.delta_max = math.pi / 6
        self.Acc_max = self.F_max / self.m
        self.Acc_min = self.F_min / self.m
        # PID params
        self.kp_x = 4000
        self.ki_x = 3
        self.kd_x = 10
        self.kp_psi = 1
        self.ki_psi = 0
        self.kd_psi = 0
        # look ahead
        self.index_add = 90
        self.speed_scale = 3
        self.turn_scale = 2.525

        self.sum_error_x = 0.0
        self.error_x_old = 0.0
        self.error_1_old = 0.0
        self.error_2_old = 0.0
        self.sum_error_psi = 0.0
        self.sum_error_psi_turn = 0.0
        self.error_psi_old = 0.0

    def getStates(self, timestep, use_slam=False):

        delT, X, Y, xdot, ydot, psi, psidot = super().getStates(timestep)

        # Initialize the EKF SLAM estimation
        if self.counter == 0:
            # Load the map
            minX, maxX, minY, maxY = -120., 450., -350., 50.
            map_x = np.linspace(minX, maxX, 7)
            map_y = np.linspace(minY, maxY, 7)
            map_X, map_Y = np.meshgrid(map_x, map_y)
            map_X = map_X.reshape(-1,1)
            map_Y = map_Y.reshape(-1,1)
            self.map = np.hstack((map_X, map_Y)).reshape((-1))
            
            # Parameters for EKF SLAM
            self.n = int(len(self.map)/2)             
            X_est = X + 0.5
            Y_est = Y - 0.5
            psi_est = psi - 0.02
            mu_est = np.zeros(3+2*self.n)
            mu_est[0:3] = np.array([X_est, Y_est, psi_est])
            mu_est[3:] = np.array(self.map)
            init_P = 1*np.eye(3+2*self.n)
            W = np.zeros((3+2*self.n, 3+2*self.n))
            W[0:3, 0:3] = delT**2 * 0.1 * np.eye(3)
            V = 0.1*np.eye(2*self.n)
            V[self.n:, self.n:] = 0.01*np.eye(self.n)
            # V[self.n:] = 0.01
            print(V)
            
            # Create a SLAM
            self.slam = EKF_SLAM(mu_est, init_P, delT, W, V, self.n)
            self.counter += 1
        else:
            mu = np.zeros(3+2*self.n)
            mu[0:3] = np.array([X, 
                                Y, 
                                psi])
            mu[3:] = self.map
            y = self._compute_measurements(X, Y, psi)
            mu_est, _ = self.slam.predict_and_correct(y, self.previous_u)

        self.previous_u = np.array([xdot, ydot, psidot])

        print("True           X, Y, psi:", X, Y, psi)
        print("Estimated X, Y, psi:", mu_est[0], mu_est[1], mu_est[2])
        print("-------------------------------------------------------")
        
        if use_slam == True:
            return delT, mu_est[0], mu_est[1], xdot, ydot, mu_est[2], psidot
        else:
            return delT, X, Y, xdot, ydot, psi, psidot

    def _compute_measurements(self, X, Y, psi):
        x = np.zeros(3+2*self.n)
        x[0:3] = np.array([X, Y, psi])
        x[3:] = self.map
        
        p = x[0:2]
        psi = x[2]
        m = x[3:].reshape((-1,2))

        y = np.zeros(2*self.n)

        for i in range(self.n):
            y[i] = np.linalg.norm(m[i, :] - p)
            y[self.n+i] = wrapToPi(np.arctan2(m[i,1]-p[1], m[i,0]-p[0]) - psi)
            
        y = y + np.random.multivariate_normal(np.zeros(2*self.n), self.slam.V)
        return y

    def glo2loc(self, X, Y, psi):
        # convert (X, Y) from global frame to inertial frame
        psi_out = wrapToPi(psi)
        XY_glo = np.array([[X], [Y]])
        convert_mat = np.array([[math.cos(psi_out), -math.sin(psi_out)],
                                [math.sin(psi_out), math.cos(psi_out)]])
        convert_mat = np.linalg.inv(convert_mat)

        xy_loc = np.matmul(convert_mat, XY_glo)
        return xy_loc[0,0], xy_loc[1,0]


    def update(self, timestep, driver):

        trajectory = self.trajectory

        lr = self.lr
        lf = self.lf
        Ca = self.Ca
        Iz = self.Iz
        m = self.m
        g = self.g

        # Fetch the states from the newly defined getStates method
        delT, X, Y, xdot, ydot, psi, psidot = self.getStates(timestep, use_slam=True)


        # Look ahead
        _, index = closestNode(X, Y, trajectory)
        next_index = index + self.index_add
        next_next_index = index + self.turn_scale * self.index_add
        if next_index > len(trajectory) - 1:  # last element of the loop
            next_index = -1
        if next_next_index > len(trajectory) - 1:  # last element of the loop
            next_next_index = -1
        X_next_ref = trajectory[next_index][0]
        Y_next_ref = trajectory[next_index][1]
        X_next_next_ref = trajectory[round(next_next_index)][0]
        Y_next_next_ref = trajectory[round(next_next_index)][1]
        psi_ref = math.atan2((Y_next_ref - Y), (X_next_ref - X))
        psi_ref = (psi_ref + math.pi * 2) % (2 * math.pi)
        psi = (psi + math.pi * 2) % (2 * math.pi)
        psi_next_ref = math.atan2((Y_next_next_ref - Y), (X_next_next_ref - X))
        psi_next_ref = (psi_next_ref + math.pi * 2) % (2 * math.pi)

        # Check for turn or straight
        # Straight  value around 0.01
        print(abs(psi_next_ref - psi_ref))
        if abs(psi_next_ref - psi_ref) < 0.4:
            Xdot_ref = (X_next_ref - X) / (delT * self.index_add)
            Ydot_ref = (Y_next_ref - Y) / (delT * self.index_add)
            xdot_ref, _ = self.glo2loc(Xdot_ref, Ydot_ref, psi)
            psi_ref = math.atan2((Y_next_ref - Y), (X_next_ref - X))
            self.speed_scale = 3.0

        else:
            Xdot_ref = (X_next_next_ref - X) / (delT * self.turn_scale * self.index_add)
            Ydot_ref = (Y_next_next_ref - Y) / (delT * self.turn_scale * self.index_add)
            xdot_ref, _ = self.glo2loc(Xdot_ref, Ydot_ref, psi)
            psi_ref = math.atan2((Y_next_next_ref - Y), (X_next_next_ref - X))
            self.speed_scale = 0.5

        psi_ref = (psi_ref + math.pi * 2) % (2 * math.pi)
        error_psi = psi_ref - psi

        # ---------------|Lateral Controller|-------------------------
        self.sum_error_psi += error_psi * delT
        delta = self.kp_psi * error_psi + \
                self.ki_psi * self.sum_error_psi + \
                self.kd_psi * (error_psi - self.error_psi_old) / delT
        delta = clamp(delta, self.delta_min, self.delta_max)
        self.error_psi_old = error_psi
        # ---------------|Longitudinal Controller|-------------------------
        error_x = xdot_ref * self.speed_scale - xdot
        # error_x = abs(error_x)
        self.sum_error_x += error_x * delT
        F = self.kp_x * error_x + \
            self.ki_x * self.sum_error_x + \
            self.kd_x * (error_x - self.error_x_old) / delT

        F = clamp(F, self.F_min, self.F_max)
        self.error_x_old = error_x


        # driver.setBrakeIntensity(clamp(someValue, 0, 1))

        # Return all states and calculated control inputs (F, delta)
        return X, Y, xdot, ydot, psi, psidot, F, delta