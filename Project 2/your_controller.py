# Fill in the respective functions to implement the controller

# Import libraries
import numpy as np
import math
from base_controller import BaseController
from scipy import signal, linalg
from util import *

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

        # Limits
        self.F_max = 16000.0
        self.F_min = 0.0
        self.delta_min = -math.pi / 6
        self.delta_max = math.pi / 6
        self.Acc_max = self.F_max/self.m
        self.Acc_min = self.F_min/self.m
        # PID params
        self.kp_x = 4000
        self.ki_x = 30
        self.kd_x = 10
        # look ahead
        self.index_add = 90
        self.speed_scale = 3
        self.turn_scale = 2.525

        self.sum_error_x = 0.0
        self.error_x_old = 0.0
        self.error_1_old = 0.0
        self.error_2_old = 0.0
        # Designed poles
        self.poles = np.array([-15,-25,-1.5,-0.001])
    def glo2loc(self, X, Y, psi):
        # convert (X, Y) from global frame to inertial frame
        psi_out = wrapToPi(psi)
        XY_glo = np.array([[X],[Y]])
        convert_mat = np.array([[math.cos(psi_out), -math.sin(psi_out)],
                                [math.sin(psi_out), math.cos(psi_out)]])
        convert_mat = np.linalg.inv(convert_mat)

        xy_loc = np.matmul(convert_mat, XY_glo)
        return xy_loc[0][0], xy_loc[1][0]
    def update(self, timestep):

        trajectory = self.trajectory

        lr = self.lr
        lf = self.lf
        Ca = self.Ca
        Iz = self.Iz
        m = self.m
        g = self.g

        # Fetch the states from the BaseController method
        delT, X, Y, xdot, ydot, psi, psidot = super().getStates(timestep)

        # Design your controllers in the spaces below. 
        # Remember, your controllers will need to use the states
        # to calculate control inputs (F, delta). 

        # Look ahead
        _, index = closestNode(X, Y, trajectory)
        next_index = index + self.index_add
        next_next_index = index + self.turn_scale * self.index_add
        if next_index > len(trajectory) - 1:  # last element of the loop
            next_index = -1
        if next_next_index > len(trajectory) - 1:  # last element of the loop
            next_next_index = -1

        X_now = trajectory[index][0]
        Y_now = trajectory[index][1]
        X_next_ref = trajectory[next_index][0]
        Y_next_ref = trajectory[next_index][1]
        X_next_next_ref = trajectory[round(next_next_index)][0]
        Y_next_next_ref = trajectory[round(next_next_index)][1]


        psi_ref = math.atan2((Y_next_ref - Y), (X_next_ref - X))
        psi_next_ref = math.atan2((Y_next_next_ref - Y),(X_next_next_ref - X))
        if abs(wrapToPi(psi_next_ref - psi_ref)) < 0.33:
            Xdot_ref = (X_next_ref - X) / (delT * self.index_add)
            Ydot_ref = (Y_next_ref - Y) / (delT * self.index_add)
            xdot_ref, _ = self.glo2loc(Xdot_ref, Ydot_ref, psi)
            psi_ref = psi_ref
            error_1 = (Y_now - Y_next_ref) * math.cos(psi_ref) - (X_now - X_next_ref) * math.sin(psi_ref)
            self.speed_scale = 3.0
        else:
            Xdot_ref = (X_next_next_ref - X) / (delT * self.turn_scale * self.index_add)
            Ydot_ref = (Y_next_next_ref - Y) / (delT * self.turn_scale * self.index_add)
            xdot_ref, _ = self.glo2loc(Xdot_ref, Ydot_ref, psi)
            psi_ref = math.atan2((Y_next_next_ref - Y), (X_next_next_ref - X))
            error_1 = (Y_now - Y_next_ref) * math.cos(psi_ref) - (X_now - X_next_ref) * math.sin(psi_ref)
            self.speed_scale = 1

        error_psi = wrapToPi(psi - psi_ref)

        # ---------------|Lateral Controller|-------------------------
        A = np.array([
            [0,1,0,0],
            [0,-4 * Ca / (m * xdot),4 * Ca / m,-2 * Ca * (lf - lr) / (m * xdot)],
            [0,0,0,1],
            [0,-2 * Ca * (lf - lr)/(Iz * xdot),2 * Ca * (lf - lr) / Iz,
             -2 * Ca * (lf ** 2 + lr ** 2) / (Iz * xdot)]
            ])
        # B matrix ignoring the F
        B = np.array([
            [0],
            [2 * Ca / m],
            [0],
            [2 * Ca * lf / Iz]
        ])

        fsf = signal.place_poles(A, B, self.poles)
        K = fsf.gain_matrix

        # Build states matrix
        error_1 = error_1
        error_1dot = ydot + xdot * error_psi
        error_2 = error_psi
        error_2dot = psidot
        States = np.asarray([
            [error_1],[error_1dot],[error_2],[error_2dot]
        ])

        delta = float(np.matmul(-K,States))
        delta = clamp(delta, -self.delta_max, self.delta_max)

        # ---------------|Longitudinal Controller|-------------------------
        error_x = xdot_ref * self.speed_scale - xdot
        self.sum_error_x += error_x * delT
        F = self.kp_x * error_x + \
            self.ki_x * self.sum_error_x + \
            self.kd_x * (error_x - self.error_x_old) / delT
        F = clamp(F, self.F_min, self.F_max)
        self.error_x_old = error_x

        # Return all states and calculated control inputs (F, delta)
        return X, Y, xdot, ydot, psi, psidot, F, delta
