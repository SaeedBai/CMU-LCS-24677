# Fill in the respective functions to implement the controller

# Import libraries
import numpy as np
from base_controller import BaseController
from scipy import signal, linalg
from util import *

import math

# CustomController class (inherits from BaseController)
class CustomController(BaseController):

    def __init__(self, trajectory):

        super().__init__(trajectory)

        # Define constants
        # These can be ignored in P1
        self.lr = 1.39  # length from rear tire to COM
        self.lf = 1.55  # length from rear tire to COM
        self.Ca = 20000 # cornering stiffness of each tire
        self.Iz = 25854 # yaw inertia
        self.m = 1888.6 # mass kg
        self.g = 9.81

        # Testing turn and straight
        self.count_turn = 0
        self.count = 0

        # Add additional member variables according to your need here.

        # Limits
        self.F_max = 16000.0
        self.F_min = 0.0
        self.delta_min = -math.pi/6
        self.delta_max = math.pi/6

        # PID params
        self.kp_x = 50000.0
        self.ki_x = 1
        self.kd_x = 1
        self.kp_psi = 100.0
        self.ki_psi = 0.0
        self.kd_psi = 0.0

        self.index_add = 90     # look ahead
        self.speed_scale = 3
        self.turn_scale = 2.525

        # Starting
        self.sum_error_x = 0.0
        self.error_x_old = 0.0
        self.sum_error_psi = 0.0
        self.sum_error_psi_turn = 0.0
        self.error_psi_old = 0.0


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
        if next_index > len(trajectory)-1: # last element of the loop
            next_index = -1
        if next_next_index > len(trajectory)-1: # last element of the loop
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
        if abs(psi_next_ref-psi_ref) < 0.4:
            Xdot_ref = (X_next_ref - X) / (delT * self.index_add)
            Ydot_ref = (Y_next_ref - Y) / (delT * self.index_add)
            xdot_ref,_ = self.glo2loc(Xdot_ref, Ydot_ref, psi)
            psi_ref = math.atan2((Y_next_ref - Y), (X_next_ref - X))
            self.speed_scale = 3.0

        else:
            Xdot_ref = (X_next_next_ref - X) / (delT * self.turn_scale * self.index_add)
            Ydot_ref = (Y_next_next_ref - Y) / (delT * self.turn_scale * self.index_add)
            xdot_ref, _ = self.glo2loc(Xdot_ref, Ydot_ref, psi)
            psi_ref = math.atan2((Y_next_next_ref - Y), (X_next_next_ref - X))
            self.speed_scale = 1

        psi_ref = (psi_ref + math.pi * 2) % (2 * math.pi)
        error_psi = psi_ref - psi

        ##############################################################
        # ---------------|Lateral Controller|-------------------------

        self.sum_error_psi += error_psi * delT
        delta = self.kp_psi * error_psi + \
        self.ki_psi * self.sum_error_psi + \
        self.kd_psi * (error_psi - self.error_psi_old) / delT
        delta = clamp(delta, self.delta_min, self.delta_max)
        self.error_psi_old = error_psi

        ###################################################################
        # ---------------|Longitudinal Controller|-------------------------

        #if abs(error_psi) < 0.05:
        error_x = xdot_ref * self.speed_scale - xdot
        self.sum_error_x += error_x * delT
        F = self.kp_x * error_x + \
        self.ki_x * self.sum_error_x + \
        self.kd_x * (error_x - self.error_x_old)/delT
        F = clamp(F, self.F_min, self.F_max)
            #print("xdot: ", xdot)
        self.error_x_old = error_x
        # print("Force: ", F)

        # Return all states and calculated control inputs (F, delta)
        return X, Y, xdot, ydot, psi, psidot, F, delta
