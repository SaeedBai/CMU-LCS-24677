%% Project 1 Excercise 1: Model Linearization
% Initializations 
m = 4500; %mass of vehicle
f = 0.028; % rolling resistance coefficient
g = 9.81;
Ca = 20000; % Cornering stiffness of each tire
lf = 1.01; % length from front tire to the center of mass
lr = 3.32; % length from rear tire to the center of mass
Iz = 29526.2; % Yaw intertia
delT = 0.032; 
syms F_input x y wheel_angle psi x_dot y_dot psi_dot X Y s
% Recreating equations in section 2.4
x_ddot = psi_dot * y_dot + 1 / m * (F_input - f*m*g);
y_ddot = -psi_dot * x_dot + 2 * Ca / m * (cos(wheel_angle)*(wheel_angle ...
    - (y_dot + lf*psi_dot))/x_dot) - (y_dot-lr*psi_dot)/x_dot;
psi_dot = psi_dot;
psi_ddot = (2*lf*Ca/Iz)*(wheel_angle- (y_dot+lf*psi_dot)/x_dot)-(2*lr*Ca) ...
    /Iz*(-(y_dot-lr*psi_dot)/x_dot);
X_dot = x_dot*cos(psi)-y_dot*sin(psi);
Y_dot = x_dot*sin(psi)+y_dot*cos(psi);
% Nonlinear state equations
Xcross_dot = [x_ddot;y_ddot;psi_dot;psi_ddot;X_dot;Y_dot];
u = [1 / m * F_input;2 * Ca / m * (cos(wheel_angle)*(wheel_angle) - (y_dot + ...
    lf*psi_dot)/x_dot);0;(2*lf*Ca/Iz)*(wheel_angle);0;0];
%% Problem 1
A = jacobian(Xcross_dot,[x_dot;y_dot;psi;psi_dot;X;Y])
B = jacobian(u,[wheel_angle;F_input])
C = [1 0 0 0 0 0;
     0 0 1 0 0 0];
D = [0 0;0 0];
%% Problem 2
x_dot = 6;
y_dot = 0;
psi_dot = y_dot;
psi = y_dot;
wheel_angle = 0;
A_new = subs(A)
B_new = subs(B)
%% Problem 3
G_s = C *inv((s*eye(length(A_new))-A_new))* B_new + D;
G_s = vpa(G_s,2)