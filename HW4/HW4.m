%% HW 4
%Problem 2
% syms x y
% eqn1 = -106 + 3*sqrt(1789) == (-106 + 3*sqrt(1789))*x^2 + (-106 + 3*sqrt(1789)-x^2*(-106 + 3*sqrt(1789)))/(-106 + 3*sqrt(1789))*(-106 + 3*sqrt(1789));
% [a1] = solve(eqn1,x)

%Problem 4
%Initializations
a=0.1;
b=0.2;
u=1;
x_ini = [2;1];
t_b = 1;
%CT system
A = [-a 0;a -b];
B = [1;0];
C = [1 0;0 1];
D = 0;
CT = ss(A,B,C,D)
%DT system
AA = [exp(-0.1) 0;exp(-0.1)-exp(-0.2) exp(-0.2)];
BB = [-10*exp(-0.1) + 10;-10*exp(-0.1)+5*exp(-0.2)+5];
CC = C;
DD = 0;
DT = ss(AA,BB,CC,DD,t_b);
%Plotting
t = 0:1:10;  % 201 points
u = ones(size(t));
lsim(CT, u, t, x_ini)
legend('CT', 'DT')
hold on
lsim(DT, u, t, x_ini)

%Problem 5
clear all;
%Initialization
uu = 1;
Kd =10;
Kp =100;
Ki =1;
Kd = 1000;
tau =0.001;
%State space
A = [0 1;0 -1/tau];
B = [0;1];
C = [Ki/tau Kp/tau-Kd/tau^2];
D = Kd/tau;
sys = ss(A,B,C,D);
t = 0:0.0001:0.01;  
u = ones(size(t));
lsim(sys, u, t)
