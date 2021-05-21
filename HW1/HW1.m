%% 24677 LCS HW- 01
%Q3 Response of the system
M = 300; %kg
K = 20000; %N/m
B = 1000; %Ns/m
Stepsize = 0.15; %m
t = 0:0.0001:0.01;
T = 0:0.0001:5;
y = 0.08* sin(100*pi*t)+abs(0.08* sin(100*pi*t));
dl = length(T) - length(y);
tmp = zeros(1,dl);
y = [y tmp];
system = tf([B,K],[M,B,K]);
output_step = step(Stepsize * system);
plot(output_step);
% lsim(system,y,T);
%Q6 Linearizing 
% syms x1 x2 x3 x4 u1 u2 m1 m2 i1 i2 l1 g d
% 
% f_x_u = [x2;
%         (u2+m2*x1*x4^2-m2*g*sin(x3))/m2;
%          x4;
%         (u1-2*m2*x1*x4*x2-(m1*l1+m2*x1)*g*cos(x3))/(m1*l1^2+i1+i2+m2*x1^2)];
% output1 = jacobian(f_x_u,[x1,x2,x3,x4]);
% output2 = jacobian(f_x_u,[u1,u2]);
% new_output1 = subs(output1,[x1,x2,x3,x4,u1,u2],[d,0,0,0,0,0])
% new_output2 = subs(output2,[x1,x2,x3,x4,u1,u2],[d,0,0,0,0,0])
