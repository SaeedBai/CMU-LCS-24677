% Project 4 exercise 1.2
clear all; clc

% INITIALIZATIONS
T = 0.002;
T_end = 1;
time_step = length(0:0.002:1);
x_est = zeros(2,time_step);
y_est = zeros(1,time_step);
y     = zeros(1,time_step);
x_bar = zeros(2,time_step);
x_hat = zeros(2,time_step);
p     = zeros(2,2,time_step);
p_hat = zeros(2,2,time_step);

anv   = 2.5;
env   = 2 * 10^-4;
u     = 0.1;

% Random Initialization
x_0 = [1    1]';

p_0 = [1    0
       0    1];
% State space
A = [1  T
     0  1];
B = [T^2 / 2    T]';
C = [1  0];
D = 0;

% Noise
wk = eye(2).* B + eye(2).*anv;
vk = env^2 / 12;

% First step

x_est(:,1) = x_0;
y_est(:,1) = C * x_est(:,1) + vk;
x_bar(:,1) = [0 0]';
x_hat(:,1) = x_0;
p(:,:,1)   = p_0;
p_hat(:,:,1) = p_0;

% Initial observer
L = [1  1]';
for i = 1 : time_step-1
    % get noise
    w  = normrnd(0,sqrt(anv));
    v  = normrnd(0,sqrt(env));
    % next step
    x_est(:,i+1) = A * x_est(:,i) + B * w;
    y_est(:,i+1) = C * x_est(:,i) + D + v;
end

for i = 1 : time_step-1
    y(i) = y_est(i);
    x_hat(:,i) = x_bar(:,i) + L * (y(i) - C * x_bar(:,i));
    p(:,:,i) = (eye(length(L)) - L * C) * squeeze(p_hat(:,:,i));
    x_bar(:,i+1) = A * x_hat(:,i) + B * u;
    p_hat(:,:,i+1) = A * squeeze(p(:,:,i)) * A' + wk;
    L = squeeze(p_hat(:,:,i+1)) * C' * inv(C * squeeze(p_hat(:,:,i+1)) * C' + vk);
end
% Get last prediction
x_hat(:,end) = x_bar(:,end) + L * (y_est(end) - C * x_bar(:,end));

% PLOT
figure;
plot(1:time_step,x_est(1,:),'b',1:time_step,x_hat(1,:),'r',1:time_step,y_est(1,:),'k');
legend('True position state','Position estimated state','position measurement')