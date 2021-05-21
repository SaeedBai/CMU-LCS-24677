%% HW3 Problem 6
%Initialization
x0 = [1;0];
x10 = [0;0];
A = [1 0.01;0 1];
B = [0;0.01];
C = [1 0];
tmax = 10;
t = 0:1:tmax-1;
x = zeros(2,tmax);
xcl = zeros(2,tmax);
% a)
M = [A^9*B A^8*B A^7*B A^6*B A^5*B A^4*B A^3*B A^2*B A^1*B A^0*B];
a = x10-A^10*x0;
u = M'*inv(M*M')*a;
x(:,1) = x0;
for i = 2: tmax
    x(:,i) = A * x(:,i-1) + B*u(i-1);
end
% subplot(3,1,1);
% plot(t,x(1,:));
% subplot(3,1,2);
% plot(t,x(2,:));
% u = [u' zeros(1,tmax)];
% t = 0:1:19;
% subplot(3,1,3);
% plot(t,u);

% b)
k = -[10000 200]; %from calculation
xcl(:,1) = x0;
for i = 2: tmax
    ucl = k * xcl(:,i-1);
    xcl(:,i) = A * xcl(:,i-1) + B*ucl;
end
subplot(3,1,1);
plot(t,xcl(1,:));
subplot(3,1,2);
plot(t,xcl(2,:));
