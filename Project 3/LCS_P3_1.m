% Excercise 1.1
% INITIALIZATIONS
syms x1 x2
A = [   0   1
        -10 -7   ];
B = [   0
        1   ];
C = eye(2);
D = 0;
x_0 = [1   1]';
T = 0.05;
R = 0.25;
Q = diag([5 1]);

S = eye(2) * 20;
K = [];
% Setting time range
convert = reshape(S,[4,1]);
c = convert;
[t,p] = ode45(@(t,p) LCS_P3_1_1_odefcn(t,p),[5 0],convert);
[~,n] = min(abs(T-t));
for i = length(p)
    holder = p(i,:);
    m = reshape(holder,[2,2]);
    K(i,:) = inv(R)*B'*m;
end
[tt,x] = ode45(@(tt,x) LCS_P3_1_1_odefcn2(A,B,n,x,K),[0 5],x_0);
plot(tt,x)

% Excercise 1.2
% INITIALIZATIONS
clear x t

N = 1000;
Q = diag([5 1]);
S = eye(2) * 20;
R = 0.25;
Ad = expm(A*T);
Bd = inv(A)*(expm(A*T)-eye(2))*B;
Kd = [];
x = [];

% Setup Initial
x(:,1) = [1 1]';

for c = N:-1:1
    Kd(c,:) = -inv(Bd'*S*Bd+R)*Bd'*S*Ad;
    S = (Ad+Bd*Kd(c,:))'*S*(Ad+Bd*Kd(c,:))+Q+Kd(c,:)'*R*Kd(c,:);
end

for i = 1:N
    x(:,i+1) = Ad*x(:,i)+Bd*Kd(i,:)*x(:,i);
end

figure
plot(0:N,x(:,1:end));
% Excercise 1.3
Q = diag([5 1]);
R = 0.5;
% State space convertion
plant = ss(A,B,C,D);
K = lqr(A,B,Q,R);

CL = feedback(plant*K,-eye(2));
initial(CL,x_0);

