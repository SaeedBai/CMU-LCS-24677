%% 24-677 HW 4 

%% Part b 
A = [1 0.01;0 1];
B = [0;0.01];
C = [1 0];
D = 0;
ts = 0.01;
[b a] = ss2tf(A,B,C,D);
sys = tf(b,a,ts);

BD = pidTuner(H,'PID'); %Getting proper Kp Ki Kd
K_p = 10.6988;
K_i = 3.1334;
K_d = 9.1327;
CC  = pid(K_p,K_i,K_d,D,ts);
G  = CC * sys;
CLTF = feedback(G,1);
step(CLTF);

%% Part c
T = ts;
T_max = 3.5; %Time interval
T_t = 0: ts : 3.5;
SE = size(T_t);
x = zeros(2,length(T_t));
y = zeros(SE);
err = zeros(SE+1);
u_d = err;
u_i = err;
u_p = err;
u_add = err;
add_err = 0;
for i = 1 : length(T_t)
    x(:,i+1) = A * x(:,i) + B * u_add(i);
    y(i) = C * x(:,i);
    err(i+1) = 1 - y(i);
    add_err = add_err + err(i+1);
    u_d(i+1) = K_d / T * (err(i+1) - err(i));
    u_i(i+1) = K_i * T * add_err;
    u_p(i+1) = K_p * err(i);
    u_add(i+1) = u_d(i+1) + u_i(i+1) + u_p(i+1);
end
plot(T_t,y)

%% Part d
Td = 0.05;
T_d = 0:0.01:Td;
%x(1)=Ax(0)+Bu(0)
%x(2)=Ax(1)+Bu(1)=A(Ax(0)+Bu(0))+Bu(1)

