%% Project 2 Excercise 2
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

A = [0                  1                   0                   0
     0             -4*Ca/(m*x_dot)        4*Ca/m        -2*Ca*(lf-lr)/(m*x_dot)
     0                  0                   0                   1
     0      -2*Ca*(lf-lr)/(Iz*x_dot)  2*Ca*(lf-lr)/Iz   -2*Ca*(lf^2+lr^2)/(Iz*x_dot)];
     
B = [0             0
      2*Ca/m        0
      0             0
      2*Ca*lf/Iz    0];

% Question 1 
[Rank1,~,~] = GetRank(2,A,B);      %v = 2m/s
[Rank2,~,~] = GetRank(5,A,B);      %v = 5m/s
[Rank3,~,~] = GetRank(8,A,B);      %v = 8m/s

% Question 2
v = 1:1:40;
sigmaratio = [];
poles = [];
% for i = 1:40
%     [~,ControlMatrix,A_sub] = GetRank(i,A,B);
%     [~,SingularVal,~] = svd(ControlMatrix);
%     Sigma1 = SingularVal(1,1);
%     Sigman = SingularVal(4,4);
%     sigmaratio = [sigmaratio log10(Sigma1 / Sigman)]; 
%     realpart = real(eig(A_sub));
%     poles = [poles realpart];
% end
% figure(1)
% plot(v,sigmaratio);
% figure(2)
% subplot(2,2,1)
% plot(v,poles(1,:));
% subplot(2,2,2)
% plot(v,poles(2,:));
% subplot(2,2,3)
% plot(v,poles(3,:));
% subplot(2,2,4)
% plot(v,poles(4,:));
% fprintf("The system are both controllable and stable\n");

% Question 3
% only heading error
C = [1 0 0 0
     0 0 1 0];
[Rank21,~,~] = GetRank2(2,A,C);      %v = 2m/s
[Rank22,~,~] = GetRank2(5,A,C);      %v = 5m/s
[Rank23,~,~] = GetRank2(8,A,C);      %v = 8m/s

% Question 4
B_SingleInput = [0             
               2*Ca/m        
                 0             
             2*Ca*lf/Iz];
p1 = -2.5;
p2 = -5.3;
p3 = -0.5   + i;
p4 = -0.5   - 1i;
p = [p1 p2 p3 p4];
[~,~,A_sub] = GetRank(15,A,B_SingleInput);
[K,prec,message] = place(A_sub,B_SingleInput,p);

K = K