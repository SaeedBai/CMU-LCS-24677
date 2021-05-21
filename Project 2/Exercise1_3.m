%%Consider the discrete time system x_k+1 = A * xk + B * uk. Design a state
%%feedback control matrix K such that the closed loop system has all poles
%%at 0
A = [1 1 -2
    0 1 1
    0 0 1];
B = [1 0 1]';
R = rank(ctrb(A,B));
Poles = [0 0 0];
K = acker(A,B,Poles);
disp('Feedback control matrix K is');
disp(K)
