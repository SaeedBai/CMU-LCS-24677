%% Consider the discrete time system x_k+1 = Ax + Bu, y  = Cx. Design an observer matrix L such that the ovserver has poles at -0.5+-0.5j
A = [-2 4
    -3 9];
B = [0;1];
C = [3 1];
Poles = [-0.5+0.5j,-0.5-0.5j];
Rank = rank(obsv(A,C));
L = place(A',C',Poles);
disp('Observer matrix L is');
disp(L)
