function [Rank,M,matrix1_sub] = GetRank2(velocity,matrix1,matrix2)
%This function calculates rank of given matrix with velocity substitutes in
x_dot = velocity;
matrix1_sub = subs(matrix1);
matrix2_sub = subs(matrix2);
CA =  matrix2_sub * matrix1_sub;
CA2 = matrix2_sub * matrix1_sub^2;
CA3 = matrix2_sub * matrix1_sub^3;
M = vertcat(matrix2_sub,CA,CA2,CA3);
Rank = rank(M);
end
