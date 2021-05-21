function [Rank,M,matrix1_sub] = GetRank(velocity,matrix1,matrix2)
%This function calculates rank of given matrix with velocity substitutes in
x_dot = velocity;
matrix1_sub = subs(matrix1);
matrix2_sub = subs(matrix2);
AB = matrix1_sub * matrix2_sub;
A2B = matrix1_sub^2 * matrix2_sub;
A3B = matrix1_sub^3 * matrix2_sub;
M = horzcat(matrix2_sub,AB,A2B,A3B);
Rank = rank(M);
end
