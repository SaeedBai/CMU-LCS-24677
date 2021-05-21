function eqn = LCS_P3_1_1_(A,B,tl,x,K)
 K_i = K(tl,:);
 eqn = A*x-B*K_i*x;
end

