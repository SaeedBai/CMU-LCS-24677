function eqn = LCS_P3_1_1_odefcn(t,p)
    


  eqn = [10*p(2)+10*p(3)+4*p(2)*p(3)-5;
            7*p(2)-p(1)+10*p(4)+4*p(2)*p(4);
            7*p(3)-p(1)+10*p(4)+4*p(3)*p(4);
            14*p(4)+4*p(4)^2             ];


end

