%Counting error E_d
function [error] = error_d(net, options, xcontrol, tcontrol)  
    y = mlpfwd(net, xcontrol);
    error = 0;  
    for r = 1:net.nout
     error = error - tcontrol(:,r)'*log(y(:,r));           
    end  