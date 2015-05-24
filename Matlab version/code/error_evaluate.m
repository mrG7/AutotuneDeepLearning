%Counting error S=(w-w_0)A(w-w_0)'+E_d
function [error] = error_evaluate(net, options, xcontrol, tcontrol)
     [A_cov,w_0]=A(net,options);
    w=netpack_nobias(net);
    y = mlpfwd(net, xcontrol);
    error = (w-w_0)'*A_cov*(w-w_0);    
    for r = 1:net.nout
        error = error - tcontrol(:,r)'*log(y(:,r));  %Are I right with this formula?           
    end
    