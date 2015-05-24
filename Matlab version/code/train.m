%this program trains the neural network using back-propagation algorithm
%for NCYCLES times with learning-rate parameter ETA

function net = train(net, xlearn, ylearn)

%Network training
options = zeros(1,18);
options(1) = 1;			% This provides display of error values.
options(14) = net.ncycles;%net.ncycles;		% Number of training cycles. 
net=UpdateStructure(net);
for i=1:net.ncycles;
    
   [y, z] = mlpfwd(net, xlearn); 	%forward propagation
   deltas = y - ylearn;
   g = mlpbkp(net, xlearn, z, deltas);
   ww = [net.w1(:)',net.b1, net.w2(:)', net.b2];
   net = mlpunpak(net, ww - net.eta*g);
   
end

end