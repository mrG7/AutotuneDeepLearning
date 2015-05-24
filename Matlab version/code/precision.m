function  precision=precision(net,xcontrol,tcontrol)
t = mlpfwd(net, xcontrol);
maxt=max(t,[],2);
tans=(t-repmat(maxt,1,net.nout))==0;
k=zeros(net.nout,1);
l=zeros(net.nout,1);
for j=1:net.nout
  for i=1:size(tcontrol,1)  
        if (tcontrol(i,j)==1)
            l(j)=l(j)+1;
            if (tans(i,j)==1) 
                k(j)=k(j)+1;
            end
        end
  end
end
precision=k./l;
    
        
        
            

