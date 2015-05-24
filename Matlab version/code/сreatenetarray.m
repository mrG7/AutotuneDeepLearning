function [net_array]=createnetarray(net,population)
net_array(size(population,1))=ziros;
for i=1:size(population,1)
net_array(i)=netunpack_nobias(net,populaton(i));
end
