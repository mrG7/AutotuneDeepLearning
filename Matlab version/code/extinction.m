% Delite the worst individual of population -> in the future it can delite
% N worst individuals
function [new_population, minn]=extinction(new_big_population,error)
[~,number]=max(error);
new_population=new_big_population;
error(number)=[];
new_population(number,:)=[];
[~,minn]=min(error);




