function [num_of_mutation]= Randn_by_our_distribution(prob_of_cross)
k=rand;
if (k<prob_of_cross(1)) 
    num_of_mutation=1;
end
if (k>sum(prob_of_cross(1:size(prob_of_cross))) )
    num_of_mutation=size(prob_of_cross);
end
for i=2:size(prob_of_cross);
    if ((k<= sum(prob_of_cross(1:i)))&&(k>=sum(prob_of_cross(1:i-1))))
        num_of_mutation=i-1; 
    end
end
