function [p, matrix] = make_pairs(f, P)

matrix = zeros(size(f,2));

for i = 1:size(f,2)
    for j = i+1:size(f,2)
        matrix(i,j) = sum(f(:,i) == f(:,j))/size(f,2);
    end
end

value = 1:P;

i=0;
p = zeros(1, P);
while sum(value)> 1
    flag = 0;
    i = i+1;
    idx = find(value ~=0, 1, 'first');
    p(i) = idx;
    value(idx) = 0;
    while flag == 0
        ind = find(matrix(idx,idx+1:end) == min(matrix(idx,idx+1:end)), 1, 'first');
        if value(idx+ind)~=0
            flag = 1;
        else
            matrix(idx, idx+ind) = Inf;
        end
    end
    i = i+1;
    p(i) = idx + ind;
    value(idx + ind) = 0;
end

p = reshape(p, 2, P/2);
p = p';
end