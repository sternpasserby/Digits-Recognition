function result = shuffleRows(a)
[n, m] = size(a);
idx = randperm(n);
result = zeros(n, m);
for s = 1:n
    result(s, :) = a(idx(s), :);
end
end
