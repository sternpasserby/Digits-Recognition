function result = dsigmoid(x)
result = exp(-x) ./ ( 1 + exp(-x) ).^2;
end