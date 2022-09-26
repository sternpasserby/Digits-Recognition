classdef Network < handle
%Network Нейронная сеть с тремя слоями. В скрытом слое n нейронов
%   Detailed explanation goes here

properties
w
b
L
Cost_x 
dC_xda_L
end

methods
function obj = Network(n, costFunc)
%Network создаёт экземпляр данного класса. n - число нейронов в
%скрытом слое
%   Detailed explanation goes here
obj.L = length(n);
for s = 2:length(n)
    obj.w{s} = randn(n(s), n(s-1));
    obj.b{s} = randn(n(s), 1);
end
if costFunc == "cross-entropy"
    obj.Cost_x = @(a, y)( -sum(y.*log(a) + (1-y).*log(1-a)) );
    obj.dC_xda_L = @(a, y)( (a - y)./a./(1-a) );
end
if costFunc == "quadric"
    %obj.Cost_x = @(a, y)(a - y)'*(a - y)/2;
    obj.Cost_x = @(a, y)( sum( (y-a).^2 , 1)/2 );
    obj.dC_xda_L = @(a, y)(a - y);
end
end

function [a, z] = feedforward(obj, a1)
%feedforward Summary of this method goes here
%   Detailed explanation goes here
a = cell(1, obj.L);
z = cell(1, obj.L);
a{1} = a1;
for s = 2:obj.L
    z{s} = obj.w{s}*a{s-1} + obj.b{s};
    a{s} = sigmoid(z{s});
end
end

function [digit, a_out] = recognizeDigit(obj, a1)
[a, ~] = feedforward(obj, a1);
[~, index] = max(a{end});
digit = index - 1;
a_out = a{end};
end

function [dw, db] = backpropagate(obj, a1, digit)
y = zeros(10, 1);
y(digit + 1) = 1;
delta = cell(1, obj.L);

[a, z] = feedforward(obj, a1);
%delta{end} = (a{end} - y).*dsigmoid(z{end});
delta{end} = obj.dC_xda_L(a{end}, y).*dsigmoid(z{end});
for s = obj.L-1:-1:2
    delta{s} = (obj.w{s+1}'*delta{s+1}) .* dsigmoid(z{s});
end

for s = obj.L:-1:2
    dw{s} = delta{s}*a{s-1}';
end
db = delta;
end

function [C, accuracy, C_test, accuracy_test] = train(obj, trainSet,...
        validationSet, testSet, eta, m, numOfEpochs)
%Cost = zeros(epoch*floor(50000/m), 1);
C = zeros(numOfEpochs, 1);
accuracy = zeros(numOfEpochs, 1);
C_test = zeros(numOfEpochs, 1);
accuracy_test = zeros(numOfEpochs, 1);
numOfMiniBatches = floor(length(trainSet)/m);
dw = cell(1, obj.L);
db = cell(1, obj.L);
for epoch = 1:numOfEpochs
    trainSet = shuffleRows(trainSet);
    for miniBatch = 1:numOfMiniBatches
        for s = 2:obj.L
            dw{s} = zeros(size(obj.w{s}));
            db{s} = zeros(size(obj.b{s}));
        end
        %ddw2 = dw2;
        %ddw3 = dw3;
        %ddb2 = db2;
        %ddb3 = db3;
        for s = 1:m
            a1 = trainSet( (miniBatch-1)*m + s, 2:end )';
            digit = trainSet( (miniBatch-1)*m + s, 1 );
            [a, ~] = obj.feedforward(a1);
            [ddw, ddb] = obj.backpropagate(a1, digit);
            for k = 2:obj.L
                dw{k} = dw{k} + ddw{k};
                db{k} = db{k} + ddb{k};
            end
            
            y = zeros(10, 1);
            y(digit + 1) = 1;
            %Cost( (o-1)*floor(50000/m) + b ) = ...
                %Cost( (o-1)*floor(50000/m) + b ) + (a{end} - y)'*(a{end} - y);
            %C(o) = C(o) + (a{end} - y)'*(a{end} - y);
            C(epoch) = C(epoch) + obj.Cost_x(a{end}, y);
        end
        %Cost( (o-1)*floor(50000/m) + b ) = Cost( (o-1)*floor(50000/m) + b )/m;
        obj.updateParametersBy(eta, m, dw, db);
    end
    C(epoch) = C(epoch)/(numOfMiniBatches*m);
    [~, accuracy(epoch)] = obj.trial(validationSet);
    [C_test(epoch), accuracy_test(epoch)] = obj.trial(testSet);
    disp("Epoch " + num2str(epoch) + "/" + num2str(numOfEpochs) +...
        ": " + num2str(accuracy(epoch), "%.4f") + "%")
end
end

function [Cost, accuracy] = trial(obj, trialSet)
Cost = 0;
success = 0;
n = length(trialSet(:, 1));
for s = 1:n
    a1 = trialSet(s, 2:end)';
    digit = trialSet(s, 1);
    y = zeros(10, 1);
    y(digit + 1) = 1;
    [netDigit, a_out] = obj.recognizeDigit(a1);
    if netDigit == digit
        success = success + 1;
    end
    Cost = Cost + obj.Cost_x(a_out, y);
end
Cost = Cost/n;
accuracy = success/n*100;
%disp("Success rate: " + num2str(percent) + "%");
end

function updateParametersBy(obj, eta, m, dw, db)
for s = 2:obj.L
    obj.w{s} = obj.w{s} - eta/m*dw{s};
    obj.b{s} = obj.b{s} - eta/m*db{s};
end
end

function [C_train, accuracy_train, C_test, accuracy_test] = matrixTrain(obj, trainSet,...
        validationSet, testSet, eta, m, numOfEpochs)
C_train = zeros(numOfEpochs, 1);
accuracy_train = zeros(numOfEpochs, 1);
C_test = zeros(numOfEpochs, 1);
accuracy_test = zeros(numOfEpochs, 1);
numOfMiniBatches = floor(length(trainSet)/m);
dw = cell(1, obj.L);
db = cell(1, obj.L);
B = cell(1, obj.L);
Z = cell(1, obj.L);
A = cell(1, obj.L);
delta = cell(1, obj.L);
for epoch = 1:numOfEpochs
    trainSet = shuffleRows(trainSet);
    for miniBatch = 1:numOfMiniBatches
        for s = 2:obj.L
            dw{s} = zeros(size(obj.w{s}));
            db{s} = zeros(size(obj.b{s}));
        end

        X = trainSet((miniBatch-1)*m+1:miniBatch*m, 2:end)';
        A{1} = X;
        for s = 2:obj.L
            B{s} = repmat(obj.b{s},1, m);
            Z{s} = obj.w{s}*A{s-1} + B{s};
            A{s} = sigmoid(Z{s});
        end

        Y = zeros(10, m);
        for s = 1:m
            digit = trainSet((miniBatch-1)*m+s, 1);
            Y(digit+1, s) = 1;
        end
        delta{end} = obj.dC_xda_L(A{end}, Y) .* dsigmoid(Z{end});
        for s = obj.L-1:-1:2
            delta{s} = (obj.w{s+1}' * delta{s+1}) .* dsigmoid(Z{s});
        end

        for s = 1:m
            for k = obj.L:-1:2
                dw{k} = dw{k} + delta{k}(:, s)*A{k-1}(:, s)';
                db{k} = sum(delta{k}, 2);
            end
        end

        %C_train(epoch) = C_train(epoch) + sum(obj.Cost_x(A{end}, Y));
        obj.updateParametersBy(eta, m, dw, db);
    end
    %C_train(epoch) = C_train(epoch)/(numOfMiniBatches*m);
    [C_train(epoch), accuracy_train(epoch)] = obj.trial(trainSet);
    [C_test(epoch), accuracy_test(epoch)] = obj.trial(testSet);
    disp("Epoch " + num2str(epoch) + ": " + num2str(accuracy_train(epoch), "%.3f") + "%")
end
end
end
end