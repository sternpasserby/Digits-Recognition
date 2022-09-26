clear;
load matlab.mat;
n = [784, 30, 10];
net = Network(n, "cross-entropy");
eta = 0.5;
m = 10;
epoch = 30;

% for s = 2:length(n)
%     for k = 1:n(s)
%         net.w{s}(k, :) = net.w{s}(k, :)./sqrt(n(s-1));
%     end
% end
[Cost_train, A_train, Cost_test, A_test] = net.matrixTrain(trainSet(1:50000, :), ...
    trainSet(50001:end, :), testSet, eta, m, epoch);

subplot(2, 2, 1)
plot(Cost_train)
title("Cost on training data")
subplot(2, 2, 2)
plot(A_train)
title("Accuracy on training data")
subplot(2, 2, 3)
plot(Cost_test)
title("Cost on test data")
subplot(2, 2, 4)
plot(A_test)
title("Accuracy on test data")

for s = 1:length(times2)
    times2(s) = times(s+1) - times(s);
end
plot(progress(2:end), times2)

% net.w = w;
% net.b = b;
% tic
% Cost2 = net.matrixTrain(trainSet, eta, m, 10);
% t1 = toc

% for s = 1:30
%     subplot(6, 5, s);
%     showW(net2.w{2}, net2.b{2}, s);
% end