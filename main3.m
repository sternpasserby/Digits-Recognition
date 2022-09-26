clear;
load matlab.mat;
n = 25;
net = Network(n);
eta = 3;
m = 20;
epoch = 60;
%C.YDataSource = 'Cost';

Cost = net.matrixTrain(trainSet, eta, m, epoch);

for s = 1:n
    subplot(6, 5, s);
    showW(net.w2, net.b2, s);
end