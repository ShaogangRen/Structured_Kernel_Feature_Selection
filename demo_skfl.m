
clear all;
close all;
addpath('skfl/');

data_path = 'data/';
dataFile = 'data_miu200.mat';

load([ data_path '/' dataFile]);
load([ data_path '/mask.mat']);


N = length(Y);
opt.lam1 = 0.5;
opt.lam2 =  5;
opt.loss = 'hinge'; 
opt.Nepoch = 1;
opt.Niter = N;
opt.beta = 10;

X = double(X);
X = X - repmat(mean(X,2),[1,size(X,2)]);
X = X./repmat(sqrt(sum(X.^2, 2)),[1,size(X,2)]);
X = X';
Y = double(Y)';

%%Q is half adjcent matrix
Adj = Q + Q';
MaxEn = 6; 
graph = AdjToGraph(Adj,MaxEn);

tL = testL{1};
trainX =  X(tL,:);
trainY =  Y(tL);
testX  = X;
testY  = Y;
testX(tL,:) = [];
testY(tL)  = []; 
NTrain = length(trainY)

tic
[w,Sigma] = skfl(X, Y, graph, opt);
time_used = toc
a =w(2:end);

testN = length(testY);
[PY] = Predict_OLGFL(testX,w,trainX,trainY,Sigma);
PreAcc = sum( PY == testY)/testN %prediction accuracy
RegAcc = RegAcc_Sort(a, vgmask,vlob_mask, height,width,depth) %region accuracy



