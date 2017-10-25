function [resY] = Predict_OLGFL(testX,w,trainX,trainY,Sigma)
trainN = size(trainX,1);
testN = size(testX,1);
X = [trainX; testX];

[N, d] = size(X);
%option.Niter = N;
KX = zeros(N^2,d);
H = eye(N) - 1/N*ones(N);

pKX = zeros(trainN,testN,d);
for i = 1:d
 %   medx = compmedDist([X(:,i)]);
 %   Sigma = [Sigma medx];
    Kx = kernel_gauss(X(:,i)',X(:,i)',Sigma(i));
    KX = H*Kx*H;
    tKX = KX(:,trainN+1:end);
    tKX = tKX(1:trainN,:);
    pKX(:,:,i) = tKX;
  %  KX(:,i) = tmp(:);
end
%pKX: trainN*testN*P


resY = [];
for iTe = 1: testN
   % size(pKX(:,iTe,:))
    
    iKTest = [ones(trainN,1),reshape(pKX(:,iTe,:),[trainN,d])];

    tv = iKTest*w;

    dy = [];
    for i = 1:trainN
        if tv(i) >0
           dy = [dy ; trainY(i)]   ;
        elseif tv(i) <0
           dy = [dy ; - trainY(i)]   ;
        elseif tv(i) == 0
            dy = [dy; 0];
        end
    end

    py = sum(dy);

    if py > 0
        y = 1;
    elseif py <0
        y = -1;
    elseif py == 0
        y = 1;
    end
    
    resY = [ resY;y];
end







