function sigma = comp_dist(X)

size1=length(X);

if size1>500
    Xmed = X(1:500,:);
    size1 = 500;
else
    Xmed = X;
end

G = sum((Xmed.*Xmed),2);
Q = repmat(G,1,size1);
R = repmat(G',size1,1);
dists = Q + R - 2*Xmed*Xmed';
dists = dists-tril(dists);
dists=reshape(dists,size1^2,1);
sigma = sqrt(0.5*median(dists(dists>0)));  

