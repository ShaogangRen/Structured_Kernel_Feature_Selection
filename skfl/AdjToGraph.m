function g = AdjToGraph(Adj,MaxEn)

EN = size(Adj,1);
g = repmat([1:EN]',[1 MaxEn]);

[rr cc] = find(Adj);

for i = 1:EN
     tt= cc(rr == i);
     for j = 1:length(tt)
         g(i,j) = tt(j);
     end
end



