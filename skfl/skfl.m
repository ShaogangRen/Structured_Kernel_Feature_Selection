function [ra,Sigma] = skfl(X, Y, graph, opt)

[N,d] = size(X);
H = eye(N) - 1/N*ones(N);
KX = zeros(N^2,d);
Sigma = [];
for i = 1:d
    medx = comp_dist([X(:,i)]);
    Sigma = [Sigma medx];
    Kx = kernel_gauss(X(:,i)',X(:,i)',medx);
    tmp = H*Kx*H;
    KX(:,i) = tmp(:);
end


KX = [ones(N^2,1),KX];
d = d+1;
tmp = Y*Y';
LY = H*tmp*H;
KY = LY(:);


histu = [];
a = zeros(d,1);
ma = zeros(d,1);
mu = zeros(d,1);

iiBuf = randi(N,[1, opt.Niter]);
iter = 1;
Nepoch = 1;

while (Nepoch<=opt.Nepoch)
    Nepoch = Nepoch+1;
    
    ii = 1;
    
    while ii<=opt.Niter
        Kx_t = KX((ii-1)*N +1 : ii*N,:);
        Ky_t = KY((ii-1)*N +1 : ii*N);
        a_t = a(:,iter);
        ii = ii + 1;
 
        yN = length(Ky_t);


        switch opt.loss
            case 'square'
                 ;
            case 'logistic'
               % disp('logistic loss!');
                Pa = size(Kx_t,2);
                temp = -Ky_t./( 1+ exp(Ky_t.*(Kx_t*a_t)));
                dem = repmat(temp, [1,Pa]);
                tt = sum(Kx_t.*dem, 1)';
                al = a_t(2:end);
                

                en = size(graph,2);
                aa = repmat(al,[1,en]);

                gg = sum(aa - al(graph),2);
                gg  = tt(2:end) + opt.lam1 * ones(Pa-1,1) + 2*opt.lam2*gg;
                u_t = [tt(1); gg];
      
            case 'hinge'
              
               Pa = size(Kx_t,2);
               tt = zeros(d,1);
               if Ky_t'*Kx_t*a_t< yN
                  tt = -Kx_t'*Ky_t;
               end
                al = a_t(2:end);
                en = size(graph,2);
                aa = repmat(al,[1,en]);
                gg = sum(aa - al(graph),2);
                gg  = tt(2:end) + opt.lam1 * ones(Pa-1,1) + 2*opt.lam2*gg;
                u_t = [tt(1); gg];      
        end
        
        iter = iter+1;

        ma(:, iter) = (iter-2)/(iter-1)*ma(:,iter-1)+1/(iter-1)*a_t;
        mu(:,iter) = (iter-2)/(iter-1)*mu(:,iter-1)+1/(iter-1)*u_t;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        g_t = mu(:,iter);
        
        if iter == 2
           a_new  = - iter*g_t/(2*opt.beta*log(iter));
        else
           a_new  = - (iter-1)*g_t/(2*opt.beta*log(iter-1));
        end
        
        tt = a_new(2:end);
        tt(tt<0) = 0;
        a_new(2:end) = tt;
        a(:,iter) = a_new;
    end
end
ra = a(:,end);




