function X=kernel_gauss(x,c,sigma)
    
  [d nx]=size(x);
  [d nc]=size(c);
  
  if nx<d || nc <d
      X = [];
      return;
  end
  
  
  x2=sum(x.^2,1);
  c2=sum(c.^2,1);

  distance2=repmat(c2,nx,1)+repmat(x2',1,nc)-2*x'*c;
  X=exp(-distance2/(2*sigma^2));
