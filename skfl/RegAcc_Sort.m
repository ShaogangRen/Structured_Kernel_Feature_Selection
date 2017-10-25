function regacc = RegAcc_Sort(img, vgmask,vlob_mask, height,width,depth)
%              p = height*width*depth;
%              vimg = zeros(p,1);
%              vimg(vgmask==1) = 1;
%              gtMask = vimg(vlob_mask);
%            
%           %  img = a;
%             bnimg = zeros(length(img),1);
%             bnimg(img ~= 0) = 1;
% 
%             detN = sum(bnimg);
%             grdN = sum(gtMask);
%             regacc = 2*sum(bnimg & gtMask)/(detN + grdN);
% 
bnimg = zeros(length(img),1);
MaskPixN = sum(vgmask);
[pixelv,idx]  = sort(abs(img),'descend');
bnimg(idx(1:MaskPixN)) = 1;
p = height*width*depth;
vimg = zeros(p,1);
vimg(vgmask==1) = 1;
gtMask = vimg(vlob_mask);
regacc = (2*MaskPixN - sum(abs(gtMask - bnimg)))/(2*MaskPixN);