%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is a demo for the U-SENC algorithm, which is proposed in the %
% following paper:                                                  %
%                                                                   %
% D. Huang, C.-D. Wang, J.-S. Wu, J.-H. Lai, and C.-K. Kwoh.        %
% "Ultra-Scalable Spectral Clustering and Ensemble Clustering."     %
% IEEE Transactions on Knowledge and Data Engineering, 2020.        %
% DOI: https://doi.org/10.1109/TKDE.2019.2903410                    %
%                                                                   %
% The code has been tested in Matlab R2016a and Matlab R2016b.      %
% Website: https://www.researchgate.net/publication/330760669       %
% Written by Huang Dong. (huangdonghere@gmail.com)                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function allScores = computeNMI(results,gt)
% Compute the NMI scores.
allScores = zeros(size(results,2),1);
for i = 1:size(results,2)
    if min(results(:,i))>0
        allScores(i) = NMImax(results(:,i), gt);
    end
end

function NMImax = NMImax(x, y)

assert(numel(x) == numel(y));
n = numel(x);
x = reshape(x,1,n);
y = reshape(y,1,n);

l = min(min(x),min(y));
x = x-l+1;
y = y-l+1;
k = max(max(x),max(y));

idx = 1:n;
Mx = sparse(idx,x,1,n,k,n);
My = sparse(idx,y,1,n,k,n);
Pxy = nonzeros(Mx'*My/n); %joint distribution of x and y
Hxy = -dot(Pxy,log(Pxy+eps));

Px = mean(Mx,1);
Py = mean(My,1);

% entropy of Py and Px
Hx = -dot(Px,log(Px+eps));
Hy = -dot(Py,log(Py+eps));

% mutual information
MI = Hx + Hy - Hxy;

% maximum normalized mutual information
NMImax = MI/max(Hx,Hy);
