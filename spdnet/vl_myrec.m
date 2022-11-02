function [Y, SS] = vl_myrec(X, epsilon, doder, SS, dzdy, dev_sc)
% Y = VL_MYREC (X, EPSILON, DZDY)
% ReEig layer

Us = cell(length(X),1);
Ss = cell(length(X),1);
Vs = cell(length(X),1);
% thres = 1e-5; % 
min_v = zeros(1,30);
Y = cell(length(X),1);
% Y_recov = cell(length(X),1);
if doder
    SS = cell(3,1);
    for ix = 1 : length(X)
        temp = X{ix}; % get each sample
        [Us{ix},Ss{ix},Vs{ix}] = svd(temp);
        min_v(ix) = min(min(diag(Ss{ix})));
    end

    for ix = 1:length(X)
        [max_S, ~]=max_eig(Ss{ix},epsilon); % the equation try to perform relu in SPDNet
        Y{ix} = Us{ix}*max_S*Us{ix}'; % use the modified eig to re-build the data in this layer
    end
    SS{1} = Us; SS{2} = Ss; SS{3} = Vs;
    
else
    Us = SS{1}; Ss = SS{2}; Vs = SS{3};
    D = size(Ss{1},2);
    for ix = 1:length(X)
        U = Us{ix}; S = Ss{ix}; V = Vs{ix};

        Dmin = D;
        
        dLdC = double(dzdy{ix}); dLdC = symmetric(dLdC); % the same as the operation in logeig layer
        
        [max_S, max_I] = max_eig(Ss{ix},epsilon); % the equation try to perform relu in SPDNet 
        dLdV = 2*dLdC*U*max_S;
        dLdS = (diag(not(max_I)))*U'*dLdC*U; % see eq.18 in SPDNet
        
        
        K = 1./(diag(S)*ones(1,Dmin)-(diag(S)*ones(1,Dmin))'); % the same as the operation in logeig layer
        K(eye(size(K,1))>0)=0; % eq.14 in spdnet
        K(find(isinf(K)==1))=0; 
        
        dzdx = U*(symmetric(K'.*(U'*dLdV))+dDiag(dLdS))*U'; % the same as the operation in logeig layer
        
        Y{ix} =  dzdx + dev_sc{ix}; % warning('no normalization');
    end
end
