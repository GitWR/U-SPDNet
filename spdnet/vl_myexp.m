function [Y, SS] = vl_myexp(X, doder, SS, dzdy)
%Y = VL_MYLOG(X, DZDY)
%ExpEig layer

Us = cell(length(X),1);
Ss = cell(length(X),1);
Vs = cell(length(X),1);
Y = cell(length(X),1);

if doder
    SS = cell(3,1);
    for ix = 1 : length(X)
        temp = X{ix};
        temp_sym = (temp + temp')/2;
        [Us{ix},Ss{ix}] = eig(temp_sym);
    end
    
    Vs = Us;


    for ix = 1:length(X)
        Y{ix} = Us{ix} * diag(exp(diag(Ss{ix}))) * Us{ix}'; % exp map
    end
    SS{1} = Us; SS{2} = Ss; SS{3} = Vs;
    
else
    
    Us = SS{1}; Ss = SS{2}; Vs = SS{3};
    D = size(Ss{1},2);
    for ix = 1:length(X)
        U = Us{ix}; S = Ss{ix};  % V = Vs{ix};
        diagS = diag(S);
        ind = diagS > -1000; % (D*eps(max(diagS)));
        Dmin = (min(find(ind,1,'last'),D));
        
        %yita = min(1,100/norm(dzdy{ix},'fro'));
        %dzdy{ix} = dzdy{ix} * yita;
        
        S = S(:,ind); U = U(:,ind);
        dLdC = double(dzdy{ix}); % double(reshape(dzdy(:,ix),[D D])); dLdC = symmetric(dLdC); % the gradient inormation of the higher layer
        dLdC = symmetric(dLdC);

        dLdV = 2*dLdC*U*diagExp(S,0); % eq.19
        dLdS = diagExp(S,0)*(U'*dLdC*U); % eq.20
        
        if sum(ind) == 1 % diag behaves badly when there is only 1d
            K = 1./(S(1)*ones(1,Dmin)-(S(1)*ones(1,Dmin))'); 
            K(eye(size(K,1))>0)=0;
        else
            K = 1./(diag(S)*ones(1,Dmin)-(diag(S)*ones(1,Dmin))');
            K(eye(size(K,1))>0)=0;
            K(find(isinf(K)==1))=0; 
        end
        if all(diagS==1)
            dzdx = zeros(D,D);
        else
            dzdx = U*(symmetric(K'.*(U'*dLdV))+dDiag(dLdS))*U';

        end
        Y{ix} =  dzdx; %warning('no normalization');        
    end
end


