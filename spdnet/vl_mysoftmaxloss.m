function Y = vl_mysoftmaxloss(X, c, dzdy)
% Softmax layer 
mass = single(c > 0) ;
mass = mass';

% convert to indexes
c_ = c - 1 ; % 155 classes from 0 to 154, so we should minus 1
for ic = 1  : length(c)
    c_(ic) = c(ic)+(ic-1)*size(X,1);
end

% compute softmaxloss
Xmax = max(X,[],1) ;
ex = exp(bsxfun(@minus, X, Xmax)) ;

% s = bsxfun(@minus, X, Xmax);
% ex = exp(s) ;
% y = ex./repmat(sum(ex,1),[size(X,1) 1]);

%n = sz(1)*sz(2) ;
if nargin < 3
  t = Xmax + log(sum(ex,1)) - reshape(X(c_), [1 size(X,2)]) ;% X(c_) is used to get the probability of each data in each batch
  Y = sum(sum(mass .* t,1)) ;
else
  Y = bsxfun(@rdivide, ex, sum(ex,1)) ; % used to conduct normalization, this is also the definition of softmax function
  Y(c_) = Y(c_) - 1; % partial derivative with respect to the form of... 
  Y = bsxfun(@times, Y, bsxfun(@times, mass, dzdy)) ; % the partial derivative of the loss with respect to x 
end