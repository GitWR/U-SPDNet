function [res] = vl_myforbackward(net, x, dzdy, res, epoch, count1, varargin)
% vl_myforbackward  evaluates a simple SPDNet

opts.res = [] ;
opts.conserveMemory = false ;
opts.sync = false ;
opts.disableDropout = false ;
opts.freezeDropout = false ;
opts.accumulate = false ;
opts.cudnn = true ;
opts.skipForward = false;
opts.backPropDepth = +inf ;
opts.epsilon = 1e-5; % this parameter is worked in the ReEig Layer

% dev_ml = cell(1,30);
% dev_re = cell(1,30);
% for p = 1 : 30
%     dev_ml{p} = zeros(50,50);
%     dev_re{p} = zeros(400,400);
% end

n = numel(net.layers) ; % count the number of layers

if (nargin <= 2) || isempty(dzdy)
  doder = false ;
else
  doder = true ; % this variable is used to control when to compute the derivative
end

if opts.cudnn
  cudnn = {'CuDNN'} ;
else
  cudnn = {'NoCuDNN'} ;
end

gpuMode = isa(x, 'gpuArray') ;

if nargin <= 3 || isempty(res)
  res = struct(...
    'x', cell(1,n+1), ...
    'dzdx', cell(1,n+1), ... % this gradient is necessary for computing the gradients in the layers below and updating their parameters  
    'dzdw', cell(1,n+1), ... % this gradient is required for updating W
    'aux', cell(1,n+1), ...
    'time', num2cell(zeros(1,n+1)), ...
    'backwardTime', num2cell(zeros(1,n+1))) ;
end
if ~opts.skipForward
  res(1).x = x ;
end


% -------------------------------------------------------------------------
%                                                              Forward pass
% -------------------------------------------------------------------------

for i=1:n
  if opts.skipForward
      break; 
  end
  l = net.layers{i} ; % each net layer stores two components: (1) layer type (2) weight
  res(i).time = tic ; % count the time spend on each layer
  switch l.type
    case 'bfc'
      res(i+1).x = vl_mybfc(res(i).x, l.weight, i, res) ; % the output data of each layer is stored in the x part
    case 'fc'
      res(i+1).x = vl_myfc(res(i).x, l.weight) ;
    case 'rec'
        if doder
            alt = doder;
        else
            alt = doder + 1;
        end
        [res(i+1).x, SS] = vl_myrec(res(i).x, opts.epsilon, alt, []) ;
        res(i+1).SS = SS;
      %res(i+1).recov = recov;
    case 'add'
      sc = res(i-1).x;
      res(i+1).x = vl_myadd(res(i).x,sc);
    case 'rec_relu'
      res(i+1).x = vl_myrec_relu(res(i).x, opts.epsilon) ;
    case 'marginloss'
      res(i+1).obj = 0.0;
      res(i+1).x = res(i).x;
    case 'reconstructionloss'
       res(i+1).obj = vl_myreconstructionloss(res(i).x, res(1).x, epoch); % 
       res(i+1).x = res(7).x;
    case 'log'
        if doder
            alt = doder;
        else
            alt = doder + 1;
        end
        if i == 10
            sc = res(i-5).x;
        elseif i == 16
            sc = res(i-13).x;
        else
            sc = res(i).x;
        end
        [res(i+1).x, SS] = vl_mylog(sc, alt, []) ;
        res(i+1).SS = SS;
    case 'exp'
        if doder
            alt = doder;
        else
            alt = doder + 1;
        end
        [res(i+1).x, SS] = vl_myexp(res(i).x, alt, []) ;
        res(i+1).SS = SS;
    case 'softmaxloss'
      res(i+1).x = vl_mysoftmaxloss(res(i).x, l.class) ;
    case 'custom'
          res(i+1) = l.forward(l, res(i), res(i+1)) ;
    otherwise
      error('Unknown layer type %s', l.type) ;
  end
  % optionally forget intermediate results
  forget = opts.conserveMemory ;
  forget = forget & (~doder || strcmp(l.type, 'relu')) ;
  forget = forget & ~(strcmp(l.type, 'loss') || strcmp(l.type, 'softmaxloss')) ;
  forget = forget & (~isfield(l, 'rememberOutput') || ~l.rememberOutput) ;
  if forget
    res(i).x = [] ;
  end 
  if gpuMode & opts.sync
    % This should make things slower, but on MATLAB 2014a it is necessary
    % for any decent performance.
    wait(gpuDevice) ;
  end
  res(i).time = toc(res(i).time) ;
end

% -------------------------------------------------------------------------
%                                                             Backward pass
% -------------------------------------------------------------------------

if doder
  res(n+1).dzdx = dzdy ; % the right hand first part of eq.6 in SPDNet. Here, its value is 1
  for i = n:-1:max(1, n-opts.backPropDepth+1) % calculate the derivate in reversed order
    l = net.layers{i} ;
    res(i).backwardTime = tic ;
    switch l.type
      case 'bfc'
        [res(i).dzdx, res(i).dzdw] = ... % all the data in a given batch share the same weight
             vl_mybfc(res(i).x, l.weight, i, res, res(i+1).dzdx) ; % 
                                                           
      case 'fc'
        [res(i).dzdx, res(i).dzdw]  = ...
              vl_myfc(res(i).x, l.weight, res(i+1).dzdx) ; 
      case 'rec'
        temp = res(i).x;
        if i == 4
            dev_sc = res(i+6).dzdx; % 
        elseif i == 2
            dev_sc = res(i+14).dzdx; % 
        else
            ZM = zeros(size(temp{1},1),size(temp{1},2));
            for num = 1 : length(temp)
                dev_sc{num} = ZM;
            end
        end
        alt = doder;
        alt = alt - 1;
        [res(i).dzdx, SS] = vl_myrec(res(i).x, opts.epsilon, alt, res(i+1).SS, res(i+1).dzdx, dev_sc) ;
        %[res(i).dzdx, recov] = vl_myrec(res(i).x, opts.epsilon, res(i+1).dzdx) ;
      case 'add'
        sc = res(i-1).x;
        res(i).dzdx = vl_myadd(res(i).x, sc, res(i+1).dzdx);
      case 'rec_relu'
        res(i).dzdx = vl_myrec_relu(res(i).x, opts.epsilon, res(i+1).dzdx) ;
      case 'marginloss'
        dev_ml_trans = cell(length(res(i).x),1);
        dzdx_recon = res(i+1).dzdx;
        dzdx_log = res(i+15).dzdx;
        for ii = 1 : length(res(i).x)
            dev_ml_trans{ii} = dzdx_recon{ii} + dzdx_log{ii};
        end
        res(i).dzdx = dev_ml_trans;
      case 'reconstructionloss'
        res(i).dzdx = vl_myreconstructionloss(res(i).x, res(1).x, epoch, res(i+1).dzdx) ; % dev_re
      case 'exp'
        alt = doder;
        alt = alt - 1; 
        [res(i).dzdx, SS] = vl_myexp(res(i).x, alt, res(i+1).SS, res(i+1).dzdx) ;  
      case 'log'
        alt = doder;
        alt = alt - 1; 
        if i == 9 || i == 15
            dev_sc = res(i+2).dzdx;
        else
            dev_sc = res(i+1).dzdx;
        end
        if i == 10
            sc = res(i-5).x;
        elseif i == 16
            sc = res(i-13).x;
        else
            sc = res(i).x;
        end
        [res(i).dzdx, SS] = vl_mylog(sc, alt, res(i+1).SS, dev_sc) ;
      case 'softmaxloss'
        res(i).dzdx = vl_mysoftmaxloss(res(i).x, l.class, res(i+1).dzdx) ;
      case 'custom'
        res(i) = l.backward(l, res(i), res(i+1));
    end
    if opts.conserveMemory
      res(i+1).dzdx = [] ;
    end
    if gpuMode & opts.sync
      wait(gpuDevice) ;
    end
    res(i).backwardTime = toc(res(i).backwardTime) ;
  end
end

