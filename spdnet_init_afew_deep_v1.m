function net = spdnet_init_afew_deep_v1(varargin)
% spdnet_init initializes the spdnet

rng('default');
rng(0) ;

opts.layernum = 6;

Winit = cell(opts.layernum,1);
opts.datadim = [51,43,37,33,37,43,51]; % the dimensionality of each bimap layer, which also indicates the kernel size of each layer


for iw = 1 : opts.layernum % designed to initialize each cov kernel 
    if iw < 4
       A = rand(opts.datadim(iw));
       [U1, ~, ~] = svd(A * A');
       Winit{iw} = U1(:,1:opts.datadim(iw+1)); % the initialized filters are all satisfy column orthogonality, residing on the Stiefel manifolds
    else
       A = rand(opts.datadim(iw+1));
       [U1, ~, ~] = svd(A * A');
       temp = U1(:,1:opts.datadim(iw));
       Winit{iw} = temp';
    end
end

f=1/100 ;
classNum = 155; % categories
fdim = size(Winit{iw-3},2) * size(Winit{iw-3},2);
theta = f * randn(fdim, classNum, 'single'); 
Winit{end+1} = theta; % the to-be-learned projection matrix of the FC layer

net.layers = {} ; % use to construct each layer of the proposed SPDNet
net.layers{end+1} = struct('type', 'bfc',...
                          'weight', Winit{1}) ;
net.layers{end+1} = struct('type', 'rec') ; 
net.layers{end+1} = struct('type', 'bfc',...
                          'weight', Winit{2}) ;
net.layers{end+1} = struct('type', 'rec') ;
net.layers{end+1} = struct('type', 'bfc',...
                          'weight', Winit{3}) ;
net.layers{end+1} = struct('type', 'marginloss') ; % is not used in our U-SPDNet
net.layers{end+1} = struct('type', 'bfc',...
                          'weight', Winit{4}) ;
net.layers{end+1} = struct('type', 'rec') ;

net.layers{end+1} = struct('type', 'log') ;
net.layers{end+1} = struct('type', 'log') ;
net.layers{end+1} = struct('type', 'add') ; % lines 46-49, the first LFE module used for feature fusion
net.layers{end+1} = struct('type', 'exp') ;

net.layers{end+1} = struct('type', 'bfc',...
                          'weight', Winit{5}) ;
net.layers{end+1} = struct('type', 'rec') ; 

net.layers{end+1} = struct('type', 'log') ;
net.layers{end+1} = struct('type', 'log') ;
net.layers{end+1} = struct('type', 'add') ; % lines 55-58, the second LFE module used for feature fusion
net.layers{end+1} = struct('type', 'exp') ;

net.layers{end+1} = struct('type', 'bfc',...
                          'weight', Winit{6}) ;
net.layers{end+1} = struct('type', 'reconstructionloss') ;
net.layers{end+1} = struct('type', 'log') ;

%% the following in the new framework may be removed 
net.layers{end+1} = struct('type', 'fc', ...
                           'weight', Winit{end}) ;
net.layers{end+1} = struct('type', 'softmaxloss') ;
