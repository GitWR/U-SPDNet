function [net, info] = spdnet_train_afew(net, spd_train, opts)

opts.errorLabels = {'top1e'};
opts.train = find(spd_train.spd.set==1) ; % 1 represents the training samples
opts.val = find(spd_train.spd.set==2) ; % 2 indicates the testing samples
count = 0;
%load my_label % the label of training data
%load data_indx_2

for epoch = 1 : opts.numEpochs 
    
    learningRate = opts.learningRate(epoch); % 
    if epoch >= 100
        lr_decay_steps = 50;
        if rem(epoch,lr_decay_steps) == 0
           count = count + 1;           
        end
        learningRate = 0.9^(count) * learningRate;
        if learningRate <= 0.00001
            learningRate = 0.00001;
        end
    end
     %% fast-forward to last checkpoint
     modelPath = @(ep) fullfile(opts.dataDir, sprintf('net-epoch-%d.mat', ep));
     modelFigPath = fullfile(opts.dataDir, 'net-train.pdf') ;
     if opts.continue
         if exist(modelPath(epoch),'file')
             if epoch == opts.numEpochs
                 load(modelPath(epoch), 'net', 'info') ;
             end
             continue ;
         end
         if epoch > 1
             fprintf('resuming by loading epoch %d\n', epoch-1) ;
             load(modelPath(epoch-1), 'net', 'info') ;
         end
     end
    
    %data_ori = cell(1,5);
    %data_recov = cell(1,5);
    
    train = opts.train(randperm(length(opts.train))) ; % data_label; shuffle, to make the training data feed into the net in disorder
    val = opts.val; % the train data is in order to pass the net
    [net,stats.train] = process_epoch(opts, epoch, spd_train, train, learningRate, net) ;
    [net,stats.val] = process_epoch(opts, epoch, spd_train, val, 0, net) ;
    
   %% doing classification
%     Train_label = spd_train.spd.label(train);
%     Val_label = spd_train.spd.label(val);
%     prediction = classification(StoreData_train, StoreData_test, Train_label, Val_label);
%     fprintf('Prediction: %d-th iteration, accuracy is %.4f',epoch,prediction);
%     fprintf('\n') ; 
    
   %% the following is to dynamicly draw the cost curve
   evaluateMode = 0;
     if evaluateMode
         sets = {'train'};
     else
         sets = {'train', 'val'};
     end
     for f = sets
         f = char(f);
         n = numel(eval(f));
         info.(f).objective(epoch) = (stats.(f)(2) + stats.(f)(3) + stats.(f)(4)) / n; % stats.(f)(2) / n;
         info.(f).acc(:,epoch) = stats.(f)(5:end) / n; % stats.(f)(3:end) / n;
     end
     if ~evaluateMode, save(modelPath(epoch), 'net', 'info') ; end
     
     if epoch >= 1
     figure(1);
     clf;
     hasError = 1;
     subplot(1,hasError+1,1);
     if ~evaluateMode
         semilogy(1:epoch,info.train.objective,'.--','linewidth',2);
     end
     grid on;
     h = legend(sets);
     set(h,'color','none');
     xlabel('training epoch');
     ylabel('cost value');
     title('objective');
     if hasError
         subplot(1,2,2);
         leg={};
         plot(1:epoch,info.val.acc','.--','linewidth',2);
         leg = horzcat(leg,strcat('val')); % ,opts.errorLabels
         set(legend(leg{:}),'color','none');
         grid on;
         xlabel('training epoch');
         ylabel('error');
         title('error')
     end
     drawnow;
     print(1,modelFigPath,'-dpdf');
     end
end  
    
    
function [net,stats] = process_epoch(opts, epoch, spd_train, trainInd, learningRate, net)

training = learningRate > 0 ;
count1 = 0;

if training
    mode = 'training' ; 
else
    mode = 'validation' ; 
end

stats = [0 ; 0; 0; 0; 0] ;
% stats = [0 ; 0; 0] ; % for softmax 

numGpus = numel(opts.gpus) ;
if numGpus >= 1
    one = gpuArray(single(1)) ;
else
    one = single(1) ;
end

batchSize = opts.batchSize;
errors = 0;
numDone = 0 ;
flag = 0;

for ib = 1 : batchSize : length(trainInd) % select the training samples. Here, 10 pairs of samples per group
    flag = flag + 1;
    fprintf('%s: epoch %02d: batch %3d/%3d:', mode, epoch, ib,length(trainInd)) ;
    batchTime = tic ; %
    res = [];
    if (ib+batchSize> length(trainInd))
        batchSize_r = length(trainInd)-ib+1;  %
    else
        batchSize_r = batchSize;
    end
    spd_data = cell(batchSize_r,1); % store the data in each batch    
    spd_label = zeros(batchSize_r,1); % store the label of the data in each batch
    for ib_r = 1 : batchSize_r
      
        spdPath = [spd_train.spd.name{trainInd(ib+ib_r-1)}];
        load(spdPath);
        spd_data{ib_r} = temp_2;
        spd_label(ib_r) = spd_train.spd.label(trainInd(ib+ib_r-1));
        
    end
    net.layers{end}.class = spd_label; % one-hot vector is used to justify your algorithm generate a right label or not 
    net.layers{6}.class = spd_label; % 
    %forward/backward spdnet
    if training
        dzdy = one; 
    else
        dzdy = [] ;
    end
    res = vl_myforbackward(net, spd_data, dzdy, res, epoch, count1) ; % its for-and-back-ward process

    %data_ori{flag} = res(4).x;
    %data_recov{flag} = res(5).recov;
    
    %%accumulating graidents
    if numGpus <= 1
      net = accumulate_gradients(opts, learningRate, batchSize_r, net, res) ;
    else
      if isempty(mmap)
        mmap = map_gradients(opts.memoryMapFile, net, res, numGpus) ;
      end
      write_gradients(mmap, net, res) ;
      labBarrier() ;
      net = accumulate_gradients(opts, learningRate, batchSize_r, net, res, mmap) ;
    end
          
    % accumulate training errors
    predictions = gather(res(end-1).x) ;
    [~,pre_label] = sort(predictions, 'descend') ;
    error = sum(~bsxfun(@eq, pre_label(1,:)', spd_label)) ;
    errors = errors+error;
    batchTime = toc(batchTime) ;
    speed = batchSize/batchTime ;
    numDone = numDone + batchSize_r ;
    
    stats = stats+[batchTime ; res(end).x; res(7).obj; res(21).obj; error]; % works even when stats=[] res(15).obj
    fprintf(' %.2f s (%.1f data/s)', batchTime, speed) ;
    fprintf(' l1-sm: %.5f', stats(2)/numDone) ; % the value of objective function
    fprintf(' l2-ml: %.5f', stats(3)/numDone) ; % 0.0 in this net
    fprintf(' l2-rebud: %.5f', stats(4)/numDone) ;
    fprintf(' l-mix: %.5f', (stats(2) + stats(3) + stats(4))/numDone) ;
    fprintf(' error: %.5f', stats(5)/numDone) ;
    fprintf(' [%d/%d]', numDone, batchSize_r);
    fprintf(' lr: %.6f',learningRate);
    fprintf('\n') ; 
    
end


% -------------------------------------------------------------------------
function [net,res] = accumulate_gradients(opts, lr, batchSize, net, res, mmap)
% -------------------------------------------------------------------------
for l = numel(net.layers):-1:1 % reverse order
  if isempty(res(l).dzdw)==0 % when the layer is defined on the SPD manifold, we must optimize W on Steifiel maniold using RCG 
    if ~isfield(net.layers{l}, 'learningRate')
       net.layers{l}.learningRate = 1 ;
    end
    if ~isfield(net.layers{l}, 'weightDecay')
       net.layers{l}.weightDecay = 1;
    end
    thisLR = lr * net.layers{l}.learningRate ;

    if isfield(net.layers{l}, 'weight')
        if strcmp(net.layers{l}.type,'bfc')==1
            W1 = net.layers{l}.weight;
            W1grad  = (1/batchSize)*res(l).dzdw;
            % gradient update on Stiefel manifolds
            problemW1.M = stiefelfactory(size(W1,1), size(W1,2));
            W1Rgrad = (problemW1.M.egrad2rgrad(W1, W1grad)); % this sentence has a problem
            %net.layers{l}.weight = (problemW1.M.retr(W1, -thisLR*W1Rgrad)); % retr indicates retraction (back to manifold)
            net.layers{l}.weight = (problemW1.M.retr(W1, -thisLR*W1Rgrad)); % retr indicates retraction (back to manifold)
        else
            net.layers{l}.weight = net.layers{l}.weight - thisLR * (1/batchSize) * res(l).dzdw ;% update W, here just for fc layer
        end
    
    end
  end
end

