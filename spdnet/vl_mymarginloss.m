function Y = vl_mymarginloss(X, c, epoch, count1, dzdy_recon, dzdx_log)
% marginloss layer
% Rui.wang implemented

% npoints = length(X); % number of data in each batch
% Dist = Compute_LED_metric(X, c); % the distance matrix

a1 = -0.1; % a margin throshold to control the intra-class and inter-class distances -5-->-10
a2 = 0.01; % a intra-class margin 1-->5

new_count = count1;
dzdy_l2 = single(1);
lambda = 1.0; % 0.05-->0.01
cur_epoch = epoch;

% lambda = 0.8^floor(epoch / 20) * lambda; % here test softmax term, we remove this setting
% if lambda < 0.002
%     lambda = 0.002;
% end

Nw = zeros(1, length(X));
Nb = zeros(1, length(X));
Sw = zeros(1, length(X)); % use to compute the sum of all the nearest distances
Sb = zeros(1, length(X)); % use to compute the sum of all the farrest distances

temp_dev_Sw = zeros(size(X{1},1),size(X{1},1),length(X));
temp_dev_Sb = zeros(size(X{1},1),size(X{1},1),length(X));

use_parfor = cell(1,length(X));
for i = 1 : size(use_parfor,2)
    use_parfor{i} = X;
end

% this is for smdml-nml-dc
% Y = cell(length(X),1);
% dev_ml = zeros(50,50);
% for j = 1 : length(X)
%     Y{j} = dev_ml + dzdy_recon{j} + dzdx_log{j};
% end

parfor j = 1 : length(X)
    K1 = 4; % the number of nearest neighbor points
    num_eachclass = find(c==c(j));% find other samples with the same label as c(j)
    temp_X = use_parfor{j};
    Xi = temp_X{j};
    Sw_temp = zeros(1,length(num_eachclass));
    temp_dev_Sw_store = zeros(size(X{1},1),size(X{1},1),length(num_eachclass));
    for k = 1 : length(num_eachclass)
        Nw(j) = Nw(j) + 1;
        Xj = temp_X{num_eachclass(k)};
        temp = logm(Xi) - logm(Xj);
        temp_dev_Sw_store(:,:,k) = Xi\temp;
        Sw_temp(k) = norm(temp, 'fro') * norm(temp, 'fro'); 
    end
    [~,idx] = sort(Sw_temp);
    if (length(num_eachclass) < K1)
        K1 = length(num_eachclass);
    end
    dist_temp = Sw_temp(idx(:,1:K1)); % get the first K1 smallest distance
    Sw(j) = sum(dist_temp); % this is the scatter matrix composed by K1 nearest neighbor
    temp_dev_Sw(:,:,j) = sum(temp_dev_Sw_store(:,:,idx(:,1:K1)),3); % it is the third of partial derivative
    if K1 == 1
        Nw(j) = K1 +1;
    else 
        Nw(j) = K1;
    end
end

parfor j = 1:length(X)
    K2 = 3;
    num_difclass=find(c~=c(j)); % find the sample share different labels with X{j}
    temp_X = use_parfor{j};
    Xi = temp_X{j};
    Sb_temp = zeros(1,length(num_difclass));
    temp_dev_Sb_store = zeros(size(X{1},1),size(X{1},1),length(num_difclass));
    for k = 1:length(num_difclass)
        Xj = temp_X{num_difclass(k)};
        Nb(j) = Nb(j) + 1;
        temp = logm(Xi) - logm(Xj); %
        temp_dev_Sb_store(:,:,k) = Xi\temp;
        Sb_temp(k) = norm(temp, 'fro') * norm(temp, 'fro');
    end
    [~,idx] = sort(Sb_temp);
    if (length(num_difclass) < K2)
        K2 = length(num_difclass);
    end
    dist_temp = Sb_temp(idx(:,1:K2)); % get the first K1 smallest distance
    Sb(j) = sum(dist_temp); % this is the scatter matrix composed by K1 nearest neighbor
    temp_dev_Sb(:,:,j) = sum(temp_dev_Sb_store(:,:,idx(:,1:K2)),3); % it is the third of partial derivative
    Nb(j) = K2;
end

alpha = 0.2; % last step is 0.6

temp_scatter = zeros(1,length(X));
Y = cell(length(X), 1);
Y_sum = 0;

td = 1.0;

for m = 1 : length(X)
    
    Sw_each = Sw(m) / (Nw(m)-1);
    Sb_each = Sb(m) / Nb(m);  
    d_inter = Sw_each - Sb_each;
    d_intra = Sw_each;
    %temp_scatter(m) = d_inter + alpha * d_intra;
    %Y_sum = Y_sum + log(1 + exp(temp_scatter(m)));
    if (d_inter > a1 && d_intra > a2)
        temp_scatter(m) = a1 + alpha * a2;
        Y_sum = Y_sum + log(1 + exp(temp_scatter(m)));
        if nargin < 5
            Y = lambda * Y_sum;
        else
            Y{m} = 0 + dzdy_recon{m} + td * dzdx_log{m};
        end
    elseif d_inter > a1 && d_intra < a2
        temp_scatter(m) = a1 + alpha * d_intra;
        Y_sum = Y_sum + log(1 + exp(temp_scatter(m)));
        if nargin < 5
            Y = lambda * Y_sum;
        else
            dev_part1 = 1 / (1 + exp(temp_scatter(m)));
            dev_part2 = exp(temp_scatter(m)); % the second part of the dev with respect to xi
            temp_dev_Sw_each = temp_dev_Sw(:,:,m) / (Nw(m) - 1);
            dev_part3 = 2 * alpha * temp_dev_Sw_each;
            dev_l2 = bsxfun(@times, (dev_part1 * dev_part2 * dev_part3), bsxfun(@times, ones(size(X{1},1)), dzdy_l2));
            Y{m} = lambda * dev_l2 + dzdy_recon{m} + td * dzdx_log{m}; % 0.5 for soft + ml
        end
    elseif d_inter < a1 && d_intra > a2
        temp_scatter(m) = d_inter + alpha * a2;
        Y_sum = Y_sum + log(1 + exp(temp_scatter(m)));
        if nargin < 5
            Y = lambda * Y_sum;
        else
            dev_part1 = 1 / (1 + exp(temp_scatter(m)));
            dev_part2 = exp(temp_scatter(m)); % the second part of the dev with respect to xi
            temp_dev_Sw_each = temp_dev_Sw(:,:,m) / (Nw(m) - 1);
            temp_dev_Sb_each = temp_dev_Sb(:,:,m) / Nb(m);
            dev_part3 = 2 * temp_dev_Sw_each - 2 * temp_dev_Sb_each;
            dev_l2 = bsxfun(@times, (dev_part1 * dev_part2 * dev_part3), bsxfun(@times, ones(size(X{1},1)), dzdy_l2));
            Y{m} = lambda * dev_l2 + dzdy_recon{m} + td * dzdx_log{m}; % 0.5 for soft + ml
        end
    else
        temp_scatter(m) = d_inter + alpha * d_intra;
        Y_sum = Y_sum + log(1 + exp(temp_scatter(m)));
        if nargin < 5
            Y = lambda * Y_sum;
        else
            dev_part1 = 1 / (1 + exp(temp_scatter(m)));
            dev_part2 = exp(temp_scatter(m)); % the second part of the dev with respect to xi
            temp_dev_Sw_each = temp_dev_Sw(:,:,m) / (Nw(m) - 1);
            temp_dev_Sb_each = temp_dev_Sb(:,:,m) / Nb(m);
            dev_part3 = 2 * temp_dev_Sw_each - 2 * temp_dev_Sb_each + 2 * alpha * temp_dev_Sw_each;
            dev_l2 = bsxfun(@times, (dev_part1 * dev_part2 * dev_part3), bsxfun(@times, ones(size(X{1},1)), dzdy_l2));
            Y{m} = lambda * dev_l2 + dzdy_recon{m} + td * dzdx_log{m}; % 0.5 for soft + ml\
        end
    end
    
end

% if nargin < 5
%     Y = lambda * Y_sum;
% else 
%     for n = 1 : length(X)
%         dev_part1 = 1 / (1 + exp(temp_scatter(n)));
%         dev_part2 = exp(temp_scatter(n)); % the second part of the dev with respect to xi
%         temp_dev_Sw_each = temp_dev_Sw(:,:,n) / (Nw(m) - 1);
%         temp_dev_Sb_each = temp_dev_Sb(:,:,n) / Nb(m);
%         dev_part3 = 2 * temp_dev_Sw_each - 2 * temp_dev_Sb_each + 2 * alpha * temp_dev_Sw_each;
%         dev_l2 = bsxfun(@times, (dev_part1 * dev_part2 * dev_part3), bsxfun(@times, ones(size(X{1},1)), dzdy_l2));
%         Y{n} = lambda * dev_l2 + dzdy{n}; % 0.5 for soft + ml
%         %Y{n} = lambda * dev_l2 + 0.1 * dzdy{n};
%     end
% end




