close
clear
clc
load label.mat%label of the dataset
load AF_all.mat%the 2D AF Plane
load train6_index.mat%the training/test partition
partitions = 10;
M = length(AF_all);%the number of AF slices in 2D AF plane
nfs = 40;%R=40
hidNodes = 200;%Q=200
C = 10 .^ (-2:1:7);
lambda = 10 .^ (-2:1:7);
ActF = 'sig';%the activation function
tt = 1;
Cpar = 10^2;%the paramter of KELM
nbclass = 30;
options = [];
options.NeighborMode = 'Supervised';
options.bLDA = 1;
Copt = 10^7;%the optimal C
lambdaopt = 10^6;%the optimal lambda
for ii = 1:partitions
    fprintf('cycle %d\n',ii);
    trainlabel = label(ind_train(:,ii));
    LDR = zeros(M,1);
    for j = 1:M
        trainsample = AF_all{j}(ind_train(:,ii),:);
        for jjj = 1:size(trainsample, 1)
            trainsample(jjj,:) = trainsample(jjj,:)./norm(trainsample(jjj,:));
        end
        [trace_sb,trace_sw] = Fisher_ratio_linear(trainsample,trainlabel);
        LDR(j) = trace_sb/trace_sw;
    end
    %ranking
    [LDR_sorted, ki] = sort(LDR,'descend');
    %keep the first largest R
    nfs_select = ki(1:nfs);
    delta = LDR_sorted(1:nfs)./sum(LDR_sorted(1:nfs));%theta used in paper
    
    ntr = length(trainlabel);
    T = zeros(ntr,nbclass);
    for j = 1:nbclass
        T(trainlabel==j,j) = 1;
    end
    
    
    Ktrtr_comb = 0;
    beta_ae = cell(nfs, 1);
    trainsample_proj = cell(nfs, 1);
    meanpd = zeros(nfs, 1);
    for j = 1:nfs
        trainsample = AF_all{nfs_select(j)}(ind_train(:,ii),:);
        for jjj = 1:size(trainsample, 1)
            trainsample(jjj,:) = trainsample(jjj,:)./norm(trainsample(jjj,:));
        end
        %feature learning stage
        [beta_ae{j}, trainsample_proj{j}, rec_err] = ...
            ELM_DAE(trainsample, trainlabel, hidNodes, Copt, lambdaopt, ActF, options);
        Dtrtr = EuDist2(trainsample_proj{j},trainsample_proj{j},0);
        meanpd(j) = sum(sum(Dtrtr))/2/(0.5*ntr*(ntr-1));
        Ktrtr = exp(-Dtrtr./(meanpd(j)));
        Ktrtr_comb = Ktrtr_comb + delta(j) * Ktrtr;%feature fusion stage
        clear Ktrtr
    end
    alpha = (Ktrtr_comb + 1/Cpar * eye(ntr))\T;
    
    %testing
    testlabel = label(ind_test(:,ii));
    nte = length(testlabel);
    for ttt = 1:nte
        Ktetr_comb = 0;
        for j = 1:nfs
            testsample = AF_all{nfs_select(j)}(ind_test(ttt,ii),:);
            for jjj = 1:size(testsample, 1)
                testsample(jjj,:) = testsample(jjj,:)./norm(testsample(jjj,:));
            end
            testsample_proj = testsample * beta_ae{j}';
            Dtetr = EuDist2(testsample_proj,trainsample_proj{j},0);
            Ktetr = exp(-Dtetr./(meanpd(j)));
            Ktetr_comb = Ktetr_comb + delta(j) * Ktetr;
            clear Ktetr
        end
        predict_label_coded = Ktetr_comb * alpha;
        [~,predicted_label(ttt,1)] = max(predict_label_coded,[],2);
    end
    correct_mk_elm_dae(ii) = mean(testlabel == predicted_label);
end