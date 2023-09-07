%DEMO
close
clear
clc
load label.mat%label of the dataset
load AF_all.mat%the 2D AF Plane
load train6_index.mat%the training/test partition
hidNodes = 200;%Q
C = 10 .^ (-2:1:7);%C
lambda = 10 .^ (-2:1:7);%lambda
ActF = 'sig';%g(.)
Cpar = 10^2;%the paramter of KELM
options = [];
options.NeighborMode = 'Supervised';
options.bLDA = 1;
partitions = 10;
nbclass = 30;
ss_opt = 5;%The optimal AF slice
correct_elm_dae = cell(partitions, 1);
for ii = 1:partitions
    fprintf('cycle %d\n',ii);
    train_label = label(ind_train(:,ii));
    trainsample = AF_all{ss_opt}(ind_train(:,ii),:);
    for jjj = 1:size(trainsample, 1)
        trainsample(jjj,:) = trainsample(jjj,:)./norm(trainsample(jjj,:));
    end
    for cc = 1:length(C)
        for la = 1:length(lambda)
            fprintf('training...\n')

            %use ELM-DAE
            [beta_ae, trainsample_proj, rec_err] = ...
                ELM_DAE(trainsample, train_label, hidNodes, C(cc), lambda(la), ActF, options);
            %train KELM
            [alpha, meanpd] = KELM_SY(trainsample_proj, train_label, Cpar);
            
            %%%%testing
            fprintf('testing...\n')
            testlabel = label(ind_test(:,ii));
            testsample = AF_all{ss_opt}(ind_test(:,ii),:);
            for jjj = 1:size(testsample, 1)
                testsample(jjj,:) = testsample(jjj,:)./norm(testsample(jjj,:));
            end
            testsample_proj = testsample * beta_ae';
            Dtetr = EuDist2(testsample_proj,trainsample_proj,0);
            Ktetr = exp(-Dtetr./(meanpd));
            predict_label_coded = Ktetr * alpha;
            [~,predicted_label] = max(predict_label_coded,[],2);
            correct_elm_dae{ii}(cc,la) = mean(testlabel == predicted_label);
        end
    end
end