function [trace_sb,trace_sw] = Fisher_ratio_linear(trainsample,trainlabel)
[n,~]=size(trainsample);
classlabel=unique(trainlabel);
nclass=length(classlabel);

totalmean= mean(trainsample);
tmp = 0;
for i = 1:nclass
    index = find(trainlabel==classlabel(i));
    classmean = mean(trainsample(index, :));
    tmp = tmp + length(index)*(classmean*classmean');
end
trace_sw = trace(trainsample*trainsample') - tmp;
trace_sb = tmp - n*(totalmean*totalmean');