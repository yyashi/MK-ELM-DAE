function [alpha, meanpd] = KELM_SY(trainsample, trainlabel, Cpar)
%preparing the kernel matrix
ntr = size(trainsample,1);
Dtrtr = EuDist2(trainsample,trainsample,0);
meanpd = sum(sum(Dtrtr))/2/(0.5*ntr*(ntr-1));
Ktrtr = exp(-Dtrtr./(meanpd));
Ktrtr = max(Ktrtr,Ktrtr');

nbclass = length(unique(trainlabel));

T = zeros(ntr,nbclass);
for j = 1:nbclass
    T(trainlabel==j,j) = 1;
end
alpha = (Ktrtr + 1/Cpar * eye(ntr))\T;