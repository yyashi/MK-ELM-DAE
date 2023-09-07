function [beta, X_proj, rec_err] = ELM_DAE(X, label, L, C, lambda, ActF, options)
%X: training data, N * d
%label: training label
%L: the low dimension
%C: the paramter C
%lambda: the paramer lambda
%ActF: the activation function

options.gnd = label;
[N, D] = size(X);%D=d

A = 2 * rand(L, D) - 1;
b = 2 * rand(L, 1) - 1;
b = orth(b);

if L > D
    A = orth(A);
else
    A = orth(A')';
end


tempH = X * A' + repmat(b', N, 1);

switch lower(ActF)
    case {'sig'}
        H = 1 ./ (1 + exp(-tempH));
    case {'tansig'}
        H = tansig(tempH);
    case {'sin','sine'}
        H = sin(tempH);
end

%compute L
W = constructW(X, options);
W = full(W);
D_W = diag(sum(W));
L_W = D_W - W;


AAA = eye(L) + C * (H' * H);
temp = X' * L_W * X;

BBB = lambda * temp;

CCC = C * H' * X;
beta = sylvester(AAA, BBB, CCC);

%project
X_proj = X * beta';

%reconstruction error
X_rec = H * beta;
rec_err = norm(X - X_rec,'fro');