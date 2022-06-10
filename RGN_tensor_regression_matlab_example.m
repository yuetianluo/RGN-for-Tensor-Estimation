% This is the example code of RGN for tensor regression in matlab.
clear;
rng('default');
rng(2017);
p = 30;
r = 3;
n = floor(p^(3/2)) * r * 5;
sig = 0;
iter_max = 20;
tol = 1e-14;
exp_time = 4;
retra_type = 'orthogra';
init_type = 'spec';
succ_tol = 1e-14;
p1 = p; p2 = p; p3 = p;
r1 = r; r2 = r; r3 = r;
S = tensor(randn(r1, r2, r3));
E1 = randn(p1, r1);
E2 = randn(p2, r2);
E3 = randn(p3, r3);
[U1,~,~] = svds(E1,r1);
[U2,~,~] = svds(E2,r2);
[U3,~,~] = svds(E3,r3);
A = tensor(randn(p1, p2, p3,n));
U = {U1, U2, U3};
X = ttm(S, U, [1:3]);
eps = sig * randn(n,1);
A_mat = tenmat(A, 4);
y = (A_mat.data) * X(:) + eps;


if strcmp(init_type,'spec')
    W = reshape(tensor(y' * A_mat),[p1, p2, p3])/n;
    Xt = hosvd(W,norm(W),'ranks',[r1,r2,r3],'sequential',true,'verbosity',0);
    Ut = Xt.u;
    St = Xt.core;
elseif strcmp(init_type,'rand')   
% random initialization
    U0 = randn(p1,r1);
    [U0,~,~] = svds(U0, r1);
    V0 = randn(p2, r2);
    [V0,~,~] = svds(V0, r2);
    Z0 = randn(p3, r3);
    [Z0,~,~] = svds(Z0, r3);
    St =  tensor(randn(r1, r2, r3));
    Ut = {U0,V0,Z0};
    Xt = ttm(St, Ut,1:3);
end

error_matrix = RGN_tensor_regression(Ut, Xt, y, A,X, r1,r2,r3, p1, p2, p3, iter_max, succ_tol, retra_type);
array2table(error_matrix)


