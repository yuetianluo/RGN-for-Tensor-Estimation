% Input: U0: initialization for three loadings; X0: initialization for X;
% y: observation; A: p1-by-p2-by-p3-by-n sensing tensor; X: true parameter tensor; 
% r1,r2,r3: input rank; p1, p2, p3: dimensions;
% iter_max: the maximum number of iterations;
% succ_tol: tolerence for successful recovery/stoping criteria
% retrac: two choices of retractions: 'orthogra' (orthographic retraction); 'hosvd' (HOSVD-based retraction)
% Return the error matrix which has three columns. They are iteration number, estimation error and runing time.

function [ error_matrix ] = RGN_tensor_regression( U0, X0, y, A,X, r1,r2,r3, p1, p2, p3, iter_max, succ_tol, retra_type)
Xt = X0;
Ut = U0;
St = ttm(Xt,{Ut{1}', Ut{2}' Ut{3}'});
Ut_perp = cell(3,1);
Vt = cell(3,1);
for i = 1:3
    Ut_perp{i} = null(Ut{i}');
    [Vt{i},~] = qr(double(tenmat(St,i))',0);
end
Xt_err = norm(tensor(Xt) - X)/norm(X);

error_matrix = [0, Xt_err, 0];
tic;
for iter = 1:iter_max
    myU1 = ttm(A,Ut{1}',1); 
    myU2 = ttm(A,Ut{2}',2);
    U1U2 = ttm(myU1,Ut{2}',2);
    U1U3 = ttm(myU1,Ut{3}',3);
    U2U3 = ttm(myU2,Ut{3}',3);
    tildeB = tenmat(ttm(U1U2,Ut{3}',3),4);
    tildeD1tm = ttm(tpartunfold(ttm(U2U3,Ut_perp{1}',1),[2,3]),Vt{1}',1);
    tildeD2tm = ttm(tpartunfold(ttm(U1U3,Ut_perp{2}',2),[1,3]),Vt{2}',1);
    tildeD3tm = ttm(tpartunfold(ttm(U1U2,Ut_perp{3}',3),[1,2]),Vt{3}',1);
    tildeD1 = tenmat(tildeD1tm,[3],[2,1]);
    tildeD2 = tenmat(tildeD2tm,[3],[2,1]);
    tildeD3 = tenmat(tildeD3tm,[3],[2,1]);
    tildeA = [tildeB.data, tildeD1.data, tildeD2.data, tildeD3.data];
    % I found doing least squares in matlab is much faster than R.
    gamma = (tildeA' * tildeA) \ tildeA' * y;
    %[gamma, flag] = lsqr(tildeA, y);
    hatB = reshape(gamma(1:r1*r2*r3),[r1,r2,r3]);
    hatD1 = reshape(gamma(r1*r2*r3+1: r1*r2*r3+r1*(p1-r1) ), [p1-r1, r1] );
    hatD2 = reshape(gamma( r1*r2*r3+r1*(p1-r1)+1 : r1*r2*r3+r1*(p1-r1) + (p2-r2)*r2 ), [p2-r2, r2]);
    hatD3 = reshape(gamma( r1*r2*r3+r1*(p1-r1) + (p2-r2)*r2+1 : r1*r2*r3+r1*(p1-r1) + (p2-r2)*r2 + (p3-r3)*r3 ), [p3-r3, r3]);
    if strcmp(retra_type,'orthogra')  
        % Newton step on the manifold
        hatL1 = Ut{1} + Ut_perp{1}* hatD1 / double(tenmat(hatB,1) * Vt{1});
        hatL2 = Ut{2} + Ut_perp{2}* hatD2 / double(tenmat(hatB,2) * Vt{2});
        hatL3 = Ut{3} + Ut_perp{3}* hatD3 / double(tenmat(hatB,3) * Vt{3});
        % retraction step
        Xt = ttm(tensor(hatB), {hatL1; hatL2; hatL3},[1:3]);
        [Ut{1},~] = qr(hatL1,0);
        [Ut{2},~] = qr(hatL2,0);
        [Ut{3},~] = qr(hatL3,0);
        St = ttm(Xt,{Ut{1}', Ut{2}' Ut{3}'});
    elseif strcmp(retra_type,'hosvd')
        tildeXt = ttm(tensor(hatB),Ut,1:3) + tensor( Ut_perp{1} * hatD1 * ( kron(Ut{3},Ut{2}) * Vt{1} )', [p1,p2,p3]) +  permute(tensor( Ut_perp{2} * hatD2 * ( kron(Ut{3},Ut{1}) * Vt{2} )', [p2,p1,p3]),[2,1,3]) +  permute(tensor( Ut_perp{3} * hatD3 * ( kron(Ut{2},Ut{1}) * Vt{3} )', [p3,p1,p2]),[2,3,1]);
        Xt = hosvd(tildeXt,norm(tildeXt),'ranks',[r1,r2,r3],'sequential',true,'verbosity',0);
        Ut = Xt.u;
        St = Xt.core;
    end
    for i = 1:3
        Ut_perp{i} = null(Ut{i}');
        [Vt{i},~] = qr(double(tenmat(St,i))',0);
    end
    Xt_err = norm(tensor(Xt) - X)/norm(X);
    time = toc;
    iter_result = [iter, Xt_err, time];
    error_matrix = vertcat(error_matrix, iter_result);
    if Xt_err < succ_tol || Xt_err > 50
            break
    end
end
end
