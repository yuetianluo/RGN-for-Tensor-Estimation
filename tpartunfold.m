% This function is to matricize the selected modes at the new mode 1 of the given tensor.
% From example, if dim(X) = (2,3,4,5), v = (3,4), you get a (12,2,5) tensor.
% In our case, the last dimension is the sample size.
function Y = tpartunfold(X,v)
    p = size(X);
    d = length(p);
    v_hat = [v,setdiff(1:d,v)];
    Y = reshape( permute(X.data, v_hat), [prod(p(v)),p(setdiff(1:d,v))] );
    Y = tensor(Y);
end