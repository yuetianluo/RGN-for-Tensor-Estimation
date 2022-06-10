# RGN for tensor completion
library(rTensor)
library(matrixcalc)
library(MASS)
library(parallel)
source('RGN_Funs.R')
set.seed(2020)
final_result <- vector()
p = c(30); r = c(3); sig = c(0);
n <- p^(3/2) * r * 6;
t_max = 30; retrac <- 'hosvd'
succ_tol <- 1e-13
p1 = p2 = p3 = p
r1 = r2 = r3 = r
S.array = array(rnorm(r1*r2*r3), dim=c(r1,r2,r3)) # Core tensor
S = as.tensor(S.array)
E1 = matrix(rnorm(p1*r1), nrow = p1, ncol=r1)
U1 <- qr.Q(qr(E1))
E2 = matrix(rnorm(p2*r2), nrow = p2, ncol=r2)
U2 <- qr.Q(qr(E2))
E3 = matrix(rnorm(p3*r3), nrow = p3, ncol=r3)
U3 <- qr.Q(qr(E3))

X = ttm(ttm(ttm(S, U1, 1), U2, 2), U3, 3) 
U <- list(U1, U2, U3)
indices <- expand.grid(1:p1, 1:p2, 1:p3)
index_select <- sample(1:(p1*p2*p3), n, replace = T)
all_index <- indices[index_select,]
obser_X <- as.tensor(array(0, dim = c(p1, p2, p3)))
y <- vector('numeric', n)
eps = sig * rnorm(n)

for (i in 1:n){
  y[i] <- X@data[all_index[i,1], all_index[i,2], all_index[i,3]] + eps[i]
  obser_X@data[all_index[i,1], all_index[i,2], all_index[i,3]] <- obser_X@data[all_index[i,1], all_index[i,2], all_index[i,3]] + y[i]
}
X_initial <- as.tensor(obser_X@data * (p1 * p2 * p3)/n)
hatU <- spectral_initia(all_index, c(p1,p2,p3), c(r1,r2,r3), obser_X, n, y)
hatU1 = hatU[[1]]; hatU2 = hatU[[2]]; hatU3 = hatU[[3]];
Z <- ttl(X_initial, list(t(hatU[[1]]), t(hatU[[2]]), t(hatU[[3]]) ), c(1:3))
Xt <- ttl(Z,hatU,c(1:3))

## RGN in the tensor completion problem.
error_matrix <- RGN_tencompletion(hatU1,hatU2,hatU3,Xt,y,all_index,X,r1, r2, r3, p1, p2, p3, t_max, succ_tol, retrac)
error_matrix
