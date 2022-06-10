########## Tensor estimation via rank-1 projection setting ###########
library(rTensor)
library(matrixcalc)
library(MASS)
library(parallel)
source('RGN_Funs.R')

set.seed(2020)
final_result <- vector()
p = c(30); r = c(3); sig = c(0);
n <- p^(3/2) * r * 8;
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
A = as.tensor(array(rnorm(n*p1*r), dim=c(n,p1,r)))

X = ttm(ttm(ttm(S, U1, 1), U2, 2), U3, 3) 
eps = sig * rnorm(n) # Error
y <- vector(mode = 'numeric', length = n)
for (i in 1:n){
  y[i] <- ttl(X, list(t(A[i,,1]@data), t(A[i,,2]@data), t(A[i,,3]@data)), c(1:3))@data + eps[i]
}

## Initialization
W <- as.tensor(array(0, dim = c(p1,p2,p3)))
for (i in 1:n){
  W <- W + y[i] * outer(outer(A[i,,1]@data,A[i,,2]@data),A[i,,3]@data) 
}
W = W/n
HOOI_result = tucker(W, c(r1,r2,r3)) # HOOI finds the probing directions
hatU = HOOI_result$U; hatU1 = hatU[[1]]; hatU2 = hatU[[2]]; hatU3 = hatU[[3]];
Xt <- ttl(HOOI_result$Z,hatU,c(1:3))

## RGN in the rank-1 sensing problem.
error_matrix <- RGN_rankone(hatU1,hatU2,hatU3,Xt,y,A,X,r1, r2, r3, p1, p2, p3, t_max, succ_tol, retrac)
error_matrix
