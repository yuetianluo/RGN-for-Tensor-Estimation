# RGN for tensor decomposition
library(rTensor)
library(matrixcalc)
library(MASS)
library(parallel)
source('RGN_Funs.R')
set.seed(2020)
final_result <- vector()
d <- 3
p = c(100); r = c(3); sig = c(1e-10);
lambda <- 3*p^(3/4) ;
t_max = 30; retrac <- 'ortho'
succ_tol <- 1e-13
p1 = p2 = p3 = p
r1 = r2 = r3 = r
S.array = array(rnorm(r1*r2*r3), dim=c(r1,r2,r3)) # Core tensor
S = as.tensor(S.array)
Sk = k_unfold(S, 1)@data
lambda.min = svd(Sk)$d[r]

for (i in 1:d){
  Sk = k_unfold(S, i)@data
  lambda.min = min(lambda.min, svd(Sk)$d[r])
}
S = S * lambda / lambda.min


E1 = matrix(rnorm(p1*r1), nrow = p1, ncol=r1)
U1 <- qr.Q(qr(E1))
E2 = matrix(rnorm(p2*r2), nrow = p2, ncol=r2)
U2 <- qr.Q(qr(E2))
E3 = matrix(rnorm(p3*r3), nrow = p3, ncol=r3)
U3 <- qr.Q(qr(E3))

X = ttm(ttm(ttm(S, U1, 1), U2, 2), U3, 3) 
U <- list(U1, U2, U3)
eps = sig * as.tensor(array(rnorm(p1*p2*p3), dim=c(p1,p2,p3)))

Y <- X + eps
## Initialization

## hosvd initialization
hosvd_result = hosvd(Y, c(r1,r2,r3)) 
hatU = hosvd_result$U; hatU1 = hatU[[1]]; hatU2 = hatU[[2]]; hatU3 = hatU[[3]];
Z = hosvd_result$Z
Xt <- ttl(hosvd_result$Z,hatU,c(1:3))
error_matrix <- RGN_tendecomposition(hatU1,hatU2,hatU3,Xt,Y,X,r1, r2, r3, p1, p2, p3, t_max, succ_tol, retrac)
error_matrix


