# Low-rank tensor estimation via Riemannian Gauss-Newton
library(rTensor)
library(matrixcalc)
library(MASS)




##################################################
# Main functions
##################################################
regression.product = function(X, A){
  # X: order-(d+1) tensor, first mode is of dimension n 
  # A: order-d tensor, the dimensions matches all but the first dimensions of X
  y = k_unfold(X, 1)@data %*% as.vector(A@data)
  return(y)
}  

tpartunfold = function(X, v){
  # unfold the modes in V for X
  # Example: X: 5*3*4*10 tensor; tpartunfold(X, c(2,3)) yields 5*10*12 tensor
  p = dim(X)
  d = length(p)
  v_hat = c(setdiff(1:d, v), v)
  Y = array(aperm(X@data, v_hat), dim = c(p[setdiff(1:d, v)], prod(p[v])))
  return(as.tensor(Y))
}


####################################################################
# Sine-theta distance between two singular vector spaces U and hatU
# Default: Frobenius sine-theta distance
####################################################################
sine.theta <- function(U, hatU, q){ # Sine-theta distance between two singular subspaces. 
  # U and hatU should be of the same dimensions
  try(if(missing("q")) q = 2) 
  try(if(nrow(U)!=nrow(hatU) | ncol(U)!=ncol(hatU)) stop("Matrix does not match") )
  
  r = ncol(U)
  v = 1 - (svd(t(U) %*% hatU)$d)^2
  if(is.infinite(q))
    return(max(v))
  else
    return((sum(v^q))^(1/q))
}

sketch_method <- function(myU1, myU2 , hatU1, hatU2, hatU3, hatU1_perp, hatU2_perp, hatU3_perp,hatV1, hatV2, hatV3, r1, r2, r3, p1, p2, p3, y){
  ### INPUT ###
  # myU1, myU2: used to save computation
  # hatU1, hatU2, hatU3: probing direction
  # hatU1_perp, hatU2_perp, hatU3_perp: orthogonal complement of the corresponding part
  # hatV1, hatV2, hatV3: second sketching direction
  # r1, r2, r3: input rank
  # p1, p2, p3: input dimension
  # y: response
  ### OUTPUT ###
  # estimate of core tensor coefficient part and the arm coefficient part
  
  U2U3 <- ttm(myU2, t(hatU3),4)
  U1U3 <- ttm(myU1, t(hatU3), 4)
  U1U2 <- ttm(myU1, t(hatU2), 3)
  
  # calculate important covariates
  tildeA_B <- as.matrix(k_unfold(ttm(U2U3, t(hatU1), 2), 1)@data)
  tildeA_D1_intermediate <- ttm(tpartunfold(ttm(U2U3, t(hatU1_perp), 2), c(3,4)), t(hatV1), 3)
  tildeA_D2_intermediate = ttm(tpartunfold(ttm(U1U3, t(hatU2_perp), 3), c(2,4)), t(hatV2), 3)
  tildeA_D3_intermediate = ttm(tpartunfold(ttm(U1U2, t(hatU3_perp), 4), c(2,3)), t(hatV3), 3) 
  tildeA_D1 = k_unfold(tildeA_D1_intermediate, 1)@data
  tildeA_D2 = k_unfold(tildeA_D2_intermediate, 1)@data
  tildeA_D3 = k_unfold(tildeA_D3_intermediate, 1)@data
  
  # calculate the least square estimates for the combined coefficient
  tildeA = cbind(tildeA_B, tildeA_D1, tildeA_D2, tildeA_D3)
  hat_estimate_combine = solve((t(tildeA)%*%tildeA), (t(tildeA) %*% y))
  hatB = as.tensor(array(hat_estimate_combine[1:(r1*r2*r3)], c(r1, r2, r3)))
  # get hatD1, hatD2, hatD3
  hatD1 = matrix(hat_estimate_combine[(r1*r2*r3+1):(r1*r2*r3+(p1-r1)*r1)], nrow=p1-r1, ncol=r1)
  hatD2 = matrix(hat_estimate_combine[(r1*r2*r3+(p1-r1)*r1+1):(r1*r2*r3+(p1-r1)*r1+(p2-r2)*r2)], nrow=p2-r2, ncol=r2)
  hatD3 = matrix(hat_estimate_combine[(r1*r2*r3+(p1-r1)*r1+(p2-r2)*r2+1):
                                               (r1*r2*r3+(p1-r1)*r1+(p2-r2)*r2+(p3-r3)*r3)], nrow=p3-r3, ncol=r3)
  result <- list(hatB = hatB, hatD1 = hatD1, hatD2=hatD2, hatD3=hatD3)
  return(result)
}

##################################################
# Main RGN FUNCTIONS
##################################################

## RGN for order-3 tensor regression
# Input: U1_ini, U2_ini, U3_ini: initialization for three loadings; X_ini: initialization for X;
# y: observation; A: n-by-p1-by-p2-by-p3 sensing tensor; X: true parameter tensor; 
# r1,r2,r3: input rank; p1, p2, p3: dimensions;
# t_max: the maximum number of iterations;
# succ_tol: tolerence for successful recovery/stoping criteria
# retrac: two choices of retractions: 'ortho' (orthographic retraction); 'hosvd' (HOSVD-based retraction)
# Return the error matrix which has three columns. They are iteration number, estimation error and runing time.
RGN_tenreg <- function(U1_ini, U2_ini, U3_ini, X_ini, y, A, X, r1, r2, r3, p1, p2, p3, t_max, succ_tol, retrac){
  U1t <- U1_ini
  U2t <- U2_ini
  U3t <- U3_ini
  Xt <- X_ini
  U1t_perp = qr.Q(qr(U1t),complete=TRUE)[,(r1+1):p1]
  U2t_perp = qr.Q(qr(U2t),complete=TRUE)[,(r2+1):p2]
  U3t_perp = qr.Q(qr(U3t),complete=TRUE)[,(r3+1):p3]
  Zt <- ttl(Xt, list(t(U1t),t(U2t),t(U3t)),c(1:3))
  V1t = qr.Q(qr(t(k_unfold(Zt, 1)@data))) 
  V2t = qr.Q(qr(t(k_unfold(Zt, 2)@data))) 
  V3t = qr.Q(qr(t(k_unfold(Zt, 3)@data))) 
  X_rela_err <- fnorm(Xt - X)/fnorm(X) 
  error_matrix = c(0,X_rela_err,0)
  t <- proc.time()
  for (i in 1:t_max){
    # to save the computation
    myU1 <- ttm(A, t(U1t), 2)
    myU2 <- ttm(A, t(U2t), 3)
    LS <- sketch_method(myU1, myU2, U1t, U2t, U3t, U1t_perp, U2t_perp, U3t_perp, V1t, V2t, V3t, r1, r2, r3, p1, p2, p3, y)
    hatB <- LS$hatB
    hatD1 = LS$hatD1
    hatD2 = LS$hatD2
    hatD3 = LS$hatD3
    if (retrac == 'ortho'){
      hatD1_alt = U1t %*% (k_unfold(hatB, 1)@data) %*% V1t + U1t_perp %*% hatD1
      hatD2_alt = U2t %*% (k_unfold(hatB, 2)@data) %*% V2t + U2t_perp %*% hatD2
      hatD3_alt = U3t %*% (k_unfold(hatB, 3)@data) %*% V3t + U3t_perp %*% hatD3
      # final estimate
      hatL1 <- hatD1_alt %*% solve(k_unfold(hatB, 1)@data %*% V1t)
      hatL2 <- hatD2_alt %*% solve(k_unfold(hatB, 2)@data %*% V2t)
      hatL3 <- hatD3_alt %*% solve(k_unfold(hatB, 3)@data %*% V3t)
      Xt = ttl(hatB, list(hatL1, hatL2, hatL3), c(1,2,3))
      U1t <- qr.Q(qr(hatL1))
      U2t <- qr.Q(qr(hatL2))
      U3t <- qr.Q(qr(hatL3))      
    } else if( retrac == 'hosvd' ){
      ## The k_unfold operation in R is U_d \otimes \cdots U_{i+1} \otimes U_{i-1} \otimes cdots
      tildeXt <- ttl(hatB, list(U1t,U2t,U3t), c(1:3)) + k_fold(U1t_perp %*% hatD1 %*% t(kronecker(U3t,U2t) %*% V1t),1,modes = c(p1, p2, p3)) +
        k_fold(U2t_perp %*% hatD2 %*% t(kronecker(U3t,U1t) %*% V2t),2,modes = c(p1, p2, p3)) + k_fold(U3t_perp %*% hatD3 %*% t(kronecker(U2t,U1t) %*% V3t),3, modes = c(p1, p2, p3))
      hosvd_result <- hosvd(tildeXt, c(r1, r2, r3))
      Xt <- hosvd_result$est
      Ut <- hosvd_result$U
      U1t <- Ut[[1]]
      U2t <- Ut[[2]]
      U3t <- Ut[[3]]
    }
    U1t_perp = qr.Q(qr(U1t),complete=TRUE)[,(r1+1):p1]
    U2t_perp = qr.Q(qr(U2t),complete=TRUE)[,(r2+1):p2]
    U3t_perp = qr.Q(qr(U3t),complete=TRUE)[,(r3+1):p3]
    Zt <- ttl(Xt, list(t(U1t),t(U2t),t(U3t)),c(1:3))
    V1t = qr.Q(qr(t(k_unfold(Zt, 1)@data))) 
    V2t = qr.Q(qr(t(k_unfold(Zt, 2)@data))) 
    V3t = qr.Q(qr(t(k_unfold(Zt, 3)@data))) 
    X_rela_err <- fnorm(Xt - X)/fnorm(X)
    error_matrix <- rbind(error_matrix,c(i, X_rela_err,(proc.time()-t)[3]))
    if (X_rela_err < succ_tol || X_rela_err > 50){
      break
    }
  }
  return(error_matrix)
}

## RGN procedure for rank-1 projection
# Input: U1_ini, U2_ini, U3_ini: initialization for three loadings; X_ini: initialization for X;
# y: observation; A: n-by-p1-by-p2-by-p3 sensing tensor; X: true parameter tensor; 
# r1,r2,r3: input rank; p1, p2, p3: dimensions;
# t_max: the maximum number of iterations;
# succ_tol: tolerence for successful recovery/stoping criteria
# retrac: two choices of retractions: 'ortho' (orthographic retraction); 'hosvd' (HOSVD-based retraction)
# Return the error matrix which has three columns. They are iteration number, estimation error and runing time.
RGN_rankone <- function(U1_ini, U2_ini, U3_ini, X_ini, y, A, X, r1, r2, r3, p1, p2, p3, t_max, succ_tol, retrac){
  U1t <- U1_ini
  U2t <- U2_ini
  U3t <- U3_ini
  Xt <- X_ini
  U1t_perp = qr.Q(qr(U1t),complete=TRUE)[,(r1+1):p1]
  U2t_perp = qr.Q(qr(U2t),complete=TRUE)[,(r2+1):p2]
  U3t_perp = qr.Q(qr(U3t),complete=TRUE)[,(r3+1):p3]
  Zt <- ttl(Xt, list(t(U1t),t(U2t),t(U3t)),c(1:3))
  V1t = qr.Q(qr(t(k_unfold(Zt, 1)@data))) 
  V2t = qr.Q(qr(t(k_unfold(Zt, 2)@data))) 
  V3t = qr.Q(qr(t(k_unfold(Zt, 3)@data))) 
  X_rela_err <- fnorm(Xt - X)/fnorm(X) 
  error_matrix = c(0,X_rela_err,0)
  m <- r1*r2*r3 + r1*(p1-r1) + r2*(p2-r2) + r3*(p3-r3)
  t <- proc.time()
  for (iter in 1:t_max){
    # to save the computation
    tildeX <- array(0, dim = c(n, m))
    for (i in 1:n){
      tildeB <- outer(outer( t(U1t) %*% A[i,,1]@data, t(U2t) %*% A[i,,2]@data ), t(U3t) %*% A[i,,3]@data )
      tildeD1 <- t(U1t_perp) %*% A[i,,1]@data %*% ( t(kronecker( t(U3t) %*% A[i,,3]@data, t(U2t) %*% A[i,,2]@data ))  %*% V1t  )
      tildeD2 <- t(U2t_perp) %*% A[i,,2]@data %*% ( t(kronecker( t(U3t) %*% A[i,,3]@data, t(U1t) %*% A[i,,1]@data ))  %*% V2t  )
      tildeD3 <- t(U3t_perp) %*% A[i,,3]@data %*% ( t(kronecker( t(U2t) %*% A[i,,2]@data, t(U1t) %*% A[i,,1]@data ))  %*% V3t  )
      tildeX[i,] <- c(as.vector(tildeB), as.vector(tildeD1), as.vector(tildeD2), as.vector(tildeD3))
    }
    hat_estimate_combine = solve((t(tildeX)%*%tildeX), (t(tildeX) %*% y))
    hatB = as.tensor(array(hat_estimate_combine[1:(r1*r2*r3)], c(r1, r2, r3)))
    hatD1 = matrix(hat_estimate_combine[(r1*r2*r3+1):(r1*r2*r3+(p1-r1)*r1)], nrow=p1-r1, ncol=r1)
    hatD2 = matrix(hat_estimate_combine[(r1*r2*r3+(p1-r1)*r1+1):(r1*r2*r3+(p1-r1)*r1+(p2-r2)*r2)], nrow=p2-r2, ncol=r2)
    hatD3 = matrix(hat_estimate_combine[(r1*r2*r3+(p1-r1)*r1+(p2-r2)*r2+1):
                                          (r1*r2*r3+(p1-r1)*r1+(p2-r2)*r2+(p3-r3)*r3)], nrow=p3-r3, ncol=r3)
    
    if (retrac == 'ortho'){
      hatD1_alt = U1t %*% (k_unfold(hatB, 1)@data) %*% V1t + U1t_perp %*% hatD1
      hatD2_alt = U2t %*% (k_unfold(hatB, 2)@data) %*% V2t + U2t_perp %*% hatD2
      hatD3_alt = U3t %*% (k_unfold(hatB, 3)@data) %*% V3t + U3t_perp %*% hatD3
      # final estimate
      hatL1 <- hatD1_alt %*% solve(k_unfold(hatB, 1)@data %*% V1t)
      hatL2 <- hatD2_alt %*% solve(k_unfold(hatB, 2)@data %*% V2t)
      hatL3 <- hatD3_alt %*% solve(k_unfold(hatB, 3)@data %*% V3t)
      Xt = ttl(hatB, list(hatL1, hatL2, hatL3), c(1,2,3))
      U1t <- qr.Q(qr(hatL1))
      U2t <- qr.Q(qr(hatL2))
      U3t <- qr.Q(qr(hatL3))      
    } else if( retrac == 'hosvd' ){
      ## The k_unfold operation in R is U_d \otimes \cdots U_{i+1} \otimes U_{i-1} \otimes cdots
      tildeXt <- ttl(hatB, list(U1t,U2t,U3t), c(1:3)) + k_fold(U1t_perp %*% hatD1 %*% t(kronecker(U3t,U2t) %*% V1t),1,modes = c(p1, p2, p3)) +
        k_fold(U2t_perp %*% hatD2 %*% t(kronecker(U3t,U1t) %*% V2t),2,modes = c(p1, p2, p3)) + k_fold(U3t_perp %*% hatD3 %*% t(kronecker(U2t,U1t) %*% V3t),3, modes = c(p1, p2, p3))
      hosvd_result <- hosvd(tildeXt, c(r1, r2, r3))
      Xt <- hosvd_result$est
      Ut <- hosvd_result$U
      U1t <- Ut[[1]]
      U2t <- Ut[[2]]
      U3t <- Ut[[3]]
    }
    U1t_perp = qr.Q(qr(U1t),complete=TRUE)[,(r1+1):p1]
    U2t_perp = qr.Q(qr(U2t),complete=TRUE)[,(r2+1):p2]
    U3t_perp = qr.Q(qr(U3t),complete=TRUE)[,(r3+1):p3]
    Zt <- ttl(Xt, list(t(U1t),t(U2t),t(U3t)),c(1:3))
    V1t = qr.Q(qr(t(k_unfold(Zt, 1)@data))) 
    V2t = qr.Q(qr(t(k_unfold(Zt, 2)@data))) 
    V3t = qr.Q(qr(t(k_unfold(Zt, 3)@data))) 
    U1_err <- sine.theta(U1t, U1, Inf)
    U2_err <- sine.theta(U2t, U2, Inf)
    U3_err <- sine.theta(U3t, U3, Inf)
    X_rela_err <- fnorm(Xt - X)/fnorm(X)
    error_matrix <- rbind(error_matrix,c(iter,X_rela_err,(proc.time()-t)[3]))
    if (X_rela_err < succ_tol || X_rela_err > 50){
      break
    }
  }
  return(error_matrix)
}

## Initialization for tensor completion
# 3-Order Tensor spectral initialization of left singular space.
spectral_initia <- function(all_index, p, r, obser_T,  n, y){
  hatU <- vector('list', 3)
  for (j in 1:3){
    Mj <- k_unfold(obser_T, j)@data # By calculating in this way, the code becomes much faster.
    Mj <- Mj %*% t(Mj)
    for(zz in 1: p[j]){
      Mj[zz, zz] <- 0 # Mj[zz, zz] - sum((y[all_index[,j] == zz])^2)
    }
    hat_N <- (p[1]*p[2]*p[3])^2/(n*(n-1)) * Mj
    hatU[[j]]  <- svd(hat_N)$u[,1:r[j]]
  }
  return(hatU)
}

## RGN in order-3 tensor completion
# Input: U1_ini, U2_ini, U3_ini: initialization for three loadings; X_ini: initialization for X;
# y: observation; index: observed indices set, it is a matrix with three columns and each row represent the (x,y,z) indices of the observed entries; X: true parameter tensor; 
# r1,r2,r3: input rank; p1, p2, p3: dimensions;
# t_max: the maximum number of iterations;
# succ_tol: tolerence for successful recovery/stoping criteria
# retrac: two choices of retractions: 'ortho' (orthographic retraction); 'hosvd' (HOSVD-based retraction)
# Return the error matrix which has three columns. They are iteration number, estimation error and runing time.
RGN_tencompletion <- function(U1_ini, U2_ini, U3_ini, X_ini, y, index, X, r1, r2, r3, p1, p2, p3, t_max, succ_tol, retrac){
  U1t <- U1_ini
  U2t <- U2_ini
  U3t <- U3_ini
  Xt <- X_ini
  U1t_perp = qr.Q(qr(U1t),complete=TRUE)[,(r1+1):p1]
  U2t_perp = qr.Q(qr(U2t),complete=TRUE)[,(r2+1):p2]
  U3t_perp = qr.Q(qr(U3t),complete=TRUE)[,(r3+1):p3]
  Zt <- ttl(Xt, list(t(U1t),t(U2t),t(U3t)),c(1:3))
  V1t = qr.Q(qr(t(k_unfold(Zt, 1)@data))) 
  V2t = qr.Q(qr(t(k_unfold(Zt, 2)@data))) 
  V3t = qr.Q(qr(t(k_unfold(Zt, 3)@data))) 
  X_rela_err <- fnorm(Xt - X)/fnorm(X) 
  error_matrix = c(0,X_rela_err,0)
  m <- r1*r2*r3 + r1*(p1-r1) + r2*(p2-r2) + r3*(p3-r3)
  t <- proc.time()
  for (iter in 1:t_max){
    # to save the computation
    m <- r1*r2*r3 + r1*(p1-r1) + r2*(p2-r2) + r3*(p3-r3)
    tildeX <- array(0, dim = c(n, m))
    for (i in 1:n){
      ind1 <- all_index[i,1] # Here I simplify the calculation a lot when the sensing tensor X has only one element is 1.
      ind2 <- all_index[i,2]
      ind3 <- all_index[i,3]
      tildeB <- outer(outer( U1t[ind1,] , U2t[ind2,] ), U3t[ind3,] )
      tildeD1 <- U1t_perp[ind1,] %*% ( t(kronecker( U3t[ind3,], U2t[ind2,] ))  %*% V1t  )
      tildeD2 <- U2t_perp[ind2,] %*% ( t(kronecker( U3t[ind3,], U1t[ind1,] ))  %*% V2t  )
      tildeD3 <- U3t_perp[ind3,] %*% ( t(kronecker( U2t[ind2,], U1t[ind1,] ))  %*% V3t  )
      tildeX[i,] <- c(as.vector(tildeB), as.vector(tildeD1), as.vector(tildeD2), as.vector(tildeD3))
    }
    hat_estimate_combine = solve((t(tildeX)%*%tildeX), (t(tildeX) %*% y))
    hatB = as.tensor(array(hat_estimate_combine[1:(r1*r2*r3)], c(r1, r2, r3)))
    hatD1 = matrix(hat_estimate_combine[(r1*r2*r3+1):(r1*r2*r3+(p1-r1)*r1)], nrow=p1-r1, ncol=r1)
    hatD2 = matrix(hat_estimate_combine[(r1*r2*r3+(p1-r1)*r1+1):(r1*r2*r3+(p1-r1)*r1+(p2-r2)*r2)], nrow=p2-r2, ncol=r2)
    hatD3 = matrix(hat_estimate_combine[(r1*r2*r3+(p1-r1)*r1+(p2-r2)*r2+1):
                                          (r1*r2*r3+(p1-r1)*r1+(p2-r2)*r2+(p3-r3)*r3)], nrow=p3-r3, ncol=r3)
    if (retrac == 'ortho'){
      hatD1_alt = U1t %*% (k_unfold(hatB, 1)@data) %*% V1t + U1t_perp %*% hatD1
      hatD2_alt = U2t %*% (k_unfold(hatB, 2)@data) %*% V2t + U2t_perp %*% hatD2
      hatD3_alt = U3t %*% (k_unfold(hatB, 3)@data) %*% V3t + U3t_perp %*% hatD3
      # final estimate
      hatL1 <- hatD1_alt %*% solve(k_unfold(hatB, 1)@data %*% V1t)
      hatL2 <- hatD2_alt %*% solve(k_unfold(hatB, 2)@data %*% V2t)
      hatL3 <- hatD3_alt %*% solve(k_unfold(hatB, 3)@data %*% V3t)
      Xt = ttl(hatB, list(hatL1, hatL2, hatL3), c(1,2,3))
      U1t <- qr.Q(qr(hatL1))
      U2t <- qr.Q(qr(hatL2))
      U3t <- qr.Q(qr(hatL3))      
    } else if( retrac == 'hosvd' ){
      ## The k_unfold operation in R is U_d \otimes \cdots U_{i+1} \otimes U_{i-1} \otimes cdots
      tildeXt <- ttl(hatB, list(U1t,U2t,U3t), c(1:3)) + k_fold(U1t_perp %*% hatD1 %*% t(kronecker(U3t,U2t) %*% V1t),1,modes = c(p1, p2, p3)) +
        k_fold(U2t_perp %*% hatD2 %*% t(kronecker(U3t,U1t) %*% V2t),2,modes = c(p1, p2, p3)) + k_fold(U3t_perp %*% hatD3 %*% t(kronecker(U2t,U1t) %*% V3t),3, modes = c(p1, p2, p3))
      hosvd_result <- hosvd(tildeXt, c(r1, r2, r3))
      Xt <- hosvd_result$est
      Ut <- hosvd_result$U
      U1t <- Ut[[1]]
      U2t <- Ut[[2]]
      U3t <- Ut[[3]]
    }
    U1t_perp = qr.Q(qr(U1t),complete=TRUE)[,(r1+1):p1]
    U2t_perp = qr.Q(qr(U2t),complete=TRUE)[,(r2+1):p2]
    U3t_perp = qr.Q(qr(U3t),complete=TRUE)[,(r3+1):p3]
    Zt <- ttl(Xt, list(t(U1t),t(U2t),t(U3t)),c(1:3))
    V1t = qr.Q(qr(t(k_unfold(Zt, 1)@data))) 
    V2t = qr.Q(qr(t(k_unfold(Zt, 2)@data))) 
    V3t = qr.Q(qr(t(k_unfold(Zt, 3)@data))) 
    X_rela_err <- fnorm(Xt - X)/fnorm(X)
    error_matrix <- rbind(error_matrix,c(iter,X_rela_err,(proc.time()-t)[3]))
    if (X_rela_err < succ_tol || X_rela_err > 50){
      break
    }
  }
  return(error_matrix)
}


## RGN for order-3 tensor decomposition
# Input: U1_ini, U2_ini, U3_ini: initialization for three loadings; X_ini: initialization for X;
# Y: observation tensor; X: true parameter tensor; 
# r1,r2,r3: input rank; p1, p2, p3: dimensions;
# t_max: the maximum number of iterations;
# succ_tol: tolerence for successful recovery/stoping criteria
# retrac: two choices of retractions: 'ortho' (orthographic retraction); 'hosvd' (HOSVD-based retraction)
# Return the error matrix which has three columns. They are iteration number, estimation error and runing time.
RGN_tendecomposition <- function(U1_ini, U2_ini, U3_ini, X_ini, Y, X, r1, r2, r3, p1, p2, p3, t_max, succ_tol, retrac){
  U1t <- U1_ini
  U2t <- U2_ini
  U3t <- U3_ini
  Xt <- X_ini
  U1t_perp = qr.Q(qr(U1t),complete=TRUE)[,(r1+1):p1]
  U2t_perp = qr.Q(qr(U2t),complete=TRUE)[,(r2+1):p2]
  U3t_perp = qr.Q(qr(U3t),complete=TRUE)[,(r3+1):p3]
  Zt <- ttl(Xt, list(t(U1t),t(U2t),t(U3t)),c(1:3))
  V1t = qr.Q(qr(t(k_unfold(Zt, 1)@data))) 
  V2t = qr.Q(qr(t(k_unfold(Zt, 2)@data))) 
  V3t = qr.Q(qr(t(k_unfold(Zt, 3)@data))) 
  X_rela_err <- fnorm(Xt - X)/fnorm(X) 
  error_matrix = c(0,X_rela_err,0)
  t <- proc.time()
  for (i in 1:t_max){
    # to save the computation
    hatB <- ttl(Y, list(t(U1t), t(U2t), t(U3t)), c(1:3))
    hatD1 = t(U1t_perp) %*% k_unfold( ttl(Y, list(t(U2t), t(U3t)), c(2,3) ), 1  )@data %*% V1t
    hatD2 = t(U2t_perp) %*% k_unfold( ttl(Y, list(t(U1t), t(U3t)), c(1,3) ), 2  )@data %*% V2t
    hatD3 = t(U3t_perp) %*% k_unfold( ttl(Y, list(t(U1t), t(U2t)), c(1,2) ), 3  )@data %*% V3t
    if (retrac == 'ortho'){
      hatD1_alt = U1t %*% (k_unfold(hatB, 1)@data) %*% V1t + U1t_perp %*% hatD1
      hatD2_alt = U2t %*% (k_unfold(hatB, 2)@data) %*% V2t + U2t_perp %*% hatD2
      hatD3_alt = U3t %*% (k_unfold(hatB, 3)@data) %*% V3t + U3t_perp %*% hatD3
      # final estimate
      hatL1 <- hatD1_alt %*% solve(k_unfold(hatB, 1)@data %*% V1t)
      hatL2 <- hatD2_alt %*% solve(k_unfold(hatB, 2)@data %*% V2t)
      hatL3 <- hatD3_alt %*% solve(k_unfold(hatB, 3)@data %*% V3t)
      Xt = ttl(hatB, list(hatL1, hatL2, hatL3), c(1,2,3))
      U1t <- qr.Q(qr(hatL1))
      U2t <- qr.Q(qr(hatL2))
      U3t <- qr.Q(qr(hatL3))      
    } else if( retrac == 'hosvd' ){
      ## The k_unfold operation in R is U_d \otimes \cdots U_{i+1} \otimes U_{i-1} \otimes cdots
      tildeXt <- ttl(hatB, list(U1t,U2t,U3t), c(1:3)) + k_fold(U1t_perp %*% hatD1 %*% t(kronecker(U3t,U2t) %*% V1t),1,modes = c(p1, p2, p3)) +
        k_fold(U2t_perp %*% hatD2 %*% t(kronecker(U3t,U1t) %*% V2t),2,modes = c(p1, p2, p3)) + k_fold(U3t_perp %*% hatD3 %*% t(kronecker(U2t,U1t) %*% V3t),3, modes = c(p1, p2, p3))
      hosvd_result <- hosvd(tildeXt, c(r1, r2, r3))
      Xt <- hosvd_result$est
      Ut <- hosvd_result$U
      U1t <- Ut[[1]]
      U2t <- Ut[[2]]
      U3t <- Ut[[3]]
    }
    U1t_perp = qr.Q(qr(U1t),complete=TRUE)[,(r1+1):p1]
    U2t_perp = qr.Q(qr(U2t),complete=TRUE)[,(r2+1):p2]
    U3t_perp = qr.Q(qr(U3t),complete=TRUE)[,(r3+1):p3]
    Zt <- ttl(Xt, list(t(U1t),t(U2t),t(U3t)),c(1:3))
    V1t = qr.Q(qr(t(k_unfold(Zt, 1)@data))) 
    V2t = qr.Q(qr(t(k_unfold(Zt, 2)@data))) 
    V3t = qr.Q(qr(t(k_unfold(Zt, 3)@data))) 
    X_rela_err <- fnorm(Xt - X)/fnorm(X)
    error_matrix <- rbind(error_matrix,c(i,X_rela_err,(proc.time()-t)[3]))
    if (X_rela_err < succ_tol || X_rela_err > 5){
      break
    }
  }
  return(error_matrix)
}




















