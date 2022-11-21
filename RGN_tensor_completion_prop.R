# Study the property of RGN in tensor completion example
library(rTensor)
library(matrixcalc)
library(MASS)
library(parallel)
source('RGN_Funs.R')
set.seed(2020)
runtime <- proc.time()
p = c(50); r = c(3); sig_total = c(0, 1e-6);
ratio_total <- c(40,45)
n_total <- floor(p^(3/2) * r) * ratio_total;
t_max = 30; retrac <- 'ortho'
init = 'rand'
succ_tol <- 1e-14
exp.time <- 1
final_result <- vector()
for (ratio.index in 1:length(ratio_total)){
  n <- n_total[ratio.index]
  for (sig in sig_total){
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
    all_index <- expand.grid(1:p1, 1:p2, 1:p3)
    sample_index <- sample(1:(p1*p2*p3), n, replace = T)
    all_index <- all_index[sample_index,]
    obser_X <- as.tensor(array(0, dim = c(p1, p2, p3)))
    y <- vector('numeric', n)
    eps = sig * rnorm(n) # Error

    for (i in 1:n){
      y[i] <- X@data[all_index[i,1], all_index[i,2], all_index[i,3]] + eps[i]
      obser_X@data[all_index[i,1], all_index[i,2], all_index[i,3]] <- obser_X@data[all_index[i,1], all_index[i,2], all_index[i,3]] + y[i]
    }
    W <- as.tensor(obser_X@data * (p1 * p2 * p3)/n)

    if (init == 'good'){
      ## Initialization
      hatU <- spectral_initia(all_index, c(p1,p2,p3), c(r1,r2,r3), obser_X, n, y)
      hatU1 = hatU[[1]]; hatU2 = hatU[[2]]; hatU3 = hatU[[3]];
    } else if(init == 'rela_good'){
      U1_perp <- qr.Q(qr(U1),complete = T)[,(r1+1):p1]
      U2_perp <- qr.Q(qr(U2),complete = T)[,(r2+1):p2]
      U3_perp <- qr.Q(qr(U3),complete = T)[,(r3+1):p3]
      O <- matrix(rnorm((p1-r1)*r1), nrow = (p1-r1), ncol=r1)
      O <- qr.Q(qr(O))
      hatU1 <- 1/sqrt(2) * (U1 + U1_perp %*% O)
      O <- matrix(rnorm((p2-r2)*r2), nrow = (p2-r2), ncol=r2)
      O <- qr.Q(qr(O))
      hatU2 <- 1/sqrt(2) * (U2 + U2_perp %*% O)
      O <- matrix(rnorm((p3-r3)*r3), nrow = (p3-r3), ncol=r3)
      O <- qr.Q(qr(O))
      hatU3 <- 1/sqrt(2) * (U3 + U3_perp %*% O)
      hatU <- list(hatU1, hatU2, hatU3)
    } else if(init == 'rand'){
      O <- matrix(rnorm(p1*r1), nrow = p1, ncol=r1)
      hatU1 <- qr.Q(qr(O))
      O <- matrix(rnorm(p2*r2), nrow = p2, ncol=r2)
      hatU2 <- qr.Q(qr(O))
      O <- matrix(rnorm(p3*r3), nrow = p3, ncol=r3)
      hatU3 <- qr.Q(qr(O))
      hatU <- list(hatU1, hatU2, hatU3)
    }
    
    Z <- ttl(W,list( t(hatU1), t(hatU2), t(hatU3)  ), c(1:3))
    Xt <- ttl(Z,hatU,c(1:3))
    hatU1_perp = qr.Q(qr(hatU1),complete=TRUE)[,(r1+1):p1]
    hatU2_perp = qr.Q(qr(hatU2),complete=TRUE)[,(r2+1):p2]
    hatU3_perp = qr.Q(qr(hatU3),complete=TRUE)[,(r3+1):p3]
    hatU1.complete = cbind(hatU1, hatU1_perp)
    hatU2.complete = cbind(hatU2, hatU2_perp)
    hatU3.complete = cbind(hatU3, hatU3_perp)
    U1t <- hatU1
    U2t <- hatU2
    U3t <- hatU3
    U1t_perp <- hatU1_perp
    U2t_perp <- hatU2_perp
    U3t_perp <- hatU3_perp
    
    RGN_time <- vector()
    
    for (i in 1:exp.time){
      RGN_error <- RGN_tencompletion(hatU1,hatU2,hatU3,Xt,y,all_index,U1,U2,U3,X,r1, r2, r3, p1, p2, p3, t_max, succ_tol, retrac)
      RGN_time <- rbind(RGN_time,RGN_error[,5])
    }
    
    RGN_time <- apply(RGN_time,2,mean)
    RGN_error[,5] <- RGN_time   
    num_row <- nrow(RGN_error)
    RGN_error <- cbind(RGN_error, rep(ratio_total[ratio.index],num_row), rep(sig, num_row) )
    final_result <- rbind(final_result, RGN_error)
  }
}


file_name <- paste("RGN_tencompletion","n",paste(n_total, sep="", collapse="_"),"p", p, "r", r, "sig", paste(sig_total, sep="", collapse="_"), "iter", t_max, 'retrac', retrac, 'tol', succ_tol, 'simu_num', exp.time,'init', init, sep = "_")
print(final_result)
print(file_name)
print((proc.time()-runtime)[3])
