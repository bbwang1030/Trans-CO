install.packages('expm')
install.packages('MASS')
install.packages('ContaminatedMixt')
install.packages('sn')
install.packages('parallel')
install.packages('Rcpp')
install.packages('RcppEigen')

library(expm)
library(MASS)
library(ContaminatedMixt)
library(sn)
library(parallel)
library(Rcpp)
library(RcppEigen)

### Generate target data
generate_simulationData<-function(n,sigma,beta,distribution=stop('t/N/SN'),seed=NULL,p){
  
  if(is.null(seed)==F){set.seed(seed)}
  
  x.Sigma=diag(rep(1,p))
  for(i in 1:p){for(j in 1:p){x.Sigma[i,j]=0.7^abs(i-j)}}
  x<-mvrnorm(n, c(rep(0,p)), x.Sigma)
  sigma<-sigma
  
  if(distribution=='t'){
    nu<-5
    tau<-rgamma(n, shape=nu/2, rate=nu/2)
    e<-sapply(1:n,function(i){rnorm(n=1,mean=0,sd=sigma/tau[i])})
    cn.alpha<-cn.eta<-skewness<-NULL
    
  }else if(distribution=='N'){
    e<-rnorm(n,0,sigma)
    tau<-nu<-cn.alpha<-cn.eta<-skewness<-NULL
    
  }else if(distribution=='CN'){
    cn.alpha=0.9;cn.eta=10
    e<-rCN(n, mu = 0, Sigma=1, cn.alpha, cn.eta )
    tau<-nu<-skewness<-NULL
    
  }else if(distribution=='St'){
    cn.alpha=0.2;cn.eta=10;skewness=3
    nu<-5
    tau<-rgamma(n, shape=nu/2, rate=nu/2)
    e<-sapply(1:n,function(i){rsn(1,0,1/tau[i])})
    nu<-cn.alpha<-cn.eta<-NULL
  }
  
  y<-x%*%beta+e
  
  if(is.null(seed)==F){set.seed(NULL)}
  return(list(n=n,x=x,y=y,beta=beta,sigma=sigma,tau=tau,nu=nu,p=p,distribution=distribution,
              cn.alpha=cn.alpha,cn.eta=cn.eta,skewness=skewness))
}

### Generate source data
generate_sourcedatasets<-function(K,M,nvec,sigmavec,beta,h,distribution,seed=NULL,p){
  sourcedatas<-list()
  if(M>0){
    for(m in 1:M){
      deltalist<-h/200*indicator_vector(p,sample(17:p,100))
      sourcedata<-generate_simulationData(n=nvec[m],
                                          sigma=sigmavec[m],
                                          beta=beta-deltalist,
                                          distribution=distribution,
                                          seed=NULL,
                                          p=p)
      sourcedatas[[m]]<-sourcedata
    }
  }
  
  if(K>M){
    for(k in (M+1):K){
      deltalist<-h/100*c(indicator_vector((p),sample(1:(p),200)))
      sourcedata<-generate_simulationData(n=nvec[k],
                                          sigma=sigmavec[k],
                                          beta=deltalist,
                                          distribution=distribution,
                                          seed=NULL,
                                          p=p)
      sourcedatas[[k]]<-sourcedata
    }
  }
  sourcedatas
}

lasso_cd_rcpp<- function(x, y, tauhat0,lambda, betahat0 = NULL, max_iter = 50) {


  # if the initial beta is null, then start at zero
  if(is.null(betahat0)) {
    betahat0 = rep(0, ncol(x))
  }
  
  # value of the loss function (MSE) at each iteration
  loss <- rep(0, max_iter)
  
  beta<-betahat0
  k=1
  while (k <= max_iter) {
    residual <- y - x %*% beta
    loss[k] <- mean(residual*tauhat0 * residual)
    
    beta <- updatebeta(x,beta,residual,tauhat0,lambda)
    
    if (all(abs(beta-betahat0)<0.001)) {
      break
    }
    k=k+1
    betahat0<-beta
  }
  
  # end-of iteration loop
  return(beta)
}

### Penalized t/N linear regression
PLR<-function(n,x,y,method,lambda=NULL,tek=50,tol=0.0001){
  sigmahat0=1
  betahat0=rep(0,dim(x)[2])
  nuhat0=5
  
  s <- 1
  while (s<=tek) {
    # tau
    if(method=='N'){
      tauhat0<-rep(1,n)
    }else if(method=='t'){
      delta<-as.vector((y-x%*%betahat0)*(y-x%*%betahat0)/sigmahat0)
      tauhat0<-(nuhat0+1)/(nuhat0+delta)
    }
    
    #beta
    betahat<-lasso_cd_rcpp(x=x, y=y,tauhat0=tauhat0,lambda=lambda,betahat0=betahat0)
    
    #sigma
    phat<-length(which(betahat>0.001))
    sigmahat<-as.vector((1/(n-phat-1)*t(tauhat0*(y-x%*%betahat))%*%(y-x%*%betahat)))
    
    if(method=='t'){
      #nu
      f<-function(nu0){
        li<--5*log(nu0/2)+2*lgamma(nu0/2)-5*(digamma((nu0+1)/2)-log((nu0+delta)/2)-tauhat0)
        L=sum(li)
        return(L)
      }
      suppressWarnings({
        nuhat=nlminb(start=1,objective=f,lower=1, upper=50)$par
      })
      nuhat
    }
    
    s<- s + 1
    if(all(is.na(betahat)==F) & all(abs(betahat-betahat0)<tol)) break
    
    if(method=='t'){nuhat0<-nuhat}
    betahat0<-betahat
    sigmahat0<-sigmahat
  }
  
  fit <- list()
  if(method=='t'){fit$nuhat<-nuhat}
  fit$beta<-betahat
  fit$sigma<-sigmahat
  fit$tau<-tauhat0

  return(fit)
}

### Penalized t/N linear regression with cross validation
PLR_CV<-function(n,x,y,method,tek=100,tol=0.01,
                 lambda.interval=seq(0.01,0.95,0.05),lambda_kfold=5){
  data_split<-kfold_split(n=n,x=x,y=y,kfold=lambda_kfold)
  
  L_mean<-c()
  for (lambda in lambda.interval) {
    L_c<-c()
    for(i in 1:lambda_kfold){
      traindata<-data_split$traindataset[[i]]
      testdata<-data_split$testdataset[[i]]
      
      model<- PLR(n=traindata$n,x=traindata$x,y=traindata$y,method=method,lambda=lambda)
      
      L<-loss(testdata,model,method)
      L_c<-c(L_c,L)
    }
    L_mean<-c(L_mean,mean(L_c))
  }
  opt.lambda<-lambda.interval[which(L_mean==max(L_mean))]
  
  if(length(opt.lambda)>1){opt.lambda <- opt.lambda[1]}
  model<- PLR(n,x,y,method=method,lambda=opt.lambda)
  model$opt.lambda<-opt.lambda
  model$L<-L_mean
  
  return(model)
}

### transfer learning
TL<-function(targetdata,sourcedatas,method,model.tar,lambda.interval){
  
  K<-length(sourcedatas)
  
  if(K==0){return(model.tar)}
  
  n<-targetdata$n
  x<-targetdata$x
  y<-targetdata$y
  
  for(k in 1:K){
    sourcedata<-sourcedatas[[k]]
    
    #step1
    w.model<-PLR_CV(n=sourcedata$n,x=sourcedata$x,y=sourcedata$y,method=method)
    w<-w.model$beta
    
    #step2
    e= targetdata$y-targetdata$x %*% w
    delta.model<-PLR_CV(n=targetdata$n,x=targetdata$x,y=e,method=method,lambda.interval=lambda.interval)
    delta<-delta.model$beta
    
    #step3
    n<-n+sourcedata$n
    x<-rbind(x,sourcedata$x)
    y<-c(y,sourcedata$y+sourcedata$x%*%delta)
  }
  
  TL.model<-PLR_CV(n=n,x=x,y=y,method=method,lambda.interval=lambda.interval)
  return(TL.model)
}

### transferable sources detection
source_detection<-function(targetdata,sourcedatas,method,detection_kfold,t.threshold,lambda.interval){
  
  K<-length(sourcedatas)
  
  #step 1  split dataset
  data_split<-kfold_split(n=targetdata$n,x=targetdata$x,y=targetdata$y,kfold=detection_kfold)
  
  L0_c<-c()
  Lk_list<-list()

  for(i in 1:detection_kfold){
    message(paste('detection_kfold:',i))
    
    # step 2.1 no transfer
    traindata<-data_split$traindataset[[i]]
    testdata<-data_split$testdataset[[i]]
    
    model0<- PLR_CV(n=traindata$n,x=traindata$x,y=traindata$y,method=method,lambda.interval=lambda.interval)
    
    L0<-loss(testdata,model0,method)
    L0_c<-c(L0_c,L0)
    
    # step 2.2 with transfer
    Lk_c<-c()
    for (k in 1:K) {
      message(paste('k:',k))
  
      modelTL<-TL(targetdata=traindata,sourcedatas=sourcedatas[k],method=method,model.tar=model0,lambda.interval=lambda.interval)
      Lk<-loss(testdata,modelTL,method)
      Lk_c<-c(Lk_c,Lk)
    }
    Lk_list[[i]]<-Lk_c
  }
  
  L0_mean<-mean(L0_c)
  Lk_table<-matrix(unlist(Lk_list),ncol=detection_kfold)
  Lk_mean<-rowMeans(Lk_table)
  T_index<-(L0_mean-Lk_mean)<sapply(1:K, function(x){max(L0_mean,0.01)})*t.threshold
  
  A<-sourcedatas[T_index]
  
  list(A=A,T_index=T_index,t=t,L0_mean=L0_mean,Lk_mean=Lk_mean,L0_c=L0_c,Lk_table=Lk_table)
}

loss<-function(testdata,model,method){
  
  y<-testdata$y
  x<-testdata$x
  if(method=='t'){
    sigma<-model$sigma
    beta<-model$beta
    nuhat<-model$nuhat
    tau<-model$tau
    L<-sum(log(gamma((nuhat+1)/2)/(gamma(nuhat/2)*sqrt(3.14*nuhat*sigma^2))*(1+(y-x%*%beta)^2/(nuhat*sigma^2))^(-(nuhat+1)/2)))
  }else if(method=='N'){
    sigma<-model$sigma
    beta<-model$beta
    L<-n/2*(log(sigma))+t(1/(2*sigma)*(y-x%*%beta))%*%(y-x%*%beta)
  }
  
  return(L=L)
}

kfold_split<-function(n,x,y,kfold,seed=NULL){
  
  if(is.null(seed)==F){set.seed(seed)}
  index<-split(1:n, sample(rep(1:kfold, n/kfold)))
  
  traindataset<-list()
  testdataset<-list()
  for(i in 1:kfold){
    test_index<-index[[i]]
    testdata<-list(n=length(test_index),y=y[test_index],x=x[test_index,])
    
    train_index<-unlist(index[-i])
    traindata<-list(n=length(train_index),y=y[train_index],x=x[train_index,])
    
    traindataset[[i]]<-traindata
    testdataset[[i]]<-testdata
  }
  set.seed(NULL)
  
  return(list(traindataset=traindataset,testdataset=testdataset))
}

indicator_vector <- function(n, position) {
  vector <- rep(0, n)  # 创建全零向量
  vector[position] <- sample(c(1,-1),length(position),replace = T)  # 在指定位置设置为1
  return(vector)
}


###############################################################################################################
Rcpp::sourceCpp('updatebeta.cpp')
lambda.interval=seq(0.01,0.5,0.05)

K=2 # number of sources
M=1 # number of transferable sources

p=500
beta<-c(rep(0.5,16),rep(0,(p-16)))
t.threshold=0.1
h=10
distribution='t' #t St CN N
method='t' # t N

### Generate simulated data
targetdata<-generate_simulationData(n=200,sigma=1,beta=beta,distribution=distribution,p=p)
sourcedatas<-generate_sourcedatasets(K=K,M=M,
                                     nvec=rep(150,K),
                                     sigmavec=rep(1,K),
                                     beta=beta,
                                     h=h,
                                     distribution=distribution,
                                     p=p)

### transferable sources detection
detection_simu<-source_detection(targetdata,sourcedatas,method,detection_kfold=2,
                                 t.threshold=t.threshold,lambda.interval=lambda.interval)

A<-detection_simu$A

### no transfer
model0=PLR_CV(n=targetdata$n,x=targetdata$x,y=targetdata$y,method=method,lambda.interval=lambda.interval)
beta0<-model0$beta

### transfer
modelTL<-TL(targetdata,A,method,model0,lambda.interval=lambda.interval)
betaTL<-modelTL$beta

### naive-transfer
modelTL_naive<-TL(targetdata,sourcedatas,method,model0,lambda.interval=lambda.interval)
betaTL_naive<-modelTL_naive$beta



