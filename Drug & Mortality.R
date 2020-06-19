library(tidyverse)
library(data.table)
library(keras)
library(tensorflow)
library(ROCR)

#read in
drug <- read.csv("drug_table_mortality_with_no.csv")
mor <- read.csv("mortality_table.csv")

#sample
ntrain <- sample(nrow(drugMor),floor(0.7*nrow(drugMor)),replace = FALSE)
train <- drugMor[ntrain,]
test <- drugMor[-ntrain,]
keras.trian.x <- as.matrix(train[,c(-1,-2)])
keras.trian.y <- as.matrix(train[,2])
keras.test.x <- as.matrix(test[,c(-1,-2)])
keras.test.y <- as.matrix(test[,2])

#keras
keras.model <- keras_model_sequential()
keras.model %>%
layer_dense(units = 64, activation = 'relu', input_shape = c(ncol(keras.test.x))) %>%
layer_dropout(rate = 0.1) %>%
layer_dense(units = 32, activation = 'relu') %>%
layer_dropout(rate = 0.1) %>%
layer_dense(units = 1, activation = 'sigmoid') %>%
compile(
optimizer = optimizer_rmsprop(),
loss = loss_binary_crossentropy,
metrics = c('accuracy')
)

history <- keras.model %>% fit(keras.train.x,keras.train.y, epochs=30, batch_size=64)
keras.pre <- predict(keras.model,keras.test.x)
predict <- prediction(keras.pre,keras.test.y)
predict.auc <- performance(predict,'auc')
predict.auc@y.values
[[1]]
[1] 0.9522891

keras.model %>% evaluate(keras.test.x,keras.test.y)
     loss  accuracy 
0.3398948 0.9195926 

plot(history)


#Gradient Descent
gradientD <- function(x,y,error,maxiter,stepmethod=T,step=0.1,alpha=0.10,beta=0.9)
{
  x <- cbind(matrix(1,nrow(x),1),x)
  theta <- matrix(rep(0,ncol(x)),ncol(x),1)
  iter <- 1
  newerror <- 1
  
  while((newerror>error)|(iter<maxiter)){
    iter <- iter+1
    h <- x%*%theta  
    des <- t(t(h-y)%*%x)
    
    if(stepmethod==T){
      step=1
      new_theta <- theta-step*des
      new_h <- x%*%new_theta
      costfunction <- t(h-y)%*%(h-y)
      new_costfunction <- t(new_h-y)%*%(new_h-y)
      
      while(new_costfunction>costfunction-alpha*step*sum(des*des)){
        step <- step*beta
        new_theta <- theta-step*des
        new_h <- x%*%new_theta
        new_costfunction <- t(new_h-y)%*%(new_h-y)  
      }
      newerror <- t(theta-new_theta)%*%(theta-new_theta)       
      theta <- new_theta     
    }
    
    
    if(stepmethod==F){        
      new_theta <- theta-Step*des
      new_h <- x%*%new_theta
      
      newerror <- t(theta-new_theta)%*%(theta-new_theta)
      theta <- new_theta 
    }
    
  }
  costfunction <- t(x%*%theta-y)%*%(x%*%theta-y)
  result <- list(theta,iter,costfunction)
  names(result) <- c('coefficient','iters','error')
  result
}

grardmodel <- gradientD(trian.x,trian.y,error=0.1,stepmethod=T,step=0.1,alpha=0.1,beta=0.9)



