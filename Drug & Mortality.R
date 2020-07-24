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
trian.x <- as.matrix(train[,c(-1,-2)])
trian.y <- as.matrix(train[,2])
test.x <- as.matrix(test[,c(-1,-2)])
test.y <- as.matrix(test[,2])

#keras
keras.model <- keras_model_sequential()
keras.model %>%
layer_dense(units = 64, activation = 'relu', input_shape = c(ncol(test.x))) %>%
layer_dropout(rate = 0.1) %>%
layer_dense(units = 32, activation = 'relu') %>%
layer_dropout(rate = 0.1) %>%
layer_dense(units = 1, activation = 'sigmoid') %>%
compile(
optimizer = optimizer_rmsprop(),
loss = loss_binary_crossentropy,
metrics = c('accuracy')
)

history <- keras.model %>% fit(train.x,train.y, epochs=30, batch_size=64)
Epoch 1/30
337/337 [==============================] - 2s 5ms/step - loss: 0.0170 - accuracy: 0.9996
Epoch 2/30
337/337 [==============================] - 1s 3ms/step - loss: 9.9411e-08 - accuracy: 1.0000
Epoch 3/30
337/337 [==============================] - 1s 3ms/step - loss: 2.3826e-09 - accuracy: 1.0000
Epoch 4/30
337/337 [==============================] - 1s 3ms/step - loss: 1.3502e-09 - accuracy: 1.0000
Epoch 5/30
337/337 [==============================] - 1s 3ms/step - loss: 1.3986e-09 - accuracy: 1.0000
Epoch 6/30
337/337 [==============================] - 1s 3ms/step - loss: 1.1093e-09 - accuracy: 1.0000
Epoch 7/30
337/337 [==============================] - 1s 3ms/step - loss: 4.5957e-10 - accuracy: 1.0000
Epoch 8/30
337/337 [==============================] - 1s 3ms/step - loss: 3.9135e-10 - accuracy: 1.0000
Epoch 9/30
337/337 [==============================] - 1s 3ms/step - loss: 2.9328e-10 - accuracy: 1.0000
Epoch 10/30
337/337 [==============================] - 1s 3ms/step - loss: 2.8226e-10 - accuracy: 1.0000
Epoch 11/30
337/337 [==============================] - 1s 3ms/step - loss: 1.2690e-10 - accuracy: 1.0000
Epoch 12/30
337/337 [==============================] - 1s 3ms/step - loss: 2.2909e-10 - accuracy: 1.0000
Epoch 13/30
337/337 [==============================] - 1s 3ms/step - loss: 3.6198e-10 - accuracy: 1.0000
Epoch 14/30
337/337 [==============================] - 1s 3ms/step - loss: 1.2287e-10 - accuracy: 1.0000
Epoch 15/30
337/337 [==============================] - 1s 3ms/step - loss: 2.5093e-10 - accuracy: 1.0000
Epoch 16/30
337/337 [==============================] - 1s 3ms/step - loss: 8.7702e-11 - accuracy: 1.0000
Epoch 17/30
337/337 [==============================] - 1s 3ms/step - loss: 1.7961e-10 - accuracy: 1.0000
Epoch 18/30
337/337 [==============================] - 1s 3ms/step - loss: 1.3158e-10 - accuracy: 1.0000
Epoch 19/30
337/337 [==============================] - 1s 3ms/step - loss: 1.2212e-10 - accuracy: 1.0000
Epoch 20/30
337/337 [==============================] - 1s 3ms/step - loss: 1.2293e-10 - accuracy: 1.0000
Epoch 21/30
337/337 [==============================] - 1s 3ms/step - loss: 1.5291e-10 - accuracy: 1.0000
Epoch 22/30
337/337 [==============================] - 1s 3ms/step - loss: 8.2111e-11 - accuracy: 1.0000
Epoch 23/30
337/337 [==============================] - 1s 3ms/step - loss: 6.3524e-11 - accuracy: 1.0000
Epoch 24/30
337/337 [==============================] - 1s 3ms/step - loss: 4.5704e-11 - accuracy: 1.0000
Epoch 25/30
337/337 [==============================] - 1s 3ms/step - loss: 7.2359e-11 - accuracy: 1.0000
Epoch 26/30
337/337 [==============================] - 1s 3ms/step - loss: 1.5442e-10 - accuracy: 1.0000
Epoch 27/30
337/337 [==============================] - 1s 2ms/step - loss: 3.7519e-11 - accuracy: 1.0000
Epoch 28/30
337/337 [==============================] - 1s 3ms/step - loss: 2.0006e-10 - accuracy: 1.0000
Epoch 29/30
337/337 [==============================] - 1s 3ms/step - loss: 6.9150e-11 - accuracy: 1.0000
Epoch 30/30
337/337 [==============================] - 1s 3ms/step - loss: 1.0500e-10 - accuracy: 1.0000
keras.pre <- predict(keras.model,test.x)
predict <- prediction(keras.pre,test.y)
predict.auc <- performance(predict,'auc')
predict.auc@y.values
[[1]]
[1] 0.9522891

keras.model %>% evaluate(keras.test.x,keras.test.y)
289/289 [==============================] - 0s 724us/step - loss: 0.0058 - accuracy: 0.9999
       loss    accuracy 
0.005806785 0.999891639 

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

grardmodel <- gradientD(trian.x,trian.y,error=0.1,1000,stepmethod=T,step=0.1,alpha=0.1,beta=0.9)


#Gradient Descent
library(gradDescent)
drugMor.split <- splitData(drugMor,dataTrainRate = 0.5,seed = 1111)
grad.train <- drugMor.split$dataTrain
grad.test <- drugMor.split$dataTest
dataInput <- (grad.test)[,1:ncol(grad.test)-1]
grad.model <- GD(grad.train)
detach("package:ROCR")
grad.prediction <- prediction(grad.model,dataInput)
grad.prediction <- grad.prediction[,ncol(grad.prediction)]
grad.prediction.auc <- performance(grad.prediction,'auc')

Error in performance(grad.prediction, "auc") : 
  Wrong argument types: First argument must be of type 'prediction'; second and optional third argument must be available performance measures!
