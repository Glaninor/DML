library(tidyverse)
library(data.table)
library(keras)
library(tensorflow)
library(ROCR)
library(pROC)

#read in
drug <- read.csv("drug_table_mortality_with_no.csv")
mor <- read.csv("mortality_table.csv")
drugMor <- merge(drug,mor)

#sample
ntrain <- sample(nrow(drugMor),floor(0.7*nrow(drugMor)),replace = FALSE)
train <- drugMor[ntrain,]
test <- drugMor[-ntrain,]
train.x <- as.matrix(train[,-2915])
train.y <- as.matrix(train$Death)
test.x <- as.matrix(test[,-2915])
test.y <- as.matrix(test$Death)

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
keras.predict <- predict(keras.model,test.x)
keras.prediction <- prediction(keras.predict,test.y)
keras.auc <- performance(keras.prediction,'auc')
keras.auc@y.values

keras.model %>% evaluate(test.x,test.y)

plot(history)

#Gradient Descent
library(gradDescent)
drugMor.split <- splitData(drugMor,dataTrainRate = 0.5,seed = 1111)
grad.train <- drugMor.split$dataTrain
grad.test <- drugMor.split$dataTest
dataInput <- (grad.test)[,1:ncol(test)-1]
grad.GD.model <- GD(grad.train, alpha = 0.1, maxIter = 500, seed = 1111)
grad.GD.prediction <- prediction(grad.GD.model,dataInput)
roc_grad.GD <- roc(grad.test[,ncol(test)],grad.GD.prediction[,ncol(test)])
roc_grad.GD$auc
#Area under the curve: 0.5617

grad.SSGD.model <- SSGD(grad.train, alpha = 0.01, maxIter = 100, seed = 222)
grad.SSGD.prediction <- prediction(grad.SSGD.model,dataInput)
roc_grad.SSGD <- roc(grad.test[,ncol(test)],grad.SSGD.prediction[,ncol(test)])
roc_grad.SSGD$auc
#Area under the curve: 0.5601

grad.ADADELTA.model <- ADADELTA(grad.train)
grad.ADADELTA.prediction <- prediction(grad.ADADELTA.model,dataInput)
roc_grad.ADADELTA <- roc(grad.test[,ncol(test)],grad.ADADELTA.prediction[,ncol(test)])
roc_grad.ADADELTA$auc
#Area under the curve: 0.5615