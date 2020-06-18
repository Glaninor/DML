library(tidyverse)
library(data.table)
library(keras)
library(tensorflow)
library(ROCR)

drug <- read.csv("drug_table_mortality_with_no.csv")
mor <- read.csv("mortality_table.csv")

ntrain <- sample(nrow(drugMor),floor(0.7*nrow(drugMor)),replace = FALSE)
train <- drugMor[ntrain,]
test <- drugMor[-ntrain,]
keras.trian.x <- as.matrix(train[,c(-1,-2)])
keras.trian.y <- as.matrix(train[,2])
keras.test.x <- as.matrix(test[,c(-1,-2)])
keras.test.y <- as.matrix(test[,2])

model <- keras_model_sequential()
model %>%
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

history <- model %>% fit(keras.train.x,keras.train.y, epochs=30, batch_size=64)
keras.pre <- predict(model,keras.test.x)
predict <- prediction(keras.pre,keras.test.y)
predict.auc <- performance(predict,'auc')
predict.auc@y.values
[[1]]
[1] 0.9522891

model %>% evaluate(keras.test.x,keras.test.y)
     loss  accuracy 
0.3398948 0.9195926 

plot(history)





