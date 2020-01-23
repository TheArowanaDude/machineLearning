nyc_airbinb <- read.csv("Downloads/AB_NYC_2019.csv")
library(tensorflow)
library(keras)
nyc_airbinb <- nyc_airbinb[complete.cases(nyc_airbinb),]
#nyc_airbinb <- na.omit(nyc_airbinb)

nyc_airbinb$id <- NULL
nyc_airbinb$host_id <- NULL
nyc_airbinb$host_name <- NULL
nyc_airbinb$name <- NULL
nyc_airbinb$last_review <- NULL
#nyc_airbinb$availability_365 <- NULL
nyc_airbinb$neighbourhood <- NULL

n <- nrow(nyc_airbinb)
set.seed(10)
train <- sample(n, 1*n)
train_data <- nyc_airbinb[train,-6]
train_labels <- nyc_airbinb[train,6]

test_data <- nyc_airbinb[-train, -6]
test_labels <- nyc_airbinb[-train, 6]


train_data$room_type <- to_categorical(as.numeric(train_data$room_type)-1)
#train_data$neighbourhood <- to_categorical(as.numeric(train_data$neighbourhood))
train_data$neighbourhood_group <- to_categorical(as.numeric(train_data$neighbourhood_group)-1)

test_data$room_type <- to_categorical(as.numeric(test_data$room_type)-1)
#test_data$neighbourhood <- to_categorical(as.numeric(test_data$neighbourhood))
test_data$neighbourhood_group <- to_categorical(as.numeric(test_data$neighbourhood_group)-1)

nyc_airbinb$room_type <- to_categorical(as.numeric(nyc_airbinb$room_type)-1)
nyc_airbinb$neighbourhood_group <- to_categorical(as.numeric(nyc_airbinb$neighbourhood_group)-1)


#train_data <- train_data[complete.cases(train_data),]
train_data <- na.omit(train_data)
#test_data <- na.omit(test_data)
nyc_airbinb <- na.omit(nyc_airbinb)

train_data <- scale(train_data)
nyc_airbinb <- scale(nyc_airbinb)

col_means_train <- attr(train_data, "scaled:center")
col_stddevs_train <- attr(train_data, "scaled:scale")
test_data <- scale(test_data,center = col_means_train,scale = col_stddevs_train)

#train_data<- data.frame(train_data)
early_stop <- callback_early_stopping(monitor = "val_loss", patience = 20)

model <- keras_model_sequential(layers=list(
  layer_dense(units = 5, activation = "relu",input_shape = dim(train_data)[2]),
  layer_dense(units = 1)))

compile(model,loss = "mse",optimizer = optimizer_rmsprop(),metrics = list("mean_absolute_error"))
history <- fit(model,train_data,train_labels,epochs = 300,validation_split = 0.2,verbose =1,callbacks = list(early_stop))

eval.results <- evaluate(model,test_data,test_labels,verbose = 1)
mae <- eval.results$mean_absolute_error
paste0("Mean absolute error on test set: ", sprintf("%.2f", mae))

