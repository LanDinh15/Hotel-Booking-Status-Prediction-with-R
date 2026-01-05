# Hotel Reservations Classification Analysis

# Load libraries
library(caret)
library(rpart)
library(rpart.plot)
library(e1071)
library(class)
library(pROC)
library(caTools)
library(dplyr)
library(ggplot2)
library(kernlab)
library(vcd)
library(car)
library(tidyverse)
library(corrgram)
library(ROSE)
library(xgboost)
library(magrittr)
library(randomForest)
library(mclust)
library(Metrics)
library(glmnet)
library(ipred)

# 1. Dataset Overview
# Loading dataset (change the file name if needed)
reservations <- read.csv("data/Hotel Reservations.csv")
head(reservations)
sum(is.na(reservations))
str(reservations)

# Converting categorical variables into factors
reservations$market_segment_type <-factor(reservations$market_segment_type)
reservations$type_of_meal_plan <- factor(reservations$type_of_meal_plan)
reservations$booking_status <- factor(reservations$booking_status)
reservations$room_type_reserved <- factor(reservations$room_type_reserved)
str(reservations)

# Remove Booking_ID as it's not useful for modeling
reservations <- reservations[,-1]

# 2. Visualization
# 2.1. Target distribution 
table(reservations$booking_status)
prop <- prop.table(table(reservations$booking_status))*100
prop
barplot(prop, col="#69b3a2",
        main="Booking Status Distribution", 
        xlab="Booking Status", ylab="Precentage(%)")

# 2.2. Numeric features
# Get numeric columns
num_cols <- reservations %>% 
  select(where(is.numeric)) %>% 
  names()
# Long format for plotting
num_long <- reservations %>% 
  pivot_longer(cols = all_of(num_cols),
               names_to = "feature",
               values_to = "value")
# Histograms (one small plot per feature)
ggplot(num_long, aes(x = value)) +
  geom_histogram(bins = 30, fill = "#69b3a2") +
  facet_wrap(~ feature, scales = "free", ncol = 3) +
  labs(title = "Distributions of Numeric Features",
       x = "Value",
       y = "Count")

ggplot(num_long, aes(x = booking_status, y = value, fill = booking_status)) +
  geom_boxplot(outlier.alpha = 0.4) +
  facet_wrap(~ feature, scales = "free_y", ncol = 3) +
  labs(title = "Boxplots of Numeric Features by Booking Status",
       x = "Booking Status",
       y = "Value") +
  theme(legend.position = "none")

# 2.3. Categorical Features
# Get categorical columns (excluding target)
cat_cols <- reservations %>% 
  select(where(~ is.character(.x)|is.factor(.x))) %>% 
  names()
cat_cols <- setdiff(cat_cols, "booking_status")  
# Long format
cat_long <- reservations %>% 
  pivot_longer(cols = all_of(cat_cols),
               names_to = "feature",
               values_to = "category")
# Bar plots 
ggplot(cat_long, aes(x = category)) +
  geom_bar(fill = "#69b3a2") +
  facet_wrap(~ feature, scales = "free", ncol = 3) +
  labs(title = "Distributions of Categorical Features",
       x = "Category",
       y = "Count") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# 2.4. Correlation
num_df <- reservations %>% 
  select(where(is.numeric))
corrgram(num_df, order=TRUE, lower.panel = panel.shade,
         upper.panel = panel.pie, text.panel = panel.txt,
         main="Corrgram of Numeric Features")

par(mfrow = c(1, 1))  # reset layout

# 3. Partitioning - Split into training (75%) and test (25%) sets
set.seed(10)
sample_split <- sample.split(Y = reservations$booking_status, SplitRatio = 0.75)
train_set <- subset(x = reservations, sample_split == TRUE)
test_set <- subset(x = reservations, sample_split == FALSE)

# 4. Model
# 4.1 Decision Tree Model
control = rpart.control(minsplit =1000,maxdepth = 10)
modelDTfinal <- rpart(booking_status~ ., data = train_set, method = "class", control = control)
modelDTfinal

rpart.plot(modelDTfinal)

predDTfinal <- predict(modelDTfinal, newdata = test_set, type = "class")


cmDTfinal <- confusionMatrix(test_set$booking_status, predDTfinal)
cmDTfinal

#Getting the feature importance
importances <- varImp(modelDTfinal)
importances %>%    
  arrange(desc(Overall))

# 4.2 Naive Bayes Model
modelNB <- naiveBayes(booking_status~., data = train_set)
modelNB

predNB <- predict(modelNB, newdata = test_set)
tableNB <- table(test_set$booking_status, predNB)
cmNB <- confusionMatrix(tableNB)
cmNB

# ROC Curve
# set probabilities and take the Canceled class
nb_prob <- predict(modelNB, newdata = test_set, 
                   type = "raw")[, "Canceled"]
roc_nb <- roc(
  response=test_set$booking_status,   
  predictor=nb_prob,                   
  positive="Canceled")

plot(roc_nb,
     legacy.axes = TRUE,     
     print.auc = TRUE,
     xlab = "False Positive Rate",
     ylab = "True Positive Rate",
     main = "ROC Curve - Naive Bayes")

# Get FPR and TPR from the roc object
fpr <- 1 - roc_nb$specificities   # x-axis
tpr <- roc_nb$sensitivities       # y-axis
auc_nb <- pROC::auc(roc_nb)

# Manual ROC plot with proper 0 -> 1 axis
plot(fpr, tpr,
     type = "l",
     xlim = c(0, 1),
     ylim = c(0, 1),
     xlab = "False Positive Rate",
     ylab = "True Positive Rate",
     main = "ROC Curve - Naive Bayes")

abline(a = 0, b = 1, lty = 2, col = "grey")          # diagonal
text(0.6, 0.2, labels = paste("AUC:", round(auc_nb, 3)))

#testing hyper parameters
modelNB1 <- naiveBayes(booking_status~., data = train_set, laplace =0.5)
modelNB1

predNB1 <- predict(modelNB1, newdata = test_set)
tableNB1 <- table(test_set$booking_status, predNB1)
cmNB1 <- confusionMatrix(tableNB1)
cmNB1

# 4.3 Feature Scaling for KNN Model
num_cols <- sapply(train_set, is.numeric)
train_scaled <- scale(train_set[, num_cols])
test_scaled <- scale(test_set[, num_cols])

# Try multiple k values to find the best k
k_values <- seq(1, 25, 2)  
accuracies <- c()

for (k in k_values) {
  knn_pred <- knn(train = train_scaled,
                  test = test_scaled,
                  cl = train_set$booking_status,
                  k = k)
  cm <- confusionMatrix(knn_pred, test_set$booking_status)
  accuracies <- c(accuracies, cm$overall["Accuracy"])
}
results <- data.frame(k = k_values, accuracy = accuracies)
# Best k
best_k <- results$k[which.max(results$accuracy)]
best_k

# KNN model
predKNN <- knn(train=train_scaled, test=test_scaled,
               cl=train_set$booking_status, k=best_k, prob=TRUE)
cmKnn <- confusionMatrix(predKNN, test_set$booking_status)
cmKnn

# ROC Curve
# Get probability of the positive class "Canceled"
knn_p_hat <- attr(predKNN, "prob")  # prob of the predicted class
knn_prob  <- ifelse(predKNN == "Canceled", knn_p_hat, 1 - knn_p_hat)
roc_knn <- roc(
  response=test_set$booking_status,   
  predictor=knn_prob,                  
  positive="Canceled")

plot(roc_knn,
     legacy.axes = TRUE,     
     print.auc = TRUE,
     xlab = "False Positive Rate",
     ylab = "True Positive Rate",
     main = "ROC Curve - KNN")


plot(results$k, results$accuracy,
     type = "b",
     xlab = "k (Number of Neighbors)",
     ylab = "Accuracy",
     main = "KNN Accuracy for Different k")

abline(v = best_k, lty = 2)   # vertical line at best k
text(best_k, max(results$accuracy),
     labels = paste("Best k =", best_k),
     pos = 4)

# Calculate out of Sample error
misClassError <- mean(predKNN != test_set$booking_status)
print(paste('Accuracy =', 1-misClassError))

# 4.4 SVM/KSVM Model
# Using linear kernel
modelsvm <- svm(booking_status~., data=train_set, kernel="linear", scale = TRUE)
modelsvm
summary(modelsvm)

predsvm <- predict(modelsvm,test_set)
cmSVMl <- confusionMatrix(test_set$booking_status, predsvm)
cmSVMl

# using radial kernel
modelsvm1 <- svm(booking_status~., data=train_set, kernel="radial", scale = TRUE)
modelsvm1
summary(modelsvm1)

predsvm1 <- predict(modelsvm1,test_set)
cmSVMr <- confusionMatrix(test_set$booking_status, predsvm1)
cmSVMr

# ksvm model using rbfdot
modelksvm <- ksvm(booking_status~., data=train_set, kernel="rbfdot",scale =TRUE, prob.model = TRUE)
modelksvm
summary(modelksvm)

predksvm <- predict(modelksvm,test_set)
cmSVMfdot <- confusionMatrix(test_set$booking_status, predksvm)
cmSVMfdot

# ksvm using tanhdot kernel
modelksvm1 <- ksvm(booking_status~., data=train_set, kernel="tanhdot",scale =TRUE, prob.model = TRUE)
modelksvm1
summary(modelksvm1)

predksvm1 <- predict(modelksvm1,test_set)
cmSVMtdot <- confusionMatrix(test_set$booking_status, predksvm1)
cmSVMtdot

# ksvm using polydot kernel
modelksvm2 <- ksvm(booking_status~., data=train_set, kernel="polydot",scale =TRUE, prob.model = TRUE)
modelksvm2
summary(modelksvm2)

predksvm2 <- predict(modelksvm2,test_set)
cmSVMpdot <- confusionMatrix(test_set$booking_status, predksvm2)
cmSVMpdot

# 4.5 XGBoost
# XGBoost only working with numeric matrices
# Design matrices 
train_x <- model.matrix(booking_status ~ . , data = train_set)[, -1]
test_x  <- model.matrix(booking_status ~ . , data = test_set)[, -1]
# We define "Canceled" as 1 and the other class as 0
train_y <- ifelse(train_set$booking_status == "Canceled", 1, 0)
test_y  <- ifelse(test_set$booking_status == "Canceled", 1, 0)
# Combining
dtrain <- xgb.DMatrix(data = as.matrix(train_x), label = train_y)
dtest  <- xgb.DMatrix(data = as.matrix(test_x),  label = test_y)

modelXGB <- xgboost(data=dtrain,
                    max.depth=2,eta=0.3, nthread=2, nrounds=100,
                    objective="binary:logistic")
# Predict probabilities
xgb_prob <- predict(modelXGB, newdata = dtest)
# Convert to class labels
xgb_pred_class <- ifelse(xgb_prob >= 0.5, "Canceled", "Not_Canceled")
xgb_pred_factor <- factor(xgb_pred_class,
                          levels = levels(test_set$booking_status))
# Confusion matrix
cmXGB <- confusionMatrix(
  data      = xgb_pred_factor,
  reference = test_set$booking_status
)
cmXGB

# reducing the eta but more nrounds
modelXGBFinal <- xgboost(
  data = dtrain,
  objective = "binary:logistic",
  eval_metric = "error",
  max_depth = 4,
  eta = 0.1,
  subsample = 0.8,
  colsample_bytree = 0.8,
  nrounds = 400,
  nthread = 2,
  verbose = 0
)
# Predict probabilities
xgb_prob_final <- predict(modelXGBFinal, newdata = dtest)
# Convert to class labels
xgb_pred_class_final <- ifelse(xgb_prob_final >= 0.5, 
                               "Canceled", "Not_Canceled")
xgb_pred_factor_final <- factor(xgb_pred_class_final,
                          levels = levels(test_set$booking_status))
# Confusion matrix
cmXGB_final <- confusionMatrix(
  data      = xgb_pred_factor_final,
  reference = test_set$booking_status
)
cmXGB_final

# ROC Curve
roc_xgb <- roc(
  response  = test_set$booking_status,  
  predictor = xgb_prob_final,           
  positive  = "Canceled")

plot(roc_xgb,
     legacy.axes = TRUE,
     print.auc   = TRUE,
     xlab = "False Positive Rate ",
     ylab = "True Positive Rate ",
     main = "ROC Curve - XGBoost")

imp_xgb <- xgb.importance(feature_names = colnames(train_x), model=modelXGBFinal)
xgb.plot.importance(importance_matrix = imp_xgb[1:10])

# 4.6 Random Forest
modelRF <- randomForest(booking_status~.,
                         data=train_set, 
                         ntree = 500, 
                         ntry=4,
                         importance = TRUE,
                         na.action = randomForest::na.roughfix,
                         replace=FALSE)
# for plotting area with separate window
varImpPlot(modelRF, col=5)

orgTG <- test_set[,18]

predRF <- predict(modelRF, newdata=test_set[,-18])
cmRF <- confusionMatrix(orgTG, predRF)
cmRF

# 4.7 Bagging
control_bag <- trainControl(method = "cv", number = 5)
modelBAG <- train(
  booking_status ~ .,
  data = train_set,
  method = "treebag",        
  trControl = control_bag
)
modelBAG

# Predictions on test set
predBAG <- predict(modelBAG, newdata = test_set)
# Confusion matrix
cmBAG <- confusionMatrix(
  data      = predBAG,
  reference = test_set$booking_status
)
cmBAG

# 4.8 Logistic Regression
# model built with numeric features
fullmodelLG <- glm(booking_status~no_of_adults+no_of_children+
                     no_of_weekend_nights+no_of_week_nights+
                     required_car_parking_space+lead_time+
                     arrival_year+arrival_month+arrival_date+
                     repeated_guest+no_of_previous_cancellations+
                     no_of_previous_bookings_not_canceled+avg_price_per_room+
                     no_of_special_requests, data=reservations, family= binomial)
summary(fullmodelLG)



reducedmodelLG <- glm(booking_status~no_of_adults+no_of_week_nights+
                        required_car_parking_space+
                        lead_time+arrival_year+arrival_month+
                        repeated_guest+no_of_previous_cancellations+
                        avg_price_per_room+no_of_special_requests,
                      data=reservations, family= binomial)
summary(reducedmodelLG)

anova(fullmodelLG,reducedmodelLG,test = "Chisq")


# 5.Model comparison
#DT
accuracyDT <- cmDTfinal$overall['Accuracy']
ceDT <- 1- accuracyDT
f1DT <- cmDTfinal$byClass["F1"]
kappaDT <- cmDTfinal$overall['Kappa']
accuracyDT
ceDT
#NB
accuracyNB <- cmNB$overall["Accuracy"]
ceNB <- 1 - accuracyNB
f1NB <- cmNB$byClass["F1"]
kappaNB <- cmNB$overall['Kappa']
#KNN
accuracyknn <- cmKnn$overall["Accuracy"]
ceknn <- 1 - accuracyknn
f1knn <- cmKnn$byClass["F1"]
kappaknn <- cmKnn$overall['Kappa']
#kSVM using rbfdot
accuracyksvm <- cmSVMfdot$overall["Accuracy"]
ceksvm <- 1 - accuracyksvm
f1ksvm <- cmSVMfdot$byClass["F1"]
kappaksvm <- cmSVMfdot$overall['Kappa']
#XGBoost
accuracyxgb <- cmXGB$overall["Accuracy"]
cexgb <- 1 - accuracyxgb
f1xgb <- cmXGB$byClass["F1"]
kappaxgb <- cmXGB$overall['Kappa']
#Random Forest
accuracyrf <- cmRF$overall["Accuracy"]
cerf <- 1 - accuracyrf
f1rf <- cmRF$byClass["F1"]
kapparf <- cmRF$overall['Kappa']
#Bagging
accuracybag <- cmBAG$overall["Accuracy"]
cebag <- 1 - accuracybag
f1bag <- cmBAG$byClass["F1"]
kappabag <- cmBAG$overall['Kappa']

# Table of results
resultsAll <- data.frame(
  ModelName = character(),
  Accuracy = numeric(),
  F1 = numeric(),
  Kappa = numeric(),
  CE = numeric(),
  stringsAsFactors = FALSE
)

resultsAll <- rbind(resultsAll, data.frame(
  ModelName = "Decision Tree",
  Accuracy = accuracyDT,
  F1 = f1DT,
  Kappa = kappaDT,
  CE = ceDT
))

resultsAll <- rbind(resultsAll, data.frame(
  ModelName = " Naive Bayes",
  Accuracy = accuracyNB,
  F1 = f1NB,
  Kappa = kappaNB,
  CE = ceNB
))

resultsAll <- rbind(resultsAll, data.frame(
  ModelName = " KNN Classifier",
  Accuracy = accuracyknn,
  F1 = f1knn,
  Kappa = kappaknn,
  CE = ceknn
))

resultsAll <- rbind(resultsAll, data.frame(
  ModelName = " KSVM Classifier",
  Accuracy = accuracyksvm,
  F1 = f1ksvm,
  Kappa = kappaksvm,
  CE = ceksvm
))

resultsAll <- rbind(resultsAll, data.frame(
  ModelName = " XGBoost",
  Accuracy = accuracyxgb,
  F1 = f1xgb,
  Kappa = kappaxgb,
  CE = cexgb
))

resultsAll <- rbind(resultsAll, data.frame(
  ModelName = " Random Forest",
  Accuracy = accuracyrf,
  F1 = f1rf,
  Kappa = kapparf,
  CE = cerf
))

resultsAll <- rbind(resultsAll, data.frame(
  ModelName = " Bagging",
  Accuracy = accuracybag,
  F1 = f1bag,
  Kappa = kappabag,
  CE = cebag
))

row.names(resultsAll) <- NULL
resultsAll

barplot(resultsAll$Accuracy,
        names.arg = resultsAll$ModelName,
        main ="Accuracy of all Models",
        ylim = c(0,1),
        col = "skyblue",
        xlab = "Model Names", ylab = "Accuracy")

barplot(resultsAll$F1,
        names.arg = resultsAll$ModelName,
        main ="F1 Scores of all Models",
        col = "lightgreen",
        ylim = c(0,0.8),
        xlab = "Model Names", ylab = "F1")


barplot(resultsAll$CE,
        names.arg = resultsAll$ModelName,
        main ="Classification Errors of all Models",
        col = "red",
        ylim = c(0,0.8),
        xlab = "Model Names", ylab = "Classification Error")

