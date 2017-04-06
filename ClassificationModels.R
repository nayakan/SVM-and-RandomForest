install.packages("caret", dependencies = c("Depends", "Suggests", "Imports"))
library(caret)


data.dir <- 'Set directory'
train.file <- paste0(data.dir, 'pima-indians-diabetesTrain.csv')
test.file  <- paste0(data.dir, 'pima-indians-diabetesTest.csv')

dtrain    <- read.csv(train.file, stringsAsFactors=F)
dtest     <- read.csv(test.file,  stringsAsFactors=F)

#######################################################################################
# Build RandomForest classification model using 10-fold-cross-validation to tune the 'm' 
# paramter (the number of features randomly selected for each split) and train the model 
# with this 'm'.Determine CV sensitivity & specificity.
set.seed(123)
ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 10, classProbs = TRUE, 
                     summaryFunction = twoClassSummary)
rndFit <- train(class ~ ., data = dtrain, method = "rf", tuneLength = 15, 
                trControl = ctrl, metric = "ROC",preProc = c("center", "scale"))
# Print
rndFit

########################################################################################
# Build SVM with a radial basis function(RBF) kernel. Tune parameters sigma/cost using
# 10-fold-cross-validation. Determine CV sensitivity & specificity.
set.seed(123)
ctrl1 <- trainControl(method = "repeatedcv", number = 10, repeats = 10, classProbs = TRUE, 
                      summaryFunction = twoClassSummary)
svmFit <-  train(class ~ ., data = dtrain, method = "svmRadial", tuneLength = 9, 
                 trControl = ctrl1, metric = "ROC",preProc = c("center", "scale"))
svmFit

# Second pass to refine the parameter space using expan grid
grid <- expand.grid(sigma = c(0.1, 0.12, 0.13), C = c(0.3, 0.4, 0.5, 0.6))
svmFit1 <-  train(class ~ ., data = dtrain, method = "svmRadial", tuneGrid = grid, 
                 trControl = ctrl1, metric = "ROC",preProc = c("center", "scale"))
# Print
svmFit1

#######################################################################################
# Compute the confusion matrix on the test data set for the above two models and
# compare their performance.

# Random Forest Model: Predict for test Data
rndFitClasses <- predict(rndFit, newdata = dtest)
# Random Forest Model: Build Confusion Matrix
confusionMatrix(data = rndFitClasses, dtest$class)

# SVM Model: Predict for test Data
svmFitClasses <- predict(svmFit, newdata = dtest)
# SVM Model: Build Confusion Matrix
confusionMatrix(data = svmFitClasses, dtest$class)
