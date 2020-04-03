library(readr)
library(caret)
library(randomForest)
library(dplyr)

setwd("C:/Users/xssok/Documents")
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", 
              "pml-training.csv")
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", 
              "pml-testing.csv")

training_pml <-read.csv("pml-training.csv")
testing_pml <- read.csv("pml-testing.csv")

dim(training_pml)
dim(testing_pml)


set.seed(1234)

inTrain <- createDataPartition(y=training_pml$classe,
                               p=0.6, list=FALSE)
training <- training_pml[inTrain,] ## subset data into training set
testing <- training_pml[-inTrain,] ## subset data into test set

# Perform exploratory analysis:
dim(training)
dim(testing)
summary(training)
str(training)
head(training) 

table(training$classe) 


#exclude identifier, timestamp, and window data (they cannot be used for prediction)
training_ex <- select(training, -(X:num_window))
#select variables with high (over 95%) missing data --> exclude them from the analysis
training_ex[training_ex==""] <- NA
NArate <- apply(training_ex, 2, function(x) sum(is.na(x)))/nrow(training_ex)
training_cl <- training_ex[!(NArate>0.95)]
str(training_cl)

# Since the number of variables are still over 50, PCA is applied
preProc <- preProcess(training_cl[,-53],method="pca",thresh=.8) #12 components to capture 80 percent of the variance 
preProc
preProc <- preProcess(training_cl[,-53],method="pca",thresh=.9) #18 components to capture 90 percent of the variance
preProc
preProc <- preProcess(training_cl[,-53],method="pca",thresh=.95) #24 components to capture 95 percent of the variance
preProc
preProc <- preProcess(training_cl[,-53],method="pca",pcaComp=24) 
preProc$rotation
training_pca <- predict(preProc,training_cl[,-53])

# Apply ramdom forest method (non-bionominal outcome & large sample size)
modFit <- randomForest(training_cl$classe ~ .,   data=training_pca, do.trace=F)
print(modFit) # view results 

# Check with test set
testing_cl <- select(testing, -(X:num_window))
testing_cl[testing_cl==""] <- NA
NArate <- apply(testing_cl, 2, function(x) sum(is.na(x)))/nrow(testing_cl)
testing_cl <- testing_cl[!(NArate>0.95)]
testing_pca <- predict(preProc,testing_cl[,-53])
confusionMatrix(testing_cl$classe,predict(modFit,testing_pca))

#Predict classes of 20 test data
testing_pml_cl <- select(testing_pml, -(X:num_window))
testing_pml_cl[testing_pml_cl==""] <- NA
NArate <- apply(testing_pml_cl, 2, function(x) sum(is.na(x)))/nrow(testing_pml_cl)
testing_pml_cl <- testing_pml_cl[!(NArate>0.95)]
testing_pml_pca <- predict(preProc,testing_pml_cl[,-53])
testing_pml_cl$classe <- predict(modFit,testing_pml_pca)
cbind (as.character(testing_pml_cl$classe))

