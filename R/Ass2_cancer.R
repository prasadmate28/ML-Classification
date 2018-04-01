library(caret)
library(MASS)
library(bnlearn)
library(class)
library(e1071)
library(bnlearn)

cancer <- read.csv("bayes_data.csv")
cancer_1 = cancer[,-1]
attach(cancer_1)
fix(cancer_1)
cancer_1$X2.1

score <- function(expected, predicted)
{
  confMat<-confusionMatrix(predicted,expected)$byClass
  
  accuracy<-confMat["Balanced Accuracy"]
  recall<-confMat["Recall"]
  precision<-confMat["Precision"]
  f.score<-confMat["F1"]
  
  r <- cbind(accuracy, recall, precision, f.score)
  colnames(r)<- c('accuracy','recall','precision','fscore') 
  return(r) 
}


trainAndTestFold <- function(folds, dataset){
  dataset$X2.1<- ifelse(dataset$X2.1=="2", 1, 0)
  train = dataset[-folds,] 
  test = dataset[folds,]
  
  ## Logistic Regression
  model1<- glm(X2.1 ~ . ,data=train, family =binomial(link='logit'))
  model1.pred=predict(model1 , newdata= test, type = "response")
  model1.pred.binary<-ifelse(model1.pred>0.50, 1, 0)
  model1.scores = score(test$X2.1,model1.pred.binary) 
  model1.pred.binary
  ##for Stacking and Cascading
  model1.pred.train=predict(model1 , newdata= train, type = "response")
  model1.pred.binary.train<-ifelse(model1.pred>0.50, 1, 0)
  
  ##LDA
  model2 = lda(X2.1~.,data=train)                         
  model2.pred = predict(model2,test,type="response")
  model2.scores = score(test$X2.1, model2.pred$class)
  model2.pred$class
  ##for Stacking and Cascading
  model2.pred.train = predict(model2,train,type="response")
  
  ##naiveBayes
  dataset[] <-lapply(dataset, as.factor)
  train = dataset[-folds,] 
  test = dataset[folds,]
  model3 = naiveBayes(X2.1~.,data=train)                  
  model3.pred = predict(model3,test)
  model3.scores = score(test$X2.1, model3.pred)
  ##for Stacking and Cascading
  model3.pred.train = predict(model3,train)
  
  ##Tree Bayes
  model4<- tree.bayes(train,"X2.1")
  model4.pred=predict(model4 ,test)
  model4.scores = score(test$X2.1,model4.pred) 
  model4.pred
  ##for Stacking and Cascading
  model4.pred.train=predict(model4 ,train)
  
  ##KNN
  model5 = knn(train,test,train$X2.1,k=1)   
  model5.pred = #test model1
    model5.scores = score(test$X2.1, model5)
  ##for Stacking and Cascading
  model5.train = knn(train,train,train$X2.1,k=1) 
  
  ##Stacking
  
  Stack_Dataset <- data.frame(model1.pred.train,model2.pred.train$class,model3.pred.train,model4.pred.train,model5.train,X2.1 = train$X2.1)
  model6 = naiveBayes(X2.1~.,data=Stack_Dataset)
  model6.pred = predict(model6,test)
  model6.scores = score(test$X2.1, model6.pred)
  
  ##Cascade
  Cascade_Dataset <- data.frame(model1.pred.train,model2.pred.train$class,model3.pred.train,model4.pred.train,model5.train,train)
  model7 = naiveBayes(X2.1~.,data=Cascade_Dataset)
  model7.pred = predict(model7,test)
  model7.scores = score(test$X2.1, model7.pred)
  
  
  r <- rbind(model1.scores,model2.scores,model3.scores,model4.scores,model5.scores,model6.scores,model7.scores)
  rownames(r)= c('Logistic','LDA','naiveBayes','Tree Bayes','KNN','Stacking','Cascade')
  return (r)
  
}


nFolds = 10
folds <- createFolds(cancer_1$X2.1, k = nFolds) 
cv<-lapply(folds, trainAndTestFold, dataset=cancer_1)
cv

meanCV = Reduce('+',cv)/ nFolds
meanCV
stdCV = sqrt(Reduce('+',lapply(cv,function(x) (x - meanCV)^2)))
stdCV


