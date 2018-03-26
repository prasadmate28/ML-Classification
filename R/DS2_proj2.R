library(readr)

score <- function(expected,predicted){
  expected = as.logical(expected)
  predicted = as.logical(predicted)
  accuracy = 
  recall =
  precision = 
  f.score = 
  r = cbind(accuracy,recall,precision,f.score)
  colnames(r)<- c('accuracy','recall','precision','fscore')
  return(r)
}

trainNtestFold <- function(fold, data){
  train =  data[-fold,]
  test = data[fold,]
  
  model1 = #train model1
  model1.pred = #test model1
  model1.scores = score(test$Purchase, model1.pred)
  
  model2 = #train model1
  model2.pred = #test model1
  model2.scores = score(test$Purchase, model2.pred)
  
  model3 = #train model1
  model3.pred = #test model1
  model3.scores = score(test$Purchase, model3.pred)
  
  model4 = #train model1
  model4.pred = #test model1
  model4.scores = score(test$Purchase, model4.pred)

  model5 = #train model1
  model5.pred = #test model1
  model5.scores = score(test$Purchase, model5.pred)
  
  r <- rbind(model1.scores, model2.score, model3.score, model4.score, model5.score)
  rownames(r)= c('KNN','NaiveBayes','Logistic','LDA','TanBayes')
  return (r)
  
}

caravan <- read_csv("IdeaProjects/ds2 classwork/caravan.csv")
nFolds = 10
folds <- createFolds(caravan$Purchase, k= nFolds)
cv <- lapply(folds,trainNtestFold, dataset=caravan)
meanCV = Reduce('+',cv)
stdCV = sqrt(Reduce('+',lapply(cv,function(x) (x - meanCV)^2)))