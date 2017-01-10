#importing data set
bcw=read.csv(file.choose())

#viewing the details of the data set
summary(bcw)
str(bcw)

#attaching the data set
attach(bcw)

#removing the log variable created while importing
bcw=subset(bcw,select = -X)

#checking the missing values in the variables
mv_bcw=sapply(bcw, function(x){sum(is.na(x))/length(x)}*100)
mv_bcw
#inference-no missing values

#checking the proportion of the target variable-"diagnosis"
round(prop.table(table(bcw$diagnosis))*100)
#inference-B(Benign)=63% and M(Malignant)=37%

#removing the id variable
bcw=subset(bcw,select = -id)

#distribution of the numeric variables
bcw_num=subset(bcw,select = -diagnosis)

#histogram
for (column in bcw_num) {
  dev.new()
  hist(column)
}

#boxplot
for (column in bcw_num) {
  dev.new()
  boxplot(column)
}
#inference-most of the variables are skewed

#transforming the target variable(diagnosis) as M==1 and B==0
bcw$diagnosis=ifelse(bcw$diagnosis == 'M', 1, 0)
bcw$diagnosis=as.factor(bcw$diagnosis)

#installing packages to plot the correlation matrix
install.packages("ggplot2")
install.packages("dplyr")
library("ggplot2")
library("dplyr")
library("plyr")

#plotting variables to check the correlation
plot(bcw[c(2:6)], col = bcw$diagnosis)
plot(bcw[c(7:11)], col = bcw$diagnosis)
plot(bcw[c(12:16)], col = bcw$diagnosis)
plot(bcw[c(17:21)], col = bcw$diagnosis)
plot(bcw[c(22:26)], col = bcw$diagnosis)
plot(bcw[c(27:31)], col = bcw$diagnosis)

#Remove the variables that have correlation coefficient > .7
#Variable reduction

#Apply cor function
descr_cor=cor(bcw_num)

#loading package "caret" for calculating correlation between numeric variables
install.packages("caret")
library("caret")

#setting cutoff value of correlation coeff=0.7
ax=findCorrelation(descr_cor, cutoff = 0.7)

#displaying highly correlated variables
highly_cor_var=colnames(bcw_num[ax])
highly_cor_var

#removing the highly correlated variables
bcw_num1=subset(bcw[,-ax])
#bcw_num1=subset(bcw_num,select=-c("concavity_mean","concave.points_mean","compactness_mean","concave.points_worst","concavity_worst","perimeter_worst","radius_worst","perimeter_mean","compactness_worst","area_worst","radius_mean","perimeter_se","radius_se","concave.points_se","compactness_se","area_se","concavity_se","smoothness_mean","fractal_dimension_mean","texture_worst"))

#Performing log transformation to remove skewness in the distribution of the variables
bcw_num1_log=log(bcw_num1)

#plotting the histogram for each variable to view the distribution after log tranformation
for (column in bcw_num1_log) {
  dev.new()
  hist(column)
}

#combining the target variable and the uncorrelated variables
bcw_full=cbind(bcw_num1_log,bcw$diagnosis)

#renaming the target variable
bcw_full=rename(bcw_full, c("bcw$diagnosis"="diagnosis"))

#setting seed for reproducibility
set.seed(1234)

#Splitting the data into train(70%) and test(30%)
#index=createDataPartition(diagnosis, p=0.75, list = F)
index=sample (1:nrow(bcw_full), 399)

#Define the train and test frame
bcw_train=bcw_full[index,]
bcw_test=bcw_full[-index,]

#importing packages to make models
library("randomForest")
library("gbm")
library("e1071")

attach(bcw_train)

#Random Forest
#creating a random forest model-1
rf=randomForest(diagnosis~.,data=bcw_train,n.trees=250,interaction.depth=7,importance=T,proximity=T)

#predicting the class for the test data set
tree.pred=predict(rf,bcw_test,type="class")

#creating confusion matrix
table(tree.pred,bcw_test$diagnosis) 

#creating a random forest model-2
rf=randomForest(diagnosis~.,data=bcw_train,n.trees=500,interaction.depth=8,importance=T,proximity=T)

#predicting the class for the test data set
tree.pred=predict(rf,bcw_test,type="class")

#creating confusion matrix
table(tree.pred,bcw_test$diagnosis) 

#tuning the parameters of random forest
rf_tune=tune.randomForest(diagnosis~.,data=bcw_train, n.trees=c(200,500,750), interaction.depth=c(6,7,8))
print(rf_tune)
#inference-Error estimation of 'randomForest' using 10-fold cross validation: 0.05269231

#Support Vector Machine
#creating a svm model-1
svm_clf=svm(diagnosis~.,data = bcw_train,kernel="linear")
summary(svm_clf)

svm.pred=predict(svm_clf,bcw_test)
table(svm.pred,bcw_test$diagnosis)

#creating a svm model-2
svm_clf=svm(diagnosis~.,data = bcw_train,kernel="poly",degree=2,gamma=100,cost=0.5)
summary(svm_clf)

svm.pred=predict(svm_clf,bcw_test)
table(svm.pred,bcw_test$diagnosis)

#creating a svm model-3
svm_clf=svm(diagnosis~.,data = bcw_train,kernel="radial",gamma=1,cost=0.05)
summary(svm_clf)

svm.pred=predict(svm_clf,bcw_test)
table(svm.pred,bcw_test$diagnosis)

#tuning SVM parameters
svm_clf_tune=tune.svm(diagnosis~.,data = bcw_train, kernel=c("poly"), degree=c(2,3),cost=c(0.3,1.1), gamma=c(1,5,10))
print(svm_clf_tune)
#inference-Parameter tuning of 'svm':

#- sampling method: 10-fold cross validation 

#- best parameters:
#  degree gamma cost
#     3     1    0.3

#- best performance: 0.07262821 

svm_clf=svm(diagnosis~.,data = bcw_train,kernel="poly",degree=3,gamma=1,cost=0.3)
summary(svm_clf)

svm.pred=predict(svm_clf,bcw_test)
svm.pred
table(svm.pred,bcw_test$diagnosis)

