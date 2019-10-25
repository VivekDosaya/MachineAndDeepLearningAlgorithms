# Importing the dataset
dataset=read.csv('Salary_Data.csv')
#dataset=dataset[,2:3]

#Splitting the dataset into training set and test set
#install.packages('caTools')
set.seed(123)
split=sample.split(dataset$Salary,SplitRatio = 2/3)
training_set=subset(dataset,split==TRUE)
test_set=subset(dataset,split==FALSE)

#Feature Scaling
#training_set[,2:3]=scale(training_set[,2:3])
#test_set[,2:3]=scale(test_set[,2:3])


#Fitting simple regression to training set
regressor=lm(formula = Salary ~ YearsExperience,data=training_set)

#predicting the test set results
y_pred=predict(regressor,newdata=test_set)

#visualising training set results
#install.packages('ggplot2')
ggplot() +
  geom_point(aes(x=training_set$YearsExperience,y=training_set$Salary),colour='red') +
  geom_line(aes(x=training_set$YearsExperience,y=predict(regressor,newdata=training_set)),colour='blue')+
  ggtitle('salary vs experience')+
  xlab('yearsExperience')+
  ylab('Salary')

#visualising test set results
ggplot() +
  geom_point(aes(x=test_set$YearsExperience,y=test_set$Salary),colour='red') +
  geom_line(aes(x=training_set$YearsExperience,y=predict(regressor,newdata=training_set)),colour='blue')+
  ggtitle('salary vs experience')+
  xlab('yearsExperience')+
  ylab('Salary')

