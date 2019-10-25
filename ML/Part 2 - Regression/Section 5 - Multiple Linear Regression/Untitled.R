#MUltiple Linear Regression

# Data Preprocessing Template

# Importing the dataset
dataset=read.csv('50_Startups.csv')
#dataset=dataset[,2:3]

# Encoding categorical data
dataset$State = factor(dataset$State,
                         levels = c('New York', 'California', 'Florida'),
                         labels = c(1, 2, 3))

#Splitting the dataset into training set and test set
#install.packages('caTools')
set.seed(123)
split=sample.split(dataset1$Profit,SplitRatio = 0.8)
training_set=subset(dataset,split==TRUE)
test_set=subset(dataset,split==FALSE)

#Feature Scaling
#training_set[,2:3]=scale(training_set[,2:3])
#test_set[,2:3]=scale(test_set[,2:3])

#Fitting Multiple Linear Regression to the Training set
regressor = lm(formula = Profit ~ .,
               data = training_set)

#when the regressor is checked it is found that only r&d spend is important

regressor_opt = lm(formula = Profit ~ R.D.Spend+Administration+Marketing.Spend+State,
               data = dataset)
#predicting the test set results

y_pred=predict(regressor,newdata=test_set)

#Building optimal model using backward elimination




