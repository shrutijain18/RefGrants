import numpy as np
import pandas as pd
import seaborn as sns
#Importing the Dataset
train = pd.read_csv("H1B_Final1.csv")
#Displaying rows and columns and rows
train.head()

#Train the dataset where X is to train to predict Y
X = train [['Job_Level', 'Skill_Set', 'Work_Experience', 'Language']]
Y = train['Visa_Status']

#Splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=400)

#Using Logistic Regression
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
predictions= logmodel.predict(X_test)

#Presenting the classificaton report whre
# 0 = Organizations not sponsoring H1B Visa
# 1 = Organizations sponsoring H1B Visa
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))

#Displaying the Accuracy score
from sklearn import metrics
print(metrics.accuracy_score(y_test, predictions))
