# Load Libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse

# Read Data
path = '/Users/liwang/desktop/ml/machine-learning-project/data/'
df = pd.read_csv(path + 'data.csv')
df.head(2)

# Only keep these columns in data: 'Followers_count', 'Friends_count', 'Listed_count',
#'Favourites_count','Statuses_count, 'Bot'
df = df.drop(df.columns[[0,1,2,3,4,5,9,11,13,14,15,16,17,18]], axis=1)
# Print Summary Table of Data
print(df.describe())

# Train-Test-Split Data
X_train, X_test, y_train, y_test = train_test_split(df.ix[:,0:-1], df.bot, stratify=df.bot, random_state=42)

# Logistic Regression with Regularization (Default C = 1)
logreg = LogisticRegression().fit(X_train, y_train)
print(logreg.coef_.T)
print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg.score(X_test, y_test)))

# Logistic Regression with Regularization (C = 100)
logreg1 = LogisticRegression(C = 100).fit(X_train, y_train)
print("Training set score for C = 100: {:.3f}".format(logreg1.score(X_train, y_train)))
print("Test set score for C = 100: {:.3f}".format(logreg1.score(X_test, y_test)))

# Logistic Regression with Regularization (C = 0.001)
logreg2 = LogisticRegression(C = 0.001).fit(X_train, y_train)
print("Training set score for C = 0.001: {:.3f}".format(logreg2.score(X_train, y_train)))
print("Test set score for C = 0.001: {:.3f}".format(logreg2.score(X_test, y_test)))

# Compute confusion matrix
y_pred = logreg.predict(X_train)
print("Confusion Matrix for C=1: ")
print(confusion_matrix(y_train, y_pred))

# Precision, Recall, Fl Scores
print(precision_score(y_train, y_pred))
print(recall_score(y_train, y_pred))
print(f1_score(y_train, y_pred))

# Top 3 weight coefficients in our learning model
top3 = logreg.coef_[0].argsort()[-3:][::-1]
attributes_list = list(df.columns.values)
names = [attributes_list[index] for index in top3]
print("Top 3: ")
print(names)

#If we desire a more interpretable model, using L1 regularization might help, as it
#limits the model to using only a few features. 
logreg3 = LogisticRegression(penalty='l1').fit(X_train, y_train)
print("Training set score: {:.3f}".format(logreg3.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg3.score(X_test, y_test)))

# Plot Coefficients of Logistic Regression (L2, C = 1)
plt.plot(logreg.coef_.T, 'o', label="C=1")
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.legend()
plt.show()