import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import *
from sklearn.feature_extraction.text import TfidfTransformer

def get_trained_NB_classifier(feature_name):
	features = ['Screen_name','Location','Description','Url','Followers_count','Friends_count','Listed_count','Created_at','Favourites_count','Verified','Statuses_count','Lang','Status','Default_profile','Default_profile_image','Has_extended_profile','name']
	numfeatures = ['Followers_count', 'Friends_count', 'Listed_count','Favourites_count','Statuses_count']
	numfeatures2 = ['followers_count', 'friends_count', 'listedcount','favourites_count','statuses_count']

	botdata = pd.read_csv('botAccounts.csv',encoding = 'ISO-8859-1')
	gooddata= pd.read_csv('GoodAccounts.csv', encoding = 'ISO-8859-1')


	all_data = botdata.append(gooddata)
	all_data = all_data.fillna("")
	numbers_data = all_data[numfeatures].as_matrix()
	matrix_data = all_data.as_matrix()
	count_vector = CountVectorizer()
	term_document_matrix = count_vector.fit_transform(all_data[feature_name])
	tf_transformer = TfidfTransformer(use_idf=False).fit(term_document_matrix)
	transformed_data = tf_transformer.transform(term_document_matrix)
	bayes = GaussianNB().fit(transformed_data.todense(), all_data["Bot"])
	#bayes = GaussianNB().fit(numbers_data, all_data["Bot"])
	bayes_predict = bayes.predict(transformed_data.todense())
	#bayes_predict = bayes.predict(all_data[numfeatures])

	bot_test = pd.read_csv('bot.csv',encoding = 'ISO-8859-1')
	good_test = pd.read_csv('good.csv',encoding = 'ISO-8859-1')
	all_test = bot_test.append(good_test).fillna("")
	test_doc = count_vector.transform(all_test[feature_name.lower()])

	#count_vector2 = CountVectorizer()
	#test_doc = count_vector2.fit_transform(all_test["location"])
	tf_transformer2 = TfidfTransformer(use_idf=False).fit(test_doc)
	transformed_data2 = tf_transformer2.transform(test_doc)


	accuracy = accuracy_score(all_data["Bot"], bayes_predict)
	print("Accuracy: ", accuracy)
	precision = precision_score(all_data["Bot"], bayes_predict)
	print("Precision: " + str(precision))
	recall = recall_score(all_data["Bot"], bayes_predict)
	print("Recall: " + str(recall))
	F1 = 2 * (precision * recall) / (precision + recall)
	print("Fl: " + str(F1))
	AUC = roc_auc_score(all_data["Bot"], bayes_predict)
	print("AUC: ", AUC)

	print(transformed_data2.shape)

	bayes_predict2 = bayes.predict(transformed_data2.todense())
	#bayes_predict2 = bayes.predict(all_test[numfeatures2])
	accuracy = accuracy_score(all_test["bot"], bayes_predict2)
	print("Accuracy: ", accuracy)
	precision = precision_score(all_test["bot"], bayes_predict2)
	print("Precision: " + str(precision))
	recall = recall_score(all_test["bot"], bayes_predict2)
	print("Recall: " + str(recall))
	F1 = 2 * (precision * recall) / (precision + recall)
	print("Fl: " + str(F1))
	AUC = roc_auc_score(all_test["bot"], bayes_predict2)
	print("AUC: ", AUC)

	return bayes




