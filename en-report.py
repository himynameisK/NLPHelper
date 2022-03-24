import numpy as np
import pandas as pd
import re
import nltk
import joblib


with open('C:/Users/krezn/PycharmProjects/NLPHelper/stopwords/english', 'r') as fp:
    # считываем сразу весь файл
    data = list(fp.read())

data_source_url = "C:/Users/krezn/PycharmProjects/NLPHelper/Tweets.csv"

airline_tweets = pd.read_csv(data_source_url)
features = airline_tweets.iloc[:, 10].values
labels = airline_tweets.iloc[:, 1].values
processed_features = []

for sentence in range(0, len(features)):
    # Remove all the special characters
    processed_feature = re.sub(r'\W', ' ', str(features[sentence]))
    # remove all single characters
    processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)
    # Remove single characters from the start
    processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature)
    # Substituting multiple spaces with single space
    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)
    # Removing prefixed 'b'
    processed_feature = re.sub(r'^b\s+', '', processed_feature)
    # Converting to Lowercase
    processed_feature = processed_feature.lower()
    processed_features.append(processed_feature)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer ( min_df=1, max_df=0.8, stop_words=data)
processed_features = vectorizer.fit_transform(processed_features).toarray()






filename = 'finalized_model.sav'

loaded_model = joblib.load(filename)

test_url = "C:/Users/krezn/PycharmProjects/NLPHelper/test.csv"
makima = pd.read_csv(test_url)
features = makima.values

test_features = []

for sentence in range(0, len(features)):
    # Remove all the special characters
    test_feature = re.sub(r'\W', ' ', str(features[sentence]))
    # remove all single characters
    test_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', test_feature)
    # Remove single characters from the start
    test_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', test_feature)
    # Substituting multiple spaces with single space
    test_feature = re.sub(r'\s+', ' ', test_feature, flags=re.I)
    # Removing prefixed 'b'
    test_feature = re.sub(r'^b\s+', '', test_feature)
    # Converting to Lowercase
    test_feature = test_feature.lower()
    test_features.append(test_feature)

print(test_features)

test_features = vectorizer.transform(test_features).toarray()

predictions = loaded_model.predict(test_features)

print(predictions)