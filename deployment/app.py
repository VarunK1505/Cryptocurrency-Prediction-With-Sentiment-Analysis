from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np
import pandas as pd
from datetime import timedelta
import statsmodels
from nltk.classify import ClassifierI
from statistics import mode

import tweepy
from apikeys import *

app = Flask(__name__)

word_feature_path = open('../models/word_features.pickle', 'rb')
word_features = pickle.load(word_feature_path)
word_feature_path.close()

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
         features[w] = (w in words)
    return features

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers
        
    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
    
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

client = tweepy.Client(bearer_token=bearertoken)

@app.route('/')
@app.route('/home')
def home_page():
    return render_template('home.html')

@app.route('/bitcoin', methods=['GET', 'POST'])
def bitcoin_page():
    if request.method == 'POST':
        time_frame = request.form.get('time_frame')
        start_date = request.form.get('start')
        end_date = request.form.get('end')
        if(time_frame == 'daily'):
            dates = pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date),freq='d')
            pickle_in = open("../models/bitcoin_daily.pkl", 'rb')
            daily_model = pickle.load(pickle_in)
            pickle_in.close()
            pred = daily_model.predict(start=start_date, end=end_date)
        elif(time_frame == 'monthly'):
            dates = pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date),freq='m')
            pickle_in = open("../models/bitcoin_monthly.pkl", 'rb')
            monthly_model = pickle.load(pickle_in)
            pickle_in.close()
            pred = monthly_model.predict(start=start_date, end=end_date)
        
        return render_template('bitcoin.html', check=True ,start_date = start_date, end_date = end_date, labels = dates, predictions = pred)

    return render_template('bitcoin.html')

@app.route('/ethereum', methods=['GET', 'POST'])    
def ethereum_page():
    if request.method == 'POST':
        time_frame = request.form.get('time_frame')
        start_date = request.form.get('start')
        end_date = request.form.get('end')
        if(time_frame == 'daily'):
            dates = pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date),freq='d')
            pickle_in = open("../models/ethereum_daily.pkl", 'rb')
            daily_model = pickle.load(pickle_in)
            pickle_in.close()
            pred = daily_model.predict(start=start_date, end=end_date)
        elif(time_frame == 'monthly'):
            dates = pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date),freq='m')
            pickle_in = open("../models/ethereum_monthly.pkl", 'rb')
            monthly_model = pickle.load(pickle_in)
            pickle_in.close()
            pred = monthly_model.predict(start=start_date, end=end_date)
        
        return render_template('ethereum.html', check=True ,start_date = start_date, end_date = end_date, labels = dates, predictions = pred)

    return render_template('ethereum.html')

@app.route('/litecoin', methods=['GET', 'POST'])
def litecoin_page():
    if request.method == 'POST':
        time_frame = request.form.get('time_frame')
        start_date = request.form.get('start')
        end_date = request.form.get('end')
        if(time_frame == 'daily'):
            dates = pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date),freq='d')
            pickle_in = open("../models/litecoin_daily.pkl", 'rb')
            daily_model = pickle.load(pickle_in)
            pickle_in.close()
            pred = daily_model.predict(start=start_date, end=end_date)
        elif(time_frame == 'monthly'):
            dates = pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date),freq='m')
            pickle_in = open("../models/litecoin_monthly.pkl", 'rb')
            monthly_model = pickle.load(pickle_in)
            pickle_in.close()
            pred = monthly_model.predict(start=start_date, end=end_date)

        return render_template('litecoin.html', check=True ,start_date = start_date, end_date = end_date, labels = dates, predictions = pred)


    return render_template('litecoin.html')

@app.route('/sentiment', methods=['GET', 'POST'])
def sentiment_page():
    if request.method == 'POST':
        pickle_in = open("../models/naivebayes_final.pickle", 'rb')
        classifier = pickle.load(pickle_in)
        pickle_in.close()
        
        pickle_in = open("../models/MNB_classifier_final.pickle", 'rb')
        MNB_classifier = pickle.load(pickle_in)
        pickle_in.close()
        
        pickle_in = open("../models/bernoulliNB_classifier_final.pickle", 'rb')
        BernoulliNB_classifier = pickle.load(pickle_in)
        pickle_in.close()

        pickle_in = open("../models/LogisticRegression_classifier_final.pickle", 'rb')
        LogisticRegression_classifier = pickle.load(pickle_in)
        pickle_in.close()

        voted_classifier = VoteClassifier(classifier,
                                MNB_classifier, 
                                BernoulliNB_classifier, 
                                LogisticRegression_classifier)
        
        form_name = request.form['form-name']
        if (form_name == 'form1'):
            tweet = request.form.get('tweet')
            feature_set = find_features(tweet)
            classification = [voted_classifier.classify(feature_set), voted_classifier.confidence(feature_set)]
            return render_template('sentiment.html', check = 2, classification = classification)
        if (form_name == 'form2'):
            currency = request.form.get('currency')
            query = currency + ' lang:en'
            tweets = client.search_recent_tweets(query=query, max_results=100)
            X = []
            Y = []
            x  = 0
            y = 0
            for tweet in tweets.data:
                x += 1
                raw_tweet = tweet.text
                clean_tweet = raw_tweet.replace("\n", "")
                feature_set = find_features(clean_tweet)
                classification = voted_classifier.classify(feature_set)
                if(classification == 'positive'):
                    y += 1
                elif(classification == 'negative'):
                    y -= 1
                else:
                    continue
                X.append(x)
                Y.append(y)
            return render_template('sentiment.html', check=1, X = X, Y = Y, currency = currency)
    return render_template('sentiment.html')