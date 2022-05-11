# Problem Statement
  A cryptocurrency is an encrypted data string that denotes a unit of currency. It is monitored and organized by a peer-to-peer network called a blockchain, which also serves as a secure ledger of transactions, e.g., buying, selling, and transferring. Unlike physical money, cryptocurrencies are decentralized, which means they are not issued by governments or other financial institutions. Here in this machine learning project, we are going to be predicting the future price of 3 cryptocurrencies (Ethereum, Bitcoin, Litecoin) using time series forecasting and along with it perform sentiment analysis to know how favourable the market is. 
  
# Dataset
For this project we have used a total of 4 dataset in which three are historical data of the cryptocurrencies taken from (https://in.investing.com/crypto) and the tweets dataset which has all the financial tweets (https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis) which we have used to train various ML models.
The historical price dataset has the following features:
- Date
- Price
- Open
- High
- Low	Vol.
- Change %

The tweets dataset has the following features:
- Sentence - The actual Tweet
- Sentiment - What is the sentiment of that tweet (Positive, neutral, negative)

# Models used:
  For the future time series forecasting we first tried a ARIMA model, ARIMA is an acronym that stands for Autoregressive Integrated Moving Average. But as ARIMA does not take seasonality into account that caused it to give poor results, Hence we then moved over to SRIMA. Seasonal Autoregressive Integrated Moving Average, SARIMA or Seasonal ARIMA, is an extension of ARIMA that explicitly supports univariate time series data with a seasonal component. This model has a total of 3 hyperparameters: 
- p: Trend autoregression order.
- d: Trend difference order.
- q: Trend moving average order.
We tuned them using the acf and pacf (Autocorrelation and partial autocorrelation) charts. This gave us much better results. We finally trained on the complete data and pickled the model. 

For the sentiment analysis part we made use of SklearnClassifier() module of nltk to train MultinomialNB, BernoulliNB, LogisticRegression, SVC, LinearSVC, SGDClassifier and pickled all of these. Then we made a vote classifier which takes all these classifiers and predicts using all of them and then returns the mode of all their outputs. Now we use tweepy library to stream real time tweets from twitter about any particular crypto and then predict them.

# Contributors:
Nihal Srivastava (https://github.com/Nihal-Srivastava05)  
Varun Kamath (https://github.com/VarunK1505)  
Ronit Naik (https://github.com/RonitNaik9)  
