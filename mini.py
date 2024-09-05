import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# Load the CSV data into a DataFrame
df = pd.read_csv('dataset.csv')

# Remove irrelevant tweets
df = df[df['Sentiment'] != 'irrelevant']

# Text normalization and tokenization
def preprocess_tweet(tweet):
    # Convert to lowercase
    tweet = tweet.lower()

    # Remove URLs
    tweet = re.sub(r"http\S+|www\S+|https\S+", "", tweet, flags=re.MULTILINE)

    # Remove mentions and hashtags
    tweet = re.sub(r"@\w+|#\w+", "", tweet)

    # Remove punctuation
    tweet = tweet.translate(str.maketrans("", "", string.punctuation))

    # Tokenization using TweetTokenizer
    tokenizer = TweetTokenizer()
    tokens = tokenizer.tokenize(tweet)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Join tokens back to a single string
    processed_tweet = ' '.join(tokens)

    return processed_tweet

df['TweetText'] = df['TweetText'].apply(preprocess_tweet)

print(df['TweetText'])

# Splitting the dataset into features (X) and labels (y)
X = df['TweetText']
y = df['Sentiment']

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Splitting the data into training and evaluation sets
X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, random_state=42)

# Training a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)
# Applying the trained model to predict labels for new, unseen tweets
y_pred = model.predict(X_eval)
 
from sklearn.metrics import classification_report

# Evaluating the model performance
print(classification_report(y_eval, y_pred))

# Analyzing the distribution of predicted labels
import matplotlib.pyplot as plt
import numpy as np

labels, counts = np.unique(y_pred, return_counts=True)
plt.bar(labels, counts)
plt.xlabel('Labels')
plt.ylabel('Counts')
plt.show()

# Exploring the important features contributing to positive and negative sentiments
feature_names = vectorizer.get_feature_names_out()
coefs = model.coef_[0]

top_positive_features = [feature_names[i] for i in np.argsort(coefs)[:10]]
top_negative_features = [feature_names[i] for i in np.argsort(coefs)[-10:]]

print("Top positive features:", top_positive_features)
print("Top negative features:", top_negative_features)

def detect_bias(sentiment_distribution):
    # Calculate the proportion of positive, negative, and neutral sentiments
    positive_proportion = sentiment_distribution['positive']
    negative_proportion = sentiment_distribution['negative']
    neutral_proportion = sentiment_distribution['neutral']

    # Determine the bias based on the sentiment proportions
    if positive_proportion > negative_proportion and positive_proportion > neutral_proportion:
        bias = "Positive Bias"
    elif negative_proportion > positive_proportion and negative_proportion > neutral_proportion:
        bias = "Negative Bias"
    else:
        bias = "Neutral Bias"

    return bias

# Calculate the distribution of sentiments
sentiment_distribution = df['Sentiment'].value_counts(normalize=True)

# Detect bias in sentiment analysis
bias = detect_bias(sentiment_distribution)
print("Bias Detected:",bias)
