import streamlit as st
import nltk
from nltk.corpus import movie_reviews
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Download NLTK movie reviews dataset
nltk.download('movie_reviews')

# Load and prepare data
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

texts = [" ".join(words) for words, label in documents]
labels = [label for words, label in documents]

# Vectorize text
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(texts)
y = labels

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Define the function for sentiment prediction
def predict_sentiment(text):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    return prediction

# Streamlit UI
st.title('Sentiment Analysis Web App')
st.write('Enter a sentence to get the sentiment (pos, neg, neu)')

# Input box for user to enter sentence
user_input = st.text_input('Enter a sentence:')

# When the user enters a sentence, predict and display sentiment
if user_input:
    sentiment = predict_sentiment(user_input)
    if sentiment == 'pos':
        st.success('Sentiment: Positive')
    elif sentiment == 'neg':
        st.error('Sentiment: Negative')
    else:
        st.warning('Sentiment: Neutral')
