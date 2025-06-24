import streamlit as st
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
import tensorflow as tf


# Load models
w2v_model = Word2Vec.load("word2vec.model")
model = tf.keras.models.load_model("sentiment.keras")
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Preprocess text
def preprocess(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text.lower())
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens

# Convert tokens to average Word2Vec vector
def avg_word2vec(tokens):
    vectors = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
    if not vectors:
        return np.zeros(w2v_model.vector_size)
    return np.mean(vectors, axis=0)

# Streamlit UI
st.title("🎬 IMDB Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment.")

review = st.text_area("Movie Review", "")

if st.button("Predict"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        tokens = preprocess(review)
        vector = avg_word2vec(tokens).reshape(1, -1)
        prediction = model.predict(vector)[0][0]

        if prediction >= 0.8:
            sentiment = "Very Good 😊"
        elif prediction >= 0.6:
            sentiment = "Good 🙂"
        elif prediction >= 0.4:
            sentiment = "Neutral 😐"
        elif prediction >= 0.2:
            sentiment = "Bad 😕"
        else:
            sentiment = "Very Bad 😠"

        st.success(f"**Sentiment:** {sentiment}")
        st.info(f"Prediction Score: `{prediction:.2f}`")
