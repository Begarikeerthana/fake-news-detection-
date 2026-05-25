import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Load trained model
model = tf.keras.models.load_model("fake_news_model.keras")

# Create tokenizer
tokenizer = Tokenizer(num_words=5000)

# App title
st.title("Fake News Detection App")

st.write("Enter a news headline or article below:")

# User input
news = st.text_area("News Text")

# Predict button
if st.button("Predict"):

    # Convert text to sequence
    seq = tokenizer.texts_to_sequences([news])

    # Padding
    padded = pad_sequences(seq, maxlen=200)

    # Prediction
    prediction = model.predict(padded)

    # Result
    if prediction[0][0] > 0.5:
        st.success("This looks like REAL news")
    else:
        st.error("This looks like FAKE news")