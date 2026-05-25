import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = tf.keras.models.load_model('fake_news_model.keras')

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

st.title('Fake News Detection App')
st.write('Enter a news article below (minimum 30 words):')

news = st.text_area('News Text')

if st.button('Predict'):
    if not news.strip():
        st.warning('Please enter some text.')
    else:
        seq = tokenizer.texts_to_sequences([news])
        padded = pad_sequences(seq, maxlen=200)
        score = model.predict(padded)[0][0]
        st.write(f'Score: {score:.4f}')
        if score > 0.5:
            st.success('This looks like REAL news')
        else:
            st.error('This looks like FAKE news')
