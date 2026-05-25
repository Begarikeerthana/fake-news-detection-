# Fake News Detection App

A deep learning web application that detects whether a news article is **Real or Fake** using an LSTM neural network built with TensorFlow and Keras, deployed as an interactive app using Streamlit.

## Demo Screenshots

### Real News Detection
![Real News](screenshots/real_news.png)

### Fake News Detection
![Fake News](screenshots/fake_news.png)

## About the Project

With the rise of misinformation and fake news on the internet, this project aims to automatically classify news articles as real or fake using Natural Language Processing and Deep Learning techniques.

The model was trained on a dataset of over 44,000 news articles and achieved **99% accuracy** on the test set.

## Tech Stack

- **Python** - Core programming language
- **TensorFlow / Keras** - Deep learning framework
- **LSTM (Long Short-Term Memory)** - Neural network architecture
- **Streamlit** - Web app framework
- **Pandas / NumPy** - Data processing
- **Scikit-learn** - Train-test split
- **Pickle** - Tokenizer serialization

## Model Architecture

- Embedding Layer (5000 vocab, 128 dimensions)
- LSTM Layer (64 units)
- Dense Output Layer (Sigmoid activation)
- Loss: Binary Crossentropy
- Optimizer: Adam
- Epochs: 5
- Test Accuracy: 99%

## Dataset

The model was trained on the Fake and Real News Dataset from Kaggle:
- 23,481 fake news articles
- 21,417 real news articles
- Total: 44,898 articles

Dataset link: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

## How to Run Locally

1. Clone the repository:
   git clone https://github.com/Begarikeerthana/fake-news-detection-.git

2. Navigate to the project folder:
   cd fake-news-detection-

3. Install dependencies:
   pip install -r requirements.txt

4. Run the app:
   streamlit run app.py

5. Open browser at:
   http://localhost:8501

## How to Use

1. Open the app in your browser
2. Paste a full news article (minimum 30 words) in the text box
3. Click the Predict button
4. The app will display:
   - A prediction score (0 to 1)
   - Score above 0.5 = Real News
   - Score below 0.5 = Fake News

## Note

This model works best with full news articles (30+ words). Short headlines may not give accurate results as the model was trained on complete articles.

## Project Structure

- app.py - Streamlit web application
- FakeNewsDetection.ipynb - Model training notebook
- fake_news_model.keras - Trained model file
- tokenizer.pkl - Saved tokenizer
- requirements.txt - Python dependencies

## Author

Begari Keerthana
ECE Final Year Student, Vasavi College of Engineering, Hyderabad
GitHub: https://github.com/Begarikeerthana
