from flask import Flask, request, jsonify
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the saved model
model = joblib.load('svm_sentiment_model.pkl')

vectorizer = joblib.load('svm_tfidf_vectorizer.pkl')

@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment():
    data = request.get_json()
    text = data['text']

    text_vectorized = vectorizer.transform([text])
    sentiment = model.predict(text_vectorized)[0]
    sentiment = int(sentiment)
    
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
