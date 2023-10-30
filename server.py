from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
CORS(app)

# Load the saved model
model = joblib.load('svm_sentiment_model.pkl')

vectorizer = joblib.load('svm_tfidf_vectorizer.pkl')

@app.route('/predict_sentiment', methods=['GET'])
def predict_sentiment():
    text = request.args.get('text')

    text_vectorized = vectorizer.transform([text])
    sentiment = model.predict(text_vectorized)[0]
    sentiment = int(sentiment)
    
    # 0: bad, 1: good
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
