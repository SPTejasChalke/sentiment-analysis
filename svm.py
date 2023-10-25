# Import necessary libraries
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib

# Sample data (replace this with your data)
texts = ["This is a positive sentence.", "This is a negative sentence.", "This is a neutral sentence."]
labels = [1, 0, 2]  # 1 for positive, 0 for negative, 2 for neutral

# Text vectorization using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Create and train the SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Evaluate the model (optional)
accuracy = svm_model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the SVM model to a file
joblib.dump(svm_model, 'svm_sentiment_model.pkl')
