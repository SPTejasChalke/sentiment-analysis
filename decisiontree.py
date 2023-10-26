# Import necessary libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, precision_score
import joblib
import pandas as pd

data = pd.read_csv('output_file.csv')

texts = data['review'].tolist()
labels = data['sentiment'].tolist()

# Text vectorization using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Save the TF-IDF vectorizer
joblib.dump(vectorizer, 'dtree_tfidf_vectorizer.pkl')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Create and train the Decision Tree model
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = dt_model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')  # You can choose 'micro', 'macro', 'weighted', or 'samples' for the average parameter
roc_auc = roc_auc_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')

# Print the evaluation metrics
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print(f"F1 Score: {f1:.2f}")
print(f"AUC Score: {roc_auc:.2f}")
print(f"Recall: {recall:.2f}")
print(f"Precision: {precision:.2f}")

# Save the Decision Tree model to a file
joblib.dump(dt_model, 'decision_tree_sentiment_model.pkl')
