import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load Dataset
data = pd.read_csv('spam.csv', encoding='latin-1')
data = data[['label', 'message']]
data.columns = ['label', 'message']

# Convert labels to binary (spam=1, ham=0)
data['label'] = data['label'].map({'spam': 1, 'ham': 0})

# Text Preprocessing
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X = vectorizer.fit_transform(data['message'])
y = data['label']

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naive Bayes Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict and Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Test with custom message
def predict_spam(message):
    vector = vectorizer.transform([message])
    prediction = model.predict(vector)
    return 'Spam' if prediction[0] == 1 else 'Not Spam'

# Test Message
message = input("Enter a message to classify: ")
print("Result:", predict_spam(message))