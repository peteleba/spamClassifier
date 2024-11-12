import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
import nltk
from nltk.corpus import stopwords
import re

# Download stopwords if you haven't already
nltk.download('stopwords')

# Step 1: Load the dataset
data = pd.read_csv("spam.csv", encoding="latin-1")
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Step 2: Preprocess the text data
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

data['message'] = data['message'].apply(preprocess_text)

# Step 3: Convert text data to numerical data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['message']).toarray()
y = data['label'].apply(lambda x: 1 if x == 'spam' else 0)

# Step 4: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

# Step 7: Test the model with a new message
def classify_message(message):
    message = preprocess_text(message)
    message_vectorized = vectorizer.transform([message]).toarray()
    prediction = model.predict(message_vectorized)
    return "Spam" if prediction[0] == 1 else "Not Spam"

# Example message
new_message = "Congratulations! You've won a free ticket."
print(f"Message: '{new_message}' is classified as: {classify_message(new_message)}")
