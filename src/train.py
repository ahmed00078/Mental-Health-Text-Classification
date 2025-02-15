import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import re

# Load the data
train_df = pd.read_csv('../data/train_cleaned.csv')
test_df = pd.read_csv('../data/test.csv')

# Basic text preprocessing function
def preprocess_text(text):
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    return ''

# Combine title and content for better context
train_df['combined_text'] = train_df['title'].fillna('') + ' ' + train_df['content'].fillna('')
test_df['combined_text'] = test_df['title'].fillna('') + ' ' + test_df['content'].fillna('')

# Preprocess the text
train_df['combined_text'] = train_df['combined_text'].apply(preprocess_text)
test_df['combined_text'] = test_df['combined_text'].apply(preprocess_text)

# Create TF-IDF features
tfidf = TfidfVectorizer(max_features=10000)
X_train = tfidf.fit_transform(train_df['combined_text'])
X_test = tfidf.transform(test_df['combined_text'])

# Train the model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, train_df['target'])

# Make predictions on test set
predictions = model.predict(X_test)

# Create submission file
submission = pd.DataFrame({
    'ID': test_df['id'],
    'Target': predictions
})

# Save submission file
submission.to_csv('submission_two.csv', index=False)

# Print sample of predictions
print("\nSample of predictions:")
print(submission.head())

# Optional: Quick validation score
# Split training data to get a validation score
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, train_df['target'], test_size=0.2, random_state=42
)

model_val = LogisticRegression(max_iter=1000, random_state=42)
model_val.fit(X_train_split, y_train_split)
val_predictions = model_val.predict(X_val)
val_accuracy = accuracy_score(y_val, val_predictions)
print(f"\nValidation Accuracy: {val_accuracy:.4f}")