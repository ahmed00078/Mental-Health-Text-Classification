import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import re
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Load the data
train_df = pd.read_csv('../data/train_cleaned.csv')
test_df = pd.read_csv('../data/test.csv')

# Enhanced text preprocessing
def preprocess_text(text):
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Replace common contractions
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"'m", " am", text)
        text = re.sub(r"'s", " is", text)
        text = re.sub(r"'re", " are", text)
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^a-zA-Z\s.,!?]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    return ''

# Combine and preprocess text
print("Preprocessing data...")
train_df['combined_text'] = (train_df['title'].fillna('') + ' . ' + 
                            train_df['content'].fillna('')).apply(preprocess_text)
test_df['combined_text'] = (test_df['title'].fillna('') + ' . ' + 
                           test_df['content'].fillna('')).apply(preprocess_text)

# Create pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )),
    ('classifier', LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    ))
])

# Perform k-fold cross-validation
print("\nPerforming cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, train_df['combined_text'], train_df['target'], 
                           cv=cv, scoring='accuracy', n_jobs=-1)

print(f"\nCross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Train final model on full training data
print("\nTraining final model...")
pipeline.fit(train_df['combined_text'], train_df['target'])

# Make predictions on test set
print("Making predictions...")
predictions = pipeline.predict(test_df['combined_text'])

# Create submission file
submission = pd.DataFrame({
    'ID': test_df['id'],
    'Target': predictions
})

# Save submission file
submission_file = 'submission_improved.csv'
submission.to_csv(submission_file, index=False)
print(f"\nSubmission saved to {submission_file}")

# Print class distribution in predictions
print("\nClass distribution in predictions:")
print(pd.Series(predictions).value_counts(normalize=True))

# Print sample of predictions with confidence scores
proba = pipeline.predict_proba(test_df['combined_text'][:5])
print("\nSample predictions with confidence scores:")
for i, (pred, prob) in enumerate(zip(predictions[:5], proba[:5])):
    confidence = max(prob) * 100
    print(f"ID: {test_df['id'].iloc[i]}, Prediction: {pred}, Confidence: {confidence:.2f}%")