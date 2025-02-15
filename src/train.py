import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import re
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load the data
print("Loading data...")
train_df = pd.read_csv('../data/train_cleaned.csv')
test_df = pd.read_csv('../data/test.csv')

# Get stopwords
stop_words = set(stopwords.words('english'))
# Add custom stopwords relevant to mental health domain
custom_stops = {'like', 'feel', 'feeling', 'think', 'know', 'get', 'would', 'could'}
stop_words.update(custom_stops)

def preprocess_text(text):
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Replace contractions
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"'m", " am", text)
        text = re.sub(r"'s", " is", text)
        text = re.sub(r"'re", " are", text)
        text = re.sub(r"'ll", " will", text)
        text = re.sub(r"'ve", " have", text)
        text = re.sub(r"'d", " would", text)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^a-zA-Z\s.,!?]', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and short words
        tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
        
        # Join tokens back
        text = ' '.join(tokens)
        
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
        max_features=12000,  # Reduced from 15000 to prevent overfitting
        ngram_range=(1, 2),
        min_df=3,  # Increased to remove more rare terms
        max_df=0.9,  # Decreased to remove more common terms
        sublinear_tf=True  # Apply sublinear scaling
    )),
    ('classifier', RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    ))
])

# Perform stratified k-fold cross-validation
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
submission_file = 'submission_robust.csv'
submission.to_csv(submission_file, index=False)
print(f"\nSubmission saved to {submission_file}")

# Print class distribution in training data vs predictions
print("\nClass distribution comparison:")
print("\nTraining data distribution:")
print(train_df['target'].value_counts(normalize=True))
print("\nPrediction distribution:")
print(pd.Series(predictions).value_counts(normalize=True))

# Get feature importances
tfidf = pipeline.named_steps['tfidf']
rf = pipeline.named_steps['classifier']
feature_names = tfidf.get_feature_names_out()
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

print("\nTop 10 most important terms:")
for f in range(10):
    print("%d. %s (%f)" % (f + 1, feature_names[indices[f]], importances[indices[f]]))