import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
import re
from sklearn.pipeline import Pipeline
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Ensure NLTK data is downloaded
for package in ['punkt', 'stopwords']:
    try:
        nltk.data.find(f'tokenizers/{package}')
    except LookupError:
        nltk.download(package)

def create_text_features(text):
    if not isinstance(text, str):
        return 0, 0, 0, 0
    
    # Count sentence-ending punctuation
    exclamations = text.count('!')
    questions = text.count('?')
    
    # Count words
    word_count = len(text.split())
    
    # Count uppercase words (potential emphasis)
    uppercase_words = sum(1 for word in text.split() if word.isupper())
    
    return exclamations, questions, word_count, uppercase_words

def preprocess_text(text):
    if not isinstance(text, str):
        return ''
    
    # Convert to lowercase but save original case info
    original_words = text.split()
    uppercase_words = [word for word in original_words if word.isupper()]
    
    text = text.lower()
    
    # Replace common mental health abbreviations
    abbrev_dict = {
        'adhd': 'attention deficit hyperactivity disorder',
        'ptsd': 'post traumatic stress disorder',
        'ocd': 'obsessive compulsive disorder',
        'bpd': 'borderline personality disorder',
        'gad': 'generalized anxiety disorder'
    }
    for abbr, full in abbrev_dict.items():
        text = text.replace(abbr, full)
    
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    
    # Handle contractions
    contractions = {
        "n't": " not", "'m": " am", "'s": " is", "'re": " are",
        "'ll": " will", "'ve": " have", "'d": " would"
    }
    for contraction, replacement in contractions.items():
        text = text.replace(contraction, replacement)
    
    # Remove special characters but keep important punctuation
    text = re.sub(r'[^a-zA-Z\s.,!?]', ' ', text)
    
    # Add back uppercase words as they might indicate emphasis
    text += ' ' + ' '.join(uppercase_words)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

print("Loading and preprocessing data...")
train_df = pd.read_csv('../data/train_cleaned.csv')
test_df = pd.read_csv('../data/test.csv')

# Create text features
print("Creating features...")
train_df['combined_text'] = train_df['title'].fillna('') + ' . ' + train_df['content'].fillna('')
test_df['combined_text'] = test_df['title'].fillna('') + ' . ' + test_df['content'].fillna('')

# Extract text features
for df in [train_df, test_df]:
    features = df['combined_text'].apply(create_text_features)
    df['exclamations'], df['questions'], df['word_count'], df['uppercase_words'] = zip(*features)

# Preprocess text
train_df['processed_text'] = train_df['combined_text'].apply(preprocess_text)
test_df['processed_text'] = test_df['combined_text'].apply(preprocess_text)

# Create multiple classifiers
tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

# Convert text to TF-IDF features
X_train_tfidf = tfidf.fit_transform(train_df['processed_text'])
X_test_tfidf = tfidf.transform(test_df['processed_text'])

# Add numerical features
numerical_features = ['exclamations', 'questions', 'word_count', 'uppercase_words']
X_train_numerical = train_df[numerical_features].values
X_test_numerical = test_df[numerical_features].values

# Combine TF-IDF and numerical features
X_train = np.hstack((X_train_tfidf.toarray(), X_train_numerical))
X_test = np.hstack((X_test_tfidf.toarray(), X_test_numerical))

# Create ensemble
clf1 = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced')
clf2 = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42
)

ensemble = VotingClassifier(
    estimators=[('lr', clf1), ('rf', clf2)],
    voting='soft'
)

# # Perform cross-validation
# print("\nPerforming cross-validation...")
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# cv_scores = cross_val_score(ensemble, X_train, train_df['target'], cv=cv, scoring='accuracy')

# print(f"\nCross-validation scores: {cv_scores}")
# print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Train final model
print("\nTraining final model...")
ensemble.fit(X_train, train_df['target'])

# Make predictions
print("Making predictions...")
predictions = ensemble.predict(X_test)

# Create submission file
submission = pd.DataFrame({
    'ID': test_df['id'],
    'Target': predictions
})
                      
submission_file = 'submission_ensemble.csv'
submission.to_csv(submission_file, index=False)
print(f"\nSubmission saved to {submission_file}")

# Print class distributions
print("\nClass distribution comparison:")
print("\nTraining data distribution:")
print(train_df['target'].value_counts(normalize=True))
print("\nPrediction distribution:")
print(pd.Series(predictions).value_counts(normalize=True))

# Get prediction probabilities
probas = ensemble.predict_proba(X_test)
confidence_scores = np.max(probas, axis=1)

print("\nConfidence statistics:")
print(f"Mean confidence: {confidence_scores.mean():.4f}")
print(f"Min confidence: {confidence_scores.min():.4f}")
print(f"Max confidence: {confidence_scores.max():.4f}")