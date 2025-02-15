import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import warnings
from tqdm import tqdm  # Pour la barre de progression

warnings.filterwarnings('ignore')

class MentalHealthDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])

        return item

def train_model(model, train_loader, val_loader, device, num_epochs=3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
            # Afficher la perte actuelle dans la barre de progression
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Afficher la perte moyenne pour l'époque
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}: Average Training Loss = {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            val_progress_bar = tqdm(val_loader, desc="Validation", leave=False)
            for batch in val_progress_bar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                preds = torch.argmax(outputs.logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(labels.cpu().numpy())
        
        val_acc = accuracy_score(val_true, val_preds)
        print(f"Epoch {epoch + 1}: Validation Accuracy = {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pt')
            print(f"New best model saved with validation accuracy: {best_val_acc:.4f}")
    
    return best_val_acc

def main():
    # Load data
    print("Loading data...")
    train_df = pd.read_csv('../data/train_cleaned.csv')
    test_df = pd.read_csv('../data/test.csv')
    
    # Combine title and content
    train_df['text'] = train_df['title'].fillna('') + ' [SEP] ' + train_df['content'].fillna('')
    test_df['text'] = test_df['title'].fillna('') + ' [SEP] ' + test_df['content'].fillna('')
    
    # Load tokenizer and model
    model_name = "emilyalsentzer/Bio_ClinicalBERT"  # or another suitable BERT variant
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Label encoding
    label_dict = {label: i for i, label in enumerate(train_df['target'].unique())}
    train_df['label'] = train_df['target'].map(label_dict)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Cross-validation setup
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df['text'], train_df['label'])):
        print(f"\nTraining fold {fold + 1}")
        
        # Create datasets
        train_dataset = MentalHealthDataset(
            texts=train_df['text'].iloc[train_idx].values,
            labels=train_df['label'].iloc[train_idx].values,
            tokenizer=tokenizer
        )
        
        val_dataset = MentalHealthDataset(
            texts=train_df['text'].iloc[val_idx].values,
            labels=train_df['label'].iloc[val_idx].values,
            tokenizer=tokenizer
        )
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8)
        
        # Initialize model
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(label_dict)
        ).to(device)
        
        # Train and evaluate
        val_acc = train_model(model, train_loader, val_loader, device)
        cv_scores.append(val_acc)
        
        print(f"Fold {fold + 1} validation accuracy: {val_acc:.4f}")
    
    print(f"\nMean CV accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores)*2:.4f})")
    
    # Train final model on full dataset
    print("\nTraining final model on full dataset...")
    final_dataset = MentalHealthDataset(
        texts=train_df['text'].values,
        labels=train_df['label'].values,
        tokenizer=tokenizer
    )
    
    final_loader = DataLoader(final_dataset, batch_size=8, shuffle=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_dict)
    ).to(device)
    
    # Create test dataset
    test_dataset = MentalHealthDataset(
        texts=test_df['text'].values,
        tokenizer=tokenizer
    )
    test_loader = DataLoader(test_dataset, batch_size=8)
    
    # Make predictions
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Making predictions"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
    
    # Create submission
    reverse_label_dict = {v: k for k, v in label_dict.items()}
    predictions = [reverse_label_dict[pred] for pred in predictions]
    
    submission = pd.DataFrame({
        'ID': test_df['id'],
        'Target': predictions
    })
    
    submission.to_csv('submission_bert.csv', index=False)
    print("\nSubmission saved to submission_bert.csv")

if __name__ == "__main__":
    main()