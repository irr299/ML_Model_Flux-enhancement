"""
Neural Network for Flux Enhancement Classification
=================================================
This script trains neural network models to classify flux enhancements
using the labeled data from flux_enhancement_detector.py

Based on the NN_examples.ipynb architecture patterns.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

# PyTorch imports
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
import joblib


class FlexibleNN(nn.Module):
    """Flexible Feed-Forward Neural Network (from NN_examples)."""
    
    def __init__(self, input_size, hidden_sizes, output_size, activation_fn=nn.ReLU, dropout_rate=0.2):
        """
        Parameters:
        -----------
        input_size : int
            Number of input features
        hidden_sizes : list of int
            Sizes of hidden layers
        output_size : int
            Number of output units
        activation_fn : PyTorch activation class
            Activation function to use
        dropout_rate : float
            Dropout rate for regularization
        """
        super(FlexibleNN, self).__init__()
        
        layers = []
        
        # Build the network
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(activation_fn())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)


class LSTMClassifier(nn.Module):
    """LSTM for time series classification."""
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.2):
        super(LSTMClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch, sequence_length, features)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Use the last time step's output
        last_output = lstm_out[:, -1, :]
        
        # Apply dropout and final layer
        out = self.dropout(last_output)
        out = self.fc(out)
        
        return out


def load_and_prepare_data(file_path='flux_enhancement_labeled_fixed.csv'):
    """Load and prepare the labeled dataset."""
    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    
    # Define feature columns
    feature_cols = [
        'flux_filtered', 'lstar_filtered', 'MLT', 'alpha_local',
        'omni_SYM_H', 'omni_B', 'omni_V', 'omni_n',
        'hour', 'day_of_year',
        'flux_rolling_mean_24h', 'flux_rolling_std_24h', 'flux_rolling_max_24h',
        'symh_rolling_mean_24h', 'symh_rolling_min_24h',
        'flux_lag_6h', 'flux_lag_12h', 'flux_lag_24h',
        'symh_lag_6h', 'symh_lag_12h',
        'flux_change_6h', 'flux_change_12h',
        'hours_since_storm'
    ]
    
    # Remove features that might have many NaNs
    available_features = [col for col in feature_cols if col in df.columns]
    
    # Handle missing values
    df = df.dropna(subset=['enhancement_label'] + available_features[:5])  # Drop if core features are NaN
    
    # Fill remaining NaNs with median
    for col in available_features:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    # Extract features and labels
    X = df[available_features].values
    y = df['enhancement_label'].values
    
    print(f"\nDataset shape: {X.shape}")
    print(f"Features used: {len(available_features)}")
    print(f"Class distribution: {np.bincount(y)}")
    print(f"Class balance: {y.mean():.3f} positive ratio")
    
    return X, y, available_features, df.index


def create_time_series_data(X, y, window_size=24):
    """Create time series windows for LSTM."""
    n_samples = len(X) - window_size + 1
    X_windows = []
    y_windows = []
    
    for i in range(n_samples):
        X_windows.append(X[i:i+window_size])
        y_windows.append(y[i+window_size-1])  # Label for the last time step
    
    return np.array(X_windows), np.array(y_windows)


def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=50, device='cpu'):
    """Train the neural network model."""
    model = model.to(device)
    history = {'train_loss': [], 'valid_loss': [], 'train_acc': [], 'valid_acc': []}
    
    # Calculate class weights for imbalanced data
    y_train = []
    for _, labels in train_loader:
        y_train.extend(labels.numpy())
    
    # Handle edge case where only one class is present
    unique_classes = np.unique(y_train)
    if len(unique_classes) == 1:
        print("WARNING: Only one class in training data!")
        # Create dummy weights for both classes
        class_weights = torch.tensor([1.0, 1.0], dtype=torch.float32).to(device)
    else:
        # Normal case - both classes present
        class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    # Update criterion with class weights
    if isinstance(criterion, nn.CrossEntropyLoss):
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Handle different output shapes
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(0)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validation phase
        model.eval()
        valid_loss = 0.0
        valid_correct = 0
        valid_total = 0
        
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                if outputs.dim() == 1:
                    outputs = outputs.unsqueeze(0)
                    
                loss = criterion(outputs, labels)
                
                valid_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                valid_total += labels.size(0)
                valid_correct += (predicted == labels).sum().item()
        
        # Calculate epoch metrics
        avg_train_loss = train_loss / train_total
        avg_valid_loss = valid_loss / valid_total
        train_acc = train_correct / train_total
        valid_acc = valid_correct / valid_total
        
        history['train_loss'].append(avg_train_loss)
        history['valid_loss'].append(avg_valid_loss)
        history['train_acc'].append(train_acc)
        history['valid_acc'].append(valid_acc)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Valid Loss: {avg_valid_loss:.4f}, Valid Acc: {valid_acc:.4f}')
    
    return history


def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate the model on test data."""
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(0)
                
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
    
    # Calculate metrics
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, 
                              target_names=['No Enhancement', 'Enhancement']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_predictions)
    print(cm)
    
    # ROC AUC Score
    auc_score = roc_auc_score(all_labels, all_probs)
    print(f"\nROC AUC Score: {auc_score:.4f}")
    
    return all_predictions, all_labels, all_probs


def plot_training_history(history):
    """Plot training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['valid_loss'], label='Valid Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['valid_acc'], label='Valid Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()


def main():
    """Main execution function."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    X, y, feature_names, timestamps = load_and_prepare_data()
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp  # 0.176 to get 15% valid
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    valid_dataset = TensorDataset(
        torch.tensor(X_valid_scaled, dtype=torch.float32),
        torch.tensor(y_valid, dtype=torch.long)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test_scaled, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )
    
    # Create data loaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model (Feed-forward NN)
    model = FlexibleNN(
        input_size=X_train_scaled.shape[1],
        hidden_sizes=[128, 64, 32],
        output_size=2,  # Binary classification
        activation_fn=nn.ReLU,
        dropout_rate=0.3
    )
    
    print("\nModel Architecture:")
    print(model)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    # Train model
    print("\nTraining model...")
    history = train_model(model, train_loader, valid_loader, criterion, optimizer, 
                         num_epochs=50, device=device)
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    predictions, labels, probs = evaluate_model(model, test_loader, device)
    
    # Save model and scaler
    torch.save(model.state_dict(), 'flux_enhancement_model.pth')
    joblib.dump(scaler, 'flux_enhancement_scaler.pkl')
    print("\nModel and scaler saved!")
    
    # Feature importance (approximate using gradient-based method)
    print("\nCalculating feature importance...")
    model.eval()
    feature_importance = np.zeros(len(feature_names))
    
    for inputs, _ in train_loader:
        inputs = inputs.to(device).requires_grad_(True)
        outputs = model(inputs)
        outputs.sum().backward()
        
        # Average absolute gradients
        feature_importance += np.abs(inputs.grad.cpu().numpy()).mean(axis=0)
    
    feature_importance /= len(train_loader)
    
    # Print top features
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10))
    
    return model, history, scaler


if __name__ == "__main__":
    model, history, scaler = main() 