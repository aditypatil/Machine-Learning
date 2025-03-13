import torch
import torch.nn as nn
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

# Wrapper classes for PyTorch models to make them scikit-learn compatible
class PytorchClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, pytorch_model_class, input_dim, num_classes=2, epochs=10, batch_size=32, lr=0.001):
        self.model_class = pytorch_model_class
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        
    def fit(self, X, y):
        # Initialize model
        self.model_ = self.model_class(input_dim=self.input_dim, num_classes=self.num_classes)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_.to(device)
        
        # Convert data to tensors
        X_tensor = torch.FloatTensor(X.to_numpy()).to(device)
        y_tensor = torch.LongTensor(y.to_numpy()).to(device)
        
        # Setup optimizer and loss
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        self.model_.train()
        for epoch in range(self.epochs):
            for i in range(0, len(X), self.batch_size):
                batch_X = X_tensor[i:i+self.batch_size]
                batch_y = y_tensor[i:i+self.batch_size]
                
                optimizer.zero_grad()
                outputs = self.model_(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        return self
    
    def predict_proba(self, X):
        self.model_.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_tensor = torch.FloatTensor(X.to_numpy()).to(device)
        
        with torch.no_grad():
            outputs = self.model_(X_tensor)
            probs = torch.softmax(outputs, dim=1)
        return probs.cpu().numpy()
    
    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

class PytorchRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, pytorch_model_class, input_dim, epochs=10, batch_size=32, lr=0.001):
        self.model_class = pytorch_model_class
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        
    def fit(self, X, y):
        # Initialize model
        self.model_ = self.model_class(input_dim=self.input_dim)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_.to(device)
        
        # Convert data to tensors
        X_tensor = torch.FloatTensor(X.to_numpy()).to(device)
        if hasattr(y,'to_numpy'):
            y_np = y.to_numpy()
        else:
            y_np = y
        y_tensor = torch.FloatTensor(y_np).reshape(-1, 1).to(device)
        
        # Setup optimizer and loss
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        
        # Training loop
        self.model_.train()
        for epoch in range(self.epochs):
            for i in range(0, len(X), self.batch_size):
                batch_X = X_tensor[i:i+self.batch_size]
                batch_y = y_tensor[i:i+self.batch_size]
                
                optimizer.zero_grad()
                outputs = self.model_(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        return self
    
    def predict(self, X):
        self.model_.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_tensor = torch.FloatTensor(X.to_numpy()).to(device)
        
        with torch.no_grad():
            outputs = self.model_(X_tensor)
        return outputs.cpu().numpy().flatten()

# PyTorch Neural Network Definitions
class SimpleNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64], num_classes=2, dropout=0.3):
        super(SimpleNNClassifier, self).__init__()
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.BatchNorm1d(dim),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
            
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class SimpleNNRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64], dropout=0.3):
        super(SimpleNNRegressor, self).__init__()
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.BatchNorm1d(dim),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
            
        # Output layer (single value for regression)
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class CNN1DRegressor(nn.Module):
    def __init__(self, input_dim, hidden_channels=32):
        super(CNN1DRegressor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden_channels, hidden_channels*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels*2),
            nn.MaxPool1d(2)
        )
        # Calculate the size after convolutions
        self.flat_size = (hidden_channels*2) * (input_dim // 4)
        self.fc = nn.Sequential(
            nn.Linear(self.flat_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.conv(x)
        x = x.view(-1, self.flat_size)
        return self.fc(x)