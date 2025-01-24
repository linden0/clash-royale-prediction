import torch
from torch import nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM expects (batch, seq, feature)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        batch_size = x.size(0)

        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)

        # LSTM output: (batch_size, seq_length, hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the last time-step
        out = out[:, -1, :]  # shape: (batch_size, hidden_size)
        
        # Fully-connected layer
        out = self.fc(out)   # shape: (batch_size, num_classes)
        return out

class CNN(nn.Module):
    def __init__(self, num_classes, in_channels=1):
        super(CNN, self).__init__()
        
        # Example: a small, 3-layer CNN
        # You can increase depth/channels for better performance
        # but be mindful of overfitting if the dataset is small.
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # halve time and mfcc dims

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # halve time and mfcc dims

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # Optionally another pooling
            # nn.MaxPool2d(kernel_size=2)
        )
        
        # After the last conv, we do adaptive average pooling so that
        # we get a fixed-size 2D feature map, regardless of input time dimension.
        # Then we flatten and do a fully connected layer to classify.
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier  = nn.Linear(64, num_classes)

    def forward(self, x):
        # x is originally (batch_size, time, n_mfcc)
        # We need to add a channel dimension => (batch_size, 1, time, n_mfcc)
        x = x.unsqueeze(1)
        
        # Pass through convolutional layers
        x = self.features(x)
        
        # Global average pooling -> (batch_size, 64, 1, 1)
        x = self.global_pool(x)
        
        # Flatten -> (batch_size, 64)
        x = x.view(x.size(0), -1)
        
        # Classify
        x = self.classifier(x)
        return x