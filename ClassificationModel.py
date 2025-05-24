import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# === 1. Load and preprocess data ===

# Load dataset
X = np.load('X_60_augmented.npy')  # shape: (1620, 60, 66)
y = np.load('y_60_augmented.npy')  # shape: (1620,)

# Load label map
with open('label_map.json', 'r') as f:
    label_map = json.load(f)

num_classes = len(label_map)

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

# Dataset and Dataloader
class PoseDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = PoseDataset(X_train_tensor, y_train_tensor)
val_dataset = PoseDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# === 2. Build the LSTM model ===

class LSTMPoseClassifier(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, num_classes):
        super(LSTMPoseClassifier, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden1, batch_first=True)
        self.bn1 = nn.BatchNorm1d(60)
        self.dropout1 = nn.Dropout(0.3)

        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.dropout2 = nn.Dropout(0.3)

        self.fc1 = nn.Linear(hidden2, 64)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.bn1(out)
        out = self.dropout1(out)

        out, _ = self.lstm2(out)
        out = self.dropout2(out)

        out = self.fc1(out[:, -1, :])  # Take the last time step
        out = self.relu(out)
        out = self.dropout3(out)
        out = self.fc2(out)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMPoseClassifier(66, 128, 64, num_classes).to(device)

# === 3. Training ===

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

train_accs, val_accs = [], []

for epoch in range(30):
    model.train()
    correct = total = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = torch.argmax(outputs, dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    train_acc = correct / total
    train_accs.append(train_acc)

    # Validation
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    val_acc = correct / total
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1}/30 - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

# === 4. Evaluate the model ===

model.eval()
all_preds = []
all_true = []

with torch.no_grad():
    for X_batch, y_batch in val_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_true.extend(y_batch.numpy())

f1 = f1_score(all_true, all_preds, average='macro')
print(f"\nF1 Score (macro): {f1:.4f}")

print("\nClassification Report:")
print(classification_report(all_true, all_preds, target_names=list(label_map.keys())))


# === 5. Plot training curves ===

plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# === 6. Save the trained model ===

torch.save(model.state_dict(), 'lstm_classification_model.pth')
print("Model saved as lstm_classification_model.pth")
