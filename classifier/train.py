import torch
from torch import nn, optim
from torchvision import models
from dataset import get_loaders
from tqdm import tqdm
import torch.nn.functional as F
from backend.models import Network

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, test_loader = get_loaders(dementia_path="./demented", normal_path="./normal")

model = Network(in_channels=1, out_channels=2).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc="Training")
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        # Apply softmax to outputs
        outputs_softmax = F.softmax(outputs, dim=1)

        # Convert labels to one-hot encoding
        labels_one_hot = F.one_hot(labels, num_classes=2).float()

        loss = criterion(outputs, labels_one_hot)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs_softmax, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        progress_bar.set_postfix({'Loss': running_loss / (progress_bar.n + 1), 'Accuracy': 100 * correct / total})

    return running_loss / len(train_loader), 100 * correct / total

def evaluate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Evaluating")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            # Apply softmax to outputs
            outputs_softmax = F.softmax(outputs, dim=1)

            # Convert labels to one-hot encoding
            labels_one_hot = F.one_hot(labels, num_classes=2).float()

            loss = criterion(outputs, labels_one_hot)

            running_loss += loss.item()
            _, predicted = torch.max(outputs_softmax, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            progress_bar.set_postfix({'Loss': running_loss / (progress_bar.n + 1), 'Accuracy': 100 * correct / total})

    return running_loss / len(test_loader), 100 * correct / total


num_epochs = 100
best_acc = 0.0

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = evaluate(model, test_loader, criterion, device)

    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    scheduler.step(val_loss)

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
        print("Saved best model")

print(f"Best validation accuracy: {best_acc:.2f}%")