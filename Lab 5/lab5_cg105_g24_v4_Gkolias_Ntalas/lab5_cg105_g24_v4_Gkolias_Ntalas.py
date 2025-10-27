import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, random_split
import time 

# For reproducibility
torch.manual_seed(42)

learning_rates = [0.001, 0.01, 0.1]
batch_sizes = [1, 64, 256]
hidden_layer_counts = [0, 1, 2]
hidden_layer_widths = [64, 128, 256]
loss_functions = {
    "CrossEntropy": nn.CrossEntropyLoss(),
    "MSE": nn.MSELoss(),
    "MAE": nn.L1Loss()
}

# Load FashionMNIST
transform = transforms.Compose([transforms.ToTensor()])
dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Split into training and validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
trainset, valset = random_split(dataset, [train_size, val_size])

def build_mlp(input_size, output_size, hidden_layers, width):
    layers = []
    last_size = input_size
    for _ in range(hidden_layers):
        layers.append(nn.Linear(last_size, width))
        layers.append(nn.ReLU())
        last_size = width
    layers.append(nn.Linear(last_size, output_size))
    return nn.Sequential(*layers)

def train_and_evaluate(learning_rate, batch_size, hidden_layers, width, loss_name, num_epochs=5):
    print(f"Training model with LR={learning_rate}, BS={batch_size}, HL={hidden_layers}, Width={width}, Loss={loss_name}")
    
    model = build_mlp(28*28, 10, hidden_layers, width)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = loss_functions[loss_name]

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valset, batch_size=batch_size)
    
    train_accuracies = []
    val_accuracies = []
    step_losses = []
    
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        correct, total = 0, 0
        for inputs, labels in train_loader:
            inputs = inputs.view(-1, 28*28)
            outputs = model(inputs)
            
            if loss_name == "CrossEntropy":
                loss = loss_fn(outputs, labels)
            else:
                labels_onehot = F.one_hot(labels, num_classes=10).float()
                loss = loss_fn(outputs, labels_onehot)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step_losses.append(loss.item())

            # Accuracy calculation
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_accuracy = correct / total
        train_accuracies.append(train_accuracy)

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.view(-1, 28*28)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_accuracy = correct / total
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")
        training_time = time.time() - start_time


    
    # After the training loop we evaluate on the test set
    test_loader = DataLoader(testset, batch_size=batch_size)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.view(-1, 28*28)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    test_accuracy = correct / total

    print(f"Final Train Acc: {train_accuracies[-1]:.4f}, Val Acc: {val_accuracies[-1]:.4f}, Test Acc: {test_accuracy:.4f}")
    print(f"Training time: {training_time:.2f} seconds")
    return step_losses, train_accuracies, val_accuracies, test_accuracy

# Example single run (feel free to loop over configs as needed)
losses, train_acc, val_acc, test_acc = train_and_evaluate(
    learning_rate=0.1,
    batch_size=64,
    hidden_layers=2,
    width=256,
    loss_name="CrossEntropy",
    num_epochs=25
)

# Plotting
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title("Loss per Training Step")
plt.xlabel("Training Step")
plt.ylabel("Loss")

plt.subplot(1, 2, 2)
plt.plot(train_acc, label="Train Accuracy")
plt.plot(val_acc, label="Validation Accuracy")
plt.title("Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.show()
