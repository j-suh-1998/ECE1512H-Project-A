import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from thop import profile
from networks import ConvNet
import time

# Hyperparameters
batch_size = 256
learning_rate = 0.01
num_epochs = 50

Start = time.time()
# Model, Loss, Optimizer, and Scheduler
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ConvNet(channel=1, num_classes=10, net_width=128, net_depth=3, net_act='relu', net_norm='instancenorm', net_pooling='avgpooling', im_size=(28, 28)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

# Data Loaders
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.MNIST(root='./mnist_dataset', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./mnist_dataset', train=False, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Training Loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        _, outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    scheduler.step()  # Update learning rate
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader.dataset):.4f}")

# Evaluation
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        _, outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
Finish = (time.time() - Start)/10
accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
print("TOTAL TIME WAS: ", Finish)

# FLOP Calculation
dummy_input = torch.randn(1, 1, 32, 32).to(device)
flops, params = profile(model, inputs=(dummy_input,))
print(f"FLOPs: {flops}, Parameters: {params}")
