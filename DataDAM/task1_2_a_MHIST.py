import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import os
from thop import profile
from networks import ConvNet
import time

# Hyperparameters
batch_size = 256
learning_rate = 0.01
num_epochs = 50
root_dir = './mhist_dataset/'
csv_file = './mhist_dataset/annotations.csv'
Start = time.time()
# Custom Dataset for MHIST
class MHISTDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, partition='train'):
        self.data_info = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.data_info = self.data_info[self.data_info['Partition'] == partition].reset_index(drop=True)

        # Define the label mapping
        self.label_mapping = {'HP': 0, 'SSA': 1}

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_info.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        
        # Map the string label to an integer using the label mapping
        label_str = self.data_info.iloc[idx, 1]
        label = self.label_mapping[label_str]

        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(label, dtype=torch.long)

        return image, label

# Model, Loss, Optimizer, and Scheduler
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ConvNet(channel=3, num_classes=2, net_width=32, net_depth=7, net_act='relu', net_norm='instancenorm', net_pooling='avgpooling', im_size=(224, 224)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

# Data Loaders
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = MHISTDataset(csv_file=csv_file, root_dir=root_dir, transform=transform, partition='train')
test_dataset = MHISTDataset(csv_file=csv_file, root_dir=root_dir, transform=transform, partition='test')
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
dummy_input = torch.randn(1, 3, 224, 224).to(device)
flops, params = profile(model, inputs=(dummy_input,))
print(f"FLOPs: {flops}, Parameters: {params}")
