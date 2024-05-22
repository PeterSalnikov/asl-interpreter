import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
import timm


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import tqdm

import pickle

import sys

print('System version:',sys.version)
print('PyTorch version:',torch.__version__)
print('Torchvision version:', torchvision.__version__)
print('Numpy version:',np.__version__)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

device = "cpu"

print(f"Using {device} device")

class HandLandmarkDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).long()
        # self.target = target
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y
    
    def __len__(self):
        return len(self.data)
    
    @property
    def classes(self):
        return self.data.classes
    
# Load the dataset

num_features = 42

data_dict = pickle.load(open('./data.pickle', 'rb'))

for i in range(len(data_dict['data'])-1,0,-1):
    if len(data_dict['data'][i]) != num_features:
        # print(len(data_dict['data'][i]))
    # print(len(data_dict['labels'][i]))
        data_dict['data'].pop(i)   
        data_dict['labels'].pop(i)   
        # del data_dict['data'][i]
        # del data_dict['labels'][i]

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

data_dir = './ASL_Alphabet_Dataset/asl_alphabet_train'
target_to_class = {v: k for v, k in ImageFolder(data_dir).class_to_idx.items()}
target = np.array([target_to_class[char] for char in labels])


# numpy_data = np.random.randn(100, 42)
# numpy_target = np.random.randint(0, 25, size=(42))

dataset = HandLandmarkDataset(data, target)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size,val_size])

# for batch_idx, (data, target) in enumerate(loader):
#     print('Batch idx {}, data shape {}, target length {}'.format(
#         batch_idx, data.shape, len(target)))
    

transform = transforms.Compose([
    transforms.Resize((42)),
    transforms.ToTensor()
])

# image, label = dataset[100]
# print(image.shape)

# Create Dataloader

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,
    # pin_memory=torch.backends.mps.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    # pin_memory=torch.backends.mps.is_available()
)

for landmarks, labels in train_loader:
    # print(landmarks.shape, len(labels))
    break

class ASLClassifier(nn.Module):
    def __init__(self, num_classes=29):
        # # initialize this object with everything from the parent class
        # super(ASLClassifier, self).__init__()
        # self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        # # remove the last layer so we can output our own
        # self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        
        # enet_out_size = 1280
        # self.classifier = nn.Linear(enet_out_size, num_classes)
        super(ASLClassifier, self).__init__()
        self.fc1 = nn.Linear(42, 128)  # Input size 42, output size 128
        self.fcx = nn.Linear(128,128)
        # self.fcy = nn.Linear(512,128)
        self.fc2 = nn.Linear(128, 64)  # Hidden layer
        self.fc3 = nn.Linear(64, 29)   # Assuming 29 classes for characters A-Z


    def forward(self,x):
        # # Connect these parts and return the output
        # x = self.features(x)
        # output = self.classifier(x)
        # return output
        x = F.relu(self.fc1(x))
        x = F.relu(self.fcx(x))
        # x = F.relu(self.fcy(x))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        return output
    
model = ASLClassifier(num_classes = 29).to(device)
# print(model)

# example_out = model(landmarks)


# for batch_idx, (data, target) in enumerate(loader):
#     # For debugging, print the shapes of data and target
#     print('Batch idx {}, data shape {}, target shape {}'.format(
#         batch_idx, data.shape, len(target)))  # Target is a list of strings
    
#     # Convert target to tensor for training (if needed)
#     # target_indices = torch.tensor([ord(t) - 65 for t in target])  # Convert A-Z to 0-25
#     target_indices = target_to_class[]
    
#     outputs = model(data)
#     print('Output shape:', outputs.shape)

if __name__ == "__main__":

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    # print(criterion(example_out, labels))

    num_epochs = 20
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for landmarks, labels in tqdm.tqdm(train_loader, desc='Training Loop'):
            landmarks, labels = landmarks.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(landmarks)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for landmarks, labels in val_loader:
                landmarks, labels = landmarks.to(device), labels.to(device)
                outputs = model(landmarks)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)
            val_loss = running_loss / len(val_loader.dataset)
            val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss}, Val loss: {val_loss}")

    torch.save(model.state_dict(), "nn_model.pt")