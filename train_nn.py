import os
import torch
from torch import nn
# from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

import numpy as np

# ptrblck's solution for number arrays as model input
# https://discuss.pytorch.org/t/input-numpy-ndarray-instead-of-images-in-a-cnn/18797
class MyDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).long()
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y
    
    def __len__(self):
        return len(self.data)

# replace this with pickle dataset
numpy_data = np.random.randn(100, 42)
numpy_target = np.random.randint(0, 26, size=(100))

dataset = MyDataset(numpy_data, numpy_target)
loader = DataLoader(
    dataset,
    batch_size=10,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

for batch_idx, (data, target) in enumerate(loader):
    print('Batch idx {}, data shape {}, target shape {}'.format(
        batch_idx, data.shape, target.shape))
    
# https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

# train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
# test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)


# Get device for training

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Define the class

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(42, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 26),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
model = NeuralNetwork().to(device)
print(model)

# X = torch.rand(1, 28, 28, device=device)
X = torch.rand(1, 42, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

# Model layers

# take a sample minibatch of 3 images
# input_image = torch.rand(3,28,28)
input_image = torch.rand(3,42)
print(input_image.size())

# Flatten (not sure this is necessary for my case?)

# flatten = nn.Flatten()
# flat_image = flatten(input_image)
# print(flat_image.size())
flat_image = input_image

# Linear
# layer1 = nn.Linear(in_features=28*28, out_features=20)
layer1 = nn.Linear(in_features=42, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())

# ReLU

print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")

# Sequential

seq_modules = nn.Sequential(
    # flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
# input_image = torch.rand(3,28,28)
input_image = torch.rand(3,42)
logits = seq_modules(input_image)

# Softmax

softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)

# Model Parameters

print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")