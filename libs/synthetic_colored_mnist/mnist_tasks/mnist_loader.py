from torch.utils.data import DataLoader, random_split
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np

labels = np.linspace(0,9,10,dtype=int)

transform=transforms.Compose([
    transforms.Grayscale(num_output_channels=3), 
    # transforms.Resize(32), 
    transforms.ToTensor(), 
    transforms.Normalize([0.5], [0.5])
])
print(transform)
train_set = datasets.MNIST(root='MNIST', transform=transform,download=True, train=True)
test_set = datasets.MNIST(root='MNIST', transform=transform,download=True, train=False)

# Random split
train_set_size = int(len(train_set) * 0.9)
valid_set_size = len(train_set) - train_set_size
train_set, valid_set = random_split(train_set, [train_set_size, valid_set_size])

# After
print('='*30)
print('Train data set:', len(train_set))
print('Test data set:', len(test_set))
print('Valid data set:', len(valid_set))

trainloader = DataLoader(train_set, batch_size=1280, shuffle=False)
testloader = DataLoader(test_set, batch_size=1280, shuffle=False)
valloader = DataLoader(valid_set, batch_size=1280, shuffle=False)