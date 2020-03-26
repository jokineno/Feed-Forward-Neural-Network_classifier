import torch
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from plotting import plot_weights


#--- hyperparameters ---
N_EPOCHS = 50
BATCH_SIZE_TRAIN = 100
BATCH_SIZE_TEST = 100
LR = 0.001


CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
NUM_CLASSES = 10



# --- CIFAR initialization ---

# We transform torchvision.datasets.CIFAR10 outputs to tensors
# Plus, we add a random horizontal transformation to the training data
train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
test_transform = transforms.Compose([transforms.ToTensor()])
train_set = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=train_transform)
test_set = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=test_transform)

# Create Pytorch data loaders
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCH_SIZE_TEST, shuffle=False)


#--- model ---
class CNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(CNN, self).__init__()
        # WRITE CODE HERE
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(8 * 8 * 64, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        # WRITE CODE HERE
        
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out



#--- set up ---
if __name__=='__main__':

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = CNN().to(device)

    # WRITE CODE HERE
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_function = nn.CrossEntropyLoss()
    

    #--- training ---
    for epoch in range(N_EPOCHS):
        train_loss = 0
        train_correct = 0
        total = 0
        for batch_num, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            # WRITE CODE HERE
            optimizer.zero_grad()
            
            output = model(data)
            
            loss = loss_function(output, target)

            loss.backward()
            optimizer.step()
            
            train_loss += loss
            total += BATCH_SIZE_TRAIN
            
            _, predicted = torch.max(output.data,1)
            correct = (predicted == target).sum()
            train_correct += correct

            print('Epoch %d - Batch %d/%d: Loss: %.4f | Train Acc: %.3f%% (%d/%d)' % (epoch, batch_num, len(train_loader), train_loss / (batch_num + 1), 100. * train_correct / total, train_correct, total))


    #--- test ---
    test_loss = 0
    test_correct = 0
    total = 0

    with torch.no_grad():
        for batch_num, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            # WRITE CODE HERE
            output = model(data)
            test_loss = loss_function(output,target)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            test_correct += (predicted == target).sum() 

            print('Evaluating: Batch %d/%d: Loss: %.4f | Test Acc: %.3f%% (%d/%d)' % (batch_num, len(test_loader), test_loss / (batch_num + 1), 100. * test_correct / total, test_correct, total))
            

    # WRITE CODE HERE
    #visualize weights for the first conv layer
    filters = model.modules()
    model_layers = [i for i in model.children()]
    first_layer = model_layers[0]
    conv_layer = first_layer[0]
    plot_weights(conv_layer.cpu())




