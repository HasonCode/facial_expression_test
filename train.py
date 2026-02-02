import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from skimage import io, transform
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
from typing import Union 
import FaceExpressionDataset as fed


train_set = fed.FaceExpressionDataset("dataset",transforms.Compose([transforms.ToTensor(),transforms.Resize((256,256))]))
train_loader = DataLoader(train_set, batch_size=64, shuffle = True, num_workers = 2)

test_set = fed.FaceExpressionDataset("dataset",transforms.Compose([transforms.ToTensor(),transforms.Resize((256,256))]))
test_loader = DataLoader(train_set, batch_size=64, shuffle = False, num_workers = 2)

classifications = ["anger","blink","frown","neutral","smile"]

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,32,3,1, padding = 1)
        self.gap = nn.AdaptiveMaxPool2d((1,1))
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32,64,3, padding = 1)
        self.conv3 = nn.Conv2d(64,128,3, padding = 1)
        self.conv4 = nn.Conv2d(128,256,3, padding = 1)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 5)

    def forward(self, x:torch.Tensor):
        out = self.pool(F.relu(self.conv1(x)))
        out = self.pool(F.relu(self.conv2(out)))
        out = self.pool(F.relu(self.conv3(out)))
        out = self.pool(F.relu(self.conv4(out)))
        out = self.gap(out)
        out = torch.flatten(out, 1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
    
if __name__ == "__main__":

    cnn = CNN()

    import torch.optim as optim
    device = torch.device("cpu")
    state_dict = torch.load("emotion_set.pth")
    cnn.load_state_dict(state_dict)
    cnn = cnn.to(device)
    num_classes = 5
    class_counts = torch.tensor([1814, 1853, 1240, 1264, 1214],dtype=torch.float)
    weights = class_counts.sum()/(num_classes*class_counts)

    loss_func = nn.CrossEntropyLoss(weight=weights.to(device))

    losses,epochs = [],[]

    optimizer = optim.SGD(cnn.parameters(), lr=0.01, momentum = 0.9)

    device = torch.device("cpu")

    for epoch in range(10):
        running_loss = 0.0
        correct = 0
        total = 0
        count = 0
        for i, data in enumerate(train_loader):

            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            outputs = cnn(inputs)
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

            loss = loss_func(outputs, labels)
            p0 = cnn.fc3.weight.detach().clone()
            loss.backward()
            # print(cnn.conv1.weight.grad.abs().mean().item())
            print(f"{i*64/len(train_set)*100:.4f}% Progress, Accuracy: {correct/total:.5f}")
            optimizer.step()
            running_loss += loss.item()
            count+=1
            # do ONE optimizer step
            p1 = cnn.fc3.weight.detach().clone()
            # print("fc3 mean |Δ|:", (p1 - p0).abs().mean().item())


        print(f"{epoch+1} loss: {running_loss/count:.7f}")
        print(f"{epoch+1} Accuracy: {correct/total:.3f}")
        losses.append(running_loss/count)
        epochs.append(epoch+1)

    print("all trained")

    torch.save(cnn.state_dict(),"emotion_set2.pth")

    plt.plot(epochs,losses)
    plt.savefig("lossvsepoch.png")
    plt.show()