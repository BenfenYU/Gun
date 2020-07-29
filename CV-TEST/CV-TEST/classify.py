import torch, torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


TRAIN_SPLIT = 0.8
TEST_SPLIT = 0.2
BATCH_SIZE = 128
PATH = './2_class.pth'
device = torch.device('cuda:0')
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
classes = ('0','1')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(3, 6, 5)
        self.fc1 = nn.Linear(6 * 117 * 117, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.relu(x))
        x = self.conv2(x)
        x = self.pool(F.relu(x))
        x = x.view(BATCH_SIZE, 6 * 117 * 117)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train():

    trainset = datasets.ImageFolder('./data/train/', transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                            shuffle=True, num_workers=1)

    net = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader,0):
            inputs,labels = data
            if(len(data[1]) != BATCH_SIZE):
                continue
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 50 == 1: 
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / (50*BATCH_SIZE)))
                running_loss = 0.0

    print('Finished Training')
    torch.save(net.state_dict(), PATH)

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def test():
    net = Net()
    net.load_state_dict(torch.load(PATH))

    testset = datasets.ImageFolder('./data/test/',transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                            shuffle=False, num_workers=1)

    dataiter = iter(testloader)
    images, labels = dataiter.next()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if(len(images) != BATCH_SIZE):
                continue
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))

    images, labels = dataiter.next()

    # print images
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(BATCH_SIZE)))
    imshow(torchvision.utils.make_grid(images))
    


if __name__ == '__main__':
    test()