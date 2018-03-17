import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 3, 5, stride=1, padding=2, dilation=1, groups=1, bias=True)
        self.conv2 = nn.Conv1d(3, 9, 5, stride=1, padding=2, dilation=1, groups=1, bias=True)
        self.pool = nn.MaxPool1d(5)
        self.fc1 = nn.Linear(9 * 40, 100)
        self.fc2 = nn.Linear(100, 20)
        self.fc3 = nn.Linear(20, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 9 * 40)         #after max-pool(5) 2 times number of data points = 9 layers of 40 points each
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return x


net = Net()
net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)

#generating training set waves in numpy, odd lines are random-sine waves even lines are random sets

wave = np.random.rand(1000,1000)
twave = torch.rand(1000, 1, 1000)

for xi in range(1000):
    if xi%2:
        freq = random.uniform(0.001, 1000)
        phase = random.randint(0, 10) * freq
        wave[xi]=np.sin(np.arange(1000)*freq*2*np.pi + phase)
        twave[xi,0]=torch.from_numpy(wave[xi])

#to plot any data from dataset
# plt.plot(np.arange(1000),wave[327])
#plt.show()

#training the neural-net

for epoch in range(100):  # loop over the whole dataset multiple times

    for b in range(20):   # batch size 50, no. of batches per iteration = 20
        running_loss = 0.0

        # label is zero for even data i.e non-sine wave
        # non-sine = +ve 1st activation, sine = +ve 2nd activation
        inputs = twave.narrow(0,50*b,50)    # alternate to slicing
        label=np.arange(50)%2
        np.transpose(label)
        labels =torch.from_numpy(label)

        # wrap them in Variable
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print('iterated '+str(epoch+1)+'-times')

print('Finished Training')
print(outputs)
#testing with the training set

print('testing...')
outputs = net(Variable(twave.cuda()))
output= outputs.data.cpu().numpy()      #convert data wrapped in variable to simple numpy array to use logic operators
correct = 0
for i in range(1000):
    a,b=output[i,0],output[i,1]
    if i%2 and b>0:
        correct+=1
    elif not(i%2) and a>0:
        correct+=1
print('accuracy on training set is '+str(correct/10)+'%')