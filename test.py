import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
data = torchvision.datasets.FashionMNIST('./', download=True)
n_epochs = 40
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.9
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)
transform = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.5,), (0.5,))
                             ])
trainset = torchvision.datasets.FashionMNIST('/files/', train=True, download=True, transform=transform)
testset = torchvision.datasets.FashionMNIST('/files/', train=False, download=True, transform=transform)
#Split trainset to validation and train
indices = list(range(len(trainset)))
np.random.shuffle(indices)
split = int(np.floor(0.2 * len(trainset)))
print(split)
val_idx, train_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)
train_data = torch.utils.data.DataLoader(trainset, sampler=train_sampler, batch_size=batch_size_train)
vali_data = torch.utils.data.DataLoader(trainset, sampler=val_sampler, batch_size=batch_size_train)

test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test)

class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
    self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
    self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
    self.fc2 = nn.Linear(in_features=120, out_features=60)
    self.out = nn.Linear(in_features=60, out_features=10)
  def forward(self, t):
    t = self.conv1(t)
    t = F.relu(t)
    t = F.max_pool2d(t, kernel_size=2, stride=2)
    t = self.conv2(t)
    t = F.relu(t)
    t = F.max_pool2d(t, kernel_size=2, stride=2)
    t = t.reshape(-1, 12*4*4)
    t = self.fc1(t)
    t = F.relu(t)
    t = self.fc2(t)
    t = F.relu(t)
    t = F.log_softmax(self.out(t), dim=1)
    return t
import matplotlib.pyplot as plt
dataiter = iter(test_loader)
print(dataiter)
images, labels = next(dataiter)

fig = plt.figure(figsize=(45, 15))
for i in range(64):
  ax = fig.add_subplot(12, 15, i + 1)
  ax.imshow(np.squeeze(images[i]), cmap="gray")


def train(train_data, vali_data, n_epochs, optimizer, loss_fn, ):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  train_losses, vali_losses = [], []
  for i in range(n_epochs):
    running_loss = 0
    validation_loss = 0
    for x, y in train_data:
      optimizer.zero_grad()
      yhat = model(x)
      loss = loss_fn(yhat, y)
      loss.backward()
      optimizer.step()
      running_loss += loss.item() * x.size(0)
    for x, y in vali_data:
      yhat = model(x)
      loss = loss_fn(yhat, y)
      validation_loss += loss.item() * x.size(0)
    running_loss /= len(train_data.sampler)
    validation_loss /= len(vali_data.sampler)
    train_losses.append(running_loss)
    vali_losses.append(validation_loss)
    print("Epoch: {}\tTraining loss: {:.6f}\t Validation loss: {:.6f}".format(i + 1, running_loss, validation_loss))
  return train_losses, vali_losses



model = Model()
loss_fn = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.009)
model.train()
train_losses, vali_losses = train(train_data, vali_data, n_epochs, optimizer, loss_fn)
plt.plot(train_losses, label="Train loss")
plt.plot(vali_losses, label="Vali loss")
plt.legend()
class_correct = [0] * 10
class_total = [0] * 10
test_loss = 0
for x, y in test_loader:
  yhat = model(x)
  loss = loss_fn(yhat, y)
  test_loss += loss.item() * x.size(0)
  _, pred = torch.max(yhat, 1)
  correct = np.squeeze(pred.eq(y.data.view_as(pred)))
  for i in range(len(y)):
    label = y.data[i]
    class_correct[label] += correct[i].item()
    class_total[label] += 1
test_loss /= len(test_loader.sampler)
print("Test loss: {:.6f}".format(test_loss))
for i in range(10):
  print("Test accuracy of class {}: {}% ({}/{})".format(i,100 * class_correct[i] /class_total[i],  class_correct[i],class_total[i]))
print("Test accuracy overall: {}".format(100 * np.sum(class_correct)/np.sum(class_total)))