#import thư viện và package
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
data = torchvision.datasets.FashionMNIST('./', download=True)


#khởi tạo hyper paremeter
n_epochs =100
batch_size_train = 128
batch_size_test = 1000
learning_rate = 0.003
momentum = 0.9
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)







#lấy dữ liệu và tạo tập train, validation và test
transform = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.5,), (0.5,))
                             ])
trainset = torchvision.datasets.FashionMNIST('/files/', train=True, download=True, transform=transform)
testset = torchvision.datasets.FashionMNIST('/files/', train=False, download=True, transform=transform)
#tách trainset to validation and train
indices = list(range(len(trainset)))
np.random.shuffle(indices)
split = int(np.floor(0.2 * len(trainset)))
print(split)
val_idx, train_idx = indices[:split], indices[split:]
train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)
train_data = torch.utils.data.DataLoader(trainset, sampler=train_sampler, batch_size=batch_size_train)
vali_data = torch.utils.data.DataLoader(trainset, sampler=val_sampler, batch_size=batch_size_train)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, shuffle=True)






#tạo Model CNN 
class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=32 ,kernel_size=3, padding="same") #convolution 1 

    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding="same") #convolution 2

    self.fc1 = nn.Linear(in_features=64*7*7, out_features=128) # tổng linear
    self.out = nn.Linear(in_features=128, out_features=10) # tôngr linear
    self.flatten = nn.Flatten()
    self.dropout1 = nn.Dropout(0.3) # random các thuộc tính và gán = 0
    self.dropout2 = nn.Dropout(0.4) # random các thuộc tính và gán = 0
    self.dropout3 = nn.Dropout(0.25) # random các thuộc tính và gán = 0
  def forward(self, t):
    
    t = self.conv1(t)
    t = F.relu(t)
    t = F.max_pool2d(t, kernel_size=2, stride=2)
    #t = self.dropout(t)
    t = self.conv2(t)
    #t = self.batch2(t)
    t = F.relu(t)
    t = F.max_pool2d(t, kernel_size=2, stride=2)
 
    t = self.flatten(t)

    t = self.fc1(t)
    t = F.relu(t)

    t = F.log_softmax(self.out(t), dim=1)
    return t




# visualize some images
import matplotlib.pyplot as plt
dataiter = iter(test_loader)
images, labels = next(dataiter)
desc = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']
print(desc[labels[0]])
plt.imshow(np.squeeze(images[0]), cmap="gray")
plt.show()


#tạo hàm train và hàm lấy learning rate
def get_lr(optimizer):
  for g in optimizer.param_groups:
    return g['lr']
def train(train_data, vali_data, n_epochs, optimizer, loss_fn, device):

  train_losses, vali_losses = [], []
  for i in range(n_epochs):
    running_loss = 0
    validation_loss = 0
    total_correct = 0
    for x, y in train_data:
      optimizer.zero_grad()
      x, y = x.to(device), y.to(device)

      yhat = model(x)
      loss = loss_fn(yhat, y)
      loss.backward()
      optimizer.step()
      running_loss += loss.item() * x.size(0)
    else:
      model.eval() 
      for x, y in vali_data:
        x, y = x.to(device), y.to(device)
        
        yhat = model(x)
        loss = loss_fn(yhat, y)
        validation_loss += loss.item() * x.size(0)
        total_correct += yhat.argmax(dim=1).eq(y).sum().item()
      running_loss /= len(train_data.sampler)
    validation_loss /= len(vali_data.sampler)
    total_correct /= len(vali_data.sampler)
    train_losses.append(running_loss)
    vali_losses.append(validation_loss)
    print("Epoch: {}\tTraining loss: {:.6f}\t Validation loss: {:.6f}\t Validation acc: {:.6f}\t Lr: {}".format(i + 1, running_loss, validation_loss, total_correct, get_lr(optimizer)))
  return train_losses, vali_losses


#chọn hàm loss, hàm tối ưu hàm loss, và train 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model().to(device)
loss_fn = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
model.train()

train_losses, vali_losses = train(train_data, vali_data, n_epochs, optimizer, loss_fn, device)



#visualize train_loss and val_loss by epochs
plt.plot(train_losses, label="Train loss")
plt.plot(vali_losses, label="Vali loss")
plt.legend()
plt.show()


#caculation accuracy score for test data
class_correct = [0] * 10
class_total = [0] * 10
test_loss = 0
for x, y in test_loader:
  x, y = x.to(device), y.to(device)
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



#predict some images

dataiter = iter(test_loader)
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)
index = 50
img, label = images[index], labels[index]
preds = model(images)
desc = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']
fig, (ax1, ax2) =  plt.subplots(figsize=(13, 6), nrows=1, ncols=2)
ax1.axis('off')
ax1.imshow(images[index].detach().numpy().squeeze())
ax1.set_title(desc[label.item()])
preds = preds.detach().numpy()[index]
preds = np.exp(preds)
ax2.bar(range(10), preds)
ax2.set_xticks(range(10))
ax2.set_xticklabels(desc, size='small')
ax2.set_title('Predicted Probabilities')
plt.tight_layout()
