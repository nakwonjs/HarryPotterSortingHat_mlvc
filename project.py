import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torchvision
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torchvision import models

from transformers import AdamW

import matplotlib.pyplot as plt

#Define dataset directory
TRAIN_DIR = #TRAIN_DIR
TEST_DIR = #TEST_DIR
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Transfer learning section
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features

model.fc = nn.Linear(num_ftrs, 27)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

#Make dataset
BATCH_SIZE = 16
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = dsets.ImageFolder(TRAIN_DIR, transform=transform)
test_data = dsets.ImageFolder(TEST_DIR, transform=transform)

trainloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

#Define fundamental variances, loss function and optimizer
t_accs, v_accs, t_loss, v_loss = [], [], [], []
epochs=25
criterion = nn.CrossEntropyLoss().cuda()
optimizer = AdamW(model.parameters(), lr=1e-5)

#Train Section
for epoch in range(epochs):
  train_loss = 0
  train_accuracy = 0
  model.train()

  for images, labels in trainloader:
    images = images.cuda()
    labels = labels.cuda()
    optimizer.zero_grad()
    output = model(images)
    ps = torch.exp(output) 
    top_p, top_class = ps.topk(1, dim = 1)
    equals = top_class == labels.view(*top_class.shape) 
    train_accuracy += torch.mean(equals.type(torch.FloatTensor))

    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()
    train_loss += loss.item()
  
  t_accs.append(train_accuracy/len(trainloader))
  t_loss.append(train_loss/len(trainloader))

  test_loss = 0
  test_accuracy = 0
  model.eval()

  for images, labels in testloader:
    images, labels = images.cuda(), labels.cuda()
    log_ps = model(images)
    test_loss += criterion(log_ps, labels).item()

    ps = torch.exp(log_ps)
    top_p, top_class = ps.topk(1, dim = 1)
    equals = top_class == labels.view(*top_class.shape)
    test_accuracy += torch.mean(equals.type(torch.FloatTensor))
  
  v_accs.append(test_accuracy/len(testloader))
  v_loss.append(test_loss/len(testloader))

  print("==> Epoch[{}/{}]".format(epoch+1, epochs))
  print("loss: {:.3f}, Accuracy: {:.3f}, val_loss: {:.3f}, val_accuracy: {:.3f}".format(t_loss[-1], t_accs[-1], v_loss[-1], v_accs[-1]))


  model_out_path = '.model.pth'
  torch.save(model.state_dict(), model_out_path)

#Plot Accuracy and Loss graph
plt.title('Model Accuracy')
plt.plot(t_accs, label= "Training Accuracy") 
plt.plot(v_accs, label= "Validation Accuracy") 
plt.ylabel('Accuracy')
plt.xlabel('Epoch') 
plt.legend(['Train','Validation'], loc='best') 
plt.grid()
plt.show()

plt.title('Model Loss')
plt.plot(t_loss, label = "Training loss") 
plt.plot(v_loss, label = "Validation loss") 
plt.ylabel('Loss')
plt.xlabel('Epoch') 
plt.legend(['Train','Validation'], loc='best') 
plt.grid()
plt.show()

#Predict Dormitory function
def SortHat(img):
  trans_img = transform(img)
  trans_img = Variable(trans_img)
  ti = trans_img.unsqueeze(0)
  ti = ti.cuda()
  output = model(ti).cuda()
  _, predicted = torch.max(output.cpu().data, 1)
  pred = predicted[0]

  if 0<=pred<=11:
    print('Slytherin!')
  elif 12<=pred<=18:
    print('Ravenclaw!')
  elif 19<=pred<=26:
    print('Griffyndor!')

#Prediction section
REAL_IMG = #Input image directory
import cv2

img = cv2.imread(REAL_IMG)
cv2.imshow(img)
SortHat(img)
