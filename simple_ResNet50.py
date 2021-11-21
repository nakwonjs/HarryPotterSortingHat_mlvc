import torch
from torch import optim
import torchvision
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models.resnet as resnet
from transformers import AdamW
from torchvision import models

DATA_DIR = #DATE_DIR
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features

model.fc = nn.Linear(num_ftrs, 27)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

BATCH_SIZE = 16
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

data = dsets.ImageFolder(DATA_DIR, transform=transform)
loader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)

model.cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = AdamW(model.parameters(), lr=1e-5)

EPOCHS = 25

for epoch in range(EPOCHS):
    for i, (images, labels) in enumerate(loader):
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    print('Epoch [%d/%d], Loss: %.4f'%(epoch+1, EPOCHS, loss.data))


model.eval()

correct = 0
total = 0
for images, labels in loader:
    images = Variable(images).cuda()
    outputs = model(images).cuda()
    _, predicted = torch.max(outputs.cpu().data, 1)
    total += labels.cpu().size(0)
    correct += (predicted == labels).sum()

print('Test Accuracy of the model on the train images: %d %%' % (100 * correct / total))
print("Correct: ", correct)
