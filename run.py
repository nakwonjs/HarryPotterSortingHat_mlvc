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

import time
from util import *


def main():
    device = check_device()
    epochs = 2
    learningRate = 1e-5

    #TRAIN_DIR = '/content/drive/MyDrive/KK/TRAIN'
    #TEST_DIR = '/content/drive/MyDrive/KK/TEST'
    TRAIN_DIR = "Data/TRAIN"
    TEST_DIR = "Data/TEST"

    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 32)

    model = model.to(device)
    BATCH_SIZE = 16
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_data = dsets.ImageFolder(TRAIN_DIR, transform=transform)
    test_data = dsets.ImageFolder(TEST_DIR, transform=transform)

    trainloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = AdamW(model.parameters(), lr=learningRate)

    t_accs, v_accs, t_loss, v_loss = [], [], [], []
    for epoch in range(epochs):
        start_time = time.time()
        train_loss = 0
        train_accuracy = 0
        model.train()

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)

            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            train_accuracy += torch.mean(equals.type(torch.FloatTensor))

            loss = criterion(output, labels)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()


        t_accs.append(train_accuracy / len(trainloader))
        t_loss.append(train_loss / len(trainloader))

        test_loss = 0
        test_accuracy = 0
        model.eval()

        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            log_ps = model(images)
            test_loss += criterion(log_ps, labels).item()

            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            test_accuracy += torch.mean(equals.type(torch.FloatTensor))

        elapsed_time = time.time() - start_time
        v_accs.append(test_accuracy / len(testloader))
        v_loss.append(test_loss / len(testloader))

        print("==> Epoch[{}/{}]".format(epoch + 1, epochs))
        print("Train loss: {:.3f} | Train Accuracy: {:.3f} | valid Loss: {:.3f} | Valid Accuracy: {:.3f} | time: {:.3f}" \
              .format(t_loss[-1], t_accs[-1], v_loss[-1], v_accs[-1], elapsed_time))

        model_out_path = '.model.pth'
        torch.save(model.state_dict(), model_out_path)

    plotResultGraph(t_accs, v_accs, t_loss, v_loss)

def plotResultGraph(t_accs, v_accs, t_loss, v_loss):
    plt.subplot(121)
    plt.title('Model Accuracy')
    plt.plot(t_accs, label="Training Accuracy")
    plt.plot(v_accs, label="Validation Accuracy")
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='best')
    plt.grid()

    plt.subplot(122)
    plt.title('Model Loss')
    plt.plot(t_loss, label="Training loss")
    plt.plot(v_loss, label="Validation loss")
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='best')
    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()
