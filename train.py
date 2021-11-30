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
import argparse

from util import *
from models import se_renet
from models import dpns
from models import Densenet
from models import inceptionv4


def main():
    device = check_device()
    BATCH_SIZE = 16
    epochs = 30
    learningRate = 3e-4
    modelIdx = 4
    optimIdx = 0
    weight_decay = 1e-4

    # TRAIN_DIR = '/content/drive/MyDrive/KK/TRAIN'
    # TEST_DIR = '/content/drive/MyDrive/KK/TEST'
    #TRAIN_DIR = "Data/data2/TRAIN"
    #TEST_DIR = "Data/data2/TEST"
    TRAIN_DIR = "Data/TRAIN"
    TEST_DIR = "Data/TEST"
    model_out_path = 'Result/'

    if modelIdx == 0:
        model = models.resnet50(pretrained=True)
        model_out_path += "resnet50/resnet50"
    elif modelIdx == 1:
        model_out_path += "densenet121/densenet121"
        model = models.densenet121(pretrained=True)
    elif modelIdx == 2:
        model_out_path += "dpn92/dpn92"
        model = dpns.dpn92(pretrained=True)
    elif modelIdx == 3:
        model_out_path += "resnet152/resnet152"
        model = models.resnet152(pretrained=True)
    elif modelIdx == 4:
        model_out_path += "dpn131/dpn131"
        model = dpns.dpn131(pretrained=True)
    elif modelIdx == 5:
        model_out_path += "inceptionv4/inceptionv4"
        model = inceptionv4.inceptionv4()

    model = model.to(device)

    transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomHorizontalFlip(p=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_data = dsets.ImageFolder(TRAIN_DIR, transform=transform)
    test_data = dsets.ImageFolder(TEST_DIR, transform=transform)

    trainloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    criterion = nn.CrossEntropyLoss().to(device)

    optimstr = ''
    if optimIdx == 0:
        optimizer = AdamW(model.parameters(), lr=learningRate, weight_decay=weight_decay)
        optimstr = '_AdamW'
    elif optimIdx == 1:
        momentum = 0.7
        optimizer = optim.SGD(model.parameters(), lr=learningRate, momentum=momentum, weight_decay=weight_decay)
        optimstr = '_SGD' + '_momentum_' + str(momentum)

    title = model_out_path + optimstr + '_lr_' + str(learningRate) + '_l2_' + str(weight_decay)
    t_accs, v_accs, t_loss, v_loss = [], [], [], []
    print("Train Start")
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

        if epoch + 1 == epochs:
            torch.save(model.state_dict(), title + "_state_result.pth")
            torch.save(model, title + "_model_result.pth")
        elif (epoch + 1) // 5 == 0:
            torch.save(model.state_dict(), title + f"_%03d_" % (epoch) + "epoch_state.pth")
            torch.save(model, title + f"_%03d_" % (epoch) + "epoch_model.pth")

    plotResultGraph(t_accs, v_accs, t_loss, v_loss, model_out_path, title)


def plotResultGraph(t_accs, v_accs, t_loss, v_loss, out_path, title):

    plt.figure(figsize=(15,8))
    plt.suptitle(title)
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
    plt.savefig(title + '_result.png')
    plt.show()


if __name__ == '__main__':
    main()
