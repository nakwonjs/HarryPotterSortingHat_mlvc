import torch
from torchvision import models
import torchvision.transforms as transforms


def check_device():
    print("### Device Check list ###")
    is_cuda = torch.cuda.is_available()
    print("GPU available?:", torch.cuda.is_available())
    if is_cuda:
        device_number = torch.cuda.current_device()
        print("Device number:", device_number)
        print("Is device?:", torch.cuda.device(device_number))
        print("Device count?:", torch.cuda.device_count())
        print("Device name?:", torch.cuda.get_device_name(device_number))
    print("### ### ### ### ### ###\n\n")
    return torch.device('cuda' if is_cuda else 'cpu')


def ComposeDataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
