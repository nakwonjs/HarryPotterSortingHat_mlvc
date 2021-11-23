import torch
from torchvision import models
import torchvision.transforms as transforms
def check_device():
	print("### Device Check list ###")
	is_cuda =  torch.cuda.is_available()
	print("GPU available?:", torch.cuda.is_available())
	if is_cuda:
		device_number = torch.cuda.current_device()
		print("Device number:", device_number)
		print("Is device?:", torch.cuda.device(device_number))
		print("Device count?:", torch.cuda.device_count())
		print("Device name?:", torch.cuda.get_device_name(device_number))
	print("### ### ### ### ### ###\n\n")
	return torch.device('cuda' if is_cuda else 'cpu')

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
