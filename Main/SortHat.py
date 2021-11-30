import facealigner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
import cv2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from tkinter import filedialog
import tkinter as tk
import cv2
from PIL import Image
from PIL import ImageTk
from dpn import *
import random

seed = 2021
torch.manual_seed(seed)
random.seed(seed)

transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomHorizontalFlip(p=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def SortHat(img):
    trans_img = transform(img)
    ti = trans_img.unsqueeze(0)
    output = model(ti)
    _, predicted = torch.max(output.data, 1)
    pred = predicted[0]
    print(pred)
    if 0 <= pred <= 11:
        return 'Slytherin!'
    elif 12 <= pred <= 18:
        return 'Ravenclaw!'
    elif 19 <= pred <= 26:
        return 'Gryffindor!'
    elif 27 <= pred <= 31:
        return 'Hufflepuff!'


def App():
    root = tk.Tk()
    root.title("Harry Potter Sorting Hat Simulator")
    root.resizable(False, False)
    def clickButton():

        file = filedialog.askopenfile(initialdir='path', title='Choose a file',
                                      filetypes=(('jpg files', '*.jpg'), ('all files', '*.*')))

        lbl.configure(text='Your Dormitory Is... ')

        if file != None:
            data = file.name
            image = cv2.imread(data)
            image = imutils.resize(image, width=800)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 2)

            for rect in rects:
                # try:
                    (x, y, w, h) = rect_to_bb(rect)
                    faceOrig = imutils.resize(image[y:y + h, x:x + w], width=128)
                    faceAligned = fa.align(image, gray, rect)

                    userImg = faceAligned
                    userImg = cv2.cvtColor(userImg, cv2.COLOR_BGR2RGB)
                    userImg = Image.fromarray(userImg)

                    userphoto = ImageTk.PhotoImage(image=userImg)
                    userphoto = userphoto._PhotoImage__photo.zoom(2)
                    lbl2.configure(image = userphoto)
                    lbl2.image = userphoto

                    faceAligned = Image.fromarray(faceAligned)
                    dormitory = SortHat(faceAligned)
                    if dormitory == 'Slytherin!':
                        lbl.configure(image=slyphoto)
                    elif dormitory == 'Ravenclaw!':
                        lbl.configure(image=ravphoto)
                    elif dormitory == 'Gryffindor!':
                        lbl.configure(image=gryphoto)
                    elif dormitory == 'Hufflepuff!':
                        lbl.configure(image=huffphoto)

                    txt.configure(text=dormitory)

                # except:
                #      txt.configure(text='UNVALID PICTURE. CHOOSE ANOTHER PICTURE')
                #      continue

            file.close()

    model.load_state_dict(ckpt)
    btn = tk.Button(None, text="UPLOAD YOUR PIC!", command=clickButton)
    btn.pack()

    pFrame = tk. Frame(master=root,
                         width=800,
                         height=500,)
    imgFrame = tk.Frame(master=pFrame)

    pFrame.pack()
    imgFrame.pack()
    pFrame.pack_propagate(0)

    hatImg = 'img/sorthat.jpg'
    baseImg = 'img/base.png'
    gryImg = 'img/gryffindor.png'
    huffImg = 'img/hufflepuff.png'
    ravImg = 'img/ravenclaw.png'
    slyImg = 'img/slytherin.png'

    im = Image.open(hatImg)
    hatphoto = ImageTk.PhotoImage(im)
    hatphoto = hatphoto._PhotoImage__photo.subsample(2)
    basephoto = tk.PhotoImage(file = baseImg).subsample(2,2)
    gryphoto= tk.PhotoImage(file = gryImg).subsample(2,2)
    huffphoto = tk.PhotoImage(file = huffImg).subsample(2,2)
    ravphoto = tk.PhotoImage(file = ravImg).subsample(2,2)
    slyphoto= tk.PhotoImage(file = slyImg).subsample(2,2)

    lbl = tk.Label(master=imgFrame, image = basephoto)
    lbl2 = tk.Label(master=imgFrame, image = hatphoto)
    lbl2.image = hatphoto

    lbl.pack(side = "left")
    lbl2.pack(side = "left")

    txtFrame = tk.Frame(master=root,
                         width=800,
                         height=50,)
    txtFrame.pack()
    txtFrame.pack_propagate(0)
    txt = tk.Label(txtFrame, text=' ', font=("Arial", 25))
    txt.pack()
    root.mainloop()


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
fa = facealigner.FaceAligner(predictor, desiredFaceWidth=128)
model = torch.load('dpn131_AdamW_lr_0.0003_l2_0.0001_model_result.pth', map_location=torch.device('cpu'))
ckpt = torch.load('dpn131_AdamW_lr_0.0003_l2_0.0001_state_result.pth', map_location=torch.device('cpu'))
App()


