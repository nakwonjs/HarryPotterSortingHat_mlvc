import facealigner 
from imutils.face_utils import rect_to_bb
import argparse 
import imutils
import dlib
import cv2
import os
import torch
import torchvision.transforms as transforms

from tkinter import filedialog
from tkinter import *
import cv2

root=Tk()
root.title("Harry Potter Sorting Hat Simulator")
root.resizable(False, False)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
fa = facealigner.FaceAligner(predictor, desiredFaceWidth=128)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def SortHat(img):
    trans_img = transform(img)
    ti = trans_img.unsqueeze(0)
    output = model(ti)
    _, predicted = torch.max(output.data, 1)
    pred = predicted[0]

    if 0 <= pred <= 11:
        return 'Slytherin!'
    elif 12 <= pred <= 18:
        return 'Ravenclaw!'
    elif 19 <= pred <= 26:
        return 'Gryffindor!'
    elif 27 <= pred <= 31:
        return 'Hufflepuff!'
    
model = torch.load('mymodel.h5', map_location=torch.device('cpu'))

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
            try:
                (x, y, w, h) = rect_to_bb(rect)
                faceOrig = imutils.resize(image[y:y+h, x:x+w], width=128)
                faceAligned = fa.align(image, gray, rect)
                dormitory = SortHat(faceAligned)
                txt.configure(text=dormitory)
                cv2.imshow('img', faceAligned)
                cv2.waitKey(0)
            except:
                txt.configure(text='UNVALID PICTURE. CHOOSE ANOTHER PICTURE')
                continue

        file.close()


btn=Button(None,text="UPLOAD YOUR PIC!",command=clickButton)
btn.pack()

lbl = Label(root, text=' ')
lbl.pack()

txt = Label(root, text=' ')
txt.pack()

root.mainloop()
