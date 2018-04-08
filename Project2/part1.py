#import all the required python libraries
import tkinter as tk
import numpy as np
import cv2
from tkinter import filedialog
import os
from matplotlib import pyplot as plt

#This function is responsible for  displaying the 11 * 11 image window on left button mouse click
#Also displays the pixel values, RGB values and intensity at the location
#mean and standard deviation of the image window 
def mouse_move(event,x,y,flags,param):
    while(1):
        if event == cv2.EVENT_LBUTTONDOWN:
           crop_img = img[y-5:y+5,x-5:x+5]
           constant= cv2.copyMakeBorder(crop_img,2,2,2,2,cv2.BORDER_CONSTANT,value=(255,255,255))
           cv2.namedWindow('cropped',cv2.WINDOW_NORMAL)
           cv2.imshow('cropped', constant)
           pix=[x,y]
           print('The location of the pixel',pix)
           b,g,r=img[x,y]
           R=np.uint16(r)
           G=np.uint16(g)
           B=np.uint16(b)
           print ('The R G B values of the pixel are',R,G,B)
           print ('The intensity of the pixel are',float((R + G + B) / 3))
           mean,sdeviation=cv2.meanStdDev(crop_img)
           bm,gm,rm=mean
           bs,gs,rs=sdeviation
           print ('The mean of the 11 * 11 window is',rm,gm,bm)
           print ('The standard deviation of the 11 * 11 window is',rs,gs,bs)
        if cv2.waitKey(20) & 0xFF == 27:
            break
#Defined the type of files that can be loaded
my_filetypes = [('png', '.png'),('bnp', '.bnp'),('tiff', '.tiff'),('jpg','.jpg')]
#Using the tkinter library used the file explorer to select the required image
answer = filedialog.askopenfilename(initialdir=os.getcwd(),
                                    title="Please select an image:",
                                    filetypes=my_filetypes)
img = cv2.imread(answer)
cv2.imshow('image',img)
cv2.waitKey(0)
#Split the channels of the image to get the coor spaces which are used to create a histogram 
channels = cv2.split(img)
colors = ("b", "g", "r")
for(channel, c) in zip(channels, colors):
    histogram = cv2.calcHist([channel], [0], None, [256], [0, 256])
    plt.plot(histogram, color = c)
    plt.xlim([0, 256])
plt.show()
#Call the function mouse_move on the image
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.setMouseCallback('image',mouse_move)
#While true, open the image and close the image window when close button the image is clicked 
while(1):
    cv2.imshow('image',img)
    cv2.resizeWindow('image',500,500)
    cv2.waitKey(0)
    if cv2.waitKey(20) & 0xFF == 27:
        break
    cv2.destroyAllWindows()
