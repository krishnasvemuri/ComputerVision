#import all the required python libraries 
import cv2
import numpy as np
import glob
import os
from matplotlib import pyplot as plt
path1 = 'C:/Python36/GrayHistograms/'
path2 = 'C:/Python36/ColorHistograms/'
path3 = 'C:/Python36/Results/'
# The function that calculates the histogram intersection 
def calculate_intersection(h1,h2):
    sum_min=0
    sum_max=0
    for i in range(0,36):
        sum_min += min(h1[i],h2[i])
        sum_max += max(h1[i],h2[i])
    return float(sum_min/sum_max)
#The function that calculates the chi-squared measure when (h1[i] + h2[i]) > 5
def calculate_chi_square(h1,h2):
    chi_square_distance = 0
    for i in range(0,36):
       if (h1[i] + h2[i]) > 5:
          chi_square_distance += (((h1[i] - h2[i])**2)/float(h1[i] + h2[i]))
    return chi_square_distance
#The function used to change the gray value to color value for the histogram intersection
def visualize_intersection(intersection):
     intersection = 255*np.array(intersection)
     return intersection
#The function used to change the gray value to color value for the chi-squared measure
#I have normalized the chi_square function.
def visualize_chi_square(chi_square):
    chi_square =(255 - (255 * (((np.array(chi_square))-(np.amin(chi_square)))/((np.amax(chi_square))-(np.amin(chi_square))))))
    return chi_square
count1=4000;
# Stores the 36-bin histograms of all gray images 
gray_hist=[]
for img in glob.glob('ST2MainHall/*.jpg'):
    image = cv2.imread(img,0)
    blur = cv2.GaussianBlur(image,(7,7),2)
    canny = cv2.Canny(blur,100,200)
    mask = np.uint8(canny)
    Gx = cv2.Sobel(blur,cv2.CV_64F,1,0,ksize=5)
    Gy = cv2.Sobel(blur,cv2.CV_64F,0,1,ksize=5)
    count1 = count1+1;
    magnitude,phase = cv2.cartToPolar(Gx,Gy,angleInDegrees=True)
    temp = np.round(np.divide(phase,10))
    angle = np.uint8(temp)
    hist=cv2.calcHist([angle],[0],mask,[36],[0,36])
    plt.plot(hist)
    plt.savefig(path1+'Gray_Histogram'+str(count1)+'.png')
    plt.clf()
    gray_hist.append(hist)
print('Plotted Gray Edge Histograms')
count2=4000;
# Stores the 36-bin histograms of all color images 
color_hist=[]
for img in glob.glob('ST2MainHall/*.jpg'):
    image = cv2.imread(img)
    blur = cv2.GaussianBlur(image,(7,7),2)
    b,g,r =cv2.split(blur)
    rcanny = cv2.Canny(r,100,200)
    gcanny = cv2.Canny(g,100,200)
    bcanny = cv2.Canny(b,100,200)
    canny = rcanny+gcanny+bcanny
    mask = np.uint8(canny)
    Rx = cv2.Sobel(r,cv2.CV_64F,1,0,ksize=5)
    Ry = cv2.Sobel(r,cv2.CV_64F,0,1,ksize=5)
    Gx = cv2.Sobel(g,cv2.CV_64F,1,0,ksize=5)
    Gy = cv2.Sobel(g,cv2.CV_64F,0,1,ksize=5)
    Bx = cv2.Sobel(b,cv2.CV_64F,1,0,ksize=5)
    By = cv2.Sobel(b,cv2.CV_64F,0,1,ksize=5)
    count2 = count2+1;
    U= Rx+Gx+Bx
    V= Ry+Gy+By
    magnitude,phase = cv2.cartToPolar(U,V,angleInDegrees=True)
    temp = np.round(np.divide(phase,10))
    angle = np.uint8(temp)
    hist=cv2.calcHist([angle],[0],mask,[36],[0,36])
    plt.plot(hist)
    plt.savefig(path2+'Color_Histogram'+str(count2)+'.png')
    plt.clf()
    color_hist.append(hist)

print('Plotted Color Edge Histograms')
#Create empty matrices of size 99 * 99 to store the histogram comparison results
gray_intersection_matrix = np.zeros((99,99))
gray_chi_square_matrix = np.zeros((99,99))

color_intersection_matrix = np.zeros((99,99))
color_chi_square_matrix = np.zeros((99,99))
for i in range(0,99):
    for j in range(0,99):
        gray_intersection_matrix[i][j] = calculate_intersection(gray_hist[i],gray_hist[j])
        color_intersection_matrix[i][j] = calculate_intersection(color_hist[i],color_hist[j])
        gray_chi_square_matrix[i][j] = calculate_chi_square(gray_hist[i],gray_hist[j])
        color_chi_square_matrix[i][j] = calculate_chi_square(color_hist[i],color_hist[j])
gray_intersection_matrix = visualize_intersection(gray_intersection_matrix)
color_intersection_matrix = visualize_intersection(color_intersection_matrix) 
gray_chi_square_matrix = visualize_chi_square(gray_chi_square_matrix)
color_chi_square_matrix = visualize_chi_square(color_chi_square_matrix) 
print("Gray Histogram Intersection Done")
#Save and plot the Gray Histogram Intersection Comparison
plt.imshow(gray_intersection_matrix,cmap='rainbow',interpolation='nearest')
plt.colorbar()
plt.savefig(path3+'Gray Intersection_Comparison.png')
plt.title('Gray Intersection Comparison')
plt.show()
plt.clf()
print("Gray Chi-Square Measure Done")
#Save and plot the Gray Chi-Square Comparison 
plt.imshow(gray_chi_square_matrix,cmap='rainbow',interpolation='nearest')
plt.colorbar()
plt.savefig(path3+'Gray_Chi_Square_Comparison.png')
plt.title('Gray Chi Square Comparison')
plt.show()
plt.clf()
   
print("Color Histogram Intersection Done")
#Save and plot the Color Histogram Intersection Comparison
plt.imshow(color_intersection_matrix,cmap='rainbow',interpolation='nearest')
plt.colorbar()
plt.savefig(path3+'Color Intersection_Comparison.png')
plt.title('Color Intersection Comparison')
plt.show()
plt.clf()
 
print("Color Chi-Square Measure Done")
#Save and plot the Chi-Square Comparison
plt.imshow(color_chi_square_matrix,cmap='rainbow',interpolation='nearest')
plt.colorbar()
plt.savefig(path3+'Color_Chi_Square_Comparison.png')
plt.title('Color Chi Square Comparison')
plt.show()
plt.clf()
