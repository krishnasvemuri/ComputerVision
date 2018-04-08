#import the required numpy and cv2 packages
import numpy as np
import cv2
# Read the required image.
img1 = cv2.imread('krishna.jpg')
# Create a window 
cv2.namedWindow('original', cv2.WINDOW_NORMAL)
# Resize the window with given height and width parameters
cv2.resizeWindow('original',500,500)
# Show the image in the window
cv2.imshow('original',img1)
# wait infinitely until the window is closed 
cv2.waitKey(0)

#Convert the the original image into grayscale 
img2 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
cv2.namedWindow('gray', cv2.WINDOW_NORMAL)
cv2.resizeWindow('gray',500,500)
cv2.imshow('gray',img2)
cv2.waitKey(0)

#Blur(average blur) the original image by increasing the kernel size by 5.
trans1a = cv2.blur(img1,(5,5))
cv2.namedWindow('blur', cv2.WINDOW_NORMAL)
cv2.resizeWindow('blur',500,500)
cv2.imshow('blur',trans1a)
cv2.waitKey(0)

#Smooth the image by using the median filtering and the kernel size is 15
trans1b = cv2.medianBlur(img2,15)
cv2.namedWindow('medianfilter', cv2.WINDOW_NORMAL)
cv2.resizeWindow('medianfilter',500,500)
cv2.imshow('medianfilter',trans1b)
cv2.waitKey(0)


#Convert the original image from RGB to BGR
trans2a = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
cv2.namedWindow('RGB2BGR', cv2.WINDOW_NORMAL)
cv2.resizeWindow('RGB2BGR',500,500)
cv2.imshow('RGB2BGR',trans2a)
cv2.waitKey(0)

#Change black and white to white and black using substract 
trans2b = cv2.subtract(255, img2) 
cv2.namedWindow('B/W2W/B', cv2.WINDOW_NORMAL)
cv2.resizeWindow('B/W2W/B',500,500)
cv2.imshow('B/W2W/B',trans2b)
cv2.waitKey(0)

#The Canny Edge detection on the original image 
trans3a=cv2.Canny(img1,200,200)
cv2.namedWindow('CannyEdge', cv2.WINDOW_NORMAL)
cv2.resizeWindow('CannyEdge',500,500)
cv2.imshow('CannyEdge',trans3a)
cv2.waitKey(0)

#The laplacian transformation is performed on the gray scale image
#Further the noise is removed by using Gaussian Blur
laplacian=cv2.Laplacian(img2,cv2.CV_64F)
trans3b=cv2.GaussianBlur(laplacian,(7,7),0)
cv2.namedWindow('laplacian', cv2.WINDOW_NORMAL)
cv2.resizeWindow('laplacian',500,500)
cv2.imshow('laplacian',trans3b)
cv2.waitKey(0)

#Series of flip transformation on the original image are performed.
trans4a= cv2.flip( img1, 0 )
cv2.namedWindow('horizontalflip', cv2.WINDOW_NORMAL)
cv2.resizeWindow('horizontalflip',500,500)
cv2.imshow('horizontalflip',trans4a)
cv2.waitKey(0)
trans4b= cv2.flip( img1, 1 )
cv2.namedWindow('verticalflip', cv2.WINDOW_NORMAL)
cv2.resizeWindow('verticalflip',500,500)
cv2.imshow('verticalflip',trans4b)
cv2.waitKey(0)
trans4c= cv2.flip( img1, -1 )
cv2.namedWindow('completeflip', cv2.WINDOW_NORMAL)
cv2.resizeWindow('completeflip',500,500)
cv2.imshow('completeflip',trans4c)
cv2.waitKey(0)

#Rotate the gray image by 45 degrees.
rows,cols = img2.shape
rotation= cv2.getRotationMatrix2D((cols/2,rows/2),45,1)
trans4d = cv2.warpAffine(img2,rotation,(cols,rows))
cv2.namedWindow('rotate', cv2.WINDOW_NORMAL)
cv2.resizeWindow('rotate',500,500)
cv2.imshow('rotate',trans4d)
cv2.waitKey(0)


# Erosion is performed on the original image
kernel = np.ones((3,3),np.uint8)
trans5a = cv2.erode(img1,kernel,iterations = 1)
cv2.namedWindow('erosion', cv2.WINDOW_NORMAL)
cv2.resizeWindow('erosion',500,500)
cv2.imshow('erosion',trans5a)
cv2.waitKey(0)

# Dilation is performed on the gray image 
trans5b = cv2.dilate(img2,kernel,iterations = 1)
cv2.namedWindow('dilate', cv2.WINDOW_NORMAL)
cv2.resizeWindow('dilate',500,500)
cv2.imshow('dilate',trans5b)
cv2.waitKey(0)




cv2.destroyAllWindows()
