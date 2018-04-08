#import all the required python libraries 
import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob
# The function that calculates the histogram intersection 
def calculate_intersection(h1,h2):
    sum_min=0
    sum_max=0
    for i in range(0,512):
        sum_min += min(h1[i],h2[i])
        sum_max += max(h1[i],h2[i])
    return float(sum_min/sum_max)
#The function that calculates the chi-squared measure when (h1[i] + h2[i]) > 5
def calculate_chi_square(h1,h2):
    chi_square_distance = 0
    for i in range(0,512):
       if (h1[i] + h2[i]) > 5:
          chi_square_distance += (((h1[i] - h2[i])**2)/float(h1[i] + h2[i]))
    return chi_square_distance
#The function used to change the gray value to color value for the histogram intersection
def color_intersection(intersection):
    return int(255*intersection)
#The function used to change the gray value to color value for the chi-squared measure
def color_chi_square(chi_square):
    return int(255-(chi_square/2000))
#The function used to create an index using the [((r/32)*64)+((g/32)*8)+(b/32)] formula
def create_index(img):
   image = cv2.imread(img)
   b,g,r = cv2.split(image)
   B= np.uint16(b)
   G= np.uint16(g)
   R= np.uint16(r)
   index = (((R>>5)<<6)+((G>>5)<<3)+(B>>5))
   return index
#The stores the 512 bin histograms of each image 
all_hist=[]
#Calculate histograms for all the images and save them and append to the exixting list
for img in glob.glob('ST2MainHall/*.jpg'):
    hist,bins = np.histogram(create_index(img),512,[0,511])
    #plt.plot(hist)
    #plt.savefig(str(img)+'_histogram.png')
    #plt.clf()
    all_hist.append(hist)
print("Saved all histograms")
#Create empty matrices of size 99 * 99
intersection_matrix = np.zeros((99,99))
chi_square_matrix = np.zeros((99,99))
#For all image combinations of the images we calculate the histogram intersection and chi_square measure
#Also the resultant value is changed to a color value using the color_intersection and color_chi_square function
for i in range(0,99):
    for j in range(0,99):
        intersection =  calculate_intersection(all_hist[i],all_hist[j])
        intersection_matrix[i][j]= color_intersection(intersection)
        chi_square = calculate_chi_square(all_hist[i],all_hist[j])
        chi_square_matrix[i][j]= color_chi_square(chi_square)
print("Histogram Intersection Done")
#Save and plot the Histogram Intersection Comparison
plt.imshow(intersection_matrix,cmap='rainbow',interpolation='nearest')
plt.colorbar()
plt.savefig('Intersection_Comparison.png')
plt.title('Intersection Comparison')
plt.show()
plt.clf()
#Save and plot the Chi-Square Comparison 
print("Chi-Square Measure Done")
plt.imshow(chi_square_matrix,cmap='rainbow',interpolation='nearest')
plt.colorbar()
plt.savefig('Chi_Square_Comparison.png')
plt.title('Chi Square Comparison')
plt.show()
plt.clf()
