#!/usr/bin/env python
import cv2
import numpy as np

# always remember:
# Any primitive type from the list can be defined by an
# identifier in the form
# CV_<bit-depth>{U|S|F}C(<number_of_channels>)


def getSobelEdge(img , size=3):
    sx = cv2.Sobel(img, cv2.CV_64F,1,0,ksize=size)
    sy = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=size)
    edges = np.sqrt(np.square(sx) + np.square(sy))
    edges = np.floor((edges / (np.max(edges)+0.001)) * 255 ).astype(np.uint8)
    return edges


# some global parameters
MEDIAN_BLUR_FILTER_SIZE = 5
LAPLACIAN_FILTER_SIZE = 5
EDGES_THRESHOLD = 10
# loading an image
img = cv2.imread("t.png")

# converting an image to the grayscale
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# median blurring the image
imgray = cv2.medianBlur(imgray , MEDIAN_BLUR_FILTER_SIZE)

# applying the laplace filter to get the edges
# edges = cv2.Laplacian(imgray, cv2.CV_8U ,  LAPLACIAN_FILTER_SIZE).astype(np.uint8)
edges = getSobelEdge(imgray , -1)
# kernel = np.ones((5,5),np.uint8)
# erosion = cv2.erode(img,kernel,iterations = 1)
# creating the mask
_ , mask = cv2.threshold(edges , EDGES_THRESHOLD , 255 , cv2.THRESH_BINARY_INV )

cv2.imread(img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

###############################################################

# generating a color painting

# A strong bilateral filter smoothes flat regions while keeping edges sharp, and
# is therefore great as an automatic cartoonifier or painting filter, except that it
# is extremely slow (that is, measured in seconds or even minutes rather than
# milliseconds!).

# trick:-
# perform bilateral filtering at a lower resolution
# therefore if the number of pixels is more than : 400 * 400
# reduce the resolution by the factor of 4
# if the resolution is below (150 * 150) : increase it by factor of 2

height , width, _ = img.shape
img = cv2.resize(img,(width//2, height//2), interpolation = cv2.INTER_LINEAR)

# four parameters that control the bilateral filter:
# color strength, positional strength, size, and repetition count

# use resizedImg instead of the img
tmp = img
# repetitions for stronger cartoon effect
REPETITIONS = 17
# filter size
ksize = 9
# Filter color strength
sigmaColor = 9
# spatial strength
sigmaSpace = 7

for i in range(REPETITIONS):
    tmp = cv2.bilateralFilter(img, ksize, sigmaColor , sigmaSpace)
    img = cv2.bilateralFilter(tmp , ksize, sigmaColor , sigmaSpace)

img = cv2.resize(img, (width, height))
newimg = np.zeros(img.shape).astype(np.uint8)
print(newimg.shape)
for i in range(3):
    newimg[:,:,i] = cv2.bitwise_and(img[:,:,i],mask)

# cv2.imshow("cartoon" , img)
cv2.imshow("cartoon-border" , newimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
