import cv2
import numpy as np
import os
import math

path = os.getcwd()

image_path = path + '/images/dataset1' 
os.chdir(image_path)
def convol(img,kernel):
    # Convolution of image matrix with 3X3 kernel
    y, x = img.shape
    padx = 1
    pady = 1
    paddedimg = np.zeros((y + 2 * pady, x + 2 * padx))
    paddedimg[pady:-pady, padx:-padx] = img[:]
    opimg = np.zeros((y,x))
    for i in range(1, y+1):
        for j in range(1, x+1):
            opimg[i-1, j-1] = np.sum(kernel * paddedimg[i-1:i + 2, j-1:j + 2])

    return opimg

def harris_corners(img,tf = 1,display = False ):
    '''

    :param img: ndarray, 2-D (greyscale) Image matrix
    :param tf: float, threshold factor (between 1 and 3)
    :param display: boolean, To create image of corners
    :return: list: 2-D list of corner indices

    '''

    # Preprocessing
    img = cv2.GaussianBlur(img, (5, 5), 0)
    y, x = img.shape
    img = cv2.normalize(img, img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Sobel partial differentiation operation
    padx = 1
    pady = 1
    paddedimg = np.zeros((y + 2 * pady, x + 2 * padx))
    paddedimg[pady:-pady, padx:-padx] = img[:]

    sobelx = np.array([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]])

    sobely = np.array([[1, 2, 1],
                       [0, 0, 0],
                       [-1, -2, -1]])

    sobelopx = convol(img, sobelx)
    sobelopy = convol(img, sobely)

    # Calculating Structure Tensor
    Ixx = sobelopx * sobelopx
    Ixy = sobelopx * sobelopy
    Iyx = sobelopy * sobelopx
    Iyy = sobelopy * sobelopy

    # Calculating windowed derivatives
    rect_filter = 1 / 8 * np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    wind_Ixx = convol(Ixx, rect_filter)
    wind_Ixy = convol(Ixy, rect_filter)
    wind_Iyx = convol(Iyx, rect_filter)
    wind_Iyy = convol(Iyy, rect_filter)

    # Harris response calculation
    M = np.zeros((2, 2))
    k = 0.04
    R = np.zeros(img.shape)
    for i in range(0, y):
        for j in range(0, x):
            M[0, 0] = wind_Ixx[i, j]
            M[0, 1] = wind_Ixy[i, j]
            M[1, 0] = wind_Iyx[i, j]
            M[1, 1] = wind_Iyy[i, j]

            determinant = np.linalg.det(M)
            trace = np.trace(M)
            R[i, j] = determinant - (k * (trace ** 2))

    # Threshold calculation
    corner_Rsum = 0
    count = 0
    finalimg = np.zeros((y, x))
    for i in range(0, y):
        for j in range(0, x):
            if R[i, j] > 0:
                corner_Rsum += R[i, j]
                count += 1
                finalimg[i, j] = 255

    threshold = tf * corner_Rsum / count

    # Non-maximal suppression
    finalimg2 = np.zeros((y, x))
    s = 10
    maxindices = []
    for i in range(0, y, s):
        for j in range(0, x, s):
            maxindex = np.unravel_index(R[i:i + s, j:j + s].argmax(), (s, s))
            try:
                if R[i + maxindex[0], j + maxindex[1]] > threshold:
                    maxindices.append([i + maxindex[0], j + maxindex[1]])
                    finalimg2[i + maxindex[0], j + maxindex[1]] = 255
            except:
                pass

    if display:
        cv2.imwrite('final2.png', finalimg2)
    return maxindices



def required_img(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)
    image,contour,heic = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contour[0]
    cnt = cnt.reshape(-1,2)
    x_min = np.min(cnt[:,0],axis = 0) 
    y_min = np.min(cnt[:,1],axis = 0)
    x_max = np.max(cnt[:,0],axis = 0)
    y_max = np.max(cnt[:,1],axis = 0)
    img2 = np.zeros((y_max,x_max,3),np.uint8)
    img2 = img[y_min:y_max,x_min:x_max]
    # cv2.waitKey(0)
    return img2

def image_stiching(img1,img2):
    gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    flag = np.array((gray1,gray2))
    indx = np.argmax(flag,axis=0)
    ind1 = np.where(indx ==0)
    ind2 = np.where(indx==1)
    img = np.zeros_like(img1)
    img[ind1] = img1[ind1]
    img[ind2] = img2[ind2]
    return  img


listdir = os.listdir()
listdir = sorted(listdir)

img1 = cv2.imread(listdir[0])
#img1 = cv2.resize(img1, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

canvas = np.zeros((img1.shape[0]*4,img1.shape[1]*5,img1.shape[2]),np.uint8)

canvas[150:150+img1.shape[0],100:100+ img1.shape[1]] = img1
img3 = canvas.copy()
i = 1

exten_list = ['.jpg','jpeg','.bmp','.png']
features_extractor = "sift"
print(listdir)
# read only when extension is of type ['.jpg','jpeg',".bmp",'.png']
while i < len(listdir):
        img1 = img3
        if  listdir[i][-4:] not in exten_list:
            continue
        img2 = cv2.imread(listdir[i])
     #  img2 = cv2.resize(img2,None, fx=0.5,fy = 0.5, interpolation=cv2.INTER_CUBIC) 
        gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
    
        if features_extractor.upper() == "SIFT":

            kp1,desc1 = sift.detectAndCompute(gray1,None)
            kp2,desc2 = sift.detectAndCompute(gray2,None)
        else:
            kp1 = []
            kp2 = []
            corner1 = harris_corners(gray1,1.2,False)
            corner2 = harris_corners(gray2,1.2,False)
            for j in range(len(corner1)):
                    kp1.append(cv2.KeyPoint(math.floor(corner1[j][1]),math.floor(corner1[j][0]),10))
            for j in range(len(corner2)):
                    kp2.append(cv2.KeyPoint(math.floor(corner2[j][1]),math.floor(corner2[j][0]),10))
            kp1,desc1 = sift.compute(gray1,kp1)
            kp2,desc2 = sift.compute(gray2,kp2)

        bf = cv2.BFMatcher(crossCheck=False)
        matches = bf.knnMatch(desc1,desc2,k=2)
        good = []
        pt1 = []
        pt2 = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)
        good = sorted(good,key = lambda x:x.distance)
        good = good
        pt1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        pt2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        print(listdir[i],pt1.shape,pt2.shape)
        if pt1.shape[0]  <= 20 or pt2.shape[0] <= 20:
            i += 1
            continue
        else:
             pass 
        
        H,mask = cv2.findHomography(pt1,pt2,cv2.RANSAC,ransacReprojThreshold=4.0)
        if H is not None:
            H = np.linalg.inv(H)
            img3 = cv2.warpPerspective(img2,H,(canvas.shape[1],canvas.shape[0]))
            canvas = image_stiching(canvas,img3)
        i += 1
        if cv2.waitKey(2) == 2:
            break

canvas = required_img(canvas)
cv2.imshow('panorama',canvas)
cv2.imwrite('panorama.jpg',canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
