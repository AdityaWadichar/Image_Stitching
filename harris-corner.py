import numpy as np
import cv2

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



# Calling Harris Algorithm

img = cv2.imread('images/dataset1/0.bmp',0)
corners = harris_corners(img, 1, True)
print(corners)