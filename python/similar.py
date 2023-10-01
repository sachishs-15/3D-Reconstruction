import cv2
import numpy as np

def setcorners(x1, x2, y1, y2, sh):

    if(x1 < 0):
        x2 += abs(x1)
        x1 = 0

    elif(x2 > sh[1]):
        x1 = x1-(x2 - sh[1])
        x2 = sh[1]

    if(y1 < 0):
        y2 += abs(y1)
        y1 = 0

    elif(y2 > sh[0]):
        y1 = y1-(y2 - sh[0])
        y2 = sh[0]

    return (x1, x2, y1, y2)

def score(img1, img2, pt1, pt2):
    w = 8

    # x1 = max(0, pt1[0] - w, pt2[0] - w)
    # y1 = max(0, pt1[1] - w, pt2[1] - w)

    # x2 = min(img1.shape[1], pt1[0] + w, pt2[0] + w)
    # y2 = min(img1.shape[0], pt1[1] + w, pt2[1] + w)

    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    x1, x2, y1, y2 = setcorners(pt1[0] - w, pt1[0] + w, pt1[1] - w, pt1[1] + w, img1.shape)
    mat1 = img1[y1: y2, x1: x2]
    
    x1, x2, y1, y2 = setcorners(pt2[0] - w, pt2[0] + w, pt2[1] - w, pt2[1] + w, img1.shape)
    mat2 = img2[y1: y2, x1: x2]

    temp = mat1 - mat2
    distance = np.linalg.norm(temp)
    #print(distance)

    return distance

def most_similar(img1, img2, pt1, pts2):

    dists = []
    for x in pts2:
        dists.append(score(img1, img2, pt1, x))

    pos = dists.index(min(dists))

    return pts2[pos]

