import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
import math

#from appscript import app

# Environment:
# OS    : Mac OS EL Capitan
# python: 3.5
# opencv: 2.4.13

# parameters
cap_region_x_begin=0.5  # start point/total width
cap_region_y_end=0.8  # start point/total width
threshold = 60  #  BINARY threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50
learningRate = 0

# variables
isBgCaptured = 0   # bool, whether the background captured
triggerSwitch = False  # if true, keyborad simulator works
listOfPoint = []

    

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def printThreshold(thr):
    print("! Changed threshold to "+str(thr))


def removeBG(frame):
    fgmask = bgModel.apply(frame,learningRate=learningRate)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # res = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res


def calculateFingers(res,drawing):  # -> finished bool, cnt: finger count
    #  convexity defect
    #listOfPoint = [()]
    global listOfPoint

    hull = cv2.convexHull(res, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(res, hull)
        if type(defects) != type(None):  # avoid crashing.   (BUG not found)

            cnt = 0
            for i in range(defects.shape[0]):  # calculate the angle
                s, e, f, d = defects[i][0]
                start = tuple(res[s][0])
                end = tuple(res[e][0])
                far = tuple(res[f][0])
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                    cnt += 1
                    cv2.circle(drawing, far, 3, [211, 84, 0], -1)
                    listOfPoint.append(far)
            return True, cnt
        
    

    return False, 0


def getOri(drawing):
    global listOfPoint
    oriIndex = listOfPoint[0]
    majIndex = listOfPoint[1]

    a = (majIndex[1] - oriIndex[1]) / (majIndex[0] - oriIndex[0])
    b = majIndex[1]
    lastPoint = ()
    pointO = [float(oriIndex[0]),float(oriIndex[1])]
    step = 1
    print (np.shape(drawing))
    while (pointO[0] < np.shape(drawing)[1]-2):
        pointO[0] =  pointO[0] + step
        pointO[1] = pointO[1] + (a * step)
        
        if drawing[int(pointO[1])][int(pointO[0])][1] == 255:
            cv2.circle(drawing, (int(pointO[0]),int(pointO[1])), 6, [255,0, 255], -1)
            lastPoint =  (int(pointO[0]),int(pointO[1]))
    listOfPoint.append(lastPoint)
           
def getIndex(drawing):
    global listOfPoint
    oriIndex = listOfPoint[2]
    majIndex = listOfPoint[1]

    a = (majIndex[1] - oriIndex[1]) / (majIndex[0] - oriIndex[0])
    b = majIndex[1]
    lastPoint = ()
    pointO = [float(oriIndex[0]),float(oriIndex[1])]
    step = 1
    print (np.shape(drawing))
    while (pointO[0] > 2):
        pointO[0] =  pointO[0] - step
        pointO[1] = pointO[1] - (a * step)
        
        if  drawing[int(pointO[1])][int(pointO[0])][1] == 255:
           lastPoint =  (int(pointO[0]),int(pointO[1]))
    listOfPoint.append(lastPoint)
    
def dist(pointA, pointB):
    return np.sqrt(np.power(pointA[0]-pointB[0],2) + np.power(pointA[1]-pointB[1],2))

def reducDist(pointA, pointB, drawing, sSize ):

    fpoint = None
    minDist = dist(pointA, pointB)
    
    for ii in range(int (pointB[1]-sSize/2), int(pointB[1]+sSize/2 ) ):
        for ij in range(int (pointB[0]-sSize/2), int (pointB[0]+sSize/2)  ):
            if  drawing[ii][ij][1] == 255:
                lDist = dist(pointA, (ij,ii))
                if (minDist > lDist ):
                    minDist = lDist
                    fpoint = (ij,ii)
    return fpoint

def reducRecDist(pointA, pointB, drawing, sSize ):
    fpoint = reducDist(pointA, pointB, drawing, sSize )
    lastPoint = fpoint
    if (fpoint is None):
        return pointB
    while (fpoint is not None):
        fpoint = reducDist(pointA, fpoint, drawing, sSize )
        if (fpoint is None):
            return lastPoint
        else:
            lastPoint = fpoint

def reducRecTotDist(pointA, pointB, drawing, sSize , sizeHist = None ):
    fpoint = reducRecDist(pointA, pointB, drawing, sSize )
    
    lastDist = dist(pointA, pointB)
    if sizeHist is not None:
        sizeHist.append(lastDist)
  
    if (dist(pointA, fpoint) < lastDist ):
        
        return reducRecTotDist(fpoint, pointA, drawing, sSize ,sizeHist)
    else :
        
        return[fpoint,pointA,sizeHist]

def imgShow(img):
    gmi = img[...,::-1]
    plt.imshow(gmi)

# Camera
img = cv2.imread('handi.jpg')



gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

## (2) Threshold
th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)


kernel = np.ones((3, 3), np.uint8)
closing = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE,
kernel, iterations=4)
cont_img = closing.copy()
img2, contours, hierarchy = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 2000 or area > 4000:
        continue
    if len(cnt) < 5:
        continue
    ellipse = cv2.fitEllipse(cnt)
    cv2.ellipse(threshed, ellipse, (0,255,0), 2)
    
thresh1 = copy.deepcopy(threshed)

imgShow(thresh1)
img2,contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
length = len(contours)
maxArea = -1
if length > 0:
    for i in range(length):  # find the biggest contour (according to area)
        temp = contours[i]
        area = cv2.contourArea(temp)
        if area > maxArea:
            maxArea = area
            ci = i

    res = contours[ci]
    hull = cv2.convexHull(res)
    drawing = np.zeros(img.shape, np.uint8)
    cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
    cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)

    cnt = calculateFingers(res,drawing)
    
    

gray = cv2.medianBlur(gray, 5)   
rows = gray.shape[0]
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8,param1=30, param2=50,minRadius=30, maxRadius=200)

circleRadius = 0
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        if (i[0] < int(drawing.shape[0]*0.2) and i[1] < int(drawing.shape[1]*0.2)):
            center = (i[0], i[1])
            # circle center
            cv2.circle(drawing, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv2.circle(drawing, center, radius, (255, 0, 255), 3)
            circleRadius = radius
        
getOri(drawing)
getIndex(drawing)

lBague = 15
moym = []
majeur = reducRecTotDist(listOfPoint[1], listOfPoint[2], drawing, dist(listOfPoint[0], listOfPoint[4])/2,moym)
print(np.average(moy))
cv2.line(img,majeur[0], majeur[1], (255,0,0),lBague)

moya = []
annulaire = reducRecTotDist(listOfPoint[0], listOfPoint[1], drawing, dist(listOfPoint[0], listOfPoint[4])/2,moya)
cv2.line(img,annulaire[0], annulaire[1], (0,255,0),lBague)

moyo = []
oriculaire = reducRecTotDist(listOfPoint[0], listOfPoint[4], drawing, dist(listOfPoint[0], listOfPoint[4])/2,moyo)
cv2.line(img,oriculaire[0], oriculaire[1], (0,0,255),lBague)

moyi = []
index = reducRecTotDist(listOfPoint[5], listOfPoint[2], drawing, dist(listOfPoint[0], listOfPoint[4])/2,moyi)
cv2.line(img,index[0], index[1], (0,0,255),lBague)

cv2.line(drawing,listOfPoint[0], listOfPoint[1], (0,0,255),2)

cv2.line(drawing,listOfPoint[1], listOfPoint[2], (0,255,0),2)
cv2.line(drawing,listOfPoint[0], listOfPoint[4], (255,0,0),2)
 
print(p1, p2)

cv2.line(drawing,listOfPoint[2], listOfPoint[5], (255,255,0),2)
def fingSize(doigt):
    fingerDistPix = dist(doigt[0], doigt[1])
    return(scaleSize(fingerDistPix))
    
def scaleSize(doigtDist):
    fingerDistPix = doigtDist
    diameter = radius * 2
    fingerDiameter = (fingerDistPix * 2.23)/diameter
    fingerSizeDoigt = fingerDiameter * 35
    return fingerSizeDoigt

main = {}
main["majeur"] = majeur
main["annulaire"] = annulaire
main["index"] = index
main["auriculaire"] = oriculaire
font = cv2.FONT_HERSHEY_SIMPLEX
    
for doigt in main.keys():
    cv2.putText(img,"%d/%d"%(scaleSize(np.average(main[doigt][2])),fingSize(main[doigt])),main[doigt][0], font, 0.5,(0,0,0),1,cv2.LINE_AA)
    print(doigt, fingSize(main[doigt]))



imgShow(drawing)



cv2.imwrite("img.png",img)
cv2.imwrite("test.png",drawing)
cv2.imwrite("threshed.png",threshed)



