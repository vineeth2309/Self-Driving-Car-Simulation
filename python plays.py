import numpy as np
from PIL import ImageGrab
import cv2
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import math


def f(x):
    return x

last_time=time.time()
kernel2 = np.ones((5,5) , np.uint8)

linelength = []
cv2.namedWindow('trackbars')
cv2.createTrackbar('Lower Edge','trackbars',0,500,f)
cv2.createTrackbar('Higher Edge','trackbars',0,500,f)
cv2.createTrackbar('Lower Edge Threshold','trackbars',0,500,f)

set1=0

def HoughLines(img,processed_image):
    lines = cv2.HoughLinesP(img,1,np.pi/180,80,np.array([]),100,5)
    lineneedssorting=draw_lines(processed_image,lines)
    length_sorted = sorted(lineneedssorting, reverse=True)
    return length_sorted

def draw_lines(img,lines):
    try:    
        for line in lines:
            coords=line[0]
            cv2.line(img,(coords[0],coords[1]),(coords[2],coords[3]),(255,255,255),10)
            #print('1')
            l1 =(pow((coords[2]-coords[0]),2) + pow((coords[3]-coords[1]),2))
            final_length=math.sqrt(l1)
            linelength.append(final_length)
    except:
        pass
    return linelength
    
def roi(image,vertices):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,vertices,255)
    masked = cv2.bitwise_and(image,mask)
    #dilation = cv2.dilate(masked,kernel2,iterations = 1)
    return masked

def process_image(original_image,lower_edge,higher_edge,lower_edge_threshold):
    new_image = cv2.cvtColor(original_image,cv2.COLOR_BGR2GRAY)
    #ret,new_image = cv2.threshold(original_image,lower_edge_threshold,255,cv2.THRESH_BINARY)
    #new_image = cv2.adaptiveThreshold(new_image1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
           # cv2.THRESH_BINARY,3,2)
    kernel = np.ones((5,5),np.float32)/25
    dst = cv2.filter2D(new_image,-1,kernel)
    blur = cv2.GaussianBlur(dst,(5,5),0)
    erosion = cv2.erode(blur,kernel2,iterations = 1)
    dilation = cv2.dilate(erosion,kernel2,iterations = 1)
    #ret,new_image = cv2.threshold(dilation,lower_edge_threshold,255,cv2.THRESH_BINARY)
    #edges = cv2.Canny(new_image,lower_edge,higher_edge,apertureSize=3)
    edges = cv2.Canny(dilation,lower_edge,higher_edge,apertureSize=3)
    #edges = cv2.Canny(blur,lower_edge,higher_edge,apertureSize=3)
    return edges


while True:

    printscreen_pil = ImageGrab.grab(bbox=(0,10,640,480))
    #current_time = time.time()
    #print("Time per Frame:",(current_time-last_time))
    le=cv2.getTrackbarPos('Lower Edge','trackbars')
    he=cv2.getTrackbarPos('Higher Edge','trackbars')
    let=cv2.getTrackbarPos('Lower Edge Threshold','trackbars')
    processed_image = process_image(np.array(printscreen_pil),le,he,let)
    vertices = np.array([[0,375],[200,190],[400,190],[640,375],[640,480],[0,480]])
    new_image=roi(processed_image,[vertices])

    sorted_lines=HoughLines(new_image,processed_image)
    

    cv2.imshow('graywindow',processed_image)
    k=cv2.waitKey(1)
    if k == ord('q'):
        break
   
