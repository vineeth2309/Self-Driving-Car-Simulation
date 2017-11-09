import numpy as np
from PIL import ImageGrab
import cv2
import tensorflow as tf
import math

def f(x):
    return x

kernel = np.ones((5,5),np.float32)/25
kernel2 = np.ones((5,5) , np.uint8)

linelength = []

set1=0


def HoughLines(img,processed_image):
    lines = cv2.HoughLinesP(img,1,np.pi/180,80,np.array([]),20,300)
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

def average_slope_intercept(lines):
    left_lines    = [] 
    left_weights  = [] 
    right_lines   = [] 
    right_weights = []
    try: 
        for line in lines:
            for x1,y1,x2,y2 in line:
                if (x1==x2):
                    continue
                #y=mx+c
                else:
                    m = ((y2-y1)/(x2-x1))
                    c = (y1 - (m*x1))
                    line_length = np.sqrt(pow((y2-y1),2)+pow((x2-x1),2))
                    if m < 0:
                        left_lines.append((m,c))
                        left_weights.append((line_length))
                    else:
                        right_lines.append((m,c))
                        right_weights.append((line_length))
    except:
        pass
    ######LEFT LINE AND RIGHT LINES SEPARATED REMOVING VERTICAL LINES####
    left_lane  = np.dot(left_weights,  left_lines) /np.sum(left_weights)  if len(left_weights) >0 else None
    right_lane = np.dot(right_weights, right_lines)/np.sum(right_weights) if len(right_weights)>0 else None
    print(left_lane)
    return left_lane, right_lane

##def make_line_points(y1, y2, line):
##    if line is None:
##        return None
##    
##    slope, intercept = line
##    
##    # make sure everything is integer as cv2.line requires it
##    if (slope==0):
##        pass
##    else:
##        x1 = int((y1 - intercept)/slope)
##        x2 = int((y2 - intercept)/slope)
##        y1 = int(y1)
##        y2 = int(y2)
##        
##    return ((x1, y1), (x2, y2))

def lane_lines(image, lines):
    left_lane, right_lane = average_slope_intercept(lines)
    
    y1 = image.shape[0] # bottom of the image
    y2 = y1*0.6         # slightly lower than the middle

    #left_line  = make_line_points(y1, y2, left_lane)
    #right_line = make_line_points(y1, y2, right_lane)
    #return left_line, right_line

def roi(image,vertices):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,vertices,255)
    masked = cv2.bitwise_and(image,mask)
    return masked
    
def process_image(original_image):

    lowerwhite = np.array([200,200,200])
    upperwhite = np.array([255,255,255])
    loweryellow = np.array([60,0,100])
    upperyellow = np.array([255,255,255])

    frameHLS=cv2.cvtColor(original_image,cv2.COLOR_BGR2HLS)
    #frameHSV=cv2.cvtColor(original_image,cv2.COLOR_BGR2LAB)

    mask_white=cv2.inRange(original_image,lowerwhite,upperwhite)
    mask_yellow=cv2.inRange(frameHLS,loweryellow,upperyellow)
    mask = cv2.bitwise_or(mask_white,mask_yellow)

    res=cv2.bitwise_and(original_image,original_image,mask=mask)
    new_image1 = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
    
    blur = cv2.GaussianBlur(new_image1,(15,15),0)
    dilation = cv2.dilate(blur,kernel2,iterations = 1)

    edges = cv2.Canny(dilation,40,150,apertureSize=3)

    return edges

while True:
    
    printscreen_pil = ImageGrab.grab(bbox=(0,10,640,480))
    processed_image = process_image(np.array(printscreen_pil))
    vertices = np.array([[50,375],[200,275],[400,275],[620,375],[620,450],[50,450]])
    new_image=roi(processed_image,[vertices])

    sorted_lines=HoughLines(new_image,processed_image)
    

    cv2.imshow('graywindow',processed_image)

    k=cv2.waitKey(1)
    if k == ord('q'):
        break

