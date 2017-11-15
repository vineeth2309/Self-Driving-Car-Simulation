
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
left_lines    = [] 
left_weights  = [] 
right_lines   = [] 
right_weights = []
right_intercepts=[]
right_slopes=[]
left_intercepts=[]
left_slopes=[]
left_lane=[]
right_lane=[]

def HoughLines(img,processed_image):
    lines = cv2.HoughLinesP(img,1,np.pi/180,80,np.array([]),20,300)
    
    i=0
    avg_left_slope =0
    left_line_length=0
    avg_left_intercepts=0
    avg_right_slope =0
    right_line_length=0
    avg_right_intercepts=0
    y1 = processed_image.shape[0] # bottom of the image
    y2 = y1*0.6         # slightly lower than the middle
   # print('1')
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                if (x1==x2):
                    continue
                #y=mx+c
                else:
                    m = ((y2-y1)/(x2-x1))
                    c = (y1 - (m*x1))
                    line_length = np.sqrt(pow((y2-y1),2)+pow((x2-x1),2))
                    if (m==0):
                        pass
                    if m < 0:
                        left_intercepts.append(c)
                        left_lines.append((m,c))
                        left_slopes.append(m)
                        left_weights.append(line_length)
                        left_line_length_sum = np.sum(left_weights)
                        
                        for i in range(len(left_slopes)):
                            avg_left_slope = avg_left_slope + (left_slopes[i]*left_weights[i])
                            avg_left_intercepts= avg_left_intercepts + (left_intercepts[i]*left_weights[i])
                        
                        avg_left_slope = avg_left_slope/left_line_length_sum
                        print(avg_left_slope)
                        avg_left_intercepts=avg_left_intercepts/left_line_length_sum
                        #print(avg_left_slope)
                        left_line_length=left_line_length_sum/len(left_weights)
                        if (avg_left_slope==0):
                            pass
                        else:
                            x1 = int((y1 - avg_left_intercepts)/avg_left_slope)
                            x2 = int((y2 - avg_left_intercepts)/avg_left_slope)
                            y1 = int(y1)
                            y2 = int(y2)
                        left_lane.append((x1, y1))
                        left_lane.append((x2, y2))
                        #print(left_lane)
                        
                    else:
                        right_intercepts.append(c)
                        right_lines.append((m,c))
                        right_slopes.append(m)
                        right_weights.append(line_length)
                        right_line_length_sum = np.sum(right_weights)
                        
                        
                        for i in range(len(right_slopes)):
                            avg_right_slope = avg_right_slope + (right_slopes[i]*right_weights[i])
                            avg_right_intercepts= avg_right_intercepts +(right_intercepts[i]*right_weights[i])
                        
                        avg_right_slope = avg_right_slope/right_line_length_sum
                        avg_right_intercepts=avg_right_intercepts/right_line_length_sum
                        #print(avg_right_slope)

                        left_line_length=right_line_length_sum/len(right_weights)
                        if (avg_right_slope==0):
                            pass
                        else:
                            xx1 = int((y1 - avg_right_intercepts)/avg_right_slope)
                            xx2 = int((y2 - avg_right_intercepts)/avg_right_slope)
                            xy1 = int(y1)
                            xy2 = int(y2)
                        right_lane.append((xx1, xy1))
                        right_lane.append((xx2, xy2))
    else:
        pass
    draw_lines(processed_image,left_lane)
            


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

    HoughLines(new_image,processed_image)
    

    cv2.imshow('graywindow',processed_image)

    k=cv2.waitKey(1)
    if k == ord('q'):
        break
