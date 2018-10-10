import cv2
import numpy as np
import matplotlib.pyplot as plt
import time



cap=cv2.VideoCapture("challenge_video.mp4")

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('o1utput.avi',fourcc, 20.0, (844,480))

kernel = np.ones((5,5),np.uint8)
sobel_kernel = 15
thresh_dir_min = 0.7
thresh_dir_max = 1.2
thresh_mag_min = 50
thresh_mag_max = 255

srcleft = np.float32([[215,315],[400,315],[270,435],[0,376]])
dstleft = np.float32([[0, 0],[750, 0],[750,480],[0, 480]])

srcmid = np.float32([[315,315],[575,315],[775,430],[100,430]])
dstmid = np.float32([[0, 0],[840, 0],[840,480],[0, 480]])

srcright = np.float32([[440,315],[595,315],[820,380],[560,430]])
dstright = np.float32([[0, 0],[750, 0],[750,480],[0, 480]])

left_lane=[]
right_lane=[]
nwindows = 10
margin = 100
minpix = 50
window_height = 48
set1=0
ploty = np.linspace(0, 479, 480)
y_eval = np.max(ploty)
img_size = (844,480)

def lane_finding(binary_warped_B):
    
    histogram = np.sum(binary_warped_B[binary_warped_B.shape[0]//2:,:], axis=0)
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    nonzero = binary_warped_B.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    left_lane_inds = []
    right_lane_inds = []
    set1=0
    for window in range(nwindows):

        win_y_low = binary_warped_B.shape[0] - (window+1)*window_height
        win_y_high = binary_warped_B.shape[0] - window*window_height

        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        #print(good_left_inds)
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

##        if len(good_left_inds) > minpix :
##            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
##            print(leftx_current)
##            set1=set1+1
##            print(set1)
##        if len(good_right_inds) > minpix:        
##            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    #print(left_lane_inds)
   
    leftx = nonzerox[left_lane_inds] 
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    return ploty, left_fitx, right_fitx, left_fit, right_fit,leftx,rightx,lefty,righty


def radius_pixels(ploty, left_fit, right_fit):


    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    
    radius_pix = (left_curverad + right_curverad)//2

    return radius_pix

def radius_m(leftx, rightx, ploty, y_eval):
    
    ym_per_pix = 30/720 
    xm_per_pix = 370/700 
    left_fit_cr = np.polyfit(leftx*ym_per_pix, (leftx*xm_per_pix), 2)
    right_fit_cr = np.polyfit(rightx*ym_per_pix, rightx*xm_per_pix, 2)

    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    radius = (left_curverad + right_curverad)//2

    return radius


def offset_cal_pix(left_fitx, right_fitx, image,lefty,righty):
    bottom=[]
    top=[]
      
    left_pointbottomx = int(left_fitx[len(left_fitx) - 10])
    right_pointbottomx = int(right_fitx[len(right_fitx) - 10])
    left_pointbottomy = int(lefty[len(lefty)-10])
    right_point_bottomy = int(righty[len(righty)-10])
    
    left_pointtopx = int(left_fitx[10])
    right_pointtopx = int(right_fitx[10])
    left_pointtopy = int(lefty[10])
    right_pointtopy = int(lefty[10])

    
    bottommidx = int((right_pointbottomx-left_pointbottomx)/2)
    topmidx = int((right_pointtopx-left_pointtopx)/2)
    bottommidy = int((right_point_bottomy+left_pointbottomy)/2)
    topmidy = int((right_pointtopy+left_pointtopy)/2)
    bottom=(bottommidx,bottommidy)
    top = (topmidx,topmidy)
    
    lane_width = right_pointbottomx - left_pointbottomx    
    lane_center = ((left_pointbottomx) + (lane_width / 2))    
    vehicle_center = (image.shape[1] / 2)
    offset = vehicle_center - lane_center
    return offset, lane_width, lane_center,bottom,top

def offset_cal_met(offset,lane_width):
    xm_per_pix = 3.0/700   
    offset_met = round(xm_per_pix*offset, 4)
    lane_width = round(xm_per_pix*lane_width, 4)    
    return offset_met,lane_width


def augument(binary_warped_B, left_fitx, right_fitx, ploty, Minv_B, img, undistorted, lane_width, prev_pts,color,bottom,top):

    warp_zero = np.zeros_like(binary_warped_B).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    if lane_width <= 750:
        
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        #pts_mid = np.vstack((pointsleft,pointsright)).astype(np.int32).T
        
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
       
        if len(pts)>0:
            cv2.fillPoly(color_warp, np.int_([pts]), color)
            cv2.line(color_warp,bottom,top,(255,0,0),50)
            prev_pts = pts
    else:
        pts = prev_pts
        if len(pts)>0:
            cv2.fillPoly(color_warp, np.int_([prev_pts]), color)
            cv2.line(color_warp,bottom,top,(255,0,0),50)
    cv2.imshow("a",color_warp)
    newwarp = cv2.warpPerspective(color_warp, Minv_B, (img.shape[1], img.shape[0])) 
    result = cv2.addWeighted(undistorted, 1, newwarp, 0.3, 0)   
    prev_pts = pts
    return result, prev_pts



def perspective_transform(img,src,dst):   
    img_size = (img.shape[1], img.shape[0])    
    m = cv2.getPerspectiveTransform(src, dst)
    m_inv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, m, img_size, flags=cv2.INTER_LINEAR)
    return warped,m_inv




def lane(image,d,frame):
    
    prev_pts = []   
    
    frameLAB=cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
    frameHLS=cv2.cvtColor(image,cv2.COLOR_BGR2HLS)
    frameYCrCb=cv2.cvtColor(image,cv2.COLOR_BGR2YCrCb)
    (Y,Cr,Cb) = cv2.split(frameYCrCb) 
    (B, G, R) = cv2.split(image)
    (L ,A, B1) = cv2.split(frameLAB)
    (H,L1,S1) = cv2.split(frameHLS)
    blurr = cv2.medianBlur(R,7)
    blurl = cv2.medianBlur(L,7)
    blury = cv2.medianBlur(Y,7)
    blurs = cv2.medianBlur(S1,7)

    rgb_r_thresh = cv2.adaptiveThreshold(blurr, 255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=25, C=-15)
    lab_l_thresh = cv2.adaptiveThreshold(blurl, 255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=25, C=-30)
    hls_s_thresh = cv2.adaptiveThreshold(blurs, 255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=25, C=-20)
    ycrcb_y_thresh = cv2.adaptiveThreshold(blury, 255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=25, C=-25)
    lab_l_thresh_noise = cv2.adaptiveThreshold(blurl, 255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=65, C=10)

    
    noise_mask = cv2.inRange(blurl, 135, 255)
    noise_bool = np.logical_or(np.logical_not(noise_mask), lab_l_thresh_noise)
    noise_mask = np.zeros_like(blurr, dtype=np.uint8)
    noise_mask[noise_bool] = 255
    m1=np.logical_or(rgb_r_thresh,hls_s_thresh)
    merged_bool = np.logical_and(np.logical_or(m1, lab_l_thresh,ycrcb_y_thresh), noise_mask)
    merged1=np.logical_or(merged_bool,m1)
    merged = np.zeros_like(R, dtype=np.uint8)
    merged[merged1] = 255

    ploty, left_fitx, right_fitx, left_fit, right_fit, leftx, rightx, lefty, righty = lane_finding(merged)
    radius_pix = radius_pixels(ploty, left_fit, right_fit)
    radius = radius_m(leftx, rightx, ploty, y_eval)
    
    
    
    return merged, left_fitx, right_fitx, d, prev_pts, lefty, righty    
            
    
   
while(cap.isOpened()):

    ret,frame=cap.read()
    start_time = time.time()
    
    aleft,dleft = perspective_transform(frame,srcleft,dstleft)
    amid,dmid = perspective_transform(frame,srcmid,dstmid)
    aright,dright = perspective_transform(frame,srcright,dstright)

    mergedm, left_fitx2, right_fitx2, d2, prev_pts2,leftmidy,rightmidy = lane(amid,dmid,frame)
    offsetmid, lane_widthmid,lane_centermid,midbottom,midtop = offset_cal_pix(left_fitx2, right_fitx2, amid,leftmidy,rightmidy)
    offset_metmid,lmid = offset_cal_met(offsetmid,lane_widthmid)
    
    mergedr, left_fitx1, right_fitx1, d1, prev_pts1,rightlefty,rightrighty = lane(aright,dright,frame)
    offsetright, lane_widthright,lane_centerright,rightbottom,righttop = offset_cal_pix(left_fitx1, right_fitx1, aright,rightlefty,rightrighty)
    offset_metright,lright = offset_cal_met(offsetright,lane_widthright)
    
    mergedl, left_fitx3, right_fitx3, d3, prev_pts3,leftlefty,leftrighty = lane(aleft,dleft,frame)
    offsetleft, lane_widthleft,lane_centerleft,leftbottom,lefttop = offset_cal_pix(left_fitx3, right_fitx3, aleft,leftlefty,leftrighty)
    offset_metleft,lleft = offset_cal_met(offsetleft,lane_widthleft)

    offleft = (lmid/2)+(lleft/2)+offset_metleft
    offright = (lmid/2)+(lright/2)+offset_metright
    
    set1=set1+1
    print(set1)
    augumented1,prev_pts11 = augument(mergedr, left_fitx1, right_fitx1, ploty, d1,frame, frame, lane_widthright, prev_pts1,(0,255, 255),rightbottom,righttop)
    augumented2,prev_pts22 = augument(mergedm, left_fitx2, right_fitx2, ploty, d2, augumented1, augumented1, lane_widthmid, prev_pts2,(0,255, 0),midbottom,midtop)
    augumented3,prev_pts33 = augument(mergedl, left_fitx3, right_fitx3, ploty, d3, augumented2, augumented2, lane_widthleft, prev_pts3,(0,255, 255),leftbottom,lefttop)
    #print("FPS: ", 1.0 / (time.time() - start_time))
    cv2.imshow('Result', augumented3)
    #out.write(augumented3)
    
    
    k=cv2.waitKey(1)
    if k==ord('q'):
        break                         
   
cv2.destroyAllWindows()
cap.release()
out.release()
