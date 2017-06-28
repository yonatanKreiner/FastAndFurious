# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 14:05:06 2017

@author: Tom
"""

import numpy as np
import cv2


def dense_optical_flow(VIDEO_PATH, th=1):
    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    grayscale = np.zeros_like(prvs)
    grayscale[...] = 0
    while(1):
        ret, frame2 = cap.read()
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        mag[mag < th] = 0
#        hsv[...,0] = ang*180/np.pi/2
#        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
#        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        cv2.imshow('original',frame2)
#        cv2.resizeWindow('original', 1200,720)
        cv2.imshow('optical_flow',mag)
#        cv2.resizeWindow('optical_flow', 1200,720)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('opticalfb.png',frame2)
            cv2.imwrite('opticalhsv.png',bgr)
        prvs = next
    cap.release()
    cv2.destroyAllWindows()

        
        
def lk_optical_flow(VIDEO_PATH):
    cap = cv2.VideoCapture(VIDEO_PATH)
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 1,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # Create some random colors
    color = np.random.randint(0,255,(100,3))
    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    #    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    p0 = cv2.cornerHarris(old_gray, 7, 5, 0.05, cv2.BORDER_DEFAULT)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    while(1):
        ret,frame = cap.read()

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, 
                                               winSize  = (15,15),
                                               maxLevel = 0,
                                               criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]
        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
        img = cv2.add(frame,mask)
        cv2.imshow('frame',img)
        cv2.resizeWindow('frame', 1200,720)        
        cv2.imshow('original',frame_gray)
        cv2.resizeWindow('original', 1200,720)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)
    cv2.destroyAllWindows()
    cap.release()
    

VIDEO_PATH = r"C:\Users\Tom\Desktop\Hackaton\3.avi"
dense_optical_flow(VIDEO_PATH)
#lk_optical_flow(VIDEO_PATH)