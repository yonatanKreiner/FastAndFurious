# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 14:05:06 2017

@author: Tom
"""

import numpy as np
import cv2


def dense_optical_flow(VIDEO_PATH, num_of_frames=20, th=1):
    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame1 = cap.read()
    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    grayscale = np.zeros_like(prvs)
    grayscale[...] = 0
    f_movement = []
    f_num_of_elements = []
    for i in range(num_of_frames - 2):
        ret, frame2 = cap.read()
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        mag[mag < th] = 0
        mag[mag >= th] = 255
        dilation = cv2.dilate(mag.astype(np.uint8),np.ones((15,15),np.uint8),iterations = 1)
        erosion = cv2.erode(dilation,np.ones((30,30),np.uint8),iterations = 1)
        moving_elements = cv2.connectedComponents(erosion, 8, cv2.CV_32S)
        is_movement = 0
        if moving_elements[0] > 0:
            is_movement = 1
        f_movement.append(is_movement)
        f_num_of_elements.append(moving_elements[0])
#        cv2.imshow('original',frame2)
#        cv2.resizeWindow('original', 1200,720)
#        cv2.imshow('optical_flow',mag)
#        cv2.imshow('optical_flow2',erosion.astype(np.float32))
#        cv2.resizeWindow('optical_flow', 1200,720)
        prvs = next
        if int(i/(num_of_frames - 2)*100) % 20 == 0:
            print(int(i/(num_of_frames - 2)*100), '% Done')
    

    cap.release()
    cv2.destroyAllWindows()
    features = []
    features.append(np.round(np.average(f_movement)))
    features.append(np.round(np.average(f_num_of_elements)))
    print(features)
    return features

def dense_optical_flow_lk(VIDEO_PATH, num_of_frames=20, th=1):
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    ret, frame1 = cap.read()
    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    grayscale = np.zeros_like(prvs)
    
    #Lucas-Kanade Optical Flow in OpenCV
    
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )
    
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0,255,(100,3))
      
    # Take first frame and find corners in it
    p0 = cv2.goodFeaturesToTrack(prvs, mask = None, **feature_params)
    
    # Create a mask image for drawing purposes
    mask = np.zeros_like(frame1)
    
    

    grayscale[...] = 0
    f_movement = []
    f_num_of_elements = []
    for i in range(num_of_frames - 2):
        ret, frame2 = cap.read()
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(prvs, next, p0, None, **lk_params)
        
        
        
        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        mag[mag < th] = 0
        mag[mag >= th] = 255
        dilation = cv2.dilate(mag.astype(np.uint8),np.ones((15,15),np.uint8),iterations = 1)
        erosion = cv2.erode(dilation,np.ones((30,30),np.uint8),iterations = 1)
        moving_elements = cv2.connectedComponents(erosion, 8, cv2.CV_32S)
        is_movement = 0
        if moving_elements[0] > 0:
            is_movement = 1
        f_movement.append(is_movement)
        f_num_of_elements.append(moving_elements[0])
#        cv2.imshow('original',frame2)
#        cv2.resizeWindow('original', 1200,720)
#        cv2.imshow('optical_flow',mag)
#        cv2.imshow('optical_flow2',erosion.astype(np.float32))
#        cv2.resizeWindow('optical_flow', 1200,720)
        prvs = next
        if int(i/(num_of_frames - 2)*100) % 20 == 0:
            print(int(i/(num_of_frames - 2)*100), '% Done')
    

    cap.release()
    cv2.destroyAllWindows()
    features = []
    features.append(np.round(np.average(f_movement)))
    features.append(np.round(np.average(f_num_of_elements)))
    print(features)
    return features
        
        
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