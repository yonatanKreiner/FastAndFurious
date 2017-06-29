import numpy as np
import cv2

class FramesCompareExtractor():
	def __init__(self):
		pass

	def get_features_names(self):
		return "frames_compare"

	def extract(self, file_path):
		features = []
		cap = cv2.VideoCapture(file_path)

		ret, prev_frame = cap.read()
		ret, prev_frame = cap.read()
		if not ret:
			return

		prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
		frames_counter = 0
		while cap.isOpened():
			ret, next_frame = cap.read()

			if not ret:
				break
			next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
			if frames_counter % 100 == 0:
				(score, diff) = compare_ssim(prev_frame, next_frame, full=True)
				diff = (diff * 255).astype("uint8")
				# cv2.imshow("Diff", diff)

			prev_frame = next_frame
			frames_counter += 1
			
		cap.release()
		cv2.destroyAllWindows()
		
		return [] 


class MotionDetectionExtractor():
	def __dense_optical_flow(self, VIDEO_PATH, threshold_1=1.5, threshold_2=50, threshold_3=5):
		cap = cv2.VideoCapture(VIDEO_PATH)
		ret, frame1 = cap.read()
		prev_frame = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
		grayscale = np.zeros_like(prev_frame)
		grayscale[...] = 0
		moved_frames = 0
		total_frames = 0

		while cap.isOpened():
			ret, frame2 = cap.read()
			if not ret:
				break
			total_frames += 1

			if total_frames % threshold_3 == 0:
				next_frame = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
				flow = cv2.calcOpticalFlowFarneback(prev_frame,next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
				mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

				num_of_moving_pixels = len(mag[mag > threshold_1])
				if num_of_moving_pixels > threshold_2:
					moved_frames +=1
				
				prev_frame = next_frame
		cap.release()
		cv2.destroyAllWindows()
		
		print([float(moved_frames) / float(total_frames / threshold_3)])

		return [float(moved_frames) / float(total_frames / threshold_3)]

	def get_features_names(self):
		return "is_motion"

	def extract(self, file_path):
		return self.__dense_optical_flow(file_path)

class TOM_FEATURES():
	def __dense_optical_flow(self, VIDEO_PATH, th=1):
        cap = cv2.VideoCapture(VIDEO_PATH)
        ret, frame1 = cap.read()
        ret, frame1 = cap.read()
        
        #Background Subtraction
        fgbg = cv2.createBackgroundSubtractorKNN()
        max_bgdiff = []
        i = 0
        bg_change_above_15000 = 0
        bg_change_above_45000 = 0
        bg_change_above_75000 = 0
        bg_change_above_105000 = 0
        bg_change_above_135000 = 0
        
        #Optical Flow
        prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        grayscale = np.zeros_like(prvs)
        grayscale[...] = 0
        f_movement = []
        f_num_of_elements = []
        f_num_of_contours = []
        f_size_of_largest_contours = []
        first = True
        while cap.isOpened():
            ret, frame2 = cap.read()
            if not ret:
    				break
            next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            mag[mag < th] = 0
            mag[mag >= th] = 255
    #        dilation = cv2.dilate(mag.astype(np.uint8),np.ones((15,15),np.uint8),iterations = 1)
    #        erosion = cv2.erode(dilation,np.ones((25,25),np.uint8),iterations = 1)
            opening = cv2.morphologyEx(mag.astype(np.uint8), cv2.MORPH_OPEN, np.ones((15,15),np.uint8))
    
            moving_elements = cv2.connectedComponentsWithStats(opening, 4, cv2.CV_32S)
            is_movement = 0
            if moving_elements[0] > 0:
                is_movement = 1
                
            # Movement Feature
            f_movement.append(is_movement)
            
            # Number of Connected Components Feature
            f_num_of_elements.append(moving_elements[0])
            
            ret, thresh = cv2.threshold(opening, 127, 255, 0)
            img, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            img = frame2
            
            # Number of Contours Feature
            f_num_of_contours.append(len(contours))
            max_contour = 0
            rows,cols = img.shape[:2]
            for c in contours:
                x,y,w,h = cv2.boundingRect(c)
                img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                if max_contour < w*h:
                    max_contour = w*h
            
            # Size of Largest Bounding Rectangle
            f_size_of_largest_contours.append(max_contour)
            
            #Background Subtraction
            fgmask = fgbg.apply(frame2)
            fgmask[fgmask < th] = 0
            fgmask[fgmask >= th] = 255
            
            closing  = cv2.morphologyEx(fgmask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((15,15),np.uint8))
            closing = cv2.erode(closing,np.ones((10,10),np.uint8),iterations = 1)
            
            ret, thresh = cv2.threshold(closing, 127, 255, 0)
            img2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            img2 = frame2
            
            if i < 10:
                i = i + 1
            else:
                max_contourArea = 0
                for c in contours:
                    if max_contourArea < cv2.contourArea(c):
                        max_contourArea = cv2.contourArea(c)
                    
                    x,y,w,h = cv2.boundingRect(c)
                    if w*h > 15000:
                        img2 = cv2.rectangle(img2,(x,y),(x+w,y+h),(0,255,0),2)
                        bg_change_above_15000 = bg_change_above_15000 + 1
                    if w*h > 45000:
    #                    img2 = cv2.rectangle(img2,(x,y),(x+w,y+h),(0,255,0),2)
                        bg_change_above_45000 = bg_change_above_45000 + 1  
                    if w*h > 75000:
    #                    img2 = cv2.rectangle(img2,(x,y),(x+w,y+h),(0,255,0),2)
                        bg_change_above_75000 = bg_change_above_75000 + 1
                    if w*h > 105000:
    #                    img2 = cv2.rectangle(img2,(x,y),(x+w,y+h),(0,255,0),2)
                        bg_change_above_105000 = bg_change_above_105000 + 1
                    if w*h > 135000:
    #                    img2 = cv2.rectangle(img2,(x,y),(x+w,y+h),(0,255,0),2)
                        bg_change_above_135000 = bg_change_above_135000 + 1
                max_bgdiff.append(max_contourArea)
    #            print(max_contourArea, max_contour)
    #        print(max_contourAreas)
    
    
        #        cv2.imshow('fgmask',fgmask)
#            cv2.imshow('closing',closing)
#            cv2.imshow('img2',img2)
    #        cv2.imshow('closing ',closing )
            
            
#            cv2.imshow('img',img)
#            cv2.imshow('opening',opening)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break                
                
            prvs = next
#            if int(i/(num_of_frames - 2)*100) % 20 == 0:
#                print(int(i/(num_of_frames - 2)*100), '% Done')
                   
        cap.release()
        cv2.destroyAllWindows()
        features = []
        features.append(np.round(np.average(f_movement)))
        features.append(np.round(np.var(f_movement)))
        features.append(np.round(np.std(f_movement)))
        features.append(np.round(np.average(f_num_of_elements)))
        features.append(np.round(np.var(f_num_of_elements)))
        features.append(np.round(np.std(f_num_of_elements)))
        features.append(np.round(np.average(f_num_of_contours)))
        features.append(np.round(np.var(f_num_of_contours)))
        features.append(np.round(np.std(f_num_of_contours)))        
        features.append(np.max(f_size_of_largest_contours))
        features.append(bg_change_above_15000)
        features.append(bg_change_above_45000)
        features.append(bg_change_above_75000)
        features.append(bg_change_above_105000)
        features.append(bg_change_above_135000)
        features.append(np.max(max_bgdiff))
        features.append(np.average(max_bgdiff))
        features.append(np.var(max_bgdiff))
        features.append(np.std(max_bgdiff))
#        print(features)
        return features
    
    def __background_subtraction(VIDEO_PATH, num_of_frames=100, th=127):
        cap = cv2.VideoCapture(VIDEO_PATH)
        
        for i in range(1):
            ret, frame1 = cap.read()
        fgbg = cv2.createBackgroundSubtractorKNN()
        max_bgdiff = []
        
        i = 0
        bg_change_above_15000 = 0
        bg_change_above_45000 = 0
        bg_change_above_75000 = 0
        bg_change_above_105000 = 0
        bg_change_above_135000 = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
    				break            
            fgmask = fgbg.apply(frame)
            fgmask[fgmask < th] = 0
            fgmask[fgmask >= th] = 255
    #        dilation = cv2.dilate(mag.astype(np.uint8),np.ones((15,15),np.uint8),iterations = 1)
    #        erosion = cv2.erode(fgmask.astype(np.uint8),np.ones((15,15),np.uint8),iterations = 1)
            closing  = cv2.morphologyEx(fgmask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((15,15),np.uint8))
            closing = cv2.erode(closing,np.ones((10,10),np.uint8),iterations = 1)
            
            ret, thresh = cv2.threshold(closing, 127, 255, 0)
            img, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            img = frame
            
            if i < 10:
                i = i + 1
            else:
                max_contourArea = 0
                for c in contours:
                    if max_contourArea < cv2.contourArea(c):
                        max_contourArea = cv2.contourArea(c)
                    
                    x,y,w,h = cv2.boundingRect(c)
                    if w*h > 15000:
                        img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                        bg_change_above_15000 = bg_change_above_15000 + 1
                    if w*h > 45000:
    #                    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                        bg_change_above_45000 = bg_change_above_45000 + 1  
                    if w*h > 75000:
    #                    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                        bg_change_above_75000 = bg_change_above_75000 + 1
                    if w*h > 105000:
    #                    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                        bg_change_above_105000 = bg_change_above_105000 + 1
                    if w*h > 135000:
    #                    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                        bg_change_above_135000 = bg_change_above_135000 + 1
                max_bgdiff.append(max_contourArea)
    #            print(max_contourArea, max_contour)
    #        print(max_contourAreas)
            
            
    #        cv2.imshow('frame',fgmask)
            cv2.imshow('closing',closing)
            cv2.imshow('frame2',img)
    #        cv2.imshow('closing ',closing )
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        cap.release()
        cv2.destroyAllWindows()   
            
        features = []
        features.append(bg_change_above_15000)
        features.append(bg_change_above_45000)
        features.append(bg_change_above_75000)
        features.append(bg_change_above_105000)
        features.append(bg_change_above_135000)
        features.append(np.max(max_bgdiff))
        features.append(np.average(max_bgdiff))
        features.append(np.var(max_bgdiff))
        features.append(np.std(max_bgdiff))
        print(features)
        return features
    
#   def get_features_names(self):
#       
	def extract(self, file_path):
		return self.__dense_optical_flow(file_path)

if __name__ == "__main__":
	extractor = MotionDetectionExtractor()
	for i in xrange(16):
		extractor.extract(r"C:\videos\1\1\1_40_" + str(i) + ".avi")
	# feature_extractor = FramesCompareExtractor()
	# feature_extractor.extract(r"C:\videos\1.avi")
